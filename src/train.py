"""
Training loops: MSE rollback (v1) and full softmax (v2).

Usage:
    python main.py train           # MSE rollback
    python main.py train softmax   # v2 full softmax
"""
import json
import os
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from src.dataset import FeatureStore, pad_history_batch, pad_history_ratings_batch
from src.model import MovieRecommender


# ── Hyperparameters ───────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def get_config() -> dict:
    """All training hyperparameters in one place."""
    return {
        # Sub-embedding sizes (inputs to the projection MLPs)
        'item_movieId_embedding_size':      32,
        'item_year_embedding_size':         8,
        'timestamp_feature_embedding_size': 4,
        'item_tag_embedding_size':          16,
        'item_genome_tag_embedding_size':   32,
        'user_genre_embedding_size':        32,
        'item_genre_embedding_size':        8,
        'user_genome_context_embedding_size': 32,
        # Projection MLP
        'proj_hidden':  256,
        'output_dim':   128,
        # Training
        'lr':           0.005,
        'momentum':     0.9,
        'weight_decay': 1e-4,
        'minibatch_size':   64,
        'training_steps':   300_000,
        'log_every':        20_000,
        'checkpoint_every': 30_000,
        'checkpoint_dir':   'saved_models',
    }


# ── Model factory ─────────────────────────────────────────────────────────────

def build_model(config: dict, fs: FeatureStore) -> MovieRecommender:
    genome_matrix = np.array(
        [fs.movieId_to_genome_tag_context[mid] for mid in fs.top_movies],
        dtype=np.float32,
    )
    pad_row = np.zeros((1, genome_matrix.shape[1]), dtype=np.float32)
    genome_context_buffer = torch.from_numpy(np.vstack([genome_matrix, pad_row]))

    genre_matrix = np.array(
        [fs.movieId_to_genre_context[mid] for mid in fs.top_movies], dtype=np.float32)
    tag_matrix = np.array(
        [fs.movieId_to_tag_context[mid] for mid in fs.top_movies], dtype=np.float32)
    year_array = np.array(
        [fs.year_to_i[fs.movieId_to_year[mid]] for mid in fs.top_movies], dtype=np.int64)
    genre_context_buffer = torch.from_numpy(
        np.vstack([genre_matrix, np.zeros((1, genre_matrix.shape[1]), dtype=np.float32)]))
    tag_context_buffer = torch.from_numpy(
        np.vstack([tag_matrix, np.zeros((1, tag_matrix.shape[1]), dtype=np.float32)]))
    year_context_buffer = torch.from_numpy(
        np.concatenate([year_array, np.zeros((1,), dtype=np.int64)]))

    return MovieRecommender(
        genres_len=len(fs.genres_ordered),
        tags_len=len(fs.tags_ordered),
        genome_tags_len=len(fs.genome_tag_ids),
        top_movies_len=len(fs.top_movies),
        all_years_len=len(fs.years_ordered),
        timestamp_num_bins=fs.timestamp_num_bins,
        user_context_size=fs.user_context_size,
        genome_context_buffer=genome_context_buffer,
        item_genre_embedding_size=config['item_genre_embedding_size'],
        item_tag_embedding_size=config['item_tag_embedding_size'],
        item_genome_tag_embedding_size=config['item_genome_tag_embedding_size'],
        item_movieId_embedding_size=config['item_movieId_embedding_size'],
        item_year_embedding_size=config['item_year_embedding_size'],
        user_genre_embedding_size=config['user_genre_embedding_size'],
        timestamp_feature_embedding_size=config['timestamp_feature_embedding_size'],
        user_genome_context_embedding_size=config.get('user_genome_context_embedding_size', 32),
        genre_context_buffer=genre_context_buffer,
        tag_context_buffer=tag_context_buffer,
        year_context_buffer=year_context_buffer,
        proj_hidden=config.get('proj_hidden', 256),
        output_dim=config.get('output_dim', 128),
    )


def print_model_summary(model: MovieRecommender) -> None:
    m = model
    def _out_dim(tower):
        for layer in reversed(tower):
            if isinstance(layer, torch.nn.Linear):
                return layer.out_features

    id_dim           = m.item_embedding_lookup.embedding_dim
    genome_dim       = _out_dim(m.item_genome_tag_tower)
    gctx_dim         = _out_dim(m.user_genome_context_tower)
    genre_dim        = _out_dim(m.user_genre_tower)
    ts_dim           = m.timestamp_embedding_lookup.embedding_dim
    item_genre_dim   = _out_dim(m.item_genre_tower)
    item_tag_dim     = _out_dim(m.item_tag_tower)
    item_movieId_dim = _out_dim(m.item_embedding_tower)
    year_dim         = _out_dim(m.year_embedding_tower)
    item_total       = item_genre_dim + item_tag_dim + genome_dim + item_movieId_dim + year_dim
    n_params         = sum(p.nelement() for p in model.parameters() if p.requires_grad)

    pool_total = 4 * id_dim
    user_total = pool_total + gctx_dim + genre_dim + ts_dim
    user_desc  = (f"4×sum_pool_id({id_dim}) + genome_ctx({gctx_dim}) + "
                  f"genre({genre_dim}) + ts({ts_dim})")

    proj_h  = m.user_projection[0].out_features
    out_dim = m.user_projection[2].out_features

    print(f"\n── Model dimensions ──")
    print(f"  User side:  {user_desc}  =  {user_total}")
    print(f"  Item side:  genre({item_genre_dim}) + tag({item_tag_dim}) + genome({genome_dim})"
          f" + movieId({item_movieId_dim}) + year({year_dim})  =  {item_total}")
    print(f"  Projection: Linear({proj_h}) → ReLU → Linear({out_dim})  [both towers]")
    print(f"  Parameters: {n_params:,}")


# ── Training loop ─────────────────────────────────────────────────────────────

def train(model: MovieRecommender, train_data: tuple, val_data: tuple,
          config: dict, fs: FeatureStore) -> str:
    """
    Run the MSE training loop. Returns the path of the best checkpoint.

    train_data / val_data: 6-tuple from dataset.build_mse_rollback_dataset()
      (X_genre, X_history, X_history_ratings, timestamp, Y, target_movieId)
    """
    (X_train, X_history_train, X_history_ratings_train, timestamp_train,
     Y_train, target_movieId_train) = train_data

    (X_val, X_history_val, X_history_ratings_val, timestamp_val,
     Y_val, target_movieId_val) = val_data

    print_model_summary(model)

    device = get_device()
    print(f"\nDevice: {device}")
    model = model.to(device)

    # Pre-move full dataset tensors; X_history* stay as lists (padded per batch)
    X_train              = X_train.to(device)
    Y_train              = Y_train.to(device)
    timestamp_train      = timestamp_train.to(device)
    target_movieId_train = target_movieId_train.to(device)
    X_val                = X_val.to(device)
    Y_val                = Y_val.to(device)
    timestamp_val        = timestamp_val.to(device)
    target_movieId_val   = target_movieId_val.to(device)

    pad_idx          = len(fs.top_movies)
    loss_fn          = torch.nn.MSELoss()
    optimizer        = torch.optim.SGD(model.parameters(), lr=config['lr'],
                                        momentum=config['momentum'],
                                        weight_decay=config.get('weight_decay', 0))
    minibatch_size   = config['minibatch_size']
    training_steps   = config['training_steps']
    scheduler        = torch.optim.lr_scheduler.CosineAnnealingLR(
                           optimizer, T_max=training_steps, eta_min=0)
    log_every        = config['log_every']
    checkpoint_every = config['checkpoint_every']
    checkpoint_dir   = config['checkpoint_dir']

    os.makedirs(checkpoint_dir, exist_ok=True)
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    arch_tag = 'gpool_gctx_proj'
    best_val_loss = float('inf')
    best_path     = os.path.join(checkpoint_dir, f'best_mse_{arch_tag}_{run_timestamp}.pth')

    loss_train = []
    loss_val   = []

    print(f"\nStarting training loop  ({training_steps:,} steps) ...")
    start = time.time()

    from tqdm import tqdm
    pbar = tqdm(range(training_steps), desc="Training")
    for i in pbar:
        is_val = (i % log_every == 0)

        if is_val:
            val_batch_size = 1024
            n_val          = X_val.shape[0]
            sq_sum         = 0.0
            n_preds        = 0
            model.eval()
            with torch.no_grad():
                for v0 in range(0, n_val, val_batch_size):
                    v1      = min(v0 + val_batch_size, n_val)
                    vix     = list(range(v0, v1))
                    hb      = pad_history_batch([X_history_val[j] for j in vix], pad_idx).to(device)
                    rb      = pad_history_ratings_batch([X_history_ratings_val[j] for j in vix]).to(device)
                    vp      = model(X_val[v0:v1], hb, rb, timestamp_val[v0:v1],
                                    target_movieId_val[v0:v1])
                    sq_sum  += ((vp - Y_val[v0:v1]) ** 2).sum().item()
                    n_preds += (v1 - v0)
            output_val = sq_sum / n_preds
            loss_val.append(output_val)
        else:
            ix = torch.randint(0, X_train.shape[0], (minibatch_size,)).tolist()
            hist_batch = pad_history_batch([X_history_train[j] for j in ix], pad_idx).to(device)
            rat_batch  = pad_history_ratings_batch([X_history_ratings_train[j] for j in ix]).to(device)
            model.train()
            preds = model(X_train[ix], hist_batch, rat_batch,
                          timestamp_train[ix], target_movieId_train[ix])
            output = loss_fn(preds, Y_train[ix])
            optimizer.zero_grad()
            output.backward()
            optimizer.step()
            scheduler.step()
            loss_train.append(output.item())

        if is_val:
            elapsed   = time.time() - start
            start     = time.time()
            avg_train = np.mean(loss_train[i-log_every:i]) if i >= log_every else loss_train[-1] if loss_train else 0.0
            val_loss  = output_val
            pbar.set_postfix(train=f"{avg_train:.4f}", val=f"{val_loss:.4f}")
            current_lr = scheduler.get_last_lr()[0] if i > 0 else config['lr']
            print(f"[{i:06d}]  train_loss={avg_train:.4f}  val_loss={val_loss:.4f}  "
                  f"lr={current_lr:.6f}  ({elapsed:.0f}s)")

            if i > 0 and val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_path)
                print(f"  → new best {best_val_loss:.4f} → {best_path}")

            if i > 0 and i % checkpoint_every == 0:
                periodic = os.path.join(checkpoint_dir,
                                        f'mse_{arch_tag}_{run_timestamp}_step_{i:06d}.pth')
                torch.save(model.state_dict(), periodic)
                print(f"  → periodic checkpoint → {periodic}")

    # Final val pass — loop covers 0..training_steps-1; last log_every steps otherwise missed
    val_batch_size = 1024
    n_val = X_val.shape[0]
    sq_sum, n_preds = 0.0, 0
    model.eval()
    with torch.no_grad():
        for v0 in range(0, n_val, val_batch_size):
            v1      = min(v0 + val_batch_size, n_val)
            vix     = list(range(v0, v1))
            hb      = pad_history_batch([X_history_val[j] for j in vix], pad_idx).to(device)
            rb      = pad_history_ratings_batch([X_history_ratings_val[j] for j in vix]).to(device)
            vp      = model(X_val[v0:v1], hb, rb, timestamp_val[v0:v1],
                            target_movieId_val[v0:v1])
            sq_sum += ((vp - Y_val[v0:v1]) ** 2).sum().item()
            n_preds += (v1 - v0)
    final_val_loss = sq_sum / n_preds
    print(f"[{training_steps:06d}]  val_loss={final_val_loss:.4f}  (final)")
    if final_val_loss < best_val_loss:
        best_val_loss = final_val_loss
        torch.save(model.state_dict(), best_path)
        print(f"  → new best {best_val_loss:.4f} → {best_path}")

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Best checkpoint: {best_path}")
    return best_path


# ── V2 full softmax training ──────────────────────────────────────────────────

def _config_path(checkpoint_path: str) -> str:
    return os.path.splitext(checkpoint_path)[0] + '_config.json'

def _save_config(config: dict, checkpoint_path: str) -> None:
    with open(_config_path(checkpoint_path), 'w') as f:
        json.dump(config, f, indent=2)

def load_config_for_checkpoint(checkpoint_path: str) -> dict:
    path = _config_path(checkpoint_path)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    cfg = get_v2_config()
    cfg['popularity_alpha'] = 0.0  # safe: never apply unknown bias to an untagged checkpoint
    return cfg


def get_v2_config() -> dict:
    return {
        # Sub-embedding sizes (same architecture as v1)
        'item_movieId_embedding_size':      32, # used for pooling user liked/disliked/full/rating-weighted watch histories (4-pool)
        'item_year_embedding_size':         8,
        'timestamp_feature_embedding_size': 4,
        'item_tag_embedding_size':          16,
        'item_genome_tag_embedding_size':   32,
        'user_genre_embedding_size':        32,
        'item_genre_embedding_size':        8,
        'user_genome_context_embedding_size': 32,
        # Projection MLP
        'proj_hidden':  256,
        'output_dim':   128,
        # Training
        'lr':               0.001,
        'weight_decay':     0.0,
        'adam_eps':         1e-6,
        'minibatch_size':   512,
        'training_steps':   150_000,
        'log_every':        5_000,
        'checkpoint_every': 30_000,
        'checkpoint_dir':   'saved_models',
        'temperature':      0.1,
        'popularity_alpha': 0.5,   # Menon et al. 2021 logit adjustment: add alpha*log1p(count_i) to all logits
    }


def train_v2_softmax(model: MovieRecommender, train_data: tuple, val_data: tuple,
                     config: dict, fs: FeatureStore) -> str:
    """
    Full softmax training loop.

    train_data / val_data: 5-tuple from dataset.build_v2_softmax_dataset()
      (X_genre, X_history, X_hist_ratings, timestamp, target_movieId)
    History tensors are pre-padded (N, MAX_HISTORY_LEN) — indexed directly per batch.

    Each step:
      V_all  = model.full_item_embedding()       (n_movies, dim)  — all corpus items
      U      = model.user_embedding(batch...)    (B, dim)
      scores = (U @ V_all.T) / temperature + alpha*log1p(count_i)   (B, n_movies)
      loss   = cross_entropy(scores, target_idx)
    Menon et al. (2021) logit adjustment: add alpha*log(count_i) to all logits.
    Popular items get a large free boost → easy positives (lazy gradient) → their embeddings
    shrink naturally. Rare items must fight for high scores → embeddings grow. Raw dot products
    at inference are then debiased without any post-hoc correction.
    """
    device = get_device()
    print(f"Using device: {device}")
    model.to(device)

    (X_genre_train, X_history_train, X_hist_liked_train, X_hist_disliked_train,
     X_hist_ratings_train, timestamp_train, target_movieId_train) = train_data

    (X_genre_val, X_history_val, X_hist_liked_val, X_hist_disliked_val,
     X_hist_ratings_val, timestamp_val, target_movieId_val) = val_data

    # Compact tensors go to device up front; history stays CPU (moved per batch)
    X_genre_train        = X_genre_train.to(device)
    timestamp_train      = timestamp_train.to(device)
    target_movieId_train = target_movieId_train.to(device)
    X_genre_val          = X_genre_val.to(device)
    timestamp_val        = timestamp_val.to(device)
    target_movieId_val   = target_movieId_val.to(device)

    print_model_summary(model)

    n_train = X_genre_train.shape[0]
    n_val   = X_genre_val.shape[0]

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'],
                                 weight_decay=config['weight_decay'],
                                 eps=config['adam_eps'])
    training_steps   = config['training_steps']
    scheduler        = torch.optim.lr_scheduler.CosineAnnealingLR(
                           optimizer, T_max=training_steps, eta_min=1e-4)
    minibatch_size   = config['minibatch_size']
    temperature      = config['temperature']
    log_every        = config['log_every']
    checkpoint_every = config['checkpoint_every']
    checkpoint_dir   = config['checkpoint_dir']

    os.makedirs(checkpoint_dir, exist_ok=True)
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    alpha_tag     = str(config['popularity_alpha']).replace('.', '') if config['popularity_alpha'] != int(config['popularity_alpha']) else str(int(config['popularity_alpha']))
    best_val_loss = float('inf')
    best_path     = os.path.join(checkpoint_dir, f'best_softmax_v2_popularity_alpha_{alpha_tag}_{run_timestamp}.pth')

    # Fixed val subset — same examples every log step so val_loss is comparable
    val_eval_size = min(8_192, n_val)
    rng_val = torch.Generator()
    rng_val.manual_seed(0)
    val_eval_idx = torch.randperm(n_val, generator=rng_val)[:val_eval_size].tolist()

    # Popularity logit adjustment (Menon et al. 2021):
    # Subtract alpha * log1p(count_i) from item i's logit before softmax.
    # Computed from training targets (corpus indices 0..n_items-1).
    pad_idx = len(fs.top_movies)
    counts_cpu = torch.bincount(target_movieId_train.cpu(), minlength=pad_idx).float()
    popularity_bias = (config['popularity_alpha'] * torch.log1p(counts_cpu)).to(device)
    print(f"  Popularity bias: alpha={config['popularity_alpha']}  "
          f"max_adj={popularity_bias.max():.3f}  min_adj={popularity_bias.min():.3f}")

    loss_train = []
    grad_norms = []

    print(f"\nStarting v2 softmax training ({training_steps:,} steps, "
          f"batch={minibatch_size}, temp={temperature}) ...")
    print(f"  Train: {n_train:,}  |  Val: {n_val:,}  (eval subset: {val_eval_size:,})")
    start = time.time()

    from tqdm import tqdm
    pbar = tqdm(range(training_steps), desc="Training (v2 softmax)")
    for i in pbar:
        is_val = (i % log_every == 0)

        if is_val:
            model.eval()
            with torch.no_grad():
                V_all = model.full_item_embedding()

                if i == 0:
                    # Logit diagnostics — verify temperature is sane before training starts
                    vidx = torch.randint(0, n_val, (minibatch_size,))
                    U = model.user_embedding(
                        X_genre_val[vidx],
                        X_history_val[vidx].to(device),
                        X_hist_liked_val[vidx].to(device),
                        X_hist_disliked_val[vidx].to(device),
                        X_hist_ratings_val[vidx].to(device),
                        timestamp_val[vidx],
                    )
                    raw  = U @ V_all.T
                    scaled = raw / temperature
                    print(f"  [step 0 logits] raw dot  mean={raw.mean():.4f}  std={raw.std():.4f}  "
                          f"min={raw.min():.4f}  max={raw.max():.4f}")
                    print(f"  [step 0 logits] /temp={temperature}  mean={scaled.mean():.4f}  "
                          f"std={scaled.std():.4f}")
                    print(f"  [step 0 logits] random-baseline loss = {np.log(V_all.shape[0]):.4f}")

                val_losses = []
                for vs in range(0, val_eval_size, minibatch_size):
                    ve   = min(vs + minibatch_size, val_eval_size)
                    vidx = val_eval_idx[vs:ve]
                    U = model.user_embedding(
                        X_genre_val[vidx],
                        X_history_val[vidx].to(device),
                        X_hist_liked_val[vidx].to(device),
                        X_hist_disliked_val[vidx].to(device),
                        X_hist_ratings_val[vidx].to(device),
                        timestamp_val[vidx],
                    )
                    logits = (U @ V_all.T) / temperature
                    scores = logits + popularity_bias
                    val_losses.append(F.cross_entropy(scores, target_movieId_val[vidx]).item())
                val_loss = float(np.mean(val_losses))

            avg_train     = np.mean(loss_train[i - log_every:i]) if i >= log_every else (loss_train[-1] if loss_train else 0.0)
            avg_grad_norm = np.mean(grad_norms[i - log_every:i]) if i >= log_every else (grad_norms[-1] if grad_norms else 0.0)
            elapsed    = time.time() - start
            start      = time.time()
            current_lr = scheduler.get_last_lr()[0] if i > 0 else config['lr']
            pbar.set_postfix(train=f"{avg_train:.4f}", val=f"{val_loss:.4f}")
            print(f"[{i:06d}]  train={avg_train:.4f}  val={val_loss:.4f}  "
                  f"lr={current_lr:.6f}  grad_norm={avg_grad_norm:.3f}  ({elapsed:.0f}s)")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_path)
                _save_config(config, best_path)
                print(f"  → new best {best_val_loss:.4f} → {best_path}")

            if i > 0 and i % checkpoint_every == 0:
                periodic = os.path.join(checkpoint_dir,
                                        f'softmax_v2_popularity_alpha_{alpha_tag}_{run_timestamp}_step_{i:06d}.pth')
                torch.save(model.state_dict(), periodic)
                _save_config(config, periodic)
                print(f"  → periodic checkpoint → {periodic}")

        else:
            model.train()
            V_all = model.full_item_embedding()
            ix    = torch.randint(0, n_train, (minibatch_size,))
            U = model.user_embedding(
                X_genre_train[ix],
                X_history_train[ix].to(device),
                X_hist_liked_train[ix].to(device),
                X_hist_disliked_train[ix].to(device),
                X_hist_ratings_train[ix].to(device),
                timestamp_train[ix],
            )
            logits = (U @ V_all.T) / temperature
            scores = logits + popularity_bias
            loss   = F.cross_entropy(scores, target_movieId_train[ix])
            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).item()
            optimizer.step()
            scheduler.step()
            loss_train.append(loss.item())
            grad_norms.append(grad_norm)

    # Final val pass
    model.eval()
    with torch.no_grad():
        V_all = model.full_item_embedding()
        val_losses = []
        for vs in range(0, val_eval_size, minibatch_size):
            ve   = min(vs + minibatch_size, val_eval_size)
            vidx = val_eval_idx[vs:ve]
            U = model.user_embedding(
                X_genre_val[vidx],
                X_history_val[vidx].to(device),
                X_hist_liked_val[vidx].to(device),
                X_hist_disliked_val[vidx].to(device),
                X_hist_ratings_val[vidx].to(device),
                timestamp_val[vidx],
            )
            logits = (U @ V_all.T) / temperature
            scores = logits + popularity_bias
            val_losses.append(F.cross_entropy(scores, target_movieId_val[vidx]).item())
    final_val_loss = float(np.mean(val_losses))
    print(f"[{training_steps:06d}]  val={final_val_loss:.4f}  (final)")
    if final_val_loss < best_val_loss:
        best_val_loss = final_val_loss
        torch.save(model.state_dict(), best_path)
        _save_config(config, best_path)
        print(f"  → new best {best_val_loss:.4f} → {best_path}")

    print(f"\nV2 softmax training complete. Best val loss: {best_val_loss:.4f}")
    print(f"Best checkpoint: {best_path}")
    return best_path
