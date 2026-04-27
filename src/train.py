"""
Training loop.

Usage:
    python main.py train
"""
import os
import time
from datetime import datetime

import numpy as np
import torch
from src.dataset import FeatureStore, pad_history_batch, pad_history_ratings_batch
from src.model import MovieRecommender


# ── Hyperparameters ───────────────────────────────────────────────────────────

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
        'use_user_genome_pool':             True,
        'use_item_pool_for_history':        False,  # True → pool full item embedding instead of id pool
        'use_item_pool_for_genome':         False,  # True → replace both id+genome pools with single item pool
        'use_user_genome_context':          True,   # True → add rating-weighted raw genome taste vector
        'user_genome_context_embedding_size': 32,
        # Projection MLP (learn cross-feature interactions after concat)
        'proj_hidden':  256,
        'output_dim':   128,
        # Training
        'lr':               0.01, # current best used 0.005, but let's double it
        'momentum':         0.9,
        'minibatch_size':   64,
        'training_steps':   150_000,
        'log_every':        5_000,
        'checkpoint_every': 30_000,
        'checkpoint_dir':   'saved_models',
    }


# ── Model factory ─────────────────────────────────────────────────────────────

def build_model(config: dict, fs: FeatureStore) -> MovieRecommender:
    # Build genome context buffer: (top_movies_len + 1, genome_tags_len)
    # Row i = genome tag context for movie at embedding index i; last row = zeros (pad)
    genome_matrix = np.array(
        [fs.movieId_to_genome_tag_context[mid] for mid in fs.top_movies],
        dtype=np.float32,
    )
    pad_row = np.zeros((1, genome_matrix.shape[1]), dtype=np.float32)
    genome_context_buffer = torch.from_numpy(np.vstack([genome_matrix, pad_row]))

    use_item_pool_for_history = config.get('use_item_pool_for_history', False)
    use_item_pool_for_genome  = config.get('use_item_pool_for_genome',  False)

    # Always build genre/tag/year buffers — forward() uses them for target-movie lookup
    # regardless of pool mode. Non-persistent: not saved in state_dict.
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

    model = MovieRecommender(
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
        use_user_genome_pool=config.get('use_user_genome_pool', True),
        use_item_pool_for_history=use_item_pool_for_history,
        use_item_pool_for_genome=use_item_pool_for_genome,
        use_user_genome_context=config.get('use_user_genome_context', False),
        user_genome_context_embedding_size=config.get('user_genome_context_embedding_size', 32),
        genre_context_buffer=genre_context_buffer,
        tag_context_buffer=tag_context_buffer,
        year_context_buffer=year_context_buffer,
        proj_hidden=config.get('proj_hidden', None),
        output_dim=config.get('output_dim', 128),
    )
    return model


def print_model_summary(model: MovieRecommender) -> None:
    """Print tower dimensions and parameter count for a built model."""
    m = model
    def _out_dim(tower):
        for layer in reversed(tower):
            if isinstance(layer, torch.nn.Linear):
                return layer.out_features

    history_dim      = m.item_embedding_lookup.embedding_dim
    genre_dim        = _out_dim(m.user_genre_tower)
    ts_dim           = m.timestamp_embedding_lookup.embedding_dim
    item_genre_dim   = _out_dim(m.item_genre_tower)
    item_tag_dim     = _out_dim(m.item_tag_tower)
    genome_tag_dim   = _out_dim(m.item_genome_tag_tower)
    item_movieId_dim = _out_dim(m.item_embedding_tower)
    year_dim         = _out_dim(m.year_embedding_tower)
    item_total       = item_genre_dim + item_tag_dim + genome_tag_dim + item_movieId_dim + year_dim
    n_params         = sum(p.nelement() for p in model.parameters() if p.requires_grad)

    gctx_dim  = _out_dim(m.user_genome_context_tower) if m.use_user_genome_context else 0
    gctx_part = f" + genome_ctx({gctx_dim})" if gctx_dim else ""

    if m.use_user_genome_pool and m.use_item_pool_for_genome:
        out_dim    = m.user_projection[2].out_features if m.user_projection is not None else history_dim
        user_desc  = f"item_emb_pool({out_dim}) + genre_tower({genre_dim}) + ts_emb({ts_dim}){gctx_part}"
        user_total = out_dim + genre_dim + ts_dim + gctx_dim
    elif m.use_user_genome_pool and m.use_item_pool_for_history:
        out_dim    = m.user_projection[2].out_features if m.user_projection is not None else history_dim
        genome_dim = _out_dim(m.item_genome_tag_tower)
        user_desc  = f"item_emb_pool({out_dim}) + genome_pool({genome_dim}) + genre_tower({genre_dim}) + ts_emb({ts_dim}){gctx_part}"
        user_total = out_dim + genome_dim + genre_dim + ts_dim + gctx_dim
    elif m.use_user_genome_pool:
        genome_dim = _out_dim(m.item_genome_tag_tower)
        user_desc  = f"item_id_pool({history_dim}) + genome_pool({genome_dim}) + genre_tower({genre_dim}) + ts_emb({ts_dim}){gctx_part}"
        user_total = history_dim + genome_dim + genre_dim + ts_dim + gctx_dim
    else:
        user_desc  = f"item_id_pool({history_dim}) + genre_tower({genre_dim}) + ts_emb({ts_dim}){gctx_part}"
        user_total = history_dim + genre_dim + ts_dim + gctx_dim

    print(f"\n── Model dimensions ──")
    print(f"  User side:  {user_desc}  =  {user_total}")
    print(f"  Item side:  genre({item_genre_dim}) + tag({item_tag_dim}) + genome({genome_tag_dim})"
          f" + movieId({item_movieId_dim}) + year({year_dim})  =  {item_total}")
    if m.user_projection is not None:
        proj_h   = m.user_projection[0].out_features
        out_dim  = m.user_projection[2].out_features
        print(f"  Projection: Linear({proj_h}) → ReLU → Linear({out_dim})  [both towers]")
    print(f"  Parameters: {n_params:,}")


# ── Training loop ─────────────────────────────────────────────────────────────

def train(model: MovieRecommender, train_data: tuple, val_data: tuple,
          config: dict, fs: FeatureStore) -> str:
    """
    Run the training loop. Returns the path of the best checkpoint.

    train_data / val_data: 6-tuple from dataset.build_mse_rollback_dataset()
      (X_genre, X_history, X_history_ratings, timestamp, Y, target_movieId)
    Item features are looked up from model buffers during forward().
    """
    (X_train, X_history_train, X_history_ratings_train, timestamp_train,
     Y_train, target_movieId_train) = train_data

    (X_val, X_history_val, X_history_ratings_val, timestamp_val,
     Y_val, target_movieId_val) = val_data

    print_model_summary(model)

    pad_idx          = len(fs.top_movies)
    loss_fn          = torch.nn.MSELoss()
    optimizer        = torch.optim.SGD(model.parameters(), lr=config['lr'], momentum=config['momentum'])
    minibatch_size   = config['minibatch_size']
    training_steps   = config['training_steps']
    scheduler        = torch.optim.lr_scheduler.CosineAnnealingLR(
                           optimizer, T_max=training_steps, eta_min=0)
    log_every        = config['log_every']
    checkpoint_every = config['checkpoint_every']
    checkpoint_dir   = config['checkpoint_dir']

    os.makedirs(checkpoint_dir, exist_ok=True)
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if config.get('use_item_pool_for_genome'):
        pool_tag = 'ipool'
    elif config.get('use_item_pool_for_history') and config.get('use_user_genome_pool', True):
        pool_tag = 'ipool_gpool'
    elif config.get('use_user_genome_pool', True):
        pool_tag = 'gpool'
    else:
        pool_tag = 'nopool'
    gctx_tag      = '_gctx' if config.get('use_user_genome_context') else ''
    opt_tag       = '_adam' if isinstance(optimizer, torch.optim.Adam) else ''
    arch_tag      = f'{pool_tag}{gctx_tag}{opt_tag}_proj' if config.get('proj_hidden') else f'{pool_tag}{gctx_tag}{opt_tag}'
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
            # Batch the val pass — genome pooling creates (n, hist_len, 1128) which is
            # too large to process all at once.
            val_batch_size = 1024
            n_val          = X_val.shape[0]
            sq_sum         = 0.0
            n_preds        = 0
            model.eval()
            with torch.no_grad():
                for v0 in range(0, n_val, val_batch_size):
                    v1        = min(v0 + val_batch_size, n_val)
                    vix       = list(range(v0, v1))
                    hb        = pad_history_batch([X_history_val[j] for j in vix], pad_idx)
                    rb        = pad_history_ratings_batch([X_history_ratings_val[j] for j in vix])
                    vp        = model(X_val[v0:v1], hb, rb, timestamp_val[v0:v1],
                                      target_movieId_val[v0:v1])
                    sq_sum  += ((vp - Y_val[v0:v1]) ** 2).sum().item()
                    n_preds += (v1 - v0)
            output_val = sq_sum / n_preds
            loss_val.append(output_val)
        else:
            ix = torch.randint(0, X_train.shape[0], (minibatch_size,)).tolist()
            hist_batch = pad_history_batch([X_history_train[j] for j in ix], pad_idx)
            rat_batch  = pad_history_ratings_batch([X_history_ratings_train[j] for j in ix])
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
            elapsed = time.time() - start
            start   = time.time()
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

    # Final val pass at step training_steps (last step is training_steps-1, not evaluated)
    val_batch_size = 1024
    n_val = X_val.shape[0]
    sq_sum, n_preds = 0.0, 0
    model.eval()
    with torch.no_grad():
        for v0 in range(0, n_val, val_batch_size):
            v1      = min(v0 + val_batch_size, n_val)
            vix     = list(range(v0, v1))
            hb      = pad_history_batch([X_history_val[j] for j in vix], pad_idx)
            rb      = pad_history_ratings_batch([X_history_ratings_val[j] for j in vix])
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


# ── Softmax training (in-batch negatives) ─────────────────────────────────────

def get_softmax_config() -> dict:
    """Hyperparameters for in-batch negatives softmax training."""
    return {
        # Sub-embedding sizes (same architecture as MSE; projection handles concat mismatch)
        'item_movieId_embedding_size':      32,
        'item_year_embedding_size':         8,
        'timestamp_feature_embedding_size': 4,
        'item_tag_embedding_size':          16,
        'item_genome_tag_embedding_size':   32,
        'user_genre_embedding_size':        32,
        'item_genre_embedding_size':        8,
        'use_user_genome_pool':             True,
        'use_item_pool_for_history':        False,  # True → pool full item embedding instead of id pool
        'use_item_pool_for_genome':         False,  # True → replace both id+genome pools with single item pool
        # Projection MLP
        'proj_hidden':  256,
        'output_dim':   128,
        # Training
        'lr':               0.001,
        'weight_decay':     1e-5,
        'minibatch_size':   512,    # 511 in-batch negatives per example
        'temperature':      0.05,
        'training_steps':   150_000,
        'log_every':        10_000,
        'checkpoint_every': 30_000,
        'checkpoint_dir':   'saved_models',
    }


def train_softmax(model: MovieRecommender, train_data: tuple, val_data: tuple,
                  config: dict, fs: FeatureStore) -> str:
    """
    Train with in-batch negatives cross-entropy (softmax).

    train_data / val_data: 5-tuple from dataset.build_softmax_dataset()
      (X_genre, X_history, X_history_ratings, timestamp, target_movieId)
    Item features are looked up from model buffers during item_embedding().

    Each minibatch of size B computes:
      U = user_embedding(...)         (B, dim)
      V = item_embedding(...)         (B, dim)
      scores = U @ V.T / temperature  (B, B)
      loss   = cross_entropy(scores, arange(B))   target is always on the diagonal
    """
    (X_genre_train, X_history_train, X_history_ratings_train, timestamp_train,
     target_movieId_train) = train_data

    (X_genre_val, X_history_val, X_history_ratings_val, timestamp_val,
     target_movieId_val) = val_data

    print_model_summary(model)

    import torch.nn.functional as F
    pad_idx          = len(fs.top_movies)
    optimizer        = torch.optim.Adam(model.parameters(), lr=config['lr'],
                                        weight_decay=config['weight_decay'])
    training_steps   = config['training_steps']
    scheduler        = torch.optim.lr_scheduler.CosineAnnealingLR(
                           optimizer, T_max=training_steps, eta_min=0)
    minibatch_size   = config['minibatch_size']
    temperature      = config['temperature']
    log_every        = config['log_every']
    checkpoint_every = config['checkpoint_every']
    checkpoint_dir   = config['checkpoint_dir']

    n_train = X_genre_train.shape[0]
    n_val   = X_genre_val.shape[0]

    os.makedirs(checkpoint_dir, exist_ok=True)
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if config.get('use_item_pool_for_genome'):
        pool_tag = 'ipool'
    elif config.get('use_item_pool_for_history') and config.get('use_user_genome_pool', True):
        pool_tag = 'ipool_gpool'
    elif config.get('use_user_genome_pool', True):
        pool_tag = 'gpool'
    else:
        pool_tag = 'nopool'
    gctx_tag      = '_gctx' if config.get('use_user_genome_context') else ''
    arch_tag      = f'{pool_tag}{gctx_tag}_proj' if config.get('proj_hidden') else f'{pool_tag}{gctx_tag}'
    best_val_loss = float('inf')
    best_path     = os.path.join(checkpoint_dir, f'best_softmax_{arch_tag}_{run_timestamp}.pth')

    loss_train = []

    print(f"\nStarting softmax training ({training_steps:,} steps, "
          f"batch={minibatch_size}, temp={temperature}) ...")
    print(f"  Train: {n_train:,} examples  |  Val: {n_val:,} examples")
    start = time.time()

    from tqdm import tqdm
    pbar = tqdm(range(training_steps), desc="Training (softmax)")
    for i in pbar:
        is_val = (i % log_every == 0)

        if is_val:
            model.eval()
            with torch.no_grad():
                vidx = torch.randint(0, n_val, (minibatch_size,)).tolist()
                vhp  = pad_history_batch([X_history_val[j] for j in vidx], pad_idx)
                vrp  = pad_history_ratings_batch([X_history_ratings_val[j] for j in vidx])
                U    = model.user_embedding(X_genre_val[vidx], vhp, vrp, timestamp_val[vidx])
                tids = target_movieId_val[vidx]
                V    = model.item_embedding(
                        model.genre_context_buffer[tids], model.tag_context_buffer[tids],
                        model.genome_context_buffer[tids], model.year_context_buffer[tids],
                        tids)
                scores   = (U @ V.T) / temperature
                labels   = torch.arange(len(vidx))
                val_loss = F.cross_entropy(scores, labels).item()

                if i == 0:
                    print(f"  [step 0 diagnostics] dot products — "
                          f"mean={scores.mean().item():.4f}  std={scores.std().item():.4f}  "
                          f"min={scores.min().item():.4f}  max={scores.max().item():.4f}")
                    print(f"  [step 0 diagnostics] random baseline loss = log({minibatch_size}) = {np.log(minibatch_size):.4f}")

            avg_train  = np.mean(loss_train[i - log_every:i]) if i >= log_every else (loss_train[-1] if loss_train else 0.0)
            elapsed    = time.time() - start
            start      = time.time()
            current_lr = scheduler.get_last_lr()[0] if i > 0 else config['lr']
            pbar.set_postfix(train=f"{avg_train:.4f}", val=f"{val_loss:.4f}")
            print(f"[{i:06d}]  train_loss={avg_train:.4f}  val_loss={val_loss:.4f}  "
                  f"lr={current_lr:.6f}  ({elapsed:.0f}s)")

            if i > 0 and val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_path)
                print(f"  → new best {best_val_loss:.4f} → {best_path}")

            if i > 0 and i % checkpoint_every == 0:
                periodic = os.path.join(checkpoint_dir,
                                        f'softmax_{arch_tag}_{run_timestamp}_step_{i:06d}.pth')
                torch.save(model.state_dict(), periodic)
                print(f"  → periodic checkpoint → {periodic}")
        else:
            model.train()
            ix  = torch.randint(0, n_train, (minibatch_size,)).tolist()
            hp  = pad_history_batch([X_history_train[j] for j in ix], pad_idx)
            rp  = pad_history_ratings_batch([X_history_ratings_train[j] for j in ix])
            U    = model.user_embedding(X_genre_train[ix], hp, rp, timestamp_train[ix])
            tids = target_movieId_train[ix]
            V    = model.item_embedding(
                       model.genre_context_buffer[tids], model.tag_context_buffer[tids],
                       model.genome_context_buffer[tids], model.year_context_buffer[tids],
                       tids)
            scores = (U @ V.T) / temperature
            labels = torch.arange(len(ix))
            loss   = F.cross_entropy(scores, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_train.append(loss.item())

    print(f"\nSoftmax training complete. Best val loss: {best_val_loss:.4f}")
    print(f"Best checkpoint: {best_path}")
    return best_path
