"""
Training loop (MSE rollback).

Usage:
    python main.py train rollback
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

    history_dim      = m.item_embedding_lookup.embedding_dim
    genome_dim       = _out_dim(m.item_genome_tag_tower)
    genre_dim        = _out_dim(m.user_genre_tower)
    ts_dim           = m.timestamp_embedding_lookup.embedding_dim
    item_genre_dim   = _out_dim(m.item_genre_tower)
    item_tag_dim     = _out_dim(m.item_tag_tower)
    item_movieId_dim = _out_dim(m.item_embedding_tower)
    year_dim         = _out_dim(m.year_embedding_tower)
    item_total       = item_genre_dim + item_tag_dim + genome_dim + item_movieId_dim + year_dim
    n_params         = sum(p.nelement() for p in model.parameters() if p.requires_grad)

    gctx_dim   = _out_dim(m.user_genome_context_tower)
    user_total = history_dim + genome_dim + genre_dim + ts_dim + gctx_dim
    user_desc  = (f"item_id_pool({history_dim}) + genome_pool({genome_dim}) + "
                  f"genre_tower({genre_dim}) + ts_emb({ts_dim}) + genome_ctx({gctx_dim})")

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
                    hb      = pad_history_batch([X_history_val[j] for j in vix], pad_idx)
                    rb      = pad_history_ratings_batch([X_history_ratings_val[j] for j in vix])
                    vp      = model(X_val[v0:v1], hb, rb, timestamp_val[v0:v1],
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
