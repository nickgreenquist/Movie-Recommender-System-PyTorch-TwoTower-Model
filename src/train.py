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
    # ── Embedding sizes ───────────────────────────────────────────────────────
    # item_movieId_embedding_size also controls the user history embedding size:
    # history pooling uses the shared item_embedding_lookup, so its output dim
    # equals this value. They cannot be set independently.
    item_movieId_embedding_size      = 40
    item_year_embedding_size         = 5
    timestamp_feature_embedding_size = 5
    item_tag_embedding_size          = 10
    # item_genome_tag_embedding_size also controls the user genome pool embedding size:
    # genome pooling uses the shared item_genome_tag_tower, so its output dim
    # equals this value. They cannot be set independently.
    item_genome_tag_embedding_size   = 35
    user_genre_embedding_size        = 30
    item_genre_embedding_size        = 20

    user_total = item_movieId_embedding_size + item_genome_tag_embedding_size + user_genre_embedding_size + timestamp_feature_embedding_size
    item_total = item_genre_embedding_size + item_tag_embedding_size + item_genome_tag_embedding_size + item_movieId_embedding_size + item_year_embedding_size
    assert user_total == item_total, (
        f"Tower size mismatch — user={user_total} "
        f"(history={item_movieId_embedding_size} + genome={item_genome_tag_embedding_size} + genre={user_genre_embedding_size} + ts={timestamp_feature_embedding_size}), "
        f"item={item_total} "
        f"(genre={item_genre_embedding_size} + tag={item_tag_embedding_size} + genome={item_genome_tag_embedding_size} + movieId={item_movieId_embedding_size} + year={item_year_embedding_size})"
    )

    return {
        # Embedding sizes
        'item_movieId_embedding_size':      item_movieId_embedding_size,
        'item_year_embedding_size':         item_year_embedding_size,
        'timestamp_feature_embedding_size': timestamp_feature_embedding_size,
        'item_tag_embedding_size':          item_tag_embedding_size,
        'item_genome_tag_embedding_size':   item_genome_tag_embedding_size,
        'user_genre_embedding_size':        user_genre_embedding_size,
        'item_genre_embedding_size':        item_genre_embedding_size,
        # Training
        'lr':               0.005,
        'momentum':         0.9,
        'minibatch_size':   64,
        'training_steps':   150_000,
        'log_every':        10_000,
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
    )
    return model


def print_model_summary(model: MovieRecommender) -> None:
    """Print tower dimensions and parameter count for a built model."""
    m = model
    history_dim      = m.item_embedding_lookup.embedding_dim
    genome_dim       = m.item_genome_tag_tower[0].out_features
    genre_dim        = m.user_genre_tower[0].out_features
    ts_dim           = m.timestamp_embedding_lookup.embedding_dim
    user_total       = history_dim + genome_dim + genre_dim + ts_dim
    item_genre_dim   = m.item_genre_tower[0].out_features
    item_tag_dim     = m.item_tag_tower[0].out_features
    genome_tag_dim   = m.item_genome_tag_tower[0].out_features
    item_movieId_dim = m.item_embedding_tower[0].out_features
    year_dim         = m.year_embedding_tower[0].out_features
    item_total       = item_genre_dim + item_tag_dim + genome_tag_dim + item_movieId_dim + year_dim
    n_params         = sum(p.nelement() for p in model.parameters() if p.requires_grad)

    print(f"\n── Model dimensions ──")
    print(f"  User side:  history({history_dim}) + genome({genome_dim}) + genre({genre_dim}) + ts({ts_dim})  =  {user_total}")
    print(f"  Item side:  movieId({item_movieId_dim}) + genome({genome_tag_dim}) + genre({item_genre_dim})"
          f" + tag({item_tag_dim}) + year({year_dim})  =  {item_total}")
    print(f"  Parameters: {n_params:,}")


# ── Training loop ─────────────────────────────────────────────────────────────

def train(model: MovieRecommender, train_data: tuple, val_data: tuple,
          config: dict, fs: FeatureStore) -> str:
    """
    Run the training loop. Returns the path of the best checkpoint.

    train_data / val_data: 10-tuple from dataset.build_dataset()
      (X, X_history, X_history_ratings, timestamp, Y,
       target_movieId, target_movieId_genre, target_movieId_tag,
       target_movieId_genome, target_movieId_year)
    """
    (X_train, X_history_train, X_history_ratings_train, timestamp_train,
     Y_train, target_movieId_train, target_movieId_genre_train, target_movieId_tag_train,
     target_movieId_genome_train, target_movieId_year_train) = train_data

    (X_val, X_history_val, X_history_ratings_val, timestamp_val,
     Y_val, target_movieId_val, target_movieId_genre_val, target_movieId_tag_val,
     target_movieId_genome_val, target_movieId_year_val) = val_data

    print_model_summary(model)

    pad_idx          = len(fs.top_movies)
    loss_fn          = torch.nn.MSELoss()
    optimizer        = torch.optim.SGD(model.parameters(),
                                        lr=config['lr'], momentum=config['momentum'])
    minibatch_size   = config['minibatch_size']
    training_steps   = config['training_steps']
    log_every        = config['log_every']
    checkpoint_every = config['checkpoint_every']
    checkpoint_dir   = config['checkpoint_dir']

    os.makedirs(checkpoint_dir, exist_ok=True)
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    best_val_loss = float('inf')
    best_path     = os.path.join(checkpoint_dir, f'best_checkpoint_{run_timestamp}.pth')

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
                                      target_movieId_genre_val[v0:v1],
                                      target_movieId_tag_val[v0:v1],
                                      target_movieId_genome_val[v0:v1],
                                      target_movieId_year_val[v0:v1],
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
                          timestamp_train[ix],
                          target_movieId_genre_train[ix], target_movieId_tag_train[ix],
                          target_movieId_genome_train[ix], target_movieId_year_train[ix],
                          target_movieId_train[ix])
            output = loss_fn(preds, Y_train[ix])
            optimizer.zero_grad()
            output.backward()
            optimizer.step()
            loss_train.append(output.item())

        if is_val:
            elapsed = time.time() - start
            start   = time.time()
            avg_train = np.mean(loss_train[i-log_every:i]) if i >= log_every else loss_train[-1] if loss_train else 0.0
            val_loss  = output_val
            pbar.set_postfix(train=f"{avg_train:.4f}", val=f"{val_loss:.4f}")
            print(f"[{i:06d}]  train_loss={avg_train:.4f}  val_loss={val_loss:.4f}  "
                  f"({elapsed:.0f}s)")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_path)
                print(f"  → new best {best_val_loss:.4f} → {best_path}")

            if i > 0 and i % checkpoint_every == 0:
                periodic = os.path.join(checkpoint_dir,
                                        f'checkpoint_{run_timestamp}_step_{i:06d}.pth')
                torch.save(model.state_dict(), periodic)
                print(f"  → periodic checkpoint → {periodic}")

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Best checkpoint: {best_path}")
    return best_path
