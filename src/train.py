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
from tqdm import tqdm

from src.dataset import FeatureStore, pad_history_batch, pad_history_ratings_batch
from src.model import MovieRecommender


# ── Hyperparameters ───────────────────────────────────────────────────────────

def get_config() -> dict:
    """
    All training hyperparameters in one place.

    Tower flags:
      user side: history(40) + user_genre(70) + ts(10) + user_tag = 120
      item side: item_genre(20) + item_tag(15) + genome(35) + movieId(40) + year(10) = 120
        user_genre = 70 - user_tag_embedding_size
        item_genre = 70 - item_tag_embedding_size - item_genome_tag_embedding_size
    """
    USE_USER_TAG_TOWER        = False
    USE_ITEM_TAG_TOWER        = True
    USE_ITEM_GENOME_TAG_TOWER = True

    item_movieId_embedding_size      = 40
    item_year_embedding_size         = 10
    timestamp_feature_embedding_size = 10

    user_tag_embedding_size        = 20 if USE_USER_TAG_TOWER        else 0
    item_tag_embedding_size        = 15 if USE_ITEM_TAG_TOWER        else 0
    item_genome_tag_embedding_size = 35 if USE_ITEM_GENOME_TAG_TOWER else 0

    user_genre_embedding_size = 70 - user_tag_embedding_size
    item_genre_embedding_size = 70 - item_tag_embedding_size - item_genome_tag_embedding_size

    return {
        # Tower flags
        'USE_USER_TAG_TOWER':        USE_USER_TAG_TOWER,
        'USE_ITEM_TAG_TOWER':        USE_ITEM_TAG_TOWER,
        'USE_ITEM_GENOME_TAG_TOWER': USE_ITEM_GENOME_TAG_TOWER,
        # Embedding sizes
        'item_movieId_embedding_size':      item_movieId_embedding_size,
        'item_year_embedding_size':         item_year_embedding_size,
        'timestamp_feature_embedding_size': timestamp_feature_embedding_size,
        'user_tag_embedding_size':          user_tag_embedding_size,
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
    model = MovieRecommender(
        genres_len=len(fs.genres_ordered),
        tags_len=len(fs.tags_ordered),
        genome_tags_len=len(fs.genome_tag_ids),
        top_movies_len=len(fs.top_movies),
        all_years_len=len(fs.years_ordered),
        timestamp_num_bins=fs.timestamp_num_bins,
        user_context_size=fs.user_context_size,
        item_genre_embedding_size=config['item_genre_embedding_size'],
        item_tag_embedding_size=config['item_tag_embedding_size'],
        item_genome_tag_embedding_size=config['item_genome_tag_embedding_size'],
        item_movieId_embedding_size=config['item_movieId_embedding_size'],
        item_year_embedding_size=config['item_year_embedding_size'],
        user_genre_embedding_size=config['user_genre_embedding_size'],
        user_tag_embedding_size=config['user_tag_embedding_size'],
        timestamp_feature_embedding_size=config['timestamp_feature_embedding_size'],
        use_user_tag_tower=config['USE_USER_TAG_TOWER'],
        use_item_tag_tower=config['USE_ITEM_TAG_TOWER'],
        use_item_genome_tag_tower=config['USE_ITEM_GENOME_TAG_TOWER'],
    )
    n_params = sum(p.nelement() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    return model


# ── Training loop ─────────────────────────────────────────────────────────────

def train(model: MovieRecommender, train_data: tuple, val_data: tuple,
          config: dict, fs: FeatureStore) -> str:
    """
    Run the training loop. Returns the path of the best checkpoint.

    train_data / val_data: 11-tuple from dataset.build_dataset()
      (X, X_history, X_history_ratings, X_tag, timestamp, Y,
       target_movieId, target_movieId_genre, target_movieId_tag,
       target_movieId_genome, target_movieId_year)
    """
    (X_train, X_history_train, X_history_ratings_train, X_tag_train, timestamp_train,
     Y_train, target_movieId_train, target_movieId_genre_train, target_movieId_tag_train,
     target_movieId_genome_train, target_movieId_year_train) = train_data

    (X_val, X_history_val, X_history_ratings_val, X_tag_val, timestamp_val,
     Y_val, target_movieId_val, target_movieId_genre_val, target_movieId_tag_val,
     target_movieId_genome_val, target_movieId_year_val) = val_data

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

    pbar = tqdm(range(training_steps), desc="Training")
    for i in pbar:
        is_val = (i % log_every == 0)

        if is_val:
            hist_batch  = pad_history_batch(X_history_val, pad_idx)
            rat_batch   = pad_history_ratings_batch(X_history_ratings_val)
            model.eval()
            with torch.no_grad():
                preds = model(X_val, X_tag_val, hist_batch, rat_batch, timestamp_val,
                              target_movieId_genre_val, target_movieId_tag_val,
                              target_movieId_genome_val, target_movieId_year_val,
                              target_movieId_val)
                output = loss_fn(preds, Y_val)
            loss_val.append(output.item())
        else:
            ix = torch.randint(0, X_train.shape[0], (minibatch_size,)).tolist()
            hist_batch = pad_history_batch([X_history_train[j] for j in ix], pad_idx)
            rat_batch  = pad_history_ratings_batch([X_history_ratings_train[j] for j in ix])
            model.train()
            preds = model(X_train[ix], X_tag_train[ix], hist_batch, rat_batch,
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
            avg_train = np.mean(loss_train[i-log_every:i]) if i >= log_every else output.item()
            val_loss  = output.item()
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
