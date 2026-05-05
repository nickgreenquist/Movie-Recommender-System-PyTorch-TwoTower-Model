"""
Stage 2 — Ranker dataset (CG-parity feature set).

Loads ranker_candidates_{train,val}.parquet plus the item-feature matrix used
for on-the-fly cross-feature computation. The model itself does not consume
item_features — it has its own registered buffers (built from the FeatureStore
in train.py via build_ranker). The dataset's item_features tensor exists only
to support cross-feature computation in sample_batch.

sample_batch returns 7-tuple:
  user_genre_ctx, X_history, X_hist_ratings, timestamp_bin, cand_idx, label,
  cross_features (B, n_cross)

Cross features are the wide-bypass scalars: [genome_cosine, genre_affinity,
era_gap, rating_calibration, popularity_match]. Computed here so the model
forward can be a pure tower-and-MLP forward pass.

Imports src/dataset.py for `load_features` (canonical FeatureStore loader).
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset import FeatureStore, load_features


# Item feature dim = genome(1128) + genre(20) + global_avg(1) + global_count_log1p(1) + year_norm(1)
# User feature dim (raw) = genre_ctx(40) + user_avg(1) + user_count_log1p(1) = 42

CANDIDATES_PER_ROW = 250   # 1 label + 249 negs


# ── Item feature matrix (used only for cross-feature computation) ────────────

def _build_item_features(fs: FeatureStore, movie_stats_df: pd.DataFrame) -> torch.Tensor:
    """
    Build (n_movies, item_dim) tensor where row i = features for corpus index i.

    Columns (in order, all float32):
      [0:1128]    genome scores
      [1128:1148] genre one-hot
      [1148]      global_avg_rating  (raw, ~[1, 5])
      [1149]      global_rating_count (log1p — already applied in parquet)
      [1150]      release_year_normalized to [0, 1]
    """
    n_movies = len(fs.top_movies)
    n_genome = len(fs.genome_tag_ids)
    n_genres = len(fs.genres_ordered)

    stats = movie_stats_df.set_index('movieId')

    int_years = []
    for mid in fs.top_movies:
        y = fs.movieId_to_year[mid]
        try:
            int_years.append(int(y))
        except (ValueError, TypeError):
            int_years.append(0)
    int_years = np.array(int_years, dtype=np.float32)
    valid = int_years > 0
    if valid.any():
        y_min, y_max = int_years[valid].min(), int_years[valid].max()
        year_norm = np.where(valid, (int_years - y_min) / max(y_max - y_min, 1), 0.0).astype(np.float32)
    else:
        year_norm = np.zeros(n_movies, dtype=np.float32)

    item_dim = n_genome + n_genres + 3
    out = np.zeros((n_movies, item_dim), dtype=np.float32)

    for i, mid in enumerate(fs.top_movies):
        out[i, 0:n_genome]                    = fs.movieId_to_genome_tag_context[mid]
        out[i, n_genome:n_genome + n_genres]  = fs.movieId_to_genre_context[mid]
        row = stats.loc[mid] if mid in stats.index else None
        out[i, n_genome + n_genres]     = float(row['global_avg_rating'])      if row is not None else 0.0
        out[i, n_genome + n_genres + 1] = float(row['global_rating_count'])    if row is not None else 0.0
        out[i, n_genome + n_genres + 2] = year_norm[i]

    return torch.from_numpy(out)


# ── Dataset class ────────────────────────────────────────────────────────────

class RankerDataset:
    """
    Holds compact per-row arrays from one ranker_candidates_{split}.parquet plus a
    shared `item_features` tensor used for cross-feature computation in sample_batch.

    The model has its own registered buffers (built from FeatureStore via build_ranker
    in train.py) — it does NOT consume self.item_features.
    """

    def __init__(self, parquet_path: str, fs: FeatureStore, movie_stats_df: pd.DataFrame,
                 item_features: torch.Tensor | None = None):
        df = pd.read_parquet(parquet_path)

        # Corpus indices fit in int32 (corpus has ~9.4k movies, well under int32 max).
        # Upcasting to int64 wasted ~3 GB for neg_idx alone.
        self.label_idx     = df['label_corpus_idx'].values.astype(np.int32)              # (N,)
        self.neg_idx       = np.stack(df['neg_corpus_idxs'].values).astype(np.int32)     # (N, 249)
        self.cg_label_rank = df['cg_label_rank'].values.astype(np.int32)                 # (N,)

        # Genome cosine — parquet-sourced wide-bypass cross feature.
        # cg_neg_scores is intentionally NOT loaded — the CG score feature is disabled
        # (n_interaction_features comment in sample_batch). Loading it cost ~3 GB for nothing.
        # If/when re-enabled, restore: np.stack(df['cg_neg_scores'].values).astype(np.float32)
        if 'genome_cosine_label' in df.columns:
            self.cg_label_score      = None
            self.cg_neg_scores       = None
            self.genome_cosine_label = df['genome_cosine_label'].values.astype(np.float32)
            self.genome_cosine_negs  = np.stack(df['genome_cosine_negs'].values).astype(np.float32)
            self.has_genome_cosine = True
        else:
            self.cg_label_score = self.cg_neg_scores = None
            self.genome_cosine_label = self.genome_cosine_negs = None
            self.has_genome_cosine = False

        self.user_genre = np.stack(df['user_genre_ctx'].values).astype(np.float32)       # (N, 40)
        self.user_avg   = df['user_avg_rating'].values.astype(np.float32)                # (N,)
        self.user_count_log1p = np.log1p(df['user_rating_count'].values).astype(np.float32)  # (N,)

        # History + timestamp — required for the user tower.
        # X_history is right-aligned, padded with len(fs.top_movies) (model.pad_idx).
        self.X_history     = np.stack(df['history_ids_padded'].values).astype(np.int64)  # (N, max_hist)
        self.X_hist_rat    = np.stack(df['history_ratings_padded'].values).astype(np.float32)
        self.timestamp_bin = df['timestamp_bin'].values.astype(np.int64)                 # (N,)

        # Item feature matrix is shared across train/val (same FeatureStore + stats).
        # Used by sample_batch / evaluate / canary for cross-feature computation only.
        self.item_features = item_features if item_features is not None \
                              else _build_item_features(fs, movie_stats_df)

        # Per-corpus-item interaction counts — used for Menon α logit adjustment.
        self.movie_interaction_counts = (
            np.asarray(fs.movie_interaction_counts, dtype=np.float32)
            if fs.movie_interaction_counts is not None else None
        )

        self.N        = len(self.label_idx)
        self.user_dim = self.user_genre.shape[1] + 2          # +2 for avg, count_log1p
        self.item_dim = self.item_features.shape[1]
        self.n_neg    = self.neg_idx.shape[1]
        self.max_hist = self.X_history.shape[1]
        self.pad_idx  = len(fs.top_movies)                     # for X_history

        # Era Bias precompute: user's mean release year_norm over watch history.
        # item_features[:, 1150] = year_norm. Padding indices clamped, masked via valid.
        n_mov    = self.item_features.shape[0]
        clamped  = np.clip(self.X_history, 0, n_mov - 1)
        year_col = self.item_features[:, 1150].cpu().numpy()                              # (n_movies,)
        hist_years = year_col[clamped]                                                     # (N, max_hist)
        valid      = (self.X_history < n_mov).astype(np.float32)
        hist_cnt   = valid.sum(axis=1).clip(min=1.0)
        self.user_mean_year_norm = torch.from_numpy(
            ((hist_years * valid).sum(axis=1) / hist_cnt).astype(np.float32)
        )
        del clamped, hist_years, valid

        # Pre-cache user-feature tensors.
        self._user_genre_t      = torch.from_numpy(self.user_genre)
        self._user_avg_t        = torch.from_numpy(self.user_avg)
        self._user_count_log1p_t = torch.from_numpy(self.user_count_log1p)
        self._X_history_t        = torch.from_numpy(self.X_history)
        self._X_hist_rat_t       = torch.from_numpy(self.X_hist_rat)
        self._timestamp_t        = torch.from_numpy(self.timestamp_bin)

    def to(self, device: torch.device):
        """
        Move feature tensors to device (call once after construction).

        After moving, drops the CPU-side numpy duplicates for arrays that are only
        accessed via the cached tensors (sample_batch / evaluate use the _xxx_t
        attributes). Saves ~2.4 GB on Apple Silicon (MPS=unified=RAM, so device copies
        are NOT free — they double-count without this cleanup).
        """
        self.item_features         = self.item_features.to(device)
        self.user_mean_year_norm   = self.user_mean_year_norm.to(device)
        self._user_genre_t         = self._user_genre_t.to(device)
        self._user_avg_t           = self._user_avg_t.to(device)
        self._user_count_log1p_t   = self._user_count_log1p_t.to(device)
        self._X_history_t          = self._X_history_t.to(device)
        self._X_hist_rat_t         = self._X_hist_rat_t.to(device)
        self._timestamp_t          = self._timestamp_t.to(device)
        # Drop CPU-side duplicates — sample_batch / evaluate use the _t versions only.
        del self.user_genre, self.user_avg, self.user_count_log1p
        del self.X_history, self.X_hist_rat, self.timestamp_bin
        return self


# ── Cross-feature computation (shared by sample_batch + evaluate + canary) ──

def compute_cross_features(user_genre_ctx: torch.Tensor,
                           user_avg: torch.Tensor,
                           user_count_log1p: torch.Tensor,
                           user_mean_year_norm: torch.Tensor,
                           cand_genre_oh: torch.Tensor,
                           cand_year_norm: torch.Tensor,
                           cand_global_avg: torch.Tensor,
                           cand_log_count: torch.Tensor,
                           genome_cosine: torch.Tensor) -> torch.Tensor:
    """
    All inputs are (B, ...) tensors on the same device.

      genome_cosine        : (B,)        — parquet-sourced
      user_genre_ctx       : (B, 40)
      user_avg / cnt / yr  : (B,)
      cand_genre_oh        : (B, n_genres) e.g. (B, 20) — item one-hot
      cand_year_norm       : (B,)
      cand_global_avg      : (B,)
      cand_log_count       : (B,)

    Returns (B, 5) — [genome_cosine, genre_affinity, era_gap, rating_cal, pop_match].
    """
    # Cross #1: Weighted Genre Affinity
    user_genre_avg = user_genre_ctx[:, :cand_genre_oh.shape[1]]                # (B, 20)
    genre_dot      = (user_genre_avg * cand_genre_oh).sum(dim=1)               # (B,)
    genre_cnt      = cand_genre_oh.sum(dim=1).clamp(min=1.0)
    genre_affinity = genre_dot / genre_cnt

    # Cross #2: Era Bias
    era_gap = torch.abs(user_mean_year_norm - cand_year_norm)

    # Cross #3: Rating Calibration
    rating_cal = user_avg - cand_global_avg

    # Cross #4: Popularity Match
    pop_match = torch.abs(user_count_log1p - cand_log_count)

    return torch.stack([genome_cosine, genre_affinity, era_gap, rating_cal, pop_match], dim=1)


# ── Random-tuple batch sampler ───────────────────────────────────────────────

def sample_batch(dataset: RankerDataset, batch_size: int, device: torch.device,
                 rng: np.random.Generator, easy_neg_frac: float = 0.5):
    """
    Mixed Negative Sampling (MNS). See class docstring for shape contract.

    Returns:
      user_genre_ctx (B, 40), X_history (B, H), X_hist_ratings (B, H), timestamp (B,),
      cand_idx (B,), label (B,), cross_features (B, 5)
    """
    rows     = rng.integers(0, dataset.N, size=batch_size)
    pos      = rng.integers(0, CANDIDATES_PER_ROW, size=batch_size)
    is_label = (pos == 0)
    is_neg   = ~is_label

    neg_col       = np.maximum(pos - 1, 0)
    cand_idx_hard = np.where(is_label, dataset.label_idx[rows], dataset.neg_idx[rows, neg_col])

    n_movies  = dataset.item_features.shape[0]
    use_easy  = is_neg & (rng.random(batch_size) < easy_neg_frac)
    rand_idx  = rng.integers(0, n_movies, size=batch_size)
    cand_idx  = np.where(use_easy, rand_idx, cand_idx_hard)

    rows_t   = torch.from_numpy(rows).long()
    cand_t   = torch.from_numpy(cand_idx).long().to(device)
    label_t  = torch.from_numpy(is_label.astype(np.float32)).to(device)

    user_genre_ctx_t = dataset._user_genre_t[rows_t].to(device)
    X_history_t      = dataset._X_history_t[rows_t].to(device)
    X_hist_rat_t     = dataset._X_hist_rat_t[rows_t].to(device)
    timestamp_t      = dataset._timestamp_t[rows_t].to(device)

    # Cross-feature scalars
    if dataset.has_genome_cosine:
        gc = np.where(is_label,
                      dataset.genome_cosine_label[rows],
                      dataset.genome_cosine_negs[rows, neg_col])
        gc = np.where(use_easy, 0.0, gc).astype(np.float32)   # easy negs were never CG-scored
        genome_cosine_t = torch.from_numpy(gc).to(device)
    else:
        genome_cosine_t = torch.zeros(batch_size, device=device)

    cand_feat       = dataset.item_features[cand_t]                      # (B, item_dim)
    cand_genre_oh   = cand_feat[:, 1128:1148]                            # (B, 20)
    cand_global_avg = cand_feat[:, 1148]                                 # (B,)
    cand_log_count  = cand_feat[:, 1149]                                 # (B,)
    cand_year_norm  = cand_feat[:, 1150]                                 # (B,)

    user_avg_b   = dataset._user_avg_t[rows_t].to(device)
    user_cnt_b   = dataset._user_count_log1p_t[rows_t].to(device)
    user_year_b  = dataset.user_mean_year_norm[rows_t.to(device)]

    cross_features = compute_cross_features(
        user_genre_ctx_t, user_avg_b, user_cnt_b, user_year_b,
        cand_genre_oh, cand_year_norm, cand_global_avg, cand_log_count,
        genome_cosine_t,
    )

    return (user_genre_ctx_t, X_history_t, X_hist_rat_t, timestamp_t,
            cand_t, label_t, cross_features)


# ── Public loader ───────────────────────────────────────────────────────────

def load_splits(data_dir: str = 'data') -> tuple:
    """Returns (train_dataset, val_dataset, FeatureStore)."""
    fs         = load_features(data_dir=data_dir, version='v1')
    stats_df   = pd.read_parquet(os.path.join(data_dir, 'ranker_movie_stats.parquet'))
    item_feats = _build_item_features(fs, stats_df)

    print(f"Item features (for cross-feature computation): shape={tuple(item_feats.shape)}  dtype={item_feats.dtype}")

    train_ds = RankerDataset(os.path.join(data_dir, 'ranker_candidates_train.parquet'),
                              fs, stats_df, item_features=item_feats)
    val_ds   = RankerDataset(os.path.join(data_dir, 'ranker_candidates_val.parquet'),
                              fs, stats_df, item_features=item_feats)
    print(f"Train: {train_ds.N:,} rollback rows  |  Val: {val_ds.N:,} rollback rows")
    print(f"User raw dim: {train_ds.user_dim}  |  Item raw dim: {train_ds.item_dim}  |  "
          f"max_hist: {train_ds.max_hist}")
    return train_ds, val_ds, fs
