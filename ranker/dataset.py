"""
Stage 1+ — Ranker dataset.

Loads ranker_candidates_{train,val}.parquet and pre-builds a `(n_movies, item_dim)`
item-feature matrix once from FeatureStore + ranker_movie_stats.parquet.

DEVIATION FROM PLAN: the plan says "expand into flat list of (user_feat, item_feat,
label) tuples at __init__". With 3.4M rows × 100 candidates = 340M tuples, that
would need many GB of RAM. We instead keep compact per-row arrays and let the
training loop sample (row_idx, candidate_position) on the fly. Item features are
looked up by `item_features[corpus_idx]` (cheap tensor indexing).

Imports src/dataset.py for `load_features` (canonical FeatureStore loader). The
plan's "ZERO src/ imports" rule was relaxed — see Import Rules section.
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
# User feature dim = genre_ctx(40) + user_avg(1) + user_count_log1p(1) = 42

CANDIDATES_PER_ROW = 250   # 1 label + 249 negs


# ── Item feature matrix ──────────────────────────────────────────────────────

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

    # Movie stats keyed by movieId; index in same order as fs.top_movies
    stats = movie_stats_df.set_index('movieId')

    # Year normalization: parse int years, scale to [0, 1]
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
    shared `item_features` tensor (built once from FeatureStore + movie_stats).

    Not a torch.utils.data.Dataset because we don't use DataLoader — training
    samples batches directly via sample_batch().
    """

    def __init__(self, parquet_path: str, fs: FeatureStore, movie_stats_df: pd.DataFrame,
                 item_features: torch.Tensor | None = None):
        df = pd.read_parquet(parquet_path)

        self.label_idx     = df['label_corpus_idx'].values.astype(np.int64)              # (N,)
        self.neg_idx       = np.stack(df['neg_corpus_idxs'].values).astype(np.int64)     # (N, 249)
        self.cg_label_rank = df['cg_label_rank'].values.astype(np.int32)                 # (N,)

        # Interaction features: CG score passthrough + genome cosine similarity.
        # Present only in parquets produced by the updated precompute.py.
        # n_interaction_features=0 preserves backward compat with old parquets.
        if 'cg_label_score' in df.columns:
            self.cg_label_score      = df['cg_label_score'].values.astype(np.float32)          # (N,)
            self.cg_neg_scores       = np.stack(df['cg_neg_scores'].values).astype(np.float32) # (N, 99)
            self.genome_cosine_label = df['genome_cosine_label'].values.astype(np.float32)     # (N,)
            self.genome_cosine_negs  = np.stack(df['genome_cosine_negs'].values).astype(np.float32)  # (N, 99)
            self.n_interaction_features = 2
        else:
            self.cg_label_score = self.cg_neg_scores = None
            self.genome_cosine_label = self.genome_cosine_negs = None
            self.n_interaction_features = 0

        self.user_genre = np.stack(df['user_genre_ctx'].values).astype(np.float32)       # (N, 40)
        self.user_avg   = df['user_avg_rating'].values.astype(np.float32)                # (N,)
        # user_rating_count stored as raw int in parquet — apply log1p here per plan.
        self.user_count_log1p = np.log1p(df['user_rating_count'].values).astype(np.float32)  # (N,)

        # Item feature matrix is shared across train/val (same FeatureStore + stats)
        self.item_features = item_features if item_features is not None \
                              else _build_item_features(fs, movie_stats_df)

        # Per-corpus-item interaction counts (np.float32, n_movies) — used for Menon α logit adjustment.
        # Loaded from fs.movie_interaction_counts (data/movie_interaction_counts_v2.npy).
        # If missing (very old setup), pop_alpha must be 0 — we don't fabricate counts.
        self.movie_interaction_counts = (
            np.asarray(fs.movie_interaction_counts, dtype=np.float32)
            if fs.movie_interaction_counts is not None else None
        )

        self.N        = len(self.label_idx)
        self.user_dim = self.user_genre.shape[1] + 2          # +2 for avg, count_log1p
        self.item_dim = self.item_features.shape[1]
        self.n_neg    = self.neg_idx.shape[1]                  # 99 by construction

        # Pre-cache user-feature concat as one (N, user_dim) tensor — small (~600 MB train),
        # avoids re-concatenating per batch / per eval step.
        self._user_features_full = torch.from_numpy(np.concatenate([
            self.user_genre,
            self.user_avg.reshape(-1, 1),
            self.user_count_log1p.reshape(-1, 1),
        ], axis=1))  # (N, user_dim)

    def user_features_for_rows(self, row_idx: np.ndarray) -> torch.Tensor:
        """Return (len(row_idx), user_dim) on whatever device user_features lives on."""
        return self._user_features_full[row_idx]

    def to(self, device: torch.device):
        """Move feature tensors to device (call once after construction)."""
        self.item_features        = self.item_features.to(device)
        self._user_features_full  = self._user_features_full.to(device)
        return self


# ── Random-tuple batch sampler ───────────────────────────────────────────────

def sample_batch(dataset: RankerDataset, batch_size: int, device: torch.device,
                 rng: np.random.Generator, easy_neg_frac: float = 0.5):
    """
    Mixed Negative Sampling (MNS): each negative is independently drawn from either
    the CG hard-negative pool (precomputed top-K) or a random corpus item.

    candidate_position == 0           → label (positive, label=1)
    candidate_position 1..n_neg, hard → CG top-K negative (high CG score)
    candidate_position 1..n_neg, easy → uniform random corpus item

    Hard negatives teach the model to distinguish near-misses (CG thought these were good).
    Easy negatives anchor the global decision boundary and prevent embedding collapse
    toward the CG candidate distribution.

    For easy negatives, interaction features (cg_score, genome_cosine) are set to 0 —
    CG never scored them, so there is no retrieval signal to pass through.

    Returns (user_feat, item_feat, cand_t, label) all on `device`.
    `cand_t` is the per-batch corpus indices — needed for Menon α popularity-bias lookup.
    """
    rows     = rng.integers(0, dataset.N, size=batch_size)
    pos      = rng.integers(0, CANDIDATES_PER_ROW, size=batch_size)
    is_label = (pos == 0)
    is_neg   = ~is_label

    neg_col       = np.maximum(pos - 1, 0)
    cand_idx_hard = np.where(is_label, dataset.label_idx[rows], dataset.neg_idx[rows, neg_col])

    # Easy negatives: uniform random across the full corpus.
    n_movies  = dataset.item_features.shape[0]
    use_easy  = is_neg & (rng.random(batch_size) < easy_neg_frac)
    rand_idx  = rng.integers(0, n_movies, size=batch_size)
    cand_idx  = np.where(use_easy, rand_idx, cand_idx_hard)

    cand_t  = torch.from_numpy(cand_idx).long().to(device)
    user_t  = dataset.user_features_for_rows(rows).to(device)
    item_t  = dataset.item_features[cand_t]
    label_t = torch.from_numpy(is_label.astype(np.float32)).to(device)

    if dataset.n_interaction_features > 0:
        cg_score   = np.where(is_label,
                               dataset.cg_label_score[rows],
                               dataset.cg_neg_scores[rows, neg_col])
        genome_cos = np.where(is_label,
                               dataset.genome_cosine_label[rows],
                               dataset.genome_cosine_negs[rows, neg_col])
        # Easy negatives were never scored by CG — zero out retrieval signal.
        cg_score   = np.where(use_easy, 0.0, cg_score).astype(np.float32)
        genome_cos = np.where(use_easy, 0.0, genome_cos).astype(np.float32)
        interact_t = torch.from_numpy(
            np.stack([cg_score, genome_cos], axis=1)
        ).to(device)
        item_t = torch.cat([item_t, interact_t], dim=1)

    return user_t, item_t, cand_t, label_t


# ── Public loader (used by train.py / evaluate.py) ──────────────────────────

def load_splits(data_dir: str = 'data') -> tuple:
    """Returns (train_dataset, val_dataset) sharing the same item_features tensor."""
    fs           = load_features(data_dir=data_dir, version='v1')
    stats_df     = pd.read_parquet(os.path.join(data_dir, 'ranker_movie_stats.parquet'))
    item_feats   = _build_item_features(fs, stats_df)

    print(f"Item features: shape={tuple(item_feats.shape)}  dtype={item_feats.dtype}")

    train_ds = RankerDataset(os.path.join(data_dir, 'ranker_candidates_train.parquet'),
                              fs, stats_df, item_features=item_feats)
    val_ds   = RankerDataset(os.path.join(data_dir, 'ranker_candidates_val.parquet'),
                              fs, stats_df, item_features=item_feats)
    print(f"Train: {train_ds.N:,} rollback rows  |  Val: {val_ds.N:,} rollback rows")
    print(f"User dim: {train_ds.user_dim}  |  Item dim: {train_ds.item_dim}  |  "
          f"MLP input dim: {train_ds.user_dim + train_ds.item_dim}")
    return train_ds, val_ds
