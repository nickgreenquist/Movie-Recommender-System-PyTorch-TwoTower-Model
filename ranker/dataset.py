"""
Stage 2 — Ranker dataset (CG-parity feature set).

Loads ranker_candidates_{train,val}.parquet plus the item-feature matrix used
for on-the-fly cross-feature computation. The model itself does not consume
item_features — it has its own registered buffers (built from the FeatureStore
in train.py via build_ranker). The dataset's item_features tensor exists only
to support cross-feature computation done by the caller.

sample_batch returns the raw inputs needed to (1) run the model and
(2) build the wide cross_features tensor. The caller (train / evaluate / canary)
runs user_embedding / item_embedding / user_genome_profile on the model first,
then assembles the 8-feature cross_features tensor by calling compute_cross_features.

Why the model-aware refactor: 3 of the 8 wide features (Dislike Similarity,
Genome Peak Match, Genre Intersection's pool inputs) require model-derived
tensors (user_concat slices, item_id_lookup outputs, user_genome_profile).
sample_batch can't compute them without the model; the cleanest split is to
have sample_batch return raw inputs and let the caller assemble cross_features.

Wide bypass features (12 total):
  1. genome_cosine        — parquet-sourced cosine of user genome pool vs item genome
  2. genre_affinity       — dot(user_genre_avg, item_genre_oh) / sum(item_genre_oh)
  3. era_gap              — abs(user_mean_year_norm - item_year_norm)
  4. rating_cal           — user_avg - item_global_avg
  5. pop_match            — abs(user_count_log1p - item_log_count)             — total user activity vs item pop
  6. jaccard              — Jaccard(user_genre_set, item_genre_set)
  7. genome_peak          — max(user_genome_profile * item_genome_scores)  ← model-aware
  8. dis_sim              — cosine(pool_disliked, item_id_lookup[cand])     ← model-aware
  9. recent_cf_sim        — last_item_emb · item_id_lookup[cand]            ← recency CF
 10. recent_genome_sim    — mean(last_5_genome) · item_genome_scores         ← recency content
 11. genome_residual      — genome_cosine - user_genome_baseline             ← per-user calibration
 12. liked_pop_gap        — cand_log_count - user_mean_liked_log_count       ← popularity taste fit (signed)

Imports src/dataset.py for `load_features` (canonical FeatureStore loader).
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

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

        # Liked-popularity precompute: user's mean log_count over LIKED history items.
        # item_features[:, 1149] = global_rating_count (already log1p). Liked = rating > 0.
        log_count_col   = self.item_features[:, 1149].cpu().numpy()                       # (n_movies,)
        hist_log_counts = log_count_col[clamped]                                           # (N, max_hist)
        liked_mask      = ((self.X_history < n_mov) & (self.X_hist_rat > 0)).astype(np.float32)
        liked_cnt       = liked_mask.sum(axis=1).clip(min=1.0)                             # (N,) — clamp guards 0-liked users
        self.user_mean_liked_log_count = torch.from_numpy(
            ((hist_log_counts * liked_mask).sum(axis=1) / liked_cnt).astype(np.float32)
        )
        del clamped, hist_years, valid, hist_log_counts, liked_mask

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
        self.item_features              = self.item_features.to(device)
        self.user_mean_year_norm        = self.user_mean_year_norm.to(device)
        self.user_mean_liked_log_count  = self.user_mean_liked_log_count.to(device)
        self._user_genre_t              = self._user_genre_t.to(device)
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
                           genome_cosine: torch.Tensor,
                           pool_disliked: torch.Tensor,
                           item_id_raw: torch.Tensor,
                           user_genome_profile: torch.Tensor,
                           cand_genome_scores: torch.Tensor,
                           last_item_emb: torch.Tensor,
                           last_k_mean_genome: torch.Tensor,
                           user_genome_baseline: torch.Tensor,
                           user_mean_liked_log_count: torch.Tensor) -> torch.Tensor:
    """
    All inputs are (B, ...) tensors on the same device.

      genome_cosine        : (B,)        — parquet-sourced
      user_genre_ctx       : (B, 40)
      user_avg / cnt / yr  : (B,)
      cand_genre_oh        : (B, n_genres) e.g. (B, 20) — item one-hot
      cand_year_norm       : (B,)
      cand_global_avg      : (B,)
      cand_log_count       : (B,)
      pool_disliked        : (B, item_id_emb_dim) — sliced from user_concat by caller
      item_id_raw          : (B, item_id_emb_dim) — model.item_id_lookup(cand_idx)
      user_genome_profile  : (B, n_genome_tags)   — model.user_genome_profile(...)
      cand_genome_scores   : (B, n_genome_tags)   — model.genome_buffer[cand_idx]
      last_item_emb        : (B, item_id_emb_dim) — from model.user_recent_signals(...)
      last_k_mean_genome   : (B, n_genome_tags)   — from model.user_recent_signals(...)
      user_genome_baseline       : (B,)            — from model.user_genome_baseline(...)
      user_mean_liked_log_count  : (B,)            — precomputed in RankerDataset.__init__

    Returns (B, 12) — [genome_cosine, genre_affinity, era_gap, rating_cal, pop_match,
                       jaccard, genome_peak, dis_sim, recent_cf_sim, recent_genome_sim,
                       genome_residual, liked_pop_gap].

    Strict ranker-only features (CG cannot meaningfully use them at retrieval):
      - jaccard: set ops, non-linear
      - genome_peak: max operator
      - dis_sim, recent_cf_sim, recent_genome_sim: require multiple user vectors
        (disliked pool, last-item embedding, recent-window genome) — CG's single
        user output can't expose them.
      - genome_residual: per-user constant shift; mathematically computable in CG
        but doesn't change retrieval ordering. Useful only for ranker calibration.
      - liked_pop_gap: subtraction of user-side scalar from item-side scalar (same
        class as era_gap, rating_cal). CG's dot product cannot represent it.
    """
    n_genres = cand_genre_oh.shape[1]

    # Cross #1: Weighted Genre Affinity
    user_genre_avg = user_genre_ctx[:, :n_genres]                              # (B, 20)
    genre_dot      = (user_genre_avg * cand_genre_oh).sum(dim=1)               # (B,)
    genre_cnt      = cand_genre_oh.sum(dim=1).clamp(min=1.0)
    genre_affinity = genre_dot / genre_cnt

    # Cross #2: Era Bias
    era_gap = torch.abs(user_mean_year_norm - cand_year_norm)

    # Cross #3: Rating Calibration
    rating_cal = user_avg - cand_global_avg

    # Cross #4: Popularity Match
    pop_match = torch.abs(user_count_log1p - cand_log_count)

    # Cross #6: Genre Intersection (Jaccard)
    # user_genre_ctx layout: [0:20]=avg_rating per genre, [20:40]=watch_fraction per genre.
    # User "set" of genres = those with any watches (watch_frac > 0).
    user_genre_mask = (user_genre_ctx[:, n_genres:2 * n_genres] > 0).float()   # (B, 20)
    inter           = (user_genre_mask * cand_genre_oh).sum(dim=1)             # (B,)
    union           = ((user_genre_mask + cand_genre_oh) > 0).float().sum(dim=1).clamp(min=1.0)
    jaccard         = inter / union                                            # (B,) in [0, 1]

    # Cross #7: Genome Peak Match — single dominant tag overlap.
    # max(user_genome_profile * cand_genome) finds the strongest co-tag, which a dot
    # product (sum) averages out. The "one tag spark" signal.
    genome_peak = (user_genome_profile * cand_genome_scores).max(dim=1).values  # (B,)

    # Cross #8: Dislike Similarity — cosine of disliked pool with raw item ID embedding.
    dis_sim = F.cosine_similarity(pool_disliked, item_id_raw, dim=1)           # (B,) in [-1, 1]

    # Cross #9: Recent CF Similarity — last item ID embedding · target item ID embedding.
    # Sequential next-item signal: "is this candidate close to what the user JUST watched?"
    # The 4 user pools are unordered sums — they have no temporal awareness. This restores it.
    recent_cf_sim = (last_item_emb * item_id_raw).sum(dim=1)                   # (B,)

    # Cross #10: Recent Genome Similarity — mean(last 5 genome) · candidate genome.
    # Content-space recency: smoothed "current taste vector" vs candidate. Distinct from
    # the all-history rating-weighted genome_cosine — this isolates the recent window.
    recent_genome_sim = (last_k_mean_genome * cand_genome_scores).sum(dim=1)   # (B,)

    # Cross #11: Genome Cosine Residual — candidate's genome cosine MINUS the user's
    # baseline (mean cosine of their profile vs their watched items). Calibrates the
    # raw cosine per user: positive = candidate is unusually compatible *for this user*,
    # negative = below the user's typical similarity bar.
    genome_residual = genome_cosine - user_genome_baseline                     # (B,)

    # Cross #12: Liked Popularity Gap — signed distance between candidate's log_count
    # and the mean log_count of items the user has LIKED (rating > 0). Captures the
    # user's preferred popularity tier: positive = candidate is more popular than the
    # user's typical pick, negative = more obscure. Distinct from pop_match (which uses
    # user's TOTAL rating count, i.e. user activity, not their popularity preference).
    liked_pop_gap = cand_log_count - user_mean_liked_log_count                 # (B,)

    return torch.stack(
        [genome_cosine, genre_affinity, era_gap, rating_cal, pop_match,
         jaccard, genome_peak, dis_sim,
         recent_cf_sim, recent_genome_sim, genome_residual, liked_pop_gap],
        dim=1,
    )


# ── Random-tuple batch sampler ───────────────────────────────────────────────

def sample_batch(dataset: RankerDataset, batch_size: int, device: torch.device,
                 rng: np.random.Generator, easy_neg_frac: float = 0.5):
    """
    Mixed Negative Sampling (MNS). Returns raw inputs needed for both the model forward
    AND for compute_cross_features (called by the caller after running model embeddings).

    Returns 15-tuple:
      user_genre_ctx (B, 40), X_history (B, H), X_hist_ratings (B, H), timestamp (B,),
      cand_idx (B,), label (B,),
      genome_cosine (B,), user_avg (B,), user_count_log1p (B,), user_mean_year_norm (B,),
      cand_genre_oh (B, 20), cand_year_norm (B,), cand_global_avg (B,), cand_log_count (B,),
      user_mean_liked_log_count (B,)
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

    user_avg_b      = dataset._user_avg_t[rows_t].to(device)
    user_cnt_b      = dataset._user_count_log1p_t[rows_t].to(device)
    rows_t_dev      = rows_t.to(device)
    user_year_b     = dataset.user_mean_year_norm[rows_t_dev]
    user_liked_pop  = dataset.user_mean_liked_log_count[rows_t_dev]

    return (user_genre_ctx_t, X_history_t, X_hist_rat_t, timestamp_t,
            cand_t, label_t,
            genome_cosine_t, user_avg_b, user_cnt_b, user_year_b,
            cand_genre_oh, cand_year_norm, cand_global_avg, cand_log_count,
            user_liked_pop)


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
