"""
Stage 3 — Dataset Loading
Reads features_*.parquet into a FeatureStore, builds PyTorch tensors.

Usage (from train.py or main.py):
    from src.dataset import load_features, make_softmax_splits
"""
import os
import random
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch


TIMESTAMP_NUM_BINS = 1_500
MAX_HISTORY_LEN    = 50   # cap watch history to most recent N movies


# ── FeatureStore ──────────────────────────────────────────────────────────────

@dataclass
class FeatureStore:
    # Vocabulary (ordered lists for index reproducibility)
    top_movies:           list   # list[int], ordered — index i = embedding row i
    genres_ordered:       list   # list[str]
    tags_ordered:         list   # list[str]
    genome_tag_ids:       list   # list[int]
    years_ordered:        list   # list[str]

    # Vocabulary index maps
    genre_to_i:           dict
    tag_to_i:             dict
    genome_tag_to_i:      dict   # tagId (int) → index
    genome_tag_names:     dict   # tagId (int) → name str
    year_to_i:            dict
    item_emb_movieId_to_i: dict  # movieId → embedding row index

    # Per-movie lookups
    movieId_to_title:         dict
    title_to_movieId:         dict
    movieId_to_year:          dict
    movieId_to_genres:        dict
    movieId_to_genre_context: dict
    movieId_to_tag_context:   dict
    movieId_to_genome_tag_context: dict

    # Derived constants
    user_context_size:                    int
    timestamp_num_bins:                   int
    timestamp_bins:                       torch.Tensor
    user_context_genre_avg_rating_to_i:   dict
    user_context_genre_watch_count_to_i:  dict

    # Per-corpus-item interaction counts (float32 array, index = corpus idx).
    # Populated by load_features() when movie_interaction_counts_v2.npy exists.
    # None if the file is missing (popularity_alpha=0 is safe in that case).
    movie_interaction_counts: object = field(default=None)  # np.ndarray | None


# ── Loader ────────────────────────────────────────────────────────────────────

def load_features(data_dir: str = 'data', version: str = 'v1') -> FeatureStore:
    """Load feature parquets and base vocab/movies into a FeatureStore."""
    vocab_df   = pd.read_parquet(os.path.join(data_dir, 'base_vocab.parquet'))
    movies_df  = pd.read_parquet(os.path.join(data_dir, 'base_movies.parquet'))
    ts_df      = pd.read_parquet(os.path.join(data_dir, 'base_timestamps.parquet'))

    movie_feat_path = os.path.join(data_dir, f'features_movies_{version}.parquet')

    # PyArrow reads list columns properly
    movie_feat_df = pq.read_table(movie_feat_path).to_pandas()

    # ── Vocab ─────────────────────────────────────────────────────────────────
    g  = vocab_df[vocab_df['type'] == 'genre'].sort_values('index')
    t  = vocab_df[vocab_df['type'] == 'tag'].sort_values('index')
    gt = vocab_df[vocab_df['type'] == 'genome_tag'].sort_values('index')
    y  = vocab_df[vocab_df['type'] == 'year'].sort_values('index')

    genres_ordered   = g['value'].tolist()
    tags_ordered     = t['value'].tolist()
    genome_tag_ids   = [int(v) for v in gt['value'].tolist()]
    years_ordered    = y['value'].tolist()

    genre_to_i       = dict(zip(g['value'], g['index'].astype(int)))
    tag_to_i         = dict(zip(t['value'], t['index'].astype(int)))
    genome_tag_to_i  = {int(r['value']): int(r['index']) for _, r in gt.iterrows()}
    genome_tag_names = {int(r['value']): r['extra']      for _, r in gt.iterrows()}
    year_to_i        = dict(zip(y['value'], y['index'].astype(int)))

    # ── Movie metadata ────────────────────────────────────────────────────────
    top_movies = movies_df['movieId'].tolist()
    item_emb_movieId_to_i = {int(mid): i for i, mid in enumerate(top_movies)}

    movieId_to_title  = {}
    title_to_movieId  = {}
    movieId_to_year   = {}
    movieId_to_genres = {}
    for _, row in movies_df.iterrows():
        mid = int(row['movieId'])
        movieId_to_title[mid]  = row['title']
        title_to_movieId[row['title']] = mid
        movieId_to_year[mid]   = str(row['year'])
        movieId_to_genres[mid] = list(row['genres'])

    # ── Per-movie feature vectors ─────────────────────────────────────────────
    movieId_to_genre_context      = {}
    movieId_to_tag_context        = {}
    movieId_to_genome_tag_context = {}
    for _, row in movie_feat_df.iterrows():
        mid = int(row['movieId'])
        movieId_to_genre_context[mid]      = list(row['genre_context'])
        movieId_to_tag_context[mid]        = list(row['tag_context'])
        movieId_to_genome_tag_context[mid] = list(row['genome_tag_context'])

    # ── Derived constants ─────────────────────────────────────────────────────
    n_genres          = len(genres_ordered)
    # user genre context = [avg_debiased_rating_per_genre (n_genres) | watch_frac_per_genre (n_genres)]
    user_context_size = 2 * n_genres

    user_context_genre_avg_rating_to_i   = {g: i           for i, g in enumerate(genres_ordered)}
    user_context_genre_watch_count_to_i  = {g: n_genres + i for i, g in enumerate(genres_ordered)}

    ts_min = int(ts_df['ts_min'].iloc[0])
    ts_max = int(ts_df['ts_max'].iloc[0])
    timestamp_bins = torch.tensor(
        np.linspace(ts_min, ts_max, TIMESTAMP_NUM_BINS)
    )

    counts_path = os.path.join(data_dir, 'movie_interaction_counts_v2.npy')
    movie_interaction_counts = np.load(counts_path) if os.path.exists(counts_path) else None

    return FeatureStore(
        top_movies=top_movies,
        genres_ordered=genres_ordered,
        tags_ordered=tags_ordered,
        genome_tag_ids=genome_tag_ids,
        years_ordered=years_ordered,
        genre_to_i=genre_to_i,
        tag_to_i=tag_to_i,
        genome_tag_to_i=genome_tag_to_i,
        genome_tag_names=genome_tag_names,
        year_to_i=year_to_i,
        item_emb_movieId_to_i=item_emb_movieId_to_i,
        movieId_to_title=movieId_to_title,
        title_to_movieId=title_to_movieId,
        movieId_to_year=movieId_to_year,
        movieId_to_genres=movieId_to_genres,
        movieId_to_genre_context=movieId_to_genre_context,
        movieId_to_tag_context=movieId_to_tag_context,
        movieId_to_genome_tag_context=movieId_to_genome_tag_context,
        user_context_size=user_context_size,
        timestamp_num_bins=TIMESTAMP_NUM_BINS,
        timestamp_bins=timestamp_bins,
        user_context_genre_avg_rating_to_i=user_context_genre_avg_rating_to_i,
        user_context_genre_watch_count_to_i=user_context_genre_watch_count_to_i,
        movie_interaction_counts=movie_interaction_counts,
    )


# ── Padding helpers ───────────────────────────────────────────────────────────

def pad_history_batch(histories: list, pad_idx: int) -> torch.Tensor:
    max_len = max((len(h) for h in histories), default=1)
    padded  = torch.full((len(histories), max_len), pad_idx, dtype=torch.long)
    for i, hist in enumerate(histories):
        if hist:
            padded[i, :len(hist)] = torch.tensor(hist, dtype=torch.long)
    return padded


def pad_history_ratings_batch(history_ratings: list) -> torch.Tensor:
    max_len = max((len(r) for r in history_ratings), default=1)
    padded  = torch.zeros(len(history_ratings), max_len)
    for i, rats in enumerate(history_ratings):
        if rats:
            padded[i, :len(rats)] = torch.tensor(rats, dtype=torch.float)
    return padded


# ── Rollback dataset (used by offline eval) ───────────────────────────────────
#
# For each val user, sample up to MAX_ROLLBACK_EXAMPLES_PER_USER chronological
# positions. At each position j: context = history[0..j-1], target = history[j].
# Val users are held out at the user level, so all positions are valid.

MAX_ROLLBACK_EXAMPLES_PER_USER = 20


def build_rollback_dataset(users: list, fs: FeatureStore, raw_df,
                            max_per_user: int = MAX_ROLLBACK_EXAMPLES_PER_USER,
                            seed: int = 42,
                            sort_by_ts: bool = True) -> tuple:
    """
    Build rollback examples for offline evaluation.

    raw_df must have columns: userId, movieId, rating, timestamp,
    already filtered to corpus movies and valid users.

    Returns 6-tuple:
        [0] X_genre            — (N, user_context_size) float  rollback genre context
        [1] X_history          — list[list[int]]  (padded at eval time)
        [2] X_history_ratings  — list[list[float]]
        [3] timestamp          — (N,) long  (binned)
        [4] Y                  — (N,) float  debiased target rating (unused in retrieval eval)
        [5] target_movieId     — (N,) long  (embedding index)
    """
    rng       = random.Random(seed)
    users_set = set(users)
    max_hist  = MAX_HISTORY_LEN
    n_genres  = len(fs.genres_ordered)

    movie_genre_idxs = {
        mid: [fs.genre_to_i[g] for g in fs.movieId_to_genres.get(mid, []) if g in fs.genre_to_i]
        for mid in fs.top_movies
    }

    df = raw_df[raw_df['userId'].isin(users_set)].copy()
    print(f"  {len(df):,} interactions, {df['userId'].nunique():,} users")

    X_genre               = []
    X_history             = []
    X_history_ratings     = []
    timestamps_raw        = []
    Y                     = []
    target_movieId        = []

    from tqdm import tqdm
    n_users = df['userId'].nunique()
    for uid, group in tqdm(df.groupby('userId'), total=n_users, desc="Building rollback examples"):
        rows    = list(zip(group['movieId'].tolist(), group['rating'].tolist(), group['timestamp'].tolist()))
        if sort_by_ts:
            rows.sort(key=lambda x: x[2])
        else:
            rng.shuffle(rows)
        movies, ratings, ts_vals = zip(*rows) if rows else ([], [], [])
        avg_rat = float(np.mean(ratings))
        n       = len(movies)

        if n < 2:
            continue

        eligible = list(range(1, n))
        k               = min(max_per_user, len(eligible))
        sampled_targets = sorted(rng.sample(eligible, k))
        sampled_set     = set(sampled_targets)

        running_count = np.zeros(n_genres, dtype=np.float32)
        running_sum   = np.zeros(n_genres, dtype=np.float32)
        ctx_ids_buf   = []
        ctx_rats_buf  = []

        for pos, (mid, rat, ts) in enumerate(zip(movies, ratings, ts_vals)):
            mid   = int(mid)
            d_rat = float(rat) - avg_rat
            t_idx = fs.item_emb_movieId_to_i[mid]

            if pos in sampled_set:
                total_assign = running_count.sum()
                genre_ctx    = np.zeros(2 * n_genres, dtype=np.float32)
                if total_assign > 0:
                    mask = running_count > 0
                    genre_ctx[:n_genres][mask] = running_sum[mask] / running_count[mask]
                    genre_ctx[n_genres:]       = running_count / total_assign

                X_genre.append(genre_ctx.tolist())
                X_history.append(list(ctx_ids_buf[-max_hist:]))
                X_history_ratings.append(list(ctx_rats_buf[-max_hist:]))
                timestamps_raw.append(ts)
                Y.append(d_rat)
                target_movieId.append(t_idx)

            ctx_ids_buf.append(t_idx)
            ctx_rats_buf.append(d_rat)
            for g_idx in movie_genre_idxs.get(mid, []):
                running_count[g_idx] += 1
                running_sum[g_idx]   += d_rat

    n = len(target_movieId)
    print(f"  {n:,} rollback examples — building tensors ...")

    X_genre_t         = torch.from_numpy(np.array(X_genre,        dtype=np.float32))
    Y_t               = torch.from_numpy(np.array(Y,              dtype=np.float32))
    target_movieId_t  = torch.from_numpy(np.array(target_movieId, dtype=np.int64))
    timestamp_t       = torch.bucketize(
        torch.from_numpy(np.array(timestamps_raw, dtype=np.float64)).float(),
        fs.timestamp_bins.float(), right=False)

    return (X_genre_t, X_history, X_history_ratings, timestamp_t, Y_t, target_movieId_t)


def get_val_users(fs: FeatureStore, data_dir: str = 'data',
                  pct_train: float = 0.9, seed: int = 42) -> tuple:
    """Return (val_users, corpus-filtered raw_df) — same split as make_softmax_splits."""
    ratings_path = os.path.join(data_dir, 'base_ratings.parquet')
    raw_df = pd.read_parquet(ratings_path)
    corpus_set = set(fs.item_emb_movieId_to_i.keys())
    raw_df = raw_df[raw_df['movieId'].isin(corpus_set)]

    valid_users = sorted(raw_df['userId'].astype(int).unique().tolist())
    rng = random.Random(seed)
    rng.shuffle(valid_users)

    split = int(len(valid_users) * pct_train)
    return valid_users[split:], raw_df


# ── Softmax dataset (full softmax training) ───────────────────────────────────
#
# Same rollback structure as eval: per user, sort by timestamp, sample up to
# MAX_SOFTMAX_EXAMPLES_PER_USER rollback positions.
#
# Returns 7-tuple:
#   [0] X_genre           — (N, user_context_size) float32
#   [1] X_history         — (N, MAX_HISTORY_LEN) int64, right-aligned, pad=len(top_movies)
#   [2] X_hist_liked      — (N, MAX_HISTORY_LEN) int64  items with positive debiased rating
#   [3] X_hist_disliked   — (N, MAX_HISTORY_LEN) int64  items with negative debiased rating
#   [4] X_hist_ratings    — (N, MAX_HISTORY_LEN) float32, right-aligned, pad=0.0
#   [5] timestamp         — (N,) int64
#   [6] target_movieId    — (N,) int64  (corpus embedding index)

MAX_SOFTMAX_EXAMPLES_PER_USER = 20


def build_softmax_dataset(users: list, fs: FeatureStore, raw_df,
                           max_per_user: int = MAX_SOFTMAX_EXAMPLES_PER_USER,
                           seed: int = 42) -> tuple:
    from tqdm import tqdm

    rng       = random.Random(seed)
    users_set = set(users)
    max_hist  = MAX_HISTORY_LEN
    n_genres  = len(fs.genres_ordered)
    pad_idx   = len(fs.top_movies)

    movie_genre_idxs = {
        mid: [fs.genre_to_i[g] for g in fs.movieId_to_genres.get(mid, []) if g in fs.genre_to_i]
        for mid in fs.top_movies
    }

    df = raw_df[raw_df['userId'].isin(users_set)].copy()
    print(f"  {len(df):,} interactions, {df['userId'].nunique():,} users")

    # ── 1. Counting pass ──────────────────────────────────────────────────────
    print("  Counting pass ...")
    n_examples   = 0
    valid_groups = []
    for uid, group in df.groupby('userId'):
        rows = list(zip(group['movieId'].tolist(), group['rating'].tolist(),
                        group['timestamp'].tolist()))
        rows.sort(key=lambda x: x[2])
        n = len(rows)
        if n >= 2:
            n_examples += min(max_per_user, n - 1)
            valid_groups.append((uid, rows))

    # ── 2. Pre-allocate NumPy buffers ─────────────────────────────────────────
    print(f"  Pre-allocating buffers for {n_examples:,} examples ...")
    X_genre         = np.zeros((n_examples, 2 * n_genres), dtype=np.float32)
    X_history       = np.full((n_examples, max_hist), pad_idx, dtype=np.int32)
    X_hist_liked    = np.full((n_examples, max_hist), pad_idx, dtype=np.int32)
    X_hist_disliked = np.full((n_examples, max_hist), pad_idx, dtype=np.int32)
    X_hist_ratings  = np.zeros((n_examples, max_hist), dtype=np.float32)
    timestamps_raw  = np.zeros(n_examples, dtype=np.float64)
    target_arr      = np.zeros(n_examples, dtype=np.int32)

    # ── 3. Fill pass ──────────────────────────────────────────────────────────
    curr = 0
    for uid, rows in tqdm(valid_groups, desc="Building softmax examples"):
        movies, ratings, ts_vals = zip(*rows)
        avg_rat = float(np.mean(ratings))
        n       = len(movies)

        eligible    = list(range(1, n))
        k           = min(max_per_user, len(eligible))
        sampled     = sorted(rng.sample(eligible, k))
        sampled_set = set(sampled)

        running_count = np.zeros(n_genres, dtype=np.float32)
        running_sum   = np.zeros(n_genres, dtype=np.float32)
        ctx_ids       = []
        ctx_rats      = []

        for pos, (mid, rat, ts) in enumerate(zip(movies, ratings, ts_vals)):
            mid   = int(mid)
            d_rat = float(rat) - avg_rat
            t_idx = fs.item_emb_movieId_to_i[mid]

            if pos in sampled_set:
                total = running_count.sum()
                if total > 0:
                    mask = running_count > 0
                    X_genre[curr, :n_genres][mask] = running_sum[mask] / running_count[mask]
                    X_genre[curr, n_genres:]       = running_count / total

                # Right-align: padding at left, most recent items at right
                take = min(len(ctx_ids), max_hist)
                if take > 0:
                    X_history[curr, max_hist - take:]      = ctx_ids[-take:]
                    X_hist_ratings[curr, max_hist - take:] = ctx_rats[-take:]

                liked_ids    = [idx for idx, r in zip(ctx_ids, ctx_rats) if r > 0]
                disliked_ids = [idx for idx, r in zip(ctx_ids, ctx_rats) if r < 0]
                take_l = min(len(liked_ids), max_hist)
                if take_l > 0:
                    X_hist_liked[curr, max_hist - take_l:] = liked_ids[-take_l:]
                take_d = min(len(disliked_ids), max_hist)
                if take_d > 0:
                    X_hist_disliked[curr, max_hist - take_d:] = disliked_ids[-take_d:]

                timestamps_raw[curr] = ts
                target_arr[curr]     = t_idx
                curr += 1

            ctx_ids.append(t_idx)
            ctx_rats.append(d_rat)
            for g_idx in movie_genre_idxs.get(mid, []):
                running_count[g_idx] += 1
                running_sum[g_idx]   += d_rat

    print(f"  {n_examples:,} softmax examples — converting to tensors ...")

    timestamp_t = torch.bucketize(
        torch.from_numpy(timestamps_raw).float(),
        fs.timestamp_bins.float(), right=False,
    ).clamp(max=TIMESTAMP_NUM_BINS - 1)

    return (
        torch.from_numpy(X_genre),
        torch.from_numpy(X_history).long(),
        torch.from_numpy(X_hist_liked).long(),
        torch.from_numpy(X_hist_disliked).long(),
        torch.from_numpy(X_hist_ratings),
        timestamp_t,
        torch.from_numpy(target_arr).long(),
    )


def make_softmax_splits(fs: FeatureStore, data_dir: str = 'data',
                         max_per_user: int = MAX_SOFTMAX_EXAMPLES_PER_USER,
                         pct_train: float = 0.9, seed: int = 42,
                         max_users: int = None) -> tuple:
    ratings_path = os.path.join(data_dir, 'base_ratings.parquet')
    print(f"Loading {ratings_path} ...")
    raw_df = pd.read_parquet(ratings_path)
    corpus_set = set(fs.item_emb_movieId_to_i.keys())
    raw_df = raw_df[raw_df['movieId'].isin(corpus_set)]
    print(f"  {len(raw_df):,} raw interactions, {raw_df['userId'].nunique():,} users")

    valid_users = sorted(raw_df['userId'].astype(int).unique().tolist())
    rng = random.Random(seed)
    rng.shuffle(valid_users)

    if max_users is not None:
        valid_users = valid_users[:max_users]
        print(f"  [debug] subsampled to {len(valid_users):,} users")

    split       = int(len(valid_users) * pct_train)
    train_users = valid_users[:split]
    val_users   = valid_users[split:]

    print(f"\nBuilding softmax train dataset ({len(train_users):,} users) ...")
    train_data = build_softmax_dataset(train_users, fs, raw_df, max_per_user, seed)
    print(f"  X_genre_train shape: {train_data[0].shape}")

    print(f"\nBuilding softmax val dataset ({len(val_users):,} users) ...")
    val_data = build_softmax_dataset(val_users, fs, raw_df, max_per_user, seed)
    print(f"  X_genre_val shape:   {val_data[0].shape}")

    return train_data, val_data


def save_softmax_splits(train_data: tuple, val_data: tuple,
                         data_dir: str = 'data', version: str = 'v2') -> None:
    torch.save(train_data, os.path.join(data_dir, f'dataset_softmax_train_{version}.pt'))
    torch.save(val_data,   os.path.join(data_dir, f'dataset_softmax_val_{version}.pt'))
    target_ids = train_data[6].numpy()
    counts = np.bincount(target_ids).astype(np.float32)
    np.save(os.path.join(data_dir, 'movie_interaction_counts_v2.npy'), counts)
    print(f"Saved dataset_softmax_train_{version}.pt, dataset_softmax_val_{version}.pt, "
          f"movie_interaction_counts_v2.npy → {data_dir}/")


def load_softmax_splits(data_dir: str = 'data', version: str = 'v2') -> tuple:
    train_path = os.path.join(data_dir, f'dataset_softmax_train_{version}.pt')
    val_path   = os.path.join(data_dir, f'dataset_softmax_val_{version}.pt')
    print(f"Loading {train_path} ...")
    train_data = torch.load(train_path, weights_only=False)
    print(f"Loading {val_path} ...")
    val_data   = torch.load(val_path, weights_only=False)
    return train_data, val_data
