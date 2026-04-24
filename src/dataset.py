"""
Stage 3 — Dataset Loading
Reads features_*.parquet into a FeatureStore, builds PyTorch tensors.
No files are written here — pure in-memory.

Usage (from train.py or main.py):
    from src.dataset import load_features, make_splits
"""
import os
import random
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch


TIMESTAMP_NUM_BINS = 1_500


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

    # Per-user lookups
    user_ids:                          list
    user_to_avg_rating:                dict
    user_to_context:                   dict
    user_to_watch_history:             dict
    user_to_watch_history_ratings:     dict
    user_to_movie_to_rating_LABEL:     dict
    user_to_movie_to_timestamp_LABEL:  dict

    # Derived constants
    user_context_size:                    int
    timestamp_num_bins:                   int
    timestamp_bins:                       torch.Tensor
    user_context_genre_avg_rating_to_i:   dict
    user_context_genre_watch_count_to_i:  dict


# ── Loader ────────────────────────────────────────────────────────────────────

def load_features(data_dir: str = 'data', version: str = 'v1') -> FeatureStore:
    """Load feature parquets and base vocab/movies into a FeatureStore."""
    vocab_df   = pd.read_parquet(os.path.join(data_dir, 'base_vocab.parquet'))
    movies_df  = pd.read_parquet(os.path.join(data_dir, 'base_movies.parquet'))
    ts_df      = pd.read_parquet(os.path.join(data_dir, 'base_timestamps.parquet'))

    movie_feat_path = os.path.join(data_dir, f'features_movies_{version}.parquet')
    user_feat_path  = os.path.join(data_dir, f'features_users_{version}.parquet')

    # PyArrow reads list columns properly
    movie_feat_df = pq.read_table(movie_feat_path).to_pandas()
    user_feat_df  = pq.read_table(user_feat_path).to_pandas()

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

    # ── Per-user feature vectors ──────────────────────────────────────────────
    user_ids                         = []
    user_to_avg_rating               = {}
    user_to_context                  = {}
    user_to_watch_history            = {}
    user_to_watch_history_ratings    = {}
    user_to_movie_to_rating_LABEL    = {}
    user_to_movie_to_timestamp_LABEL = {}

    for _, row in user_feat_df.iterrows():
        uid = int(row['userId'])
        user_ids.append(uid)
        user_to_avg_rating[uid]            = float(row['avg_rating'])
        user_to_context[uid]               = list(row['genre_context'])
        user_to_watch_history[uid]         = list(row['watch_history'])
        user_to_watch_history_ratings[uid] = list(row['watch_history_ratings'])

        lbl_movies = list(row['label_movieIds'])
        lbl_rats   = list(row['label_ratings'])
        lbl_times  = list(row['label_timestamps'])
        user_to_movie_to_rating_LABEL[uid]    = dict(zip(lbl_movies, lbl_rats))
        user_to_movie_to_timestamp_LABEL[uid] = dict(zip(lbl_movies, lbl_times))

    # ── Derived constants ─────────────────────────────────────────────────────
    n_genres          = len(genres_ordered)
    # user genre context = [avg_debiased_rating_per_genre (n_genres) | watch_frac_per_genre (n_genres)]
    user_context_size = 2 * n_genres

    user_context_genre_avg_rating_to_i   = {g: i           for i, g in enumerate(genres_ordered)}  # indices 0..n_genres-1
    user_context_genre_watch_count_to_i  = {g: n_genres + i for i, g in enumerate(genres_ordered)} # indices n_genres..2*n_genres-1

    ts_min = int(ts_df['ts_min'].iloc[0])
    ts_max = int(ts_df['ts_max'].iloc[0])
    timestamp_bins = torch.tensor(
        np.linspace(ts_min, ts_max, TIMESTAMP_NUM_BINS)
    )

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
        user_ids=user_ids,
        user_to_avg_rating=user_to_avg_rating,
        user_to_context=user_to_context,
        user_to_watch_history=user_to_watch_history,
        user_to_watch_history_ratings=user_to_watch_history_ratings,
        user_to_movie_to_rating_LABEL=user_to_movie_to_rating_LABEL,
        user_to_movie_to_timestamp_LABEL=user_to_movie_to_timestamp_LABEL,
        user_context_size=user_context_size,
        timestamp_num_bins=TIMESTAMP_NUM_BINS,
        timestamp_bins=timestamp_bins,
        user_context_genre_avg_rating_to_i=user_context_genre_avg_rating_to_i,
        user_context_genre_watch_count_to_i=user_context_genre_watch_count_to_i,
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


# ── Dataset builder ───────────────────────────────────────────────────────────

def build_dataset(users: list, fs: FeatureStore) -> tuple:
    """
    Build training/validation tensors for a list of user IDs.
    Returns a tuple of 10 elements:
      X, X_history (list), X_history_ratings (list),
      timestamp, Y, target_movieId,
      target_movieId_genre_context, target_movieId_tag_context,
      target_movieId_genome_tag_context, target_movieId_year
    """
    X                              = []
    X_history                      = []
    X_history_ratings              = []
    timestamp                      = []
    target_movieId                 = []
    target_movieId_genre_context   = []
    target_movieId_tag_context     = []
    target_movieId_genome_context  = []
    target_movieId_year            = []
    Y                              = []

    from tqdm import tqdm
    for user in tqdm(users, desc="Collecting samples"):
        for movieId, rating in fs.user_to_movie_to_rating_LABEL[user].items():
            movieId = int(movieId)
            X.append(fs.user_to_context[user])
            X_history.append(fs.user_to_watch_history[user])
            X_history_ratings.append(fs.user_to_watch_history_ratings[user])
            timestamp.append(fs.user_to_movie_to_timestamp_LABEL[user][movieId])
            target_movieId.append(fs.item_emb_movieId_to_i[movieId])
            target_movieId_genre_context.append(fs.movieId_to_genre_context[movieId])
            target_movieId_tag_context.append(fs.movieId_to_tag_context[movieId])
            target_movieId_genome_context.append(fs.movieId_to_genome_tag_context[movieId])
            target_movieId_year.append(fs.year_to_i[fs.movieId_to_year[movieId]])
            Y.append(float(rating - fs.user_to_avg_rating[user]))

    n = len(Y)
    print(f"  {n:,} samples — building tensors ...")
    print("  X, Y ...")
    X               = torch.from_numpy(np.array(X,     dtype=np.float32))
    Y               = torch.from_numpy(np.array(Y,     dtype=np.float32))
    print("  target_movieId, year, timestamp ...")
    target_movieId_t = torch.from_numpy(np.array(target_movieId,      dtype=np.int64))
    target_movieId_year_t = torch.from_numpy(np.array(target_movieId_year, dtype=np.int64))
    timestamp_t = torch.bucketize(
        torch.from_numpy(np.array(timestamp, dtype=np.float32)),
        fs.timestamp_bins, right=False)
    print("  genre context ...")
    target_movieId_genre_t = torch.from_numpy(np.array(target_movieId_genre_context, dtype=np.float32))
    print("  tag context ...")
    target_movieId_tag_t   = torch.from_numpy(np.array(target_movieId_tag_context,   dtype=np.float32))
    print("  genome tag context ...")
    target_movieId_genome_t = torch.from_numpy(np.array(target_movieId_genome_context, dtype=np.float32))

    return (X, X_history, X_history_ratings, timestamp_t, Y,
            target_movieId_t, target_movieId_genre_t, target_movieId_tag_t,
            target_movieId_genome_t, target_movieId_year_t)


# ── Disk cache helpers ────────────────────────────────────────────────────────

def save_splits(train_data: tuple, val_data: tuple,
                data_dir: str = 'data', version: str = 'v1') -> None:
    torch.save(train_data, os.path.join(data_dir, f'dataset_train_{version}.pt'))
    torch.save(val_data,   os.path.join(data_dir, f'dataset_val_{version}.pt'))
    print(f"Saved dataset_train_{version}.pt and dataset_val_{version}.pt → {data_dir}/")


def load_splits(data_dir: str = 'data', version: str = 'v1') -> tuple:
    train_path = os.path.join(data_dir, f'dataset_train_{version}.pt')
    val_path   = os.path.join(data_dir, f'dataset_val_{version}.pt')
    print(f"Loading {train_path} ...")
    train_data = torch.load(train_path, weights_only=False)
    print(f"Loading {val_path} ...")
    val_data   = torch.load(val_path, weights_only=False)
    return train_data, val_data


# ── Softmax dataset (rollback, implicit feedback) ─────────────────────────────
#
# Each user contributes multiple training examples via "rollback":
# for a user who watched movies [A, B, C, D, E] in order, we generate examples like:
#   context=[A],       target=B
#   context=[A,B,C],   target=D
#   context=[A,B,C,D], target=E
# Each example uses only movies watched *before* the target — no future leakage.
# MAX_SOFTMAX_EXAMPLES_PER_USER caps how many rollback examples we randomly sample per user.
#
# Why rollback for BOTH train and val:
# Rollback produces examples with varying context lengths (short to long).
# If val used only the last rating per user (full history as context), val would always
# have long contexts while train has short+long — a distribution mismatch that makes
# val loss unreliable. Using rollback for both keeps context length distribution consistent.

MAX_SOFTMAX_EXAMPLES_PER_USER = 20


def build_softmax_dataset(users: list, fs: FeatureStore, raw_df,
                           max_per_user: int = MAX_SOFTMAX_EXAMPLES_PER_USER,
                           seed: int = 42,
                           min_target_rating: float = 0.0) -> tuple:
    """
    Build rollback training examples for in-batch negatives softmax training.

    raw_df must have columns: userId, movieId, rating, timestamp.
    All rows are assumed to be already filtered to corpus movies and valid users.

    min_target_rating: if > 0, only use watch events with raw rating >= this value as
        targets. Context (history) remains unfiltered — low-rated watches still inform
        the user embedding. Set to 4.0 to restrict targets to movies the user enjoyed.

    Returns 9-tuple:
        [0] X_genre            — (N, user_context_size) float  rollback genre context
        [1] X_history          — list[list[int]]  (padded at training time)
        [2] X_history_ratings  — list[list[float]]
        [3] timestamp          — (N,) long  (binned)
        [4] target_movieId     — (N,) long  (embedding index)
        [5] target_genre       — (N, genres_len) float
        [6] target_tag         — (N, tags_len) float
        [7] target_genome      — (N, genome_tags_len) float
        [8] target_year        — (N,) long
    """
    from src.features import MAX_HISTORY_LEN
    rng       = random.Random(seed)
    users_set = set(users)
    max_hist  = MAX_HISTORY_LEN
    n_genres  = len(fs.genres_ordered)

    # Precompute movie → genre index list (avoids repeated dict lookups in inner loop)
    movie_genre_idxs = {
        mid: [fs.genre_to_i[g] for g in fs.movieId_to_genres.get(mid, []) if g in fs.genre_to_i]
        for mid in fs.top_movies
    }

    df = raw_df[raw_df['userId'].isin(users_set)].copy()
    print(f"  Sorting {len(df):,} interactions by user + timestamp ...")
    df = df.sort_values(['userId', 'timestamp'])
    print(f"  {df['userId'].nunique():,} users")

    X_genre               = []
    X_history             = []
    X_history_ratings     = []
    timestamps_raw        = []
    target_movieId        = []
    target_genre_context  = []
    target_tag_context    = []
    target_genome_context = []
    target_year           = []

    from tqdm import tqdm
    n_users = df['userId'].nunique()
    for uid, group in tqdm(df.groupby('userId'), total=n_users, desc="Building softmax examples"):
        avg_rat = fs.user_to_avg_rating.get(uid, 3.0)

        movies  = group['movieId'].tolist()
        ratings = group['rating'].tolist()
        ts_vals = group['timestamp'].tolist()
        n       = len(movies)

        if n < 2:
            continue

        # Sample target positions upfront — avoids generating all rollbacks then discarding.
        # Valid targets: positions 1..n-1 (position 0 has no prior context).
        # If min_target_rating set, only positions with raw rating >= threshold are eligible.
        # Sorting ensures a single left-to-right scan maintains genre accumulators correctly.
        eligible = [i for i in range(1, n)
                    if min_target_rating == 0.0 or ratings[i] >= min_target_rating]
        if not eligible:
            continue
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
            t_idx = fs.item_emb_movieId_to_i[mid]   # safe: raw_df filtered to corpus movies

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
                target_movieId.append(t_idx)
                target_genre_context.append(fs.movieId_to_genre_context[mid])
                target_tag_context.append(fs.movieId_to_tag_context[mid])
                target_genome_context.append(fs.movieId_to_genome_tag_context[mid])
                target_year.append(fs.year_to_i.get(fs.movieId_to_year[mid], 0))

            # Update accumulators and context buffer with current movie
            ctx_ids_buf.append(t_idx)
            ctx_rats_buf.append(d_rat)
            for g_idx in movie_genre_idxs.get(mid, []):
                running_count[g_idx] += 1
                running_sum[g_idx]   += d_rat

    n = len(target_movieId)
    print(f"  {n:,} softmax examples — building tensors ...")

    X_genre_t         = torch.from_numpy(np.array(X_genre,               dtype=np.float32))
    target_movieId_t  = torch.from_numpy(np.array(target_movieId,        dtype=np.int64))
    target_year_t     = torch.from_numpy(np.array(target_year,           dtype=np.int64))
    target_genre_t    = torch.from_numpy(np.array(target_genre_context,  dtype=np.float32))
    target_tag_t      = torch.from_numpy(np.array(target_tag_context,    dtype=np.float32))
    target_genome_t   = torch.from_numpy(np.array(target_genome_context, dtype=np.float32))
    timestamp_t       = torch.bucketize(
        torch.from_numpy(np.array(timestamps_raw, dtype=np.float64)).float(),
        fs.timestamp_bins.float(), right=False)

    return (X_genre_t, X_history, X_history_ratings, timestamp_t,
            target_movieId_t, target_genre_t, target_tag_t, target_genome_t, target_year_t)


def make_softmax_splits(fs: FeatureStore, data_dir: str = 'data',
                        max_per_user: int = MAX_SOFTMAX_EXAMPLES_PER_USER,
                        pct_train: float = 0.9, seed: int = 42,
                        max_users: int = None,
                        min_target_rating: float = 0.0) -> tuple:
    """
    Load raw interactions (watch + labels), split users 90/10, build softmax datasets.
    Returns (train_data, val_data) each a 9-tuple from build_softmax_dataset().

    max_users: if set, subsample to this many total users (for fast debug runs).
    min_target_rating: if > 0, only use highly-rated watches as targets (e.g. 4.0).
    """
    import pandas as pd
    watch_path  = os.path.join(data_dir, 'base_ratings_watch.parquet')
    labels_path = os.path.join(data_dir, 'base_ratings_labels.parquet')
    print(f"Loading {watch_path} + {labels_path} ...")
    raw_df = pd.concat([
        pd.read_parquet(watch_path),
        pd.read_parquet(labels_path),
    ], ignore_index=True)
    print(f"  {len(raw_df):,} raw interactions, {raw_df['userId'].nunique():,} users")

    valid_users = fs.user_ids[:]
    rng = random.Random(seed)
    rng.shuffle(valid_users)

    if max_users is not None:
        valid_users = valid_users[:max_users]
        print(f"  [debug] subsampled to {len(valid_users):,} users")

    split       = int(len(valid_users) * pct_train)
    train_users = valid_users[:split]
    val_users   = valid_users[split:]

    if min_target_rating > 0:
        print(f"  min_target_rating={min_target_rating} — targets filtered to highly-rated watches only")

    print(f"\nBuilding softmax train dataset ({len(train_users):,} users) ...")
    train_data = build_softmax_dataset(train_users, fs, raw_df, max_per_user, seed, min_target_rating)
    print(f"  X_genre_train shape: {train_data[0].shape}")

    print(f"\nBuilding softmax val dataset ({len(val_users):,} users) ...")
    val_data = build_softmax_dataset(val_users, fs, raw_df, max_per_user, seed, min_target_rating)
    print(f"  X_genre_val shape:   {val_data[0].shape}")

    return train_data, val_data


def save_softmax_splits(train_data: tuple, val_data: tuple,
                        data_dir: str = 'data', version: str = 'v1') -> None:
    torch.save(train_data, os.path.join(data_dir, f'dataset_softmax_train_{version}.pt'))
    torch.save(val_data,   os.path.join(data_dir, f'dataset_softmax_val_{version}.pt'))
    print(f"Saved dataset_softmax_train_{version}.pt and dataset_softmax_val_{version}.pt → {data_dir}/")


def load_softmax_splits(data_dir: str = 'data', version: str = 'v1') -> tuple:
    train_path = os.path.join(data_dir, f'dataset_softmax_train_{version}.pt')
    val_path   = os.path.join(data_dir, f'dataset_softmax_val_{version}.pt')
    print(f"Loading {train_path} ...")
    train_data = torch.load(train_path, weights_only=False)
    print(f"Loading {val_path} ...")
    val_data   = torch.load(val_path, weights_only=False)
    return train_data, val_data


# ── MSE rollback dataset ──────────────────────────────────────────────────────
#
# Same rollback logic as the softmax dataset, but produces the MSE 10-tuple
# (includes Y = debiased target rating) so the existing train() loop works
# unchanged.  No 90/10 within-user history/label split — every chronological
# position in each user's full history is a valid (context, target) pair.
# Train vs val is a user-level split only (same 90/10 user split as softmax).

MAX_MSE_ROLLBACK_EXAMPLES_PER_USER = 20


def build_mse_rollback_dataset(users: list, fs: FeatureStore, raw_df,
                                max_per_user: int = MAX_MSE_ROLLBACK_EXAMPLES_PER_USER,
                                seed: int = 42) -> tuple:
    """
    Build rollback training examples for MSE training.

    Mirrors build_softmax_dataset() but adds Y (debiased rating) and returns
    the same 10-tuple as build_dataset() so train() works unchanged.

    raw_df must have columns: userId, movieId, rating, timestamp,
    already filtered to corpus movies and valid users.

    Returns 10-tuple:
        [0] X_genre            — (N, user_context_size) float  rollback genre context
        [1] X_history          — list[list[int]]  (padded at training time)
        [2] X_history_ratings  — list[list[float]]
        [3] timestamp          — (N,) long  (binned)
        [4] Y                  — (N,) float  debiased target rating
        [5] target_movieId     — (N,) long  (embedding index)
        [6] target_genre       — (N, genres_len) float
        [7] target_tag         — (N, tags_len) float
        [8] target_genome      — (N, genome_tags_len) float
        [9] target_year        — (N,) long
    """
    from src.features import MAX_HISTORY_LEN
    rng       = random.Random(seed)
    users_set = set(users)
    max_hist  = MAX_HISTORY_LEN
    n_genres  = len(fs.genres_ordered)

    movie_genre_idxs = {
        mid: [fs.genre_to_i[g] for g in fs.movieId_to_genres.get(mid, []) if g in fs.genre_to_i]
        for mid in fs.top_movies
    }

    df = raw_df[raw_df['userId'].isin(users_set)].copy()
    print(f"  Sorting {len(df):,} interactions by user + timestamp ...")
    df = df.sort_values(['userId', 'timestamp'])
    print(f"  {df['userId'].nunique():,} users")

    X_genre               = []
    X_history             = []
    X_history_ratings     = []
    timestamps_raw        = []
    Y                     = []
    target_movieId        = []
    target_genre_context  = []
    target_tag_context    = []
    target_genome_context = []
    target_year           = []

    from tqdm import tqdm
    n_users = df['userId'].nunique()
    for uid, group in tqdm(df.groupby('userId'), total=n_users, desc="Building MSE rollback examples"):
        avg_rat = fs.user_to_avg_rating.get(uid, 3.0)

        movies  = group['movieId'].tolist()
        ratings = group['rating'].tolist()
        ts_vals = group['timestamp'].tolist()
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
                target_genre_context.append(fs.movieId_to_genre_context[mid])
                target_tag_context.append(fs.movieId_to_tag_context[mid])
                target_genome_context.append(fs.movieId_to_genome_tag_context[mid])
                target_year.append(fs.year_to_i.get(fs.movieId_to_year[mid], 0))

            ctx_ids_buf.append(t_idx)
            ctx_rats_buf.append(d_rat)
            for g_idx in movie_genre_idxs.get(mid, []):
                running_count[g_idx] += 1
                running_sum[g_idx]   += d_rat

    n = len(target_movieId)
    print(f"  {n:,} MSE rollback examples — building tensors ...")

    X_genre_t         = torch.from_numpy(np.array(X_genre,               dtype=np.float32))
    Y_t               = torch.from_numpy(np.array(Y,                     dtype=np.float32))
    target_movieId_t  = torch.from_numpy(np.array(target_movieId,        dtype=np.int64))
    target_year_t     = torch.from_numpy(np.array(target_year,           dtype=np.int64))
    target_genre_t    = torch.from_numpy(np.array(target_genre_context,  dtype=np.float32))
    target_tag_t      = torch.from_numpy(np.array(target_tag_context,    dtype=np.float32))
    target_genome_t   = torch.from_numpy(np.array(target_genome_context, dtype=np.float32))
    timestamp_t       = torch.bucketize(
        torch.from_numpy(np.array(timestamps_raw, dtype=np.float64)).float(),
        fs.timestamp_bins.float(), right=False)

    return (X_genre_t, X_history, X_history_ratings, timestamp_t, Y_t,
            target_movieId_t, target_genre_t, target_tag_t, target_genome_t, target_year_t)


def make_mse_rollback_splits(fs: FeatureStore, data_dir: str = 'data',
                              max_per_user: int = MAX_MSE_ROLLBACK_EXAMPLES_PER_USER,
                              pct_train: float = 0.9, seed: int = 42,
                              max_users: int = None) -> tuple:
    """
    Load raw interactions (watch + labels), split users 90/10 at the user level,
    build MSE rollback datasets. No within-user history/label split.
    Returns (train_data, val_data) each a 10-tuple from build_mse_rollback_dataset().
    """
    watch_path  = os.path.join(data_dir, 'base_ratings_watch.parquet')
    labels_path = os.path.join(data_dir, 'base_ratings_labels.parquet')
    print(f"Loading {watch_path} + {labels_path} ...")
    raw_df = pd.concat([
        pd.read_parquet(watch_path),
        pd.read_parquet(labels_path),
    ], ignore_index=True)
    # Filter to corpus movies only
    corpus_set = set(fs.item_emb_movieId_to_i.keys())
    raw_df = raw_df[raw_df['movieId'].isin(corpus_set)]
    print(f"  {len(raw_df):,} raw interactions, {raw_df['userId'].nunique():,} users")

    valid_users = fs.user_ids[:]
    rng = random.Random(seed)
    rng.shuffle(valid_users)

    if max_users is not None:
        valid_users = valid_users[:max_users]
        print(f"  [debug] subsampled to {len(valid_users):,} users")

    split       = int(len(valid_users) * pct_train)
    train_users = valid_users[:split]
    val_users   = valid_users[split:]

    print(f"\nBuilding MSE rollback train dataset ({len(train_users):,} users) ...")
    train_data = build_mse_rollback_dataset(train_users, fs, raw_df, max_per_user, seed)
    print(f"  X_genre_train shape: {train_data[0].shape}")

    print(f"\nBuilding MSE rollback val dataset ({len(val_users):,} users) ...")
    val_data = build_mse_rollback_dataset(val_users, fs, raw_df, max_per_user, seed)
    print(f"  X_genre_val shape:   {val_data[0].shape}")

    return train_data, val_data


def save_mse_rollback_splits(train_data: tuple, val_data: tuple,
                              data_dir: str = 'data', version: str = 'v1') -> None:
    torch.save(train_data, os.path.join(data_dir, f'dataset_mse_rollback_train_{version}.pt'))
    torch.save(val_data,   os.path.join(data_dir, f'dataset_mse_rollback_val_{version}.pt'))
    print(f"Saved dataset_mse_rollback_train_{version}.pt and dataset_mse_rollback_val_{version}.pt → {data_dir}/")


def load_mse_rollback_splits(data_dir: str = 'data', version: str = 'v1') -> tuple:
    train_path = os.path.join(data_dir, f'dataset_mse_rollback_train_{version}.pt')
    val_path   = os.path.join(data_dir, f'dataset_mse_rollback_val_{version}.pt')
    print(f"Loading {train_path} ...")
    train_data = torch.load(train_path, weights_only=False)
    print(f"Loading {val_path} ...")
    val_data   = torch.load(val_path, weights_only=False)
    return train_data, val_data


# ── Train / val split ─────────────────────────────────────────────────────────

def make_splits(fs: FeatureStore, pct_train: float = 0.9, seed: int = 42) -> tuple:
    """
    Split users into train/val, build tensors for each.
    Returns (train_data, val_data) where each is the 11-tuple from build_dataset().
    """
    # Filter to users that have at least 2 label examples
    final_users = [
        u for u in fs.user_ids
        if 2 <= len(fs.user_to_movie_to_rating_LABEL.get(u, {})) < 500
    ]
    print(f"Final users for training: {len(final_users):,}  "
          f"(skipped {len(fs.user_ids) - len(final_users):,})")

    rng = random.Random(seed)
    rng.shuffle(final_users)
    split = int(len(final_users) * pct_train)
    train_users = final_users[:split]
    val_users   = final_users[split:]

    print(f"Building train dataset ({len(train_users):,} users) ...")
    train_data = build_dataset(train_users, fs)
    print(f"  X_train shape: {train_data[0].shape}")

    print(f"Building val dataset ({len(val_users):,} users) ...")
    val_data   = build_dataset(val_users, fs)
    print(f"  X_val shape:   {val_data[0].shape}")

    return train_data, val_data
