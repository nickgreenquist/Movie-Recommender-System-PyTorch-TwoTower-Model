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
