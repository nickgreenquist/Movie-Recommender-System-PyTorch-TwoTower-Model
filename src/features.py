"""
Stage 2 — Feature Engineering
Reads base_*.parquet, builds per-movie and per-user feature vectors.
Re-run this (not preprocess) when iterating on feature ideas.

Usage:
    python main.py features
"""
import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


FEATURES_VERSION = 'v1'
MAX_HISTORY_LEN  = 50   # cap watch history to most recent N movies


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_base(data_dir: str) -> dict:
    return {
        'movies':       pd.read_parquet(os.path.join(data_dir, 'base_movies.parquet')),
        'vocab':        pd.read_parquet(os.path.join(data_dir, 'base_vocab.parquet')),
        'watch':        pd.read_parquet(os.path.join(data_dir, 'base_ratings_watch.parquet')),
        'labels':       pd.read_parquet(os.path.join(data_dir, 'base_ratings_labels.parquet')),
        'timestamps':   pd.read_parquet(os.path.join(data_dir, 'base_timestamps.parquet')),
        'movie_tags':   pd.read_parquet(os.path.join(data_dir, 'base_movie_tags.parquet')),
        'movie_genome': pd.read_parquet(os.path.join(data_dir, 'base_movie_genome.parquet')),
    }


def parse_vocab(vocab_df: pd.DataFrame) -> dict:
    """Extract ordered vocabulary lists and index maps from base_vocab.parquet."""
    g  = vocab_df[vocab_df['type'] == 'genre'].sort_values('index')
    t  = vocab_df[vocab_df['type'] == 'tag'].sort_values('index')
    gt = vocab_df[vocab_df['type'] == 'genome_tag'].sort_values('index')
    y  = vocab_df[vocab_df['type'] == 'year'].sort_values('index')

    return {
        'genres_ordered':    g['value'].tolist(),
        'tags_ordered':      t['value'].tolist(),
        'genome_tag_ids':    [int(v) for v in gt['value'].tolist()],
        'years_ordered':     y['value'].tolist(),
        'genre_to_i':        dict(zip(g['value'], g['index'].astype(int))),
        'tag_to_i':          dict(zip(t['value'], t['index'].astype(int))),
        'genome_tag_to_i':   {int(r['value']): int(r['index']) for _, r in gt.iterrows()},
        'genome_tag_names':  {int(r['value']): r['extra']      for _, r in gt.iterrows()},
        'year_to_i':         dict(zip(y['value'], y['index'].astype(int))),
    }


# ── Per-movie features ────────────────────────────────────────────────────────

def build_movie_features(base: dict, vocab: dict) -> pd.DataFrame:
    """
    Returns DataFrame with one row per movie:
      movieId, year, genre_context, tag_context, genome_tag_context
    All context columns are list[float].
    """
    movies_df       = base['movies']
    movie_tags_df   = base['movie_tags']
    movie_genome_df = base['movie_genome']

    genre_to_i      = vocab['genre_to_i']
    tag_to_i        = vocab['tag_to_i']
    genome_tag_to_i = vocab['genome_tag_to_i']
    n_genres        = len(genre_to_i)
    n_tags          = len(tag_to_i)
    n_genome        = len(genome_tag_to_i)

    # Index lookups
    movieId_to_genres = {int(r['movieId']): r['genres'] for _, r in movies_df.iterrows()}

    # Per-movie tag context
    from tqdm import tqdm
    movieId_to_tag_ctx = {}
    for _, row in tqdm(movie_tags_df.iterrows(), total=len(movie_tags_df), desc="Movie tag contexts"):
        mid   = int(row['movieId'])
        total = int(row['total_tag_count'])
        vec   = [0.0] * n_tags
        if total > 0:
            for tag, cnt in zip(row['tags'], row['tag_counts']):
                if tag in tag_to_i:
                    vec[tag_to_i[tag]] = float(cnt) / total
        movieId_to_tag_ctx[mid] = vec

    # Per-movie genome tag context
    movieId_to_genome_ctx = {}
    for _, row in tqdm(movie_genome_df.iterrows(), total=len(movie_genome_df), desc="Movie genome contexts"):
        mid = int(row['movieId'])
        vec = [0.0] * n_genome
        for tid, score in zip(row['tagIds'], row['scores']):
            tid = int(tid)
            if tid in genome_tag_to_i:
                vec[genome_tag_to_i[tid]] = float(score)
        movieId_to_genome_ctx[mid] = vec

    rows = []
    for _, mrow in tqdm(movies_df.iterrows(), total=len(movies_df), desc="Movie features"):
        mid  = int(mrow['movieId'])
        year = str(mrow['year'])

        genre_ctx  = [0.0] * n_genres
        for g in movieId_to_genres.get(mid, []):
            if g in genre_to_i:
                genre_ctx[genre_to_i[g]] = 1.0

        rows.append({
            'movieId':            mid,
            'year':               year,
            'genre_context':      genre_ctx,
            'tag_context':        movieId_to_tag_ctx.get(mid,    [0.0] * n_tags),
            'genome_tag_context': movieId_to_genome_ctx.get(mid, [0.0] * n_genome),
        })

    df = pd.DataFrame(rows)
    print(f"  Movie features: {len(df)} movies  "
          f"(genres={n_genres}, tags={n_tags}, genome_tags={n_genome})")
    return df


# ── Per-user features ─────────────────────────────────────────────────────────

def build_user_features(base: dict, movie_df: pd.DataFrame, vocab: dict) -> pd.DataFrame:
    """
    Returns DataFrame with one row per user:
      userId, avg_rating, genre_context, watch_history (emb indices),
      watch_history_ratings, tag_context,
      label_movieIds, label_ratings, label_timestamps
    All list columns are list[float] or list[int].
    """
    watch_df   = base['watch']
    labels_df  = base['labels']
    movies_df  = base['movies']

    genre_to_i = vocab['genre_to_i']
    genres_ord = vocab['genres_ordered']
    n_genres   = len(genre_to_i)
    n_tags     = len(vocab['tag_to_i'])

    top_movies            = movies_df['movieId'].tolist()
    item_emb_movieId_to_i = {int(mid): i for i, mid in enumerate(top_movies)}

    movieId_to_genres = {int(r['movieId']): r['genres'] for _, r in movies_df.iterrows()}

    # movieId → tag_context (from movie_df)
    movieId_to_tag_ctx = {int(r['movieId']): r['tag_context'] for _, r in movie_df.iterrows()}

    # Per-user avg rating
    avg_ratings = watch_df.groupby('userId')['rating'].mean().to_dict()

    # Per-user genre stats (from watch history) — vectorized via explode + groupby
    print("  Computing user genre stats ...")
    _wg = watch_df[['userId', 'movieId', 'rating']].copy()
    _wg['genre'] = _wg['movieId'].map(movieId_to_genres)
    _wg = _wg.explode('genre').dropna(subset=['genre'])
    _agg = _wg.groupby(['userId', 'genre']).agg(N=('rating', 'count'), S=('rating', 'sum')).reset_index()

    # genre index helpers
    avg_idx   = {g: i           for i, g in enumerate(genres_ord)}
    watch_idx = {g: n_genres + i for i, g in enumerate(genres_ord)}

    # Vectorized genre context matrix — numpy scatter from _agg
    print("  Building genre context matrix ...")
    total_N     = _agg.groupby('userId')['N'].sum()
    all_uids    = list(total_N.index)
    uid_to_row  = {int(uid): i for i, uid in enumerate(all_uids)}

    _ctx = _agg.copy()
    _ctx['total_N']   = _ctx['userId'].map(total_N)
    _ctx['avg_rat']   = _ctx['userId'].map(avg_ratings)
    _ctx['avg_g']     = _ctx['S'] / _ctx['N']
    _ctx['val_avg']   = _ctx['avg_g'] - _ctx['avg_rat']
    _ctx['val_watch'] = _ctx['N'] / _ctx['total_N']
    _ctx['col_avg']   = _ctx['genre'].map(avg_idx)
    _ctx['col_watch'] = _ctx['genre'].map(watch_idx)
    _ctx = _ctx.dropna(subset=['col_avg'])
    _ctx['col_avg']   = _ctx['col_avg'].astype(int)
    _ctx['col_watch'] = _ctx['col_watch'].astype(int)
    _ctx['row']       = _ctx['userId'].map(uid_to_row).astype(int)

    genre_ctx_matrix = np.zeros((len(all_uids), 2 * n_genres), dtype=np.float32)
    genre_ctx_matrix[_ctx['row'].values, _ctx['col_avg'].values]   = _ctx['val_avg'].values
    genre_ctx_matrix[_ctx['row'].values, _ctx['col_watch'].values] = _ctx['val_watch'].values

    # Pre-build tag matrix (n_top_movies × n_tags) for fast numpy row-mean
    print("  Building tag matrix ...")
    tag_matrix = np.zeros((len(top_movies), n_tags), dtype=np.float32)
    for mid, idx in item_emb_movieId_to_i.items():
        ctx = movieId_to_tag_ctx.get(mid)
        if ctx:
            tag_matrix[idx] = ctx

    # Aggregate watch/label history per user
    watch_agg = (watch_df
                 .groupby('userId')
                 .agg(movieIds=('movieId', list), ratings=('rating', list))
                 .reset_index())
    label_agg = (labels_df
                 .groupby('userId')
                 .agg(movieIds=('movieId', list), ratings=('rating', list),
                      timestamps=('timestamp', list))
                 .reset_index())

    watch_by_user = {int(r['userId']): r for _, r in watch_agg.iterrows()}
    label_by_user = {int(r['userId']): r for _, r in label_agg.iterrows()}

    from tqdm import tqdm
    rows = []
    for uid in tqdm(all_uids, desc="User features"):
        uid     = int(uid)
        avg_rat = float(avg_ratings.get(uid, 3.0))

        # Genre context — O(1) lookup from precomputed matrix
        ctx = genre_ctx_matrix[uid_to_row[uid]].tolist()

        # Watch history → emb indices + debiased ratings, cap to MAX_HISTORY_LEN
        wrow = watch_by_user.get(uid)
        if wrow is not None:
            pairs = [
                (item_emb_movieId_to_i[int(mid)], float(rat) - avg_rat)
                for mid, rat in zip(wrow['movieIds'], wrow['ratings'])
                if int(mid) in item_emb_movieId_to_i
            ][-MAX_HISTORY_LEN:]
        else:
            pairs = []
        hist_ids     = [p[0] for p in pairs]
        hist_ratings = [p[1] for p in pairs]

        # Tag context — numpy row-mean over watched movie tag vectors
        if hist_ids:
            tag_ctx = tag_matrix[hist_ids].mean(axis=0).tolist()
        else:
            tag_ctx = [0.0] * n_tags

        # Labels
        lrow = label_by_user.get(uid)
        if lrow is not None:
            lbl_movies = [int(m) for m in lrow['movieIds']]
            lbl_rats   = [float(r) for r in lrow['ratings']]
            lbl_times  = [int(t) for t in lrow['timestamps']]
        else:
            lbl_movies = lbl_rats = lbl_times = []

        rows.append({
            'userId':                uid,
            'avg_rating':            avg_rat,
            'genre_context':         ctx,
            'watch_history':         hist_ids,
            'watch_history_ratings': hist_ratings,
            'tag_context':           tag_ctx,
            'label_movieIds':        lbl_movies,
            'label_ratings':         lbl_rats,
            'label_timestamps':      lbl_times,
        })

    df = pd.DataFrame(rows)
    print(f"  User features: {len(df)} users")
    return df


# ── Parquet writer (handles list columns) ─────────────────────────────────────

def _write_list_parquet(df: pd.DataFrame, path: str) -> None:
    arrays = {}
    for col in df.columns:
        sample = df[col].iloc[0] if len(df) > 0 else None
        if isinstance(sample, list) and sample and isinstance(sample[0], float):
            arrays[col] = pa.array(df[col].tolist(), type=pa.list_(pa.float32()))
        elif isinstance(sample, list) and sample and isinstance(sample[0], int):
            arrays[col] = pa.array(df[col].tolist(), type=pa.list_(pa.int32()))
        elif isinstance(sample, list):
            arrays[col] = pa.array(df[col].tolist(), type=pa.list_(pa.float32()))
        else:
            arrays[col] = pa.array(df[col].tolist())
    pq.write_table(pa.table(arrays), path)


# ── Orchestrator ──────────────────────────────────────────────────────────────

def run(data_dir: str = 'data', version: str = FEATURES_VERSION) -> None:
    print(f"Loading base parquets from {data_dir}/ ...")
    base  = load_base(data_dir)
    vocab = parse_vocab(base['vocab'])

    print("\n── Building movie features ──")
    movie_df = build_movie_features(base, vocab)

    print("\n── Building user features ──")
    user_df = build_user_features(base, movie_df, vocab)

    movie_out = os.path.join(data_dir, f'features_movies_{version}.parquet')
    user_out  = os.path.join(data_dir, f'features_users_{version}.parquet')

    print(f"\nWriting {movie_out} ...")
    _write_list_parquet(movie_df, movie_out)
    print(f"Writing {user_out} ...")
    _write_list_parquet(user_df, user_out)

    print(f"\n✓ features_movies_{version}.parquet and features_users_{version}.parquet → {data_dir}/")
