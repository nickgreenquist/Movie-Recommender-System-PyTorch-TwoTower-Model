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
    files = [
        ('movies',       'base_movies.parquet'),
        ('vocab',        'base_vocab.parquet'),
        ('timestamps',   'base_timestamps.parquet'),
        ('movie_tags',   'base_movie_tags.parquet'),
        ('movie_genome', 'base_movie_genome.parquet'),
    ]
    result = {}
    for key, filename in files:
        print(f"  Loading {filename} ...")
        result[key] = pd.read_parquet(os.path.join(data_dir, filename))
    return result


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

    movie_out = os.path.join(data_dir, f'features_movies_{version}.parquet')
    print(f"\nWriting {movie_out} ...")
    _write_list_parquet(movie_df, movie_out)

    print(f"\n✓ features_movies_{version}.parquet → {data_dir}/")
