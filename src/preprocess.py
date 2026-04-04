"""
Stage 1 — Base Preprocessing
Run once on raw CSVs. Outputs data/base_*.parquet.

Usage:
    python main.py preprocess
"""
import os
import re

import pandas as pd
from tqdm import tqdm


# ── Constants ────────────────────────────────────────────────────────────────

MIN_RATINGS_PER_MOVIE = 1_000
MIN_RATINGS_PER_USER  = 20
MAX_RATINGS_PER_USER  = 500
MIN_NUM_TAGS          = 1_000       # user-applied tags must appear this often across all movies
PERCENT_WATCH_HISTORY = 0.9         # fraction of each user's ratings used as watch history


# ── Loaders ──────────────────────────────────────────────────────────────────

def load_raw(raw_dir: str) -> dict:
    """Load all five raw CSVs. Returns a dict of DataFrames."""
    print("Loading ratings.csv ...")
    df_ratings = pd.read_csv(os.path.join(raw_dir, 'ratings.csv'), nrows=35_000_000)
    df_ratings = df_ratings.dropna()
    df_ratings['movieId'] = df_ratings['movieId'].astype(int)

    print("Loading movies.csv ...")
    df_movies = pd.read_csv(os.path.join(raw_dir, 'movies.csv'))

    print("Loading tags.csv ...")
    df_tags = pd.read_csv(os.path.join(raw_dir, 'tags.csv'))
    df_tags['tag'] = df_tags['tag'].str.lower().str.strip()

    print("Loading genome-tags.csv ...")
    df_genome_tags = pd.read_csv(os.path.join(raw_dir, 'genome-tags.csv'))

    print("Loading genome-scores.csv ...")
    df_genome_scores = pd.read_csv(os.path.join(raw_dir, 'genome-scores.csv'))

    return {
        'ratings':       df_ratings,
        'movies':        df_movies,
        'tags':          df_tags,
        'genome_tags':   df_genome_tags,
        'genome_scores': df_genome_scores,
    }


# ── Corpus building ───────────────────────────────────────────────────────────

def build_corpus(dfs: dict) -> tuple:
    """
    Filter to movies with MIN_RATINGS_PER_MOVIE+ ratings.
    Returns (top_movies: list[int], movies_df: DataFrame).
    movies_df columns: movieId, title, year, genres (list[str]).
    """
    df_ratings = dfs['ratings']
    df_movies  = dfs['movies']

    counts = df_ratings.groupby('movieId')['rating'].count().reset_index()
    counts.columns = ['movieId', 'num_ratings']
    top_movie_ids = set(counts.loc[counts['num_ratings'] > MIN_RATINGS_PER_MOVIE, 'movieId'].tolist())

    print(f"Total movies in corpus: {len(counts)}")
    print(f"Movies with {MIN_RATINGS_PER_MOVIE}+ ratings: {len(top_movie_ids)}")

    rows = []
    for _, row in df_movies.iterrows():
        mid = int(row['movieId'])
        if mid not in top_movie_ids:
            continue
        title = row['title']
        match = re.search(r"\(\d+\)\s*$", title)
        year  = title[match.start()+1:match.end()-1] if match else '-1'
        genres = [g for g in str(row['genres']).split('|') if g]
        rows.append({'movieId': mid, 'title': title, 'year': year, 'genres': genres})

    movies_df  = pd.DataFrame(rows)
    top_movies = movies_df['movieId'].tolist()   # ordered list; order matters for embedding index
    return top_movies, movies_df


# ── Vocabulary building ───────────────────────────────────────────────────────

def build_vocab(dfs: dict, top_movies: list) -> pd.DataFrame:
    """
    Build ordered vocabularies for genres, user-applied tags, genome tags, and years.
    Returns a single DataFrame with columns: type, index, value, extra.
      type='genre'      value=genre_name    extra=''
      type='tag'        value=tag_name      extra=''
      type='genome_tag' value=str(tagId)    extra=tag_name
      type='year'       value=year_str      extra=''
    """
    df_tags        = dfs['tags']
    df_genome_tags = dfs['genome_tags']
    df_movies_meta = dfs['movies']

    top_movies_set = set(top_movies)

    # Genres (sorted for determinism)
    all_genres = set()
    for _, row in df_movies_meta.iterrows():
        if int(row['movieId']) not in top_movies_set:
            continue
        for g in str(row['genres']).split('|'):
            if g:
                all_genres.add(g)
    genres_ordered = sorted(all_genres)

    # User-applied tags (min count threshold, sorted for determinism)
    counts = df_tags.groupby('tag').size().reset_index(name='count')
    final_tags = sorted(counts.loc[counts['count'] > MIN_NUM_TAGS, 'tag'].tolist())

    # Genome tag IDs (sorted for determinism)
    genome_tag_ids   = sorted(df_genome_tags['tagId'].tolist())
    genome_tag_names = dict(zip(df_genome_tags['tagId'], df_genome_tags['tag']))

    # Years from top movies
    years_seen = set()
    for _, row in df_movies_meta.iterrows():
        mid = int(row['movieId'])
        title = row['title']
        match = re.search(r"\(\d+\)\s*$", title)
        year  = title[match.start()+1:match.end()-1] if match else '-1'
        years_seen.add(year)
    years_ordered = sorted(years_seen)

    rows = []
    for i, g in enumerate(genres_ordered):
        rows.append({'type': 'genre', 'index': i, 'value': g, 'extra': ''})
    for i, t in enumerate(final_tags):
        rows.append({'type': 'tag', 'index': i, 'value': t, 'extra': ''})
    for i, tid in enumerate(genome_tag_ids):
        rows.append({'type': 'genome_tag', 'index': i, 'value': str(tid), 'extra': genome_tag_names[tid]})
    for i, y in enumerate(years_ordered):
        rows.append({'type': 'year', 'index': i, 'value': y, 'extra': ''})

    print(f"Vocab sizes — genres: {len(genres_ordered)}, tags: {len(final_tags)}, "
          f"genome_tags: {len(genome_tag_ids)}, years: {len(years_ordered)}")
    return pd.DataFrame(rows)


# ── User history splitting ────────────────────────────────────────────────────

def split_user_history(dfs: dict, top_movies: list) -> tuple:
    """
    Filter ratings to top movies, filter users by rating count,
    split each user's history 90% watch / 10% labels (chronological order).
    Returns (watch_df, labels_df) each with columns: userId, movieId, rating, timestamp.
    """
    df_ratings    = dfs['ratings']
    top_movies_set = set(top_movies)

    df_filtered = df_ratings[df_ratings['movieId'].isin(top_movies_set)].copy()
    df_filtered = df_filtered.sort_values(['userId', 'timestamp'])

    df_agg = df_filtered.groupby('userId').agg(
        movieId   = ('movieId',   list),
        rating    = ('rating',    list),
        timestamp = ('timestamp', list),
    ).reset_index()

    watch_rows  = []
    label_rows  = []
    too_few = too_many = 0

    for _, row in tqdm(df_agg.iterrows(), total=len(df_agg), desc="Splitting user histories"):
        n = len(row['movieId'])
        if n < MIN_RATINGS_PER_USER:
            too_few += 1
            continue
        if n > MAX_RATINGS_PER_USER:
            too_many += 1
            continue

        uid     = int(row['userId'])
        split   = int(n * PERCENT_WATCH_HISTORY)
        movies  = row['movieId']
        ratings = row['rating']
        times   = row['timestamp']

        for i in range(split):
            watch_rows.append({'userId': uid, 'movieId': movies[i],
                                'rating': ratings[i], 'timestamp': times[i]})
        for i in range(split, n):
            label_rows.append({'userId': uid, 'movieId': movies[i],
                                'rating': ratings[i], 'timestamp': times[i]})

    watch_df  = pd.DataFrame(watch_rows)
    labels_df = pd.DataFrame(label_rows)

    print(f"Users kept: {len(watch_df['userId'].unique())}  "
          f"(skipped too_few={too_few}, too_many={too_many})")
    print(f"Watch rows: {len(watch_df):,}   Label rows: {len(labels_df):,}")
    return watch_df, labels_df


# ── Per-movie tag / genome helpers ───────────────────────────────────────────

def _build_movie_tag_counts(dfs: dict, top_movies: list, vocab_df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame (movieId, tags: list[str], tag_counts: list[int], total_tag_count: int)
    for use by features.py to build tag context vectors.
    Only tags in the vocab (final_movie_tags) are included.
    """
    df_tags = dfs['tags']
    final_tags = set(vocab_df.loc[vocab_df['type'] == 'tag', 'value'].tolist())
    top_movies_set = set(top_movies)

    # movieId → {tag: count}
    tag_counts: dict = {}
    for _, row in tqdm(df_tags.iterrows(), total=len(df_tags), desc="Building movie tag counts"):
        mid = int(row['movieId'])
        if mid not in top_movies_set:
            continue
        tag = row['tag']
        if tag not in final_tags:
            continue
        if mid not in tag_counts:
            tag_counts[mid] = {}
        tag_counts[mid][tag] = tag_counts[mid].get(tag, 0) + 1

    rows = []
    for mid in top_movies:
        tc = tag_counts.get(mid, {})
        tags_list   = list(tc.keys())
        counts_list = [tc[t] for t in tags_list]
        total       = sum(counts_list)
        rows.append({'movieId': mid, 'tags': tags_list,
                     'tag_counts': counts_list, 'total_tag_count': total})

    return pd.DataFrame(rows)


def _build_movie_genome_scores(dfs: dict, top_movies: list) -> pd.DataFrame:
    """
    Returns a DataFrame (movieId, tagIds: list[int], scores: list[float])
    for use by features.py to build genome tag context vectors.
    """
    df_genome_scores = dfs['genome_scores']
    top_movies_set   = set(top_movies)

    df_top = df_genome_scores[df_genome_scores['movieId'].isin(top_movies_set)]
    agg = df_top.groupby('movieId').agg(
        tagIds = ('tagId',     list),
        scores = ('relevance', list),
    ).reset_index()

    # Ensure all top movies are present (fill missing with empty)
    present = set(agg['movieId'].tolist())
    extra_rows = [{'movieId': mid, 'tagIds': [], 'scores': []}
                  for mid in top_movies if mid not in present]
    if extra_rows:
        agg = pd.concat([agg, pd.DataFrame(extra_rows)], ignore_index=True)

    return agg


# ── Orchestrator ──────────────────────────────────────────────────────────────

def run(raw_dir: str = 'data/ml-32m', out_dir: str = 'data') -> None:
    os.makedirs(out_dir, exist_ok=True)

    dfs = load_raw(raw_dir)

    print("\n── Building corpus ──")
    top_movies, movies_df = build_corpus(dfs)

    print("\n── Building vocabulary ──")
    vocab_df = build_vocab(dfs, top_movies)

    print("\n── Splitting user histories ──")
    watch_df, labels_df = split_user_history(dfs, top_movies)

    # Write parquets
    movies_df.to_parquet(os.path.join(out_dir, 'base_movies.parquet'), index=False)
    vocab_df.to_parquet(os.path.join(out_dir, 'base_vocab.parquet'), index=False)
    watch_df.to_parquet(os.path.join(out_dir, 'base_ratings_watch.parquet'), index=False)
    labels_df.to_parquet(os.path.join(out_dir, 'base_ratings_labels.parquet'), index=False)

    # Per-movie tag counts (for features.py tag context vectors)
    movie_tags_df = _build_movie_tag_counts(dfs, top_movies, vocab_df)
    movie_tags_df.to_parquet(os.path.join(out_dir, 'base_movie_tags.parquet'), index=False)

    # Per-movie genome scores (for features.py genome tag context vectors)
    movie_genome_df = _build_movie_genome_scores(dfs, top_movies)
    movie_genome_df.to_parquet(os.path.join(out_dir, 'base_movie_genome.parquet'), index=False)

    # Global min/max timestamp for bucketizing (needed by dataset.py)
    ts_df = pd.DataFrame({'ts_min': [int(dfs['ratings']['timestamp'].min())],
                          'ts_max': [int(dfs['ratings']['timestamp'].max())]})
    ts_df.to_parquet(os.path.join(out_dir, 'base_timestamps.parquet'), index=False)

    print(f"\n✓ Wrote base_movies, base_vocab, base_ratings_watch, base_ratings_labels, "
          f"base_movie_tags, base_movie_genome, base_timestamps  →  {out_dir}/")
