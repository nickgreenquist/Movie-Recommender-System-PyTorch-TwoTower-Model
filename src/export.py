"""
Stage 5 — Export serving artifacts.

Generates serving/ directory with three files:
  model.pth             — model state_dict
  movie_embeddings.pt   — {movieId: {MOVIE_EMBEDDING_COMBINED, sub-embeddings}}
  feature_store.pt      — inference-only dict (no user data)

Usage:
    python main.py export
    python main.py export <checkpoint_path>
"""
import glob
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.dataset import load_features
from src.evaluate import build_movie_embeddings
from src.train import build_model, get_config

SERVING_DIR = 'serving'


def run_export(data_dir: str = 'data', checkpoint_path: str = None,
               version: str = 'v1') -> None:
    # Resolve checkpoint
    if checkpoint_path is None:
        cfg = get_config()
        checkpoint_dir = cfg['checkpoint_dir']
        candidates = sorted(
            glob.glob(os.path.join(checkpoint_dir, 'best_mse_*.pth')),
            key=os.path.getmtime, reverse=True,
        )
        if not candidates:
            print("No checkpoint found in saved_models/. Train a model first.")
            return
        checkpoint_path = candidates[0]

    config = get_config()

    print(f"Checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, weights_only=True)

    sd = state_dict
    item_id_dim = sd['item_embedding_lookup.weight'].shape[1]
    ts_dim      = sd['timestamp_embedding_lookup.weight'].shape[1]
    genre_dim   = sd['user_genre_tower.0.weight'].shape[0]
    genome_dim  = sd['item_genome_tag_tower.0.weight'].shape[0]
    config['item_movieId_embedding_size']      = item_id_dim
    config['user_genre_embedding_size']        = genre_dim
    config['timestamp_feature_embedding_size'] = ts_dim
    config['item_genre_embedding_size']        = sd['item_genre_tower.0.weight'].shape[0]
    config['item_tag_embedding_size']          = sd['item_tag_tower.0.weight'].shape[0]
    config['item_genome_tag_embedding_size']   = genome_dim
    config['item_year_embedding_size']         = sd['year_embedding_lookup.weight'].shape[1]

    config['user_genome_context_embedding_size'] = sd['user_genome_context_tower.0.weight'].shape[0]

    config['proj_hidden'] = sd['user_projection.0.weight'].shape[0]
    config['output_dim']  = sd['user_projection.2.weight'].shape[0]

    print("Loading features ...")
    fs = load_features(data_dir, version)

    model = build_model(config, fs)
    model.load_state_dict(state_dict)
    model.eval()

    print("Building movie embeddings ...")
    movie_embeddings = build_movie_embeddings(model, fs)

    os.makedirs(SERVING_DIR, exist_ok=True)

    # ── model.pth ────────────────────────────────────────────────────────────
    model_path = os.path.join(SERVING_DIR, 'model.pth')
    torch.save(state_dict, model_path)
    print(f"Saved {model_path}  ({os.path.getsize(model_path) / 1e6:.1f} MB)")

    # ── movie_embeddings.pt ──────────────────────────────────────────────────
    emb_path = os.path.join(SERVING_DIR, 'movie_embeddings.pt')
    torch.save(movie_embeddings, emb_path)
    print(f"Saved {emb_path}  ({os.path.getsize(emb_path) / 1e6:.1f} MB)")

    # ── Popularity ordering (for app dropdowns) ──────────────────────────────
    print("Computing popularity order ...")
    watch_df    = pd.read_parquet(os.path.join(data_dir, 'base_ratings_watch.parquet'))
    mid_counts  = watch_df.groupby('movieId').size()
    sorted_mids = mid_counts.sort_values(ascending=False).index.tolist()
    # Keep only top_movies, preserve popularity rank
    top_set             = set(fs.top_movies)
    popularity_ordered_titles = [
        fs.movieId_to_title[mid]
        for mid in sorted_mids
        if mid in top_set and mid in fs.movieId_to_title
    ]
    # Append any movies missing from watch data at the end
    covered = set(popularity_ordered_titles)
    for mid in fs.top_movies:
        t = fs.movieId_to_title.get(mid)
        if t and t not in covered:
            popularity_ordered_titles.append(t)

    # ── feature_store.pt ─────────────────────────────────────────────────────
    feature_store = {
        # Movie titles ordered by rating count (for app dropdowns)
        'popularity_ordered_titles': popularity_ordered_titles,
        # Vocabularies
        'top_movies':              fs.top_movies,
        'genres_ordered':          fs.genres_ordered,
        'tags_ordered':            fs.tags_ordered,
        'genome_tag_ids':          fs.genome_tag_ids,
        'genome_tag_names':        fs.genome_tag_names,
        'years_ordered':           fs.years_ordered,
        # Index maps
        'genre_to_i':              fs.genre_to_i,
        'tag_to_i':                fs.tag_to_i,
        'genome_tag_to_i':         fs.genome_tag_to_i,
        'year_to_i':               fs.year_to_i,
        'item_emb_movieId_to_i':   fs.item_emb_movieId_to_i,
        # Per-movie lookups
        'movieId_to_title':        fs.movieId_to_title,
        'title_to_movieId':        fs.title_to_movieId,
        'movieId_to_year':         fs.movieId_to_year,
        'movieId_to_genres':       fs.movieId_to_genres,
        # Context dicts stored as numpy float32 arrays (not Python lists) to avoid
        # pickle overhead — Python floats are ~28 bytes each vs 4 bytes for float32.
        'movieId_to_genre_context': {
            mid: np.array(v, dtype=np.float32)
            for mid, v in fs.movieId_to_genre_context.items()
        },
        'movieId_to_tag_context': {
            mid: np.array(v, dtype=np.float32)
            for mid, v in fs.movieId_to_tag_context.items()
        },
        'movieId_to_genome_tag_context': {
            mid: np.array(v, dtype=np.float32)
            for mid, v in fs.movieId_to_genome_tag_context.items()
        },
        # User context index maps (needed for canary-style inference in the app)
        'user_context_genre_avg_rating_to_i':  fs.user_context_genre_avg_rating_to_i,
        'user_context_genre_watch_count_to_i': fs.user_context_genre_watch_count_to_i,
        # Derived constants
        'user_context_size':    fs.user_context_size,
        'timestamp_num_bins':   fs.timestamp_num_bins,
        'timestamp_bins':       fs.timestamp_bins,
        # Model config — needed to reconstruct the model in the Streamlit app
        'model_config':         config,
    }
    fs_path = os.path.join(SERVING_DIR, 'feature_store.pt')
    torch.save(feature_store, fs_path)
    print(f"Saved {fs_path}  ({os.path.getsize(fs_path) / 1e6:.1f} MB)")

    total_mb = sum(
        os.path.getsize(os.path.join(SERVING_DIR, f)) / 1e6
        for f in ('model.pth', 'movie_embeddings.pt', 'feature_store.pt')
    )
    print(f"\nTotal serving/ size: {total_mb:.1f} MB")
    print("Done. Verify with: python -c \"import torch; fs=torch.load('serving/feature_store.pt', weights_only=False); print(len(fs['top_movies']), 'movies')\"")
