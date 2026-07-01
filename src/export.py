"""
Stage 5 — Export serving artifacts.

Generates serving/ directory with three files:
  model.pth             — model state_dict
  movie_embeddings.pt   — {movieId: {MOVIE_EMBEDDING_COMBINED, sub-embeddings}}
  feature_store.pt      — inference-only dict (no user data)

The feature_store also carries a baked TRUE 3D (volumetric) projection of every item embedding
(`item_coords_3d`) for the app's 3D Map tab — see `_project_embeddings_3d`. Points fill 3D
space (not pinned to a sphere surface, which would be only 2-dimensional and no more
informative than a flat 2D scatter). The reducers that compute it (UMAP / scikit-learn) are
EXPORT-TIME-ONLY dependencies; the deployed app only ever loads the coordinates, never
re-projects, so they stay out of the serving requirements. `pip install umap-learn` for the
best projection; the export degrades to t-SNE then PCA when it is absent.

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

from src.checkpoint import load_checkpoint
from src.dataset import load_features
from src.evaluate import build_movie_embeddings
from src.features import FEATURES_VERSION
from src.train import build_model, get_config

SERVING_DIR = 'serving'
FACET_STORE_PATH = os.path.join('llm_features', 'cache', 'facet_store.pt')


def _center_3d(xyz: np.ndarray) -> np.ndarray:
    """Center a 3D embedding on the origin (so it rotates about its own middle) WITHOUT touching the
    radial spread — every point keeps its own distance from center, so all THREE axes carry real
    structure. (Contrast a sphere-surface projection, which discards the radius and is therefore
    only 2-dimensional in disguise — no more informative than a flat 2D scatter.)"""
    xyz = np.asarray(xyz, dtype=np.float32)
    return (xyz - xyz.mean(axis=0, keepdims=True)).astype(np.float32)


def _project_embeddings_3d(embeddings: np.ndarray) -> tuple:
    """
    Project (N, D) item embeddings to a TRUE volumetric (N, 3) layout for the app's 3D Map tab.

    Points fill 3D space — they are NOT pinned to a sphere surface. A sphere surface has only two
    intrinsic degrees of freedom (latitude / longitude), so it carries no more information than a 2D
    scatter; a volumetric 3D embedding uses all three axes and preserves more of the high-dimensional
    cosine neighbourhood (lower distortion than 2D).

    Tries reducers in descending quality, returning the first whose library is installed — so the
    export never hard-fails on a missing OPTIONAL, export-time-only viz dependency:
      1. UMAP    (umap-learn)   — n_components=3; the `cosine` input metric mirrors the model's own
                                  similarity (cosine over the genome-tag tower output).
      2. t-SNE   (scikit-learn) — 3 components, cosine; no extra dependency, slower on ~9k points.
      3. PCA     (scikit-learn) — 3 components; linear last resort, always available.

    All centered on the origin (radius preserved). `random_state` is pinned so re-exports are
    reproducible. Returns (xyz (N,3) float32, reducer_name) — the name is baked alongside the coords
    as provenance for the tab caption. None of these libraries are in requirements.txt: the coords
    ship pre-computed, so the deployed app only LOADS them and never imports a reducer.
    """
    n = int(embeddings.shape[0])
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

    try:
        import umap  # umap-learn — export-time only
        xyz = umap.UMAP(
            n_components=3, n_neighbors=15, min_dist=0.1,
            metric='cosine', random_state=42,
        ).fit_transform(embeddings)
        return _center_3d(xyz), 'UMAP (cosine, 3D)'
    except ImportError:
        pass

    try:
        from sklearn.manifold import TSNE
        # perplexity must stay below the sample count; cap it so tiny corpora still project.
        perplexity = min(30, max(5, (n - 1) // 3))
        xyz = TSNE(
            n_components=3, metric='cosine', init='pca',
            perplexity=perplexity, random_state=42,
        ).fit_transform(embeddings)
        return _center_3d(xyz), 't-SNE (cosine, 3D)'
    except ImportError:
        pass

    from sklearn.decomposition import PCA
    xyz = PCA(n_components=3, random_state=42).fit_transform(embeddings)
    return _center_3d(xyz), 'PCA (3D)'


def run_export(data_dir: str = 'data', checkpoint_path: str = None,
               version: str = FEATURES_VERSION, variant: str = None) -> None:
    # variant=None writes the canonical serving artifacts (model.pth, movie_embeddings.pt,
    # feature_store.pt). A non-None variant (e.g. 'no_alpha') is a SECONDARY model deployed
    # alongside prod for an A/B in the app: it writes only its weight-dependent artifacts under
    # suffixed names (model_<variant>.pth, movie_embeddings_<variant>.pt) and SKIPS the shared
    # feature_store.pt — the variant has identical architecture, vocabs and feature buffers to
    # prod (only the trained weights differ, e.g. popularity α), so the app builds BOTH models
    # from the one feature_store. The variant therefore relies on a prior canonical export having
    # already written serving/feature_store.pt.
    suffix     = f'_{variant}' if variant else ''
    model_name = f'model{suffix}.pth'
    emb_name   = f'movie_embeddings{suffix}.pt'
    if variant and not os.path.exists(os.path.join(SERVING_DIR, 'feature_store.pt')):
        print(f"WARNING: serving/feature_store.pt is absent — the '{variant}' variant shares it "
              "with prod. Run a canonical `python main.py export <prod_ckpt>` first.")

    # Resolve checkpoint
    if checkpoint_path is None:
        cfg = get_config()
        checkpoint_dir = cfg['checkpoint_dir']
        candidates = sorted(
            glob.glob(os.path.join(checkpoint_dir, 'best_*.pth')),
            key=os.path.getmtime, reverse=True,
        )
        if not candidates:
            print("No checkpoint found in saved_models/. Train a model first.")
            return
        checkpoint_path = candidates[0]

    print(f"Checkpoint: {checkpoint_path}")
    config, state_dict = load_checkpoint(checkpoint_path)

    print("Loading features ...")
    fs = load_features(data_dir, version)

    model = build_model(config, fs)
    model.load_state_dict(state_dict)
    model.eval()

    print("Building movie embeddings ...")
    movie_embeddings = build_movie_embeddings(model, fs)

    os.makedirs(SERVING_DIR, exist_ok=True)

    # ── model.pth ────────────────────────────────────────────────────────────
    model_path = os.path.join(SERVING_DIR, model_name)
    torch.save(state_dict, model_path)
    print(f"Saved {model_path}  ({os.path.getsize(model_path) / 1e6:.1f} MB)")

    # ── movie_embeddings.pt ──────────────────────────────────────────────────
    emb_path = os.path.join(SERVING_DIR, emb_name)
    torch.save(movie_embeddings, emb_path)
    print(f"Saved {emb_path}  ({os.path.getsize(emb_path) / 1e6:.1f} MB)")

    # A variant export stops here — feature_store.pt (vocabs, buffers, config) is shared with the
    # canonical export, so the app reconstructs the variant model from prod's feature_store.
    if variant:
        print(f"\nVariant '{variant}' export complete — wrote {model_name} + {emb_name}, "
              "reusing the existing serving/feature_store.pt.")
        return

    # ── Popularity ordering (for app dropdowns) ──────────────────────────────
    print("Computing popularity order ...")
    watch_df    = pd.read_parquet(os.path.join(data_dir, 'base_ratings.parquet'))
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

    # ── 3D (volumetric) projection for the Map tab ───────────────────────────
    # Project the item embeddings to a true volumetric (N, 3) layout once, offline, so the app can
    # plot the whole catalog in 3D without ever running UMAP/t-SNE at request time (and without
    # reading data/). coord_movie_ids records the row order — it equals build_movie_embeddings'
    # fs.top_movies order, but baking it explicitly keeps the app robust to any future reorder.
    #
    # We project the GENOME-TAG TOWER output (item_genome_tag_tower, 32-dim) — NOT the 128-dim
    # combined item embedding. The combined vector is dominated by the year/timestamp + item-ID
    # signal and clusters movies by RELEASE ERA (measured head-to-head: mean nearest-neighbor
    # year-gap 4.2y vs 10.4y, and its neighbor lists collapse to same-year titles regardless of
    # genre). The genome projection — the same space the app's Similar tab ranks in — clusters by
    # THEME/CONTENT, which is what the map exists to make legible. Falls back to the combined
    # embedding only for a checkpoint with no genome tower.
    print("Projecting item embeddings to 3D for the Map tab ...")
    coord_movie_ids = list(movie_embeddings.keys())
    sample = movie_embeddings[coord_movie_ids[0]]
    if 'MOVIE_GENOME_TAG_EMBEDDING' in sample:
        coord_key, coord_space = 'MOVIE_GENOME_TAG_EMBEDDING', 'genome-tag content'
    else:
        coord_key, coord_space = 'MOVIE_EMBEDDING_COMBINED', 'combined retrieval'
    emb_matrix = torch.cat(
        [movie_embeddings[m][coord_key] for m in coord_movie_ids], dim=0
    ).cpu().numpy()
    item_coords_3d, reducer_name = _project_embeddings_3d(emb_matrix)
    print(f"  + projected {item_coords_3d.shape[0]} movies to 3D via {reducer_name} on the "
          f"{coord_space} embedding ({coord_key}, {emb_matrix.shape[1]}-dim)")

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
        # 2D embedding projection for the Map tab (additive — older bundles lack these keys
        # and the app hides the tab body behind a notice when they're absent).
        'item_coords_3d':       item_coords_3d,       # (N, 3) float32, true volumetric 3D layout
        'item_coords_movie_ids': coord_movie_ids,     # movieId per row, aligned to item_coords_3d
        'item_coords_reducer':  reducer_name,         # reducer provenance for the tab caption
        'item_coords_space':    coord_space,          # which embedding was projected (genome vs combined)
        # Model config — needed to reconstruct the model in the Streamlit app
        'model_config':         config,
    }

    # ── Self-contained LLM feature buffer ────────────────────────────────────
    # The 'llm'/'both' models' user tower reads llm_feature_buffer at inference. build_model
    # sources it from data/llm_features_*.pt — gitignored, and absent on Streamlit Cloud — so
    # bake the (already-padded, top_movies-ordered) buffer into serving here. Genome needs no
    # equivalent: its buffer is rebuilt in the app from movieId_to_genome_tag_context above.
    if getattr(model, 'has_llm', False):
        feature_store['llm_feature_buffer'] = model.llm_feature_buffer.cpu()
        feature_store['llm_feature_len']    = int(model.llm_feature_buffer.shape[1])
        print(f"  + llm_feature_buffer {tuple(model.llm_feature_buffer.shape)} baked into feature_store")

    # ── Scraped-facet store (people facets) ──────────────────────────────────
    # The front-end resolves/filters people ("Tom Hanks movies") against the facet store built by
    # llm_features/build_facet_store.py from the TMDB credits scrape (cache absent on Streamlit
    # Cloud), so bake its tables into serving here — mirroring llm_feature_buffer above. It's
    # optional: if the build artifact is missing (no scrape), skip with a warning rather than fail.
    if os.path.exists(FACET_STORE_PATH):
        facets = torch.load(FACET_STORE_PATH, weights_only=False)
        feature_store['facets'] = facets
        meta = facets.get('meta', {})
        print(f"  + facets baked into feature_store "
              f"({meta.get('n_persons', '?')} persons, {meta.get('n_movies_covered', '?')} movies)")
    else:
        print(f"  ! {FACET_STORE_PATH} absent — people facets NOT baked "
              f"(run: python llm_features/build_facet_store.py)")

    fs_path = os.path.join(SERVING_DIR, 'feature_store.pt')
    torch.save(feature_store, fs_path)
    print(f"Saved {fs_path}  ({os.path.getsize(fs_path) / 1e6:.1f} MB)")

    total_mb = sum(
        os.path.getsize(os.path.join(SERVING_DIR, f)) / 1e6
        for f in ('model.pth', 'movie_embeddings.pt', 'feature_store.pt')
    )
    print(f"\nTotal serving/ size: {total_mb:.1f} MB")
    print("Done. Verify with: python -c \"import torch; fs=torch.load('serving/feature_store.pt', weights_only=False); print(len(fs['top_movies']), 'movies')\"")
