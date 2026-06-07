"""
Stage 4 (tensor) — Build the LLM content-feature tensor from merged extractions
                   (LLM-vs-genome ablation)

Converts the per-movie merged dicts (cache/llm_merged/<tag>/<movieId>.json, Stage 3)
into a single (n_corpus + 1, 132) float32 tensor that fills the model's swappable
content slot EXACTLY the way the genome buffer does in src/train.build_model:
  • row i           = the movie at corpus index i (fs.top_movies order)
  • final row       = the zero pad row (pad_idx = len(top_movies))
This mirrors `np.vstack([content_matrix, pad_row])` there, so Model B can load it into
content_context_buffer with content_feature_source='llm' (the 'llm' branch in build_model
is the remaining bridge — it must point the item_content_tower input at THIS tensor's
width, 132, instead of the genome 1128).

Corpus order is read from base_movies<corpus>.parquet (movieId column) — IDENTICAL to
src/dataset.load_features' `top_movies = movies_df['movieId'].tolist()` — so tensor rows
align to the trained item-embedding rows. Set CORPUS=phase1 for the reduced corpus.

Movies with no extraction get an all-zero row (and are excluded from the similarity
check). Also runs a cosine-similarity sanity check over the movies that DO have features
(genre / sub-genre should cluster; unrelated films should not) — the Stage 4 gate before
training, the LLM-side analogue of the plan's "Toy Story 1/2/3 similar" check.

Usage (standalone — no API; needs the merged cache from Stage 3; run with -m so `data/`
is found from repo root):
    CORPUS=phase1 python -m llm_features.build_features                    # DEFAULT_TAG
    CORPUS=phase1 python -m llm_features.build_features manual-opus-4-8    # specific tag
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

from llm_features.schemas import FEATURE_ORDER


# ── Paths / constants ─────────────────────────────────────────────────────────

REPO_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(REPO_ROOT, 'data')
MERGED_DIR = os.path.join(REPO_ROOT, 'llm_features', 'cache', 'llm_merged')

DEFAULT_TAG      = 'claude-code-sonnet'
FEATURES_VERSION = 'v1'

# Recognisable movies for the similarity readout (label only; any extracted movie is
# included in the check). Same spirit as the plan's "similar movies → similar features".
SANITY_TITLES = {
    1: 'Toy Story', 2: 'Jumanji', 296: 'Pulp Fiction', 741: 'Ghost in the Shell',
    1587: 'Conan', 48394: "Pan's Labyrinth", 68237: 'Moon', 61240: 'Let the Right One In',
}


# ── Corpus order (mirror src/dataset.load_features) ───────────────────────────

def corpus_suffix() -> tuple:
    """('' | '_<corpus>', corpus) from the CORPUS env — mirrors src/corpus.py."""
    corpus = os.environ.get('CORPUS', 'full')
    if corpus not in ('full', 'phase1'):
        raise ValueError(f"Unknown CORPUS={corpus!r}; expected 'full' or 'phase1'")
    return ('' if corpus == 'full' else f'_{corpus}'), corpus


def load_top_movies(sfx: str) -> list:
    """Corpus movieId order — read the SAME way src/dataset.load_features reads it."""
    df = pd.read_parquet(os.path.join(DATA_DIR, f'base_movies{sfx}.parquet'))
    return df['movieId'].tolist()


# ── Tensor build ──────────────────────────────────────────────────────────────

def build_tensor(tag: str, top_movies: list) -> tuple:
    """(n+1, 132) float32: row i = top_movies[i] features (0.0 if unextracted), pad last."""
    merged_dir = os.path.join(MERGED_DIR, tag)
    n_dims = len(FEATURE_ORDER)
    matrix = np.zeros((len(top_movies), n_dims), dtype=np.float32)
    covered = []
    for i, mid in enumerate(top_movies):
        path = os.path.join(merged_dir, f'{mid}.json')
        if not os.path.exists(path):
            continue
        feats = json.load(open(path))
        matrix[i] = [float(feats.get(name, 0.0)) for name in FEATURE_ORDER]
        covered.append(i)
    pad = np.zeros((1, n_dims), dtype=np.float32)
    return torch.from_numpy(np.vstack([matrix, pad])), covered


# ── Similarity sanity check ───────────────────────────────────────────────────

def similarity_check(tensor: torch.Tensor, top_movies: list, covered: list) -> None:
    """Cosine similarity among extracted movies; print each one's nearest neighbours."""
    if len(covered) < 2:
        print("   (need ≥2 extracted movies for a similarity check — skipped)")
        return
    mids = [top_movies[i] for i in covered]
    rows = torch.nn.functional.normalize(tensor[covered], dim=1)   # zero rows → zero (no nan)
    sims = rows @ rows.T

    print("\n── Cosine similarity among extracted movies (nearest neighbours) ──")
    for r, mid in enumerate(mids):
        order = torch.argsort(sims[r], descending=True).tolist()
        nbrs  = [(mids[c], float(sims[r, c])) for c in order if c != r][:3]
        label = SANITY_TITLES.get(mid, str(mid))
        nbr_s = ', '.join(f"{SANITY_TITLES.get(m, m)}={s:.2f}" for m, s in nbrs)
        print(f"   {label:<20} → {nbr_s}")


# ── Orchestrator ──────────────────────────────────────────────────────────────

def run(tag: str) -> None:
    sfx, corpus = corpus_suffix()
    print(f"Corpus: {corpus}  (suffix {sfx!r})  |  model tag: {tag}")

    top_movies = load_top_movies(sfx)
    tensor, covered = build_tensor(tag, top_movies)
    print(f"Tensor: {tuple(tensor.shape)}  "
          f"({len(covered)}/{len(top_movies)} movies extracted, rest zero-padded)")

    out_path   = os.path.join(DATA_DIR, f'llm_features_{tag}_{FEATURES_VERSION}{sfx}.pt')
    names_path = os.path.join(DATA_DIR, f'llm_feature_names_{FEATURES_VERSION}.json')
    torch.save(tensor, out_path)
    with open(names_path, 'w') as f:
        json.dump(FEATURE_ORDER, f, indent=2)
    print(f"✓ Saved {os.path.relpath(out_path, REPO_ROOT)}  +  "
          f"{os.path.relpath(names_path, REPO_ROOT)}")

    similarity_check(tensor, top_movies, covered)


if __name__ == '__main__':
    tag = next((a for a in sys.argv[1:] if not a.startswith('-')), DEFAULT_TAG)
    run(tag)
