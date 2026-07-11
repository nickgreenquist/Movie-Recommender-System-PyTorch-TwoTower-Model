"""
Stage 0 — Phase 1 Corpus Filtering  (LLM-vs-genome ablation)

Filters the full MovieLens corpus down to the most-rated ~4-5k movies for the
Phase 1 pilot of the LLM-vs-genome content-feature ablation, and writes the
resulting movieId list to data/llm_experiment_movies_phase1.json. Every Phase 1
stage (scrape -> extract -> train all three models) operates on this reduced
list — it let the approach be validated cheaply before the full-corpus
extraction (both since completed). See docs/plans/llm_vs_genome_ablation_plan.md.

This is the ONLY place the phase1 *filter* lives. src/corpus.py provides the
artifact-naming plumbing (the '_phase1' filename suffix) but deliberately does
NOT pick which movies are in the reduced corpus — that decision is here. Scope:
produce the list only; scraping and extraction are later stages.

Counting matches src/preprocess.py:build_corpus() exactly — raw ratings.csv
counts (NOT the user-filtered counts in top_movies_by_popularity.csv, which run
~5x smaller), strict '>' threshold — so the phase1 threshold is directly
comparable to the full corpus's MIN_RATINGS_PER_MOVIE = 200. The chosen movies
are intersected with the full corpus (base_movies.parquet), guaranteeing the
phase1 set is a clean subset for which every downstream artifact (genome scores,
tag vectors, year) already exists.

Usage (standalone — not part of the main.py pipeline CLI):
    python llm_features/filter_corpus.py
"""
import json
import os

import pandas as pd


# ── Constants ────────────────────────────────────────────────────────────────

# Higher than the full corpus's MIN_RATINGS_PER_MOVIE = 200 (src/preprocess.py).
# At '> 1000' raw ratings the full 9,375-movie corpus reduces to 4,461 movies —
# squarely in the ~4-5k Phase 1 target, a clean 5x the base threshold, and
# matching the plan's ~4,500-movie / ~27k-call cost model. Neighbours for
# tuning (within-corpus raw counts): >800 -> 5,025, >900 -> 4,705, >1200 -> 4,047.
MIN_RATINGS_PER_MOVIE_PHASE1 = 1_000

REPO_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_RATINGS = os.path.join(REPO_ROOT, 'data', 'ml-32m', 'ratings.csv')
FULL_MOVIES = os.path.join(REPO_ROOT, 'data', 'base_movies.parquet')
OUT_PATH    = os.path.join(REPO_ROOT, 'data', 'llm_experiment_movies_phase1.json')


# ── Counting ─────────────────────────────────────────────────────────────────

def corpus_raw_counts() -> pd.Series:
    """
    Raw ratings.csv count per full-corpus movie, indexed by movieId in full
    corpus order (base_movies.parquet). Same counting as build_corpus(): count
    every raw rating row, before the 20-500-ratings user filter is applied.
    """
    full_ids = pd.read_parquet(FULL_MOVIES)['movieId'].astype(int).tolist()
    print(f"Full corpus: {len(full_ids)} movies (data/base_movies.parquet)")

    print("Counting raw ratings per movie (data/ml-32m/ratings.csv) ...")
    raw = pd.read_csv(RAW_RATINGS, usecols=['movieId'])['movieId'].value_counts()
    return raw.reindex(full_ids).fillna(0).astype(int)


# ── Orchestrator ─────────────────────────────────────────────────────────────

def run() -> None:
    print(f"Phase 1 corpus filter — threshold: > {MIN_RATINGS_PER_MOVIE_PHASE1} "
          f"raw ratings (full corpus uses > 200)\n")
    counts = corpus_raw_counts()

    print("\nThreshold sensitivity (raw count > T  →  # movies kept):")
    for T in (800, 900, 1_000, 1_100, 1_200):
        marker = '  ← chosen' if T == MIN_RATINGS_PER_MOVIE_PHASE1 else ''
        print(f"  > {T:5d}  ->  {int((counts > T).sum()):5d}{marker}")

    movie_ids = counts[counts > MIN_RATINGS_PER_MOVIE_PHASE1].index.tolist()
    if not 4_000 <= len(movie_ids) <= 5_000:
        print(f"\n⚠  {len(movie_ids)} movies is outside the ~4-5k target — "
              f"tune MIN_RATINGS_PER_MOVIE_PHASE1 and re-run.")

    payload = {
        'corpus':                'phase1',
        'threshold_min_ratings': MIN_RATINGS_PER_MOVIE_PHASE1,
        'full_corpus_threshold': 200,    # src/preprocess.py MIN_RATINGS_PER_MOVIE
        'n_movies':              len(movie_ids),
        'movie_ids':             movie_ids,
    }
    with open(OUT_PATH, 'w') as f:
        json.dump(payload, f)

    print(f"\n✓ Wrote {len(movie_ids)} movieIds  →  {OUT_PATH}")


if __name__ == '__main__':
    run()
