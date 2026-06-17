"""
Offline retrieval evaluation — Recall@K, NDCG@K, Hit Rate@K, MRR.

Rollback protocol: for each val user (held out at user level), sample up to
MAX_ROLLBACK_EXAMPLES_PER_USER chronological positions. At each position j,
context = history[0..j-1], target = history[j]. All positions are
valid since val users were never seen in training.

Results are written to eval_results/<checkpoint_stem>.txt

Long-tail split: in addition to the whole-corpus block, metrics are reported
restricted to popularity tiers of the *target* movie (raw ratings.csv counts —
the same basis that defines the corpus and the Phase 1 threshold). Two cuts:
  • HEAD / TAIL at the Phase 1 threshold (> 1,000 raw ratings) — HEAD is exactly
    the Phase 1 corpus, TAIL is exactly the long tail Phase 1 excluded.
  • Population quartiles (Q1 rarest … Q4 most popular), equal movie counts.
Content features are expected to earn their lift on the tail, where collaborative
signal is sparse — see docs/plans/llm_vs_genome_ablation_plan.md Stage 6.

Usage:
    python main.py eval
    python main.py eval <checkpoint_path>
"""
import math
import os
import random

import numpy as np
import torch

from src.corpus import corpus_suffix
from src.dataset import (FeatureStore, pad_history_batch, pad_history_ratings_batch,
                          build_rollback_dataset, get_val_users,
                          MAX_ROLLBACK_EXAMPLES_PER_USER)
from src.evaluate import build_movie_embeddings
from src.model import MovieRecommender

EVAL_BATCH_SIZE = 512

# Phase 1 corpus filter threshold (llm_features/filter_corpus.py). Movies with
# > this many raw ratings are the Phase 1 ("head") corpus; the rest are the long
# tail Phase 1 excluded. Kept in sync with MIN_RATINGS_PER_MOVIE_PHASE1.
PHASE1_RATING_THRESHOLD = 1_000


def _build_emb_matrix(model, fs):
    movie_embeddings = build_movie_embeddings(model, fs)
    device   = next(model.parameters()).device
    all_ids  = list(movie_embeddings.keys())
    all_embs = torch.cat(
        [movie_embeddings[mid]['MOVIE_EMBEDDING_COMBINED'] for mid in all_ids], dim=0
    ).to(device)
    return all_ids, all_embs


def _corpus_raw_rating_counts(fs, data_dir):
    """
    Raw ratings.csv count per corpus movie, aligned to corpus index (= fs.top_movies
    order). Same counting as src/preprocess.py:build_corpus() and
    llm_features/filter_corpus.py — every raw rating row, before the user filter.
    Cached to data/corpus_raw_rating_counts<sfx>.npy (one-time ~20s, then instant).
    Returns None if ratings.csv is absent and no cache exists (tier split skipped).
    """
    sfx   = corpus_suffix()
    cache = os.path.join(data_dir, f'corpus_raw_rating_counts{sfx}.npy')
    if os.path.exists(cache):
        return np.load(cache)

    raw_path = os.path.join(data_dir, 'ml-32m', 'ratings.csv')
    if not os.path.exists(raw_path):
        print(f"  ⚠  {raw_path} not found and no cached counts — skipping long-tail split.")
        return None

    import pandas as pd
    print(f"  Computing raw rating counts per movie from {raw_path} (one-time, cached) ...")
    vc     = pd.read_csv(raw_path, usecols=['movieId'])['movieId'].value_counts()
    counts = np.array([int(vc.get(int(mid), 0)) for mid in fs.top_movies], dtype=np.int64)
    np.save(cache, counts)
    return counts


def _build_tiers(counts):
    """
    Given per-corpus-index raw rating counts, return an ordered list of
    (label, corpus_index_mask) popularity tiers: HEAD/TAIL at the Phase 1
    threshold, then equal-movie-count population quartiles (Q1 rarest … Q4 most
    popular). Masks are boolean arrays over corpus index.
    """
    n_movies = len(counts)
    tiers = []

    head = counts > PHASE1_RATING_THRESHOLD
    tail = ~head
    tiers.append((f"HEAD (> {PHASE1_RATING_THRESHOLD:,} raw ratings) — Phase 1 corpus", head))
    tiers.append((f"TAIL (≤ {PHASE1_RATING_THRESHOLD:,}) — Phase 1-excluded long tail", tail))

    # Rank-based quartiles: equal number of movies per tier, Q1 = rarest.
    order            = np.argsort(counts, kind='stable')   # ascending by count
    pop_rank         = np.empty(n_movies, dtype=np.int64)
    pop_rank[order]  = np.arange(n_movies)
    quartile_of_idx  = pop_rank * 4 // n_movies            # 0..3
    for q in range(4):
        mask = quartile_of_idx == q
        lo, hi = int(counts[mask].min()), int(counts[mask].max())
        tiers.append((f"Q{q + 1} ({'rarest' if q == 0 else 'most popular' if q == 3 else 'mid'}; "
                      f"{lo:,}–{hi:,} ratings)", mask))
    return tiers


def _metrics_from_ranks(ranks, ks):
    """Vectorized retrieval metrics from a 1-D array of per-example target ranks."""
    n = int(ranks.shape[0])
    if n == 0:
        return None
    inv_log = 1.0 / np.log2(ranks + 1.0)
    out = {'n': n, 'mrr': float((1.0 / ranks).mean())}
    for k in ks:
        hit          = ranks <= k
        out[(k, 'hit')]  = float(hit.mean())
        out[(k, 'ndcg')] = float((hit * inv_log).mean())
    return out


def _format_block(metrics, ks, title, n_corpus):
    """Render one metrics block (whole-corpus or a single tier) as text lines."""
    lines = [f"\n{title}  (n={metrics['n']:,} rollbacks)"]
    header = f"{'K':>6}  {'Recall@K':>10}  {'Hit Rate@K':>11}  {'NDCG@K':>8}"
    lines.append(header)
    lines.append("─" * len(header))
    for k in ks:
        hit = metrics[(k, 'hit')]
        lines.append(f"{k:>6}  {hit:>10.4f}  {hit:>11.4f}  {metrics[(k, 'ndcg')]:>8.4f}")
    lines.append("─" * len(header))
    lines.append(f"MRR: {metrics['mrr']:.4f}")
    return lines


def run_offline_eval(model: MovieRecommender, fs: FeatureStore,
                     checkpoint_path: str = '',
                     n_users: int | None = None,   # None → all val users (full eval)
                     ks: tuple = (1, 5, 10, 20, 50, 100, 150, 200, 250),
                     seed: int = 42,
                     data_dir: str = 'data') -> None:
    _run_rollback_eval(model, fs, checkpoint_path, data_dir, n_users, ks, seed)


def _run_rollback_eval(model, fs, checkpoint_path, data_dir, n_users, ks, seed):
    model.eval()

    print("Building movie embeddings ...")
    all_ids, all_embs = _build_emb_matrix(model, fs)

    print("Loading val users ...")
    val_users, raw_df = get_val_users(fs, data_dir)
    rng = random.Random(seed)
    rng.shuffle(val_users)
    eval_users = val_users[:n_users]

    print(f"Building rollback examples for {len(eval_users):,} val users (chronological, ≤{MAX_ROLLBACK_EXAMPLES_PER_USER}/user) ...")
    (X_genre, X_history, X_history_ratings, timestamp, _, target_movieId) = \
        build_rollback_dataset(eval_users, fs, raw_df,
                               MAX_ROLLBACK_EXAMPLES_PER_USER, seed=seed + 1)

    n_examples = int(target_movieId.shape[0])

    # Pre-pad histories once so the scoring loop just slices pre-allocated tensors
    device = next(model.parameters()).device
    hist_idx_padded = pad_history_batch(X_history, model.pad_idx).to(device)
    hist_wts_padded = pad_history_ratings_batch(X_history_ratings).to(device)
    X_genre   = X_genre.to(device)
    timestamp = timestamp.to(device)

    ranks = np.empty(n_examples, dtype=np.int64)   # per-example target rank
    pad   = torch.tensor(model.pad_idx, device=device)

    with torch.no_grad():
        for s in range(0, n_examples, EVAL_BATCH_SIZE):
            e = min(s + EVAL_BATCH_SIZE, n_examples)

            hist_idx_t    = hist_idx_padded[s:e]
            hist_wts_t    = hist_wts_padded[s:e]
            hist_liked_t  = torch.where(hist_wts_t > 0, hist_idx_t, pad)
            hist_disliked_t = torch.where(hist_wts_t < 0, hist_idx_t, pad)
            user_embs  = model.user_embedding(X_genre[s:e], hist_idx_t,
                                              hist_liked_t, hist_disliked_t,
                                              hist_wts_t, timestamp[s:e])
            scores     = user_embs @ all_embs.T  # (B, n_items)

            for i in range(e - s):
                t_pos        = int(target_movieId[s + i].item())
                target_score = scores[i, t_pos]
                ranks[s + i] = int((scores[i] > target_score).sum().item()) + 1

    if n_examples == 0:
        print("No rollback positions evaluated — check that feature parquets are loaded.")
        return

    # ── Whole-corpus block ────────────────────────────────────────────────────
    max_k    = max(ks)
    n_corpus = len(all_ids)
    label    = f"rollback (≤{MAX_ROLLBACK_EXAMPLES_PER_USER}/user)"
    head = [
        f"\n── Offline Evaluation [{label}]  (n={n_examples:,} rollbacks, "
        f"{checkpoint_path or 'latest'}) " + "─" * 10,
        f"Corpus: {n_corpus:,} movies  |  Random Hit Rate@{max_k} baseline: {max_k / n_corpus:.3%}",
    ]
    out_lines = head + _format_block(_metrics_from_ranks(ranks, ks), ks, "Whole corpus", n_corpus)

    # ── Long-tail split ───────────────────────────────────────────────────────
    counts = _corpus_raw_rating_counts(fs, data_dir)
    if counts is not None:
        target_idx    = target_movieId.cpu().numpy()
        example_count = counts[target_idx]
        out_lines.append("\n" + "═" * 60)
        out_lines.append("Long-tail split — metrics restricted by target-movie popularity")
        out_lines.append("(raw ratings.csv counts; tiers are properties of the corpus)")
        out_lines.append("═" * 60)
        for tier_label, corpus_mask in _build_tiers(counts):
            n_movies_tier = int(corpus_mask.sum())
            ex_mask       = corpus_mask[target_idx]
            tier_metrics  = _metrics_from_ranks(ranks[ex_mask], ks)
            title         = f"── {tier_label}  [{n_movies_tier:,} movies] " + "─" * 6
            if tier_metrics is None:
                out_lines.append(f"\n{title}\n  (no rollback examples target this tier)")
                continue
            out_lines += _format_block(tier_metrics, ks, title, n_corpus)

    output = "\n".join(out_lines)
    print(output)

    out_dir  = 'eval_results'
    os.makedirs(out_dir, exist_ok=True)
    stem     = os.path.splitext(os.path.basename(checkpoint_path))[0] if checkpoint_path else 'latest'
    out_path = os.path.join(out_dir, f'{stem}.txt')
    with open(out_path, 'w') as f:
        f.write(output + "\n")
    print(f"\n  → saved to {out_path}")
