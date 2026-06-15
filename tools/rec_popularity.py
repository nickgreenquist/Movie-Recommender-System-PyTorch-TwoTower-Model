"""
Recommendation-popularity analysis — does a checkpoint over-serve the popular head?

`main.py eval` measures where the *held-out target* lands (rank/MRR), tiered by the
target's popularity. It does NOT measure how popular the model's *own* top-K
recommendations are. That second quantity is the direct evidence for the Menon
popularity correction: an α=0 model ranks on raw dot products that still encode the
catalog's popularity prior, so its top-K skews to the head; the α=0.5 model was
trained with `+α·log(1+count_i)` added to each logit, learned to deflate popular
items, and at inference (correction dropped) surfaces a flatter, more on-taste list.

For each held-out val user (rollback protocol, identical to src/offline_eval.py),
we build the user embedding, score every corpus item, drop items already in the
user's context history, take the top-K the model would actually serve, and look up
each recommended item's raw ratings.csv count (`_corpus_raw_rating_counts` — the
same popularity basis the correction is applied against). Aggregated over all
contexts:

  mean / median rec popularity (raw rating count)   ↓  less head-heavy
  median log10(1 + count)                            ↓  robust central tendency
  HEAD slot share   (recs with > 1,000 ratings)      ↓  the Phase-1 head
  TAIL slot share   (recs with ≤ 1,000 ratings)       ↑  the long tail — the goal
  catalog coverage  (# distinct items ever served)    ↑  aggregate diversity
  Gini of the rec-impression distribution             ↓  more equal exposure

Both twins are scored on the IDENTICAL contexts (same seed), so every delta is
attributable to α alone.

Usage:
    python tools/rec_popularity.py <checkpoint.pth>
Env:
    REC_POP_N_USERS   val users to sample (default 3,000)
    REC_POP_TOPK      recommendations per context (default 20)
Writes tools/results/rec_popularity_<stem>.json and prints a summary block.
"""
import json
import os
import random
import sys

# Running `python tools/rec_popularity.py` puts tools/ on sys.path, not the
# repo root — add the root so `src` imports resolve.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.dataset import (build_rollback_dataset, get_val_users,
                         pad_history_batch, pad_history_ratings_batch,
                         MAX_ROLLBACK_EXAMPLES_PER_USER)
from src.evaluate import _setup
from src.offline_eval import _corpus_raw_rating_counts, PHASE1_RATING_THRESHOLD

SCORE_BATCH_SIZE = 512


# ── Distributional helpers ────────────────────────────────────────────────────

def _gini(x: np.ndarray) -> float:
    """Gini coefficient of a non-negative vector (0 = perfectly equal exposure)."""
    x = np.sort(x.astype(np.float64))
    n = x.size
    total = x.sum()
    if n == 0 or total == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2.0 * np.sum(idx * x) - (n + 1) * total) / (n * total))


# ── Core ──────────────────────────────────────────────────────────────────────

def analyze(checkpoint_path: str, data_dir: str = 'data',
            n_users: int = 3_000, top_k: int = 20, seed: int = 42) -> dict:
    # Load model + features + item-embedding matrix (all_ids in fs.top_movies order).
    result = _setup(data_dir, checkpoint_path)
    if result[0] is None:
        raise SystemExit("No checkpoint / features.")
    model, fs, _, all_ids, all_embs = result[0], result[1], result[2], result[3], result[4]
    checkpoint_path = result[-1]
    device   = next(model.parameters()).device
    all_embs = all_embs.to(device)
    n_items  = len(all_ids)

    # Raw ratings.csv counts, aligned to all_ids (both are fs.top_movies order).
    counts = _corpus_raw_rating_counts(fs, data_dir)
    if counts is None:
        raise SystemExit("No popularity counts (ratings.csv / cache absent).")
    counts = np.asarray(counts, dtype=np.int64)
    assert counts.shape[0] == n_items, (counts.shape, n_items)
    n_head_movies = int((counts > PHASE1_RATING_THRESHOLD).sum())

    # Build the identical held-out contexts both twins are scored on.
    print(f"Loading val users ...")
    val_users, raw_df = get_val_users(fs, data_dir)
    rng = random.Random(seed)
    rng.shuffle(val_users)
    eval_users = val_users[:n_users]
    print(f"Building rollback contexts for {len(eval_users):,} val users "
          f"(≤{MAX_ROLLBACK_EXAMPLES_PER_USER}/user) ...")
    (X_genre, X_history, X_history_ratings, timestamp, _, _target) = \
        build_rollback_dataset(eval_users, fs, raw_df,
                               MAX_ROLLBACK_EXAMPLES_PER_USER, seed=seed + 1)
    n_ctx = int(X_genre.shape[0])

    # Pre-pad histories once, then slice per batch (mirrors offline_eval).
    hist_idx_padded = pad_history_batch(X_history, model.pad_idx).to(device)
    hist_wts_padded = pad_history_ratings_batch(X_history_ratings).to(device)
    X_genre   = X_genre.to(device)
    timestamp = timestamp.to(device)
    pad       = torch.tensor(model.pad_idx, device=device)
    neg_inf   = torch.finfo(all_embs.dtype).min

    rec_counts   = np.empty((n_ctx, top_k), dtype=np.int64)   # popularity of each rec slot
    impressions  = np.zeros(n_items, dtype=np.int64)          # how often each item is served

    print(f"Scoring {n_ctx:,} contexts → top-{top_k} recs (seen items masked) ...")
    model.eval()
    with torch.no_grad():
        for s in range(0, n_ctx, SCORE_BATCH_SIZE):
            e = min(s + SCORE_BATCH_SIZE, n_ctx)
            hist_idx_t   = hist_idx_padded[s:e]
            hist_wts_t   = hist_wts_padded[s:e]
            hist_liked   = torch.where(hist_wts_t > 0, hist_idx_t, pad)
            hist_dislike = torch.where(hist_wts_t < 0, hist_idx_t, pad)
            user_embs = model.user_embedding(X_genre[s:e], hist_idx_t,
                                             hist_liked, hist_dislike,
                                             hist_wts_t, timestamp[s:e])
            scores = user_embs @ all_embs.T                  # (B, n_items), raw cosine

            # Mask the user's own context history so we score true recommendations.
            # Append a throwaway column for pad_idx (== n_items, out of catalog range);
            # scatter -inf into seen positions, then drop the pad column.
            B   = e - s
            ext = torch.cat([scores, scores.new_full((B, 1), neg_inf)], dim=1)
            ext.scatter_(1, hist_idx_t, neg_inf)
            scores = ext[:, :n_items]

            topk = scores.topk(top_k, dim=1).indices.cpu().numpy()   # corpus indices
            rec_counts[s:e] = counts[topk]
            np.add.at(impressions, topk.reshape(-1), 1)

    # ── Aggregate ─────────────────────────────────────────────────────────────
    flat       = rec_counts.reshape(-1)
    total_slot = int(flat.shape[0])
    head_slots = int((flat > PHASE1_RATING_THRESHOLD).sum())
    served     = int((impressions > 0).sum())

    # True head = top 10% of items by popularity rank (more discriminating than the
    # 1,000-rating cut, which is ~half the catalog). Computed exactly from the
    # per-item impression vector.
    order = np.argsort(counts, kind='stable')
    rank  = np.empty(n_items, dtype=np.int64)
    rank[order] = np.arange(n_items)
    head_decile_mask = rank >= int(np.ceil(0.9 * n_items))
    top_decile_share = float(impressions[head_decile_mask].sum() / total_slot)

    stats = {
        'checkpoint':           os.path.basename(checkpoint_path),
        'alpha':                None,   # filled by caller from sidecar; informational
        'n_users':              len(eval_users),
        'n_contexts':           n_ctx,
        'top_k':                top_k,
        'n_items':              n_items,
        'n_head_movies':        n_head_movies,
        'n_tail_movies':        n_items - n_head_movies,
        'mean_rec_popularity':  float(flat.mean()),
        'median_rec_popularity': float(np.median(flat)),
        'p90_rec_popularity':   float(np.percentile(flat, 90)),
        'median_log10_pop':     float(np.median(np.log10(1.0 + flat))),
        'head_slot_share':      head_slots / total_slot,
        'tail_slot_share':      1.0 - head_slots / total_slot,
        'top_decile_share':     top_decile_share,
        'catalog_coverage':     served / n_items,
        'n_distinct_served':    served,
        'impression_gini':      _gini(impressions),
    }

    # ── Report ────────────────────────────────────────────────────────────────
    print(f"\n{'═' * 68}")
    print(f"Recommendation popularity — {stats['checkpoint']}")
    print(f"{'═' * 68}")
    print(f"  contexts={n_ctx:,}  top_k={top_k}  catalog={n_items:,} "
          f"({n_head_movies:,} head / {n_items - n_head_movies:,} tail movies)")
    print(f"  mean rec popularity (raw ratings) : {stats['mean_rec_popularity']:>12,.0f}")
    print(f"  median rec popularity             : {stats['median_rec_popularity']:>12,.0f}")
    print(f"  p90 rec popularity                : {stats['p90_rec_popularity']:>12,.0f}")
    print(f"  median log10(1+pop)               : {stats['median_log10_pop']:>12.3f}")
    print(f"  HEAD slot share (> {PHASE1_RATING_THRESHOLD:,})         : {stats['head_slot_share']:>12.1%}")
    print(f"  TAIL slot share (≤ {PHASE1_RATING_THRESHOLD:,})         : {stats['tail_slot_share']:>12.1%}")
    print(f"  top-decile (true head) share      : {stats['top_decile_share']:>12.1%}")
    print(f"  catalog coverage (distinct served): {stats['catalog_coverage']:>12.1%}  "
          f"({served:,}/{n_items:,})")
    print(f"  impression Gini                   : {stats['impression_gini']:>12.3f}")

    out_dir = 'tools/results'
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(checkpoint_path))[0]
    out_path = os.path.join(out_dir, f'rec_popularity_{stem}.json')
    with open(out_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\n  → saved to {out_path}")
    return stats


if __name__ == '__main__':
    ckpt = sys.argv[1] if len(sys.argv) > 1 else None
    if ckpt is None:
        print("Usage: python tools/rec_popularity.py <checkpoint.pth>")
        sys.exit(1)
    n_users = int(os.environ.get('REC_POP_N_USERS', 3_000))
    top_k   = int(os.environ.get('REC_POP_TOPK', 20))
    analyze(ckpt, n_users=n_users, top_k=top_k)
