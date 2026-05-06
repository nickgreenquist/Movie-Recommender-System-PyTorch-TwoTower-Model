"""
Stage 0 — Ranker candidate precompute.

For each rollback example (user, position N), build the v2-softmax-style user
context from history[0:N-1], score against all corpus items via the PROD CG
model (v2 softmax), and save the top-99 negatives + label as a single row.

Output:
    data/ranker_candidates_train.parquet
    data/ranker_candidates_val.parquet
    data/ranker_movie_stats.parquet
"""
import glob
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset import (MAX_HISTORY_LEN, TIMESTAMP_NUM_BINS, FeatureStore,
                         load_features)
from src.train import build_model, get_device, get_config


MAX_ROLLBACK_PER_USER = 20      # match MAX_V2_SOFTMAX_EXAMPLES_PER_USER
TOP_K_CANDIDATES      = 250     # 1 label + 249 negatives per row
SCORING_BATCH_SIZE    = 512
SPLIT_SEED            = 42
PCT_TRAIN             = 0.9


# ── CG model loader ──────────────────────────────────────────────────────────

def _resolve_checkpoint(checkpoint_arg: str | None) -> str:
    """Default to current PROD softmax checkpoint; allow override."""
    if checkpoint_arg:
        return checkpoint_arg
    matches = sorted(glob.glob('saved_models/PROD_best_softmax_v2_popularity_alpha_*.pth'))
    if not matches:
        raise FileNotFoundError(
            "No PROD softmax checkpoint found in saved_models/. "
            "Pass an explicit path: python ranker/main.py precompute <checkpoint>"
        )
    return matches[-1]


def load_cg(checkpoint_path: str, fs: FeatureStore, device: torch.device):
    config = get_config()
    model = build_model(config, fs)
    state = torch.load(checkpoint_path, weights_only=True, map_location='cpu')
    model.load_state_dict(state)
    model.eval().to(device)
    return model


# ── Rollback builder (tracks user_id and rollback_n; copies v2 softmax logic) ─

def _build_rollback_arrays(users: list, fs: FeatureStore, raw_df,
                           max_per_user: int, seed: int):
    """
    Mirrors build_v2_softmax_dataset (sort_by_ts, max_per_user, right-aligned padding)
    but additionally tracks (user_id, rollback_n) per example and returns user-level
    avg_rating / total_rating_count for each example.

    Returns dict of pre-allocated tensors/arrays:
        user_id, rollback_n, label_corpus_idx, user_avg_rating, user_rating_count,
        X_genre, X_history, X_hist_ratings, timestamp
    """
    rng       = random.Random(seed)
    users_set = set(users)
    max_hist  = MAX_HISTORY_LEN
    n_genres  = len(fs.genres_ordered)
    pad_idx   = len(fs.top_movies)

    movie_genre_idxs = {
        mid: [fs.genre_to_i[g] for g in fs.movieId_to_genres.get(mid, []) if g in fs.genre_to_i]
        for mid in fs.top_movies
    }

    df = raw_df[raw_df['userId'].isin(users_set)]
    print(f"  {len(df):,} interactions, {df['userId'].nunique():,} users")

    print("  Counting pass ...")
    valid_groups = []
    n_examples = 0
    for uid, group in df.groupby('userId'):
        rows = list(zip(group['movieId'].tolist(), group['rating'].tolist(),
                        group['timestamp'].tolist()))
        rows.sort(key=lambda x: x[2])
        n = len(rows)
        if n >= 2:
            _, ratings_col, _ = zip(*rows)
            avg_r = float(np.mean(ratings_col))
            n_liked = sum(1 for i in range(1, n) if ratings_col[i] > avg_r)
            n_examples += min(max_per_user, n_liked)
            valid_groups.append((int(uid), rows))

    print(f"  Pre-allocating buffers for {n_examples:,} examples ...")
    user_id           = np.zeros(n_examples, dtype=np.int64)
    rollback_n        = np.zeros(n_examples, dtype=np.int32)
    label_corpus_idx  = np.zeros(n_examples, dtype=np.int64)
    user_avg_rating   = np.zeros(n_examples, dtype=np.float32)
    user_rating_count = np.zeros(n_examples, dtype=np.int32)
    X_genre           = np.zeros((n_examples, 2 * n_genres), dtype=np.float32)
    X_history         = np.full((n_examples, max_hist), pad_idx, dtype=np.int32)
    X_hist_ratings    = np.zeros((n_examples, max_hist), dtype=np.float32)
    timestamps_raw    = np.zeros(n_examples, dtype=np.float64)

    curr = 0
    for uid, rows in tqdm(valid_groups, desc="Building rollback examples"):
        movies, ratings, ts_vals = zip(*rows)
        avg_rat        = float(np.mean(ratings))
        n_user_ratings = len(movies)

        # Only sample positions where the target was rated above the user's mean.
        # Context (watch history) remains unfiltered — disliked watches still inform
        # the user embedding, but we don't train the ranker to recommend things
        # the user rated below their own average.
        eligible    = [i for i in range(1, n_user_ratings) if ratings[i] > avg_rat]
        k           = min(max_per_user, len(eligible))
        sampled     = sorted(rng.sample(eligible, k))
        sampled_set = set(sampled)

        running_count = np.zeros(n_genres, dtype=np.float32)
        running_sum   = np.zeros(n_genres, dtype=np.float32)
        ctx_ids       = []
        ctx_rats      = []

        for pos, (mid, rat, ts) in enumerate(zip(movies, ratings, ts_vals)):
            mid   = int(mid)
            d_rat = float(rat) - avg_rat
            t_idx = fs.item_emb_movieId_to_i[mid]

            if pos in sampled_set:
                total = running_count.sum()
                if total > 0:
                    mask = running_count > 0
                    X_genre[curr, :n_genres][mask] = running_sum[mask] / running_count[mask]
                    X_genre[curr, n_genres:]       = running_count / total

                take = min(len(ctx_ids), max_hist)
                if take > 0:
                    X_history[curr, max_hist - take:]      = ctx_ids[-take:]
                    X_hist_ratings[curr, max_hist - take:] = ctx_rats[-take:]

                user_id[curr]           = uid
                rollback_n[curr]        = pos
                label_corpus_idx[curr]  = t_idx
                user_avg_rating[curr]   = avg_rat
                user_rating_count[curr] = n_user_ratings
                timestamps_raw[curr]    = ts
                curr += 1

            ctx_ids.append(t_idx)
            ctx_rats.append(d_rat)
            for g_idx in movie_genre_idxs.get(mid, []):
                running_count[g_idx] += 1
                running_sum[g_idx]   += d_rat

    timestamp_t = torch.bucketize(
        torch.from_numpy(timestamps_raw[:curr]).float(),
        fs.timestamp_bins.float(), right=False,
    ).clamp(max=TIMESTAMP_NUM_BINS - 1).numpy().astype(np.int32)

    return {
        'user_id':           user_id[:curr],
        'rollback_n':        rollback_n[:curr],
        'label_corpus_idx':  label_corpus_idx[:curr],
        'user_avg_rating':   user_avg_rating[:curr],
        'user_rating_count': user_rating_count[:curr],
        'X_genre':           X_genre[:curr],
        'X_history':         X_history[:curr],
        'X_hist_ratings':    X_hist_ratings[:curr],
        'timestamp_bin':     timestamp_t,
    }


# ── CG scoring (batched) ─────────────────────────────────────────────────────

@torch.no_grad()
def _score_candidates(model, V_all, arrays, device, batch_size, top_k):
    """
    Run CG scoring in batches. For each rollback example, returns:
      - neg_corpus_idxs:  (n, top_k - 1) — top-99 negatives by CG score (label masked out)
      - cg_label_rank:    (n,)           — label's rank within the (label + 99 negs) group
                                            of 100 candidates, 1-indexed, capped at top_k.

    ─── Why cg_label_rank ───
    The 99 negatives are CG's top-99 corpus items EXCLUDING the label. The label could
    be anywhere in CG's full corpus ranking; we want to know the label's rank within
    just the 100-candidate group to compute the "what would CG do?" baseline NDCG@10/MRR.

    Three cases for the label's true rank in the full corpus (call it r_full):
      1. r_full == 1: label scores higher than ALL 99 negs (negs are corpus ranks 2..100).
                      → label rank within group = 1.
      2. 1 < r_full <= top_k: label scores higher than (top_k - r_full) negs and lower
                              than (r_full - 1) negs (negs are corpus ranks 1..r_full-1
                              and r_full+1..top_k).
                              → label rank within group = r_full.
      3. r_full > top_k: label scores lower than ALL 99 negs (negs are corpus ranks 1..99).
                         → label rank within group = top_k (capped, since we only have
                            99 negatives that beat it; we can't measure beyond that).

    Equivalently:
        cg_label_rank = min(r_full, top_k) = (count of negs with score > label_score) + 1

    We compute this BEFORE masking the label position to -inf, so the label's score is
    intact when comparing against the topk-1 negative scores.

    Why this matters for evaluation:
      - Without cg_label_rank: baseline assumes label is always at rank 100 (no info).
        That gives MRR = 1/100 = 0.01 and NDCG@10 = 0 for every example — flat baseline.
      - With cg_label_rank: baseline reflects CG's actual quality per example. If CG
        already ranks the label at #1 for many examples, the baseline NDCG@10 is high
        and the ranker has to beat THAT, not a strawman.
    """
    n = arrays['X_genre'].shape[0]
    n_neg = top_k - 1

    X_genre        = torch.from_numpy(arrays['X_genre'])
    X_history      = torch.from_numpy(arrays['X_history']).long()
    X_hist_ratings = torch.from_numpy(arrays['X_hist_ratings'])
    timestamp      = torch.from_numpy(arrays['timestamp_bin']).long()
    label_idx      = torch.from_numpy(arrays['label_corpus_idx']).long()

    neg_corpus_idxs     = np.zeros((n, n_neg), dtype=np.int32)
    cg_label_rank       = np.zeros(n, dtype=np.int32)
    cg_label_score      = np.zeros(n, dtype=np.float32)
    cg_neg_scores       = np.zeros((n, n_neg), dtype=np.float32)
    genome_cosine_label = np.zeros(n, dtype=np.float32)
    genome_cosine_negs  = np.zeros((n, n_neg), dtype=np.float32)

    pad_idx_v = model.pad_idx  # == len(fs.top_movies)

    pad = torch.tensor(pad_idx_v, device=device)

    for s in tqdm(range(0, n, batch_size), desc="Scoring candidates"):
        e = min(s + batch_size, n)
        B = e - s
        hist_idx_b = X_history[s:e].to(device)
        hist_wts_b = X_hist_ratings[s:e].to(device)
        hist_liked_b    = torch.where(hist_wts_b > 0, hist_idx_b, pad)
        hist_disliked_b = torch.where(hist_wts_b < 0, hist_idx_b, pad)
        U = model.user_embedding(
            X_genre[s:e].to(device),
            hist_idx_b,
            hist_liked_b,
            hist_disliked_b,
            hist_wts_b,
            timestamp[s:e].to(device),
        )
        scores = U @ V_all.T  # (B, n_movies)
        # NaN scores (e.g. from a degenerate user embedding) cause topk to return
        # non-unique indices on MPS. Clamp to -inf so they sort last and are excluded.
        scores = scores.nan_to_num(nan=float('-inf'))
        b_idx    = torch.arange(B, device=device)
        label_b  = label_idx[s:e].to(device)
        label_sc = scores[b_idx, label_b]                       # (B,) label scores

        # Store CG label score BEFORE masking.
        cg_label_score[s:e] = label_sc.cpu().numpy()

        # Compute label rank in full corpus, then cap at top_k (see docstring).
        full_rank = (scores > label_sc.unsqueeze(1)).sum(dim=1) + 1   # (B,) 1-indexed
        cg_label_rank[s:e] = full_rank.clamp(max=top_k).cpu().numpy()

        # Mask label and retrieve top negatives with their CG scores.
        scores[b_idx, label_b] = float('-inf')
        top_result           = scores.topk(n_neg, dim=1)
        neg_corpus_idxs[s:e] = top_result.indices.cpu().numpy()
        cg_neg_scores[s:e]   = top_result.values.cpu().numpy()

        # ── Genome cosine similarity (per-batch, freed immediately) ──────────
        # Rating-weighted avg of raw genome scores over watch history → user genome pool.
        hist_ids_b  = X_history[s:e].to(device)           # (B, max_hist)
        hist_rats_b = X_hist_ratings[s:e].to(device)       # (B, max_hist)
        pad_mask_b  = (hist_ids_b != pad_idx_v).float()   # 0 for padding positions
        rat_w_b     = hist_rats_b * pad_mask_b             # (B, max_hist)
        w_sum_b     = rat_w_b.abs().sum(dim=1, keepdim=True).clamp(min=1e-6)
        # Clamp to valid genome buffer range; padding rows (weight=0) don't affect pool.
        safe_ids    = hist_ids_b.clamp(max=pad_idx_v - 1)
        genome_hist = model.genome_context_buffer[safe_ids]         # (B, max_hist, genome_dim)
        genome_pool = (genome_hist * rat_w_b.unsqueeze(-1)).sum(dim=1) / w_sum_b  # (B, genome_dim)
        gp_norm     = F.normalize(genome_pool, p=2, dim=1)

        lbl_genome              = F.normalize(model.genome_context_buffer[label_b], p=2, dim=1)
        genome_cosine_label[s:e] = (gp_norm * lbl_genome).sum(dim=1).cpu().numpy()

        neg_ids_t              = top_result.indices                  # (B, n_neg)
        neg_genome             = F.normalize(model.genome_context_buffer[neg_ids_t], p=2, dim=-1)
        genome_cosine_negs[s:e] = (gp_norm.unsqueeze(1) * neg_genome).sum(dim=-1).cpu().numpy()

    return neg_corpus_idxs, cg_label_rank, cg_label_score, cg_neg_scores, genome_cosine_label, genome_cosine_negs


# ── Movie stats (global per-movie features) ──────────────────────────────────

def compute_movie_stats(fs: FeatureStore, data_dir: str, output_dir: str):
    """
    Per-movie global average rating + log1p(global rating count).
    Computed from raw base_ratings.parquet (corpus-filtered).
    """
    raw_df = pd.read_parquet(os.path.join(data_dir, 'base_ratings.parquet'))
    corpus_set = set(fs.item_emb_movieId_to_i.keys())
    raw_df = raw_df[raw_df['movieId'].isin(corpus_set)]

    grouped = raw_df.groupby('movieId').agg(
        global_avg_rating=('rating', 'mean'),
        n_ratings=('rating', 'count'),
    ).reset_index()
    grouped['global_rating_count'] = np.log1p(grouped['n_ratings']).astype(np.float32)
    grouped['global_avg_rating']   = grouped['global_avg_rating'].astype(np.float32)
    grouped = grouped[['movieId', 'global_avg_rating', 'global_rating_count']]

    # Add a row for any corpus movies missing from base_ratings (shouldn't happen, but defensive)
    have = set(grouped['movieId'])
    missing = [mid for mid in fs.top_movies if mid not in have]
    if missing:
        print(f"  WARN: {len(missing)} corpus movies have no rating rows; filling with zeros")
        pad = pd.DataFrame({
            'movieId': missing,
            'global_avg_rating': np.zeros(len(missing), dtype=np.float32),
            'global_rating_count': np.zeros(len(missing), dtype=np.float32),
        })
        grouped = pd.concat([grouped, pad], ignore_index=True)

    out_path = os.path.join(output_dir, 'ranker_movie_stats.parquet')
    grouped.to_parquet(out_path, index=False)
    print(f"  Movie stats: {len(grouped):,} rows → {out_path}")
    print(f"    avg_rating: mean={grouped['global_avg_rating'].mean():.3f}  "
          f"min={grouped['global_avg_rating'].min():.3f}  "
          f"max={grouped['global_avg_rating'].max():.3f}")
    print(f"    log1p_count: mean={grouped['global_rating_count'].mean():.3f}  "
          f"min={grouped['global_rating_count'].min():.3f}  "
          f"max={grouped['global_rating_count'].max():.3f}")


# ── Verification ─────────────────────────────────────────────────────────────

def _verify_split(arrays, neg_corpus_idxs, cg_label_rank, fs, label, n_movies, top_k):
    """Hard-fail checks before writing parquet."""
    n = arrays['X_genre'].shape[0]
    label_idx = arrays['label_corpus_idx']
    pad_idx   = len(fs.top_movies)

    # 1. label never in negatives
    in_neg = (neg_corpus_idxs == label_idx[:, None]).any(axis=1).sum()
    assert in_neg == 0, f"[{label}] label appears in negatives in {in_neg} rows"

    # 2. negatives all valid corpus indices
    assert neg_corpus_idxs.min() >= 0,         f"[{label}] negative idx < 0"
    assert neg_corpus_idxs.max() < n_movies,   f"[{label}] negative idx >= n_movies"

    # 3. each row has exactly (top_k - 1) unique negatives
    n_neg = top_k - 1
    sample_uniq = np.array([len(np.unique(neg_corpus_idxs[i])) for i in range(min(1000, n))])
    bad_mask = sample_uniq != n_neg
    if bad_mask.any():
        bad_counts = sample_uniq[bad_mask]
        print(f"  [{label}] non-unique negatives in {bad_mask.sum()} / {len(sample_uniq)} sampled rows")
        print(f"    unique count distribution: min={bad_counts.min()}  max={bad_counts.max()}  "
              f"mean={bad_counts.mean():.1f}  most common={np.bincount(bad_counts).argmax()}")
        print(f"    first bad row index: {np.where(bad_mask)[0][0]}")
        print(f"    first bad row values: {neg_corpus_idxs[np.where(bad_mask)[0][0]]}")
    assert not bad_mask.any(), f"[{label}] non-unique negatives in some rows (expected {n_neg})"

    # 4. history never contains label (label is at position N, history is 0..N-1)
    sample_n = min(1000, n)
    sample_idx = np.random.choice(n, sample_n, replace=False)
    hist = arrays['X_history'][sample_idx]
    label_sample = label_idx[sample_idx]
    label_in_hist = ((hist == label_sample[:, None]) & (hist != pad_idx)).any(axis=1).sum()
    assert label_in_hist == 0, f"[{label}] label appears in history for {label_in_hist}/{sample_n} sampled rows"

    # 5. cg_label_rank in [1, top_k]
    assert cg_label_rank.min() >= 1,     f"[{label}] cg_label_rank < 1"
    assert cg_label_rank.max() <= top_k, f"[{label}] cg_label_rank > {top_k}"

    print(f"  [{label}] verification passed: n={n:,}  unique_users={len(np.unique(arrays['user_id'])):,}")
    # Quick CG quality summary — see how often CG ranks the label well unaided.
    cg_mrr      = float((1.0 / cg_label_rank).mean())
    log2_term   = np.log2(cg_label_rank + 1)
    cg_ndcg10   = float(np.where(cg_label_rank <= 10, 1.0 / log2_term, 0.0).mean())
    cg_hit10    = float((cg_label_rank <= 10).mean())
    cg_hit1     = float((cg_label_rank == 1).mean())
    cg_max_hit  = float((cg_label_rank < top_k).mean())  # label was inside CG top-(top_k-1)
    print(f"  [{label}] CG baseline: MRR={cg_mrr:.4f}  NDCG@10={cg_ndcg10:.4f}  "
          f"Hit@1={cg_hit1:.4f}  Hit@10={cg_hit10:.4f}  Hit@<{top_k}={cg_max_hit:.4f}")


def _build_dataframe(arrays, neg_corpus_idxs, cg_label_rank,
                     cg_label_score, cg_neg_scores,
                     genome_cosine_label, genome_cosine_negs):
    """Convert arrays + negatives + cg rank + interaction features into a parquet-ready DataFrame."""
    return pd.DataFrame({
        'user_id':                arrays['user_id'],
        'rollback_n':             arrays['rollback_n'].astype(np.int32),
        'label_corpus_idx':       arrays['label_corpus_idx'].astype(np.int32),
        'neg_corpus_idxs':        list(neg_corpus_idxs.astype(np.int32)),
        'cg_label_rank':          cg_label_rank.astype(np.int32),
        'cg_label_score':         cg_label_score,
        'cg_neg_scores':          list(cg_neg_scores),
        'genome_cosine_label':    genome_cosine_label,
        'genome_cosine_negs':     list(genome_cosine_negs),
        'user_avg_rating':        arrays['user_avg_rating'],
        'user_rating_count':      arrays['user_rating_count'].astype(np.int32),
        'user_genre_ctx':         list(arrays['X_genre']),
        'history_ids_padded':     list(arrays['X_history'].astype(np.int32)),
        'history_ratings_padded': list(arrays['X_hist_ratings']),
        'timestamp_bin':          arrays['timestamp_bin'].astype(np.int32),
    })


# ── Main entry ───────────────────────────────────────────────────────────────

def precompute(checkpoint_path: str | None = None,
               data_dir: str = 'data',
               output_dir: str = 'data',
               max_per_user: int = MAX_ROLLBACK_PER_USER,
               batch_size: int = SCORING_BATCH_SIZE,
               top_k: int = TOP_K_CANDIDATES):
    checkpoint_path = _resolve_checkpoint(checkpoint_path)
    print(f"CG checkpoint: {checkpoint_path}")

    print("\nLoading FeatureStore ...")
    fs = load_features(data_dir=data_dir, version='v1')
    n_movies = len(fs.top_movies)
    print(f"  {n_movies:,} corpus movies")

    device = get_device()
    print(f"\nDevice: {device}")
    model = load_cg(checkpoint_path, fs, device)

    with torch.no_grad():
        V_all = model.full_item_embedding()
    print(f"  V_all: shape={tuple(V_all.shape)}  device={V_all.device}")

    # Reproduce the v2-softmax 90/10 user-level split exactly
    raw_df = pd.read_parquet(os.path.join(data_dir, 'base_ratings.parquet'))
    corpus_set = set(fs.item_emb_movieId_to_i.keys())
    raw_df = raw_df[raw_df['movieId'].isin(corpus_set)]
    print(f"\n{len(raw_df):,} corpus-filtered interactions, "
          f"{raw_df['userId'].nunique():,} valid users")

    valid_users = sorted(raw_df['userId'].astype(int).unique().tolist())
    rng = random.Random(SPLIT_SEED)
    rng.shuffle(valid_users)
    split = int(len(valid_users) * PCT_TRAIN)
    train_users, val_users = valid_users[:split], valid_users[split:]
    print(f"  Train users: {len(train_users):,}   Val users: {len(val_users):,}")

    for split_name, users in [('train', train_users), ('val', val_users)]:
        print(f"\n── {split_name.upper()} split ──")
        arrays = _build_rollback_arrays(users, fs, raw_df, max_per_user, SPLIT_SEED)
        neg, cg_rank, cg_lscore, cg_nscores, gc_label, gc_negs = _score_candidates(
            model, V_all, arrays, device, batch_size, top_k)
        _verify_split(arrays, neg, cg_rank, fs, split_name, n_movies, top_k)
        df = _build_dataframe(arrays, neg, cg_rank, cg_lscore, cg_nscores, gc_label, gc_negs)
        out_path = os.path.join(output_dir, f'ranker_candidates_{split_name}.parquet')
        df.to_parquet(out_path, index=False)
        print(f"  → {out_path}  ({len(df):,} rows)")

    print("\n── Movie stats ──")
    compute_movie_stats(fs, data_dir, output_dir)

    print("\nPrecompute complete.")
