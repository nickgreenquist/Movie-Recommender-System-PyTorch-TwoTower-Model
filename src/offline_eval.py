"""
Offline retrieval evaluation — Recall@K, NDCG@K, Hit Rate@K, MRR.

Protocol: leave-label-out per user.
  Context = user_to_watch_history (90% of ratings, pre-mapped movie indices)
  Targets = user_to_movie_to_rating_LABEL (remaining 10% of ratings)

No new splits needed — reuses the existing 90/10 split from preprocess.py.

Usage:
    python main.py eval
    python main.py eval <checkpoint_path>
"""
import math
import random

import torch

from src.dataset import FeatureStore, pad_history_batch, pad_history_ratings_batch
from src.evaluate import build_movie_embeddings
from src.model import MovieRecommender


def run_offline_eval(model: MovieRecommender, fs: FeatureStore,
                     checkpoint_path: str = '',
                     n_users: int = 5_000,
                     ks: tuple = (1, 5, 10, 20, 50),
                     seed: int = 42) -> None:
    model.eval()

    # ── Pre-compute item embedding matrix ────────────────────────────────────
    print("Building movie embeddings ...")
    movie_embeddings = build_movie_embeddings(model, fs)
    all_ids  = list(movie_embeddings.keys())
    all_embs = torch.cat(
        [movie_embeddings[mid]['MOVIE_EMBEDDING_COMBINED'] for mid in all_ids], dim=0
    )  # (n_movies, 110)
    mid_to_pos = {mid: i for i, mid in enumerate(all_ids)}

    # ── Reconstruct MSE val users (held-out 10% — never in MSE training set) ───
    # MSE trains on (90%-context, label-movie) pairs; using those users would inflate
    # MSE metrics since the eval is testing the exact pairs it trained on.
    # Replicate make_splits() logic (same filter + seed) to get true held-out users.
    all_eligible = [u for u in fs.user_ids
                    if 2 <= len(fs.user_to_movie_to_rating_LABEL.get(u, {})) < 500]
    split_rng = random.Random(42)
    split_rng.shuffle(all_eligible)
    split = int(len(all_eligible) * 0.9)
    val_users_set = set(all_eligible[split:])

    eligible = [u for u in val_users_set
                if fs.user_to_watch_history.get(u)
                and fs.user_to_movie_to_rating_LABEL.get(u)]
    rng = random.Random(seed)
    eval_users = rng.sample(eligible, min(n_users, len(eligible)))

    # ── Timestamp: use max bin (same as canary) ───────────────────────────────
    ts_max_bin = torch.bucketize(
        torch.tensor([float(fs.timestamp_bins[-1].item())]),
        fs.timestamp_bins, right=False,
    )

    # ── Accumulators ─────────────────────────────────────────────────────────
    recall   = {k: 0.0 for k in ks}
    hit_rate = {k: 0   for k in ks}
    ndcg     = {k: 0.0 for k in ks}
    mrr_sum  = 0.0
    n_eval   = 0

    with torch.no_grad():
        for user in eval_users:
            hist_indices = fs.user_to_watch_history[user]           # list[int]  (emb indices)
            hist_ratings = fs.user_to_watch_history_ratings[user]   # list[float] debiased

            if not hist_indices:
                continue

            # label_movieIds are raw movie IDs; filter to those in corpus
            target_mids = [int(mid) for mid in fs.user_to_movie_to_rating_LABEL[user]
                           if int(mid) in mid_to_pos]
            if not target_mids:
                continue

            # ── Build user embedding via model.user_embedding() ───────────────
            genre_ctx  = fs.user_to_context[user]
            hist_idx_t = pad_history_batch([hist_indices], model.pad_idx)
            hist_wts_t = pad_history_ratings_batch([hist_ratings])
            user_emb   = model.user_embedding(
                torch.tensor([genre_ctx]),
                hist_idx_t, hist_wts_t,
                ts_max_bin,
            )  # (1, embedding_dim) — respects use_user_genome_pool flag

            # ── Score all movies ───────────────────────────────────────────────
            scores = (all_embs @ user_emb.T).squeeze(-1)  # (n_movies,)

            # ── Metrics ───────────────────────────────────────────────────────
            n_targets        = len(target_mids)
            target_positions = [mid_to_pos[mid] for mid in target_mids]
            target_scores    = scores[target_positions]

            # Rank of each target: number of ALL movies scoring higher + 1
            ranks = (scores.unsqueeze(1) > target_scores.unsqueeze(0)).sum(dim=0) + 1

            best_rank = ranks.min().item()
            mrr_sum  += 1.0 / best_rank

            for k in ks:
                hits_k = (ranks <= k).sum().item()
                recall[k]   += hits_k / n_targets
                hit_rate[k] += int(hits_k > 0)
                dcg   = sum(1.0 / math.log2(r + 1) for r in ranks.tolist() if r <= k)
                ideal = sum(1.0 / math.log2(i + 2) for i in range(min(n_targets, k)))
                ndcg[k] += dcg / ideal if ideal > 0 else 0.0

            n_eval += 1

    if n_eval == 0:
        print("No users evaluated — check that feature parquets are loaded.")
        return

    # ── Print results ─────────────────────────────────────────────────────────
    max_k = max(ks)
    random_hit_baseline = max_k / len(all_ids)

    print(f"\n── Offline Evaluation  (n={n_eval:,} users, leave-label-out) "
          + "─" * 20)
    if checkpoint_path:
        print(f"Checkpoint: {checkpoint_path}")
    print(f"Corpus: {len(all_ids):,} movies  |  "
          f"Random Hit Rate@{max_k} baseline: {random_hit_baseline:.3%}\n")

    header = f"{'K':>6}  {'Recall@K':>10}  {'Hit Rate@K':>11}  {'NDCG@K':>8}"
    print(header)
    print("─" * len(header))
    for k in ks:
        print(f"{k:>6}  "
              f"{recall[k]/n_eval:>10.4f}  "
              f"{hit_rate[k]/n_eval:>11.4f}  "
              f"{ndcg[k]/n_eval:>8.4f}")
    print("─" * len(header))
    print(f"MRR: {mrr_sum/n_eval:.4f}")
