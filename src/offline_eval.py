"""
Offline retrieval evaluation — Recall@K, NDCG@K, Hit Rate@K, MRR.

MSE protocol: leave-label-out per user.
  Context = user_to_watch_history (90% of ratings, pre-mapped movie indices)
  Targets = user_to_movie_to_rating_LABEL (remaining 10% of ratings)
  Users   = 10% MSE val users (held out from make_splits)

Softmax protocol: rollback eval on softmax val users.
  Val users (10% of all users) were never seen during softmax training, so
  every chronological position in their full history is a valid test target.
  For each user, sample up to MAX_ROLLBACKS_PER_USER positions; at each
  position j, context = history[0..j-1] and target = history[j].

Usage:
    python main.py eval
    python main.py eval <checkpoint_path>
"""
import math
import os
import random

import torch

from src.dataset import FeatureStore, pad_history_batch, pad_history_ratings_batch
from src.evaluate import build_movie_embeddings
from src.model import MovieRecommender

MAX_ROLLBACKS_PER_USER = 20
MIN_CONTEXT            = 3


def _build_emb_matrix(model, fs):
    movie_embeddings = build_movie_embeddings(model, fs)
    all_ids  = list(movie_embeddings.keys())
    all_embs = torch.cat(
        [movie_embeddings[mid]['MOVIE_EMBEDDING_COMBINED'] for mid in all_ids], dim=0
    )
    mid_to_pos = {mid: i for i, mid in enumerate(all_ids)}
    return all_ids, all_embs, mid_to_pos


def _ts_max_bin(fs):
    return torch.bucketize(
        torch.tensor([float(fs.timestamp_bins[-1].item())]),
        fs.timestamp_bins, right=False,
    )


def _print_results(recall, hit_rate, ndcg, mrr_sum, n_eval, ks, all_ids, checkpoint_path, label):
    max_k = max(ks)
    random_hit_baseline = max_k / len(all_ids)
    print(f"\n── Offline Evaluation [{label}]  (n={n_eval:,} rollbacks, {checkpoint_path or 'latest'}) "
          + "─" * 10)
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


def run_offline_eval(model: MovieRecommender, fs: FeatureStore,
                     checkpoint_path: str = '',
                     n_users: int = 5_000,
                     ks: tuple = (1, 5, 10, 20, 50),
                     seed: int = 42,
                     data_dir: str = 'data',
                     force_rollback: bool = False) -> None:
    if force_rollback or 'softmax' in checkpoint_path:
        _run_softmax_eval(model, fs, checkpoint_path, data_dir, n_users, ks, seed)
    else:
        _run_mse_eval(model, fs, checkpoint_path, n_users, ks, seed)


def _run_mse_eval(model, fs, checkpoint_path, n_users, ks, seed):
    model.eval()

    print("Building movie embeddings ...")
    all_ids, all_embs, mid_to_pos = _build_emb_matrix(model, fs)
    ts_bin = _ts_max_bin(fs)

    # MSE val users — held-out 10% from make_splits (same filter + seed)
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

    recall   = {k: 0.0 for k in ks}
    hit_rate = {k: 0   for k in ks}
    ndcg     = {k: 0.0 for k in ks}
    mrr_sum  = 0.0
    n_eval   = 0

    with torch.no_grad():
        for user in eval_users:
            hist_indices = fs.user_to_watch_history[user]
            hist_ratings = fs.user_to_watch_history_ratings[user]
            if not hist_indices:
                continue

            target_mids = [int(mid) for mid in fs.user_to_movie_to_rating_LABEL[user]
                           if int(mid) in mid_to_pos]
            if not target_mids:
                continue

            genre_ctx  = fs.user_to_context[user]
            hist_idx_t = pad_history_batch([hist_indices], model.pad_idx)
            hist_wts_t = pad_history_ratings_batch([hist_ratings])
            user_emb   = model.user_embedding(
                torch.tensor([genre_ctx]),
                hist_idx_t, hist_wts_t,
                ts_bin,
            )

            scores = (all_embs @ user_emb.T).squeeze(-1)

            n_targets        = len(target_mids)
            target_positions = [mid_to_pos[mid] for mid in target_mids]
            target_scores    = scores[target_positions]
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

    _print_results(recall, hit_rate, ndcg, mrr_sum, n_eval, ks, all_ids, checkpoint_path, "MSE leave-label-out")


def _run_softmax_eval(model, fs, checkpoint_path, data_dir, n_users, ks, seed):
    """
    Rollback eval for softmax models.

    Softmax val users were never seen during training (user-level split),
    so every chronological position in their full history is a valid target.
    Uses up to MAX_ROLLBACKS_PER_USER randomly-sampled positions per user.
    """
    import pandas as pd
    from src.features import MAX_HISTORY_LEN

    model.eval()

    print("Building movie embeddings ...")
    all_ids, all_embs, mid_to_pos = _build_emb_matrix(model, fs)
    ts_bin = _ts_max_bin(fs)

    # Softmax val users — 10% held out from make_softmax_splits (seed=42, all user_ids)
    valid_users = fs.user_ids[:]
    split_rng = random.Random(42)
    split_rng.shuffle(valid_users)
    split = int(len(valid_users) * 0.9)
    softmax_val_users = valid_users[split:]

    rng = random.Random(seed)
    rng.shuffle(softmax_val_users)
    eval_users = softmax_val_users[:n_users]

    # Load full chronological history (watch + labels) for eval users
    print("Loading raw interactions for softmax val users ...")
    watch_path  = os.path.join(data_dir, 'base_ratings_watch.parquet')
    labels_path = os.path.join(data_dir, 'base_ratings_labels.parquet')
    eval_set = set(eval_users)
    raw_df = pd.concat([
        pd.read_parquet(watch_path),
        pd.read_parquet(labels_path),
    ], ignore_index=True)
    raw_df = raw_df[raw_df['userId'].isin(eval_set)].sort_values(['userId', 'timestamp'])

    user_history = {}
    for uid, group in raw_df.groupby('userId'):
        uid = int(uid)
        movies  = [int(m) for m in group['movieId'].tolist()]
        ratings = [float(r) for r in group['rating'].tolist()]
        user_history[uid] = list(zip(movies, ratings))

    recall   = {k: 0.0 for k in ks}
    hit_rate = {k: 0   for k in ks}
    ndcg     = {k: 0.0 for k in ks}
    mrr_sum  = 0.0
    n_eval   = 0

    with torch.no_grad():
        for user in eval_users:
            history = user_history.get(user, [])
            avg_rat = fs.user_to_avg_rating.get(user, 3.0)

            valid_positions = [
                j for j in range(MIN_CONTEXT, len(history))
                if history[j][0] in mid_to_pos
            ]
            if not valid_positions:
                continue

            # Random sample up to MAX_ROLLBACKS_PER_USER positions
            positions = rng.sample(valid_positions, min(MAX_ROLLBACKS_PER_USER, len(valid_positions)))

            for j in positions:
                target_mid = history[j][0]

                # Context: corpus-filtered movies before j, capped to MAX_HISTORY_LEN
                ctx = [(fs.item_emb_movieId_to_i[m], r - avg_rat)
                       for m, r in history[:j]
                       if m in fs.item_emb_movieId_to_i][-MAX_HISTORY_LEN:]
                if not ctx:
                    continue

                hist_indices = [p[0] for p in ctx]
                hist_ratings = [p[1] for p in ctx]
                genre_ctx    = fs.user_to_context[user]

                hist_idx_t = pad_history_batch([hist_indices], model.pad_idx)
                hist_wts_t = pad_history_ratings_batch([hist_ratings])
                user_emb   = model.user_embedding(
                    torch.tensor([genre_ctx]),
                    hist_idx_t, hist_wts_t,
                    ts_bin,
                )

                scores = (all_embs @ user_emb.T).squeeze(-1)
                target_score = scores[mid_to_pos[target_mid]]
                rank = int((scores > target_score).sum().item()) + 1

                mrr_sum += 1.0 / rank
                for k in ks:
                    hit   = int(rank <= k)
                    recall[k]   += hit
                    hit_rate[k] += hit
                    ndcg[k]     += (1.0 / math.log2(rank + 1)) if hit else 0.0

                n_eval += 1

    if n_eval == 0:
        print("No rollback positions evaluated — check that feature parquets are loaded.")
        return

    _print_results(recall, hit_rate, ndcg, mrr_sum, n_eval, ks, all_ids, checkpoint_path,
                   f"Softmax rollback (≤{MAX_ROLLBACKS_PER_USER}/user)")
