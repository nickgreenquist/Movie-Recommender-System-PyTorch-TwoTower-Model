"""
Offline retrieval evaluation — Recall@K, NDCG@K, Hit Rate@K, MRR.

Rollback protocol: for each val user (held out at user level), sample up to
MAX_MSE_ROLLBACK_EXAMPLES_PER_USER chronological positions.  At each position j,
context = history[0..j-1], target = history[j].  All positions are
valid since val users were never seen in training.

Results are written to eval_results/<checkpoint_stem>.txt

Usage:
    python main.py eval
    python main.py eval <checkpoint_path>
"""
import math
import os
import random

import torch

from src.dataset import (FeatureStore, pad_history_batch, pad_history_ratings_batch,
                          build_mse_rollback_dataset, get_val_users,
                          MAX_MSE_ROLLBACK_EXAMPLES_PER_USER)
from src.evaluate import build_movie_embeddings
from src.model import MovieRecommender

EVAL_BATCH_SIZE = 512


def _build_emb_matrix(model, fs):
    movie_embeddings = build_movie_embeddings(model, fs)
    all_ids  = list(movie_embeddings.keys())
    all_embs = torch.cat(
        [movie_embeddings[mid]['MOVIE_EMBEDDING_COMBINED'] for mid in all_ids], dim=0
    )
    return all_ids, all_embs


def _print_results(recall, hit_rate, ndcg, mrr_sum, n_eval, ks, all_ids, checkpoint_path, label):
    max_k = max(ks)
    random_hit_baseline = max_k / len(all_ids)
    lines = [
        f"\n── Offline Evaluation [{label}]  (n={n_eval:,} rollbacks, {checkpoint_path or 'latest'}) "
        + "─" * 10,
        f"Corpus: {len(all_ids):,} movies  |  "
        f"Random Hit Rate@{max_k} baseline: {random_hit_baseline:.3%}\n",
    ]
    header = f"{'K':>6}  {'Recall@K':>10}  {'Hit Rate@K':>11}  {'NDCG@K':>8}"
    lines.append(header)
    lines.append("─" * len(header))
    for k in ks:
        lines.append(f"{k:>6}  "
                     f"{recall[k]/n_eval:>10.4f}  "
                     f"{hit_rate[k]/n_eval:>11.4f}  "
                     f"{ndcg[k]/n_eval:>8.4f}")
    lines.append("─" * len(header))
    lines.append(f"MRR: {mrr_sum/n_eval:.4f}")

    output = "\n".join(lines)
    print(output)

    out_dir  = 'eval_results'
    os.makedirs(out_dir, exist_ok=True)
    stem     = os.path.splitext(os.path.basename(checkpoint_path))[0] if checkpoint_path else 'latest'
    out_path = os.path.join(out_dir, f'{stem}.txt')
    with open(out_path, 'w') as f:
        f.write(output + "\n")
    print(f"  → saved to {out_path}")


def run_offline_eval(model: MovieRecommender, fs: FeatureStore,
                     checkpoint_path: str = '',
                     n_users: int = 5_000,
                     ks: tuple = (1, 5, 10, 20, 50),
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

    print(f"Building rollback examples for {len(eval_users):,} val users ...")
    (X_genre, X_history, X_history_ratings, timestamp, _, target_movieId) = \
        build_mse_rollback_dataset(eval_users, fs, raw_df,
                                   MAX_MSE_ROLLBACK_EXAMPLES_PER_USER, seed=seed + 1)

    n_examples = int(target_movieId.shape[0])

    # Pre-pad histories once so the scoring loop just slices pre-allocated tensors
    hist_idx_padded = pad_history_batch(X_history, model.pad_idx)
    hist_wts_padded = pad_history_ratings_batch(X_history_ratings)

    recall   = {k: 0.0 for k in ks}
    hit_rate = {k: 0   for k in ks}
    ndcg     = {k: 0.0 for k in ks}
    mrr_sum  = 0.0
    n_eval   = 0

    with torch.no_grad():
        for s in range(0, n_examples, EVAL_BATCH_SIZE):
            e = min(s + EVAL_BATCH_SIZE, n_examples)

            hist_idx_t = hist_idx_padded[s:e]
            hist_wts_t = hist_wts_padded[s:e]
            user_embs  = model.user_embedding(X_genre[s:e], hist_idx_t, hist_wts_t, timestamp[s:e])
            scores     = user_embs @ all_embs.T  # (B, n_items)

            for i in range(e - s):
                t_pos        = int(target_movieId[s + i].item())
                target_score = scores[i, t_pos]
                rank         = int((scores[i] > target_score).sum().item()) + 1

                mrr_sum += 1.0 / rank
                for k in ks:
                    hit = int(rank <= k)
                    recall[k]   += hit
                    hit_rate[k] += hit
                    ndcg[k]     += (1.0 / math.log2(rank + 1)) if hit else 0.0
                n_eval += 1

    if n_eval == 0:
        print("No rollback positions evaluated — check that feature parquets are loaded.")
        return

    _print_results(recall, hit_rate, ndcg, mrr_sum, n_eval, ks, all_ids, checkpoint_path,
                   f"MSE rollback (≤{MAX_MSE_ROLLBACK_EXAMPLES_PER_USER}/user)")
