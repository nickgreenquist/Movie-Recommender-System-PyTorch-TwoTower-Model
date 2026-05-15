"""
Ranker evaluation: NDCG@10, MRR, plus the CG baseline (read from precompute parquet).

For each rollback group (1 label + 249 hard negs = 250 candidates):
  - Score all 250 candidates with the ranker
  - Compute label's rank within the group (1-indexed)
  - NDCG@10 = 1/log2(rank+1) if rank <= 10 else 0
  - MRR     = 1/rank

E2E ceiling: if CG didn't organically retrieve the label (cg_label_rank >= n_cand),
the ranker never sees it in production → rank = n_cand + 1 (score = 0).
"""
import numpy as np
import torch

from ranker.dataset import RankerDataset, compute_cross_features
from ranker.model import USER_CONCAT_LAYOUT


@torch.no_grad()
def compute_label_ranks(model, dataset: RankerDataset, device: torch.device,
                        batch_size: int = 32,
                        eval_indices: np.ndarray | None = None) -> np.ndarray:
    """
    Score rollback groups with `model`. Return label ranks (1-indexed, E2E-adjusted).

    eval_indices: optional np.int64 array of row indices to evaluate. If None, evaluates
                  every val row (slow). For training-time logging, pass a deterministic
                  sample (fixed seed) so successive evals are directly comparable.
    """
    model.eval()
    n_cand      = 1 + dataset.n_neg
    if eval_indices is None:
        eval_indices = np.arange(dataset.N, dtype=np.int64)
    n_eval      = len(eval_indices)
    label_ranks = np.zeros(n_eval, dtype=np.int32)

    for s in range(0, n_eval, batch_size):
        e = min(s + batch_size, n_eval)
        B = e - s
        rows = eval_indices[s:e]
        rows_t = torch.from_numpy(rows).long()

        # ── User side: compute ONCE per row (not per candidate) ─────────────
        ugc_b = dataset._user_genre_t[rows_t].to(device)         # (B, 40)
        xh_b  = dataset._X_history_t[rows_t].to(device)          # (B, max_hist)
        xhr_b = dataset._X_hist_rat_t[rows_t].to(device)         # (B, max_hist)
        ts_b  = dataset._timestamp_t[rows_t].to(device)          # (B,)
        # ONE user-side pass — shares the (B, H, n_genome) genome lookup across pools,
        # genome_ctx, profile, baseline, and recent signals.
        us = model.user_forward(ugc_b, xh_b, xhr_b, ts_b, k=5)
        user_concat       = us.user_concat
        user_g_prof_b     = us.profile
        last_item_emb_b   = us.last_item_emb
        last_k_mean_g_b   = us.last_k_mean_genome
        user_g_baseline_b = us.baseline

        # ── Item side: compute per candidate ────────────────────────────────
        cand = np.empty((B, n_cand), dtype=np.int64)
        cand[:, 0]  = dataset.label_idx[rows]
        cand[:, 1:] = dataset.neg_idx[rows]
        cand_flat   = torch.from_numpy(cand.reshape(-1)).to(device)         # (B*n_cand,)
        item_concat = model.item_embedding(cand_flat)                       # (B*n_cand, item_concat_dim)
        item_id_raw = model.item_id_lookup(cand_flat)                       # (B*n_cand, item_id_dim)
        cand_g_scores = model.genome_buffer[cand_flat]                      # (B*n_cand, n_genome)

        # ── Cross features (per row × candidate) ────────────────────────────
        cand_feat       = dataset.item_features[cand_flat]
        cand_genre_oh   = cand_feat[:, 1128:1148]
        cand_global_avg = cand_feat[:, 1148]
        cand_log_count  = cand_feat[:, 1149]
        cand_year_norm  = cand_feat[:, 1150]

        ugc_exp       = ugc_b.unsqueeze(1).expand(-1, n_cand, -1).reshape(B * n_cand, -1)
        user_avg_exp  = dataset._user_avg_t[rows_t].to(device).unsqueeze(1).expand(-1, n_cand).reshape(-1)
        user_cnt_exp  = dataset._user_count_log1p_t[rows_t].to(device).unsqueeze(1).expand(-1, n_cand).reshape(-1)
        user_year_exp = dataset.user_mean_year_norm[rows_t.to(device)].unsqueeze(1).expand(-1, n_cand).reshape(-1)

        if dataset.has_genome_cosine:
            gc = np.empty((B, n_cand), dtype=np.float32)
            gc[:, 0]  = dataset.genome_cosine_label[rows]
            gc[:, 1:] = dataset.genome_cosine_negs[rows]
            genome_cos_exp = torch.from_numpy(gc.reshape(-1)).to(device)
        else:
            genome_cos_exp = torch.zeros(B * n_cand, device=device)

        # User concat is per-row → expand across candidates. expand() is a view (cheap).
        user_concat_exp = user_concat.unsqueeze(1).expand(-1, n_cand, -1).reshape(B * n_cand, -1)
        pool_disliked_exp = user_concat_exp[:, slice(*USER_CONCAT_LAYOUT['pool_disliked'])]
        # Per-row recency / content profiles broadcast across candidates.
        user_g_prof_exp     = user_g_prof_b.unsqueeze(1).expand(-1, n_cand, -1).reshape(B * n_cand, -1)
        last_item_emb_exp   = last_item_emb_b.unsqueeze(1).expand(-1, n_cand, -1).reshape(B * n_cand, -1)
        last_k_mean_g_exp   = last_k_mean_g_b.unsqueeze(1).expand(-1, n_cand, -1).reshape(B * n_cand, -1)
        user_g_baseline_exp = user_g_baseline_b.unsqueeze(1).expand(-1, n_cand).reshape(B * n_cand)
        user_liked_pop_exp  = (dataset.user_mean_liked_log_count[rows_t.to(device)]
                               .unsqueeze(1).expand(-1, n_cand).reshape(B * n_cand))

        cross = compute_cross_features(
            ugc_exp, user_avg_exp, user_cnt_exp, user_year_exp,
            cand_genre_oh, cand_year_norm, cand_global_avg, cand_log_count,
            genome_cos_exp,
            pool_disliked_exp, item_id_raw, user_g_prof_exp, cand_g_scores,
            last_item_emb_exp, last_k_mean_g_exp,
            user_g_baseline_exp,
            user_liked_pop_exp,
        )

        # ── Score: cheap broadcast of user_concat across candidates ─────────
        scores = model.score_pairs(user_concat_exp, item_concat, cross).reshape(B, n_cand)

        label_score = scores[:, 0].unsqueeze(1)
        rank_np = ((scores > label_score).sum(dim=1) + 1).cpu().numpy()

        cg_found         = dataset.cg_label_rank[rows] < n_cand
        label_ranks[s:e] = np.where(cg_found, rank_np, n_cand + 1)

    return label_ranks


def evaluate_ndcg_mrr(model, dataset: RankerDataset, device: torch.device,
                      batch_size: int = 32,
                      eval_indices: np.ndarray | None = None) -> tuple[float, float]:
    return _ndcg_mrr_from_ranks(
        compute_label_ranks(model, dataset, device, batch_size, eval_indices=eval_indices))


def _ndcg_mrr_from_ranks(ranks: np.ndarray) -> tuple[float, float]:
    mrr  = float((1.0 / ranks).mean())
    ndcg = float(np.where(ranks <= 10, 1.0 / np.log2(ranks + 1), 0.0).mean())
    return ndcg, mrr


def cg_baseline(dataset: RankerDataset,
                eval_indices: np.ndarray | None = None) -> tuple[float, float]:
    """CG baseline NDCG@10 and MRR, E2E-consistent with compute_label_ranks."""
    n_cand = 1 + dataset.n_neg
    raw    = dataset.cg_label_rank if eval_indices is None else dataset.cg_label_rank[eval_indices]
    ranks  = np.where(raw < n_cand, raw, n_cand + 1)
    return _ndcg_mrr_from_ranks(ranks)


def hit_rates_from_ranks(ranks: np.ndarray, ks: list = (1, 5, 10, 20, 50, 100, 150, 200, 250)) -> dict:
    return {f'Hit@{k}': float((ranks <= k).mean()) for k in ks}
