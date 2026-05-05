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
        user_concat = model.user_embedding(ugc_b, xh_b, xhr_b, ts_b)        # (B, user_concat_dim)

        # ── Item side: compute per candidate ────────────────────────────────
        cand = np.empty((B, n_cand), dtype=np.int64)
        cand[:, 0]  = dataset.label_idx[rows]
        cand[:, 1:] = dataset.neg_idx[rows]
        cand_flat   = torch.from_numpy(cand.reshape(-1)).to(device)         # (B*n_cand,)
        item_concat = model.item_embedding(cand_flat)                       # (B*n_cand, item_concat_dim)

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

        cross = compute_cross_features(
            ugc_exp, user_avg_exp, user_cnt_exp, user_year_exp,
            cand_genre_oh, cand_year_norm, cand_global_avg, cand_log_count,
            genome_cos_exp,
        )

        # ── Score: cheap broadcast of user_concat across candidates ─────────
        # expand() creates a view (no copy of the genome pool work).
        user_concat_exp = user_concat.unsqueeze(1).expand(-1, n_cand, -1).reshape(B * n_cand, -1)
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
