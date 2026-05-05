"""
Ranker evaluation: NDCG@10, MRR, plus the CG baseline (read from precompute parquet).

For each rollback group (1 label + 99 hard negs = 100 candidates):
  - Score all 100 candidates with the ranker
  - Compute label's rank within the group (1-indexed)
  - NDCG@10 = 1/log2(rank+1) if rank <= 10 else 0
  - MRR     = 1/rank

Imports src/dataset.load_features through ranker.dataset (canonical FeatureStore loader).
"""
import numpy as np
import torch

from ranker.dataset import RankerDataset


@torch.no_grad()
def compute_label_ranks(model, dataset: RankerDataset, device: torch.device,
                        batch_size: int = 64) -> np.ndarray:
    """
    Score every rollback group with `model`. Return (N,) array of label ranks (1-indexed).

    E2E semantics (the Golden Rule of two-stage evaluation):
      If CG did not organically retrieve the label (cg_label_rank >= n_cand), the ranker
      never sees it in production → rank is set to n_cand + 1 (score = 0 for all metrics).
      This enforces Ranker_Hit@K ≤ CG_Recall@n_cand.

    cg_label_rank is capped at n_cand in precompute. cg_label_rank < n_cand means the
    label's true full-corpus rank < n_cand (unambiguous: label was in CG's organic top-99).
    cg_label_rank == n_cand is ambiguous (rank n_cand or rank > n_cand); treated conservatively
    as "not found" to avoid overcounting retrieval successes.
    """
    model.eval()
    n          = dataset.N
    n_cand     = 1 + dataset.n_neg                # 100
    user_dim   = dataset.user_dim
    item_dim   = dataset.item_dim + dataset.n_interaction_features
    n_interact = dataset.n_interaction_features
    label_ranks = np.zeros(n, dtype=np.int32)

    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        B = e - s

        rows = np.arange(s, e)
        user_feat = dataset.user_features_for_rows(rows)               # (B, user_dim) on device

        cand = np.empty((B, n_cand), dtype=np.int64)
        cand[:, 0]  = dataset.label_idx[s:e]
        cand[:, 1:] = dataset.neg_idx[s:e]
        cand_t      = torch.from_numpy(cand).to(device)

        item_feat = dataset.item_features[cand_t]                     # (B, 100, item_dim_base)
        user_exp  = user_feat.unsqueeze(1).expand(-1, n_cand, -1)     # (B, 100, user_dim)

        if n_interact > 0:
            cg_b = np.empty((B, n_cand), dtype=np.float32)
            cg_b[:, 0]  = dataset.cg_label_score[s:e]
            cg_b[:, 1:] = dataset.cg_neg_scores[s:e]
            gc_b = np.empty((B, n_cand), dtype=np.float32)
            gc_b[:, 0]  = dataset.genome_cosine_label[s:e]
            gc_b[:, 1:] = dataset.genome_cosine_negs[s:e]
            interact_t = torch.from_numpy(
                np.stack([cg_b, gc_b], axis=2).astype(np.float32)
            ).to(device)                                               # (B, 100, 2)
            item_feat = torch.cat([item_feat, interact_t], dim=2)

        scores = model(user_exp.reshape(B * n_cand, user_dim),
                       item_feat.reshape(B * n_cand, item_dim)).reshape(B, n_cand)

        label_score = scores[:, 0].unsqueeze(1)
        rank_np = ((scores > label_score).sum(dim=1) + 1).cpu().numpy()

        # E2E ceiling: zero out ranker contribution for examples CG didn't retrieve.
        cg_found        = dataset.cg_label_rank[s:e] < n_cand
        label_ranks[s:e] = np.where(cg_found, rank_np, n_cand + 1)

    return label_ranks


def evaluate_ndcg_mrr(model, dataset: RankerDataset, device: torch.device,
                      batch_size: int = 64) -> tuple[float, float]:
    return _ndcg_mrr_from_ranks(compute_label_ranks(model, dataset, device, batch_size))


def _ndcg_mrr_from_ranks(ranks: np.ndarray) -> tuple[float, float]:
    mrr  = float((1.0 / ranks).mean())
    ndcg = float(np.where(ranks <= 10, 1.0 / np.log2(ranks + 1), 0.0).mean())
    return ndcg, mrr


def cg_baseline(dataset: RankerDataset) -> tuple[float, float]:
    """
    CG baseline NDCG@10 and MRR, E2E-consistent with compute_label_ranks.

    cg_label_rank is the label's true full-corpus rank capped at n_cand.
    cg_label_rank == n_cand is treated as "not found" (rank = n_cand + 1, score = 0)
    to match the same ceiling applied to the ranker.
    """
    n_cand = 1 + dataset.n_neg
    ranks  = np.where(dataset.cg_label_rank < n_cand,
                      dataset.cg_label_rank,
                      n_cand + 1)
    return _ndcg_mrr_from_ranks(ranks)


def hit_rates_from_ranks(ranks: np.ndarray, ks: list = (1, 5, 10, 20, 50, 100, 150, 200, 250)) -> dict:
    return {f'Hit@{k}': float((ranks <= k).mean()) for k in ks}
