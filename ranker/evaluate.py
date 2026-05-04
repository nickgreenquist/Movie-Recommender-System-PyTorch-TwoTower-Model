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
    Rank = (count of negs with score > label_score) + 1, in [1, n_cand].
    """
    model.eval()
    n          = dataset.N
    n_cand     = 1 + dataset.n_neg                # 100
    user_dim   = dataset.user_dim
    item_dim   = dataset.item_dim
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

        item_feat   = dataset.item_features[cand_t]                   # (B, 100, item_dim)
        user_exp    = user_feat.unsqueeze(1).expand(-1, n_cand, -1)   # (B, 100, user_dim)

        scores = model(user_exp.reshape(B * n_cand, user_dim),
                       item_feat.reshape(B * n_cand, item_dim)).reshape(B, n_cand)

        label_score = scores[:, 0].unsqueeze(1)
        rank = (scores > label_score).sum(dim=1) + 1
        label_ranks[s:e] = rank.cpu().numpy()

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
    CG baseline NDCG@10 and MRR from the precomputed cg_label_rank column.

    cg_label_rank is the label's rank within the 100-candidate group when scored
    by CG itself (1-indexed, capped at 100). See ranker/precompute.py docstring.
    """
    return _ndcg_mrr_from_ranks(dataset.cg_label_rank)


def hit_rates_from_ranks(ranks: np.ndarray, ks: list = (1, 5, 10, 20, 50)) -> dict:
    return {f'Hit@{k}': float((ranks <= k).mean()) for k in ks}
