# MovieLens Ranker: Implementation Plan (Code-Aware)

## What I Learned From Reading The CG Code

### FeatureStore (`src/dataset.py: load_features()`)
All movie metadata lives here. Key fields for ranker:
- `fs.top_movies`: ordered list of movieIds (corpus index = position in list)
- `fs.item_emb_movieId_to_i`: movieId → corpus index
- `fs.movieId_to_genome_tag_context`: movieId → 1128-dim genome vector
- `fs.movieId_to_genre_context`: movieId → 18-dim genre one-hot
- `fs.movieId_to_tag_context`: movieId → tag frequency vector
- `fs.movieId_to_genres`: movieId → list of genre strings
- `genome_context_buffer` in model: `(n_movies+1, 1128)` — same data, as tensor

### CG Scoring Interface (`src/model.py: MovieRecommender`)
- `model.user_embedding(genre_ctx, history_ids, history_ratings, timestamps)` → `(batch, 128)` L2-normalized user vector
- `model.full_item_embedding()` → `(n_movies, 128)` all corpus item embeddings
- Scoring: `scores = user_emb @ V_all.T` → argsort → top-K corpus indices

### Rollback Logic (`src/dataset.py: build_mse_rollback_dataset`)
Already does exactly what we need:
- `sort_by_ts=True` by default
- Samples up to `MAX_MSE_ROLLBACK_EXAMPLES_PER_USER` positions per user
- history = corpus indices (`item_emb_movieId_to_i`)
- ratings = debiased (`rating - user_avg_rating`)
- genre context = running avg/frac up to position N-1
- timestamp = binned

**We REUSE this logic directly. Do not reimplement it.**

### History Format Difference
- MSE dataset: `X_history = list of lists` (padded at batch time with `pad_history_batch`)
- V2 softmax dataset: `X_history = (N, MAX_HISTORY_LEN)` pre-padded tensor

Ranker precompute should use the v2 softmax format (pre-padded tensor) — easier to index directly without `pad_history_batch` calls.

### `build_model()` (`src/train.py`)
Builds `MovieRecommender` with all buffers populated from `FeatureStore`. Ranker precompute imports and calls this directly.

---

## Repo Structure

```
Movie-Recommender-System-PyTorch-TwoTower-Model/
├── CLAUDE.md
├── assets/
├── src/                              ← CG — never modified by ranker work
│   ├── dataset.py                    ← FeatureStore, load_features(), rollback logic
│   ├── model.py                      ← MovieRecommender
│   ├── train.py                      ← build_model(), get_config()
│   ├── features.py
│   └── main.py
├── ranker/                           ← fully self-contained
│   ├── precompute.py                 ← generate candidates using CG, save to disk
│   ├── dataset.py                    ← load precomputed parquet → PyTorch dataset
│   ├── model.py                      ← MLPRanker
│   ├── train.py                      ← BCE training loop
│   ├── evaluate.py                   ← NDCG@10, MRR, CG baseline
│   └── main.py                       ← entry point
├── data/                             ← shared (existing)
│   ├── ranker_candidates.parquet     ← written by ranker/precompute.py
│   ├── ranker_movie_stats.parquet    ← global avg_rating, rating_count per movie
│   └── ... existing CG data files
├── saved_models/                     ← existing CG checkpoints
│   └── ranker/                       ← new subdirectory for ranker checkpoints
└── serving/                          ← Streamlit (untouched)
```

---

## Stage 0: `ranker/precompute.py`

### How to load CG

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset import load_features
from src.train import build_model, get_v2_config

# Load FeatureStore
fs = load_features(data_dir='data', version='v1')

# Build and load trained CG model
config = get_v2_config()
model = build_model(config, fs)
model.load_state_dict(torch.load('saved_models/best_mse_gpool_gctx_proj_*.pth',
                                  weights_only=True))
model.eval()
```

### CG scoring for a single rollback example

```python
# Compute V_all ONCE before the loop — static, reuse for all users
with torch.no_grad():
    V_all = model.full_item_embedding()        # (n_movies, 128)

# Per rollback example:
with torch.no_grad():
    U = model.user_embedding(
        genre_ctx,        # (1, user_context_size)
        history_ids,      # (1, MAX_HISTORY_LEN) pre-padded corpus indices
        history_ratings,  # (1, MAX_HISTORY_LEN) debiased ratings
        timestamp,        # (1,) binned int
    )                                          # (1, 128)
    scores = (U @ V_all.T).squeeze(0)         # (n_movies,)
    top100_corpus_idx = scores.argsort(descending=True)[:100].tolist()
```

### Key implementation notes
- Compute `V_all` ONCE before the loop, not per example. It's `(n_movies, 128)` — static.
- `history_ids` must be pre-padded to `MAX_HISTORY_LEN` with `pad_idx = len(fs.top_movies)`
- Rollback context = `history[0:N-1]` — replicate exactly from `build_v2_softmax_dataset`
- Remove label corpus index from top-100 before saving
- Corpus index → movieId: `fs.top_movies[corpus_idx]`

### Rollback implementation
Do not reimplement rollback from scratch. Copy the inner user loop from `build_v2_softmax_dataset()` in `src/dataset.py`, add CG scoring at each sampled rollback position, and write output rows directly.

### Also compute in precompute.py: movie stats
```python
# Load base_ratings.parquet, compute per-movie global stats
# Save as data/ranker_movie_stats.parquet
# Columns: movieId, global_avg_rating, global_rating_count (log1p scaled)
```

### Output schema: `data/ranker_candidates.parquet`
One row per rollback example:

| Column | Type | Description |
|--------|------|-------------|
| `user_id` | int | |
| `rollback_n` | int | position index in user's sorted history |
| `label_corpus_idx` | int | positive item corpus index |
| `neg_corpus_idxs` | list[int] | 99 negatives in CG rank order |
| `user_avg_rating` | float | |
| `user_rating_count` | int | |
| `user_genre_ctx` | list[float] | length = `user_context_size` = 36 |
| `history_ids_padded` | list[int] | MAX_HISTORY_LEN = 50 |
| `history_ratings_padded` | list[float] | MAX_HISTORY_LEN = 50 |
| `timestamp_bin` | int | |

**Note:** genome/genre/tag vectors for each candidate looked up at dataset load time from FeatureStore. Do not store 1128-dim vectors per-candidate in parquet (too large). Store corpus indices only; `ranker/dataset.py` resolves features.

### Verify before proceeding
- `label_corpus_idx` never in `neg_corpus_idxs`
- All `neg_corpus_idxs` are valid corpus indices (0 to `len(fs.top_movies)-1`)
- `user_genre_ctx` never computed using label item (only `history[0:N-1]`)
- Print: n_examples, avg negatives per example (should be 99), label ratio
- Spot check 3 known users against CG Streamlit app results

---

## Stage 1: Baseline MLP Ranker

### `ranker/dataset.py`

Load `data/ranker_candidates.parquet`. Load `FeatureStore` for feature lookups. Load `ranker_movie_stats.parquet` for global stats.

For each row, expand into 100 `(user_features, item_features, label)` tuples:
- 1 positive: `label_corpus_idx` with `label=1`
- 99 negatives: each `neg_corpus_idx` with `label=0`

**User features** (identical for all 100 candidates in a group):
- `user_genre_ctx`: 36-dim (from parquet)
- `user_avg_rating`: scalar (from parquet)
- `user_rating_count`: log1p scalar (from parquet)

**Item features** (looked up from FeatureStore + movie_stats per corpus index):
- `genome_vector`: 1128-dim (`fs.movieId_to_genome_tag_context`)
- `genre_vector`: 18-dim one-hot (`fs.movieId_to_genre_context`)
- `global_avg_rating`: scalar (from `ranker_movie_stats.parquet`)
- `global_rating_count`: log1p scalar (from `ranker_movie_stats.parquet`)
- `release_year`: normalized scalar (`fs.movieId_to_year`)

```python
class RankerDataset(Dataset):
    def __init__(self, candidates_df, fs, movie_stats_df):
        # Expand all rows into flat list of (user_feat, item_feat, label) tuples
        # at __init__ time — not lazily. Corpus is small enough.
        ...

    def __len__(self): ...

    def __getitem__(self, idx):
        return user_features, item_features, label  # all float tensors
```

### `ranker/model.py`

```python
class MLPRanker(nn.Module):
    def __init__(self, user_dim, item_dim, hidden_dims=[256, 128, 64]):
        super().__init__()
        input_dim = user_dim + item_dim
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers += [nn.Linear(in_dim, 1)]  # NO activation — BCEWithLogitsLoss handles sigmoid
        self.mlp = nn.Sequential(*layers)

    def forward(self, user_features, item_features):
        x = torch.cat([user_features, item_features], dim=1)
        return self.mlp(x).squeeze(-1)  # (batch,) scalar logits
```

### `ranker/train.py`
- Loss: `F.binary_cross_entropy_with_logits(scores, labels.float())`
- Optimizer: `Adam lr=1e-3`
- Gradient clipping: `clip_grad_norm_(model.parameters(), max_norm=1.0)`
- Batch sampling: random across all `(user, rollback, candidate)` tuples — NOT within rollback groups (avoids 99:1 imbalance within each batch)
- Val metric: NDCG@10 on held-out rollback groups (not val loss)
- Checkpoint: save when val NDCG@10 improves

### `ranker/evaluate.py`

```python
def evaluate_ndcg_mrr(model, groups, device):
    """
    groups: list of dicts, each with:
      - user_features: (1, user_dim)   — same for all 100 candidates
      - item_features: (100, item_dim)
      - label_idx: int (which of the 100 is the positive)
    Returns: mean NDCG@10, mean MRR
    """
    ndcg_scores, mrr_scores = [], []
    model.eval()
    with torch.no_grad():
        for group in groups:
            scores = model(
                group['user_features'].expand(100, -1).to(device),
                group['item_features'].to(device)
            ).sigmoid()
            rank = (scores > scores[group['label_idx']]).sum().item() + 1
            mrr_scores.append(1.0 / rank)
            ndcg_scores.append(1.0 / np.log2(rank + 1) if rank <= 10 else 0.0)
    return float(np.mean(ndcg_scores)), float(np.mean(mrr_scores))


def cg_baseline_ndcg(groups):
    """
    CG ranked candidates in order — neg_corpus_idxs is in CG rank order.
    Label was removed from top-100, so label's true CG rank is unknown (≤100).
    Baseline: treat label as rank 101 (just outside top-100) for all groups.
    This is a conservative CG baseline — actual CG performance is better.
    Record this alongside ranker NDCG@10.
    """
    ...
```

### `ranker/main.py`
```
python ranker/main.py precompute   # Stage 0: generate candidates
python ranker/main.py train        # Stage 1+: train ranker
python ranker/main.py evaluate     # eval only, loads best checkpoint
```

---

## Stages 2–5: Planned Ablations

**Do not implement until Stage 1 baseline metrics are recorded.**

Each adds ONE thing and measures NDCG@10 delta.

### Stage 2 — History Genome Pool
Add to `ranker/precompute.py`:
- Weighted avg of genome vectors over positive history items
- Weight = debiased rating, clamp negatives to 0 (positive history only)
- Store as `history_genome_pool` (1128-dim) in parquet

Add to user_features in `ranker/dataset.py`. No model architecture change.

### Stage 3 — Cross Features
Add to `ranker/precompute.py` (precomputed scalars):
- `genome_cosine_sim`: `cosine(candidate_genome, history_genome_pool)`
- `genre_overlap`: fraction of candidate genres in user's history genres
- `user_avg_rating_for_genre`: mean debiased rating for user's history movies in candidate's primary genre

No model architecture change — wider input concat only.

### Stage 4 — Item ID Embeddings
Add to `ranker/model.py`:
- `nn.Embedding(n_movies, 64)` — trained from scratch, not shared with CG
- `history_id_pool`: weighted avg of ID embeddings over positive history
- `candidate_id_emb`: single lookup for candidate item
- `dot_sim`: `(history_id_pool * candidate_id_emb).sum(dim=1)` — explicit interaction scalar

Architecture change required. New inputs needed from dataset.

### Stage 5 — DCN V2
Replace `MLPRanker` with `DCNRanker` (cross network + deep network). Same features as Stage 4, different architecture.
Paper: https://arxiv.org/abs/2008.13535

---

## Ablation Tracking Table

| Stage | Description | NDCG@10 | MRR | Delta vs prev |
|-------|-------------|---------|-----|---------------|
| CG baseline | CG ordering, no ranker | ? | ? | — |
| 1 | MLP, static features only | ? | ? | vs CG |
| 2 | + history genome pool | ? | ? | vs 1 |
| 3 | + cross features | ? | ? | vs 2 |
| 4 | + item ID embeddings | ? | ? | vs 3 |
| 5 | DCN V2 cross network | ? | ? | vs 4 |

---

## CLAUDE.md Additions

Add the following section to `CLAUDE.md`:

```
## Ranker (ranker/)

Architecture: two-stage pipeline
  Stage 1 (CG):     two-tower MSE model in src/ — retrieves top-100 candidates
  Stage 2 (Ranker): MLP ranker in ranker/ — reranks candidates with richer features

Key paths:
  CG model checkpoint:    saved_models/best_mse_gpool_gctx_proj_*.pth
  CG config:              src/train.py → get_v2_config() or get_config()
  Ranker candidates:      data/ranker_candidates.parquet
  Ranker movie stats:     data/ranker_movie_stats.parquet
  Ranker checkpoints:     saved_models/ranker/

Import rules:
  ranker/precompute.py    → imports src/ (CG model + FeatureStore only)
  ranker/dataset.py       → ZERO src/ imports (reads parquet + FeatureStore directly)
  ranker/model.py         → ZERO src/ imports
  ranker/train.py         → ZERO src/ imports
  ranker/evaluate.py      → ZERO src/ imports

Do NOT:
  - Modify any file in src/ when working on ranker
  - Put ranker checkpoints in saved_models/ root (use saved_models/ranker/)
  - Re-run precompute.py unless CG model or rollback logic changes
  - Use BCELoss — only BCEWithLogitsLoss
  - Add sigmoid to MLPRanker.forward()
  - Reimplement FeatureStore or rollback logic — import from src/
```

---

## Critical Warnings for Implementation

1. **`V_all` is computed once.** Call `model.full_item_embedding()` once before the precompute loop, not per user or per rollback example.

2. **Rollback context must never include the label item.** `history[0:N-1]` only. Verify explicitly before writing candidates to disk.

3. **`BCEWithLogitsLoss` only.** Never `BCELoss`. Never add `sigmoid` to `MLPRanker.forward()`.

4. **Batch sampling is across all tuples.** Do not group all 100 candidates from one rollback into the same batch — the 99:1 imbalance within a group would dominate gradients.

5. **Evaluation groups by rollback.** NDCG@10 and MRR require ranking within each `(user, rollback)` group of 100 candidates. Global ranking across users is meaningless.

6. **Record CG baseline NDCG@10 first.** Before training anything, measure how well CG's own ranking of the 99 hard negatives compares. This is the number every stage is measured against.

7. **Genome is 1128-dim.** This makes the initial input dim large (~1200+). If training is slow, apply PCA to 128-dim at precompute time. Decide after seeing actual training speed — do not pre-optimize.
