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

CG = current PROD = v2 softmax model (`PROD_best_softmax_v2_popularity_alpha_05_*.pth`). The MSE model is dead and ignored everywhere in ranker code. Architecture is identical between MSE and v2 softmax (same `MovieRecommender`); both produce L2-normalized 128-dim embeddings, so dot product = cosine similarity for retrieval.

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset import load_features
from src.train import build_model, get_v2_config, get_device

# Load FeatureStore
fs = load_features(data_dir='data', version='v1')

# Build and load trained CG model — config is the v2 softmax config
config = get_v2_config()
model = build_model(config, fs)
model.load_state_dict(torch.load('saved_models/PROD_best_softmax_v2_popularity_alpha_05_*.pth',
                                  weights_only=True, map_location='cpu'))
model.eval().to(get_device())
```

### CG scoring (batched)

Build all rollback examples first into pre-padded tensors (matching the v2 softmax dataset format). Then score in batches — orders of magnitude faster than per-example on MPS/CUDA.

```python
# Compute V_all ONCE before the loop — static, reuse for all users
with torch.no_grad():
    V_all = model.full_item_embedding()        # (n_movies, 128)

# Batched scoring (batch_size=512):
with torch.no_grad():
    U = model.user_embedding(
        genre_ctx_batch,        # (B, user_context_size)
        history_ids_batch,      # (B, MAX_HISTORY_LEN) pre-padded corpus indices
        history_ratings_batch,  # (B, MAX_HISTORY_LEN) debiased ratings
        timestamp_batch,        # (B,) binned int
    )                                          # (B, 128)
    scores = U @ V_all.T                       # (B, n_movies)
    # Mask the label position to -inf BEFORE topk so the label is never returned
    scores[torch.arange(B), label_corpus_idx_batch] = float('-inf')
    top99 = scores.topk(99, dim=1).indices     # (B, 99) — pure negatives
```

The plan originally said "take top-100, remove label". The masking approach is equivalent but cleaner: ask for exactly 99 negatives in CG rank order, no post-filtering needed.

### Key implementation notes
- Compute `V_all` ONCE before the loop, not per example. It's `(n_movies, 128)` — static.
- `history_ids` must be pre-padded to `MAX_HISTORY_LEN` with `pad_idx = len(fs.top_movies)`
- Right-aligned padding (most recent items at right) — same as `build_v2_softmax_dataset`
- Rollback context = `history[0:N-1]` — replicate exactly from `build_v2_softmax_dataset`
- Mask label corpus index to `-inf` before `topk(99)` — guarantees label not in negatives
- Corpus index → movieId: `fs.top_movies[corpus_idx]`
- Use `get_device()` (MPS on Apple Silicon) — model and V_all stay on device throughout

### Rollback implementation
Do not reimplement rollback from scratch. Copy the inner user loop from `build_v2_softmax_dataset()` in `src/dataset.py`, but track `userId` and `rollback_n` per example (the existing CG dataset doesn't expose these). Accumulate examples into batches; flush a batch through CG when full.

### Train/val split
- 90/10 user-level split — same seed/scheme as `make_v2_softmax_splits` (`random.Random(42).shuffle(valid_users)`)
- **Users only** — never split a single user's watch history. Rollback handles temporal/cold-start signal entirely.
- Two output parquets — one per split. The plan originally listed one file; the ranker needs a held-out group for NDCG@10 eval, so we write `_train.parquet` and `_val.parquet`.

### Also compute in precompute.py: movie stats
```python
# Load base_ratings.parquet, compute per-movie global stats
# Save as data/ranker_movie_stats.parquet
# Columns: movieId, global_avg_rating, global_rating_count (log1p scaled)
```

### Output schema: `data/ranker_candidates_{train,val}.parquet`
One row per rollback example. Two files (one per user-level split):

| Column | Type | Description |
|--------|------|-------------|
| `user_id` | int | |
| `rollback_n` | int | position index in user's sorted history |
| `label_corpus_idx` | int | positive item corpus index |
| `neg_corpus_idxs` | list[int] | 99 negatives in CG rank order |
| `cg_label_rank` | int | label's rank within the (label + 99 negs) group as scored by CG, 1-indexed, capped at 100. Equals `min(label_full_corpus_rank, 100)`. Used as the CG baseline for ranker NDCG@10/MRR comparison. |
| `user_avg_rating` | float | |
| `user_rating_count` | int | |
| `user_genre_ctx` | list[float] | length = `user_context_size` = 40 (= 2 × n_genres = 2 × 20) |
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
- `user_genre_ctx`: 40-dim (from parquet)  — actual `user_context_size` = 2 × 20 genres
- `user_avg_rating`: scalar (from parquet)
- `user_rating_count`: log1p scalar (from parquet)

**Item features** (looked up from FeatureStore + movie_stats per corpus index):
- `genome_vector`: 1128-dim (`fs.movieId_to_genome_tag_context`)
- `genre_vector`: 20-dim one-hot (`fs.movieId_to_genre_context`)
- `global_avg_rating`: scalar (from `ranker_movie_stats.parquet`)
- `global_rating_count`: log1p scalar (from `ranker_movie_stats.parquet`)
- `release_year`: normalized scalar (`fs.movieId_to_year`)

**DEVIATION FROM PLAN:** the original plan said "expand all rows into flat list of tuples at __init__". With 3.4M rows × 100 candidates = 340M tuples, the expansion would need many GB of RAM (genome alone is 1128 floats × 340M = ~1.5 TB). Instead:

- `RankerDataset` keeps compact per-row arrays: `(N, 99)` neg indices, `(N,)` label, `(N, 40)` genre ctx, scalars
- `item_features` is a single `(n_movies=9375, item_dim=1151)` tensor built once and shared across train/val
- Lookups are vectorized: `item_features[corpus_idx]`
- No `Dataset` / `DataLoader` — the training loop calls `sample_batch(dataset, batch_size, device, rng)` directly. It samples random `(row, candidate_position)` tuples in pure NumPy, then indexes the feature tensors.

```python
class RankerDataset:
    def __init__(self, parquet_path, fs, movie_stats_df, item_features=None):
        # compact per-row arrays + shared item_features tensor
        ...
    def to(self, device): ...   # move feature tensors to GPU/MPS
    def user_features_for_rows(self, row_idx): ...

def sample_batch(dataset, batch_size, device, rng):
    rows = rng.integers(0, dataset.N, size=batch_size)
    pos  = rng.integers(0, 100, size=batch_size)  # 0=label, 1-99=neg index
    is_label = (pos == 0)
    cand_idx = np.where(is_label, dataset.label_idx[rows],
                        dataset.neg_idx[rows, np.maximum(pos-1, 0)])
    user_t  = dataset.user_features_for_rows(rows).to(device)
    item_t  = dataset.item_features[torch.from_numpy(cand_idx).long().to(device)]
    label_t = torch.from_numpy(is_label.astype(np.float32)).to(device)
    return user_t, item_t, label_t
```

Memory: train dataset ≈ 700 MB, val ≈ 80 MB, item_features ≈ 43 MB. All comfortably fit in 24 GB RAM.

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
- Loss: `F.binary_cross_entropy_with_logits(adjusted_logits, labels.float())`
  - **Menon α logit adjustment** (see "Unified popularity debiasing" section below): `adjusted_logits = raw_logits + α · log1p(count_i)` during training; raw logits at inference.
- Optimizer: `Adam lr=1e-3`
- Gradient clipping: `clip_grad_norm_(model.parameters(), max_norm=1.0)`
- Batch sampling: random across all `(user, rollback, candidate)` tuples — NOT within rollback groups (avoids 99:1 imbalance within each batch)
- Val metric: NDCG@10 on held-out rollback groups (not val loss)
- Checkpoint: save when val NDCG@10 improves; sidecar `_config.json` records `popularity_alpha`

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


def cg_baseline(cg_label_rank: np.ndarray) -> tuple[float, float]:
    """
    CG baseline NDCG@10 / MRR computed from cg_label_rank column in parquet.

    cg_label_rank[i] = label's rank within the 100-candidate group (label + 99 hard negs)
                      as scored by CG. 1-indexed, capped at 100.

    Returns (ndcg10, mrr) — these are the numbers every ranker stage is compared against.
    """
    mrr   = float((1.0 / cg_label_rank).mean())
    ndcg  = float(np.where(cg_label_rank <= 10, 1.0 / np.log2(cg_label_rank + 1), 0.0).mean())
    return ndcg, mrr
```

### `ranker/main.py`
```
python ranker/main.py precompute   # Stage 0: generate candidates
python ranker/main.py train        # Stage 1+: train ranker
python ranker/main.py evaluate     # eval only, loads best checkpoint
```

---

---

## Unified popularity debiasing (Menon α across CG and ranker)

### The finding

**Stage 1 ranker without popularity debiasing scored well on offline metrics but failed qualitatively.** Side-by-side canary inspection (`ranker/canary_results/ranker_mlp_20260504_160616.txt`) showed the ranker collapsing diverse genres to IMDb top-100:

- **Sci-Fi Lover**: CG → Pi, Stalker, Primer, Fifth Element, Clockwork Orange. Ranker → Shawshank, Pulp Fiction, Forrest Gump, Silence of the Lambs, Dark Knight.
- **Heist Lover**: CG → RocknRolla, Layer Cake, Quantum of Solace. Ranker → Forrest Gump, Matrix, Dark Knight, Inception.
- **Anime Lover**: CG → Kiki's Delivery Service, Porco Rosso, Whisper of the Heart. Ranker → Matrix, Inception, Interstellar, Up, WALL·E.
- **Crime Lover**: CG → Killing Them Softly, Shot Caller. Ranker → Forrest Gump, Pulp Fiction, Matrix, Fight Club.
- **Children's Movie Lover**: ranker even injected Dark Knight at #1 (!), Inception at #2.

Offline metrics looked great (NDCG@10 +47%, Hit@1 +71%) — but the ranker had clearly learned the cheap shortcut "popular = positive" rather than personalized ranking.

### Why this happens (mathematical root cause)

Same bias as CG had pre-Menon, transferred down the pipeline:

- **Implicit feedback is popularity-skewed.** In the data, popular movies are positive labels far more often (users watch what's surfaced).
- **CG's Menon α=0.5 only fixes CG's retrieval.** It shrinks popular item embeddings during CG training so CG's *top-100 set* is not popularity-biased.
- **The ranker is trained on a fresh BCE task with its own loss function.** The popularity bias re-enters: BCE positives correlate strongly with item popularity. The ranker gets a free shortcut by learning "popularity → label=1".
- **Worse, the ranker has explicit popularity features** (`global_avg_rating`, `log1p(global_rating_count)`). Even if popularity weren't correlated with positives, the ranker has a direct popularity input it can latch onto.

The +47% NDCG@10 was partly real ranker improvement and partly popularity arbitrage, exactly the same way α=0 CG looked great offline but had genre collapse on canary.

### The fix: Menon α applied to ranker BCE

```python
# During training:
adjusted_logit = raw_logit + α · log1p(count_i)
loss           = BCE(adjusted_logit, label)

# At inference:
score = raw_logit  # no adjustment — popular items' representations have shrunk
```

**Why this works for BCE specifically** (Menon et al. 2021 + label-dependent cost-sensitive learning):
- Popular item, target=1 (positive): `σ(z + 7) ≈ 1`, BCE loss ≈ 0 → "lazy positive", model gets little gradient from it
- Popular item, target=0 (negative): `σ(z + 7) ≈ 1` but target=0 → BCE loss is huge, model aggressively pushes weights down
- Net effect: the raw `f(u, i)` is forced to learn user/content match because popularity is "absorbed" by the offset term. At inference, the raw logit reflects content match alone.

### Why ranker α may differ from CG α

| Stage | Job | Failure mode | Typical α |
|---|---|---|---|
| CG (softmax) | Recall | Tail items missing from top-100 | 0.5 (current PROD) |
| Ranker (BCE) | Precision | Popular items dominate top-N | **1.0** (default in `ranker/train.py:get_config`) |

Per Menon et al. 2021 and production folklore: rankers are precision-critical and far more prone to popularity memorization than retrievers. A more aggressive α at the ranker stage is standard.

### Pipeline summary (unified)

| Component | Training strategy | Inference strategy |
|---|---|---|
| CG (two-tower) | logits + α_cg · log1p(count) before softmax (α_cg = 0.5) | raw `U·V^T`, no offset |
| Ranker (MLP)   | logits + α_ranker · log1p(count) before BCE   (α_ranker = 1.0) | raw `f(u, i)`, no offset |

Both stages "speak the same language" — popularity is absorbed during training, raw scores at inference are debiased.

### Code

- `ranker/train.py`: `popularity_alpha` in config (default 1.0); applied to logits in the BCE step.
- `ranker/dataset.py`: `RankerDataset.movie_interaction_counts` from `fs.movie_interaction_counts` (loaded from `data/movie_interaction_counts_v2.npy`).
- Inference paths (`evaluate.py`, `canary.py`) unchanged.
- Checkpoints: filename includes alpha (`ranker_mlp_alpha_1_<ts>.pth`); JSON sidecar records full config.

---

## Stages 2–5: Planned Ablations

**Do not implement until Stage 1 baseline metrics are recorded.**

Each adds ONE thing and measures NDCG@10 delta.

### Stage 2 — User Genome Context (analogous to CG's `user_genome_context_tower`)

**Motivation:** CG's `user_genome_context_tower` (rating-weighted avg of raw genome scores → Linear(1128→32)) delivered +8% MRR over CG's base projection MLP. The ranker has zero genome signal on the user side — this is the most likely reason Stage 1 loses to CG despite having richer item features.

The ranker doesn't need to replicate CG's sub-tower projection — it can pass the raw 1128-dim genome context into the MLP and let the MLP learn to compress it. This is simpler and avoids adding a trainable sub-tower.

**⚠️ MEMORY PROBLEM — DO NOT IMPLEMENT AS WRITTEN ⚠️**

The naive approach (store 1128-dim genome_ctx per example in parquet + `_user_features_full`) uses ~15 GB of RAM for train alone, vs CG's ~12 GB for its entire dataset. That is unacceptable. Before implementing, study how CG keeps memory low and replicate the pattern.

**How CG avoids the per-example genome expansion (key insight):**

CG does NOT store 1128-dim genome context per training example. It stores compact corpus indices (`X_history`, 50 ints/example, ~680 MB for 3.4M examples). At forward-pass time, the model indexes `genome_context_buffer` (a `(n_movies+1, 1128)` buffer, ~42 MB) to recover genome vectors on-the-fly:

```python
# model.py:161-166 — computed per batch, not pre-expanded per example
watched_genome = self.genome_context_buffer[user_watch_history]   # (B, 50, 1128) — small batch
genome_ctx_raw = (watched_genome * rating_weights).sum(dim=1) / weight_sum  # (B, 1128)
```

Per-batch cost: `batch_size × 50 × 1128 × 4` bytes. At batch=1024: 230 MB for one pass, immediately freed. No per-example accumulation at all.

**The ranker already stores `history_ids_padded` and `history_ratings_padded` in the parquet** — the same information CG uses. The fix is to replicate CG's approach in the ranker:

1. **`ranker/dataset.py`**: Load `history_ids_padded` (N, 50) and `history_ratings_padded` (N, 50) into the dataset as compact arrays (680 MB × 2 = 1.4 GB total — acceptable). Build a `genome_context_buffer` (n_movies+1, 1128) tensor from FeatureStore (~42 MB). Keep both in RAM.
2. **`ranker/dataset.py` `sample_batch()`**: Compute genome_ctx_raw per batch from the history arrays — same formula as CG. Do NOT add genome_ctx to `_user_features_full`.
3. **`ranker/model.py` or `sample_batch()`**: Return genome_ctx_raw as part of user features for each batch, concatenating with genre_ctx on the fly. user_dim stays logically 1170 but is never fully materialized for all N examples simultaneously.
4. **Do not store `user_genome_ctx` in parquet** — no precompute changes needed; the history columns already exist.

**Memory budget (corrected):**
- `history_ids_padded` (N=3.4M, 50 ints): 680 MB
- `history_ratings_padded` (N=3.4M, 50 floats): 680 MB
- `genome_context_buffer` (9376, 1128): 42 MB
- `_user_features_full` stays at 42 dims: 572 MB (genre_ctx + avg + log1p only)
- Per-batch genome computation: batch×50×1128×4 ≈ 230 MB, immediately freed
- **Total additional RAM: ~1.4 GB** (vs 15 GB naive approach)

**Canary:** `_build_synthetic_user_features()` can compute genome_ctx directly from movie lists (no history arrays needed — small set of fav/anchor/disliked titles).

**No precompute rerun needed** — history columns already in parquet from Stage 0.

**Before implementing Stage 2, read `src/model.py:149-173` and `src/dataset.py:build_v2_softmax_dataset` in full to understand the memory-efficient pattern, then replicate it exactly.**

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

All numbers on val split (382,138 rollback groups, 1 label + 99 hard negs each).

**α baseline rule:** All ranker experiments use α=0.5 as the fixed baseline. Do NOT tune α until the ranker beats CG on offline metrics (NDCG@10 > 0.0965). Changing α and features simultaneously makes results uninterpretable — change one thing at a time.

| Stage | Description | α | NDCG@10 | MRR | Notes |
|-------|-------------|---|---------|-----|-------|
| CG baseline | CG ordering, no ranker | (CG α=0.5) | 0.0965 | 0.0897 | — |
| 1a | MLP, static features only | **0** | 0.1422 | 0.1338 | Great offline, **bad canary** (popular drift) |
| 1b | MLP, static features + Menon α | **1.0** | < CG | < CG | **Loses to CG on offline eval** |
| 1c | MLP, static features + Menon α | **0.5** | < CG | < CG | **Also loses to CG — user features too weak** |
| 2 | + user_genome_context (1128-dim) | 0.5 | ? | ? | Next experiment — see Stage 2 notes |
| 3 | + cross features | 0.5 | ? | ? | vs 2 |
| 4 | + item ID embeddings | 0.5 | ? | ? | vs 3 |
| 5 | DCN V2 cross network | 0.5 | ? | ? | vs 4 |
| (α tuning) | tune α after beating CG | tbd | ? | ? | Only attempt after Stage ≥2 beats CG |

### Finding: Stage 1b and 1c both lose to CG — root cause: weak user features

Stages 1b (α=1.0) and 1c (α=0.5) both fail to beat CG on offline NDCG@10/MRR despite the ranker seeing item content features CG cannot. Root cause: **the ranker's user representation is far weaker than CG's**:

| | Ranker (Stage 1) | CG user tower |
|---|---|---|
| Genre signal | genre_ctx 40-dim (raw fraction/avg) | genre_tower(32) — learned projection |
| Genome signal | **NONE** | genome_pool(32) + genome_ctx_tower(32) — the +8% MRR feature |
| History signal | **NONE** | id_pool(32) — item co-watch CF |
| Timestamp | **NONE** | ts_emb(4) |
| User stats | avg_rating + log1p_count (2 scalars) | implicit in history pool |

The genome context tower was +8% MRR for CG. Without it, the ranker cannot even match CG's ordering of the same 100 candidates it retrieved. α=0 wins offline *only because it memorizes popularity* — the identical failure mode as CG without Menon.

Stage 1a offline numbers (α=0, published in the "Full Hit@K table" below) are **NOT valid ranker wins** — they reflect popularity arbitrage, same as CG α=0. Discard them from the success story.

### Stage 1a details (α=0, for reference only)
- Checkpoint: `saved_models/ranker/ranker_mlp_20260504_160616.pth` (1.4 MB, 346,881 params)
- Config: Adam lr=1e-3 + cosine schedule, batch=1024, 50K steps, hidden=[256,128,64], grad_clip=1.0
- Best at step 36K. BCE loss plateaued ~0.055 from step 4K onward. Pos rate stayed at 0.010 (1%, natural data ratio).
- Runtime: ~6 min on MPS.

#### Full Hit@K table — α=0 (popularity-memorizing; NOT a valid win)

| Metric | CG baseline | Stage 1a ranker (α=0) | Delta | Relative |
|---|---|---|---|---|
| Hit@1  | 0.0404 | 0.0690 | +0.0286 | +71% |
| Hit@5  | 0.1155 | 0.1613 | +0.0458 | +40% |
| Hit@10 | 0.1717 | 0.2454 | +0.0737 | +43% |
| Hit@20 | 0.2460 | 0.3803 | +0.1344 | +55% |
| Hit@50 | 0.3766 | 0.6611 | +0.2845 | +76% |
| NDCG@10 | 0.0965 | 0.1422 | +0.0457 | +47% |
| MRR    | 0.0897 | 0.1338 | +0.0441 | +49% |

These gains are spurious — the ranker learned "popular = positive" without any user-content matching. Same failure as CG α=0.

---

## CLAUDE.md Additions

Add the following section to `CLAUDE.md`:

```
## Ranker (ranker/)

Architecture: two-stage pipeline
  Stage 1 (CG):     v2 softmax two-tower model in src/ — retrieves top-100 candidates
  Stage 2 (Ranker): MLP ranker in ranker/ — reranks candidates with richer features

Key paths:
  CG model checkpoint:    saved_models/PROD_best_softmax_v2_popularity_alpha_05_*.pth
  CG config:              src/train.py → get_v2_config()
  Ranker candidates:      data/ranker_candidates_{train,val}.parquet
  Ranker movie stats:     data/ranker_movie_stats.parquet
  Ranker checkpoints:     saved_models/ranker/

Import rules:
  ranker/precompute.py    → imports src/ (CG model + FeatureStore)
  ranker/dataset.py       → imports src/ (FeatureStore via load_features only — canonical loader)
  ranker/model.py         → ZERO src/ imports
  ranker/train.py         → ZERO src/ imports
  ranker/evaluate.py      → imports src/ (FeatureStore via load_features only — canonical loader)
  Note: re-implementing FeatureStore in ranker/ would duplicate ~100 lines of vocab/index logic
  and risk drifting out of sync. Use src.dataset.load_features() instead.

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

1. **CG = v2 softmax PROD only.** The MSE model is dead. Do not load `best_mse_*` checkpoints — they will load fine (same architecture) but they're a worse retriever (6.6× lower MRR).

2. **`V_all` is computed once.** Call `model.full_item_embedding()` once before the precompute loop, not per user or per rollback example.

3. **Rollback context must never include the label item.** `history[0:N-1]` only. Verify explicitly before writing candidates to disk.

4. **`BCEWithLogitsLoss` only.** Never `BCELoss`. Never add `sigmoid` to `MLPRanker.forward()`.

5. **Batch sampling is across all tuples.** Do not group all 100 candidates from one rollback into the same batch — the 99:1 imbalance within a group would dominate gradients.

6. **Evaluation groups by rollback.** NDCG@10 and MRR require ranking within each `(user, rollback)` group of 100 candidates. Global ranking across users is meaningless.

7. **Record CG baseline NDCG@10 first.** Before training anything, measure how well CG's own ranking of the 99 hard negatives compares. This is the number every stage is measured against.

8. **User-level split, never within-user.** The 90/10 train/val split partitions *users*, not interactions. Every rollback example for a given user lands entirely in either train or val. This avoids label leakage and matches the existing CG split.

9. **Genome is 1128-dim.** This makes the initial input dim large (~1200+). If training is slow, apply PCA to 128-dim at precompute time. Decide after seeing actual training speed — do not pre-optimize.
