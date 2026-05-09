# MovieLens Ranker: Implementation Plan (Current State)

## Pipeline Overview

Two-stage retrieve-and-rank:
1. **CG** (v3 softmax two-tower, 4-pool user tower, L2-normalized, 128-dim) — retrieves top-250 candidates per rollback example
2. **Ranker** (Wide & Deep MLP) — reranks the 250 candidates using richer features

**CG baselines:**

| Model | Protocol | NDCG@10 | MRR | Hit@1 | Hit@10 | Recall@250 |
|-------|----------|---------|-----|-------|--------|-----------|
| v2 (old prod) | Full-corpus rollback | 0.0984 | 0.0878 | — | 17.49% | 67.48% |
| v2 (old prod) | E2E-adjusted, 250-cand | 0.0938 | 0.0845 | — | — | — |
| v3 (current prod) | Full-corpus rollback | 0.1296 | 0.1153 | 5.99% | 22.06% | 72.52% |
| **v3 (current prod)** | **E2E-adjusted, 250-cand** | **0.1233** | **0.1106** | **5.60%** | **21.17%** | **72.65%** |

Every ranker experiment is measured against **NDCG@10 = 0.1233**.

---

## Core Principle: CG-Parity Baseline First

**The ranker baseline must contain EVERY CG input/feature, projected through the same per-feature towers CG uses, with the same dimensions and the same initialization.** Cross features come *on top* of this baseline — they are the differentiator, not a replacement.

**Architectural difference vs CG:** No separate user/item tower with late L2-normalized dot product. All per-feature embeddings (user-side and item-side) are concatenated into ONE vector that feeds the deep MLP. **We cross features early via the MLP**, then add wide cross features straight to the head.

> **Why:** A ranker that sees less than CG cannot be expected to beat CG regardless of clever cross features. Industry practice (Google, YouTube, Netflix): the ranker always receives every signal CG had, plus more. Skipping a CG feature in the ranker is information loss, not simplification.

> **No CG coupling:** The ranker owns its own copies of every parameter (its own `item_embedding_lookup`, its own genome tower, its own genre tower, etc.). It just *replicates the architecture* of each CG sub-tower. No shared `nn.Module`, no loaded CG state_dict tensors.

### Ranker baseline concat layout (CG-parity, 292-dim deep input)

```
USER SIDE — mirrors v3 CG user tower:
  pool_full         : 32   [LayerNorm]   sum(item_emb_lookup[full_history])
  pool_liked        : 32   [LayerNorm]   sum(item_emb_lookup[liked_history])
  pool_disliked     : 32   [LayerNorm]   sum(item_emb_lookup[disliked_history])
  pool_weighted     : 32   [LayerNorm]   rating-weighted sum(item_emb_lookup[full])
  user_genome_ctx   : 32                 Linear(1128 → 32) over rating-wtd avg genome
  user_genre_emb    : 32                 Linear(40 → 32) from user_genre_ctx
  ts_emb            :  4                 Embedding(N_ts_bins, 4)

ITEM SIDE — mirrors v3 CG item tower:
  item_id_emb       : 32                 item_emb_lookup → Linear+ReLU(32 → 32)
  item_genre_emb    :  8                 Linear(20 → 8) from genre_onehot
  item_tag_emb      : 16                 Linear(306 → 16) from user-applied tag vector
  item_genome_emb   : 32                 Linear(1128 → 32) from genome_scores
  year_emb          :  8                 Embedding(N_year_bins, 8)

TOTAL deep concat: 4×32 + 32 + 32 + 4 + 32 + 8 + 16 + 32 + 8 = 292 dims

Deep MLP:  Linear(292 → 256) → ReLU → Linear(256 → 128) → ReLU → Linear(128 → 64) → ReLU
                                                                            → deep_out (64)
Wide:      cat(genome_cosine, future cross features)  — bypasses MLP
Head:      Linear(64 + |wide| → 1) → raw logit
```

### Implementation invariants

1. **Ranker owns its own `item_embedding_lookup`** (32-dim, `nn.Embedding(n_movies+1, 32)`). Shared across all 4 user pools and the item-side `item_id_emb`. Owned and trained by the ranker.
2. **LayerNorm on each of the 4 pools** — exactly as CG.
3. **Sub-tower init:** Xavier uniform `gain=0.1` on per-feature linears. Projection-style layers (deep MLP layers, head) at `gain=1.0`. Skipping this is what made CG's dot products collapse early in training.
4. **Year is bucketed and embedded** — not a raw scalar. Replace `release_year_norm` in `item_features` with `release_year_bin`.
5. **Item user-applied tag vector (306-dim) must be added to `item_features`** — currently absent. Required for `item_tag_tower` parity.
6. **Timestamp bin is fed into the model** — `timestamp_bin` is already in the parquet but currently unused.

### Warm-start initialization (preferred)

**Where shapes match, initialize ranker sub-tower weights from the corresponding v3 CG tower** instead of random. This is a one-time copy at construction; the tensors then live entirely in the ranker `state_dict` and train freely. **No runtime coupling** — no shared modules, no frozen tensors, no references back to CG after construction.

| Ranker tower | CG source (v3 prod state_dict) |
|---|---|
| `item_emb_lookup` (32-dim, n_movies+1) | `item_embedding_lookup.weight` |
| `item_id_emb` Linear+ReLU(32→32) | `item_embedding_tower.*` |
| `item_genre_emb` Linear(20→8) | `item_genre_tower.*` |
| `item_tag_emb` Linear(306→16) | `item_tag_tower.*` |
| `item_genome_emb` Linear(1128→32) | `item_genome_tag_tower.*` |
| `year_emb` Embedding(N, 8) | `year_embedding_tower.*` |
| `user_genome_ctx` Linear(1128→32) | `user_genome_context_tower.*` |
| `user_genre_emb` Linear(40→32) | `user_genre_tower.*` |
| `ts_emb` Embedding(N_ts, 4) | `timestamp_embedding_tower.*` |
| Pool LayerNorms (4× 32-dim) | `hist_full_norm`, `hist_liked_norm`, `hist_disliked_norm`, `hist_weighted_norm` |

The deep MLP, the wide head, and any new cross-feature weights stay random-init — they have no CG counterpart. Warm-start is a faster path to the CG-parity baseline; it does not change what the model can ultimately learn.

### What gets added on top of the baseline (cross features)

After the CG-parity baseline is built, the wide bypass adds *new* signal CG doesn't compute — interaction features like Weighted Genre Affinity, Era Bias, Rating Calibration, etc. (See "Cross-Feature Roadmap" below.) These are the ranker's actual edge.

---

## E2E Evaluation Rule (Golden Rule of Two-Stage Systems)

**Ranker_Hit@K ≤ CG_Recall@250 in production.** If CG didn't retrieve the label, the ranker never sees it.

In offline eval, we enforce this: if `cg_label_rank >= n_cand (250)`, rank is set to `n_cand + 1` (score = 0 for all metrics). Both CG baseline and ranker numbers use this same ceiling so comparison is apples-to-apples.

`cg_label_rank = n_cand` is ambiguous (could be rank 250 or rank > 250 — cap doesn't distinguish). Treated conservatively as "not found."

Eval outputs a `Recall@250` ceiling metric = fraction of examples where CG organically retrieved the label.

---

## Repo Structure

```
ranker/
├── precompute.py     ← CG scoring: builds ranker_candidates_{train,val}.parquet
├── dataset.py        ← RankerDataset, sample_batch() (Mixed Negative Sampling)
├── model.py          ← WideDeepRanker
├── train.py          ← BCE training loop
├── evaluate.py       ← NDCG@10, MRR, Hit@K, CG baseline, E2E ceiling
├── canary.py         ← side-by-side CG vs Ranker top-N for synthetic users
├── main.py           ← entry point
├── eval_results/     ← evaluate output saved here
└── canary_results/   ← canary output saved here

data/
├── ranker_candidates_train.parquet   ← 3,439,197 rows
├── ranker_candidates_val.parquet     ← 382,138 rows
└── ranker_movie_stats.parquet        ← global avg_rating, log1p(count) per movie
```

---

## Stage 0: Precompute (`ranker/precompute.py`) — DONE

**Configuration:** `TOP_K_CANDIDATES = 250` (1 label + 249 hard negatives per row)

**Train/val split:** 90/10 user-level split, `random.Random(42)`, same seed as CG splits.

**Parquet schema** (`ranker_candidates_{train,val}.parquet`):

| Column | Type | Description |
|--------|------|-------------|
| `user_id` | int | |
| `rollback_n` | int | position index in user's chronological history |
| `label_corpus_idx` | int | positive item corpus index |
| `neg_corpus_idxs` | list[int] | 249 hard negatives in CG score order |
| `cg_label_rank` | int | label's rank in full corpus (1-indexed, capped at 250) |
| `cg_label_score` | float | CG cosine similarity for the label, pre-masking |
| `cg_neg_scores` | list[float] | CG scores for the 249 negatives |
| `genome_cosine_label` | float | cosine(user_genome_pool, label_genome_vec) |
| `genome_cosine_negs` | list[float] | genome cosine for each negative |
| `user_avg_rating` | float | |
| `user_rating_count` | int | raw count (log1p applied in dataset.py) |
| `user_genre_ctx` | list[float] | 40-dim running genre avg/frac up to position N-1 |
| `X_history` | list[int] | padded corpus indices, MAX_HISTORY_LEN |
| `X_hist_ratings` | list[float] | debiased ratings, MAX_HISTORY_LEN |
| `timestamp_bin` | int | binned timestamp |

**Genome cosine computation (memory-efficient, per-batch):**
- Computed in `_score_candidates()` using `model.genome_context_buffer` (42 MB static buffer)
- Rating-weighted avg of raw 1128-dim genome vectors over watch history → normalize → dot with candidate genome → stored as scalar in parquet
- No per-example 1128-dim storage; intermediate tensors freed after each batch

**Fixes applied:**
- `nan_to_num(nan=-inf)` before `topk` — handles degenerate MPS scores causing non-unique indices
- Verification assertion uses `top_k - 1` (249) not hardcoded 99

---

## Stage 1: Target Architecture — CG-Parity WideDeepRanker

> **Status:** Current `ranker/model.py` is a partial implementation (only ~3 of CG's signals + genome bottleneck). The next experiment **builds the full CG-parity baseline** described above before any cross feature is added.

### Current implementation gap (what needs adding)

| CG signal | Current ranker | Action |
|---|---|---|
| `pool_full` (32) | missing | add — sum_pool over ranker `item_emb_lookup` + LayerNorm |
| `pool_liked` (32) | missing | add — same pool, masked to liked subset + LayerNorm |
| `pool_disliked` (32) | only as wide cosine | replace with full 32-dim pool concat + LayerNorm |
| `pool_weighted` (32) | missing | add — rating-weighted sum + LayerNorm |
| `user_genome_ctx` (32) | covered indirectly via `genome_cosine` only | add — `Linear(1128 → 32)` over rating-wtd avg genome |
| `user_genre_emb` (32) | raw 40-dim in concat | replace with `Linear(40 → 32)` |
| `ts_emb` (4) | missing | add — `Embedding(N_ts_bins, 4)` |
| `item_id_emb` (32) | missing | add — ranker `nn.Embedding(n_movies+1, 32)` shared with 4 user pools, then `Linear+ReLU(32→32)` for item-side use |
| `item_genre_emb` (8) | raw 20-dim onehot in concat | replace with `Linear(20 → 8)` |
| `item_tag_emb` (16) | **feature absent from `item_features`** | **add 306-dim user-applied tag vector to `item_features`**, then `Linear(306 → 16)` |
| `item_genome_emb` (32) | bottleneck `Linear(1128 → 64)` | reshape to `Linear(1128 → 32)` to match CG dim |
| `year_emb` (8) | raw scalar | bucket year into N bins, replace with `Embedding(N, 8)` |

### Original feature layout (for reference, will be replaced)

**User features** (42-dim):
- `user_genre_ctx`: 40-dim (2 × 20 genres: avg rating + watch fraction)
- `user_avg_rating`: scalar
- `user_count_log1p`: log1p(user_rating_count)

**Item features** (1151-dim base + 1 interaction = 1152-dim total):
- `genome_scores[0:1128]`: raw genome tag scores — compressed by genome bottleneck
- `genre_onehot[1128:1148]`: 20-dim
- `global_avg_rating[1148]`: scalar
- `global_rating_count[1149]`: log1p-scaled
- `release_year_norm[1150]`: normalized to [0,1]
- `genome_cosine[1151]`: user–item genome cosine similarity (appended by `sample_batch`)

**Interaction features:** `n_interaction_features = 1` (genome cosine only)
- `cg_label_score` / `cg_neg_scores` are loaded from parquet but **not used as model input**
- Reason: ranker must beat CG using independent signal before being given CG's own retrieval score

### `ranker/model.py` — WideDeepRanker

**Architecture:** Wide & Deep bridge

```
Item features (1152-dim)
│
├─ Genome bottleneck: genome[0:1128] → Linear(1128→64) + ReLU → 64-dim
│
├─ Deep path: cat(user(42), genome_bn(64), rest(24)) = 130-dim
│   └─ Linear(130→256) → ReLU → Linear(256→128) → ReLU → Linear(128→64) → ReLU
│   └─ deep_out: 64-dim
│
├─ Wide path: genome_cosine (last 1 dim of item_features) — bypasses all deep layers
│
└─ Head: cat(deep_out(64), genome_cosine(1)) → Linear(65→1) → raw logit
```

**Why Wide & Deep:**
The genome_cosine scalar has one learned weight in the head — a direct gradient path. Without the wide bypass, this single scalar must compete against 130 input dims for attention in the first hidden layer. Any useful signal can get washed out during backprop through many layers.

**Why genome bottleneck:**
Raw 1128-dim genome dominates the first hidden layer numerically — 1128 of 130 inputs (after bottleneck: 64 of 130). Without compression, genome swamps all other features.

**Config** (`get_config()` in `train.py`):
```python
hidden_dims:        [256, 128, 64]
genome_dim:         1128
genome_bottleneck_dim: 64
wide_dim:           1      # genome_cosine bypass
```

All architecture params saved in `_config.json` sidecar alongside each checkpoint.

### `ranker/dataset.py` — Mixed Negative Sampling (MNS)

`sample_batch()` uses 50% hard / 50% easy negatives:

- **Hard negatives** (50%): CG-retrieved candidates from parquet (CG thought these were relevant)
- **Easy negatives** (50%): uniform random corpus items (anchor global decision boundary)
- Easy negatives get genome_cosine = 0.0 (CG never scored them, no retrieval signal)

`CANDIDATES_PER_ROW = 250` → positive rate ≈ 1/250 = 0.004 per batch.

### `ranker/train.py` — Training Config

```python
lr:               1e-3  (Adam + cosine annealing, T_max = training_steps)
weight_decay:     0.0
batch_size:       1024
training_steps:   200_000
log_every:        10_000
grad_clip:        1.0
hidden_dims:      [256, 128, 64]
dropout:          0.0
popularity_alpha: 0.0    ← DISABLED (see below)
easy_neg_frac:    0.5
genome_dim:       1128
genome_bottleneck_dim: 64
wide_dim:         1
```

### `ranker/evaluate.py`

- NDCG@10, MRR, Hit@K for K = (1, 5, 10, 20, 50, 100, 150, 200, 250)
- CG baseline uses same E2E ceiling
- `Recall@250` printed as the production ceiling for all Hit@K metrics
- `evaluate_only()` auto-finds most recent checkpoint; reads arch params from JSON sidecar

---

## Popularity Alpha: CLOSED — Menon Correction Incompatible with BCE Ranker

**Status:** `popularity_alpha = 0.0` permanently. Do not revisit.

**Experiment:** Ran full 150k steps at alpha=0.5 (same as CG prod). Full-val E2E result:
- NDCG@10: **0.1146** (vs CG 0.1233, **−7.1%**) — 129 points *below* CG
- MRR: **0.1037** (vs CG 0.1106, −6.2%)
- Hit@50: −0.0528, Hit@100: −0.0413 — massive regression at all K
- Train/val loss gap: train=0.0210, **val=0.0337** (+60% gap) — training and inference objectives misaligned

**Why it fails in BCE but works in softmax (CG):**  
In softmax, the Menon logit adjustment shifts the full probability distribution over all ~9,375 items — the effect is normalized across the denominator. In BCE at 1:249 positive:negative ratio, each example is trained independently. Adding `alpha * log(count_i)` to a popular positive's logit during training forces the model to produce a *lower raw score* to minimize loss. Since popular items appear constantly as true positives, the model is trained to systematically undervalue exactly the items it needs to rank high. The result is NDCG collapse, not genre correction.

**Canary at alpha=0.5 (not shown separately):** Genre drift was *worse* than alpha=0.0, not better. The correction that works in CG's retrieval stage does not transfer to BCE reranking.

**How we address popularity drift instead:** Rich cross features that give the model enough explicit user-item compatibility signal to make genre-specific choices without a loss-function penalty. See ablation sequence.

---

## CG Score: Disabled Until Ranker Beats CG Without It

**Current status:** `n_interaction_features = 1` (genome cosine only, CG score excluded)

**Why disabled:** Using the CG score as a ranker feature is circular — the ranker can trivially learn "follow CG's score" and "beat" CG only because it's a wrapper around CG's own output. We want to prove the ranker adds independent value from item/user content features.

**Evidence:** The 50k run with CG score got NDCG@10=0.0885 immediately at step 5k. That fast convergence is the CG score doing all the work, not the ranker learning.

**Re-enable (as a final improvement):** After the ranker beats CG on content features alone, add CG score back as one more wide input. Set `n_interaction_features = 2` in `dataset.py`.

---

## Feature Roadmap

Two distinct categories — both required, in this order.

### A. CG-parity features (the baseline — must all be present together)

These mirror v3 CG's feature towers exactly. They are not optional and are not added one at a time — they ship as one cohesive baseline because partial CG parity makes ablations uninterpretable. (You can't tell whether a missing pool or a missing tag tower is what's holding the model back.)

| # | Feature | Path | Formula / Source | Dim |
|---|---------|------|------------------|-----|
| 1 | `pool_full` | Deep concat | `sum(item_emb_lookup[full_history])` + LayerNorm | 32 |
| 2 | `pool_liked` | Deep concat | `sum(item_emb_lookup[liked_history])` + LayerNorm | 32 |
| 3 | `pool_disliked` | Deep concat | `sum(item_emb_lookup[disliked_history])` + LayerNorm | 32 |
| 4 | `pool_weighted` | Deep concat | `rating-weighted sum(item_emb_lookup[full])` + LayerNorm | 32 |
| 5 | `user_genome_ctx` | Deep concat | `Linear(1128 → 32)` over rating-wtd avg raw genome | 32 |
| 6 | `user_genre_emb` | Deep concat | `Linear(40 → 32)` from `user_genre_ctx` | 32 |
| 7 | `ts_emb` | Deep concat | `Embedding(N_ts_bins, 4)` from `timestamp_bin` | 4 |
| 8 | `item_id_emb` | Deep concat | `item_emb_lookup → Linear+ReLU(32 → 32)` (lookup shared w/ user pools) | 32 |
| 9 | `item_genre_emb` | Deep concat | `Linear(20 → 8)` from `genre_onehot` | 8 |
| 10 | `item_tag_emb` | Deep concat | `Linear(306 → 16)` from user-applied tag vector | 16 |
| 11 | `item_genome_emb` | Deep concat | `Linear(1128 → 32)` from `genome_scores` | 32 |
| 12 | `year_emb` | Deep concat | `Embedding(N_year_bins, 8)` from bucketed year | 8 |
| 13 | `genome_cosine` | **Wide** bypass | precomputed cross feature, already in parquet | 1 |

Total deep concat: **292** | Total wide: **1** (pre-cross-features) | Head input: **64 + 1 = 65**

### B. Cross features (the ranker's edge — added on top of the baseline, one at a time)

These are signals the ranker has that CG doesn't compute. Each one is added in isolation and measured against the CG-parity baseline.

| Priority | Feature | Path | Formula / Source |
|----------|---------|------|------------------|
| 1 | **Weighted Genre Affinity** | Wide | `dot(user_genre_ctx_avg, item_genre_onehot) / sum(item_genre_onehot)` |
| 2 | **Era Bias** | Wide | `log1p(abs(user_median_release_year - item_release_year))` |
| 3 | **Genome Cosine Residual** | Wide | `genome_cosine - user_mean_genome_cosine` (replaces raw `genome_cosine`) |
| 4 | **Rating Calibration** | Wide | `user_avg_rating - item_global_avg_rating` |
| 5 | **Popularity Match** | Wide | `abs(user_avg_log_count - item_log_count)` |
| 6 | **Dislike Similarity** | Wide | `cosine(user_disliked_pool, item_id_emb)` |
| 7 | **Genre Intersection** | Wide | `Jaccard(user_top_genres, item_genres)` |
| 8 | **Genome Peak Match** | Wide | `max(user_genome_profile * item_genome_scores)` — the "one tag spark" |
| 9 | **Rating Entropy** | Deep | Shannon entropy of user rating distribution — calibrates criticality |
| 10 | **User Rating Variance** | Deep | Std dev of user ratings — opinionated vs indifferent |
| 11 | **Genre Diversity** | Deep | Entropy of history genre distribution — specialist vs generalist |
| 12 | **History Confidence** | Deep | `log1p(total_user_ratings)` — trust in latent signals |
| 13 | **CG Score** | Wide | Raw CG dot-product (re-enable last; see CG Score section above) |
| — | **Popularity Alpha** | — | Re-enable Menon α=0.5 in loss (see Popularity Alpha section above) |

### Wide-feature normalization

All wide features beyond `genome_cosine` and `Dislike Similarity` must be **normalized before concatenation** using fixed statistics registered as model buffers (not BatchNorm — train/eval batch composition differs, causing mismatch). Compute per-feature mean/std from a single pass over training data, register as `register_buffer(persistent=False)`.

```
Expected ranges (pre-normalization):
  Genre Affinity       [0, 1]        → Z-score
  Era Bias             [0, ~4.6]     → Z-score (log1p of year diff up to ~100yr)
  Genome Residual      [~−0.5, 0.5]  → Z-score (centered by construction)
  Rating Calibration   [~−4, +4]     → Z-score
  Popularity Match     [0, ~8]       → Z-score (log1p)
  Dislike Similarity   [−1, 1]       → no normalization needed
  Genome Peak Match    [0, 1]        → Z-score
  Genre Intersection   [0, 1]        → Z-score
genome_cosine          [−1, 1]       → no normalization needed
```

When all wide features are added:
```
wide = cat(genome_cosine(1), cross_features(N))   # (B, N+1)
head = Linear(64 + N + 1 → 1)
```

Initialize new wide-feature weights at 0.1 so they start with a small non-zero signal without swamping the deep path.

---

## Planned Ablation Sequence

Change **one thing per experiment**. Measure NDCG@10 delta before proceeding.

| Priority | Experiment | Phase | Change | Hypothesis |
|----------|------------|-------|--------|------------|
| ~~**Done**~~ | WideDeepRanker, liked-only labels, α=0 | baseline (legacy) | — | Established baseline: NDCG@10=0.0532 (43% of v3 CG) |
| ~~**Done**~~ | Recompute candidates with v3 CG | infra | Re-ran `ranker/precompute.py` with v3 checkpoint | V3 CG baseline: NDCG@10=0.1233, MRR=0.1106, Recall@250=72.65% |
| ~~**Done**~~ | **CG-parity baseline + 5 wide cross features** | **CG parity + cross** | Full 12 deep-concat features (292-dim) with v3 CG warm-start; wide bypass = `[genome_cosine, genre_affinity, era_gap, rating_cal, pop_match]` (n_cross=5); BCE on 1:249, lr=1e-3 cosine, MNS easy_neg_frac=0.5. | **Result: sampled NDCG@10=0.1299 (+0.0083, +6.8% vs CG sampled 0.1216).** Cleared CG cleanly with stable monotonic climb. Acts as the new starting line; further cross features add on top of this number. |
| ~~**Done**~~ | ~~Re-enable popularity alpha~~ | ~~regularization~~ | `popularity_alpha = 0.5` | **Result: NDCG@10=0.1146 (−7.1% vs CG). Menon correction is incompatible with BCE ranker training.** See "Popularity Alpha: CLOSED" section. |
| **Next** | **Genre Intersection** | cross #6 | Wide bypass — `Jaccard(user_top_genres, item_genres)` | Sharpest genre-precision signal available. Directly attacks the popularity gravity well by giving the model an explicit measure of how well an item's genre matches the user's demonstrated taste. |
| Then | **Dislike Similarity** | cross #7 | Wide bypass — `cosine(user_disliked_pool, item_id_emb)` | Veto signal: negative cosine = item is close to something the user disliked. Leverages the warm-started disliked pool directly. |
| Then | **Genome Peak Match** | cross #8 | Wide bypass — `max(user_genome_profile * item_genome_scores)` | "One tag spark" — finds whether a single dominant tag creates a match. Complementary to cosine (global similarity vs single strongest signal). |
| Then | **Genome Cosine Residual** | replaces cross #1 | Swap raw `genome_cosine` → `genome_cosine - user_mean_genome_cosine` | Centers genome cosine per-user, removing the average-user bias. Not additive — replaces one existing wide slot. |
| Then | CG score | retrieval signal | `n_interaction_features += 1` — re-enable CG score passthrough | Final boost: give ranker CG's own retrieval confidence |
| Later | DCN V2 cross network | architecture | Replace deep path with explicit cross layers | Feature crossing at scale once feature set is locked |

> **Note on the bundled CG-parity-plus-5-cross experiment.** The plan originally called for building pure CG-parity first (only `genome_cosine` in the wide bypass) and adding cross features one at a time. The actual code shipped with all 5 wide cross features wired from the start, so the first run measured CG-parity *plus* `[genre_affinity, era_gap, rating_cal, pop_match]` together. We're treating the +0.0083 result as the new starting line rather than re-running for an isolated baseline — pragmatic for a portfolio project, would matter more in production.

**Strategy for popularity drift:** Menon logit adjustment (alpha) is closed as a tool (see below). The path forward is richer cross features — give the model enough explicit user-item compatibility signal that the MLP learns to trust content over popularity rather than fighting popularity in the loss function. Genre Intersection, Dislike Similarity, and Genome Peak Match are the priority signals: together they give the model a sharp genre fingerprint, an explicit veto channel, and a tag-level "spark" match.

**Rule:** Beat CG on CG-parity + cross features before adding CG score. But do add CG's feature set — that is not the same as leaking the CG score.

**Industry principle:** In production two-stage systems (Google, YouTube, Netflix), the ranker always receives every signal the CG model had access to — and then adds more on top. A ranker that's informationally poorer than CG cannot be expected to beat CG regardless of how good its cross features are.

---

## No CG Coupling — Ranker Owns All Its Parameters

"Mirror CG features" means replicating the same *architecture and dimensions* CG uses, with ranker-owned parameters. **Warm-starting weights from CG at construction time is encouraged** (see "Warm-start initialization" above) — it gives the ranker a head start at the CG-parity baseline. After construction, those tensors live entirely in the ranker's `state_dict` and train freely.

What is NOT allowed: shared `nn.Module` objects, frozen CG tensors referenced at runtime, or any cross-graph reference back to CG after construction. Runtime coupling creates optimization conflicts (CG embeddings are trained for retrieval, not reranking), deployment brittleness, and invalid ablations.

**The rule:** The ranker's only runtime connection to CG is: (1) the candidate list output of CG (the 250 corpus indices), (2) precomputed features in the parquet (`genome_cosine`, CG scores), and (3) static feature data both models read from disk (genome scores, genre vectors, tag vectors). Init-time warm-start is fine. No shared modules at runtime.

---

## Diagnostic: If Ranker Still Can't Beat CG After Full CG Parity + Cross Features

If a ranker with all CG features + all cross features + CG score still can't beat CG's NDCG@10, the bug is in the training objective or architecture, not features:

- **BCE vs softmax:** BCE on random (row, candidate) tuples has very different gradient structure than softmax over the full candidate pool. Try evaluating with softmax loss over the 250-candidate pool.
- **MNS easy-negative fraction:** easy negatives may be too easy, wasting batch capacity. Try reducing `easy_neg_frac` from 0.5 to 0.2.
- **Negative sampling leakage:** verify hard negatives are genuinely hard (high CG score) and not contaminated by easy items appearing in the hard slot.
- **Architecture capacity:** `[256, 128, 64]` with 130-dim input may not be deep enough for the expanded feature set. Try `[512, 256, 128]`.

A ranker with strictly more information than CG that still loses is a bug, not a feature gap.

---

## Ablation Results Log

Runs before 2026-05-05 used v2 CG candidates. V3 CG baseline (E2E-adjusted, 250-cand): **NDCG@10=0.1233 · MRR=0.1106**. All future runs use v3 candidates.

| Run | Key config | Val NDCG@10 | Val MRR | Delta vs CG | Notes |
|-----|-----------|-------------|---------|-------------|-------|
| CG baseline (old parquets) | — | 0.0965 | 0.0871 | — | Recall@250=0.6737, all-movie labels |
| CG baseline v2 (liked-only parquets) | — | 0.0938 | 0.0845 | — | Old baseline; superseded by v3 |
| **CG baseline v3 (250-cand, E2E-adj)** | — | **0.1233** | **0.1106** | — | Full-val target |
| **CG baseline v3 (sampled, n=20k)** | — | **0.1216** | **0.1094** | — | Training-time sampled target |
| 50k, MLPRanker, α=0.5, CG score + genome cosine | alpha=0.5 | 0.0885 | 0.0798 | −0.0080 | Best at step 20k, degraded after. CG score leakage — ranker learned to follow CG, not beat it. |
| 200k, WideDeepRanker, α=0, genome cosine only, all-movie labels | alpha=0 | 0.0324 | 0.0348 | −0.0641 | Step 10k only — run superseded. Shows label noise effect. |
| 200k, WideDeepRanker, α=0, genome cosine only, liked-only labels | `ranker_mlp_alpha_0_20260505_111114.pth` | 0.0532 | 0.0521 | −0.0406 | Pre-CG-parity baseline. 56.7% of CG NDCG. |
| 80k partial, lr=1e-2, CG-parity + 5 cross, warm-start | (run abandoned) | 0.0702 peak | — | — | Aggressive lr washed out warm-start; NDCG oscillated ±0.05 between checkpoints. Killed at step 28k. |
| **150k, lr=1e-3, CG-parity + 5 cross, v3 warm-start** | `ranker_mlp_alpha_0_20260508_170820.pth` | **0.1287** (full-val E2E) · 0.1299 sampled | **0.1155** (full-val) · 0.1168 sampled | **+0.0054 E2E** (+4.4% vs CG) | First ranker to clear CG. Smooth monotonic climb. Full-val confirmed. **New starting line.** |
| **150k, same as above but α=0.5** | `ranker_mlp_alpha_05_20260508_193850.pth` | **0.1146** (full-val E2E) | **0.1037** | **−0.0087 (−7.1%)** | BCE + Menon correction is incompatible. Train/val loss gap +60% (0.0210→0.0337). Hit@50 −0.0528. Closed — do not retry alpha correction in ranker. |

### Run 0508 — full feature inventory

What was actually active in the +0.0083 result:

**Deep concat (292 dims, all warm-started where shapes match):**
- `pool_full`, `pool_liked`, `pool_disliked`, `pool_weighted` (4 × 32 + LayerNorm) — derived inside `user_embedding` from `(X_history, X_hist_ratings)` via `torch.where`
- `user_genome_ctx` (1128→32, pool-then-tower)
- `user_genre_emb` (40→32)
- `ts_emb` (Embedding lookup, 4-dim)
- `item_id_emb` (32-dim lookup → 32→32 + ReLU; lookup shared with all 4 pools)
- `item_genre_emb` (20→8)
- `item_tag_emb` (306→16)
- `item_genome_emb` (1128→32)
- `year_emb` (Embedding lookup, 8-dim)

**Wide bypass (5 cross features):**
1. `genome_cosine` — precomputed in parquet, cosine(user_genome_pool, item_genome)
2. `genre_affinity` — `dot(user_genre_ctx_avg, item_genre_onehot) / sum(item_genre_onehot)`
3. `era_gap` — `abs(user_mean_year_norm - item_year_norm)`
4. `rating_cal` — `user_avg_rating - item_global_avg_rating`
5. `pop_match` — `abs(user_count_log1p - item_log_count)`

**Training:** lr=1e-3 cosine→1e-5, batch=4096, 150k steps, BCE on 1:249, MNS easy_neg_frac=0.5, popularity_alpha=0, no dropout, grad_clip=1.0.

**Warm-start:** 27 of 36 ranker tensors loaded from `PROD_best_softmax_v2_popularity_alpha_05_20260505_182728.pth`. The 9 not warm-started are deep MLP (4 layers), head, and non-persistent buffers — random/fresh init.

### Training curve — Run 0508

| Step | NDCG@10 | MRR | Loss | Note |
|------|---------|-----|------|------|
| 2k | 0.0700 | 0.0645 | 0.0237 | already past prior 200k checkpoint's final |
| 8k | 0.0817 | 0.0734 | 0.0223 | smooth climb |
| 18k | 0.0927 | 0.0832 | 0.0218 | crossing prior CG sample baseline (0.0924) |
| 30k | 0.1020 | 0.0910 | 0.0218 | crossed 0.10 |
| 50k | 0.1082 | 0.0966 | 0.0216 | |
| 70k | 0.1170 | 0.1047 | 0.0213 | |
| 78k | 0.1201 | 0.1067 | 0.0212 | within 0.0015 of CG sample |
| 100k | ~0.124 | — | ~0.021 | (interpolated) crossed CG sample |
| 146k | 0.1299 | 0.1168 | 0.0209 | best checkpoint |
| 150k | 0.1295 | 0.1165 | 0.0210 | final (tied with 0.1299 best) |

Monotonic climb, near-zero oscillation. Loss dropped 0.0237 → 0.0210 in lockstep — actual learning, not just lucky rerankings. Cosine annealing to lr=1e-5 by end produced graceful saturation rather than a hard plateau.

### Run 0508 — full-val E2E results

| Metric | CG | Ranker | Delta |
|--------|----|--------|-------|
| NDCG@10 | 0.1233 | **0.1287** | +0.0054 (+4.4%) |
| MRR | 0.1106 | **0.1155** | +0.0049 (+4.4%) |
| Hit@1 | 0.0560 | 0.0590 | +0.0030 |
| Hit@10 | 0.2117 | 0.2192 | +0.0075 |
| Hit@100 | 0.5630 | 0.5868 | +0.0238 |
| Hit@150 | 0.6366 | 0.6628 | +0.0262 |
| Hit@250 | 0.7265 | 0.7265 | +0.0000 (ceiling) |

Pure reranking quality (label in CG set only, n=252,101): NDCG@10=**0.1772** vs CG 0.1697 (+4.4%). Gains compound at the tail (Hit@100+, Hit@150+) — ranker is meaningfully reordering the back half.

### Run 0508 — canary analysis

**Improvements vs CG:**
- **Children's**: Coco, Emperor's New Groove, Shrek 2, Frozen, Lilo & Stitch — clearly higher quality than CG's Cars 3 / Turbo / Cloudy 2 sequels
- **Superhero**: MCU quality slate (Doctor Strange, Guardians 2, Winter Soldier, Ragnarok, Infinity War) vs CG's lower-quality superhero picks
- **Anime**: Well-known classics (Spirited Away #1, Howl's #2, Ghost in Shell, Akira, Grave of Fireflies) — trades depth for coverage quality
- **Nick's list**: Matrix, Pulp Fiction, Shaun of Dead, Spirited Away, Fight Club, Eternal Sunshine, Pan's Labyrinth — excellent diverse cinephile list

**Regressions vs CG (popularity drift):**
- **Sci-Fi**: CG gives arthouse slow-burn (Stalker, Soylent Green, Fantastic Planet, Metropolis). Ranker: Star Wars ESB #2, Fight Club #3 (not sci-fi), Terminator #10, Jurassic Park #12, Total Recall #13
- **WW2**: Ranker: LotR Return of King #1, Gladiator #2, Inception #5, Troy #6, Last Samurai #8 — wrong genre entirely
- **Western**: Ranker: Raiders of the Lost Ark #1, Godfather #2, Apocalypse Now #3 — pure IMDb Top 250, no genre specificity
- **Musical**: Princess Bride, LotR Two Towers, Shrek — not musicals
- **War Movie**: Unforgiven #7 (western), Ford v Ferrari #8 (racing), Lincoln #13 (political drama)

**Root cause:** alpha=0 means no popularity penalty. When popular movies align with the user's genre (Superhero, Children's, Anime mainstream), the ranker excels. When popular movies are a different genre entirely (LotR for WW2, Godfather for Western), the ranker recommends those blockbusters anyway. The ranker has the right genre signals — it's just overridden by popularity for niche tastes.

**Implication for next experiment:** The ranker has now beaten CG (0.1287 > 0.1233), which is the plan's stated trigger for enabling popularity correction. Adding more cross features on a popularity-drifting foundation makes ablations harder to interpret. **Popularity alpha correction should be the next experiment** before adding cross features #6+.

### Historical (pre-250-candidate schema — not directly comparable)

| Run | α | Val NDCG@10 | Notes |
|-----|---|-------------|-------|
| MLP, α=0, static features only | 0 | 0.1422 | **Invalid** — popularity arbitrage, bad canary |
| MLP, α=1.0, static features | 1.0 | < CG | Both lose — user features too weak without genome context |
| MLP, α=0.5, static features | 0.5 | < CG | Same finding |

---

## Experiment Discipline Rules

1. **One change at a time.** Every run isolates exactly one variable. If two things change simultaneously, the result is uninterpretable.
2. **Beat CG on content features before adding CG score.** Ranker should earn its NDCG@10 > 0.1233 from independent signal.
3. **Beat CG before tuning alpha.** Don't chase canary quality until offline metrics confirm the ranker works.
4. **No src/ modifications.** Ranker is fully self-contained; CG code is read-only.
5. **No streamlit/export changes** until a model is verified better by eval + canary.
6. **`BCEWithLogitsLoss` only.** Never `BCELoss`. Never sigmoid in `WideDeepRanker.forward()`.
7. **Batch sampling is across all tuples** — not within rollback groups (avoids 249:1 imbalance dominating gradients).
8. **E2E ceiling always enforced** in both eval and CG baseline — apples-to-apples comparison only.
