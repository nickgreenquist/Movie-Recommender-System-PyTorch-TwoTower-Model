# MovieLens Ranker: Implementation Plan (Current State)

## Pipeline Overview

Two-stage retrieve-and-rank:
1. **CG** (v3 softmax two-tower, 4-pool user tower, L2-normalized, 128-dim) — retrieves top-250 candidates per rollback example
2. **Ranker** (Wide & Deep MLP) — reranks the 250 candidates using richer features

> **Note:** Ranker candidates were precomputed with v2 CG. Re-running `ranker/precompute.py` with the v3 checkpoint will give a stronger CG baseline and harder negatives. Do this before the next training run.

CG baseline (val, 250-candidate pool, E2E-adjusted, liked-only labels): **NDCG@10=0.0938 · MRR=0.0845**

Every ranker experiment is measured against this number.

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

## Stage 1: Current Architecture — WideDeepRanker

### Feature Layout

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

## Popularity Alpha: Disabled Until Ranker Beats CG

**Current status:** `popularity_alpha = 0.0` (Menon debiasing OFF)

**Why disabled:** Without a strong user-item interaction signal (CG score), the Menon bias at α=0.5 overwhelms weak content signals early in training:
- Hard CG negatives (items CG also liked) tend to be popular
- α=0.5 adds a large upward boost to popular items' logits during training → model is pushed hard to score popular items DOWN
- Labels are also often popular → model learns "popular = bad" → labels score below random at inference
- Result: NDCG@10 < 0.01, below random chance (~0.023 for 250-candidate pool)

**Principle:** The ranker needs to learn clean content-based personalization signals first. Re-enable by setting `popularity_alpha = 0.5` in `get_config()` after beating CG (NDCG@10 > 0.0938).

---

## CG Score: Disabled Until Ranker Beats CG Without It

**Current status:** `n_interaction_features = 1` (genome cosine only, CG score excluded)

**Why disabled:** Using the CG score as a ranker feature is circular — the ranker can trivially learn "follow CG's score" and "beat" CG only because it's a wrapper around CG's own output. We want to prove the ranker adds independent value from item/user content features.

**Evidence:** The 50k run with CG score got NDCG@10=0.0885 immediately at step 5k. That fast convergence is the CG score doing all the work, not the ranker learning.

**Re-enable (as a final improvement):** After the ranker beats CG on content features alone, add CG score back as one more wide input. Set `n_interaction_features = 2` in `dataset.py`.

---

## Feature Engineering Roadmap

The whole point of a ranking model is **cross features** — signals that capture the interaction between a specific user and a specific item that neither pure user nor pure item features can express alone. Each feature below is assigned to the **wide bypass** (direct skip to head) or the **deep path** (through MLP).

**Wide path** — sparse, interpretable, scalar interactions. One learned weight each in the head. Prevents dilution by deep layer competition.

**Deep path** — dense features that benefit from non-linear mixing (user behavioral stats, item side signals).

All wide features beyond genome_cosine must be **normalized before concatenation** using fixed statistics registered as model buffers (not BatchNorm — train/eval batch composition differs, causing mismatch). Compute per-feature mean/std from a single pass over training data, register as `register_buffer(persistent=False)`.

```
Expected ranges (pre-normalization):
  Genre Affinity       [0, 1]        → Z-score (skewed distribution)
  Era Bias             [0, ~4.6]     → Z-score (log1p of year diff up to ~100yr)
  Genome Residual      [~−0.5, 0.5]  → Z-score (centered by construction)
  Rating Calibration   [~−4, +4]     → Z-score
  Popularity Match     [0, ~8]       → Z-score (log1p)
  Dislike Similarity   [−1, 1]       → no normalization needed
  Genome Peak Match    [0, 1]        → Z-score
  Genre Intersection   [0, 1]        → Z-score
genome_cosine (slot 0) is already in [−1, 1] — no normalization needed.
```

When all wide features are added, the head becomes:
```
wide = cat(genome_cosine(1), cross_features(N))   # (B, N+1)
logit = head(cat(deep_out(64), wide))              # Linear(65+N → 1)
```

Initialize new wide feature weights at 0.1 so they start with a small non-zero signal without swamping the deep path's learned representation.

### Complete Feature Table

| Priority | Feature | Path | Formula / Source | Status |
|----------|---------|------|-----------------|--------|
| — | **Genome Cosine** | Wide | `cosine(user_genome_pool, item_genome_vec)` — from parquet | **Done (baseline)** |
| 1 | **Weighted Genre Affinity** | Wide | `dot(user_genre_ctx_avg, item_genre_onehot) / sum(item_genre_onehot)` | Pending |
| 2 | **Era Bias** | Wide | `log1p(abs(user_median_release_year - item_release_year))` | Pending |
| 3 | **Genome Cosine Residual** | Wide | `genome_cosine - user_mean_genome_cosine` (replaces raw genome_cosine) | Pending |
| 4 | **Rating Calibration** | Wide | `user_avg_rating - item_global_avg_rating` | Pending |
| 5 | **Popularity Match** | Wide | `abs(user_avg_log_count - item_log_count)` | Pending |
| 6 | **Dislike Similarity** | Wide | `cosine(user_disliked_pool, item_embedding)` — ranker-owned embedding | CG parity |
| 7 | **Genre Intersection** | Wide | Jaccard(user_top_genres, item_genres) — prevents embedding dilution | CG parity |
| 8 | **User Genome Context** | Deep | Rating-weighted avg of raw genome scores over X_history → ranker Linear(1128→64) | CG parity |
| 9 | **User History Pool** | Deep | Rating-weighted avg of ranker item embeddings over X_history → 32-dim | CG parity |
| 10 | **Item ID Embedding** | Deep | Ranker `nn.Embedding(n_movies, 32)` lookup for candidate (shared table with pool) | CG parity |
| 11 | **Genome Peak Match** | Deep | `max(user_genome_profile * item_genome_scores)` — the "one tag spark" | Later |
| 12 | **Rating Entropy** | Deep | Shannon entropy of user rating distribution — calibrates criticality | Later |
| 13 | **User Rating Variance** | Deep | Std dev of user ratings — distinguishes opinionated critics from indifferent raters | Later |
| 14 | **Genre Diversity** | Deep | Entropy of history genre distribution — specialist vs generalist | Later |
| 15 | **History Confidence** | Deep | `log1p(total_user_ratings)` — how much to trust latent signals | Later |
| 16 | **CG Score** | Wide | Raw CG dot-product — see CG Score section above | After beating CG |
| 17 | **Popularity Alpha** | — | Menon logit adjustment α=0.5 — see Popularity Alpha section above | After beating CG |

### No CG Coupling — Ranker Owns All Its Parameters

"Porting CG features" means replicating the same *types* of signals CG computes, using ranker-owned parameters trained from scratch. It does not mean loading CG's `item_embedding_lookup` or freezing CG tensors into the ranker graph. Coupling creates optimization conflicts (CG embeddings are trained for retrieval, not reranking), deployment brittleness, and invalid ablations.

**The rule:** The ranker's only connection to CG is: (1) the candidate list output of CG (the 250 corpus indices), (2) precomputed features in the parquet (`genome_cosine`, CG scores), and (3) static feature data both models read from disk (genome scores, genre vectors). No shared `nn.Module`, no shared `state_dict` tensors.

---

## Planned Ablation Sequence

Change **one thing per experiment**. Measure NDCG@10 delta before proceeding.

| Priority | Experiment | Phase | Change | Hypothesis |
|----------|------------|-------|--------|------------|
| ~~**Done**~~ | WideDeepRanker, liked-only labels, α=0 | baseline | — | Established baseline: NDCG@10=0.0532 (56.7% of CG) |
| **Next** | Recompute candidates with v3 CG | infra | Re-run `ranker/precompute.py` with v3 checkpoint | Stronger CG negatives; establishes new CG baseline |
| **Then** | Weighted Genre Affinity | cross #1 | `dot(user_genre_ctx_avg, item_genre_onehot) / sum(item_genre_onehot)` → wide | Modal genre kill switch — strongest cheap signal |
| **Then** | Era Bias | cross #2 | `log1p(abs(user_median_year - item_year))` → wide | Look-and-feel fit orthogonal to genre |
| **Then** | Genome Cosine Residual | cross #3 | `genome_cosine - user_mean_cosine` replaces raw genome_cosine | Denoises existing signal by subtracting user baseline |
| **Then** | Rating Calibration | cross #4 | `user_avg_rating - item_global_avg` → wide | Hidden gem vs guilty pleasure signal |
| **Then** | Popularity Match | cross #5 | `abs(user_avg_log_count - item_log_count)` → wide | Head-seeker vs tail-hunter alignment |
| **Then** | User Genome Context | CG parity | Rating-weighted avg of raw genome over X_history → ranker Linear(1128→64) | CG's strongest user content signal |
| **Then** | User History Pool + Item ID Embedding | CG parity | Ranker-owned embeddings for watch history pool and candidate | CG's primary CF signal; add together (they share one embedding table) |
| **Then** | Dislike Similarity + Genre Intersection | cross #6-7 | cosine(disliked_pool, item_emb); Jaccard(user_genres, item_genres) | Veto signal + genre precision |
| **Then** | Re-enable popularity alpha | regularization | `popularity_alpha = 0.5` | Prevent popular drift once content signal is established |
| **Then** | CG score | retrieval signal | `n_interaction_features = 2` — re-enable CG score passthrough | Final boost: give ranker CG's own retrieval confidence |
| **Later** | DCN V2 cross network | architecture | Replace deep path with explicit cross layers | Feature crossing at scale once feature set is locked |

**Rule:** Beat CG on independent content features before adding CG score. But do add CG's feature set — that is not the same as leaking the CG score.

**Industry principle:** In production two-stage systems (Google, YouTube, Netflix), the ranker always receives every signal the CG model had access to — and then adds more on top. A ranker that's informationally poorer than CG cannot be expected to beat CG regardless of how good its cross features are.

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

CG baseline shifted at liked-only parquet rerun: **NDCG@10=0.0938 · MRR=0.0845** (harder task — liked-only val labels). All runs below use this baseline unless noted.

| Run | Key config | Val NDCG@10 | Val MRR | Delta vs CG | Notes |
|-----|-----------|-------------|---------|-------------|-------|
| CG baseline (old parquets) | — | 0.0965 | 0.0871 | — | Recall@250=0.6737, all-movie labels |
| CG baseline (liked-only parquets) | — | 0.0938 | 0.0845 | — | Harder task; use this for all future comparisons |
| 50k, MLPRanker, α=0.5, CG score + genome cosine | alpha=0.5 | 0.0885 | 0.0798 | −0.0080 | Best at step 20k, degraded after. CG score leakage — ranker learned to follow CG, not beat it. |
| 200k, WideDeepRanker, α=0, genome cosine only, all-movie labels | alpha=0 | 0.0324 | 0.0348 | −0.0641 | Step 10k only — run superseded. Shows label noise effect. |
| **200k, WideDeepRanker, α=0, genome cosine only, liked-only labels** | `ranker_mlp_alpha_0_20260505_111114.pth` | **0.0532** | **0.0521** | **−0.0406** | Steady improvement 10k→190k (0.0440→0.0532). Still improving at end. 56.7% of CG NDCG. |

### Training curve (liked-only WideDeep baseline)

| Step | NDCG@10 | MRR | Notes |
|------|---------|-----|-------|
| 10k | 0.0440 | 0.0440 | |
| 50k | 0.0471 | 0.0460 | |
| 100k | 0.0491 | 0.0482 | |
| 150k | 0.0522 | 0.0513 | |
| 190k | 0.0532 | 0.0521 | best checkpoint |
| 200k | 0.0532 | — | final (tied best) |

Monotonically improving through end of run with no sign of overfitting — more steps or cross features are the right next move, not early stopping.

### Historical (pre-250-candidate schema — not directly comparable)

| Run | α | Val NDCG@10 | Notes |
|-----|---|-------------|-------|
| MLP, α=0, static features only | 0 | 0.1422 | **Invalid** — popularity arbitrage, bad canary |
| MLP, α=1.0, static features | 1.0 | < CG | Both lose — user features too weak without genome context |
| MLP, α=0.5, static features | 0.5 | < CG | Same finding |

---

## Experiment Discipline Rules

1. **One change at a time.** Every run isolates exactly one variable. If two things change simultaneously, the result is uninterpretable.
2. **Beat CG on content features before adding CG score.** Ranker should earn its NDCG@10 > 0.0938 from independent signal.
3. **Beat CG before tuning alpha.** Don't chase canary quality until offline metrics confirm the ranker works.
4. **No src/ modifications.** Ranker is fully self-contained; CG code is read-only.
5. **No streamlit/export changes** until a model is verified better by eval + canary.
6. **`BCEWithLogitsLoss` only.** Never `BCELoss`. Never sigmoid in `WideDeepRanker.forward()`.
7. **Batch sampling is across all tuples** — not within rollback groups (avoids 249:1 imbalance dominating gradients).
8. **E2E ceiling always enforced** in both eval and CG baseline — apples-to-apples comparison only.
