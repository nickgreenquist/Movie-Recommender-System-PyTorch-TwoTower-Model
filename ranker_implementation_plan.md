# MovieLens Ranker: Implementation Plan

> **Note for future AI agents:** This doc is the plan, not the current code state. For uncommitted changes and recent edits, run `git status` and `git log --oneline -20 -- ranker/ ranker_implementation_plan.md`. Treat code as the source of truth for what is wired up; this doc captures intent and rationale.

---

## 1. Pipeline

Two-stage retrieve-and-rank:
1. **CG** (v3 softmax two-tower, 4-pool user, L2-normalized 128-dim) — retrieves top-250 candidates per rollback example.
2. **Ranker** (Wide & Deep MLP) — reranks the 250 candidates using richer features.

### CG baselines (target to beat)

| Model | Protocol | NDCG@10 | MRR | Hit@1 | Hit@10 | Recall@250 |
|-------|----------|---------|-----|-------|--------|-----------|
| v3 prod | Full-corpus rollback | 0.1296 | 0.1153 | 5.99% | 22.06% | 72.52% |
| **v3 prod** | **E2E-adjusted, 250-cand** | **0.1233** | **0.1106** | **5.60%** | **21.17%** | **72.65%** |
| v3 prod | E2E, sampled (n=20k) | 0.1216 | 0.1094 | — | — | — |

Every ranker experiment is measured against **NDCG@10 = 0.1233** (E2E full-val) or **0.1216** (sampled, training-time).

---

## 2. Core Principles

**CG-parity baseline first.** The ranker must contain every CG input/feature, projected through the same per-feature towers CG uses, with the same dimensions and the same initialization. Cross features come *on top* of this baseline — they are the differentiator, not a replacement. A ranker that sees less than CG cannot be expected to beat CG regardless of clever cross features. (Industry practice: Google/YouTube/Netflix rankers always receive every signal CG had, plus more.)

**Wide & Deep architecture.** Unlike CG (separate user/item towers + late L2-normalized dot product), the ranker concatenates all per-feature embeddings (user-side + item-side) into ONE vector that feeds a deep MLP. Cross features bypass the MLP and go straight to the head — giving each one a direct learned weight.

**No CG coupling at runtime.** The ranker owns its own copies of every parameter (its own `item_embedding_lookup`, genome tower, genre tower, etc.). It just *replicates the architecture* of each CG sub-tower. **Warm-starting weights from CG at init is encouraged** — it's a one-time copy at construction; the tensors then live entirely in the ranker `state_dict` and train freely. What is NOT allowed: shared `nn.Module` objects, frozen CG tensors referenced at runtime, cross-graph references after construction. The ranker's only runtime connection to CG is (1) the 250 candidate indices, (2) precomputed features in the parquet, (3) static feature data both models read from disk.

---

## 3. Architecture

### Deep concat layout (292 dims, mirrors v3 CG towers)

```
USER SIDE:
  pool_full         : 32   [LayerNorm]   sum(item_emb_lookup[full_history])
  pool_liked        : 32   [LayerNorm]   sum(item_emb_lookup[liked_history])
  pool_disliked     : 32   [LayerNorm]   sum(item_emb_lookup[disliked_history])
  pool_weighted     : 32   [LayerNorm]   rating-weighted sum(item_emb_lookup[full])
  user_genome_ctx   : 32                 Linear(1128 → 32) over rating-wtd avg genome
  user_genre_emb    : 32                 Linear(40 → 32) from user_genre_ctx
  ts_emb            :  4                 Embedding(N_ts_bins, 4)

ITEM SIDE:
  item_id_emb       : 32                 item_emb_lookup → Linear+ReLU(32 → 32)
  item_genre_emb    :  8                 Linear(20 → 8) from genre_onehot
  item_tag_emb      : 16                 Linear(306 → 16) from user-applied tag vector
  item_genome_emb   : 32                 Linear(1128 → 32) from genome_scores
  year_emb          :  8                 Embedding(N_year_bins, 8)

TOTAL: 4×32 + 32 + 32 + 4 + 32 + 8 + 16 + 32 + 8 = 292

Deep MLP:  Linear(292 → 256) → ReLU → Linear(256 → 128) → ReLU → Linear(128 → 64) → ReLU
           → deep_out (64)
Wide:      cat(genome_cosine, cross_features)  — bypasses MLP, direct to head
Head:      Linear(64 + |wide| → 1) → raw logit
```

### Implementation invariants

1. **Ranker owns its own `item_embedding_lookup`** (32-dim, `nn.Embedding(n_movies+1, 32)`). Shared across all 4 user pools and the item-side `item_id_emb`.
2. **LayerNorm on each of the 4 pools** — exactly as CG.
3. **Sub-tower init:** Xavier uniform `gain=0.1` on per-feature linears; `gain=1.0` on deep MLP + head. Skipping this is what made CG's dot products collapse early in training.
4. **Year is bucketed and embedded** — not a raw scalar.
5. **Timestamp bin** is fed into the model (via `ts_emb`).
6. **Item user-applied tag vector** (306-dim) lives in `item_features`.

### Warm-start mapping (init from v3 CG state_dict)

| Ranker tower | CG source |
|---|---|
| `item_emb_lookup` (32, n_movies+1) | `item_embedding_lookup.weight` |
| `item_id_emb` (32→32) | `item_embedding_tower.*` |
| `item_genre_emb` (20→8) | `item_genre_tower.*` |
| `item_tag_emb` (306→16) | `item_tag_tower.*` |
| `item_genome_emb` (1128→32) | `item_genome_tag_tower.*` |
| `year_emb` | `year_embedding_tower.*` |
| `user_genome_ctx` (1128→32) | `user_genome_context_tower.*` |
| `user_genre_emb` (40→32) | `user_genre_tower.*` |
| `ts_emb` | `timestamp_embedding_tower.*` |
| 4 pool LayerNorms | `hist_{full,liked,disliked,weighted}_norm` |

Deep MLP, head, and cross-feature weights stay random-init — no CG counterpart.

### Why Wide & Deep

The wide bypass gives each cross feature a single learned weight in the head — a direct gradient path. Without it, those scalar signals must compete against ~290 dims for attention in the first hidden layer and get washed out during backprop.

### Why item_genome at 32 dims (not raw 1128)

Raw 1128-dim genome would numerically dominate the first hidden layer (1128 of 290 inputs vs. compressed 32 of 290). Without compression, genome swamps all other features.

---

## 4. E2E Evaluation Rule (Golden Rule of Two-Stage Systems)

**Ranker_Hit@K ≤ CG_Recall@250.** If CG didn't retrieve the label, the ranker never sees it.

In offline eval: if `cg_label_rank >= n_cand (250)`, rank is set to `n_cand + 1` (score = 0 for all metrics). Both CG baseline and ranker numbers use this same ceiling — apples-to-apples comparison only. `cg_label_rank = n_cand` is ambiguous (could be rank 250 or > 250); treated conservatively as "not found."

Eval outputs `Recall@250` as the production ceiling.

---

## 5. Repo Structure

```
ranker/
├── precompute.py     ← CG scoring: builds ranker_candidates_{train,val}.parquet
├── dataset.py        ← RankerDataset, sample_batch (Mixed Negative Sampling)
├── model.py          ← WideDeepRanker
├── train.py          ← BCE training loop
├── evaluate.py       ← NDCG@10, MRR, Hit@K, CG baseline, E2E ceiling
├── canary.py         ← side-by-side CG vs Ranker top-N for synthetic users
├── main.py           ← entry point
├── eval_results/
└── canary_results/

data/
├── ranker_candidates_train.parquet   ← 3,439,197 rows
├── ranker_candidates_val.parquet     ← 382,138 rows
└── ranker_movie_stats.parquet        ← global avg_rating, log1p(count) per movie
```

---

## 6. Stage 0: Precompute — DONE

- `TOP_K_CANDIDATES = 250` (1 label + 249 hard negatives per row).
- Train/val: 90/10 user-level split, `random.Random(42)`, same seed as CG splits.
- Genome cosine computed per-batch (no per-example 1128-dim storage), stored as scalar in parquet.

### Parquet schema

| Column | Type | Description |
|--------|------|-------------|
| `user_id` | int | |
| `rollback_n` | int | position index in user's chronological history |
| `label_corpus_idx` | int | positive item corpus index |
| `neg_corpus_idxs` | list[int] | 249 hard negatives in CG score order |
| `cg_label_rank` | int | label's rank in full corpus (1-indexed, capped at 250) |
| `cg_label_score` | float | CG cosine for the label, pre-masking |
| `cg_neg_scores` | list[float] | CG scores for the 249 negatives |
| `genome_cosine_label` | float | cosine(user_genome_pool, label_genome_vec) |
| `genome_cosine_negs` | list[float] | genome cosine per negative |
| `user_avg_rating` | float | |
| `user_rating_count` | int | raw count (log1p applied in dataset.py) |
| `user_genre_ctx` | list[float] | 40-dim running genre avg/frac up to position N-1 |
| `X_history` | list[int] | padded corpus indices, MAX_HISTORY_LEN |
| `X_hist_ratings` | list[float] | debiased ratings, MAX_HISTORY_LEN |
| `timestamp_bin` | int | |

---

## 7. Training Stack

### Mixed Negative Sampling (MNS)

`sample_batch` uses 50% hard / 50% easy negatives:
- **Hard** (50%): CG-retrieved candidates from parquet (CG thought these were relevant).
- **Easy** (50%): uniform random corpus items (anchor global decision boundary).

Easy negatives get `genome_cosine = 0.0` (CG never scored them). Positive rate per batch ≈ 1/250 = 0.004.

### Training config (Run 0508, current canonical)

```
lr:               1e-3  (Adam + cosine annealing, T_max = training_steps)
weight_decay:     0.0
batch_size:       4096
training_steps:   150_000
grad_clip:        1.0
hidden_dims:      [256, 128, 64]
dropout:          0.0
popularity_alpha: 0.0           ← permanently disabled, see §8
easy_neg_frac:    0.5
```

Architecture params saved to `_config.json` sidecar alongside each checkpoint.

### Eval (`evaluate.py`)

- NDCG@10, MRR, Hit@K for K ∈ {1, 5, 10, 20, 50, 100, 150, 200, 250}.
- CG baseline computed with the same E2E ceiling.
- `Recall@250` printed as the production ceiling.
- `evaluate_only()` auto-finds the most recent checkpoint; reads arch params from the JSON sidecar.

---

## 8. Closed Gates

### Popularity Alpha — CLOSED, do not revisit

`popularity_alpha = 0.0` permanently.

**Experiment (Run 0508-α05):** 150k steps at α=0.5 (CG prod value). Full-val E2E:
- NDCG@10 = **0.1146** (vs CG 0.1233, **−7.1%**), MRR = 0.1037 (−6.2%), Hit@50 −0.0528.
- Train/val loss gap +60% (0.0210 → 0.0337) — training and inference objectives misaligned.
- Canary genre drift was *worse* than α=0, not better.

**Why it fails in BCE but works in softmax (CG):** In softmax, Menon's logit adjustment shifts the full probability distribution over all ~9,375 items — the effect is normalized by the denominator. In BCE at 1:249, each example is trained independently. Adding `alpha * log(count_i)` to a popular positive's logit forces the model to produce a *lower raw score* to minimize loss. Since popular items appear constantly as true positives, the model is trained to systematically undervalue exactly the items it needs to rank high. The result is NDCG collapse, not genre correction.

**How we address popularity drift instead:** rich cross features (Genre Intersection, Dislike Similarity, Genome Peak Match) that give the model explicit user-item compatibility signal — not a loss-function penalty.

### CG Score — gated, re-enable last

Currently disabled (`n_interaction_features = 1`, genome cosine only).

**Why disabled:** using the CG score as a ranker input is circular — the ranker can trivially learn to follow CG. We want the ranker to add *independent* value from content features. The earlier 50k run with CG score hit NDCG@10=0.0885 at step 5k — that fast convergence was CG doing the work, not the ranker learning.

**Re-enable:** only after the ranker beats CG on content features alone. Set `n_interaction_features = 2` in `dataset.py`.

---

## 9. Cross-Feature Roadmap

The cross-feature stack lives in the wide bypass (and a few in deep concat). Each one is a signal CG cannot meaningfully use at retrieval. **Add one at a time, measure NDCG@10 delta before proceeding** — except where bundled (see Run 0508 note below).

| # | Feature | Path | Formula | Status |
|---|---------|------|---------|--------|
| 1 | Genome Cosine | Wide | precomputed in parquet | **shipped (Run 0508)** |
| 2 | Weighted Genre Affinity | Wide | `dot(user_genre_ctx_avg, item_genre_onehot) / sum(item_genre_onehot)` | **shipped (Run 0508)** |
| 3 | Era Bias | Wide | `abs(user_mean_year_norm - item_year_norm)` | **shipped (Run 0508)** |
| 4 | Rating Calibration | Wide | `user_avg_rating - item_global_avg_rating` | **shipped (Run 0508)** |
| 5 | Popularity Match | Wide | `abs(user_count_log1p - item_log_count)` | **shipped (Run 0508)** |
| 6 | Genre Intersection | Wide | `Jaccard(user_top_genres, item_genres)` | next priority |
| 7 | Dislike Similarity | Wide | `cosine(user_disliked_pool, item_id_emb)` | next |
| 8 | Genome Peak Match | Wide | `max(user_genome_profile * item_genome_scores)` | next |
| 9 | Genome Cosine Residual | Wide | `genome_cosine - user_mean_genome_cosine` (replaces #1) | replace later |
| 10 | CG Score | Wide | raw CG dot-product (re-enable last; see §8) | gated |
| 11 | Rating Entropy / Variance / Genre Diversity / History Confidence | Deep | user-state scalars | not started |
| 12 | DCN V2 cross network | architecture | replace deep MLP with explicit cross layers | not started |

> **Code may be ahead of this table.** Check `n_cross_features` in `train.py` and `compute_cross_features` in `dataset.py` for what is actually wired.

### Wide-feature normalization

Wide features beyond `genome_cosine` and `Dislike Similarity` must be **normalized before concatenation** using fixed statistics registered as model buffers (not BatchNorm — train/eval batch composition differs). Compute per-feature mean/std from a single pass over training data, register with `register_buffer(name, tensor)` (persistent — the default). The stats MUST be saved with the checkpoint so eval/inference Z-score with training-time mean/std. **Do not pass `persistent=False`** — that excludes the buffer from `state_dict`, so the stats reset to constructor defaults on load and silently degrade the model.

```
Expected ranges (pre-normalization):
  Genre Affinity       [0, 1]        → Z-score
  Era Bias             [0, ~4.6]     → Z-score (log1p of year diff up to ~100yr)
  Genome Residual      [~−0.5, 0.5]  → Z-score (centered by construction)
  Rating Calibration   [~−4, +4]     → Z-score
  Popularity Match     [0, ~8]       → Z-score (log1p)
  Genome Peak Match    [0, 1]        → Z-score
  Genre Intersection   [0, 1]        → Z-score
  Dislike Similarity   [−1, 1]       → no normalization
  genome_cosine        [−1, 1]       → no normalization
```

Initialize new wide-feature weights at 0.1 — small non-zero signal without swamping the deep path.

### Strategy for popularity drift

α-based correction is closed (§8). The path forward is richer cross features — give the model enough explicit user-item compatibility signal that the MLP learns to trust content over popularity. Genre Intersection, Dislike Similarity, and Genome Peak Match are the priority signals: sharp genre fingerprint, explicit veto channel, single-tag spark match.

### Diagnostics if the next experiments stall

If adding cross features stops moving the needle, check before adding more:
- **Easy-negative fraction** — try reducing `easy_neg_frac` 0.5 → 0.2 (easy negs may be wasting batch capacity).
- **Hard-negative quality** — verify hard negatives have genuinely high CG scores.
- **Architecture capacity** — try `[512, 256, 128]` for the deep MLP.
- **Loss** — BCE on random tuples has very different gradient structure than softmax over the candidate pool; try softmax-over-250 loss.

---

## 10. Results Log

V3 CG baseline (E2E full-val): **NDCG@10 = 0.1233, MRR = 0.1106**. All runs since 2026-05-05 use v3 candidates.

| Run | Config | NDCG@10 | MRR | Δ vs CG | Notes |
|-----|--------|---------|-----|---------|-------|
| 50k MLP, α=0.5, CG score + genome cosine | (legacy) | 0.0885 | 0.0798 | −0.0080 | CG score leakage — ranker followed CG |
| 200k WideDeep, α=0, genome cosine only, liked-only | `..._20260505_111114.pth` | 0.0532 | 0.0521 | −0.0406 | Pre-CG-parity baseline |
| 80k partial, lr=1e-2, CG-parity + 5 cross | (abandoned) | 0.0702 peak | — | — | Aggressive lr washed out warm-start |
| **150k Run 0508**, lr=1e-3, CG-parity + 5 cross, v3 warm-start | `..._20260508_170820.pth` | **0.1287** | **0.1155** | **+0.0054 (+4.4%)** | **First ranker to clear CG.** New starting line. |
| Run 0508-α05 (α=0.5 ablation) | `..._20260508_193850.pth` | 0.1146 | 0.1037 | −0.0087 (−7.1%) | Closed alpha experiment (§8) |

### Run 0508 highlights

**Training curve:**

| Step | NDCG@10 | Loss | Note |
|------|---------|------|------|
| 2k   | 0.0700 | 0.0237 | already past prior 200k checkpoint's final |
| 100k | ~0.124 | ~0.021 | crossed CG sample baseline |
| 146k | **0.1299** | 0.0209 | best checkpoint |

Monotonic climb, near-zero oscillation, loss dropped in lockstep with NDCG — actual learning. Cosine annealing to lr=1e-5 produced graceful saturation.

**Full-val E2E:**

| Metric | CG | Ranker | Δ |
|--------|----|--------|---|
| NDCG@10 | 0.1233 | 0.1287 | +0.0054 (+4.4%) |
| MRR | 0.1106 | 0.1155 | +0.0049 (+4.4%) |
| Hit@1 | 0.0560 | 0.0590 | +0.0030 |
| Hit@10 | 0.2117 | 0.2192 | +0.0075 |
| Hit@100 | 0.5630 | 0.5868 | +0.0238 |
| Hit@150 | 0.6366 | 0.6628 | +0.0262 |
| Hit@250 | 0.7265 | 0.7265 | 0 (ceiling) |

Pure reranking (label in CG set only, n=252,101): NDCG@10 = **0.1772** vs CG 0.1697 (+4.4%). Gains compound at the tail — ranker meaningfully reorders the back half.

**Feature inventory:** all 12 CG-parity deep features (4 pools + 5 user/item towers warm-started), wide bypass = `[genome_cosine, genre_affinity, era_gap, rating_cal, pop_match]`. 27 of 36 ranker tensors warm-started from `PROD_best_softmax_v2_popularity_alpha_05_20260505_182728.pth`; deep MLP + head + buffers were fresh init.

> **Bundling caveat.** The plan called for pure CG-parity first (only `genome_cosine` in wide), then adding cross features one at a time. The actual code shipped with 5 wide features from the start, so Run 0508 measured CG-parity *plus* 4 extras together. We treat the +0.0054 as the new starting line rather than re-running for an isolated baseline — pragmatic for a portfolio project; would matter more in production.

### Canary analysis — Run 0508

**Wins:** Children's (Coco / Shrek 2 / Frozen vs CG's Cars 3 / Turbo); Superhero (MCU quality slate); Anime (well-known classics — Spirited Away, Howl's, Ghost in Shell); Nick's list (Matrix / Pulp Fiction / Pan's Labyrinth — diverse cinephile).

**Regressions (popularity drift):**
- Sci-Fi: ESB / Fight Club / Terminator vs CG's Stalker / Soylent Green / Metropolis.
- WW2: LotR / Gladiator / Inception / Troy — wrong genre entirely.
- Western: Raiders / Godfather / Apocalypse Now — pure IMDb Top 250.
- Musical: Princess Bride / LotR Two Towers — not musicals.
- War: Unforgiven (western) / Ford v Ferrari (racing) / Lincoln (political drama).

**Root cause:** α=0 means no popularity penalty. When popular movies align with the user's genre (Superhero, Children's, mainstream Anime), the ranker excels. When popular movies are a *different* genre, the ranker recommends them anyway. The genre signals are there — they're overridden by popularity for niche tastes. Next-priority cross features (Genre Intersection, Dislike Similarity, Genome Peak Match) directly target this.

### Pre-250-candidate runs (not directly comparable)

Earlier MLP runs with `α=0` on static features hit NDCG@10 = 0.1422 — but flagged as invalid (popularity arbitrage, bad canary). `α=0.5` and `α=1.0` runs both lost to CG.

---

## 11. Experiment Discipline Rules

1. **One change at a time.** Every run isolates exactly one variable.
2. **Beat CG on content features before adding CG score.** Earn NDCG@10 > 0.1233 from independent signal.
3. **No src/ modifications.** Ranker is fully self-contained; CG code is read-only.
4. **No streamlit/export changes** until a model is verified better by eval + canary.
5. **`BCEWithLogitsLoss` only.** Never `BCELoss`. Never sigmoid in `WideDeepRanker.forward()`.
6. **Batch sampling is across all tuples** — not within rollback groups (avoids 249:1 imbalance dominating gradients).
7. **E2E ceiling always enforced** in both ranker eval and CG baseline.
