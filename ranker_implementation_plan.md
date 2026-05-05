# MovieLens Ranker: Implementation Plan (Current State)

## Pipeline Overview

Two-stage retrieve-and-rank:
1. **CG** (v2 softmax two-tower, L2-normalized, 128-dim) — retrieves top-250 candidates per rollback example
2. **Ranker** (Wide & Deep MLP) — reranks the 250 candidates using richer features

CG baseline (val, 250-candidate pool, E2E-adjusted): **NDCG@10=0.0965 · MRR=0.0871 · Hit@10=0.1717 · Recall@250=0.6737**

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
- Re-enable by setting `n_interaction_features = 2` in `dataset.py`

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
The genome_cosine scalar has one learned weight in the head — a direct gradient path. Without the wide bypass, this single scalar must compete against 130 input dims for attention in the first hidden layer (256 neurons). Any useful signal can get washed out during backprop through many layers before reaching the output weight.

**Why genome bottleneck:**
Raw 1128-dim genome dominates the first hidden layer numerically — 1128 of 130 inputs (after bottleneck: 64 of 130). Without compression, genome swamps all other features.

**Config** (`get_config()` in `train.py`):
```python
hidden_dims:        [256, 128, 64]
genome_dim:         1128
genome_bottleneck_dim: 64
wide_dim:           1      # genome_cosine bypass
```

All architecture params saved in `_config.json` sidecar alongside each checkpoint. `evaluate_only()` and `canary.py` read these for correct model reconstruction.

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

**Principle:** The ranker needs to learn clean content-based personalization signals first. Popularity correction can be layered on top once the ranker demonstrates it can beat CG without it.

**Re-enable:** Set `popularity_alpha = 0.5` in `get_config()` after beating CG baseline (NDCG@10 > 0.0965).

---

## CG Score: Disabled Until Ranker Beats CG Without It

**Current status:** `n_interaction_features = 1` (genome cosine only, CG score excluded)

**Why disabled:** Using the CG score as a ranker feature is circular — the ranker can trivially learn "follow CG's score" and "beat" CG only because it's a wrapper around CG's own output. We want to prove the ranker adds independent value from item/user content features.

**Evidence:** The 50k run with CG score got NDCG@10=0.0885 immediately at step 5k. That fast convergence is the CG score doing all the work, not the ranker learning.

**Re-enable (as a final improvement):** After the ranker beats CG on content features alone, add CG score back as one more input. Set `n_interaction_features = 2` in `dataset.py` and uncomment the CG score block in `sample_batch()` and `evaluate.py`.

---

## Planned Ablation Sequence

The whole point of a ranking model is **cross features** — signals that capture the interaction between a specific user and a specific item that neither pure user nor pure item features can express alone. Prioritize cross features before adding more user-side or item-side features in isolation.

Change ONE thing per experiment. Measure NDCG@10 delta before proceeding.

### Cross-Feature Priority Ranking

Five cross features ranked by expected NDCG impact on MovieLens. All are computed on-the-fly in `sample_batch` / `evaluate.py` / `canary.py` — no precompute rerun needed unless noted.

| Rank | Feature | Importance | Formula | Rationale |
|------|---------|------------|---------|-----------|
| 1 | **Weighted Genre Affinity** | Critical | `dot(user_genre_ctx[avg_slot], item_genre_vec) / sum(item_genre_vec)` | MovieLens users are highly modal — if they hate Horror, they really hate Horror. Acts as a categorical kill switch. Normalizing by genre count prevents multi-genre movies from dominating. |
| 2 | **Era Bias (Year Gap)** | High | `log1p(abs(user_median_release_year - item_release_year))` | Captures look-and-feel fit. A user who loves 70s grain is unlikely to enjoy 2020s CGI-heavy films regardless of genre match. `user_median_release_year` computed from X_history via `fs.movieId_to_year`. |
| 3 | **Genome Cosine Residual** | High | `genome_cosine - user_mean_genome_cosine` | Subtracts the user's mean cosine similarity across the corpus to isolate true item-specific relevance from "user generally likes everything" baseline noise. `user_mean_genome_cosine` precomputed once per user batch. |
| 4 | **Rating Calibration** | Medium | `user_avg_rating - item_global_avg` | Positive = hidden gem (user is harsher than crowd, so item is underrated). Negative = guilty pleasure (user rates higher than crowd). Captures alignment between user pickiness and item quality perception. |
| 5 | **Popularity Match** | Medium | `abs(user_avg_log_count - item_log_count)` | Identifies head-seeker vs tail-hunter. High value = mismatch (blockbuster fan shown indie, or vice versa). `user_avg_log_count` = mean `log1p(global_rating_count)` over watch history. |

### Wide Bypass Architecture for Cross Features

All five cross features are passed through the **wide bypass** (skip-connection directly to the head), not through the deep MLP. This prevents dilution in the first hidden layer where they'd compete against 130+ input dims.

```
interaction_features = [genre_affinity, era_gap, genome_residual, rating_cal, pop_match]  # (B, 5)

deep_out   = MLP(user + genome_bn + rest)      # (B, 64)
wide       = cat(genome_cosine, interaction_features)  # (B, 6)  ← existing + 5 new
combined   = cat(deep_out, wide)                # (B, 70)
logit      = head(combined)                     # Linear(70, 1)
```

Weight initialization for the 5 new wide features: `nn.init.constant_(head.weight[:, 65:], 0.1)` — gives them a non-zero starting signal without swamping the deep path's learned representation.

### Feature Normalization (Required Before Head)

All 5 cross features **must be normalized before concatenation into the wide vector**. Without normalization, the head weight for a large-magnitude feature (e.g. era_gap in years) will dominate gradient updates over a unit-range feature (e.g. genre_affinity ∈ [0,1]) — gradient competition between wide features is unfair.

**Normalization method: fixed statistics registered as model buffers (not BatchNorm).**

Do NOT use `nn.BatchNorm1d` on the wide block. BatchNorm has different behavior at train vs eval time (running stats vs batch stats), which introduces a train/inference mismatch in a ranking model where batch composition (hard + easy negatives) differs from eval (full candidate pool). Fixed statistics are stable and deterministic at both train and eval time.

**Implementation:**
1. After all 5 features are added and their value ranges are understood from one pass over the training set, compute per-feature `mean` and `std` (for signed features) or `min`/`max` (for non-negative bounded features).
2. Register as non-trainable buffers in `WideDeepRanker`:
   ```python
   self.register_buffer('wide_mean', torch.zeros(5))   # exclude genome_cosine — already in [-1,1]
   self.register_buffer('wide_std',  torch.ones(5))
   ```
3. Apply in `forward()` before concatenation:
   ```python
   interact_norm = (interaction_features - self.wide_mean) / (self.wide_std + 1e-8)
   wide = torch.cat([genome_cosine_vec, interact_norm], dim=1)  # (B, 6)
   ```
4. Stats are computed once from training data (e.g. a 50k-sample pass in `train.py`) and stored in the checkpoint via the buffer mechanism — no separate file needed.

**Expected ranges per feature (pre-normalization):**

| Feature | Range | Method |
|---------|-------|--------|
| Genre Affinity | [0, 1] (already normalized by genre count) | Z-score still recommended — distribution is skewed |
| Era Gap | [0, ~4.6] after log1p (year diffs up to ~100 years) | Z-score |
| Genome Cosine Residual | roughly [−0.5, 0.5] (centered by construction) | Z-score |
| Rating Calibration | roughly [−4, +4] (ratings ∈ [0.5, 5], mean ~3.5) | Z-score |
| Popularity Match | [0, ~8] after log1p | Z-score |

`genome_cosine` (the existing wide feature, slot 0) is already in [−1, 1] and needs no additional normalization.

### Industry Principle: CG Features First, Then Cross Features on Top

**Industry rankers are built on top of the retrieval model's own feature set, not instead of it.**

In production two-stage systems (Google, YouTube, Netflix), the ranker always receives every signal the CG model had access to — and then adds more on top. The ranker's advantage is not that it ignores CG's features; it's that it can use them more expressively (cross products, non-linear interactions, richer user representations) in a computation budget that wouldn't be feasible at CG scale.

Our ranker currently knows far less than CG about each user:
- **CG user representation:** rating-weighted pool of 128-dim projected item embeddings + genome pool + genre tower + timestamp
- **Ranker user representation:** 40-dim genre ctx (averages only) + scalar avg rating + scalar log count

A ranker that's informationally poorer than CG cannot be expected to beat CG, no matter how good its cross features are. The strategy is:

1. **Phase 1 (cross features):** Add the 5 cross features to test the architecture's ability to learn user×item interactions — these are cheap (no precompute rerun) and isolate the cross-signal hypothesis cleanly.
2. **Phase 2 (CG parity):** Add CG's own feature set to the ranker so it has at least as much information as CG. A ranker with strictly more information than CG that still loses is a bug, not a feature gap.
3. **Phase 3 (combined):** Cross features + CG features together — this is the full ranker as industry would build it.

### No CG Coupling — Ranker Owns All Its Parameters

**The ranker does not share weights, embeddings, or towers with the CG model. Ever.**

"Porting CG features" means replicating the same *types* of signals CG computes, using ranker-owned parameters trained from scratch. It does not mean loading CG's `item_embedding_lookup`, extracting CG's linear layers, or freezing CG tensors into the ranker graph. Coupling would create:

- **Optimization conflict:** CG's embeddings are trained for retrieval (maximize dot product between user and item towers). The ranker needs item representations optimized for reranking (maximize BCE over the 250-candidate pool). Same embedding table cannot serve both objectives.
- **Deployment brittleness:** upgrading CG requires redeploying the ranker. They must evolve independently.
- **False ablations:** if the ranker uses CG's learned embeddings, any "CG parity" experiment actually measures the quality of CG's learned representations, not the ranker's ability to learn from raw features.

**The rule:** The ranker's only connection to CG is: (1) the candidate list output of CG (the 250 corpus indices), (2) the raw precomputed features in the parquet (`genome_cosine`, CG scores, etc.), and (3) static feature data that both models read from disk (genome scores, genre vectors). No shared `nn.Module`, no shared `state_dict` tensors.

### CG Features to Port to the Ranker

All implemented with ranker-owned parameters. Raw feature data (genome scores, genre vectors, year) comes from the feature store / parquet — the same source CG reads at training time. No CG checkpoint loading needed.

| CG Feature | Ranker-Owned Implementation | Data Source | Why It Matters |
|------------|----------------------------|-------------|----------------|
| **User history pool** | Ranker's own `nn.Embedding(n_movies, 32)` lookup; rating-weighted avg over `X_history` → 32-dim | `X_history`, `X_hist_ratings` from parquet | CF signal: user's taste in embedding space, learned by the ranker independently |
| **User genome context** | Rating-weighted avg of raw genome scores (1128-dim) over `X_history` → ranker's own `Linear(1128→64)` → 64-dim | `item_features[:, 0:1128]` (already in memory) + `X_history` | Content texture of watch history; genome scores are static data, projection is ranker-trained |
| **Item ID embedding** | Same `nn.Embedding(n_movies, 32)` shared with user pool (same design as CG — item and history pool share one table) | candidate corpus index | Candidate CF signal, trained by ranker |
| **Item genome tower** | Ranker's own `Linear(1128→32)` applied to `item_features[:, 0:1128]`; replaces the current genome bottleneck | `item_features[:, 0:1128]` (already in memory) | Gives genome a dedicated trained projection; same data as bottleneck, different (ranker-owned) weights |

Note: `item_features[:, 0:1128]` is already loaded into the ranker's item feature matrix — the genome scores are static data derived from `genome-scores.csv`, not learned by CG. Using them does not create any CG coupling.

### Ablation Sequence

| Priority | Experiment | Phase | Change | Hypothesis |
|----------|------------|-------|--------|------------|
| **Now** | WideDeepRanker, liked-only labels, α=0 | baseline | Current (liked-only parquets) | Establish new baseline with cleaner labels |
| **Next** | Weighted Genre Affinity | cross #1 | `dot(user_genre_ctx_avg, item_genre_vec) / sum(item_genre_vec)` → wide bypass | Modal genre kill switch — strongest cheap signal |
| **Then** | Era Bias | cross #2 | `log1p(abs(user_median_year - item_year))` → wide bypass | Look-and-feel fit orthogonal to genre |
| **Then** | Genome Cosine Residual | cross #3 | `genome_cosine - user_mean_cosine` replaces raw genome_cosine | Denoises existing signal by subtracting user baseline |
| **Then** | Rating Calibration | cross #4 | `user_avg_rating - item_global_avg` → wide bypass | Hidden gem vs guilty pleasure signal |
| **Then** | Popularity Match | cross #5 | `abs(user_avg_log_count - item_log_count)` → wide bypass | Head-seeker vs tail-hunter alignment |
| **Then** | User genome context | CG parity | rating-weighted avg of raw genome over X_history → Linear(1128→64) → user concat | CG's +8% MRR feature; ranker must have at least this |
| **Then** | User history pool | CG parity | rating-weighted avg of item ID embeddings over X_history → user concat | CG's primary CF user signal |
| **Then** | Item ID embedding | CG parity | `nn.Embedding(n_movies, 64)` lookup for candidate → item concat | CG's candidate CF signal |
| **Then** | Re-enable popularity alpha | regularization | `popularity_alpha = 0.5` | Prevent popular drift once content signal is established |
| **Then** | CG score | retrieval signal | `n_interaction_features = 2` — re-enable CG score passthrough | Final boost: give ranker CG's own retrieval confidence |
| **Later** | DCN V2 cross network | architecture | Replace WideDeepRanker deep path with explicit cross layers | Feature crossing at scale once feature set is locked |

**Rule:** Beat CG on independent content features before adding CG score. But do add CG's feature set — that is not the same as leaking the CG score.

---

## Diagnostic: If Ranker Still Can't Beat CG After Full CG Parity + Cross Features

If a ranker with all CG features + all 5 cross features + CG score still can't beat CG's NDCG@10, the bug is in the training objective or architecture, not features:

- **BCE vs softmax:** BCE on random (row, candidate) tuples has very different gradient structure than softmax over the full candidate pool. Try evaluating with softmax loss over the 250-candidate pool.
- **MNS easy-negative fraction:** easy negatives (random corpus items) may be too easy, wasting batch capacity. Try reducing `easy_neg_frac` from 0.5 to 0.2.
- **Negative sampling leakage:** verify hard negatives are genuinely hard (high CG score) and not contaminated by easy items appearing in the hard slot.
- **Architecture capacity:** `[256, 128, 64]` with a 130-dim input may not be deep enough for the expanded feature set. Try `[512, 256, 128]`.

A ranker with strictly more information than CG that still loses is a bug, not a feature gap.

---

## Ablation Results Log

CG baseline shifted at liked-only parquet rerun: **NDCG@10=0.0938 · MRR=0.0845** (harder task — liked-only val labels). All runs below use this baseline unless noted.

| Run | Key config | Val NDCG@10 | Val MRR | Delta vs CG | Notes |
|-----|-----------|-------------|---------|-------------|-------|
| CG baseline (old parquets) | — | 0.0965 | 0.0871 | — | Recall@250=0.6737, all-movie labels |
| CG baseline (liked-only parquets) | — | 0.0938 | 0.0845 | — | Harder task; use this for all future comparisons |
| 50k, MLPRanker, α=0.5, CG score + genome cosine | 200k steps, alpha=0.5 | 0.0885 | 0.0798 | −0.0080 | Best at step 20k, degraded after. CG score leakage — ranker learned to follow CG, not beat it. |
| 200k, WideDeepRanker, α=0, genome cosine only, all-movie labels | 200k steps, alpha=0 | 0.0324 | 0.0348 | −0.0641 | Step 10k only — run was superseded. Shows label noise effect. |
| **200k, WideDeepRanker, α=0, genome cosine only, liked-only labels** | `ranker_mlp_alpha_0_20260505_111114.pth` | **0.0532** | **0.0521** | **−0.0406** | Steady improvement 10k→190k (0.0440→0.0532). Still improving at end of run. 56.7% of CG NDCG. |

### Training curve (liked-only baseline)

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
2. **Beat CG on content features before adding CG score.** Ranker should earn its NDCG@10 > 0.0965 from independent signal.
3. **Beat CG before tuning alpha.** Don't chase canary quality until offline metrics confirm the ranker works.
4. **No src/ modifications.** Ranker is fully self-contained; CG code is read-only.
5. **No streamlit/export changes** until a model is verified better by eval + canary.
6. **`BCEWithLogitsLoss` only.** Never `BCELoss`. Never sigmoid in `WideDeepRanker.forward()`.
7. **Batch sampling is across all tuples** — not within rollback groups (avoids 249:1 imbalance dominating gradients).
8. **E2E ceiling always enforced** in both eval and CG baseline — apples-to-apples comparison only.
