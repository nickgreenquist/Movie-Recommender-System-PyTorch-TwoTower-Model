# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Current State

The model is complete and deployed. Best checkpoint: `best_mse_gpool_gctx_proj_20260426_202808.pth` (MSE rollback, genome pool ON, genome context tower ON, projection MLP, 128-dim output, 512k params). This is the prod model running in Streamlit — do not replace it without a clearly better eval result.

To re-export serving artifacts from prod checkpoint:

```bash
python main.py export saved_models/best_mse_gpool_gctx_proj_20260426_202808.pth
```

## Project Overview

A PyTorch Two-Tower neural network recommender system trained on the MovieLens 32M dataset. The model predicts ratings via dot product of user and item embeddings.

**Critical design choice: no user ID embedding.** Users are represented entirely by their taste signals: watch history (rating-weighted avg pooling of movie embeddings), genre affinity, genome content pooling, and timestamp. Any user can be represented at inference time as long as you have even a few movies they liked — no retraining required.

## Running the Code

The project is a Python CLI (`main.py`). Notebooks in `jupyter/` are archived references only — the canonical code is in `src/`.

```bash
python main.py preprocess          # Stage 1: raw CSVs → data/base_*.parquet
python main.py features            # Stage 2: base parquets → data/features_*.parquet
python main.py dataset             # Stage 3: features → data/dataset_mse_rollback_*_v1.pt
python main.py train               # Stage 4: MSE training (SGD, rating regression)
python main.py train softmax       # Stage 4: in-batch negatives softmax training
python main.py canary              # Canary user recommendations (most recent checkpoint)
python main.py canary <path>       # Canary user recommendations (specific checkpoint)
python main.py probe               # Embedding probes (most recent checkpoint)
python main.py probe <path>        # Embedding probes (specific checkpoint)
python main.py eval                # Offline eval: Recall@K, NDCG@K, Hit Rate@K, MRR
python main.py eval <path>         # Same, specific checkpoint
python main.py export              # Stage 5: export serving artifacts for Streamlit
python main.py export <path>       # Export using specific checkpoint
python main.py posters             # Fetch movie poster URLs from TMDB → serving/posters.json
python main.py                     # Run all stages in order (MSE)
```

### Serving artifacts (`serving/`)

After `export`, the `serving/` directory contains:
- `model.pth` — model state_dict
- `movie_embeddings.pt` — precomputed item embeddings for all corpus movies
- `feature_store.pt` — inference-only dict (vocabularies, index maps, model config)
- `posters.json` — TMDB poster URLs: `{"<movieId>": "<url>", ...}` (empty string = no poster)

To regenerate posters (run once; safe to interrupt and resume — skips already-fetched):
```bash
TMDB_API_KEY=your_key python main.py posters
```
Get a free key at https://www.themoviedb.org/settings/api. Fetches ~9,375 corpus movies at 0.25s/request (~40 min). Current coverage: 9,293/9,375 (99.3%); 64 failures are invalid TMDB IDs in links.csv (not transient — won't resolve on retry).

The Streamlit app (`streamlit_app.py`) loads `posters.json` at startup and shows a poster grid (5 columns) for all recommendation results. Falls back to a placeholder tile if a poster is missing.

Stages 1–3 are slow and cache results to disk. Re-run only what changed:
- Changed raw data → rerun from `preprocess`
- Changed feature engineering → rerun from `features`
- Changed train/val split logic → rerun from `dataset`
- Changed model/hyperparams only → rerun from `train` (skip 1–3)
- Softmax dataset and MSE dataset are cached separately

## Dataset

The `data/ml-32m/` directory must be present (not in git). Required files:
- `ratings.csv` — 33M rows: userId, movieId, rating, timestamp
- `movies.csv` — 86k movies: movieId, title, genres (pipe-separated)
- `tags.csv` — free-form user-applied tags (userId, movieId, tag, timestamp)
- `genome-scores.csv` — ML-derived relevance scores (0–1) per (movie, tag) pair
- `genome-tags.csv` — curated vocabulary of ~1,128 genome tag names

Only movies with **200+ ratings** are kept (~9,375 movies). Only users with 20–500 ratings are kept.

### Tag data

**User-applied tags (`tags.csv`):** Each movie's tag vector is built by counting how many users applied each tag, then normalizing to sum to 1. Only tags applied 1,000+ times across all movies are kept (306 tags survive).

**Genome tags (`genome-scores.csv`):** ML-derived relevance scores (0.0–1.0) per (movie, tag) pair. 1,128 tags. Much denser and more semantically meaningful than user-applied tags — movies get non-zero scores for relevant tags even when users didn't explicitly apply them.

## Model Architecture (canonical — prod checkpoint, projection MLP)

The prod checkpoint (`best_mse_gpool_gctx_proj_20260426_202808.pth`) uses the projection MLP architecture with genome context:

```
User Tower:
  rating_weighted_avg_pool(item_embeddings[watch_history])              →  history_emb  (32)
  rating_weighted_avg_pool(item_genome_tag_tower(genome_ctx[history]))  →  genome_emb   (32)  [shared tower]
  user_genre_tower([avg_rating_per_genre | watch_frac])                 →  genre_emb    (32)
  timestamp_embedding_tower(watch_month)                                →  ts_emb       (4)
  user_genome_context_tower(rating_weighted_avg(raw_genome[history]))   →  genome_ctx   (32)
  concat (132) → Linear(256) → ReLU → Linear(128) → user_emb (128)

Item Tower:
  item_genre_tower(genre_onehot)        →  item_genre_emb   (8)
  item_tag_tower(tag_vector)            →  item_tag_emb     (16)
  item_genome_tag_tower(genome_scores)  →  item_genome_emb  (32)  [shared with user genome pool]
  item_embedding_tower(movie_id)        →  item_emb         (32)  [shared with user history pool]
  year_embedding_tower(release_year)    →  year_emb         (8)
  concat (96) → Linear(256) → ReLU → Linear(128) → item_emb (128)

Prediction: dot_product(user_emb, item_emb)
```

Sub-tower linears: Xavier uniform `gain=0.1`. Projection linears re-initialized at `gain=1.0` after the rest of the model — without this, `gain=0.1²` compounds and collapses dot products to zero before training starts.

**Shared genome tower:** `item_genome_tag_tower` is shared between item side and user genome pooling, ensuring both live in the same embedding space. Genome contexts are stored in `genome_context_buffer` (non-trainable buffer, saved in state_dict).

**Shared item embedding:** `item_embedding_lookup` is shared between item tower and user history pool. They cannot be decoupled without a projection layer.

### Key implementation notes

- **Rating-weighted avg pool:** Each watched movie embedding is weighted by the user's debiased rating. Abs-value normalization prevents negative ratings from cancelling positive ones in the denominator.
- **Shared item embedding:** `item_embedding_lookup` is shared between the item tower and the user history pool. They cannot be decoupled without adding a projection layer. Removing `item_embedding_tower` causes severe genre clustering — do not remove it.
- **Shared genome tower:** `item_genome_tag_tower` is shared between the item side and user genome pooling, ensuring both live in the same embedding space. Genome contexts are stored in `genome_context_buffer` (non-trainable buffer, saved in state_dict).
- **Item ID embedding does not learn meaningful CF signal under MSE.** Content towers (genre, genome) dominate the gradient; ID embeddings don't develop independent co-watch structure. User taste representation comes mostly from the genome pool and genre tower.
- **Genome sub-tower size matters:** genome compresses 1,128 → 32 dims. Do not shrink below 32 — information is lost before the projection MLP can mix it. Tested at 16, was noticeably worse.
- **Backward compatibility:** `proj_hidden=None` → legacy flat model. All checkpoint-loading code infers architecture fully from state-dict shapes, so old and new checkpoints coexist.

## Training Details

### MSE rollback (`python main.py train rollback`) — canonical objective

- **Loss**: MSE on de-biased ratings (raw rating − user mean)
- **Optimizer**: SGD, `lr=0.005`, `momentum=0.9`
- **Batch size**: 64
- **Steps**: 150,000
- **Val logging**: every 10,000 steps (full val pass)
- **Checkpointing**: `saved_models/best_mse_<pool>_<timestamp>.pth`
- **LR=0.01**: Too aggressive; collapses genre boundaries — avoid
- **Adam is a failure for MSE training — do not use.** Causes user tower collapse (same recs for every user regardless of taste). Root causes: adaptive per-parameter lr blows up sparse embeddings early then freezes them; β2=0.999 accumulates stale gradients from unrelated users; dot-product scoring is norm-sensitive so a few high-norm items dominate every user. SGD with momentum generalizes better here because noisier uniform updates force distributed representations. Tested at lr=0.005 — complete failure. Do not retry.
- **Dataset structure**: rollback examples — for each watch event, context = all prior watches (chronological). User-level 90/10 train/val split; no within-user history split. Cap per user: `MAX_MSE_ROLLBACK_EXAMPLES_PER_USER=20`. Better cold-start generalization than fixed-split.
- **Dataset build**: `python main.py dataset rollback` → `data/dataset_mse_rollback_*_v1.pt`

### Softmax (`python main.py train softmax`) — implemented but not better on MovieLens

- **Loss**: cross-entropy over in-batch negatives (B×B score matrix, diagonal = correct targets)
- **Dataset**: rollback examples — for each watch event, context = all prior watches. Capped at `MAX_SOFTMAX_EXAMPLES_PER_USER`.
- **Optimizer**: Adam, `lr=0.001`, `weight_decay=1e-5`
- **Batch size**: 512, **Temperature**: 0.05, **Steps**: 150,000
- **Checkpointing**: `saved_models/best_softmax_<pool>_<timestamp>.pth`
- **`F.normalize` must NOT be used in `train_softmax`.** Applying it makes training optimize cosine similarity while inference uses raw dot products — train/inference mismatch. Always use raw dot products in both.
- **Similarity metric rule — do not revisit:** Raw dot product for user-to-item scoring (training, eval, canary). Cosine similarity for item-to-item (probe_similar, tab_similar). Raw dot for item-item causes high-norm items to dominate every neighborhood. Tested and confirmed worse — do not switch again.

## Saving / Loading Models

Checkpoints are weights-only (~1MB). The `saved_models/` directory is gitignored.

Naming convention: `best_<loss>_<pool>_<timestamp>.pth`
- `<loss>`: `mse` or `softmax`
- `<pool>`: `gpool` (genome pool ON) or `nopool` (genome pool OFF)

Auto-detection (no path given) picks the most recently modified checkpoint from `best_mse_*.pth`, `best_softmax_*.pth`, and legacy `best_checkpoint_*.pth`.

The evaluate setup loads the checkpoint first (fast), then features (slow). Invalid checkpoint fails immediately before the slow features load.

## Offline Evaluation (`python main.py eval`)

`src/offline_eval.py` — Recall@K, NDCG@K, Hit Rate@K, MRR at K = 1, 5, 10, 20, 50.

Two protocols — auto-selected based on checkpoint name, or forced with `python main.py eval <path> rollback`:

**Rollback protocol** (default for MSE rollback and softmax checkpoints): For each val user (held out at user level), sample up to 20 chronological positions. Context = history[0..j-1], target = history[j]. All positions valid since val users were never seen in training.

**Leave-label-out protocol** (legacy MSE fixed-split): Context = `user_to_watch_history` (90% of ratings), targets = `user_to_movie_to_rating_LABEL` (remaining 10%). 5,000 users sampled with `random.Random(42)`.

Note: rollback eval produces lower absolute numbers than leave-label-out (harder task), so compare only within the same protocol.

## Embedding Probes (`python main.py probe`)

**`probe_genre(genre)`** — one-hot → `item_genre_tower` → cosine vs all genre embeddings.

**`probe_tag(tags)`** — tag vector → `item_tag_tower` → cosine vs all tag embeddings.

**`probe_genome_tag(genome_tags)`** — averages top-k representative movie genome embeddings as query, cosine vs all genome embeddings. Uses real movie embeddings to avoid OOD synthetic inputs.

**`probe_similar(titles)`** — pairwise cosine similarity on `MOVIE_EMBEDDING_COMBINED` (128-dim with proj). Most reliable probe.

## Canary Users for Eval

> **Timestamp:** All canary users receive `ts_max_bin` (most recent timestamp bin).

- **Horror Lover** and **Sci-Fi Lover** — most sensitive to genre drift; if wrong, model is failing
- **Comedy Lover** and **Romance Lover** — good sanity checks, tend to work well
- **Crime Lover** — stress test; expect imperfect results due to genre overlap

### Known persistent issues (prod proj model)

- **War Movie Lover** drifts to prestige drama (Godfather, Shawshank) — "acclaimed serious film" cluster
- **WW2 Lover** drifts to prestige drama — only Defiance, Allied, Imitation Game stay on-genre
- **Western Lover** drifts to samurai/Japanese cinema (Kurosawa) — genome overlap with serious classic films
- **Anime Lover** drifts to European arthouse — genome overlap between Ghibli and slow/contemplative world cinema
- **Comedy Lover** has slight crime-comedy drift (Snatch, Lock Stock, Trainspotting) — acceptable
- **Sci-Fi Lover** is now fixed in the proj model (Interstellar, Gattaca, Sunshine — clean)

## MSE vs Softmax: Findings

MSE with genome pooling is the right objective for MovieLens. Softmax (validated on Goodreads at 3× improvement) failed here.

### Offline eval results — rollback protocol (updated 2026-04-28)

All numbers below use the rollback eval protocol (harder than leave-label-out; compare only within this table).

| Metric | Old prod: MSE flat | Softmax proj (genome=16) | MSE rollback proj | **Current prod: + genome context** |
|---|---|---|---|---|
| Hit Rate@1 | 0.19% | 0.14% | 0.43% | **0.44%** |
| Hit Rate@5 | 0.94% | 0.57% | 1.66% | **1.85%** |
| Hit Rate@10 | 1.70% | 1.00% | 2.70% | **3.01%** |
| Hit Rate@20 | 2.93% | 1.66% | 4.28% | **4.78%** |
| Hit Rate@50 | 5.62% | 3.45% | 7.59% | **8.41%** |
| **MRR** | 0.0084 | 0.0058 | 0.0135 | **0.0146** |

Current prod beats old prod by **+74% MRR** and +77% Hit Rate@10. Adding the genome context tower over the base MSE rollback proj gave an additional **+8% MRR** and fixed Sci-Fi genre drift in canary. Softmax is the worst — confirms MSE is correct for MovieLens.

### Key architecture findings

- **Genome pooling is essential** — always use `gpool`.
- **Projection MLP is a clear win** — +57% MRR over flat architecture with rollback training.
- **Rollback training > fixed-split** — model sees users at every history length, better cold-start.
- **Genome sub-embedding size matters** — 32 is the minimum; 16 noticeably hurts.
- **User genome context tower is a real win** — rating-weighted avg of raw genome scores (1,128-dim) → `user_genome_context_tower` (32-dim) added to user concat gave **+8% MRR** over the base projection MLP model and fixed Sci-Fi genre drift in canary. See `user_genome_context_tower` in `src/model.py`. Always on; not a flag.
- **MSE + gpool + genome context + projection MLP + rollback training is the optimal configuration.**
- **Pooling over the full 128-dim projected item embedding does not help** — see experiment log below.

### Softmax proj gpool findings (2026-04-24)

Projection MLP is a clear improvement over flat softmax — at only ~10k/150k steps, the proj model reached 69% of MSE prod's MRR on rollback eval (MRR 0.0058 vs 0.0084). But **canary results were poor** despite promising eval numbers.

Root cause: softmax treats every watched movie as a positive target regardless of rating. A user who rated a movie 1★ generates a training signal saying "recommend this to users like them." The model learns to predict the next movie a user *watches*, not the next movie they *like*. Eval rewards this (any watch counts as a hit) but canary exposes it (recommendations include movies the user would dislike).

**Fix to try next:** rebuild the softmax dataset with `min_target_rating=4.0` so only liked movies are positive targets:
```bash
python main.py dataset softmax 4.0
python main.py train softmax
```
Context (watch history) remains unfiltered — low-rated watches still inform the user embedding, only the target label is restricted.

### Why softmax fails on MovieLens

MovieLens interactions are shaped by the platform's own recommender — users rate movies they were *shown*, not movies they independently sought out. Softmax with in-batch negatives systematically suppresses popular items (frequent negatives), but on MovieLens popular = genuinely watched. Goodreads is user-driven; MovieLens is not.

Softmax also excels at niche taste clusters (Western, Martial Arts, Anime) but fails badly on high-traffic genres (Horror, Children, Comedy). The tradeoff is not worth it.

**Norm diagnostic:** Mean item embedding norm = 7.133, std = 0.110 — extremely tight. High-norm items are niche classics, not popular trash. Rules out runaway norm hypothesis.

**Conclusion:** Do not invest further in softmax for MovieLens. The objective mismatch is structural.

### Goodreads comparison (same architecture, different dataset)

| Metric | MSE | Softmax |
|---|---|---|
| Hit Rate@10 | 4.3% | **11.7%** |
| Hit Rate@50 | 16.8% | **31.3%** |
| MRR | 0.021 | **0.057** |

The cross-dataset result is the key finding: softmax wins on user-driven data, MSE wins on platform-driven data. The architecture is the same; the objective choice is dataset-dependent.

## Model Architecture (legacy — flat, pre-2026-04-24)

The old prod checkpoint (`best_checkpoint_20260406_210158.pth`) used the flat architecture (no projection MLP):

```
User Tower: history(40) + genome(35) + genre(30) + ts(5) = 110 → dot product directly
Item Tower: genre(20) + tag(10) + genome(35) + movieId(40) + year(5) = 110 → dot product directly
```

Kept for reference; superseded by the projection MLP architecture.

## Architecture Experiment Log (branch: full-item-pool, 2026-04-25)

**Motivation:** The user history pool uses only the 32-dim item ID lookup — a low-capacity signal. The item tower produces a full 128-dim projected embedding combining genre, tag, genome, ID, and year. Pooling over this richer representation is the "proper" way to represent watch history. We tested two variants.

### Experiment 1 — Full item pool replacing both id and genome pools (`use_item_pool_for_genome=True`)

User concat: `item_pool(128) + genre(32) + ts(4) = 164 → user_proj(164→256→128)`

**Result: null / worse than prod.** Canary showed arthouse attractor — Horror, Sci-Fi, Martial Arts, War all drifted toward Ghibli/European art cinema. The genome pool, which previously provided a discrete content-texture signal, was now baked silently into item_pool and the model couldn't weight it separately. Dropped this variant.

### Experiment 2 — Full item pool replacing id pool, genome pool kept separately (`use_item_pool_for_history=True`)

User concat: `item_pool(128) + genome_pool(32) + genre(32) + ts(4) = 196 → user_proj(196→256→128)`

Checkpoint: `best_mse_ipool_gpool_proj_20260425_143734.pth`

**Result: null / marginally worse than prod.** Canary was mixed — Sci-Fi improved, Martial Arts/Crime regressed. Offline eval (leave-label-out protocol):

| Metric | **ipool_gpool (new)** | **Prod (gpool_proj)** |
|---|---|---|
| Hit Rate@1 | 1.74% | **1.92%** |
| Hit Rate@5 | **6.46%** | 6.42% |
| Hit Rate@10 | 10.72% | **10.88%** |
| Hit Rate@20 | 17.38% | **17.64%** |
| Hit Rate@50 | 30.40% | 30.40% |
| **MRR** | 0.0510 | **0.0521** |

*Leave-label-out protocol; not comparable to rollback numbers in the MSE vs Softmax table above.*

Prod wins on MRR (−2.1%) and most Hit Rate@K. The 32-dim id pool carries signal the full projected embedding doesn't — likely because the projection is trained to match item-side targets, making it noisier as a history-pooling signal than a purpose-built id lookup.

**Conclusion: the separate 32-dim id pool + 32-dim genome pool architecture is optimal. Do not revisit full-item-pool variants.**

### What we know does NOT work

- Removing the item ID embedding from the item tower entirely — canary genre discrimination collapsed. Do not try again.
- Removing `item_embedding_tower` (the Linear+Tanh on top of the ID lookup) — causes severe genre clustering.
- Pooling over the full projected item embedding (128-dim) in the user tower — confirmed null result across two variants.

## Training Experiment Log (2026-04-27)

### Experiment: 300k training steps

**Motivation:** Check whether the model fully converges at 150k–200k steps.

Checkpoint: `best_mse_gpool_proj_20260427_211802.pth` (trained on 20-example/user dataset)

**Result: null.** MRR 0.0145 vs prod 0.0146 — tied within noise. Canary identical to prod. The model converges well before 300k steps; extra training provides no benefit.

### Experiment: 30 examples per user (dataset rebuild, 2026-04-28)

**Motivation:** More rollback examples per user (30 vs 20) exposes the model to more chronological positions in long watch histories, potentially improving generalization.

Dataset stats: 4,942,281 train examples (+43% vs 20-example dataset), ~2.6GB on disk, ~16GB RAM (vs ~12GB at 20 examples).

Checkpoint: `best_mse_gpool_proj_20260428_070226.pth`

**Result: null.** MRR 0.0145 vs prod 0.0146 — tied. Marginally better at Hit Rate@10 (3.07% vs 3.01%) and Hit Rate@20 (4.81% vs 4.78%) but no MRR improvement. Canary identical.

**Conclusion: canonical dataset cap is 20 examples/user. Do not increase — extra signal does not help given current architecture and the additional 4GB RAM is not worth it.**

## Dataset Memory Fix (2026-04-25)

The dataset `.pt` files previously stored redundant per-item feature tensors (genre, tag, genome, year for the target movie) that were also available via the feature store. This bloated RAM to 33 GB on a 24 GB machine.

**Fix:** Strip those tensors from the dataset builders in `src/dataset.py`. Dataset is now a 6-tuple (MSE) and 5-tuple (softmax):
- MSE: `(X_genre, X_history, X_history_ratings, timestamp, Y, target_movieId)`
- Softmax: `(X_genre, X_history, X_history_ratings, timestamp, target_movieId)`

Item features are looked up from model buffers during training (no dataset rebuild needed for model-only changes). RAM dropped from 33 GB → ~12 GB.

If loading old 10-tuple `.pt` files: `load_mse_rollback_splits()` slices to `[:6]` for backward compat.

## Potential Next Improvements

ROI on further MovieLens tuning is low. If continuing:

1. **Rating variance per genre** in the user context vector — distinguishes genuine fans from casual watchers.
2. **Genre + year in the same item tower** — directly attacks era confusion in recommendations (War/WW2/Western drift).
3. **Embedding size tuning:** Experiment with larger genome/genre/ID embedding dims — may improve representation capacity.
4. **Remove timestamp from user tower:** It adds only 4 dims; removing it simplifies the user concat and may not hurt quality.
5. **Freeze item embeddings during user pooling:** Prevent gradients from flowing back through the item tower when computing the user pool — decouples item representation learning from user pooling.

## Known Code Inconsistency

**Label movieIds stored as raw IDs, watch history as embedding indices** — `watch_history` is mapped to embedding indices in `features.py`, but `label_movieIds` are stored as raw IDs and mapped later in `dataset.py`. Both work correctly but the inconsistency is confusing. Fix: map label movieIds in `features.py`.

---

## Git Workflow

Never commit and push in the same command. Always commit first, then ask before pushing.

## Experiment Discipline

**Change one thing at a time.** Every training run should isolate exactly one variable. If asked to make multiple architectural changes in the same run, stop and warn before proceeding — the canary comparison becomes uninterpretable when multiple things change simultaneously.

**Do not touch `streamlit_app.py` or `src/export.py` until a model change is verified good.** The workflow is: train → canary → eval → if better, then update export/streamlit. Updating serving code before verification wastes time and risks shipping a broken model. This applies even if the code changes are "obviously needed" — wait for a confirmed win first.
