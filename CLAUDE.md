# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Current State

The model is complete and deployed. Best checkpoint: `best_checkpoint_20260406_210158.pth` (MSE, genome pool ON, single-layer towers, 110-dim, 429k params). This is the prod model running in Streamlit — do not replace it without a clearly better eval result.

To re-export serving artifacts from prod checkpoint:

```bash
python main.py export saved_models/best_checkpoint_20260406_210158.pth
```

## Project Overview

A PyTorch Two-Tower neural network recommender system trained on the MovieLens 32M dataset. The model predicts ratings via dot product of user and item embeddings.

**Critical design choice: no user ID embedding.** Users are represented entirely by their taste signals: watch history (rating-weighted avg pooling of movie embeddings), genre affinity, genome content pooling, and timestamp. Any user can be represented at inference time as long as you have even a few movies they liked — no retraining required.

## Running the Code

The project is a Python CLI (`main.py`). Notebooks in `jupyter/` are archived references only — the canonical code is in `src/`.

```bash
python main.py preprocess          # Stage 1: raw CSVs → data/base_*.parquet
python main.py features            # Stage 2: base parquets → data/features_*.parquet
python main.py dataset             # Stage 3: features → data/dataset_*_v1.pt  (MSE)
python main.py dataset softmax     # Stage 3: features → data/dataset_softmax_*_v1.pt
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
python main.py                     # Run all stages in order (MSE)
```

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

## Model Architecture (canonical)

**Two-Tower design** with dot product prediction:

```
User Tower:
  rating_weighted_avg_pool(item_embeddings[watch_history])              →  history_emb  (40)
  rating_weighted_avg_pool(item_genome_tag_tower(genome_ctx[history]))  →  genome_emb   (35)  [shared tower]
  user_genre_tower([avg_rating_per_genre | watch_frac])                 →  genre_emb    (30)
  timestamp_embedding_tower(watch_month)                                →  ts_emb       (5)
  concat → user_combined  (110)

Item Tower:
  item_genre_tower(genre_onehot)        →  item_genre_emb   (20)
  item_tag_tower(tag_vector)            →  item_tag_emb     (10)
  item_genome_tag_tower(genome_scores)  →  item_genome_emb  (35)  [shared with user genome pool]
  item_embedding_tower(movie_id)        →  item_emb         (40)
  year_embedding_tower(release_year)    →  year_emb         (5)
  concat → item_combined  (110)

Prediction: dot_product(user_combined, item_combined)
```

All towers use `nn.Linear → Tanh`. Weights initialized with Xavier uniform (gain=0.01).

**Shared genome tower:** `item_genome_tag_tower` is shared between item side and user genome pooling, ensuring both live in the same embedding space. Genome contexts are stored in `genome_context_buffer` (non-trainable buffer, saved in state_dict).

**Shared item embedding:** `item_embedding_lookup` is shared between item tower and user history pool. They cannot be decoupled without a projection layer.

### Critical dimension constraint

`len(user_combined) == len(item_combined)` must hold. The model raises `ValueError` at construction if violated.

### Key implementation notes

- **Rating-weighted avg pool:** Each watched movie embedding is weighted by the user's debiased rating. Abs-value normalization prevents negative ratings from cancelling positive ones in the denominator.
- **Removing `item_embedding_tower` causes severe genre clustering** — do not remove it.
- **Item ID embedding does not learn meaningful CF signal under MSE.** Content towers (genre, genome) dominate the gradient; ID embeddings don't develop independent co-watch structure. User taste representation comes mostly from the genome pool and genre tower.
- **Tanh saturation:** Small sub-embedding spaces (genre=20, tag=10) saturate after training, causing sub-embedding cosine probes to return 1.0 for many movies. Known limitation.

## Training Details

### MSE (`python main.py train`) — canonical objective

- **Loss**: MSE on de-biased ratings (raw rating − user mean)
- **Optimizer**: SGD, `lr=0.005`, `momentum=0.9`
- **Batch size**: 64
- **Steps**: 150,000
- **Val logging**: every 10,000 steps (full val pass)
- **Checkpointing**: `saved_models/best_mse_<pool>_<timestamp>.pth`
- **LR=0.01**: Too aggressive; collapses genre boundaries — avoid
- **Train/val split**: 90/10 user-based; 90% of each user's history as fixed context, 10% as labels
- **Dataset structure**: flat label expansion — each user has one fixed context; each label movie becomes one training example. No rollbacks.

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

**Protocol:** Leave-label-out per user. 5,000 users sampled with `random.Random(42)`.
- **Context** = `user_to_watch_history` (90% of ratings)
- **Targets** = `user_to_movie_to_rating_LABEL` (remaining 10%)

## Embedding Probes (`python main.py probe`)

**`probe_genre(genre)`** — one-hot → `item_genre_tower` → cosine vs all genre embeddings.

**`probe_tag(tags)`** — tag vector → `item_tag_tower` → cosine vs all tag embeddings.

**`probe_genome_tag(genome_tags)`** — averages top-k representative movie genome embeddings as query, cosine vs all genome embeddings. Uses real movie embeddings to avoid OOD synthetic inputs.

**`probe_similar(titles)`** — pairwise cosine similarity on `MOVIE_EMBEDDING_COMBINED` (110-dim). Most reliable probe.

## Canary Users for Eval

> **Timestamp:** All canary users receive `ts_max_bin` (most recent timestamp bin).

- **Horror Lover** and **Sci-Fi Lover** — most sensitive to genre drift; if wrong, model is failing
- **Comedy Lover** and **Romance Lover** — good sanity checks, tend to work well
- **Crime Lover** — stress test; expect imperfect results due to genre overlap

### Known persistent issues

- **Fantasy Lover** drifts to arthouse/surreal (Brazil, Dark City)
- **Sci-Fi Lover** drifts to arthouse/cult (Videodrome, Naked Lunch) — prestige sci-fi shares genome signals with surreal cinema
- **War Movie Lover** drifts to epic/action (LotR, Matrix) — "acclaimed serious film" cluster

## MSE vs Softmax: Findings

MSE with genome pooling is the right objective for MovieLens. Softmax (validated on Goodreads at 3× improvement) failed here.

### Offline eval results (2026-04-21)

| Metric | **Prod: MSE gpool 1-layer** | MSE gpool 2-layer | MSE nopool 2-layer | Softmax nopool 2-layer + ≥4★ |
|---|---|---|---|---|
| Hit Rate@1 | **1.12%** | 0.92% | 1.14% | 0.44% |
| Hit Rate@5 | **4.86%** | 4.18% | 3.68% | 1.82% |
| **Hit Rate@10** | **8.36%** | 7.72% | 6.38% | 3.54% |
| Hit Rate@20 | **13.60%** | 13.10% | 11.70% | 6.32% |
| Hit Rate@50 | **24.54%** | 24.60% | 23.04% | 12.64% |
| Recall@10 | **0.0140** | 0.0122 | 0.0096 | 0.0057 |
| NDCG@10 | **0.0133** | 0.0117 | 0.0104 | 0.0056 |
| **MRR** | **0.0381** | 0.0348 | 0.0334 | 0.0183 |

**The prod model was never beaten.** Despite trying deeper towers (2-layer), different objectives (softmax), genome pool ON/OFF, and rating filters — the original MSE gpool single-layer model remains the best on every metric. Do not redeploy without a clearly better eval result.

### Key architecture findings

- **Genome pooling is essential** — nopool 2-layer loses to gpool 1-layer on every metric. Always use `gpool`.
- **Deeper towers did not help** — gpool 2-layer underperforms gpool 1-layer despite 113k more parameters. The single-layer towers are expressive enough for the input sizes.
- **MSE + gpool + single-layer is the optimal configuration for this dataset.**

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

## Potential Next Improvements

ROI on further MovieLens tuning is low. If continuing:

1. **MSE with rollbacks** — currently MSE uses one fixed context per user (90% of history). Rollback training would expose the model to users at every history length, improving cold-start generalization for the Streamlit use case (users inputting 5–10 favorite movies). Modest expected gain.
2. **Rating variance per genre** in the user context vector — distinguishes genuine fans from casual watchers.
3. **Genre + year in the same item tower** — directly attacks era confusion in recommendations.

## Known Code Inconsistency

**Label movieIds stored as raw IDs, watch history as embedding indices** — `watch_history` is mapped to embedding indices in `features.py`, but `label_movieIds` are stored as raw IDs and mapped later in `dataset.py`. Both work correctly but the inconsistency is confusing. Fix: map label movieIds in `features.py`.

---

## Git Workflow

Never commit and push in the same command. Always commit first, then ask before pushing.
