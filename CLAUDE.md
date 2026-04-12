# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A PyTorch Two-Tower neural network recommender system trained on the MovieLens 32M dataset. The model predicts ratings via dot product of user and item embeddings.

**Critical design choice: no user ID embedding.** Most recommender tutorials embed unique user IDs, which means inference is only possible for users seen during training. A new user has no embedding — the only options are full retraining, partial fine-tuning, or proxying to a similar existing user. This model deliberately avoids that trap. Users are represented entirely by their taste signals: watch history (rating-weighted avg pooling of movie embeddings), genre affinity, genome content pooling, and timestamp. Any user can be represented at inference time as long as you have even a few movies they liked — no retraining required. This makes the model production-friendly and eliminates user-level cold start.

## Running the Code

The project is a Python CLI (`main.py`). Notebooks in `jupyter/` are archived references only — the canonical code is in `src/`.

```bash
python main.py preprocess      # Stage 1: raw CSVs → data/base_*.parquet
python main.py features        # Stage 2: base parquets → data/features_*.parquet
python main.py dataset         # Stage 3: features → data/dataset_*_v1.pt (cached tensor splits)
python main.py train           # Stage 4: load cached splits, train, save checkpoints
python main.py canary          # Canary user recommendations (most recent checkpoint)
python main.py canary <path>   # Canary user recommendations (specific checkpoint)
python main.py probe           # Embedding probes (most recent checkpoint)
python main.py probe <path>    # Embedding probes (specific checkpoint)
python main.py                 # Run all stages in order
```

Stages 1–3 are slow and cache results to disk. Re-run only what changed:
- Changed raw data → rerun from `preprocess`
- Changed feature engineering → rerun from `features`
- Changed train/val split logic → rerun from `dataset`
- Changed model/hyperparams only → rerun from `train` (skip 1–3)

## Dataset

The `data/ml-32m/` directory must be present (not in git). Required files:
- `ratings.csv` — 33M rows: userId, movieId, rating, timestamp
- `movies.csv` — 86k movies: movieId, title, genres (pipe-separated)
- `tags.csv` — free-form user-applied tags (userId, movieId, tag, timestamp)
- `genome-scores.csv` — ML-derived relevance scores (0–1) per (movie, tag) pair
- `genome-tags.csv` — curated vocabulary of ~1,128 genome tag names

Only movies with **200+ ratings** are kept (~9,375 movies). Only users with 20–500 ratings are kept.

### Tag data

**User-applied tags (`tags.csv`):** Each movie's tag vector is built by counting how many users applied each tag, then normalizing to sum to 1. Only tags applied 1,000+ times across all movies are kept (306 tags survive). Controlled by `USE_ITEM_TAG_TOWER` / `USE_USER_TAG_TOWER`.

**Genome tags (`genome-scores.csv`):** ML-derived relevance scores (0.0–1.0) per (movie, tag) pair. 1,128 tags. Much denser and more semantically meaningful than user-applied tags — movies get non-zero scores for relevant tags even when users didn't explicitly apply them. Controlled by `USE_ITEM_GENOME_TAG_TOWER`.

## Model Architecture (v3 — canonical)

**Two-Tower design** with dot product prediction:

```
User Tower:
  rating_weighted_avg_pool(item_embeddings[watch_history])              →  history_emb  (size: item_movieId_embedding_size)
  rating_weighted_avg_pool(item_genome_tag_tower(genome_ctx[history]))  →  genome_emb   (size: item_genome_tag_embedding_size)  [shared tower]
  user_genre_tower([avg_rating_per_genre | watch_frac])                 →  genre_emb    (size: user_genre_embedding_size)
  timestamp_embedding_tower(watch_month)                                →  ts_emb       (size: timestamp_feature_embedding_size)
  concat → user_combined

Item Tower:
  item_genre_tower(genre_onehot)        →  item_genre_emb       (size: item_genre_embedding_size)
  item_tag_tower(tag_vector)            →  item_tag_emb         (size: item_tag_embedding_size)
  item_genome_tag_tower(genome_scores)  →  item_genome_tag_emb  (size: item_genome_tag_embedding_size)  [shared with user genome pool]
  item_embedding_tower(movie_id)        →  item_emb             (size: item_movieId_embedding_size)
  year_embedding_tower(release_year)    →  year_emb             (size: item_year_embedding_size)
  concat → item_combined

Prediction: dot_product(user_combined, item_combined)
```

All towers use `nn.Linear → Tanh`. Weights initialized with Xavier uniform (gain=0.01).

**Shared tower:** `item_genome_tag_tower` is used for both the item side (target movie genome scores) and the user genome pooling (each watched movie's genome context → pool). This ensures both live in the same embedding space so the dot product is directly meaningful. The genome contexts are stored in a `genome_context_buffer` registered as a non-trainable buffer in the model, indexed by the same embedding indices already in `user_watch_history` — no dataset changes required.

### Critical dimension constraint

`len(user_combined) == len(item_combined)` must hold for the dot product. The model raises `ValueError` at construction if violated.

### Current embedding sizes (110-dim)

```python
item_movieId_embedding_size      = 40   # shared: user history pool + item tower
item_year_embedding_size         = 5
timestamp_feature_embedding_size = 5
item_tag_embedding_size          = 10
item_genome_tag_embedding_size   = 35   # shared: item tower + user genome pool (cannot be set independently)
user_genre_embedding_size        = 30
item_genre_embedding_size        = 20

# user:  40 + 35 + 30 + 5       = 110
# item:  20 + 10 + 35 + 40 + 5  = 110  ✓
```

**Previous model (120-dim, with user genome pooling):**
```
item_year=10, ts=10, item_tag=15, user_genre=35
user: 40+35+35+10=120 / item: 20+15+35+40+10=120
```

**Previous model (120-dim, no user genome pooling):**
```
user_genre=70, no user genome pool
user: 40+70+10=120 / item: 20+15+35+40+10=120
```

**Previous model (80-dim, checkpoint `best_checkpoint_20260403_210601.pth`):**
```
item_movieId=20, genre_budget=50, item_tag=15, genome=25
user: 20+50+10=80 / item: 10+15+25+20+10=80
```

### Key implementation notes

- **Shared item embedding:** `item_embedding_lookup` is shared between the item tower (target movie) and the user history avg pool. The lookup has `padding_idx = top_movies_len`. History pooling output size = `item_movieId_embedding_size` — they cannot be decoupled without adding a projection layer.
- **Shared genome tower:** `item_genome_tag_tower` is shared between item side and user genome pooling. `nn.Linear` + `nn.Tanh` broadcast over leading dims, so `(batch, hist_len, 1128)` → `(batch, hist_len, 35)` works without reshaping.
- **genome_context_buffer:** Registered as a non-trainable buffer (saved in state_dict, device-portable). Row `i` = genome context for movie at embedding index `i`; last row = zeros (pad index). Built in `build_model()` from `fs.movieId_to_genome_tag_context`.
- **Rating-weighted avg pool:** Each watched movie embedding is weighted by the user's debiased rating. Abs-value normalization prevents negative ratings from cancelling positive ones in the denominator.
- **Removing `item_embedding_tower` causes severe genre clustering** — do not remove it.
- **Item ID embedding does not learn meaningful CF signal.** `probe_similar` on `MOVIEID_EMBEDDING` alone returns near-random neighbors (e.g. LotR → Hero, Thin Blue Line, Earth vs. Flying Saucers). Content towers (genre, genome) dominate the gradient and ID embeddings don't develop independent co-watch structure under MSE loss. Consequence: the user history pool (which pools over item ID embeddings) is also not capturing genuine CF signal — user taste representation comes mostly from the genome pool and genre tower. Fix requires sampled softmax with implicit feedback.
- **Tanh saturation:** Small sub-embedding spaces (genre=20, tag=15) saturate after training, causing sub-embedding cosine probes to return 1.0 for many movies. This is a known limitation. ReLU + larger dims would fix it but requires retraining.

## Training Details

- **Loss**: MSE on de-biased ratings (raw rating − user mean)
- **Optimizer**: SGD, `lr=0.005`, `momentum=0.9`
- **Batch size**: 64
- **Steps**: 150,000
- **Val logging**: every 10,000 steps (full val pass)
- **Checkpointing**: best val checkpoint saved on improvement; periodic checkpoint every 30,000 steps
- **LR=0.01**: Too aggressive; collapses genre boundaries — avoid
- **Train/val split**: 90/10 user-based; 90% of each user's history as context, 10% as labels

## Saving / Loading Models

Checkpoints are weights-only (`~1MB`). The `saved_models/` directory is gitignored.

```bash
python main.py canary saved_models/best_checkpoint_<timestamp>.pth
python main.py probe  saved_models/best_checkpoint_<timestamp>.pth
```

The evaluate setup loads the checkpoint first (fast, ~1MB), then features (slow). If the checkpoint is invalid it fails immediately before the slow features load.

## Embedding Probes (`python main.py probe`)

Three sub-embedding probes + one combined-space probe:

**`probe_genre(genre)`** — one-hot genre vector → `item_genre_tower` → cosine vs all `MOVIE_GENRE_EMBEDDING`. Shows what the item genre embedding space learned. Scores near 1.0 indicate Tanh saturation (embedding space too small).

**`probe_tag(tags)`** — tag vector → `item_tag_tower` → cosine vs all `MOVIE_TAG_EMBEDDING`. Default query: `['pixar', 'animation']`.

**`probe_genome_tag(genome_tags)`** — finds top-k_anchors most representative movies by raw genome score, averages their `MOVIE_GENOME_TAG_EMBEDDING` as query, cosine vs all `MOVIE_GENOME_TAG_EMBEDDING`. Seeds are marked `[seed]` in output. Avoids synthetic OOD inputs. Default queries: `['horror', 'gore', 'torture']` and `['martial arts', 'kung fu']`.

**`probe_similar(titles)`** — computes pairwise cosine similarity on `MOVIE_EMBEDDING_COMBINED` (full 120-dim space). Most reliable probe — uses the space the model actually optimizes. Output is a table: seed title | top-1 | top-2 | ... | top-5.

> **Note on genome tag probing:** Genome scores are dense (0–1, nearly every movie has non-zero scores for most tags). Synthetic 0-filled vectors are OOD for the genome tower. The anchor approach avoids this by using real movie embeddings as the query.

## Canary Users for Eval

Use these synthetic users to quickly assess model quality after training:

> **Timestamp:** canary users are synthetic and have no real watch timestamps. All canary users receive `ts_max_bin` (the most recent timestamp bin in the training data), meaning the timestamp tower sees them as current users. This is set once in `run_canary_eval` and passed to every `_build_user_embedding` call.

- **Horror Lover** and **Sci-Fi Lover** — most sensitive to genre drift; if these are wrong, the model is failing
- **Comedy Lover** and **Romance Lover** — tend to work well across all runs; good sanity checks
- **Crime Lover** — stress test for edge cases; expect imperfect results due to genre overlap

### Known persistent issues (as of 120-dim model)

- **Fantasy Lover** drifts to arthouse/surreal (Howl's Moving Castle is correct; Brazil, Dark City, After Hours are not)
- **Sci-Fi Lover** drifts to arthouse/cult (Videodrome, Naked Lunch, Blue Velvet) — prestige sci-fi shares genome signals with surreal cinema
- **War Movie Lover** drifts to epic/action (LotR, Matrix, Doctor Strange) — "acclaimed serious film" cluster
- Removing `item_embedding_tower` causes severe genre clustering — do not remove it

## Future Training Objective Improvements

1. **Sampled softmax (implicit feedback)** — replace MSE regression on ratings with sampled softmax
   classification over the item corpus (YouTube DNN, 2016). Frame recommendation as "predict the
   next watched item" rather than "predict a rating". Use implicit feedback (every watch = positive
   example) with sampled negatives and cross-entropy loss. Yields orders of magnitude more training
   signal since ratings are sparse but watches are plentiful. At serving time reduces to the same
   nearest neighbor search in dot product space. Requires: new label construction (item IDs, not
   ratings), dropping debiasing, adding negative sampling to the training loop.

2. **Implicit vs explicit feedback tradeoff** — explicit ratings (current) give clean preference
   signal but are sparse. Implicit feedback (watches) is abundant but noisy — a user may finish
   a movie they disliked. Consider a hybrid: use implicit feedback for candidate generation
   (softmax) and explicit ratings for a separate ranking stage.

---

## Future User Tower Improvements

Roughly ordered by implementation cost:

1. **User genome tag affinity context** — analogous to the genre context vector, compute rating-weighted avg genome tag relevance scores over a user's watch history. Mirrors the item-side genome tag tower. Likely high impact since item side already uses genome scores.
2. **Recency-weighted pooling** — exponential decay so recent watches matter more than old ones
3. **Rating variance per genre** — consistency signal (always loves horror vs. sometimes likes it)
4. **Explicit dislikes** — low-rated movies (1–2 stars) pooled separately as a negative taste embedding
5. **Short-term vs. long-term history** — two pooled embeddings (e.g., last 10 vs. all history) concatenated
6. **Transformer over history** — replace avg pooling with a small Transformer encoder; `[CLS]` token becomes the history embedding

## Richer Cross-Signal Features to Explore

The key insight: when two signals are concatenated into the **same tower**, the linear layer can learn cross-term interactions between them. This is more powerful than separate towers, which can only learn each signal independently.

### User side — additions to the genre context tower

Currently the genre context tower takes `[avg_debiased_rating_per_genre | watch_frac_per_genre]` (2 × n_genres). Good candidates to add alongside:

- **Rating variance per genre** — `high watch + high avg + low variance` = genuine fan; `high watch + low avg + high variance` = casual/completionist watcher.
- **Recency weight per genre** — fraction of the user's *last N* watches that were genre X, alongside the all-time fraction.

### Item side — cross-signal tower opportunities

- **Genre + year in the same tower** — genre conventions change drastically by decade. Currently year and genre are completely separate towers with no interaction — directly attacks era confusion in recommendations.
- **Genre + genome tags in the same tower** — genre labels are coarse (20 categories); genome tags are 1128 fine-grained signals. Cross-learning discovers sub-genres.
- **Global avg rating + genre** (new feature, not currently used) — highly-rated Drama is fundamentally different from low-rated Drama. Directly attacks the War/Drama clustering problem.

### Priority

1. **Genre + year (item side)** — directly attacks era confusion, low implementation cost
2. **Rating variance (user side)** — sharpens genre signals by distinguishing fans from casual watchers
3. **Genre + genome tags (item side)** — highest potential impact but requires merging two large input spaces

## Known Code Inconsistencies (to fix)

- **Label movieIds stored as raw IDs, watch history as embedding indices** — in `features.py`,
  `watch_history` is mapped to embedding indices via `item_emb_movieId_to_i` at feature
  engineering time, but `label_movieIds` are stored as raw movie IDs and only mapped to embedding
  indices later in `dataset.py`. Both work correctly but the inconsistency is confusing.
  Fix: map label movieIds to embedding indices in `features.py` the same way watch history is.

---

## Git Workflow

Never commit and push in the same command. Always commit first, then ask before pushing.
