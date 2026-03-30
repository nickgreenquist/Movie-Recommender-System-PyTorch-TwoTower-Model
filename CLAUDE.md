# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A PyTorch Two-Tower neural network recommender system trained on the MovieLens 32M dataset. The model predicts ratings via dot product of user and item embeddings. There is no user ID embedding — users are represented entirely by their watch history (rating-weighted avg pooling of movie embeddings), genre affinity, and tag signals. This makes the model cold-start-friendly and generalizable to new users.

## Running the Code

All code lives in Jupyter notebooks. There are no standalone scripts, Makefile, or CLI entrypoints.

```bash
# Start Jupyter
jupyter notebook

# Open the latest version
# MovieLens_Two_Tower_Embedding_NN_v3_user_embedding_avg_pooling.ipynb
```

The notebooks run top-to-bottom. Each notebook is self-contained (data loading through inference). GPU acceleration uses Apple MPS when available (`torch.device("mps")`).

## Dataset

The `ml-32m/` directory must be present (not in git). Required files:
- `ratings.csv` — 33M rows: userId, movieId, rating, timestamp
- `movies.csv` — 86k movies: movieId, title, genres (pipe-separated)
- `tags.csv` — free-form user-applied tags (userId, movieId, tag, timestamp)
- `genome-scores.csv` — ML-derived relevance scores (0–1) per (movie, tag) pair
- `genome-tags.csv` — curated vocabulary of ~1,128 genome tag names

Only movies with **1,000+ ratings** are kept (~4,461 movies). Only users with 20–500 ratings are kept.

> **Note:** The threshold was lowered from 3,000 → 1,000 ratings. Earlier architectures were unstable at this corpus size, but the current v3 architecture (rating-weighted pooling + genome tags) handles it well and produces better recommendations.

### Tag data

**User-applied tags (`tags.csv`):** Each movie's tag vector is built by counting how many users applied each tag, then normalizing to sum to 1. Only tags applied 1,000+ times across all movies are kept (306 tags survive). Controlled by `USE_ITEM_TAG_TOWER` / `USE_USER_TAG_TOWER`.

**Genome tags (`genome-scores.csv`):** ML-derived relevance scores (0.0–1.0) per (movie, tag) pair. 1,128 tags. Much denser and more semantically meaningful than user-applied tags — movies get non-zero scores for relevant tags even when users didn't explicitly apply them. Controlled by `USE_ITEM_GENOME_TAG_TOWER`.

## Model Architecture (v3 — canonical)

**Two-Tower design** with dot product prediction:

```
User Tower:
  rating_weighted_avg_pool(item_embeddings[watch_history])  →  history_emb  (size: item_movieId_embedding_size)
  user_genre_tower([avg_rating_per_genre | watch_frac])     →  genre_emb    (size: user_genre_embedding_size)
  user_tag_tower(avg_tag_vector[watch_history])             →  tag_emb      (size: user_tag_embedding_size)   [optional]
  timestamp_embedding_tower(watch_month)                    →  ts_emb       (size: timestamp_feature_embedding_size)
  concat → user_combined

Item Tower:
  item_genre_tower(genre_onehot)        →  item_genre_emb       (size: item_genre_embedding_size)
  item_tag_tower(tag_vector)            →  item_tag_emb         (size: item_tag_embedding_size)         [optional]
  item_genome_tag_tower(genome_scores)  →  item_genome_tag_emb  (size: item_genome_tag_embedding_size)  [optional]
  item_embedding_tower(movie_id)        →  item_emb             (size: item_movieId_embedding_size)
  year_embedding_tower(release_year)    →  year_emb             (size: item_year_embedding_size)
  concat → item_combined

Prediction: dot_product(user_combined, item_combined)
```

All towers use `nn.Linear → Tanh`. Weights initialized with Xavier uniform (gain=0.01).

### Tower flags

Three independent boolean flags in the hyperparameter cell:

```python
USE_USER_TAG_TOWER        = False   # user tag profile (avg of watched movies' user-applied tag vectors)
USE_ITEM_TAG_TOWER        = True    # item tower: user-applied tags (tags.csv, 306 tags)
USE_ITEM_GENOME_TAG_TOWER = True    # item tower: genome relevance scores (genome-scores.csv, 1128 tags)
```

### Critical dimension constraint

`len(user_combined) == len(item_combined)` must hold for the dot product. The model raises `ValueError` at construction if violated.

- user side = `item_movieId_embedding_size + user_genre_embedding_size + timestamp_feature_embedding_size + (user_tag if enabled else 0)`
- item side = `item_genre_embedding_size + (item_tag if enabled else 0) + (genome_tag if enabled else 0) + item_movieId_embedding_size + item_year_embedding_size`

The genre towers absorb freed dims automatically:
```python
user_genre_embedding_size = 50 - user_tag_embedding_size
item_genre_embedding_size = 50 - item_tag_embedding_size - item_genome_tag_embedding_size
```

### Default embedding sizes (current)

```python
item_movieId_embedding_size      = 40   # fixed
item_year_embedding_size         = 10   # fixed
timestamp_feature_embedding_size = 10   # fixed

user_tag_embedding_size        = 20   # if USE_USER_TAG_TOWER else 0
item_tag_embedding_size        = 15   # if USE_ITEM_TAG_TOWER else 0
item_genome_tag_embedding_size = 25   # if USE_ITEM_GENOME_TAG_TOWER else 0

user_genre_embedding_size = 50 - user_tag_embedding_size           # = 50 (with user_tag off)
item_genre_embedding_size = 50 - item_tag_embedding_size - item_genome_tag_embedding_size  # = 10
# user: 40 + 50 + 10 + 0  = 100
# item: 10 + 15 + 25 + 40 + 10 = 100  ✓
```

### Key implementation notes

- **Shared item embedding:** `item_embedding_lookup` is shared between the item tower (target movie) and the user history avg pool. The lookup has `padding_idx = top_movies_len`.
- **Rating-weighted avg pool:** Each watched movie embedding is weighted by the user's debiased rating. Abs-value normalization prevents negative ratings from cancelling positive ones in the denominator.
- **Removing `item_embedding_tower` causes severe genre clustering** — do not remove it.

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

```python
# Save (done automatically by training loop)
torch.save(model.state_dict(), 'saved_models/best_checkpoint_{run_timestamp}.pth')

# Load — must instantiate model with same flags and sizes as when it was saved
model = MovieRecommender(
    genres_len=len(genres),
    tags_len=len(final_movie_tags),
    genome_tags_len=len(final_movie_genome_tags),
    top_movies_len=len(top_movies),
    all_years_len=len(year_to_num_movies),
    timestamp_num_bins=timestamp_num_bins,
    user_context_size=user_context_size,
    item_genre_embedding_size=item_genre_embedding_size,
    item_tag_embedding_size=item_tag_embedding_size,
    item_genome_tag_embedding_size=item_genome_tag_embedding_size,
    item_movieId_embedding_size=item_movieId_embedding_size,
    item_year_embedding_size=item_year_embedding_size,
    user_genre_embedding_size=user_genre_embedding_size,
    user_tag_embedding_size=user_tag_embedding_size,
    timestamp_feature_embedding_size=timestamp_feature_embedding_size,
    use_user_tag_tower=USE_USER_TAG_TOWER,
    use_item_tag_tower=USE_ITEM_TAG_TOWER,
    use_item_genome_tag_tower=USE_ITEM_GENOME_TAG_TOWER,
)
model.load_state_dict(torch.load(PATH, weights_only=True))
model.eval()
```

Checkpoints are weights-only (~500–600KB). The `saved_models/` directory is gitignored.

## Notebook Versions

| Notebook | Description |
|---|---|
| `MovieLens_Two_Tower_Embedding_NN.ipynb` | v1 — raw tensor ops, no nn.Module |
| `MovieLens_Two_Tower_Embedding_NN-v2.ipynb` | v2 — refactored to nn.Module, adds tags/year/timestamp features |
| `MovieLens_Two_Tower_Embedding_NN_v3_user_embedding_avg_pooling.ipynb` | **v3 (latest)** — rating-weighted avg pooling over watch history embeddings, genome tag tower |

The key v3 innovation: user watch history is represented as the **rating-weighted mean of watched movie embeddings** (with padding mask). Each movie embedding is weighted by the user's debiased rating (abs-value normalized). This is more expressive and handles variable-length histories naturally.

## Known Issues

- **Fantasy Lover** drifts to arthouse/surreal films (Stalker, Wings of Desire, City of Lost Children) — persistent open problem
- **Sci-Fi Lover** drifts to "cerebral/surreal" cluster (Videodrome, Eraserhead, Naked Lunch) — prestige sci-fi shares too many signals with arthouse cinema
- **War Lover** drifts to epic adventure/blockbuster (Star Wars, Dark Knight) — structural issue with "acclaimed serious film" cluster
- **Thriller Lover** drifts to disaster/action movies — Thriller genre is too broad to separate psychological from action-thriller
- **Crime Lover** drifts to comedy — Pulp Fiction and Fargo carry Comedy genre tag which dominates
- Removing `item_embedding_tower` causes severe genre clustering — do not remove it

## Future User Tower Improvements

Roughly ordered by implementation cost:

1. **User genome tag affinity context** — analogous to the genre context vector, compute rating-weighted avg genome tag relevance scores over a user's watch history. Mirrors the item-side genome tag tower. Likely high impact since item side already uses genome scores.
2. **Recency-weighted pooling** — exponential decay so recent watches matter more than old ones
3. **Rating variance per genre** — consistency signal (always loves horror vs. sometimes likes it)
4. **Explicit dislikes** — low-rated movies (1–2 stars) pooled separately as a negative taste embedding
5. **Short-term vs. long-term history** — two pooled embeddings (e.g., last 10 vs. all history) concatenated
6. **Transformer over history** — replace avg pooling with a small Transformer encoder; `[CLS]` token becomes the history embedding

## Git Workflow

Never commit and push in the same command. Always commit first, then ask before pushing.

## Canary Users for Eval

Use these synthetic users to quickly assess model quality after training:
- **Horror Lover** and **Sci-Fi Lover** — most sensitive to genre drift; if these are wrong, the model is failing
- **Comedy Lover** and **Romance Lover** — tend to work well across all runs; good sanity checks
- **Thriller Lover** and **Crime Lover** — stress tests for edge cases; expect imperfect results due to genre overlap
