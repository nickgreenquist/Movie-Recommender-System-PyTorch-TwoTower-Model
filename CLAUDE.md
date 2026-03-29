# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A PyTorch Two-Tower neural network recommender system trained on the MovieLens 32M dataset. The model predicts ratings via dot product of user and item embeddings. There is no user ID embedding — users are represented entirely by their watch history (average pooling of watched movie embeddings) and genre affinity (average rating per genre). This makes the model cold-start-friendly and generalizable to new users.

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
- `genome-scores.csv` — tag relevance scores per movie
- `genome-tags.csv` — tag index

Only movies with 1,000+ ratings are kept (~2,070 movies). Only users with 20–500 ratings are kept.

## Model Architecture (v3 — canonical)

**Two-Tower design** with dot product prediction:

```
User Tower:
  avg_pool(item_embeddings[watch_history])  →  history_emb (size: item_movieId_embedding_size)
  user_genre_tower(avg_genre_ratings)       →  genre_emb   (size: user_genre_embedding_size)
  timestamp_embedding(watch_month)          →  ts_emb      (size: timestamp_feature_embedding_size)
  concat → user_combined

Item Tower:
  item_genre_tower(genre_onehot)            →  item_genre_emb  (size: item_genre_embedding_size)
  item_tag_tower(tag_vector)               →  item_tag_emb    (size: item_tag_embedding_size)
  item_embedding_tower(movie_id)           →  item_emb        (size: item_movieId_embedding_size)
  year_embedding_tower(release_year)       →  year_emb        (size: item_year_embedding_size)
  concat → item_combined

Prediction: dot_product(user_combined, item_combined)
```

**Critical dimension constraint:** `len(user_combined) == len(item_combined)` must hold for the dot product. In v3:
- user side = `item_movieId_embedding_size + user_genre_embedding_size + timestamp_feature_embedding_size`
- item side = `item_genre_embedding_size + item_tag_embedding_size + item_movieId_embedding_size + item_year_embedding_size`

Default embedding sizes (v3): item_genre=10, item_tag=40, item_movieId=40, item_year=10, user_genre=50, timestamp=10 → both sides = 100.

## Training Details

- **Loss**: MSE on de-biased ratings (user mean subtracted)
- **Best run**: LR=0.005, 150K steps — best val checkpoint at step 110K (~0.749 val loss)
- **LR=0.01**: Too aggressive; collapses genre boundaries — avoid
- **Train/val split**: 90/10 user-based; 90% of each user's history as context, 10% as labels

## Saving / Loading Models

```python
# Save
torch.save(model.state_dict(), 'saved_models/YYYYMMDD.pth')

# Load
model.load_state_dict(torch.load('saved_models/YYYYMMDD.pth'))
model.eval()
```

Checkpoints are weights-only (~500–600KB). The `saved_models/` directory is gitignored.

## Notebook Versions

| Notebook | Description |
|---|---|
| `MovieLens_Two_Tower_Embedding_NN.ipynb` | v1 — raw tensor ops, no nn.Module |
| `MovieLens_Two_Tower_Embedding_NN-v2.ipynb` | v2 — refactored to nn.Module, adds tags/year/timestamp features |
| `MovieLens_Two_Tower_Embedding_NN_v3_user_embedding_avg_pooling.ipynb` | **v3 (latest)** — avg pooling over watch history embeddings |

The key v3 innovation: user watch history is represented as the **unweighted mean of watched movie embeddings** (with padding mask) rather than categorical features. This is more expressive and handles variable-length histories naturally.

## Known Issues

- **Horror Lover** synthetic user drifts to quirky art films (Big Lebowski, Delicatessen) — persistent across all runs
- **Fantasy Lover** skews toward Studio Ghibli animated films rather than live-action epic fantasy
- **War Lover** gets tonally coherent serious dramas (Goodfellas, Wolf of Wall Street) rather than strict war films
- Removing `item_embedding_tower` causes severe genre clustering — do not remove it
- Fantasy/Horror genre drift is a known open problem

## Future User Tower Improvements

Roughly ordered by implementation cost:

1. **Rating-weighted pooling** — weight each movie embedding by the user's rating instead of equal weights
2. **Recency-weighted pooling** — exponential decay so recent watches matter more than old ones
3. **Rating variance per genre** — consistency signal (always loves horror vs. sometimes likes it)
4. **Explicit dislikes** — low-rated movies (1–2 stars) pooled separately as a negative taste embedding
5. **Short-term vs. long-term history** — two pooled embeddings (e.g., last 10 vs. all history) concatenated
6. **Tag-based user profile** — average the tag vectors of all watched movies (weighted by rating), mirroring the item tag tower
7. **Transformer over history** — replace avg pooling with a small Transformer encoder; `[CLS]` token becomes the history embedding

## Git Workflow

Never commit and push in the same command. Always commit first, then ask before pushing.

## Canary Users for Eval

Use these synthetic users to quickly assess model quality after training:
- **Horror Lover** and **Sci-Fi Lover** — most sensitive to genre drift; if these are wrong, the model is failing
- **Comedy Lover** and **Romance Lover** — tend to work well across all runs; good sanity checks
