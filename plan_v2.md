# Recommender System V2 Plan

This file is used to plan a major next version of the recommender system. It is based on industry proven recommender systems.

## Key Reference: Book Recommender
**Before implementing any part of v2, read `/Users/nickgreenquist/Documents/Book-Recommender-System-PyTorch-TwoTower-Model/src/` first.** The book recommender already uses this v2 approach (full softmax, Adam, L2 norm, ReLU) on a structurally similar dataset (ratings, shelves ≈ genome, genres). It has optimized implementations across preprocess, features, dataset, model, and train that should be ported/adapted rather than written from scratch.

## Plan Overview

### Model Changes
- **Adam optimizer** — `lr=0.001`, `weight_decay=0.0`, `adam_eps=1e-6`
- **Full softmax** instead of in-batch negatives (corpus is small enough at ~9k items)
- **ReLU** for all activations (replace TanH)
- **L2 Norm Layer** — applied after user and item towers; dot product of L2-normalized vectors = cosine similarity
- **Gradient clipping** — 1.0
- **Temperature** — fixed at 0.1 (tunable later; same value used in game and book recommenders)
- **Cosine LR schedule** — CosineAnnealingLR, `eta_min=1e-4`, `T_max=training_steps`
- **No weight decay** (L2 norm layer provides sufficient regularization)
- **No LayerNorm on history pools** — L2 norm at tower output is sufficient
- **Save model config** — save config as JSON sidecar alongside checkpoint `.pth`

### Hyperparameters
```python
'lr':             0.001,
'weight_decay':   0.0,
'adam_eps':       1e-6,
'minibatch_size': 512,
'training_steps': 150_000,
```

### User Tower (unchanged from v1)
Same pooling (rating-weighted avg), same concat, same signals — do not touch pooling until the BIG CHANGE below.
`id_pool(32) + genome_pool(32) + genre_emb(32) + ts_emb(4) + genome_ctx(32) → proj MLP → 128`
Shared `item_embedding_lookup` between item tower and history pool. Item ID embedding also concatenated in item tower.

### Training Loop — Full Softmax Implementation Note
Full softmax requires all 9k item embeddings every training step to build the `(batch_size × 9375)` score matrix. The item tower must be run over the full corpus each step as a **single batched forward pass** (same pattern as `build_movie_embeddings` in `evaluate.py` — do NOT loop per item). This is the dominant cost per step; on MPS it is fast but must be batched.

### Dataset Changes
Dataset must be completely rebuilt — the label structure changes. Core concept unchanged: for every user, sort watches by timestamp and create rollback examples (same `MAX_MSE_ROLLBACK_EXAMPLES_PER_USER` cap). Changes are on the **label side** only:
- **Positive targets** — movies rated **above** the user's avg rating only. Low-rated watches are NOT targets.
- **Dataset order** — keep timestamp sort (`sort_by_ts=True`). Shuffle introduced massive popularity bias (+7.7% relative increase in top-20 label frequency). Do not change.

---

## To Try After V2 is Trained/Evaled/Canaried

### [BIG CHANGE] Quadruple-History Pools
Replace the single rating-weighted avg pool with four separate **sum** pools over item ID embeddings (shared lookup):
1. **Liked pool** — watches rated above user avg
2. **Disliked pool** — watches rated below user avg
3. **Full history pool** — all watches, unweighted sum
4. **Rating-weighted pool** — all watches, weight = `rating - avg_rating`; denominator = `abs(weights).sum()`

User tower concat becomes: `liked_pool + disliked_pool + full_pool + rated_pool + genome_pool + genre_emb + ts_emb + genome_ctx → proj MLP → 128`