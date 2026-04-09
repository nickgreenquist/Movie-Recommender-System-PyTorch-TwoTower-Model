# Plan: Build Books Two-Tower Model, Then Unify Architecture

## Approach

Build the books pipeline end-to-end first — as a parallel, self-contained implementation.
Validate it works (canary eval, embedding probes, sensible recommendations). Then, with two
working pipelines to compare, unify only what's demonstrably duplicated.

Premature generalization risks refactoring working movie code based on assumptions about
what books needs. The right seams for splitting `preprocess.py`, what FeatureStore needs
to be generic, which user tower components books actually uses — all become obvious once
both pipelines exist.

---

## Phase 1: Books Pipeline (self-contained, no changes to movie code)

### Dataset

Target: **Goodreads Book Graph** (UCSD, public).

Schema mapping to existing pipeline:

| Pipeline concept | Movies (MovieLens) | Books (Goodreads) |
|---|---|---|
| Item ID | `movieId` | `book_id` |
| Title | title string | title string |
| Categories | genres (pipe-sep) | shelves / genres |
| Year | parsed from title `(YYYY)` | `original_publication_year` column |
| User tags | tags.csv (user-applied) | shelves applied per user |
| Genome equivalent | genome-scores.csv (1,128 tags) | shelf-relevance scores (computed) |
| Ratings | userId, movieId, rating, timestamp | user_id, book_id, rating, date_added |

**Genome equivalent for books:** shelf-relevance score = how strongly a shelf characterizes a
book across all users (fraction of users who shelved the book who applied that shelf).
Analogous to MovieLens genome scores. Computed during preprocessing from aggregated shelf data.

### New files (touch nothing in existing `src/`)

```
books/
├── preprocess.py   # load Goodreads CSVs → same 7 base_*.parquet schema
├── features.py     # identical logic to src/features.py, books column names
├── dataset.py      # identical logic to src/dataset.py, books FeatureStore
├── model.py        # identical logic to src/model.py (or direct import — see note)
├── train.py        # identical logic to src/train.py
├── evaluate.py     # canary users defined with book titles
└── main.py         # python books/main.py preprocess|features|train|canary
```

> **Note on model.py:** `src/model.py` has no MovieLens-specific code — it can be imported
> directly by `books/train.py` without copying. Only `preprocess.py`, `evaluate.py` (canary
> defs), and the FeatureStore field names are movie-specific.

### Books canary users (examples)

```python
USER_TYPE_TO_FAVORITE_GENRES = {
    'Mystery Lover':   ['Mystery', 'Thriller'],
    'Fantasy Lover':   ['Fantasy'],
    'Romance Lover':   ['Romance'],
    'Sci-Fi Lover':    ['Science Fiction'],
}
USER_TYPE_TO_FAVORITE_BOOKS = {
    'Mystery Lover':   ['Gone Girl', 'The Girl with the Dragon Tattoo'],
    'Fantasy Lover':   ['The Name of the Wind', 'The Way of Kings'],
    ...
}
USER_TYPE_TO_GENOME_TAGS = {
    'Mystery Lover':   ['mystery', 'suspense', 'crime'],
    'Fantasy Lover':   ['magic', 'epic fantasy', 'world-building'],
    ...
}
```

### Verification (Phase 1 complete when all pass)

1. `python books/main.py preprocess` — produces `data/books/base_*.parquet`
2. `python books/main.py features` — produces `data/books/features_*.parquet`
3. `python books/main.py train` — trains, val loss decreases, checkpoint saved
4. `python books/main.py canary` — sensible book recommendations per canary user
5. `python books/main.py probe` — embedding probes show genre/tag separation

---

## Phase 2: Unify Architecture (after Phase 1 is working)

With two working pipelines, identify what's actually duplicated vs. what genuinely differs.
Expected outcome based on current knowledge — but verify against reality first:

**Likely truly generic (move to `src/`):**
- Training loop, optimizer, loss (`train.py`)
- Export/serialization (`export.py`)
- Feature math: one-hot encoding, tag frequency, genome score normalization (`features.py`)
- Tensor construction, padding, bucketizing (`dataset.py`)
- Inference logic: `_build_user_embedding()`, `run_canary_eval()`, probe functions (`evaluate.py`)
- Model architecture (`model.py` — already generic, minor rename)

**Likely domain-specific (stays in `domains/movies/` and `domains/books/`):**
- CSV loading and schema mapping (`preprocess.py`)
- Canary user definitions with item titles (`evaluate.py` canary dicts)

**Defer until Phase 2:**
- Whether user tower (explicit genre + genome towers) should generalize or collapse to pure pooling
- FeatureStore field naming (`top_movies` → `top_items` etc.)
- `--domain` flag routing in `main.py`
- `MovieRecommender` → `TwoTowerRecommender` rename

These decisions are best made with both pipelines in hand.
