# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Current State

**Prod checkpoint:** `best_softmax_genome_tags_llm_features_popularity_alpha_05_20260607_195227.pth` (Model D — full softmax, L2 norm, Menon α=0.5, 4-pool user tower, 128-dim output; `feature_towers='both'` — **genome + LLM** content towers fused). Do not replace without a clearly better eval result.

Promoted 2026-06-07, replacing the genome-only v3 prod (`best_softmax_v2_popularity_alpha_05_20260505_182728.pth`, now demoted). **Deliberate, portfolio-motivated lateral swap:** one deployed model that fuses the dataset's curated genome tags with my own web-scraped + LLM-extracted features. On par with the old prod — MRR 0.1123 vs 0.1144 (−1.8%) at the canonical n=382,138 protocol, identical deep-tail MRR (0.0159), canary on par across personas (comparison table under Offline Evaluation). Trades a negligible aggregate-MRR cost for the fused-feature story — not a metrics upgrade, intended.

To re-export serving artifacts from prod checkpoint:

```bash
python main.py export saved_models/best_softmax_genome_tags_llm_features_popularity_alpha_05_20260607_195227.pth
```

Export bakes the LLM feature buffer into `serving/feature_store.pt` (the `data/llm_features_*.pt` tensor is gitignored and absent on Streamlit Cloud), so the deployed app reconstructs the genome + LLM user tower from serving artifacts alone — `streamlit_app.py` does not read `data/`.

## Project Overview

A PyTorch Two-Tower neural network recommender system trained on the MovieLens 32M dataset.

**Critical design choice: no user ID embedding.** Users are represented entirely by their taste signals: 4-pool watch history (full, liked, disliked, rating-weighted sum pools of item ID embeddings), genre affinity, genome context, and timestamp. Any user can be represented at inference time with just a few movies they liked — no retraining required.

## Running the Code

The project is a Python CLI (`main.py`). Notebooks in `archive/notebooks/` are archived references only — the canonical code is in `src/`.

```bash
python main.py preprocess          # Stage 1: raw CSVs → data/base_*.parquet
python main.py features            # Stage 2: base parquets → data/features_*.parquet
python main.py dataset             # Stage 3: features → data/dataset_mse_rollback_*_v1.pt
python main.py train softmax       # Stage 4: full softmax training (canonical)
python main.py train               # Stage 4: MSE training (legacy, not prod)
python main.py canary              # Canary user recommendations (most recent checkpoint)
python main.py canary <path>       # Canary user recommendations (specific checkpoint)
python main.py probe               # Embedding probes (most recent checkpoint)
python main.py probe <path>        # Embedding probes (specific checkpoint)
python main.py eval                # Offline eval: Recall@K, NDCG@K, Hit Rate@K, MRR
python main.py eval <path>         # Same, specific checkpoint
python main.py export              # Stage 5: export serving artifacts for Streamlit
python main.py export <path>       # Export using specific checkpoint
python main.py posters             # Fetch movie poster URLs from TMDB → serving/posters.json
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

**Phase 1 reduced corpus (LLM-vs-genome ablation).** `llm_features/filter_corpus.py` applies a higher raw-ratings threshold — **>1,000 ratings → 4,461 movies** (a clean subset of the full corpus) — and writes the list to `data/llm_experiment_movies_phase1.json`. Counting matches `build_corpus()` (raw `ratings.csv` counts, strict `>`), so the threshold is directly comparable to the full corpus's `> 200`. This is the reduced set the Phase 1 pilot scrapes/extracts/trains on; `src/corpus.py` (`CORPUS=phase1`) namespaces the resulting artifacts. See `docs/plans/llm_vs_genome_ablation_plan.md` (Stage 0).

### Tag data

**User-applied tags (`tags.csv`):** Each movie's tag vector is built by counting how many users applied each tag, then normalizing to sum to 1. Only tags applied 1,000+ times across all movies are kept (306 tags survive).

**Genome tags (`genome-scores.csv`):** ML-derived relevance scores (0.0–1.0) per (movie, tag) pair. 1,128 tags. Much denser and more semantically meaningful than user-applied tags — movies get non-zero scores for relevant tags even when users didn't explicitly apply them.

## Model Architecture (canonical — v3 prod)

```
User Tower (4-pool):
  sum_pool(item_embedding_lookup[full_history])            →  pool_full      (32)  [LayerNorm]
  sum_pool(item_embedding_lookup[liked_history])           →  pool_liked     (32)  [LayerNorm]
  sum_pool(item_embedding_lookup[disliked_history])        →  pool_disliked  (32)  [LayerNorm]
  rating_weighted_sum(item_embedding_lookup[full_history]) →  pool_weighted  (32)  [LayerNorm]
  user_content_tower(rating_weighted_avg(raw_genome[history]))  →  content_ctx  (32)
  user_genre_tower([avg_rating_per_genre | watch_frac])   →  genre_emb      (32)
  timestamp_embedding_tower(watch_month)                  →  ts_emb         (4)
  concat (196) → Linear(256) → ReLU → Linear(128) → L2-normalize → user_emb (128)

Item Tower:
  item_genre_tower(genre_onehot)        →  item_genre_emb   (8)
  item_tag_tower(tag_vector)            →  item_tag_emb     (16)
  item_content_tower(genome_scores)  →  item_content_emb  (32)
  item_embedding_tower(movie_id)        →  item_emb         (32)  [shared lookup with all 4 user pools]
  year_embedding_tower(release_year)    →  year_emb         (8)
  concat (96) → Linear(256) → ReLU → Linear(128) → L2-normalize → item_emb (128)

Prediction: dot_product(user_emb, item_emb) = cosine similarity (both L2-normalized)
```

Sub-tower linears: Xavier uniform `gain=0.1`. Projection linears re-initialized at `gain=1.0` after the rest of the model — without this, `gain=0.1²` compounds and collapses dot products to zero before training starts.

### Key implementation notes

- **4-pool user tower:** Full, liked, disliked, and rating-weighted sum pools all operate on `item_embedding_lookup` (32-dim raw ID embedding). Each has its own LayerNorm. The liked/disliked pools are pre-computed in the dataset builder as right-aligned padded tensors; the model receives them directly.
- **Shared item embedding:** `item_embedding_lookup` is shared between the item tower and all four user history pools. Removing `item_embedding_tower` (the Linear+ReLU on top of the lookup) causes severe genre clustering — do not remove it.
- **Content context tower:** `user_content_tower` runs a single Linear(1128→32) over the rating-weighted average of raw genome scores across the user's full history. This is a compact dense content fingerprint. It is NOT a pool over `item_content_tower` outputs — do not confuse the two.
- **`item_content_tower` is item-side only.** It is no longer shared with the user tower. It compresses per-movie genome scores (1128→32) for the item embedding.
- **Swappable content slot:** `item_content_tower` / `user_content_tower` are a swappable slot set by `content_feature_source` (`'genome'` | `'llm'` | `None`), filled by **genome** in prod. `None` omits both towers and the buffer (the ablation's no-content baseline). Legacy checkpoints load via `LEGACY_KEY_REMAP` in `src/checkpoint.py`. The genome *product feature* (probes, anchors, Explore-Genome tab) stays genome-named — only the slot was renamed.
- **Genome sub-tower size matters:** genome compresses 1,128 → 32 dims. Do not shrink below 32 — information is lost before the projection MLP can mix it. Tested at 16, was noticeably worse.
- **L2 normalization:** both towers output unit-norm vectors. Dot product = cosine similarity. No post-hoc popularity correction needed at inference.

## Training Details

### V3 Softmax (`python main.py train softmax`) — canonical, prod

- **Loss**: full softmax cross-entropy over all ~9,375 corpus items
- **Optimizer**: Adam, `lr=0.001`, `weight_decay=0.0`, `adam_eps=1e-6`
- **Batch size**: 512, **Temperature**: 0.1, **Steps**: 150,000
- **L2 normalization** at end of both towers
- **Popularity bias correction (Menon et al. 2021):** logit-adjusted loss — **add** `alpha * log(count_i)` to item i's logit before softmax. `alpha=0.5` chosen (see alpha comparison below). **Do NOT subtract** — wrong sign.
- **Inference:** raw dot products — no post-hoc correction. Menon α is training-only.
- **Config sidecar:** `alpha` and `temperature` saved as JSON alongside each checkpoint.
- **Dataset**: rollback examples — for each watch event, context = all prior watches (chronological). 7-tuple: `(X_genre, X_history, X_hist_liked, X_hist_disliked, X_hist_ratings, timestamp, target_movieId)`. Capped at `MAX_SOFTMAX_EXAMPLES_PER_USER` per user.
- **Checkpointing**: `saved_models/best_softmax_<timestamp>.pth`; periodic checkpoints every 30,000 steps.

### Alpha selection (Menon popularity correction)

**Workflow rule — alpha is a deployment-time knob, never a training-search knob.** Always train at `alpha=0` while iterating on model/architecture: it gives the best offline MRR and the cleanest signal for comparing variants, and it is the setting all ablation/experiment runs use. Only once a new best model/architecture is locked in do you fine-tune `alpha` — sweep values, pick the one with the best **canary** results — before deployment. The committed `get_config` default is therefore `alpha=0.0` (search mode); the `alpha=0.5` below is the fine-tuned value chosen for the *current prod model*, not the training default.

`alpha=0.5` chosen over alternatives:
- `alpha=0.0` — best offline MRR but severe popular drift on canary (War/Fantasy/Heist/Crime collapse to IMDb top-10)
- `alpha=0.5` — small offline MRR cost (−4.9% vs alpha=0) buys clean genre discrimination
- `alpha=1.0` — over-corrects to obscure/low-quality items

### MSE rollback (`python main.py train rollback`) — legacy, not prod

- **Loss**: MSE on de-biased ratings (raw rating − user mean)
- **Optimizer**: SGD, `lr=0.005`, `momentum=0.9`
- **Batch size**: 64, **Steps**: 300,000
- **Adam is a failure for MSE training — do not use.** Causes user tower collapse. SGD with momentum generalizes better.
- **Dataset build**: `python main.py dataset rollback` → `data/dataset_mse_rollback_*_v1.pt`
- **Canonical dataset cap**: 20 examples/user. Tested 30 — null result, +4GB RAM not worth it.
- **Always use timestamp sort** (`sort_by_ts=True`). Shuffle biases training toward popular items.

## Saving / Loading Models

Checkpoints are weights-only (~1MB). (Checkpoints saved before the genome context buffer was made non-persistent are ~42MB until re-saved/re-trained — the prod checkpoint above is one of these.) The `saved_models/` directory is gitignored.

Auto-detection (no path given) picks the most recently modified `best_*.pth` checkpoint.

The evaluate/export setup loads the checkpoint first (fast), then features (slow). Invalid checkpoint fails immediately before the slow features load.

`_resolve_config` in `src/evaluate.py` and `src/export.py` reads all embedding dimensions directly from the state_dict's weight shapes — no separate config file needed. Old and new checkpoints coexist.

## Offline Evaluation (`python main.py eval`)

`src/offline_eval.py` — Recall@K, NDCG@K, Hit Rate@K, MRR at K = 1, 5, 10, 20, 50, 100, 150, 200, 250.

**Rollback protocol** (default for all softmax checkpoints): For each val user (held out at user level), sample up to 20 chronological positions. Context = history[0..j-1], target = history[j]. All positions valid since val users were never seen in training.

Note: rollback eval is the harder protocol — compare only within the same protocol.

### Offline eval results — rollback protocol

| Metric | MSE flat (old) | MSE rollback proj | v2 softmax α=0.5 | **v3 softmax α=0.5 (PROD)** |
|---|---|---|---|---|
| Hit Rate@1 | 0.19% | 0.43% | 4.14% | **5.99%** |
| Hit Rate@5 | 0.94% | 1.66% | 11.73% | **15.49%** |
| Hit Rate@10 | 1.70% | 2.70% | 17.49% | **22.06%** |
| Hit Rate@20 | 2.93% | 4.28% | 25.00% | **30.44%** |
| Hit Rate@50 | 5.62% | 7.59% | 37.80% | **44.38%** |
| **MRR** | 0.0084 | 0.0135 | 0.0878 | **0.1153** |

v3 beats v2 prod by **+31% MRR**. v3 beats MSE prod by **+8.7× MRR**.

### Prod swap: genome-only → genome+LLM (Model D) at α=0.5 — canonical protocol

2026-06-07 the genome-only v3 prod was swapped for the **Model D** (genome + LLM, `feature_towers='both'`)
α=0.5 checkpoint. Deliberate, portfolio-motivated lateral swap — one deployed model fusing the
dataset's curated genome tags with my own web-scraped + LLM-extracted features — verified on par with
the old prod, not a metrics upgrade. **Different protocol from the legacy table above** (this is the
canonical rollback eval, all 19,134 val users, n=382,138), so it is a separate table, not an edit of
those numbers.

| Metric        | OLD prod genome α0.5 | NEW prod D α0.5 (genome+LLM) |
|---|---|---|
| Hit Rate@1    | 0.0592 | 0.0573 |
| Hit Rate@5    | 0.1538 | 0.1513 |
| Hit Rate@10   | 0.2191 | 0.2173 |
| Hit Rate@20   | 0.3020 | 0.3009 |
| Hit Rate@50   | 0.4425 | 0.4413 |
| TAIL ≤1k MRR  | 0.0159 | 0.0159 |
| **MRR**       | **0.1144** | **0.1123** |

Aggregate MRR −1.8%; deep-tail (≤1k ratings) MRR identical (0.0159). Eval files:
`eval_results/best_softmax_v2_popularity_alpha_05_20260505_182728.txt` (old) and
`eval_results/best_softmax_genome_tags_llm_features_popularity_alpha_05_20260607_195227.txt` (new).

**Canary (vs old prod):** on par across personas — clean Horror / Sci-Fi / Comedy / Romance /
Children's / Anime; strong Arthouse / Superhero / Western; D **beats** old prod on WW2. Fantasy and
Heist are washes (both drift — Fantasy to sci-fi / low-quality, Heist to spy-action). Net: a lateral,
portfolio-motivated swap with negligible quality cost.

## Embedding Probes (`python main.py probe`)

**`probe_genre(genre)`** — one-hot → `item_genre_tower` → cosine vs all genre embeddings.

**`probe_tag(tags)`** — tag vector → `item_tag_tower` → cosine vs all tag embeddings.

**`probe_genome_tag(genome_tags)`** — averages top-k representative movie genome embeddings as query, cosine vs all genome embeddings. Uses real movie embeddings to avoid OOD synthetic inputs.

**`probe_similar(titles)`** — pairwise cosine similarity on `MOVIE_EMBEDDING_COMBINED` (128-dim). Most reliable probe.

## Canary Users for Eval

> **Timestamp:** All canary users receive `ts_max_bin` (most recent timestamp bin).

- **Horror Lover** and **Sci-Fi Lover** — most sensitive to genre drift; if wrong, model is failing
- **Comedy Lover** and **Romance Lover** — good sanity checks
- **WW2 Lover** — stress test for drift into non-Western cinema
- **Crime Lover** — stress test; expect some genre overlap

### Known issues (v3 prod)

- **Fantasy Lover** drifts to low-quality genre-adjacent films (Wrath of Titans, Hansel & Gretel) — anchor movies (LotR, Dune) sit in an ambiguous area of the embedding space
- **Western Lover** occasionally surfaces WW2 films (Cross of Iron, Paths of Glory) — war/western genome overlap
- **War Movie Lover** occasionally has one off-genre pick (e.g. Cosmos: A Spacetime Odyssey) — acceptable

### Improvements vs v2 prod (from canary)

- **Sci-Fi**: purer classic hard sci-fi (Metropolis, Fantastic Planet, Silent Running) vs v2's Pi/Fight Club drift
- **Horror**: higher-quality picks (Ring, Others, Session 9) vs v2's Annabelle/Insidious spam
- **War**: Band of Brothers, Pacific, Zulu vs v2's Red Dawn/Big Short drift
- **Crime, Arthouse, Martial Arts**: all improved
- **WW2**: no longer recommends Indian films (was a clear failure at early checkpoints; resolved by 105k steps)

## LLM-vs-Genome Ablation: Model → Checkpoint Map

The four ablation arms (A/B/C/D) are identical architecture + hyperparameters (α=0, full corpus,
150k steps, val-MRR checkpoint selection, trained 2026-06-07); the *only* difference is what fills
the content slot. **All four checkpoints (and their `eval_results/`/`canary_results/` files) were
renamed on 2026-06-07 to the current explicit-tower-families naming scheme** (commit `7e5b34a`), so
the filenames below are now self-consistent — A/B/C originally used a legacy scheme (see "History").

| Model | Content slot | `FEATURE_TOWERS` | Best checkpoint (`saved_models/`) | MRR\* |
|---|---|---|---|---|
| **A** | genome tags | `genome` | `best_softmax_genome_tags_popularity_alpha_0_20260607_101027.pth` | 0.1146 |
| **B** | LLM features | `llm` | `best_softmax_llm_features_popularity_alpha_0_20260607_105646.pth` | 0.1157 |
| **C** | none (floor) | `none` | `best_softmax_popularity_alpha_0_20260607_112755.pth` | 0.1143 |
| **D** | genome + LLM | `both` | `best_softmax_genome_tags_llm_features_popularity_alpha_0_20260607_131924.pth` | 0.1154 |

\*Whole-corpus MRR, canonical Phase 2 eval (all 19,134 val users, n=382,138). Per-model
eval/canary outputs live in `eval_results/` and `canary_results/` under the same stems. (A
5,000-user `eval` run reads ~1% higher — e.g. A=0.1163, B=0.1173, C=0.1158, D=0.1154 — same
C < A < B ordering; use the 19,134-user numbers above as canonical.)

**Config resolution does not depend on the filename.** `src/checkpoint.py:load_checkpoint`
resolves each model's towers from the *state_dict weight keys* (`item_genome_tag_tower` /
`item_llm_feature_tower`), so renaming a `.pth` is safe. The **one exception**: A and B are
"content-era" checkpoints whose state_dict still uses the generic `item_content_tower` keys — for
those, genome-vs-llm is read from the `_config.json` sidecar (`content_feature_source`, legacy key).
So the sidecar **must keep the same stem as its `.pth`**; the rename moved both together.

**History (why git/older artifacts show different names):**
- A/B/C `.pth` files (and their result txts) used to read `best_softmax_v2_…`,
  `best_softmax_v2_llm_…`, `best_softmax_v2_nocontent_…`. The `_v2` token and the unmarked-genome
  default are gone; the names now compose tower families like D always did.
- A/B/C **sidecar JSON still contains the legacy `content_feature_source` key internally** (only
  the *filenames* were changed, not the JSON contents). The loader reads both `feature_towers` and
  the legacy `content_feature_source`, so this is harmless.

**None of these four α=0 arms is itself prod.** Prod is now a separate deployment fine-tune of the
**D architecture** (genome + LLM, `feature_towers='both'`) at α=0.5 — see Current State; the swap was
portfolio-motivated, not driven by these metrics. Verdict
(full analysis in `docs/plans/llm_vs_genome_ablation_plan.md` Stage 6): B ≥ A on the popular head,
A ≈ B on the deep tail, D shows no clear additive benefit over the better single source.

## Architecture Experiment Log

### V3: 4-pool user tower (2026-05-05) — PROD

**Motivation:** The v2 user tower used a rating-weighted avg pool over `item_genome_tag_tower` outputs for each movie in history. This was expensive (Linear(1128→32) over 50 movies per batch example = ~924M multiply-adds/step) and couldn't distinguish liked from disliked watches. Replaced with 4 sum pools over the raw 32-dim `item_embedding_lookup` (no tower pass, just table lookup + sum):

- `pool_full` — unweighted sum, all history
- `pool_liked` — sum of items with positive debiased rating
- `pool_disliked` — sum of items with negative debiased rating
- `pool_weighted` — rating-weighted sum, all history

Each pool has its own LayerNorm. The `user_genome_context_tower` (raw genome fingerprint) was kept. The genome embedding pool (per-movie tower pass) was removed entirely.

User concat: 4×32 + 32 + 32 + 4 = 196 → proj(256) → 128.

**Result: +31% MRR (0.1153 vs 0.0878).** Clean canary across all personas at 105k/150k steps.

### Full-item-pool experiments (2026-04-25, MSE era) — null results

Tested pooling over the full 128-dim projected item embedding instead of the 32-dim ID lookup. Two variants:
1. Replace both ID pool and genome pool → arthouse attractor, all genres drifted
2. Replace ID pool only, keep genome pool → MRR 0.0510 vs prod 0.0521 (−2.1%)

**Conclusion: the 32-dim ID lookup is a better pooling signal than the 128-dim projected embedding.** The projection is trained to match item-side targets, making it noisier as a history-pooling signal. Do not revisit.

### What we know does NOT work

- Removing `item_embedding_tower` (Linear+ReLU on top of the ID lookup) — severe genre clustering
- Pooling over the full projected item embedding (128-dim) in the user tower — confirmed null across two variants
- Adam for MSE training — user tower collapse
- Shuffle in rollback dataset build — biases toward popular items
- Increasing dataset cap beyond 20 examples/user — null MRR, +4GB RAM

## Dataset

### Softmax dataset tuple format

7-tuple: `(X_genre, X_history, X_hist_liked, X_hist_disliked, X_hist_ratings, timestamp, target_movieId)`

- `X_hist_liked` / `X_hist_disliked`: right-aligned padded integer tensors (pad = `len(top_movies)`), built in `build_v2_softmax_dataset()`. Derived from `ctx_rats > 0` / `< 0` at each rollback position.
- Built in `src/dataset.py: build_v2_softmax_dataset()`.

### MSE dataset tuple format (legacy)

6-tuple: `(X_genre, X_history, X_history_ratings, timestamp, Y, target_movieId)`

### Memory

Dataset RAM: ~12 GB for the softmax dataset. Per-item feature tensors (genre, tag, genome, year for target movie) are NOT stored in the dataset — they're looked up from model buffers during training.

## MPS GPU Support

`get_device()` in `src/train.py` returns `mps > cuda > cpu`. Train/eval/inference all run on device. Export stays CPU-only — `src/export.py` uses `map_location='cpu'` when loading checkpoints, ensuring serving artifacts are device-agnostic for Streamlit Cloud.

**Critical:** Checkpoints saved during MPS training contain MPS tensor locations. `export.py` must use `map_location='cpu'` — without it, `model.pth` crashes on Linux (Streamlit Cloud has no MPS).

MPS training speed: ~145 it/s.

## Goodreads Comparison

The two-tower architecture was validated on Goodreads (user-driven data) where softmax gives 3× MRR over MSE. MovieLens is platform-driven (users rate what they were shown), which changes the objective tradeoff — but full softmax with popularity correction still dominates MSE on MovieLens.

| Metric | MSE | Softmax |
|---|---|---|
| Hit Rate@10 | 4.3% | **11.7%** |
| Hit Rate@50 | 16.8% | **31.3%** |
| MRR | 0.021 | **0.057** |

## Potential Next Improvements

1. **Rating variance per genre** in the user context vector — distinguishes genuine fans from casual watchers.
2. **Genre + year interaction in the item tower** — directly attacks era confusion in War/WW2/Western drift.
3. **Embedding size tuning** — larger genome/genre dims may improve representation capacity.

## Known Code Inconsistency

**Label movieIds stored as raw IDs, watch history as embedding indices** — `watch_history` is mapped to embedding indices in `features.py`, but `label_movieIds` are stored as raw IDs and mapped later in `dataset.py`. Both work correctly but the inconsistency is confusing.

---

## Working Style and Guidelines

### Git workflow

Never commit and push in the same command. Always commit first, then ask before pushing.

For changes that require retraining to validate (hyperparameters, optimizer, scheduler, loss, dataset logic, model architecture): write the code, then stop. Do not commit until the user has run training and confirmed the results look better.

**After any model/training change: always wait for the user to run `python main.py train`, then `python main.py canary`, then `python main.py eval`, and confirm the results are acceptable before committing anything.**

**Full training runs go to the user — never launch `python main.py train` via a Claude-spawned background process or a Workflow.** Claude-launched background jobs run ~10× slower and degrade over the run (observed ~6 it/s and falling, vs a steady ~67 it/s in the user's own terminal). The cause is macOS deprioritizing detached/background-spawned processes (low QoS / App Nap), **not** thermal — a foreground run on the same machine does not throttle. Claude writes and smoke-tests the training code, then hands the exact `CONTENT_SOURCE=… CORPUS=… python main.py train …` command(s) to the user to run themselves. Shorter MPS jobs (`eval`, `canary`, `probe`) are fine to run in the background — it's the long training loop where the slowdown compounds.

### Behavioral guidelines

These supplement (not replace) the Claude Code system prompt. The standard "don't speculate, don't over-abstract, don't over-comment" rules already live there — what follows is project-specific or worth re-stating because it's bitten us before.

- **Match the existing style.** This codebase has a strong house style: long docstring headers on every util, NamedTuple bundles for related buffers, multi-paragraph comment banners on training-loop functions, named slice offsets instead of magic numbers, parquet column comments that line up vertically. When you add code, match what's around it even if you'd structure it differently.

- **Surgical changes.** Touch only what the task requires. Don't "improve" adjacent code, comments, or formatting. If you notice unrelated dead code or a refactor opportunity, mention it — don't act on it. The test: every changed line should trace directly to the user's request.

- **Verification belongs to the user for model changes.** The Git workflow above is the contract: any change that affects training behavior gets the code-only treatment. You verify imports compile, shapes match, smoke-test passes — the user verifies metrics. Don't claim success on a model/dataset change based on smoke tests alone, and don't update results tables in CLAUDE.md or the implementation plan until the user reports numbers back.

- **Surface tradeoffs early, in one or two sentences.** If multiple interpretations of a request exist, name them briefly and pick the one you think is right with a stated assumption — don't silently choose, and don't open a multi-option AskUserQuestion for routine calls. Default to action with a stated assumption; the user can redirect.

- **Use TaskCreate for multi-step work, not text-form plans.** When a task has 3+ steps with their own verifications (e.g. "implement Bucket N"), track them with TaskCreate/TaskUpdate so progress is visible in the UI. Don't also write the plan out as inline prose or a separate `.md` file — that's redundant.

**Do not touch `streamlit_app.py` or `src/export.py` until a model change is verified good.** The workflow is: train → canary → eval → if better, then update export/streamlit.
