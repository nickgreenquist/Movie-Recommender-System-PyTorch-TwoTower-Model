# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project Overview

A PyTorch Two-Tower neural network recommender trained on the MovieLens 32M dataset. Python CLI (`main.py`); canonical code in `src/` (notebooks in `archive/notebooks/` are archived references only).

**Purpose:** this is a portfolio piece to showcase ML/recsys mastery to an ML engineer or CTO (and may be demoed live in ML-engineering interviews) — weigh every feature by whether it sharpens that signal.

**Critical design choice: no user ID embedding.** Users are represented entirely by taste signals — 4-pool watch history (full, liked, disliked, rating-weighted sum of item ID embeddings), genre affinity, genome context, timestamp. Any user can be represented at inference from just a few liked movies — no retraining.

## Current State (prod)

**Prod checkpoint:** `PROD_best_softmax_genome_tags_llm_features_popularity_alpha_05_20260612_080719.pth` — Model D: full softmax, L2 norm, Menon α=0.5, 4-pool user tower, 128-dim, `feature_towers='both'` (genome + LLM content towers fused). Trained 250k steps on the **corrected** LLM feature tensor (the 141-movie misalignment fix); whole-corpus rollback MRR 0.1132 (vs 0.1123 for the prior prod). Promoted & deployed 2026-06-12; prior prod is now `OLD_PROD_..._20260607_195227.pth`. Do not replace without a clearly better eval result.

Re-export serving artifacts:
```bash
python main.py export saved_models/PROD_best_softmax_genome_tags_llm_features_popularity_alpha_05_20260612_080719.pth
```
Export bakes the LLM feature buffer into `serving/feature_store.pt` (the `data/llm_features_*.pt` tensor is gitignored / absent on Streamlit Cloud), so the deployed app rebuilds the genome+LLM user tower from `serving/` alone — `streamlit_app.py` never reads `data/`.

**⚠️ TRAIN/SERVE SKEW — iterate on LLM-front-end prompts against SERVING DATA ONLY (learned the hard way 2026-07-09).** Export also bakes the **facet store** into `serving/feature_store.pt['facets']` — including the `require_keyword_concepts` resolver tables `movieId_to_keyword_concepts` (curated concepts) and `keyword_to_movieIds` (the raw-TMDB index, rung 3 of `resolve_topic_term`). These lag the local `llm_features/cache/facet_store.pt` whenever concepts grow between gated exports. The deployed Ask tab resolves keywords against `serving/` ALONE, so a keyword-only pill (no `liked_items` anchor) whose term isn't a genome tag (e.g. `ancient rome`, `mathematician`) **falls back to popular titles live** if its table isn't baked — even though it looks perfect locally. `tools/llm_frontend_probe.py:Serving` **overlays** the local store by default, which MASKS this; **always pass `Serving(serving_only=True)` when generating or grading Ask-tab pills** (`gen_ask_examples.py` and `ask_live_vs_frozen.py` now do). After ANY change to `KEYWORD_CONCEPTS` / the facet build, **re-export** so `serving/` matches, then regenerate pills. `liked_items` (named-title) anchors always resolve live via the two-tower — only the keyword/facet *resolver tables* are at risk.

**Ask tab (launched 2026-07-09, stable):** 7 roots + 42 leaf pills + 4 hidden backburner = 53 pre-generated boards in `serving/ask_examples.json`, built by `python tools/gen_ask_examples.py` (no API key) from committed frozen extractions `tools/ask_extractions/<id>.json` (spec: `tools/ask_examples_spec.py`). Boards are pinned by those extractions — bare regen only, **never `--live`** (re-extracts everything and destroys curation). Pill↔live honesty was verified pill-by-pill pre-launch; grade drift read-only via `tools/ask_live_vs_frozen.py --k 3`. Tuning rule (prompt-only, never hand-write extraction JSON) + routing learnings: `tools/ask_extractions/README.md`. After a model promotion: re-export, bare-regen boards, re-eyeball. Design/validation record: `docs/llm_frontend/`.

## Running the Code

```bash
python main.py preprocess     # Stage 1: raw CSVs → data/base_*.parquet
python main.py features       # Stage 2: base parquets → data/features_*.parquet
python main.py dataset        # Stage 3: features → data/dataset_mse_rollback_*_v1.pt
python main.py train softmax  # Stage 4: full softmax training (canonical, prod)
python main.py train          # Stage 4: MSE training (legacy, not prod)
python main.py canary [path]  # Canary user recommendations
python main.py probe [path]   # Embedding probes
python main.py eval [path]    # Offline eval: Recall@K, NDCG@K, Hit Rate@K, MRR
python main.py export [path]  # Stage 5: export serving artifacts for Streamlit
python main.py posters        # Fetch TMDB poster URLs → serving/posters.json
```
No path → auto-detects most recently modified `best_*.pth`.

Stages 1–3 are slow and cache to disk; re-run only from the earliest stage you changed (raw data→`preprocess`, features→`features`, split logic→`dataset`, model/hyperparams only→`train`). Softmax and MSE datasets cache separately.

Posters: `TMDB_API_KEY=... python main.py posters` (free key at themoviedb.org/settings/api). Safe to interrupt/resume. The 64 uncovered movies are invalid TMDB IDs in links.csv — won't resolve on retry.

**Serving artifacts (`serving/`, written by `export`):** `model.pth` (state_dict), `movie_embeddings.pt` (precomputed item embeddings), `feature_store.pt` (vocabs, index maps, config), `posters.json` (`{"<movieId>": "<url>"}`).

**Dev/analysis scripts (`tools/`, standalone — not part of the `main.py` CLI):** blog/canary tooling (`batch_canary.py`, `persona_tools.py`, `poster_board.py`, `shoot_boards.py`, `rec_popularity.py`) + diagnostics (`analyze_target_distribution.py`, `similar_movies_diagnostic.py`); run from repo root as `python tools/<script>.py`. Persona inputs in `tools/personas/`; outputs in `tools/results/` (`*.json` committed, `figures/` + `*.md` gitignored).

## Dataset

`data/ml-32m/` must be present (not in git): `ratings.csv`, `movies.csv`, `tags.csv`, `genome-scores.csv`, `genome-tags.csv`. Keep movies with **200+ ratings** (~9,375) and users with 20–500 ratings.

- **User-applied tags** (`tags.csv`): per-movie vector = user-count per tag, normalized to sum 1; keep tags applied 1,000+ times (306 survive).
- **Genome tags** (`genome-scores.csv`): ML-derived relevance 0–1 per (movie, tag), 1,128 tags — denser and more semantic than user tags.

Phase 1 reduced corpus (LLM ablation): `llm_features/filter_corpus.py` keeps **>1,000 ratings → 4,461 movies**; `CORPUS=phase1` namespaces artifacts. See `docs/plans/llm_vs_genome_ablation_plan.md`.

## Model Architecture (canonical, v3 prod)

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
  item_content_tower(genome_scores)     →  item_content_emb (32)
  item_embedding_tower(movie_id)        →  item_emb         (32)  [shared lookup with all 4 user pools]
  year_embedding_tower(release_year)    →  year_emb         (8)
  concat (96) → Linear(256) → ReLU → Linear(128) → L2-normalize → item_emb (128)

Prediction: dot_product(user_emb, item_emb) = cosine similarity (both L2-normalized)
```

Sub-tower linears init Xavier uniform `gain=0.1`; **projection linears re-init at `gain=1.0` after the rest of the model** — otherwise `gain=0.1²` compounds and collapses dot products to zero before training.

### Key implementation notes (gotchas)
- **4-pool user tower:** full/liked/disliked/rating-weighted sum pools all over `item_embedding_lookup` (32-dim ID embedding), each with its own LayerNorm. Liked/disliked pools are pre-computed right-aligned padded tensors in the dataset builder.
- **Shared item embedding:** `item_embedding_lookup` is shared by the item tower and all 4 user pools. Removing `item_embedding_tower` (Linear+ReLU on the lookup) causes severe genre clustering — do not remove.
- **`user_content_tower` ≠ `item_content_tower`:** user side runs a single Linear(1128→32) over the rating-weighted avg of raw genome scores across history (a dense fingerprint), NOT a pool over item-tower outputs. `item_content_tower` is item-side only, compressing per-movie genome 1128→32.
- **Swappable content slot:** `content_feature_source` ∈ `'genome'`|`'llm'`|`None`; prod fuses both (`feature_towers='both'`). `None` omits both towers + buffer (no-content baseline). The genome *product feature* (probes, anchors, Explore-Genome tab) keeps its genome name — only the slot was renamed.
- **Genome sub-tower ≥ 32 dims:** 1128→32; do not shrink below 32 (tested 16, worse).
- **L2 norm both towers:** dot product = cosine; no inference-time popularity correction.

## Training

### V3 Softmax (`train softmax`) — canonical, prod
- Full softmax CE over all ~9,375 items. Adam `lr=0.001`, `weight_decay=0`, `adam_eps=1e-6`. Batch 512, temperature 0.1, 150,000 steps. L2 norm both towers.
- **Menon popularity correction:** logit-adjusted — **add** `alpha*log(count_i)` to logit i before softmax (**not** subtract). Inference uses raw dot products; α is training-only. `alpha`/`temperature` saved as a JSON sidecar per checkpoint.
- Dataset: rollback 7-tuple (below), capped at `MAX_SOFTMAX_EXAMPLES_PER_USER`. Checkpoints `saved_models/best_softmax_<ts>.pth` every 30k steps.

**Alpha is a deployment knob, never a training-search knob.** Train at `alpha=0` while iterating (best offline MRR, cleanest variant comparison; what all ablations use); fine-tune α only on a locked-in model, picking by **canary**. Committed `get_config` default is `alpha=0.0`. Prod's `alpha=0.5` is the fine-tuned value (α=0 drifts popular — War/Fantasy/Heist/Crime collapse to IMDb top-10; α=1.0 over-corrects to obscure).

### MSE rollback (`train rollback`) — legacy, not prod
MSE on de-biased ratings (rating − user mean). SGD `lr=0.005`, momentum 0.9, batch 64, 300k steps. **Adam collapses the user tower — do not use.** Cap 20 examples/user; always `sort_by_ts=True` (shuffle biases toward popular).

## Saving / Loading

Weights-only (~1MB); pre-buffer-fix checkpoints are ~42MB until re-saved (prod is one). `saved_models/` gitignored. `_resolve_config` (`src/evaluate.py`/`src/export.py`) and `src/checkpoint.py:load_checkpoint` read all dims and content/base towers from state_dict weight shapes, so **renaming a `.pth` is safe** and old/new checkpoints coexist. Eval/export load the checkpoint first (fast-fail on invalid), then features.

Legacy "content-era" checkpoints (old 06-07 A/B/D) use generic `item_content_tower` keys disambiguated by a `_config.json` sidecar — keep its stem matching the `.pth`. Low-variance/stripped checkpoints use explicit `item_genome_tag_tower`/`item_llm_feature_tower` keys natively.

## Offline Eval (`main.py eval`)

`src/offline_eval.py` — Recall/NDCG/HitRate/MRR @ K∈{1,5,10,20,50,100,150,200,250}. **Rollback protocol** (default for softmax): per held-out val user, sample up to 20 chronological positions, context=history[0..j-1], target=history[j]. It's the harder protocol — **compare only within the same protocol.** Writes `eval_results/<stem>.txt` (overwrites by stem — a small-user run destroys the canonical artifact). Current prod: MRR 0.1123, tail-≤1k MRR 0.0159, HitRate@10 0.2173.

## Probes (`main.py probe`)
- `probe_genre(genre)` — one-hot → `item_genre_tower` → cosine vs genre embeddings.
- `probe_tag(tags)` — tag vector → `item_tag_tower` → cosine vs tag embeddings.
- `probe_genome_tag(genome_tags)` — avg of top-k representative movie genome embeddings as query (real movies avoid OOD inputs).
- `probe_similar(titles)` — pairwise cosine on `MOVIE_EMBEDDING_COMBINED` (128-dim). Most reliable.

## Canary Users (`main.py canary`)
All get `ts_max_bin` (most recent timestamp). Horror & Sci-Fi are most genre-drift-sensitive (if wrong, model is failing); Comedy & Romance are sanity checks; WW2 stresses drift into non-Western cinema; Crime expects some overlap. Known-acceptable v3 drift: Fantasy → low-quality genre-adjacent (LotR/Dune sit in an ambiguous region); Western occasionally surfaces WW2 (war/western genome overlap); War may have one off-genre pick.

## Dataset Tuples
- **Softmax (7-tuple):** `(X_genre, X_history, X_hist_liked, X_hist_disliked, X_hist_ratings, timestamp, target_movieId)`. Liked/disliked are right-aligned padded int tensors (pad=`len(top_movies)`) from `ctx_rats >0`/`<0`; built in `src/dataset.py:build_v2_softmax_dataset()`.
- **MSE (6-tuple, legacy):** `(X_genre, X_history, X_history_ratings, timestamp, Y, target_movieId)`.
- RAM ~12GB (softmax). Per-item target features (genre/tag/genome/year) are looked up from model buffers at train time, not stored.

**Known inconsistency:** `watch_history` is mapped to embedding indices in `features.py`, but `label_movieIds` stay raw IDs until mapped in `dataset.py`. Both correct, just confusing.

## MPS
`get_device()` (`src/train.py`) = mps > cuda > cpu; train/eval/inference on-device. **Export must `map_location='cpu'`** — MPS-saved checkpoints crash on Linux (Streamlit Cloud has no MPS). ~145 it/s.

## What does NOT work (don't revisit)
- Removing `item_embedding_tower` — severe genre clustering.
- Pooling over the full 128-dim projected item embedding in the user tower — null across two variants. The 32-dim ID lookup is the better pooling signal; the projection is trained for item-side targets, so it's noisier for pooling.
- Adam for MSE — user tower collapse.
- Shuffle in rollback dataset build — biases toward popular.
- Dataset cap > 20/user — null MRR, +4GB RAM.
- Soft negatives (removed in v3) — do not bring back.

## LLM-vs-Genome Ablation
Narrative `docs/llm_vs_genome_ablation/llm_vs_genome_ablation.md`; full record + checkpoint→model map `docs/plans/llm_vs_genome_ablation_plan.md`. Question: does an item content vector help, genome vs LLM-extracted, on genome's 132 axes? Verdict (low-variance, seeded, α=0): content beats the CF floor in the stripped `BASE_TOWERS=idonly` setting (C′<A′<B′, LLM +3.0%); in the rich base the lift vanishes (content is redundant with curated genre/tags). LLM ≥ genome on every tier of both corpora — LLM features are less redundant with cheap metadata, i.e. more genuinely additive. Model D (genome+LLM) shows no clear additive benefit; prod is a separate α=0.5 portfolio deployment of the D architecture, not metrics-driven.

---

## Working Style

**Git:** never commit and push in one command — commit, then ask before pushing. For changes that need retraining to validate (hyperparams, optimizer, loss, dataset logic, architecture): write code, smoke-test, then **stop** — do not commit, and do not update results tables in CLAUDE.md/plans, until the user runs `train` → `canary` → `eval` and confirms the numbers. Verifying metrics is the user's job; you only verify imports compile / shapes match / smoke passes.

**Never launch a full `python main.py train` yourself** (background process or Workflow): Claude-spawned background jobs run ~10× slower and degrade (macOS deprioritizes detached processes — low QoS / App Nap, not thermal). Write + smoke-test the training code, then hand the exact `CONTENT_SOURCE=… CORPUS=… python main.py train …` command to the user. Short MPS jobs (`eval`, `canary`, `probe`) are fine in the background.

**Do not touch `streamlit_app.py` or `src/export.py` until a model change is verified good.** Order: train → canary → eval → if better, then export/streamlit.

**Code style:** match the house style (long docstring headers, NamedTuple buffer bundles, comment banners on training-loop fns, named slice offsets over magic numbers, vertically-aligned parquet column comments). Make surgical changes — every changed line traces to the request; mention adjacent issues, don't fix them. Surface interpretation tradeoffs in a sentence and proceed with a stated assumption rather than asking. Track 3+ step work with TaskCreate, not inline prose plans.
