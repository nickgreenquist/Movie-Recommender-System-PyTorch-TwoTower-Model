# CLAUDE.md

Guidance for Claude Code in this repo. Kept deliberately short — read the code/docs it points at for detail.

## Project

PyTorch Two-Tower recommender on MovieLens 32M. CLI `main.py`; canonical code in `src/`; Streamlit demo `streamlit_app.py` (deploys from `serving/` artifacts alone — never reads `data/`); LLM feature pipeline `llm_features/` (see its README); dev scripts `tools/` (run as `python tools/<script>.py`). `archive/` notebooks are reference-only.

**Purpose: portfolio piece** showcasing ML/recsys mastery to an ML engineer or CTO (may be demoed live in interviews) — weigh every feature by whether it sharpens that signal.

**Core design choice: no user-ID embedding.** Users are represented by taste signals only — 4 history pools over the shared 32-dim item-ID lookup (full / liked / disliked / rating-weighted, each LayerNormed), a genome-context fingerprint (Linear over the rating-weighted RAW genome average — not a pool over item outputs), genre affinity, timestamp — so any user is embeddable at inference from a few liked movies, no retraining. Both towers L2-normalize to 128-dim; dot product = cosine, no inference-time popularity correction. Full architecture: `src/model.py` (the Streamlit About tab has an accurate prose walkthrough).

## Prod

**Checkpoint:** `saved_models/PROD_best_softmax_genome_tags_llm_features_popularity_alpha_05_20260612_080719.pth` — Model D: full softmax, Menon α=0.5, `feature_towers='both'` (genome + 132-dim LLM content towers fused), trained 250k steps. Whole-corpus rollback **MRR 0.1132**, tail-≤1k MRR 0.0157, HitRate@10 0.2174 (`eval_results/PROD_…txt`). Promoted 2026-06-12; prior prod = `OLD_PROD_…20260607_195227.pth` (MRR 0.1123). **Do not replace without a clearly better eval.** Re-export serving artifacts: `python main.py export saved_models/PROD_…pth` (`--variant no_alpha` writes the α=0 twin behind the app's popularity toggle; export bakes the LLM feature buffer + facet store into `serving/feature_store.pt`).

**⚠ TRAIN/SERVE SKEW (Ask tab, learned the hard way 2026-07-09):** the deployed app resolves keyword/facet terms against the tables baked into `serving/feature_store.pt['facets']` ALONE — they lag the local `llm_features/cache/facet_store.pt` between exports, and `tools/llm_frontend_probe.py:Serving` overlays the local store by default, which MASKS the lag (a keyword-only pill can look perfect locally and fall back to popular titles live). **Iterate/grade Ask pills only with `Serving(serving_only=True)`; after any `KEYWORD_CONCEPTS`/facet change: re-export, then regenerate pills.** Named-title (`liked_items`) anchors always resolve live — only the resolver tables are at risk.

**Ask tab (launched 2026-07-09):** 67 pinned boards (9 roots + 53 leaves + 5 backburner) in `serving/ask_examples.json` = `python tools/gen_ask_examples.py` (no API key) over frozen extractions `tools/ask_extractions/*.json` (spec: `tools/ask_examples_spec.py`). **Bare regen only — NEVER `--live`** (re-extracts everything, destroys curation). Tune pills prompt-only, never hand-edit extraction JSON; grade drift with `tools/ask_live_vs_frozen.py --k 3`. Rules + routing learnings: `tools/ask_extractions/README.md`; design/validation record: `docs/llm_frontend/`. After a model promotion: re-export → bare regen → re-eyeball.

## Commands

```bash
python main.py preprocess   # raw CSVs → data/base_*.parquet        (slow, cached)
python main.py features     # → data/features_*.parquet             (slow, cached)
python main.py dataset      # → data/dataset_softmax_*_v2.pt        (slow, cached; ~12GB RAM)
python main.py train        # full-softmax training — the ONLY trainer (extra args silently ignored)
python main.py canary|probe|eval|export [ckpt]   # no path → most recently modified best_*.pth
python main.py posters      # TMDB poster URLs → serving/posters.json (needs TMDB_API_KEY)
```

Re-run only from the earliest stage you changed. `data/ml-32m/` (not in git): keep movies with 200+ ratings (~9,375), users 20–500. Ablation env knobs (`FEATURE_TOWERS`, `BASE_TOWERS`, `USER_POOLS`, `CORPUS=phase1`, `SEED`) reproduce the committed experiments; all unset = prod config, byte-identical. Tests: `pytest tests/` or `python -m tests.test_model_shapes` (CPU-only, no data/checkpoint needed).

## Training (v3 full softmax — the only path; the old MSE trainer is deleted, git history only)

CE over all ~9,375 items; Adam lr=0.001, batch 512, temperature 0.1, 200k steps default. **Menon correction: ADD `alpha*log(count_i)` to logit i during training only** (inference stays raw dot products); α/temperature saved in a JSON sidecar per checkpoint. **α is a deployment knob, never a training-search knob:** iterate and ablate at α=0 (the committed default), fine-tune α only on a locked-in model, pick by **canary** (prod α=0.5; α=0 drifts popular, α=1.0 over-corrects to obscure).

Gotchas that will bite:
- Projection linears are re-inited at gain=1.0 AFTER the sub-towers' Xavier gain=0.1 init — otherwise the gains compound and dot products collapse to zero.
- `item_embedding_lookup` is shared by the item tower and all 4 user pools — removing `item_embedding_tower` causes severe genre clustering.
- `watch_history` is already mapped to embedding indices in `features.py` while `label_movieIds` stay raw IDs until `dataset.py` — both correct, just confusing.
- Export must load `map_location='cpu'` (MPS-saved checkpoints crash Linux/Streamlit Cloud). `get_device()`: mps > cuda > cpu (~145 it/s).
- Checkpoint dims/towers resolve from state_dict weight shapes (`src/checkpoint.py`, `_resolve_config`) — renaming a `.pth` is safe. Legacy "content-era" checkpoints need their `_config.json` sidecar stem-matched.

## Eval / canary

`main.py eval` = rollback protocol (per held-out val user, ≤20 chronological positions; context = history so far, target = next watch), K∈{1..250} — **compare only within the same protocol**. **It overwrites `eval_results/<stem>.txt` by stem — a small-`EVAL_N_USERS` run destroys the canonical artifact; smoke-test some other way.** Canary personas live in `src/evaluate.py`; Horror & Sci-Fi are the sensitive drift canaries. Known-acceptable drift: Fantasy → genre-adjacent, Western ↔ WW2 genome overlap.

## Do NOT revisit (all tested, all failed)

Removing `item_embedding_tower` (genre clustering) · pooling the 128-dim projected embedding instead of the 32-dim ID lookup (null across two variants) · genome sub-tower below 32 dims (16 tested, worse) · Adam for the old MSE path (user-tower collapse) · shuffle in the rollback dataset build (popularity bias) · dataset cap >20/user (null MRR, +4GB) · soft negatives (removed in v3) · an explicit LLM query-router for the Ask tab (sim-rejected: 64% hybrids, silent misroutes) · HNSW at 9k items (full scoring is a single matmul).

## Ablation records (complete — cite, don't rerun)

- **LLM-vs-genome content features:** `docs/plans/llm_vs_genome_ablation_plan.md` (+ narrative in `docs/llm_vs_genome_ablation/`). Verdict: content beats the CF floor only in the stripped `idonly` base; in the rich base the lift vanishes; LLM ≥ genome on every tier (less redundant with curated metadata). Prod's genome+LLM fusion is a portfolio deployment choice, not metrics-driven.
- **Multipool user tower (12 arms):** `docs/plans/multipool_user_tower_ablation_plan.md` (+ narrative in `docs/multipool_user_tower_ablation/`). Verdict: recency/last-item dominates, rating-valence channels null, order-2 = noise.

## Working style

- **Git:** commit only when asked; never commit+push in one command — commit, then ask before pushing.
- **Blog posts ship with a LinkedIn counterpart:** most `docs/<topic>/` narratives carry a `<topic>_linkedin_post.txt` beside them — the copy Nick actually pastes into LinkedIn. Hand-written, `.txt` so it copies clean. **Never prune these** (a "superseded files" sweep deleted all three in 889379f; recovered 2026-07-12).
- **Changes that need retraining to validate** (hyperparams, loss, dataset logic, architecture): write code, smoke-test, then **stop** — no commits, no results-table updates until the user runs train → canary → eval and confirms. You verify imports/shapes/smoke only.
- **Never launch a full `python main.py train` yourself** (background or Workflow) — Claude-spawned jobs run ~10× slower (macOS deprioritizes detached processes). Hand the exact env+command to the user. Short MPS jobs (eval, canary, probe) in the background are fine.
- **Don't touch `streamlit_app.py` or `src/export.py` until a model change is verified good** (train → canary → eval → only then export/app).
- **Style:** match the house style (long docstring headers, NamedTuple buffer bundles, named slice offsets over magic numbers, aligned column comments). Surgical diffs — every changed line traces to the request; mention adjacent issues, don't fix them. Surface interpretation tradeoffs in a sentence and proceed with a stated assumption. Track 3+-step work with TaskCreate.
- **`tools/` inventory:** Ask tab (`gen_ask_examples`, `ask_examples_spec`, `ask_live_vs_frozen`, `llm_frontend_probe`, `llm_frontend_trace`, `llm_frontend_eval`) · canary/figures (`batch_canary`, `persona_tools`, `poster_board`, `shoot_boards`, `rec_popularity`, `make_multipool_figure`) · diagnostics (`analyze_target_distribution`, `similar_movies_diagnostic`). Persona inputs in `tools/personas/`; committed outputs in `tools/results/*.json`.
