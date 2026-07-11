# llm_features/ — LLM content-feature pipeline

Extracts **132 content dimensions** (0–1 scores on genome-derived axes) per movie from
scraped TMDB + Wikipedia text — the tensor fused into the prod item tower. Full design +
results record: `docs/plans/llm_vs_genome_ablation_plan.md`.

**Pipeline order for a new corpus** (everything caches under `llm_features/cache/` and resumes):

| Step | Script | What it does |
|---|---|---|
| 0 | `filter_corpus.py` | pick a reduced corpus (phase1 = >1k ratings); full corpus skips this |
| 1 | `scrape.py` | raw TMDB + Wikipedia → `cache/scraped/` (store raw, truncate at feed time) |
| 2 | `derive_schema.py` | genome discriminability → `data/llm_schema_dimensions.json` (imported by `schemas.py`/`prompts.py`) |
| 3 | `cc_extract.py` ★ | **the canonical extractor** (Claude-Code Sonnet, no API key; produced the ablation + prod tensor). Fan out subagents/terminals over `list_remaining()`; genre-guarded `ingest()` |
| 4 | `merge_extractions.py` | six group JSONs → one 132-dim dict per movie + consistency scan |
| 5 | `build_features.py` | merged dicts → `(n_corpus+1, 132)` float32 tensor for train/export |

Alternate extractors (both wrap the same core artifacts as `cc_extract.py`):
`llm_extract.py` — metered-API path, retained for reproduction only (the planned
Sonnet-vs-Haiku bake-off was dropped); `batch_extract.py` — 25-movie calibration harness,
not a full-run driver.

Separate from the pipeline:
- `build_facet_store.py` — **live product machinery**: builds the people/keyword/attribute
  resolver tables for the Ask tab. Baked into `serving/feature_store.pt['facets']` at export —
  **re-export after changing it** (see the train/serve-skew warning in CLAUDE.md).
- `make_figures.py`, `feature_level_analysis.py` — reproduce the ablation write-up figures.
