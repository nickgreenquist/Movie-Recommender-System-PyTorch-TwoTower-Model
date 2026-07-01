# LLM Conversational Front-End — project docs

Home for **one feature**: a natural-language "Ask AI" layer over the trained two-tower recommender. A
fast hosted LLM parses a free-form request into the structured input the model expects; the two-tower
model does the actual retrieval; the user never sees raw LLM output. (Industry name: the **"front
bookend"** of the Bookend-LLM pattern — see `research/bookend_llm_framework.md`.)

Everything for this project lives under `docs/llm_frontend/` so it's no longer scattered across
`docs/plans/` (beside unrelated ablation plans) and `docs/research/`.

## Document map

| Path | What it is | Status |
|---|---|---|
| **`llm_frontend_plan.md`** | The **umbrella plan** — core principle, v1→v1.5→v2 rollout, build handoff | root spec |
| **`facet_store_plan.md`** | The **v1.5 sub-plan** — scraped-facet store + the "three engines" (membership facets / vibe-affect path / demoted plot backfill). **Active work.** | active |
| `research/hybrid_llm_recommender_research.md` | Prior-art survey (Gemini Deep Research) the plan is grounded on | reference |
| `research/bookend_llm_framework.md` | The Bookend-LLM production-pattern writeup a colleague sanity-checked us against | reference |
| `validation/llm_frontend_haiku_validation.md` | The **v1→v5 mass-Haiku validation writeup** (results) | done |
| `validation/haiku_validation_handoff.md` | Archived resume prompt for that validation | ✅ done/archived |
| `validation/v4_resume/`, `validation/v5/` | Validation run artifacts (extraction prompts, judge scripts, cases, summaries) | archive |
| `validation/ask_ai_holes/` | **Paused** run: 500 real Ask-AI prompts + oracle-rec/coverage harness to find plan holes | ⏸ paused |

## Current status (2026-06-30)
- **v1 front-end:** built + verified in-repo (uncommitted). Shared core `src/llm_frontend.py`; prompt/schema
  `src/llm_frontend_prompt.py`; hosted call `src/llm_frontend_extraction.py`; Streamlit "Ask" tab.
- **v1→v5 validation:** DONE (see the writeup).
- **Facet store (v1.5):** Phase 0 committed; Phase 1 (people hard-filter) built, uncommitted; "Expansion II"
  rewritten into the three-engine design after a measurement pass.
- **⏸ Immediate next step:** the **Ask-AI holes** run — see the "▶ IMMEDIATE NEXT STEP" callout in
  `facet_store_plan.md` (Expansion II). 500 prompts generated + saved; oracle-rec + synthesis remain.
- **Back bookend (LLM explanations):** deferred by company decision — build nothing; just preserve per-rec
  provenance (noted in `facet_store_plan.md` → "Back-bookend readiness").

## Related (outside this dir)
- Code: `src/llm_frontend*.py`, `tools/llm_frontend_probe.py`, `llm_features/build_facet_store.py`.
- Memory: `project_llm_frontend_v1`, `project_llm_frontend_validation`, `project_facet_store_plan`,
  `project_facet_expansion_measurement`, `reference_bookend_llm_framework`.
