# LLM Conversational Front-End — project docs

Home for **one feature**: the natural-language **Ask** tab over the trained two-tower recommender. A
fast hosted LLM parses a free-form request into the structured input the model expects; the two-tower
model does the actual retrieval; the user never sees raw LLM output. (Industry name: the **"front
bookend"** of the Bookend-LLM pattern — see `research/bookend_llm_framework.md`.)

**STATUS: LAUNCHED (2026-07-09).** v1 + v1.5 are live in the deployed Streamlit app — the tuned
extraction prompt, facet hard-filters (people / keyword topics / rating floors & ceilings / runtime /
year fences), the keyword-concept resolver, the step-4 transparency layer, and the pre-generated
example-pill tour. This directory is now the design/validation **record**; the operational docs live
next to the code (see "Where things landed" below).

## Document map

| Path | What it is | Status |
|---|---|---|
| **`llm_frontend_plan.md`** | The **umbrella plan** — core principle, v1→v1.5→v2 rollout | shipped — kept as design record |
| `research/hybrid_llm_recommender_research.md` | Prior-art survey (Gemini Deep Research) the plan is grounded on | reference |
| `research/bookend_llm_framework.md` | The Bookend-LLM production-pattern writeup a colleague sanity-checked us against | reference |
| `validation/llm_frontend_haiku_validation.md` | The **v1→v5 mass-Haiku validation writeup** (results) | done |
| `validation/retrieval_eval/` | Facet/retrieval regression cases + reports — **live input** to `tools/llm_frontend_eval.py` (169 cases, 80 promoted from the 500-query run) | live eval data |
| `validation/v4_resume/` *(local-only, gitignored)*, `validation/v5/` | Validation-run artifacts (cases, judge scripts, summaries) | archive |
| **`validation/test_prompts_500.md`** | **The 500 test prompts** — realistic Ask queries + per-prompt intent, the reusable prompt-iteration test set | live test asset |

Working docs deleted after ship (`facet_store_plan.md`, `hybrid_retrieval_ask_tab_plan.md`, the
`ask_tab_*_handoff` series, validation resume notes) live in git history (removed in `425c633`).
Code comments still cite `facet_store_plan` section names as design rationale — resolve those
against the historical doc via `git show 425c633^:docs/llm_frontend/facet_store_plan.md`.

## Where things landed (post-launch)

- **Code:** `src/llm_frontend.py` (retrieval pipeline, facet post-filters, `KEYWORD_CONCEPTS`
  resolver), `src/llm_frontend_prompt.py` (prompt + schema — the tuned asset),
  `src/llm_frontend_extraction.py` (hosted forced-tool Haiku call), `llm_features/build_facet_store.py`
  (facet-store build; baked into `serving/feature_store.pt` by export).
- **Example pills:** spec `tools/ask_examples_spec.py` → frozen extractions `tools/ask_extractions/`
  (apply loop + routing learnings in its README) → `python tools/gen_ask_examples.py` →
  `serving/ask_examples.json` (7 roots + 42 shown leaves + 4 backburner = 53 boards).
- **Eval / regression:** `tools/llm_frontend_eval.py` over `validation/retrieval_eval/eval_cases.json`;
  read-only pill-vs-live grader `tools/ask_live_vs_frozen.py --k 3`; single-query live trace
  `tools/llm_frontend_trace.py` (the `/trace` skill).
- **The 500-query quality program:** prompt set `validation/test_prompts_500.md` (committed — the
  reusable test set; the original `ask_ai_holes/` harvest harness + run results were pruned to git
  history post-launch); run artifacts local-only in `tools/results/traces/run500/` (gitignored).
  Headline: good-board rate 32.0% → 50.8% across the tuning waves.
- **Back bookend (LLM explanations):** deferred by decision — build nothing; per-rec provenance is
  preserved for it.

## Related (outside this dir)

- CLAUDE.md: the **TRAIN/SERVE SKEW** warning (iterate on pills against `serving/` only) and the
  Ask-tab pipeline summary.
- Memory: `project_ask_serving_skew`, `project_ask_tab_launched`, `project_run500_quality_grade`,
  `project_query_router_rejected`, `reference_bookend_llm_framework`, `reference_trace_command`.
