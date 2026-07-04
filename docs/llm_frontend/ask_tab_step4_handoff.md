# Ask-Tab Step-4 Execution Handoff (post week-1, 2026-07-04)

> **Who you are.** You executed the week-1 Ask-tab work on 2026-07-03 (fresh context now — this
> file is your memory). All of it is COMMITTED & pushed (`af8c5ee`): split-brain fix, genome-first
> dynamic resolver (`resolve_topic_term`), prompt deletion diff, popularity demotion. Ruler is
> **165/165**; mock loop improved (fix2 good2/partial6/bad23 → good5/partial5/bad21, fallbacks
> 31→1). Your job now: EXECUTE step 4 below, then bring Nick the two decisions.
>
> **Read first:** this file → `docs/plans/hybrid_retrieval_ask_tab_plan.md` §step 4 + guardrails +
> don't-relitigate (binding) → `docs/llm_frontend/ask_tab_week1_handoff.md` §validation protocol.

## Task — plan step 4: transparency / polish layer (portfolio-grade degradation)

Build at the HARNESS level first — extend the `recommend()` report + the trace/probe renderers;
no `serving/` or `streamlit_app.py` edits (gated until decision 1). The raw material already
exists — this is mostly routing it to the surface:

a. **Intent echo** — "Understood: 🐕 dog · cozy vibe · ranked by taste." Source: extraction +
   `report['topic_resolution']` notes + `mood_tags` / `anchors` / people & facet logs.
b. **Per-rec provenance chip** — `matched: dog @0.98, heartwarming`. Source: resolver route notes
   + per-film genome relevance (`fs['movieId_to_genome_tag_context']` via `ctx.genome_name_to_idx`).
c. **Relaxation notice** — honest wording over the existing `relaxed_constraints` (deb1040).
d. **Capability boundary** — ceiling asks (plot-structure "like X where…", autumn-vibes-class
   metadata gaps — see the still-bad list in the week-1 mock loop) get an honest "I match
   themes/topics/people/era, not specific plot mechanics — closest read:" line, not silence.
e. **Out-of-domain catch + seeded example chips** — "a game with dogs" → "I only know movies 🎬";
   pick 4–6 showcase queries.

Acceptance: 165/165 stays green; ~20-casual-query smoke (plan done-criteria): every query
succeeds or degrades TRANSPARENTLY — never a silent generic dump; trace/probe render the new
fields sensibly. Bar = "never looks broken, always looks like it understood."

## Then STOP and bring Nick decision 1 — Streamlit wiring + gated export (plan step 6)

Scope the wiring, then ask for the explicit go before touching `streamlit_app.py`/`serving/`.
The one real design call to present: `keyword_to_movieIds` (resolver rung 3) is LOCAL-only —
deployed Cloud app would silently lose rung 3 (rungs 1–2 still work). Baking it into
`serving/feature_store.pt` is ~1–2 MB but grows the git-tracked ~84 MB artifact. Recommendation
from last session: bake it. Export must `map_location='cpu'`; export bakes local facet tables.

## And decision 2 — full-500 re-measure (optional, high-value evidence)

Pre-resolver baseline: good_pct 32% (memory `project_run500_quality_grade`). A fresh full-500
mock-loop run + Sonnet grade gives the before/after headline for the blog/interviews. Cost: 500
Haiku extractions + ~50 judge agents — ASK Nick before burning that budget; run in batches
(tens per Workflow, per `feedback_agent_fleet_budget`). Tooling: `run500/gen_workflow.py` →
`phase2_fix1.py` → `gen_grade_workflow.py` → `aggregate_grades.py`; compare vs `grades.json`.

## Footnotes (only if Nick asks)

Batched small items: `buddy cop` concept also matches `buddy comedy` keywords (White Chicks
leaks; one-entry split of the frozen list = Nick's call); the 80-case `eval_cases_run500.json`
harvest is still unpromoted.

## Working style (unchanged)

165/165 after every change; validate live behavior via the mock loop, not export. Surgical
diffs, house style, TaskCreate for the tasks. Commit only with Nick's explicit go — never
commit+push in one command. If a scope call comes up, state the assumption in one sentence and
proceed; batch open questions for Nick at the end.
