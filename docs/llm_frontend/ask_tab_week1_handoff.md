# Ask-Tab Week-1 Execution Handoff (staff-review follow-up, 2026-07-03)

> **Who you are.** You are the senior staff engineer who audited the Ask-tab / LLM front-end
> direction on 2026-07-03 (fresh context now — this file is your memory). Nick reviewed your
> assessment and green-lit the week-1 items. Your job is to EXECUTE tasks 1–3 below, in order.
> Do not re-run the assessment; the evidence is banked in §Evidence.
>
> **Read first (in this order):** this file → `docs/plans/hybrid_retrieval_ask_tab_plan.md`
> (THE spec — its step 2/3 are your tasks 2/3; its gotchas, guardrails, and don't-relitigate
> list are binding) → skim `src/llm_frontend_prompt.py` and the constants/table block of
> `src/llm_frontend.py` (lines ~50–800).

## State of the tree (do not fight it)

- UNCOMMITTED step-1 + fix-wave work is already in the working tree: `src/llm_frontend.py`,
  `src/llm_frontend_prompt.py`, `docs/llm_frontend/validation/retrieval_eval/eval_cases.json`,
  `tools/llm_frontend_eval.py`, plus the untracked plan doc. It is GOOD (validated 165/165).
  Build on top of it. Never commit — and never commit+push — without Nick's explicit go.
- Regression ruler currently passes **165/165**: `python tools/llm_frontend_eval.py`. Any drop
  is a regression you introduced.
- `serving/` and `streamlit_app.py` are GATED — no export, no edits there. All iteration goes
  through the mock loop (`tools/results/traces/run500/mock_serving.py:get_mock_serving()`).

## Evidence bank (from the 2026-07-03 audit — do not re-derive)

**"movies with dogs", four ways** (live Haiku extraction reproduced + `recommend()` runs on the
fresh local facet store; genome tags `dog`=col 315 / `dogs`=col 316; 87 films ≥0.7):

| Path | Result | Top of list |
|---|---|---|
| `/trace` today (stale baked store) | popularity fallback after empty-pool relax | Shawshank, Pulp Fiction — garbage |
| ctrl-F (genome ≥0.7, relevance sort) | all dog films, junk-first ordering | See Spot Run (pop#8167) #1 |
| live extraction on FRESH store: `require_keyword_concepts:["dog"]` | fallback=True → popularity inside boolean pool | The Mask (dog-genome 0.33) #1; 6/15 incidental |
| `require_genome_tags:["dogs"]` (eval-case path) | floor 0.35 + λ=2.0 rerank + anchors | Lassie, Balto, Beethoven, Hachiko — all 15 ≥0.93 |

Row 4 is the target behavior and already works — but live extraction can't reach it: the prompt
routes dual-membership terms (dog is a genome tag AND a keyword concept) concept-list-first
(deliberate step-1 interim; the genome-first tie-break was deferred to the step-2 resolver).

**The split-brain (root cause of Nick's bad trace):** `tools/llm_frontend_probe.py:107` —
`Serving()` prefers the facet table baked into `serving/feature_store.pt` (Jul 2, **57
concepts, no `dog`**) and only falls back to the rebuilt local store
(`llm_features/cache/facet_store.pt`, Jul 3, **95 concepts**) when no baked table exists. The
eval ruler (`tools/llm_frontend_eval.py:350`) and `mock_serving.py` patch the local
`movieId_to_keyword_concepts` table in; `/trace` (via `tools/llm_frontend_trace.py:259 → Serving()`)
does NOT. So the team's rulers said "fixed" while Nick's smoke tool showed popularity garbage.

**Ruler-vs-live gap (keep in mind whenever you read "165/165 green"):** eval cases are
hand-authored extractions; the "dogs" case tests the genome path that live extraction currently
never emits for that query. Only the mock loop (real Haiku extraction end-to-end) measures live
behavior. Validate every task against BOTH.

**Prompt-size baseline** (for the task-2 deletion target): full system prompt 34,244 chars
(~8.6k tok) = genome vocab ~4.0k tok + **English policy ~4.6k tok**. The English half is where
the fragile hardcoding lives (routing order, concrete-topic guard, concept-list injection).

## Task 1 — kill the split-brain: `/trace` must see the fresh facet store (do this first)

Recommended: fix it in `Serving.__init__` (`tools/llm_frontend_probe.py`) — after selecting the
baked `fs['facets']`, OVERLAY the local `llm_features/cache/facet_store.pt`
`movieId_to_keyword_concepts` table when that file exists (same single-table patch as
`mock_serving.py` / the eval fix; the file already has a harness-only disk-fallback precedent at
line ~107). That makes `/trace`, the probe CLI, and the rulers agree by construction. Alternative
(if you find a reason Serving must stay pristine): patch inside `llm_frontend_trace.py` instead.
Check whether `mock_serving.py` is git-tracked before importing it from a committed tool
(`git ls-files tools/results/traces/run500/mock_serving.py`) — if untracked, replicate the
3-line patch rather than importing.

Acceptance: (a) `python tools/llm_frontend_eval.py` → 165/165; (b) re-run the dogs trace
(`/trace` skill flow or `tools/llm_frontend_trace.py` with extraction
`{"hard_constraints":{"require_keyword_concepts":["dog"]}}`) → §5 must show dog-keyword films
(Up / Best in Show / John Wick tier), NOT Shawshank; §4 must show no relaxation.

## Task 2 — plan step 2: genome-first dynamic resolver, shipped as a REPLACEMENT

Spec = plan §"Target mechanism" + step-2 gotchas (exact-token never substring; homonym
denylist; confidence gate before HARD; genome-over-keyword tie-break; hard when defining topic,
soft when modifier; reuse the empty-pool relaxation). Implementation notes from the audit:

- **Keyword vocab source:** the 17,820-keyword index is regenerable from
  `llm_features/cache/scraped/*.json`; bake an inverted `keyword → movieIds` table via
  `python llm_features/build_facet_store.py` (extend the builder). Consider a min-coverage
  floor (e.g. ≥3 films) and the homonym denylist at build time. Local store only — no export.
- **Schema shape:** prefer evolving `require_keyword_concepts` into free concept terms (or add
  a parallel free-terms field) resolved server-side. BACK-COMPAT IS REQUIRED: the 165 ruler
  cases hand-author `require_keyword_concepts` / `require_genome_tags` extractions — old slots
  must keep working in `recommend()`.
- **Genome-first for dual terms:** a term that exact-token-matches the genome vocab (try
  verbatim, then naive singular/plural variant — still exact-token against vocab entries) routes
  to the genome machinery (anchors + 0.35 floor + REQUIRE_GT_RERANK_LAMBDA), NOT the boolean
  keyword pool. Keyword membership is the fallback for the genome's long-tail misses.
  ("movies with dogs" must land on the Lassie/Balto row; idx 243 "trapped underwater" should
  land on submarine-genome ranking, not Austin-Powers-by-popularity.)
- **The deletion diff (non-negotiable — this is what makes it a simplification, not a 4th
  channel):** with the resolver live, strip from `_SYSTEM_TEMPLATE` + schema: the (1)→(4)
  routing-order prose, the CONCRETE-TOPIC GUARD enumeration, and the `_CONCEPT_KEYS_STR`
  injection (keep `KEYWORD_CONCEPTS` in code as the frozen high-precision homonym-safe core per
  the plan guardrail — but the LLM no longer needs the list; it emits free terms). Target: the
  ~4.6k-token English half shrinks by O(1–2k tokens). If the prompt doesn't shrink, you added a
  channel instead of replacing one — stop and reconsider.

Acceptance: 165/165 green; mock loop on the 54-bad subset + a live "movies with dogs"
re-extraction reaches the genome path (top-15 all high dogs-relevance); no real regressions on
the good spot-checks (separate judge noise ±5 from actual rec changes per the plan's protocol);
prompt token count measurably down.

## Task 3 — plan step 3: demote popularity to true last resort

In `recommend()` (`src/llm_frontend.py` — `fallback` computed ~line 1553): popularity ordering
may fire only when NO genome/keyword/anchor signal resolved. After task 2, most topic queries
carry genome anchors, so the remaining exposure is pure boolean-membership pools (keyword hit,
no genome coverage). Guardrail: the genre-centroid "seed of last resort" for BARE-CATEGORY
(genre-only) queries was REJECTED by data — popularity IS the correct floor there
(memory `project_year_fence_vibe_collapse`); step 3 narrows popularity's reach, it does not
abolish it. Scope the change so bare-genre behavior is unchanged.

Acceptance: 165/165; idx-243-class mock-loop queries improve; bare-category queries unchanged.

## Validation protocol (after EVERY task, no exceptions)

1. `python tools/llm_frontend_eval.py` → must stay 165/165.
2. Mock loop for live behavior: `gen_workflow.py ids <ext_dir> <name> <idxs>` (Haiku subagent
   extractions — keep fan-outs to tens, not hundreds) → `phase2_fix1.py` (recommend() on
   `get_mock_serving()`) → `gen_grade_workflow.py` (Sonnet judge) → `compare_progress.py`.
   Tooling paths + the 500 queries: plan §"Tooling that already exists".

## Stop conditions / working style

- Write code, validate on the rulers, then STOP and hand Nick the numbers + diff summary. He
  decides commits. No export, no `streamlit_app.py`, no serving/ writes, no pushes.
- Surgical diffs in house style (long docstring headers, banner comments, named constants).
  Track the 3 tasks with TaskCreate. Don't relitigate the plan's settled list (no ANN, no LLM
  router, no 17k-keyword prompt inlining — token math: ~50k tok/call vs 8.6k today).
- If a task forces a scope call the plan doesn't answer, state the assumption in one sentence
  and proceed; batch open questions for Nick at the end.
