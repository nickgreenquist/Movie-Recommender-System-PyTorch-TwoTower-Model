# Ask-Tab Bad-136 Triage Handoff (post step-4 + full-500 re-measure, 2026-07-04)

> **Who you are.** You executed plan step 4 (transparency/polish layer) AND the full-500
> re-measure on 2026-07-04 (fresh context now — this file is your memory). All of it is
> COMMITTED & pushed. Headline: **good_pct 32.0% → 50.8%** on the same 500 queries / same strict
> Sonnet judge (good 156→252, partial 213→108, bad 119→136, unservable 12→4; fallbacks 76→57).
> Ruler is **165/165**. Your job now: EXECUTE Nick's triage command below, report, and STOP.
>
> **Read first:** this file → memory `project_run500_quality_grade` (baseline + re-measure blocks)
> → `tools/results/traces/run500/step4_remeasure_summary.md` (confusion matrix + good→bad triage)
> → `docs/plans/hybrid_retrieval_ask_tab_plan.md` §guardrails + §don't-relitigate (binding).

## Task — Nick's command (2026-07-04, verbatim intent)

> "Go over the 136 'bad' prompt/results, and tell me if you even think those are servable. I'm
> betting a lot of them are just not worth chasing. So we could have: bad / unservable / not worth
> optimizing for / etc."

Re-triage ALL 136 step-4 bads (`grades_step4.json`, verdict == "bad") into servability tiers.
Suggested tiers (refine if the data argues for different cuts, but keep them decision-oriented):

- **WORTH-FIXING** — a real loss with a known cheap lever (e.g. the two parked candidates below,
  an alias/fewshot gap, a resolver miss on an in-data term). Name the lever per item.
- **CEILING-DECLINE** — representational limit (plot mechanics / craft technique / aesthetic with
  no tag; the like-X-nuance class). The RIGHT answer is the step-4 capability notice firing, not
  new retrieval machinery — verify whether it DID fire (read the record/extraction) and flag the
  silent ones.
- **NOT-WORTH-OPTIMIZING** — ultra-long-tail / bizarre / low-portfolio-value asks a demo will
  never field; chasing them is negative ROI. (Nick's bet: many land here.)
- **MISCALLED** — judge should have said unservable (rubric's own class: meme/soundtrack/social-
  framing) or the recs are actually defensible (judge strictness / noise — check whether recs
  changed vs baseline `records/rec_<idx>.json` before crying regression).
- **HONEST-THIN** — correct behavior judged as under-delivery (e.g. idx 456 "last 5 years":
  catalog is sparse post-2019 → 4 films). Arguably a presentation ask, not retrieval.

Method: for each of the 136, read `grades_step4.json` (reason + offenders) + `records_step4/
rec_<idx>.json` (query, recs, fallback/relaxed flags) + `ext_step4/ext_<idx>.json` (what the
extractor emitted). Classify, cluster, count. Deliverable = a report for Nick: tier counts, the
per-tier lists with one-line justifications (a summary MD in the run500 dir is the right shape),
and a recommendation — which tier(s), if any, justify a fix wave, and what the good_pct ceiling
looks like if CEILING + NOT-WORTH + MISCALLED are excluded from the denominator (the honest
"what's actually left to win" number). This is ANALYSIS ONLY — no code/prompt changes until Nick
picks; batch open questions at the end. Subagents (Haiku/Sonnet readers over batches of records)
are fine if you want them — size fleets to tens (memory `feedback_agent_fleet_budget`; two 5h-limit
cutoffs already bit this project — durable file-per-agent + Workflow resume recovered both).

## Parked next-fix candidates (mention in the recommendation, do NOT build unasked)

1. Bare topic-phrase queries occasionally route to soft genome_tags instead of
   require_keyword_concepts ("post apocalyptic survival" idx 470, Blair-Witch found-footage idx
   388) — a fewshot would tighten; the same class mostly WORKS (winter/summer/serial-killer/
   secret-societies all recovered).
2. "live-action" (idx 176) resolves to nothing — right route is exclude_genres ["Animation"]
   (prompt guidance).
3. REJECTED: unsupported_notes salvage rung — only 1/46 notes resolver-salvageable.

## State of the tree / artifacts

- **Committed & pushed:** step-4 transparency layer — `src/llm_frontend.py` (intent echo,
  rec_provenance, relaxation/capability notices incl. honest-empty + thin-pool tail,
  out_of_domain report, SHOWCASE_QUERIES, 'scare me' mood alias), `src/llm_frontend_prompt.py`
  (out_of_domain + unsupported_notes schema/prompt + 2 fewshots), trace/probe renderers.
  Report fields are presentation-only; rec 4-tuples unchanged.
- **Local, gitignored (do not expect on a fresh clone):** `tools/results/traces/run500/`
  {ext,records,grades}_step4/, grades_step4.json, compare_step4.py, aggregate_step4.py,
  step4_remeasure_summary.md; baseline = grades.json (32.0%) + records/.
- **GATED, unchanged:** `serving/` + `streamlit_app.py`. Nick's decision 1 (Streamlit wiring +
  gated export; recommendation = bake `keyword_to_movieIds` ~1–2MB into feature_store.pt) is
  STILL OPEN — do not touch without his explicit go.
- Step-4 extraction telemetry at 500-scale: out_of_domain 1/500 (idx 443 meme lookup — the judge
  rubric's own unservable example); unsupported_notes 46/500, genuine residue.

## Working style (unchanged)

165/165 (`python tools/llm_frontend_eval.py`) after any change; live behavior via the mock loop
(`get_mock_serving()`), never export. Surgical diffs, house style. Commit only with Nick's
explicit go — never commit+push in one command. State scope assumptions in one sentence and
proceed; batch open questions for Nick at the end.
