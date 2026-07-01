# v5 validation — RESUME (conditional tag-gate)

**Goal:** confirm the v5 prompt fix undoes v4's see-saw — keep soft_genre's recovery (≈2.86) AND
restore constraint (toward v3's 3.67), by gating the "NO titles → always emit mood tags" rule on
soft-vibe vs hard-constraint. Validated on the **32-case subset** where the gate changes behavior
(`constraint` 15 / `soft_genre` 7 / `family` 5 / `era` 5). Budget-conscious (≈64 agents vs ≈320 full).

## State
- **v5 prompt edit DONE** — `src/llm_frontend_prompt.py`, the "NO titles given" bullet now branches:
  PURE VIBE / SOFT-GENRE TASTE → always emit 3–5 tags; HARD-CONSTRAINT (head-noun genre and/or
  year) → set require_genres/year, suppress invented tags. Renders clean (22,411 chars).
- **Stage 1 extraction** — `wf_extract_v5.js` (32 Haiku, embedded v5 prompt+schema+subset). Launched
  as workflow `w4p8qi4p3`. Save its `result` to `extractions_v5_subset.json` here.
- Stages 2–4 scripts written and node-checked. Baselines (v1/v3/v4 summaries) live in `../v4_resume/`.

## Steps to finish
1. Save the extraction workflow `result` → `extractions_v5_subset.json` (list of {id,cat,text,extraction}).
2. **Recommend (local, no agents):** `python docs/llm_frontend/validation/v5/run_recommend_v5.py docs/llm_frontend/validation/v5` → writes `cases_v5/case_*.json`.
3. **Judge (32 sonnet):** `Workflow(scriptPath: "docs/llm_frontend/validation/v5/wf_judge_v5.js")`; save its `result` for compare.
4. **Compare:** `python docs/llm_frontend/validation/v5/compare_v5.py docs/llm_frontend/validation/v4_resume docs/llm_frontend/validation/v5 <v5_judge_output>` → writes `judge_summary_v5_subset.json`, prints v1/v3/v4/v5 per-category recs + v4→v5 deltas.
5. **Finalize:** fold the v5 subset result into `../llm_frontend_haiku_validation.md` + memory `[[project_llm_frontend_validation]]`.

## Regenerate if lost
`wf_extract_v5.js` / `wf_judge_v5.js` are generated (prompt+subset embedded); re-run the generators in
the session, or: extraction subset = cats {constraint,soft_genre,family,era} from
`../v4_resume/cases_v4_full/`, ids in `subset_ids.json`. v5 system prompt frozen in
`extraction_system_prompt_v5.txt`.

## Don't
- Don't stack the judge pass with other fan-outs back-to-back (budget — see memory
  `feedback_agent_fleet_budget`). One workflow at a time; recommend (local) sits between extract and judge.
- Nothing committed; working tree = v5 prompt (v4 harness still in place).
