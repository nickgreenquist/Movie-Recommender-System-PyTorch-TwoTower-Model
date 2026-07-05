# Ask-Tab Next-Levers Handoff (post waves 1+2, 2026-07-05)

> **Who you are.** A fresh-context session; this file is your memory. Your predecessor executed
> bad-136 waves 1+2 — committed+pushed as `a20b942` (that commit is your baseline). Nick approved
> this exact worklist 2026-07-05: **items 1→3 below, in order, then ONE full-500 re-measure.**
> **Export/Streamlit is DEAD LAST by Nick's explicit call — do not touch `serving/`,
> `streamlit_app.py`, or run `main.py export` for ANY reason in this worklist.** (Known and
> accepted: kw-rung wins stay mock-only until that final export.)
>
> **Read first:** this file → `tools/results/traces/run500/wave12_results.md` (what just landed +
> residuals) → `tools/results/traces/run500/bad136_triage.md` §levers D and E + tables D/E →
> memory `project_bad136_triage`.

## Binding method (unchanged from waves 1+2)

- After EVERY change: `python tools/llm_frontend_eval.py` → **165/165**.
- Measure via the mock loop only: `gen_workflow.py ids <ext_dir> <name> <idxs>` (fresh Haiku on
  the CURRENT prompt; embeds it at gen time) → `phase2_fix1.py <idxs> <ext_dir> <rec_dir>` →
  `gen_grade_workflow.py 10 <rec_dir> <grade_dir> <name> <idxs>` → merge grade_*.json →
  `compare_wave.py <name> <idxs>` (exists now; compares vs `grades_step4.json` + recs-changed
  flag). ±5/run judge noise; identical recs + verdict flip = noise. `get_mock_serving()` already
  patches `movieId_to_keyword_concepts` + `keyword_to_movieIds`.
- Fleets sized to tens; judge stage budgeted first. Commit only on Nick's explicit go (never
  commit+push in one command); docs/memory ride the commit.
- Don't relitigate: single-title like-X ranks in GENOME space; no hard router; popularity = floor
  for bare category; no KEYWORD_CONCEPTS growth; ceilings (triage CEILING tiers) stay declined —
  honest declines grade bad under the judge protocol (open Q3), don't chase them.

## Item 1 — mood-table micro-wave (~30 min, cheapest points)

Wave-1 found extractions that were PERFECT but died on `MOOD_TAGS` gaps (`resolve_mood` returned
[]): **'terrifying'** (idx 130: halloween + exclude_mood ['terrifying'] + mood fun — the anti-floor
never fired, list was Halloween-franchise slashers) and **'patriotic'** (26/195 — mood half lost).
Grep `MOOD_TAGS = {` and its alias map in `src/llm_frontend.py`. 'terrifying' (+ 'genuinely
terrifying' phrasing) → alias into the 'scary' family (exclude-side is what matters: the 0.4
anti-floor must drop strongly-scary films). 'patriotic' → probe what genome tags actually fit
(patriotism is NOT a genome tag — it kw-resolves as a topic; the mood may want an
inspirational-family route, or nothing — decide by probing, don't force it). Stretch (prompt-side
only): 451's lookup emitted 'television show' not 'reality tv' — consider one worked example for
tip-of-tongue lookups. Validate: ruler + mock loop on [130, 26, 195, 451] vs `grades_step4.json`.

## Item 2 — E-centering validation spike (gated, decide by data)

One-line candidate in `_content_similar_scores` (`src/llm_frontend.py`): mean-center the genome
matrix before cosine — `normalize(M − M.mean(0))`. Evidence (triage, live-verified): sparse 2018+
genome rows form a ~0.95-cosine hub; centered: Truman Show → Eternal Sunshine/Pleasantville/Being
There/Network (raw gave Bo Burnham/AlphaGo); BR2049 → GitS/Blade Runner 1982/Children of Men;
Whiplash → Sound of Metal/Crazy Heart. Adversarial verdict: seed-dependent (in-hub 2019+ seeds
like EEAAO unrescuable — the signal isn't in the data).

GATE before adopting: (a) ruler 165/165, (b) ~30-seed like-X re-grade in the mock loop — D idxs
[302,305,324,334,342,343,348,349,355,362,364,368,371,375,376,380,389,394,395] + E idxs
[315,322,326,360,366] + ~6 baseline-GOOD like-X regression guards (grep records_step4 for
`ranked_by_similarity: true` with step4-good verdicts). Adopt only on net improvement with no
guard regressions; else report and keep raw cosine. Optional rider if adopted: a "too new for my
content data" capability notice for 2019+ seeds (378-class) — presentation-only.

## Item 3 — D Mode-1.5 like-X qualifier-refine (design → build, after E)

The 19-item D table is the test set (its pool may shrink if E lands — re-check before building).
Documented failure: when `liked_items` arrives WITH mood/genre/topic qualifiers, anchors/gates
REPLACE similarity (334 Her → grief dramas; 362 Nice Guys lost Kiss Kiss Bang Bang; 348 Black
Swan's require-gate turned sim into art-docs). Design principle (triage): when a title resolves,
rank by SIMILARITY FIRST; qualifiers re-rank WITHIN the neighborhood (genome-expressible
qualifiers exist: amnesia 84, mother-daughter 69, con artists 137); a hard topic gate on a like-X
query should REFINE (re-rank), not gate the sim pool. Write the design into
`docs/plans/hybrid_retrieval_ask_tab_plan.md` first (house pattern), get the shape stable, build,
validate on the 19 + like-X guards + ruler.

## Then: ONE full-500 re-measure

After items 1–3 settle, refresh the headline (50.8% → ?) with one full-500 mock re-measure —
**ask Nick before launching the ~100-agent spend** (extraction + judge fleets; two 5h-limit
cutoffs have hit this project — durable file-per-agent + Workflow resume recovers).

## Batch for Nick at the end (don't build unasked)

- Q3 grading protocol for honest declines (recommended: keep strict series, report adjusted
  good_pct alongside). Q4: CS disclosure fewshots for the 13 silent ceilings (honesty, not
  good_pct) — still unconfirmed.
- Residuals intentionally left: 246 compound 'family secrets' form, 173 country-primacy ordering
  (not small), G hidden-gem, H studio. Export/Streamlit: dead last, after everything.
