# Ask-Tab Wave-1 + Wave-2 Execution Handoff (post bad-136 triage, 2026-07-04)

> **Who you are.** A fresh-context session; this file is your memory. Your predecessor executed
> the bad-136 triage (`tools/results/traces/run500/bad136_triage.md` — read it, it is the
> evidence base for everything below). Nick reviewed it 2026-07-04 and said: **"I agree with your
> assessment and recommendations"** — wave 1 and wave 2 are green-lit for BUILD + VALIDATE.
> Committing still requires his explicit go (never commit+push in one command).
>
> **Read first:** this file → `tools/results/traces/run500/bad136_triage.md` (headline + lever
> sections + per-item tables) → memory `project_bad136_triage` + `project_run500_quality_grade`
> (mock-pipeline block) → `docs/plans/hybrid_retrieval_ask_tab_plan.md` §Guardrails +
> §Don't-relitigate (binding).

## Scope

- **Wave 1 — routing/fewshot wave (clusters A + F + C, 40 idxs):** prompt-side routing
  discipline in `src/llm_frontend_prompt.py`, two one-line resolver-table touches + one small
  franchise-exclude change in `src/llm_frontend.py`.
- **Wave 2 — union match-count ordering (cluster B, 10 idxs):** one rank-layer change in
  `recommend()`.
- **NOT in scope (queued, do not build):** E sim-centering (separate gated validation spike),
  D Mode-1.5 like-X design, CS disclosure fewshots for the 13 silent ceilings (Nick's open Q4 —
  reconfirm before touching), G hidden-gem ordering, H studio facet, Streamlit wiring / export
  (Decision 1 STILL OPEN — `serving/` + `streamlit_app.py` remain GATED).

## Verified facts you can trust (probed 2026-07-04 against the local facet store — don't re-derive)

- `resolve_topic_term()` (`src/llm_frontend.py:911`) rungs: genome vocab → curated concepts →
  raw `keyword_to_movieIds`. Verified member pools: `small town` 807, `dreams` 904, `1970s` 944
  (genome has decade tags natively), `true story` 767, `murder mystery` 621, `car chase` 589,
  `underdog` 342, `aging` 383, `military` 359, `utopia` 272, `gothic` 254, `musicians` 247,
  `isolation` 251, `villain` 214 (kw), `college` 193, `reality tv` 160, `con artists` 137,
  `amnesia` 84, `mother daughter relationship` 69, `whodunit` 58 (kw), `vacation` 46 (kw),
  `breaking the fourth wall` 43 (kw), `halloween` 30, `patriotism` 26 (kw), `europe` 21 (kw),
  `one night` 21 (kw), `neo western` 15 (kw), `time loop` 14, `deaf` 14 (kw), `election` 12 (kw).
  Resolve-to-NOTHING (true voids — do not chase): neon, autumn(7, thin), one room, single
  location, silence, breakup, wanderlust, fake relationship, old age, day/daylight.
- Multi-term require topics OR their member sets (`src/llm_frontend.py:1937`,
  `topic_pool_ok | members`) — deliberate (cross-channel empty-AND fix). The gate is right; the
  ORDER is the bug (wave 2).
- `campaign` raw keyword = 6 films, mostly military campaigns (Last Samurai, Kelly's Heroes) —
  homonym.
- Franchise: `facets['franchise_universe_aliases']['marvel']` expands to 10 MCU collections only;
  Fantastic Four / Ghost Rider / Blade / Punisher have their own collections (pass the exclude),
  Elektra + Incredible Hulk have `collection=None` (can NEVER match collection-based exclusion).
  Genome tag `marvel` (110 members) contains every offender — verified. Matching code:
  `_franchise_match` (~:1254), applied at ~:1355–1362.
- 165/165 ruler = `python tools/llm_frontend_eval.py` (it swaps the LOCAL facet store's
  keyword-concepts table — trust it, not the stale baked serving copy).
- The step-4 report fields (`intent_echo`/`capability_notice`/`relaxation_notice`) are
  presentation-only and recomputed per call — your changes must keep rec 4-tuples' shape
  unchanged.

## Wave 1 — what to build

**Guardrails first:** do NOT re-add a concept/vocab enumeration to the prompt (step 2
deliberately deleted the concept-list injection; re-adding a vocab dump relitigates that).
Teach PATTERNS via a small number of worked fewshots + one-line guidance rules. Do NOT grow
`KEYWORD_CONCEPTS` (standing guardrail). Allowed table touches are exactly two:
`TOPIC_HOMONYM_DENYLIST += 'campaign'` and an alias-precision trim (see A6).

### Cluster A — routing fewshots/guidance (35 idxs)
`[11, 26, 28, 31, 45, 50, 90, 96, 130, 150, 151, 176, 182, 189, 195, 224, 230, 231, 241, 245,
246, 253, 260, 274, 275, 280, 288, 383, 388, 407, 422, 424, 451, 453, 470]`

Patterns to teach (per-item evidence: triage MD table A):

1. **Subject nouns → the hard path.** A concrete subject/setting/topic noun goes in
   `require_keyword_concepts` (or `require_genome_tags`), NOT soft `genome_tags`, NOT mood.
   Worked fewshot candidates: "cozy small-town vibe" → require small town; "gothic romance" →
   require gothic + Romance; "nostalgic for college life" → require college. Covers 26, 28, 31,
   50, 96, 130, 151, 182, 189, 230, 231, 246, 274, 275, 280, 288, 388, 407, 424, 451, 470.
2. **Canonical SHORT term form.** Emit 'space' not 'space exploration'; 'election' not
   'political campaign'; the resolver is exact-token. A resolver miss already degrades honestly
   (capability notice), so emitting is SAFE — try a topic term BEFORE parking a phrase in
   `unsupported_notes`. Converts written-off "ceilings": fourth wall → 'breaking the fourth
   wall' (241), one-single-night → 'one night' (253), 70s-setting → '1970s' (383), whodunit
   (182), patriotism (26/195). NOTE: this is prompt-side FORM discipline, distinct from the
   REJECTED notes-salvage rung (verbatim notes only resolve ~1/46).
3. **Era/decade discipline (extends the existing decade guard).** "nostalgic FOR the 90s" /
   "feels like the 90s" → the decade SETTING concept (fix#2 micro-regression, 11); "feels like
   high school in the early 2000s" → year window + high school tag (45); "movies from the 80s"
   stays a year window (existing guard — don't break it); release-year ≠ setting: "western set
   in modern times" → 'neo western', not year_min (245); "popular right now" → year_min ≈
   corpus max (2023) (422).
4. **Slot-routing one-liners:** "live-action" → `exclude_genres ['Animation']` (parked
   candidate #2, 176); "animated" → require the native **Animation GENRE**, not the noisy
   'animated' attribute (453); vacation/Europe → 'vacation' topic + require_country (the
   country resolver has a multi-code 'european' region entry — referee-verified; confirm the
   exact key in `resolve_facet`/country tables and wire 'Europe' to it) (150).
5. **Comfort-vibe tag choice.** "lazy Sunday / cozy / comfort" queries → feel-good/gentle/
   heartwarming anchor tags, NOT meditative/dreamlike/slow-paced (arthouse-coded in genome
   space; they swapped a good list for Bergman/Tarkovsky — 90).
6. **Table touches:** add `'campaign'` to `TOPIC_HOMONYM_DENYLIST` + guidance "political
   campaign → 'election'" (260); alias precision so marijuana-DRAMAS stop qualifying as stoner
   COMEDY — either trim the 'marijuana'→stoner alias or teach the prompt to pair the stoner
   concept with require Comedy when the ask is comedic (424 — referee-endorsed; pick the
   narrower diff at execution).

### Cluster F — social-context embedded preference (3 idxs prompt-only + 1 stretch)
`[121, 137, 161]` + stretch `[173]`

Fewshot the EXTRACT pattern: mine the taste signal inside a social frame — "watch with my
teenage son" → liked_genres Action/Adventure/Sci-Fi (137); "grandfather who was in the
military" → 'military' topic/genome (359 members) (161); "double date night" → romantic/fun
crowd-pleaser mood (121). A pure social frame with NO taste signal stays empty → popularity +
honest echo (that part is correct today; 101/140 are the judge-rubric's own unservable class —
don't chase). **173 (British drama) is NOT prompt-only** — its lever is country/genre-PRIMACY
ordering inside a filtered pool (half its list is indefensible: Dead Man Walking has no UK
credit) — treat as a stretch goal; if it isn't a small diff, leave it and note it.

### Cluster C — franchise-universe exclude (1 idx)
`[169]`

When an `exclude_franchise` spec names a UNIVERSE (marvel/dc — i.e., it's a
`franchise_universe_aliases` key), also drop films via the topic resolver's genome rung
(exclude semantics: genome ∪ concept — `resolve_topic_term(ctx, spec, exclude=True)`), so
collection-less Marvel films (Elektra) and non-MCU Marvel collections (FF/Ghost Rider/Blade/
Punisher) are caught. Keep plain collection specs ("no Saw movies") on the existing
`_franchise_match` path. Surgical: gate the new behavior on alias-key membership.

## Wave 2 — union match-count ordering (10 idxs)
`[214, 222, 228, 240, 247, 264, 282, 290, 291, 300]`

In `recommend()`'s topic machinery (~:1922–1977): the union GATE stays (never re-introduce a
hard AND across sibling terms — that's the settled cross-channel-empty fix). Change the ORDER:
films matching MORE of the resolved require terms rank first (match-count as the primary
re-rank key over `topic_require_resolved` member sets), then the existing graded
REQUIRE_GT-strength re-rank / anchor ordering within a match-count band. Implementation freedom:
a strong per-extra-term re-rank bonus is fine if it's cleaner than a hard sort key — but it must
be decisive (a 2-of-2 match must beat any 1-of-2, or the flagships below won't convert).

**Acceptance evidence (from the triage, all verified in-corpus):** idx 264 "train journey crime"
must surface *Murder on the Orient Express (1974)* (it's in the 12-film train∩crime
intersection); 282 → *Cast Away*; 291 → *The Thomas Crown Affair*; 290 → RV / National
Lampoon's Vacation-class. Known secondary issue NOT to fix here: 228's attribute∩topic
AND-across-gate-types over-tightening (2-film list) is by design; match-count doesn't change it.
222 (chef/restaurant keyword-incidentality) may only partially convert — keyword prominence is
an accepted limitation (memory `project_facet_expansion_measurement`).

## Validation (binding — the user's standing method)

After EVERY change: `python tools/llm_frontend_eval.py` → **165/165**. Never validate by
exporting; live behavior via the mock loop (`get_mock_serving()`).

Per-wave measurement (mock pipeline, tools in `tools/results/traces/run500/`, all parametrized —
usage in memory `project_run500_quality_grade`):
1. Fresh Haiku extractions for the wave's idx list on the NEW prompt (`gen_workflow.py`
   fan-out pattern; durable file-per-agent — two 5h-limit cutoffs have hit this project;
   `Workflow` resume recovers).
2. `python phase2_fix1.py <idxs> ext_wave1 records_wave1` (and `_wave2`).
3. Sonnet judge via `gen_grade_workflow.py <batch> records_wave1 grades_wave1 wave1 <idxs>`.
4. Compare vs `grades_step4.json`. Success bar: net conversion on the wave's idxs at roughly the
   historical ~50% rate, ZERO real regressions — before crying regression, check whether recs
   actually changed (`records_step4/rec_<idx>.json`) and remember ±5/run judge noise.
5. Regression spot-check beyond the ruler: re-run ~15 baseline-GOOD idxs (prompt changes are
   global); recs-identical-or-equivalent expected.

Report results to Nick with the same before/after framing as the triage (per-idx table +
conversion count), then STOP — commit only on his explicit go, docs/memory updates ride with
the commit decision.

## Working style (unchanged)

Surgical diffs, house style, every changed line traces to a cluster above. `serving/` +
`streamlit_app.py` untouched. No full training runs. State scope assumptions in one sentence
and proceed; batch open questions at the end. Fleets sized to tens, judge stage budgeted first.
