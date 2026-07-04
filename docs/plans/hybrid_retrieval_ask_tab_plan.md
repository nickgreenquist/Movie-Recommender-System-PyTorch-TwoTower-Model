# Ask-Tab Signal Resolution — Build Plan (genome-first)

> **What this is.** The single forward plan for the LLM movie front-end (Streamlit "Ask" tab).
> Supersedes the run500 grading/concept-patching handoffs (that work is DONE — see *Already done*).
> A fresh-context agent should continue from this doc + two memories
> ([[project_run500_quality_grade]], [[project_query_router_rejected]]) alone.
>
> **One-line goal:** stop asking the two-tower to "retrieve" cold queries via popularity fallback.
> Instead, **resolve the query's terms against the signals we already have (genome matrix + TMDB
> keywords), use them as anchors and/or hard filters, and let the model rank the full 9k corpus** —
> with a transparency layer so it degrades gracefully and is worth showing off.
>
> **⚠ BEFORE YOU START and AFTER EVERY CHANGE — validate against BOTH rulers (non-negotiable):**
> 1. **160-case regression** — `python tools/llm_frontend_eval.py` must stay **160/160** (no-API,
>    deterministic; hand-authored extractions → `recommend()`).
> 2. **The 500 real prompts** (or at least the 54-bad subset) via the **mock loop** — re-extract with
>    the current prompt (Haiku subagents) → `recommend()` against `get_mock_serving()` → Sonnet grade
>    → confirm improvement AND no new regressions. Tooling in *Tooling that already exists*.
> Do not declare any step done until both rulers agree it helped and broke nothing.

## The reframe that drives everything (read this first)

The corpus is **~9,375 movies** and the two-tower **scores every movie exhaustively** on each query.
So there is **no candidate-reduction / ANN problem** — a separate "retrieval index" is a solution to a
problem we don't have. The real problem is **signal resolution**: map the query to facets/anchors that
already exist, then let the model rank the full corpus.

And **the genome matrix (9,376 × 1,128 relevance scores) is already a content index** — dense,
per-movie, pre-scored. It gives us, for free:
- **Anchors (soft):** high-relevance movies for a tag → seed the two-tower → tag-flavored ranking.
- **Hard filter (already built!):** movies with `tag relevance ≥ θ`. This is the exact machinery
  behind "set in Japan" (`require_genome_tags`) — today it's just *described* as settings-only.

Worked example that motivated this plan: **"movies with dogs"** failed → popularity → generic
blockbusters. But `dog`/`dogs` ARE genome tags (IDs 316/317; 187 movies ≥0.5, 133 ≥0.7). The genome
index already *solved* this query — the extractor just never routed `dog` to it (it's a concrete
object, and the genome fields are *described* as "vibe" / "setting", so a concrete noun matches
neither description and fell through to an unwired keyword slot → popularity).

## Why popularity fallback is the wrong default

Popularity ranking ignores the two-tower entirely and returns the same IMDb-top-N for any signal-less
query — weak, and reached far too often. At 9k with full scoring, **almost any query with a genome or
keyword hit should be served by anchors/filter instead.** Popularity should be the **true last
resort**, not the common path.

## Target mechanism — a dynamic resolver, genome-first

Replace the hand-curated closed `KEYWORD_CONCEPTS` list as the *foundation* with a **dynamic
resolver**: the LLM emits free concept terms; the resolver matches them against the vocabularies we
already have and routes them to the right signal. No per-term hardcoding.

```
LLM extracts: hard filters (year/genre/people/rating/excl) + taste anchors + FREE concept terms
  → resolver, per term (e.g. "dog"):
      1. EXACT-token match vs GENOME vocab (1,128)      → genome hit:
            • use as ANCHOR (soft rank), and/or
            • require_genome_tags relevance filter (HARD)  ← generalize beyond "settings"
      2. else EXACT-token match vs TMDB keyword vocab (17,820) → keyword-membership facet
            (the long tail genome misses; homonym denylist applies)
      3. else → text-embedding semantic match (DEFERRED — vibe gaps like "autumn"/"neon" only)
  → rank the FULL 9k corpus: two-tower (with anchors / taste) over the filtered pool
  → popularity ONLY if there is genuinely NO signal (rare last resort)
  → existing hard-facet post-filter + graceful relaxation (deb1040)
```

Key properties: **genome-first** (graded relevance beats boolean membership for ranking, and it's
already computed); **no hand-curation** (terms resolve because they're *in the data*); **interpretable**
("matched genome tag `dog` @0.82" is a perfect provenance chip); **the two-tower stays the ranker**
(its real job — cold queries just get genome anchors instead of a null user embedding).

## Build steps (in order; validate each against BOTH rulers before moving on)

1. **Generalize `require_genome_tags` beyond settings. — ✅ DONE 2026-07-03 (uncommitted).**
   Prompt/schema-only change (zero harness edits): schema + system-prompt descriptions un-restricted
   to any genome-vocab SUBJECT/TOPIC, with an explicit ROUTING ORDER — genre → keyword-concept list →
   verbatim genome tag → drop/soft. **Concept-list-first for dual-membership terms** (dog, samurai):
   a misroute to the concept path is benign (boolean floor still correct), a misroute to a
   nonexistent genome tag silently collapses to popularity — the deterministic genome-first tie-break
   is step 2's resolver job, not Haiku's. **θ decision: keep `GENOME_HARD_FLOOR=0.35`** — measured
   top-15 is IDENTICAL at 0.35 vs 0.5 (REQUIRE_GT_RERANK_LAMBDA=2.0 puts 0.9+ carriers on top
   regardless); the floor only sets tail depth, and 0.35 leaves pool room for compound filters
   (trains: 44 films @0.35 vs 25 @0.5). Also added AND-semantics discipline after a live regression:
   "a city like Tokyo or Paris" got hard-AND'd → now exemplars/"or" alternatives stay SOFT.
   Validated: regression **165/165** (5 new `genome_topic_facet` cases: dogs / courtroom+Drama /
   journalism / high school+2000s fence / wedding); mock loop on 13 bad + 6 good-spot-check idxs →
   4 target improvements (160 ocean/sharks bad→GOOD, 239 hs-noir → Brick #1, 298 hs-reunion, 389),
   0 real regressions (269 exemplar-AND found+fixed+re-verified; 243/307 = judge/Haiku variance).
2. **Dynamic resolver, genome-first.** LLM emits free concept terms → exact-token match vs genome
   vocab, then TMDB keyword vocab. Route genome hits to anchor+filter, keyword hits to the membership
   facet. Gotchas to design around (these are why the old list was hand-curated):
   - **Exact-token match, never substring** (cat≠catastrophe); keep a small **homonym denylist** for
     semantic homonyms an exact match still can't catch (bat, ring, heist/atheist).
   - **Confidence/relevance gate** before a HARD filter fires (a false hard filter empties the pool).
   - **Genome-vs-keyword tie-break:** prefer genome's graded relevance for ranking; keyword membership
     as a boolean floor.
   - **Soft vs hard:** hard-filter when the term is the *defining topic* ("dog movies"); soft-anchor
     when it's a modifier on a broader vibe. Reuse the empty-pool relaxation for over-tight filters.
3. **Demote popularity to last resort** in `recommend()`: it fires only when no genome/keyword/anchor
   signal resolved. (Keeps the settled rule that popularity is the correct *floor* for a genuinely
   bare category — just reached far less often.)
4. **Transparency / polish layer** (what makes it portfolio-grade, not just functional):
   a. **Intent echo** — "Understood: 🐕 dog · cozy vibe · ranked by taste/popularity."
   b. **Per-rec provenance chip** — `matched: dog @0.82, heartwarming`.
   c. **Graceful relaxation notice** — "no exact match for 'single night', showing closest, relaxed
      that filter." (Seed exists: `deb1040`.)
   d. **Honest capability boundary** — ceiling asks (specific plot twist / craft technique): "I match
      themes/vibe/topics/people/era — not specific plot twists. Here's my closest read."
   e. **Out-of-domain catch + seeded example chips** — "a game with dogs" → "I only know movies 🎬";
      seed the box with 4–6 queries that shine.
5. **Text-embedding — DEFERRED, semantic-gap only.** Build only if grading shows a real residual of
   vibe/aesthetic queries neither vocab has a literal tag for (autumn, neon, campfire). Dense-embed
   item docs; consulted only when the resolver + genome + keyword all miss. Do NOT lead with this.
6. **Streamlit** last: only after both rulers pass on the mock loop, wire into the Ask tab, then the
   gated `serving/` export if needed.

## Validation protocol (the user explicitly asked future-you to remember this)

Every change runs through **both** rulers, always:
- **Regression (must stay 160/160):** `python tools/llm_frontend_eval.py`. Adding signals is usually
  non-destructive, but the resolver + threshold + demoted popularity CAN move existing cases — check.
- **Quality (the 500 / 54-bad):** mock loop — regenerate the system prompt, re-extract the target idxs
  with Haiku subagents (`gen_workflow.py`), `recommend()` against `get_mock_serving()`
  (`phase2_fix1.py`), grade with Sonnet (`gen_grade_workflow.py`), diff with `compare_progress.py`.
  Watch for ~±5 run-to-run noise (stochastic Haiku × judge) — trust the *targeted-class* delta, and
  separate real regressions from noise by checking whether recs actually changed.

## Done criteria (stop here — do NOT chase the long tail past this)

- `require_genome_tags` generalized + resolver routing genome-first + popularity demoted.
- Genome-covered topics served from genome (dog, samurai, etc. — verified in the mock loop).
- **160/160 regression green** AND the 54-bad quality re-measured up with no real regressions.
- All 5 polish items live; ~20 casual-query smoke test: every query succeeds or **degrades
  gracefully** (never a silent generic popularity dump). Bar = *"never looks broken, always looks like
  it understood,"* NOT total coverage. Another unhandleable long-tail input is fine if it degrades
  transparently.

## Guardrails

- **Don't bury the thesis.** A signal-resolution Ask tab still leans on the two-tower as ranker, but
  cold queries won't showcase *personalization*. Give the two-tower its own taste-driven showcase
  ("pick 5 films you love → personalized recs") or lean the Similar/Explore tabs into it. Framing:
  Ask tab = *front bookend* ([[reference_bookend_llm_framework]]); two-tower tab = *the model*.
- **Serving/export is gated.** `serving/feature_store.pt` is git-TRACKED (~84MB) and feeds the live
  deploy. Do NOT re-export or touch `streamlit_app.py` until mock-verified good. Iterate via the
  mock loop (`get_mock_serving()`), never by exporting.
- **Do NOT resume hand-adding `KEYWORD_CONCEPTS`.** The resolver against the full vocab replaces
  curation. Existing concepts stay (harmless, high-precision homonym-safe core); don't grow the list.

## Don't-relitigate (settled, with evidence)

- **The problem is signal resolution, not retrieval** — 9k corpus, model scores everything; genome
  matrix is already a content index. No ANN / retrieval-index needed for scale.
- **Hard LLM router** → rejected ([[project_query_router_rejected]]): hybrids dominate, misroutes
  silent. Resolve-and-rank pipeline, not a switch.
- **Classic RAG / LLM-as-retriever** → not this. We rank the real corpus; the two-tower + cheap
  harness supercharging a traditional model IS the thesis.
- **Popularity = correct floor** for a genuinely bare category — but a last resort, not the default.
- **Single-title "like X"** ranks in GENOME space (combined item embedding is era/co-watch biased).
- **Explicit per-word term→facet mapping** → rejected (that's the whack-a-mole); the resolver is the
  dynamic, data-driven alternative.

## Already done (context, not tasks — do NOT redo)

- **Build step 1 — DONE 2026-07-03, UNCOMMITTED** (see the step-1 entry above for full detail).
  Files: `src/llm_frontend_prompt.py` (routing + AND discipline), `eval_cases.json` (+5 topic
  cases → ruler is now **165/165**), `tools/llm_frontend_eval.py` (see next bullet). Step-2 evidence
  banked from the loop: idx 243 ("trapped underwater") shows a PURE keyword-concept extraction has no
  taste signal → popularity order inside the concept pool floats Austin Powers over Das Boot unless
  Haiku happens to emit soft tags — exactly why the resolver should genome-first dual terms
  (submarine IS a genome tag → anchors + graded re-rank) and why popularity must demote (step 3).
  Step-1b mock artifacts: `ext_step1{,b}/`, `records_step1{,b}/`, `grades_step1{,b}/`.
- **Regression-ruler store fix (same session):** `tools/llm_frontend_eval.py` now swaps the local
  `llm_features/cache/facet_store.pt` `movieId_to_keyword_concepts` table into `ctx.facets` (same
  single-table patch as `mock_serving.py`). Before this, plain `Serving()` preferred the STALE
  57-concept store baked into git-tracked `serving/feature_store.pt`, so the 95-concept fix#1/#2
  code was being ruler-tested against a store missing 38 concepts (3 oracle cases failed; a
  require on any new concept would empty-pool). serving/ itself remains untouched/gated.

- **500-query grade + triage + two fix waves — DONE, UNCOMMITTED.** fix#1 (34 keyword/season/decade
  concepts) + fix#2 (extraction-routing prompt guards + `reunion`/`dog`/`cat`/`horse`). Regression
  **160/160**; concepts now 95; measured recovery on 54-bad 26→31 (±~5 noise). Full detail:
  [[project_run500_quality_grade]]. **These stay as the high-precision facet core** — but the concept
  path is CLOSED; the genome-first resolver is the new foundation. (Note: `dog` is BOTH a genome tag
  and now a keyword concept — genome-first should prefer the genome tag for graded ranking.)
- Known low-priority fix#2 micro-regressions: idx 11 (decade guard over-suppresses "nostalgic *for*
  the 90s"), idx 195 (spurious `outer space` emit). Fold in or ignore.
- Uncommitted files: `src/llm_frontend.py`, `src/llm_frontend_prompt.py`,
  `docs/llm_frontend/validation/retrieval_eval/eval_cases.json` (3 oracle updates). `serving/`
  untouched.

## Tooling that already exists (use it — mock loop = no API key, no export)

- `tools/results/traces/run500/mock_serving.py` — `get_mock_serving()` = real `Serving()` with the
  locally-rebuilt `llm_features/cache/facet_store.pt` swapped into `ctx.facets`. **This is how you
  test front-end changes without exporting to `serving/`.** Rebuild the store after facet/vocab edits:
  `python llm_features/build_facet_store.py`.
- Extractor mock: `gen_workflow.py ids <ext_dir> <name> <idxs>` (embeds `build_system_prompt()`,
  fans out Haiku extractions to files). `phase2_fix1.py <idxs> [ext_sub] [rec_sub]` runs
  `recommend()` against the mock serving. `gen_grade_workflow.py <BATCH> [REC_DIR] [GRADE_DIR] [NAME]
  [IDXS]` fans out Sonnet grades. `compare_progress.py` diffs stages.
- Regression ruler: `python tools/llm_frontend_eval.py`
  (`docs/llm_frontend/validation/retrieval_eval/eval_cases.json`, **160/160**).
- Grade + triage artifacts (local, gitignored): `tools/results/traces/run500/{grades.json,
  grades_summary.md, triage.json, ext/, records/}`. The 500 queries: `.../queries.json`.
- Genome vocab lookups live in `serving/feature_store.pt` (`genome_tag_names`, `genome_tag_to_i`) and
  per-movie genome context; the raw TMDB keyword index was built in scratchpad from
  `llm_features/cache/scraped/*.json` (regenerable — 17,820 distinct keywords).

## Pointers

- Memories: [[project_run500_quality_grade]] (grade + triage + fix#1/#2 + mock-loop recipe + this
  plan's forward pointer), [[project_run500_gap_analysis]], [[project_query_router_rejected]],
  [[reference_bookend_llm_framework]], [[project_facet_store_plan]].
