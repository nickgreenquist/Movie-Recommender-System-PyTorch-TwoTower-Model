# Scraped-Facet Store — Design Spec & Build Plan (v1.5)

> **▶ HANDOFF PROMPT — paste this verbatim after clearing context:**
>
> Read `docs/plans/facet_store_plan.md` in full — it is a self-contained design spec for the next
> major feature on the LLM conversational front-end: a **scraped-facet store** that lets requests like
> "Tom Hanks movies" / "directed by Sofia Coppola" work, by mapping people → movies from TMDB data we
> already scraped but never wired in. You (Claude) wrote this spec in a prior session and have **no
> memory of it** — the plan is the source of truth.
>
> Before writing any code: (1) reload the front-end's current state by skimming `src/llm_frontend.py`,
> `src/llm_frontend_prompt.py`, and the validation writeup `docs/llm_frontend_validation/llm_frontend_haiku_validation.md`
> (the front-end is at an unshipped "**v5**" prompt + "**v4**" harness in the working tree; nothing is
> committed); (2) confirm `llm_features/cache/scraped/` still holds the scraped JSONs. Then **start at
> Phase 0** (build the facet store + `resolve_person`, smoke-test) and **STOP for my review before the
> Phase 1 schema/prompt changes**. Follow `CLAUDE.md` working style: surgical changes, smoke-test only
> (never run training), and ask before committing or pushing.

---

## TL;DR

The two-tower model represents a movie by genome tags + genres + user-tags + an LLM-distilled 132-dim
vector + movieId + year. The richest discriminative facets from the TMDB/Wikipedia scrape — **cast,
director, writers, studio, keywords** — were collapsed away (only the 132-dim semantic vector
survived). So today the front-end **cannot** serve "movies with Tom Hanks", "directed by X", "scored
by Hans Zimmer": the extraction prompt correctly *drops* those as unsupported, leaving an empty query
that falls back to popularity.

This feature builds a **facet store** from the already-scraped data (99% cast/director coverage, with
canonical TMDB person IDs) and wires two paths into the existing harness:
- **Filter path (MVP):** "Tom Hanks movies" → hard-filter the corpus to `actor=31` → the existing
  empty-embedding fallback ranks by popularity *within the Hanks pool* → his top films.
- **Seed path (later):** "movies *like* Sofia Coppola's" → expand person → representative films →
  feed as user-embedding anchors (reusing the genome-anchor machinery).

It unlocks the entire `unsupported` request class (currently impossible) and is the planned "v1.5
scraped-facet store" referenced in the validation doc's residual list. It **subsumes** the "Mode-3
constraint-seeding" idea (facet→movies→seeds is constraint-seeding generalized).

## Why now (motivation, grounded in measurement)

From the v1→v5 validation + a post-filter ablation done this session:
- The hard post-filter (`require_genres`/`exclude_genres`/year) does the **heavy lifting** on
  constraint requests — turn it off and ~85% of surfaced recs violate the request. Filters are
  deterministic, correct, and compose with a popularity fallback. **This feature extends that proven
  mechanism to people/keywords.**
- **16% of all inputs hit fallback** (empty extraction → popularity). Breaking down the fallbacks: most
  are *correct* (vague/injection, or pure browse like "70s cinema" with no taste to parse). But a hard
  core — the `unsupported` eval category (≈14/160: "Anything by Christopher Nolan", "Movies with Tom
  Hanks", "Films scored by Hans Zimmer", "Movies directed by women", "Nothing R-rated") — fall back
  **only because the requested facet isn't in the system**. No prompt can fix that; it needs data.
- These are not parse failures. The prompt is doing its job. The gap is a **capability/data** gap.

## The data reality (the enabling assumption — verified)

- Scraper: `llm_features/scrape.py`. Output cache: `llm_features/cache/scraped/{movieId}.json`,
  **9,366 files** present (≈ the whole 9,375 corpus).
- Each record nests a `tmdb` sub-dict with convenience fields **`cast`** (full, billing-ordered),
  **`director`**, **`writers`**, **`production_companies`**, `genres`, `runtime`, `release_date`,
  `vote_average/count`, plus **`details_raw`** (TMDB `/movie/{id}` incl. appended `keywords`) and
  **`credits_raw`** (`/movie/{id}/credits`).
- **Coverage: cast 99%, director 99%.**
- **Canonical person IDs** live in `credits_raw.cast[].id` / `.crew[].id` (e.g. Sally Field = 35).
  Index by ID, not name → unambiguous resolution; same-name people disambiguate for free. This avoids
  the fuzzy-title pain (`resolve_title`'s accent/year bugs) entirely.
- Keywords are under `tmdb.details_raw.keywords.keywords[].name`. Studios under
  `tmdb.production_companies` (names; IDs in `details_raw.production_companies[].id`).

## Key design decisions (the reasoning — do not relitigate without cause)

### 1. Filter path is the MVP; soft-seed is later (this is a deliberate priority flip)
The intuitive ordering is "seed the embedding with matched movies, add a hard filter later." Reverse it:
1. **It matches the default intent.** "Tom Hanks movies" means *films he is in* — a membership query
   = a filter. "Movies *like* Tom Hanks's" (the seed intent) is the rarer phrasing.
2. **It composes with the validated fallback.** require_people + empty embedding → fallback →
   popularity within the person-filtered pool → that person's top films. Zero anchor tuning; reuses
   the exact filter+fallback machinery the ablation proved carries constraint requests.
3. **Actors are NOT a coherent embedding neighborhood — so seeding them is muddy.** The model's item
   embeddings cluster by co-watch + content (genre/genome/year). Tom Hanks spans *Big* (comedy),
   *Philadelphia* (drama), *Saving Private Ryan* (war), *Toy Story* (animation). Seeding the user
   embedding with all of them averages into a diffuse centroid → diffuse retrieval, and the neighbors
   it returns are *not even guaranteed to be Hanks films* (the model has no actor concept). The filter
   gives the deterministic membership the embedding cannot.

### 2. Soft-seed is facet-type-dependent
The seed path works **well for directors** (auteur style is somewhat coherent in embedding space) and
**keywords/vibes**, **poorly for actors** (centroid problem above). So when built, default actor →
filter, director → offer both, keyword → seed.

### 3. Reuse existing machinery (small new-code surface)
- **Filter** → extend `_passes_constraints(mid, fs, hc)` in `src/llm_frontend.py` to also check
  person-ID membership. The fallback path already calls it, so the empty-embedding case is free.
- **Seed** → `people_to_anchors()` is `anchors_for()` generalized: today genome-tag → top-relevance
  movies → seeds; here person-ID → top-billed/most-popular films → seeds. Same shape, same
  subordination-to-named-titles weighting.
- **Resolution** → `resolve_person()` mirrors `resolve_title()` but is *simpler* (exact normalized
  name → ID; IDs disambiguate). No fuzzy-year guards needed.

### 4. Must be baked into `serving/` at export time
The deployed app (Streamlit) and the harness load **only `serving/`**; `data/` and
`llm_features/cache/` are gitignored / absent on Streamlit Cloud. So the facet tables must be exported
into a serving artifact (mirroring how `export` already bakes `llm_feature_buffer` into
`serving/feature_store.pt`). Recommend **folding into `feature_store.pt`** (it already holds all the
`movieId_to_*` metadata dicts — `movieId_to_people` sits naturally beside `movieId_to_genres`); a
separate `serving/facet_store.pt` is the modular alternative.

## Architecture

### Build-time (data → serving)
New builder (e.g. `llm_features/build_facet_store.py`, run as a pre-export step or folded into
`src/export.py`). Reads `llm_features/cache/scraped/*.json`, emits:
- `movieId_to_people: {mid: {'actors': [pid…], 'directors': [pid…], 'writers': [pid…]}}`
  — **actors capped to the top-N billed** (design knob; start N≈10 — beyond that is bit parts, and
  "Tom Hanks movies" wants films where he leads, not cameos). Directors/writers: all.
- `person_id_to_name: {pid: 'Tom Hanks'}` (display + reverse lookup).
- `person_name_to_ids: {normalized_name: [pid…]}` for resolution (`_norm_name` ≈ `_norm_title`'s
  accent-fold + lowercase + punctuation strip, no article inversion).
- (Phase 3) `movieId_to_keywords`, `movieId_to_studios`, and derived attribute tables (e.g.
  `director_gender` from TMDB `credits_raw.crew[].gender`, for "directed by women").
Export bakes these into `feature_store.pt`. Keep it int-keyed and compact.

### Inference (front-end: `src/llm_frontend.py` + `src/llm_frontend_prompt.py`)
- **Schema** (`build_schema`): add people facets, soft vs hard, mirroring the genre soft/hard split —
  `liked_people: [str]` (soft) and under `hard_constraints`: `require_people: [str]`,
  `exclude_people: [str]`. (Phase 3: `require_studios`, `require_keywords`, attribute flags.)
- **Prompt** (`_SYSTEM_TEMPLATE`): remove actor/director/writer/studio from the "NOT SUPPORTED →
  silently drop" list; teach the soft/hard distinction — "movies **with/by/only** X" / "X movies" =
  HARD `require_people`; "I **like** X", "movies **like** X's" = SOFT `liked_people`. Keep
  content-rating/runtime unsupported for now (no clean data). Names are emitted as free strings (the
  harness resolves to IDs); do not enum-constrain (the person vocab is huge).
- **`resolve_person(raw, ctx) -> (pid|None, note)`**: `_norm_name(raw)` → `person_name_to_ids`. Exact
  normalized hit → ID (popularity/most-films tie-break on collision). Miss → None + report (drop, like
  unresolved titles). Optional light fuzzy as a last resort, high cutoff.
- **Filter path** (`_passes_constraints`): for `require_people`, movie must include **all** required
  person IDs (consistent with `require_genres` = ALL semantics); `exclude_people` drops any match.
  Check against `movieId_to_people[mid]` (union of actors+directors+writers, or role-scoped if the
  schema distinguishes — start union-of-all for simplicity).
- **Seed path** (Phase 2, `people_to_anchors`): for `liked_people`, expand each ID → top-K
  representative films (rank by popularity, or billing order for actors) → add to
  `liked_with_weights` at an anchor weight, subordinated to any named titles exactly like
  `anchors_for`.
- **Fallback interaction (the MVP magic):** require_people with no other signal → `fallback=True` →
  the fallback branch walks `popularity_ordered_titles` applying `_passes_constraints` → returns the
  person's most-popular films. Already works once the filter knows about people.

### Bonus: structured `facet:value` syntax
Pre-parse in the harness/UI: raw text matching `^(actor|director|writer|studio):(.+)$` routes straight
to `resolve_person` + hard filter, bypassing the LLM. A cheap power-user path once the store + resolver
exist; the LLM may also emit it.

## Phased build plan (front-load value, defer risk; stop/review between phases)

- **Phase 0 — Facet store + resolution (no inference change).** Build the builder; produce the tables;
  add `resolve_person`. Smoke-test deterministically: "tom hanks" → person ID → list his catalog films;
  spot-check a director and a same-name collision. **STOP for review.**
- **Phase 1 — Filter MVP.** Schema `require_people`/`exclude_people` + prompt extraction (HARD only) +
  `_passes_constraints` extension + export bake. End-to-end: "Tom Hanks movies" → his top films via
  fallback. Validate on the `unsupported` eval category (those cases go from fallback → served).
  **STOP for review.**
- **Phase 2 — Soft-seed.** `liked_people` + `people_to_anchors`. Start with directors; measure actors
  (expect muddy). Validate "movies like X's".
- **Phase 3 — Attribute facets + power syntax.** Keywords, studios, derived attributes ("directed by
  women" via TMDB gender), and the `actor:X` structured syntax.

## Validation approach (mirror the v1→v5 harness)

Pipeline already exists and is documented in `docs/llm_frontend_validation/`: extract (Haiku, forced
schema) → `recommend()` over real `serving/` → per-case sonnet judge. For facets, add:
- A **deterministic facet-correctness check** (no agents): for a `require_people` request, what % of
  the top-15 actually contain that person? (Analogous to the constraint-violation check in
  `docs/llm_frontend_validation/v5/ablate_postfilter.py`.) This is the primary metric — for a hard
  facet, membership *is* correctness.
- Extend the eval set's `unsupported` category with more actor/director/studio/keyword cases.
- Watch coverage/sparsity: a given actor has ≈10–25 films in the 9,375 corpus — fine for
  filter+popularity, thin for pure soft-seed.

## Caveats / risks (carry forward)

- **Schema reopening needs its own validation.** Adding facets reopens the "what's supported" surface;
  the LLM must reliably split "with X" (hard) from "like X's" (soft). Budget a judged pass.
- **Attribute facets are a second tier.** Name-facets (actor/director/studio X) are easy + high
  coverage. "Directed by women", "foreign", "silent", "black & white" need *derived* metadata (TMDB
  `gender` is incomplete; others need other signals).
- **Top-N billed cutoff is a real knob.** Too high → cameos pollute "X movies"; too low → misses
  genuine supporting roles. Start N≈10, tune on examples.
- **Actor-centroid problem** (see decision #1) — keep actors on the filter path.
- **False negatives** if the scrape missed a credit (rare at 99%, but a hard filter *excludes* on
  miss). Acceptable for headline actors; note it.
- **Cost:** this is the biggest item discussed — a data→store→schema→resolver→filter→validation
  feature, not a prompt tweak. The 99%-ready data removes the long pole, but it is multi-session.

## Context pointers (for fresh-context me)

- **Front-end current state:** working tree = "v5" prompt (conditional Mode-2 tag-gate +
  head-noun→`require_genres`) + "v4" harness (year-guard). **Nothing committed.** Full v1→v5 history,
  the post-filter ablation, and the residual list are in
  `docs/llm_frontend_validation/llm_frontend_haiku_validation.md`.
- **Key code:**
  - `src/llm_frontend.py` — `recommend()`, `resolve_title()`, `anchors_for()`, `_passes_constraints()`,
    fallback logic, `FrontendContext`.
  - `src/llm_frontend_prompt.py` — `build_schema()`, `_SYSTEM_TEMPLATE`, `build_system_prompt()`.
  - `src/llm_frontend_extraction.py` — the hosted (Streamlit "Ask" tab) extraction path; keep it in
    step with the schema/prompt.
  - `tools/llm_frontend_probe.py` — `Serving` loader + `--smoke`; the in-repo QA harness.
  - `src/export.py` — bakes serving artifacts (the pattern to mirror for the facet tables).
  - `streamlit_app.py` — loads `serving/` only; must build the same model + facet store from `serving/`.
- **Scrape:** `llm_features/scrape.py` (fields), `llm_features/cache/scraped/{mid}.json` (data; facet
  fields under the `tmdb` sub-dict; person IDs in `tmdb.credits_raw`).
- **Serving artifacts:** `serving/feature_store.pt` (vocabs + `movieId_to_*` dicts + baked buffers),
  `serving/movie_embeddings.pt` (128-d item vectors — the retrieval substrate),
  `serving/model.pth`. The front-end is **serving-only** (never reads `data/`).
- **Conventions (`CLAUDE.md`):** house style (long docstring headers, named offsets, NamedTuple
  bundles); **never run `python main.py train`** (hand training to the user); smoke-test then stop;
  ask before committing/pushing; do not touch `streamlit_app.py`/`src/export.py` for a change until
  it's verified good.
