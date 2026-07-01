# Scraped-Facet Store — Design Spec & Build Plan (v1.5)

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

## Expansion II — Handling the real ask distribution: membership facets + a vibe/affect path

> Added 2026-06-30, then **substantially revised the same day after a measurement pass** over the full
> scrape (9,366 records) plus two LLM prominence-judge sweeps (172 objects, ~210 actions/events). The
> original framing — "extract way more *metadata/props* from plot text to handle more query types" — was
> measured and **largely rejected**: an LLM-over-plot extraction (the old M2) is *not* additive over the
> crowd-sourced TMDB keywords for props, actions, themes, OR settings. See **"What the measurement
> overturned."** This section is reorganized around the **real distribution of user prompts** and splits
> the work into **three engines**, not one keyword-vocab phase. Measurement scripts + judge outputs:
> memory `project_facet_expansion_measurement`; scratchpad `m1*.py`/`m2_*.py`/`actions.py`/`queries.py`.

> ### ▶ IMMEDIATE NEXT STEP (resume here — 2026-06-30, paused end-of-day)
> **Continue the "Ask-AI holes" measurement, then apply the plan fix it produces.** In progress: a
> subagent run that (1) generated **500 realistic "Ask AI" prompts** across 50 user archetypes, (2)
> oracle-recommends each grounded in our real per-movie data, tagging every signal's coverage
> (HAVE/PARTIAL/MISSING) and layer (intake vs post-filter), (3) synthesizes the **holes** in this plan
> (missing data axes, intents the intake LLM isn't taught, filters we can't compute, unhandled intent
> classes, composition/thin-pool risks). The 500-prompt generation is DONE and saved; only the
> rec+synthesis remain. Run cost-controlled (**Sonnet**, ~20 agents at a time — the full 50 exhausts the
> usage window).
>
> **Durable artifacts** (survive session reset — under `docs/llm_frontend/validation/ask_ai_holes/`):
> `harvest_500_prompts.json` (the 500 prompts, 50 groups × 10, + 100 already-done Opus rec records),
> `corpus_helper.py` (grounding CLI: `prompts <i>` / `title` / `search` / `genome` / `kw`; needs
> `serving/feature_store.pt` + `llm_features/cache/facet_store.pt`; rebuild `agg.pkl` from the scrape via
> the scratchpad `aggregate.py` if absent), `wf_rec_20_subset.js` (**tomorrow's launch** — 20 hole-rich
> archetypes, Sonnet, rec-only), `wf_rec_synth_50.js` (full 50 + 6 synthesis lenses).
>
> **To resume:** point `corpus_helper.py` at the harvest, run `wf_rec_20_subset.js` via the Workflow
> tool, aggregate the returned coverage table + holes, then fold the findings into "The three engines"
> below as concrete new intake slots / post-filter checks / data-build tasks. THEN the earlier "commit
> the plan revision" question is still open.

### The real ask distribution (measured, not assumed)
The users' own example prompts, mapped to what actually carries the signal and which engine serves it:

| User ask (real phrasing) | Category | Signal & measured coverage | Engine |
|---|---|---|---|
| "French / Korean / foreign films" | nationality | `production_countries`, ISO-clean (FR 842, JP 297, KR 55…) | **1 · membership** |
| "black & white", "directed by women", "based on a book" | format/attribute | single clean top-25 tag (364 / 560 / 1149) | **1 · membership** |
| "boxing", "martial arts", "heist films", "space operas" | clean theme/subgenre | keyword head (boxing 48, martial arts 219, heist 225, space opera 45) | **1 · membership** |
| "food is the **main theme**" | theme + prominence | **keyword** clean (food-theme 93) vs plot noise (653) | **1** (plot *hurts*) |
| "movies in **Tokyo / NYC / Paris**" | setting | 217 `"city, region"` place tags; curated rollup | **1 · membership (location)** |
| "80s / medieval / **WWII**" | era | decade = native year feature; WWII = 36-tag semantic cluster | year native; era→**3** |
| "**Tom Hanks**", "directed by **Nolan**" | people | TMDB credits store | **1** (Phase 1, built) |
| "epic space opera **but more mature**", "**darker**" | tone modifier | **already in genome vocab** (dark, gritty, epic, bleak…) | **2 · vibe/affect** |
| "**make me cry**", "feel-good", "something scary", "cozy" | mood/affect | **already in genome vocab** (12 sad, 22 feel-good, 11 tense…) + subject proxies (loss 235, cancer 47) | **2 · vibe/affect** |
| "a thing for **ferraris**", "car chases", "lightsaber" | object / under-tagged action | plot recovers recall but 62–95% incidental after gating | **3** (demoted, optional) |
| "movies **like** Blade Runner but ___" | similarity + modifier | existing title-anchor + tone | existing anchor path |
| "**French heist films from the 70s**" | composition | intersect facets + year | **1** (multi-facet AND) |

### What the measurement overturned (this is the evidence — do not relitigate without re-measuring)
1. **The crowd keyword IS the prominence gate — for themes, actions, AND objects.** TMDB taggers apply a
   keyword only when the concept is a *feature* of the film. Verified: the 93 `food`-keyworded films are
   genuinely food-central (Big Night, Chef, Ratatouille) while 653 plot-`food` mentions are 85% incidental
   (people eat); the 80 `heist`-keyworded films are genuinely heist-central (Baby Driver, Die Hard, Rififi).
2. **So an LLM-over-plot pass is not additive.** Judge sweep (10 props, 172 snippets): **23% central / 62%
   incidental / 15% name-collision** (a *character* named Ferrari — Casablanca, Umberto D. — which no
   regex can catch). Judge sweep (11 actions, ~210 snippets): the keyword pool is **≥** the LLM-gated
   central plot pool in nearly every case — heist kw 225 vs 167 gated-central; courtroom 114 vs 66;
   kidnapping 230 vs 82; shootout 5% central; wedding 20%. Even the poster child **`car chase`** (keyword
   genuinely ≈0, plot 69) is only **15% central** → ~10 films, the rest single incidental chases (Bad Boys,
   48 Hrs). The "props live in the plot ~13×" recall gain is **real but ~80% incidental noise**, and after
   the mandatory gate the residual ≤ what the keyword already gives. Net: **plot extraction earns its cost
   only for distinctive coined objects** (lightsaber 92% central, delorean 75%) — a tiny, franchise-bound,
   already-discoverable set. Demote it.
3. **Affect is already in the model.** The 1,128 genome tags (and 306 user tags) carry a full affect/tone
   vocabulary — sad/cry (12: emotional, heartbreaking, poignant, tragedy, "sad but good"), mature/dark
   (13), feel-good (22), tense/scary (11), epic (8), mind-bending (12). These feed the item content tower
   and the LLM-132 vector is genome-derived, so **the embedding already encodes mood.** "Make me cry" /
   "more mature" is therefore a **routing** problem (phrase → genome tags → the existing anchor retrieval),
   **not a new-data problem.** The facet *filter* is the wrong tool for affect; the *anchor/embedding* path
   is the right one.
4. **Naive substring matching is unsafe** (drives the location resolver's cost, not vocab size): the
   `world war i` probe is **85% WWII/WWIII bleed-through** (`world war ii` contains `world war i`);
   `paris`→`parish`, `trench`→`mariana trench`. Resolution must be exact-tag membership + an explicit map.
5. **Country ≠ setting.** Paris-*keyworded* films carry `production_countries` US:7/FR:6/GB:5 — a Paris-set
   film is often a US/UK *production*. Two independent axes: `production_countries` = nationality of
   production; keyword/location = where it's set. Keep them separate.

### The three engines (replaces the old M1/M2/M3 phases)

**Engine 1 — Membership facets (hard filter + candidate map).** "What is *in* / *of* the movie": person,
country, language, format/attribute, clean theme/subgenre, place. **All from data already on disk** (TMDB
keywords + `production_countries` + credits) — **no LLM extraction.** This is where nearly all the user
value is, and it generalizes the people store directly (`_passes_constraints` + `facets_to_anchors`).

**Engine 2 — Vibe/affect path (soft anchor over the existing embedding).** "How the movie *feels*": mood
(sad, feel-good, scary, cozy, mind-bending) and tone modifiers (mature, darker, epic, lighthearted). A
small curated **mood-phrase → genome-tag-set** table routes into the **existing** `anchors_for` /
`probe_genome_tag` machinery — the affect signal is already baked into `serving/`. No scrape, no new
facet tables. Optional subject-proxy union for reach (`make me cry` ∪ {loss of loved one, terminal
illness, grief}). This engine is *new capability the current plan omitted entirely*, and it's cheap.

**Engine 3 — (Demoted / optional) LLM plot extraction.** Measured low-value. Keep only as a narrow,
deferred backfill: (a) distinctive iconic objects where keyword recall is truly 0 but the object *is* the
film (lightsaber, delorean, batmobile) — a short curated list, not a corpus sweep; (b) the ~227 movies
with no keywords at all; (c) "set in" vs "filmed in" disambiguation for the top ~50 locations if it ever
proves worth it. **Do not run an undirected plot sweep** — it costs ~10 LLM judgments per usable film and
returns an 80%-incidental pool the keywords already covered.

### Engine 1 detail — membership facets

**Sub-tiers by resolver cost (the tractability stratifies — do not treat as one vocab problem):**
- **1a · Structured, trivial (ship first):** *country/language* from `production_countries` /
  `original_language` (ISO-clean; a ~26-code region table gives "Scandinavian/European/East-Asian"), and
  *format/attribute* — single clean top-distribution keywords (`black and white` 364, `woman director`
  560 = "directed by women", `based on novel or book` 1149, `based on true story` 642). **Zero ontology.**
- **1b · Clean theme/subgenre keywords:** a curated top-N keyword vocab (the head is small — 338 keywords
  in ≥50 films, 117 in ≥100; 44% of the 17,820 are singletons) + a light spelling-alias map
  (`car chase`/`car chases`, `post-apocalyptic`/`…future`). Membership filter + candidate anchors.
- **1c · Location (curated rollup, not a synonym map):** TMDB already normalizes to `"city, region"` — only
  **217** comma-pattern place tags, **38** in ≥10 films. The work is an explicit **child→parent hierarchy**
  over *exact* tags (borough→city rollup: NYC +24%, LA +26% recall; Paris/London +4%) — **never** substring
  (see overturned #4). City→country only where a nationality query needs it.
- **Excluded from Engine 1 (goes to Engine 3 / year):** era/war **semantic clusters** (WWII, Cold War) have
  *no canonical head keyword* (36 scattered strings, contamination) — a hand-built keyword union is the trap
  the original plan warned about; let an LLM closed-ontology tag own "is this a WWII film," and let the
  native year feature own numeric decades.

**Resolution — `resolve_facet(phrase, kind) → facet value(s)`** (the analogue of `resolve_person`, but
alias-mapped + hierarchical for 1c; exact-membership for 1a/1b). Normalize → alias map → canonical
value(s), expanding the location hierarchy. Miss → drop + report (like unresolved titles/people). This is
where the Engine-1 design effort concentrates; the underlying data is free.

**Build-time** (`build_facet_store.py`, extended): `movieId_to_keywords` (raw + curated-vocab),
`movieId_to_countries`, `movieId_to_languages` from the scrape; `facet_vocab` (curated values) +
`facet_alias` (synonym→canonical) + `location_hierarchy` (child→parent, exact tags) + `country_region`.
**Schema/prompt:** HARD slots `require_country`, `require_language`, `require_location`, `require_keywords`,
format/attribute flags — closed enums where the vocab is small (country/language/format), harness-resolved
free strings for location/keyword. **Filter** (`_passes_constraints`): membership, ALL/ANY consistent with
`require_genres`/`require_people`. **Seed** (`facets_to_anchors`): facet → top-relevance member films →
soft anchors, subordinated to named titles exactly like `anchors_for`.

### Engine 2 detail — vibe/affect path
**Schema/prompt:** a soft `mood`/`vibe` slot (free strings). **Resolution:** `resolve_mood(phrase) →
[genome_tag…]` via a small curated table (~15 moods: cry→{emotional, heartbreaking, poignant, tragedy,
sad but good}; feel-good→{feel-good, heartwarming, uplifting, happy ending}; mature→{dark, gritty, bleak,
adult themes}; scary→{tense, creepy, atmospheric, claustrophobic}; mind-bending→{mindfuck, cerebral, twist
ending, nonlinear}; …). **Route:** genome tags → existing `anchors_for`/`probe_genome_tag` → soft seeds
(NOT a hard filter). Tone modifiers compose onto a membership base ("epic space opera **but mature**" =
`require_keywords=[space opera]` + mood-anchor{dark, epic}). Optional reach: union the mood's subject-proxy
keywords. **No new data — validate the routing, not a store.**

### Composition (the actually-hard inference problem)
Real prompts stack constraints: "French heist films from the 70s", "feel-good movie about food", "epic
space opera but more mature". The engines compose — Engine-1 hard filters AND together, Engine-2 mood
becomes soft anchors *within* the filtered pool, year bounds the set — but the LLM must **split membership
(hard) from vibe (soft)** reliably, and thin intersections must degrade gracefully (fall back to the
loosest hard constraint + anchors, never empty). Budget a judged validation pass on composition; it's the
real risk now that extraction is off the table.

### Back-bookend readiness — preserve per-rec provenance (BUILD NOTHING NOW)
This whole front-end is the **"front bookend"** of the industry Bookend-LLM pattern (LLM parses NL →
structured model input · classical recommender ranks · LLM explains results). The **back bookend**
(LLM-generated, grounded explanations) is **deferred by company decision — do not build it.** But it's a
bolt-on later *only if we don't design out its inputs now*. The facet store is exactly the metadata such
an explanation layer would ground on, so one cheap discipline:
- `recommend()` already returns solid **query-level** provenance (`extraction`, `resolution`,
  `people_resolution`, `anchors`, `anchor_weight`, `fallback`, `filtered`). Keep that.
- The gap is **per-rec attribution**: each rec is `(title, genres, year, score)` — it does *not* record
  *why THIS film* (which anchor it sits nearest, which mood/genome-tag or facet it matched, which
  constraint it passed). As Engines 1–3 extend `_passes_constraints` (today returns a bool) and add
  `facets_to_anchors`, **keep those paths able to attribute** — i.e. don't collapse "which facet/anchor
  matched" to a bare pass/fail or a flat anchor list. Structure to record it; leave it unsurfaced.
- The other back-bookend need — an "is it really in the catalog" guard before showing any generated text
  — we **already have** in `_passes_constraints` + the corpus-bounded ranking.

Net: add no provenance plumbing now; just don't throw the signal away when building the engines.

### Phasing
- **F1 — Engine 1a/1b (structured + clean theme keywords).** Country/language/format/attribute + curated
  theme vocab → `resolve_facet` + filter + candidate anchors + export bake. Ships "French/Korean films",
  "black and white", "directed by women", "based on a book", "heist/space-opera/martial-arts/food films".
  Most of the value, no LLM, no ontology. Measure per-facet coverage + correctness.
- **F2 — Engine 1c (location) + Engine 2 (vibe/affect).** Location hierarchy rollup; mood/tone routing to
  genome anchors. Ships "movies in Tokyo/NYC/Paris", "make me cry", "something to make me feel good",
  "darker/more mature". Both reuse existing machinery.
- **F3 — Composition validation + (optional) Engine 3 backfill.** Judged pass on multi-constraint asks;
  only then, if a gap remains, the narrow curated-object / no-keyword-movie plot backfill.

### Validation & caveats (this expansion)
- **Membership (Engine 1):** deterministic **facet-correctness** (% of top-N truly carrying the facet) +
  per-value **coverage/sparsity audit** (flag thin facets so the UI never promises an empty pool). Verified
  the crowd keyword is high-precision (food/heist spot-checks) — trust it as the gate.
- **Vibe (Engine 2):** subjective — judge with the sonnet harness ("does this feel *sad / mature / cozy*?"),
  not a membership check. The metric is anchor *relevance*, not filter correctness.
- **Keyword recall gaps:** crowd-sourced — a Paris-set film may lack the `paris` tag (recall ≤ coverage). A
  hard filter *excludes on a missing tag*; for thin facets prefer a soft boost or union with the structured
  signal, and surface counts.
- **Substring is unsafe** (overturned #4): resolvers use exact-tag membership + explicit alias/hierarchy,
  never `if probe in keyword`.
- **Scope discipline:** the win is *wiring existing data* (keywords + countries + credits + genome), not
  new extraction. Engine 3 is the boil-the-ocean trap the measurement closed — keep it demoted.

## Validation approach (mirror the v1→v5 harness)

Pipeline already exists and is documented in `docs/llm_frontend/validation/`: extract (Haiku, forced
schema) → `recommend()` over real `serving/` → per-case sonnet judge. For facets, add:
- A **deterministic facet-correctness check** (no agents): for a `require_people` request, what % of
  the top-15 actually contain that person? (Analogous to the constraint-violation check in
  `docs/llm_frontend/validation/v5/ablate_postfilter.py`.) This is the primary metric — for a hard
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
  `docs/llm_frontend/validation/llm_frontend_haiku_validation.md`.
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
