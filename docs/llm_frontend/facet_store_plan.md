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
  **[Correction, 2026-07-01 — the "no clean data" premise here is OVERTURNED.** The Ask-AI holes
  measurement found US certification (`details_raw.release_dates`, 98%) and `tmdb.runtime` (100%) are
  already in the scrape → both become cheap Engine-1 structured facets. See "Ask-AI holes — measured
  results" at the end of Expansion II.]
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

> ### ▶ RESUME HERE — non-API build campaign COMPLETE; next = API-wiring / F2 (2026-07-02)
> **The non-API-key / non-Streamlit build campaign (steps A→C) is DONE and COMMITTED.** Everything on the
> retrieval/filter/data/resolver side that is buildable + deterministically verifiable without an API key has
> shipped; what remains needs either the hosted extractor (schema/prompt slots feeding these engines) or the
> Streamlit Ask tab. A deterministic eval harness anchors it: `tools/llm_frontend_eval.py` over
> `docs/llm_frontend/validation/retrieval_eval/eval_cases.json` (subagent-authored cases → recommend() →
> machine-checkable assertions; no API), now at **REGRESSION 69/69, SPEC 6/9** (snapshot `report_facets_stepC.txt`).
> **Win #1 (committed):** soft genome re-rank `GENOME_RERANK_LAMBDA=0.5`. Full detail + the ranked backlog are in
> **"Non-API build campaign"** below.
>
> **Win #2 — step A · F1 structured facets — BUILT + verified + adversarially reviewed (uncommitted, 2026-07-01).**
> content-rating (US MPAA cert), runtime, franchise/collection (require/exclude incl. a curated MCU/DCEU
> universe-alias table), composer, and a `min_vote_average` quality floor. All from the on-disk scrape, no LLM,
> no model change. `_passes_constraints` now reads the baked facet tables (absent metadata passes, mirroring the
> year gate). The eval gained faithful facet-membership assertion types (`max_runtime`/`max_rating`/
> `excludes_franchise`) + an independent `oracle` ground-truth assertion — before, its genre proxies didn't gate
> on the facet. **A 4-lens adversarial review (9 confirmed findings) then hardened it**, all fixed:
>   • **composer resolution is role-aware** — composers live in a SEPARATE `composer_name_to_ids` namespace so
>     "movies with John Williams" (actor) never shadows to the prolific same-named composer; `require_composers`
>     resolves composer-only, `require_people` is billed-preferred with a composer fallback.
>   • **franchise matching is contiguous-token, not raw substring** — kills 'Saw'→Texas Chainsaw, 'The Ring'→LotR,
>     'It'→65 collections. Universe aliases tightened ('spider man'→'spider man mcu' spares Sony Spider-Man;
>     dropped dead 'incredible hulk'/'green lantern'/'justice league'); documented as a broad family heuristic
>     (standalone films TMDB gives no collection can't be caught by name — precise collection-id/movieId roster
>     deferred to F3).
>   • **runtime/vote gates coerce stringified LLM bounds** (`max_runtime:"90"`) instead of crashing.
>   • **eval no longer passes vacuously** on an empty/all-unknown pool; built facets carry `built:true` so they
>     move into the must-stay-green REGRESSION bucket (were hiding in SPEC, skipped by `--only-regression`).
> **Final: facet-correctness 100%; the 11 built-facet cases (rating/runtime/franchise/composer + oracle +
> adversarial low-ceiling-rating) are guarded REGRESSIONs, all green; 0 new regressions** (the 10 REGRESSION
> failures are pre-existing genre-coherence cases with no facet assertions — step-B). 2 `max_runtime` cases stay
> SPEC on genre-count (runtime filter is correct; `liked_genres`-only retrieval under-supplies the genre → step B).
> Snapshot: `retrieval_eval/report_facets_f1.txt` (34/44 reg, 19/28 spec). Files: `llm_features/build_facet_store.py`,
> `src/llm_frontend.py`, `tools/llm_frontend_eval.py` + augmented cases. Export bakes the whole facet_store dict
> already, so **no `src/export.py` change** — canonical `python main.py export <PROD α=0.5 ckpt>` reproduces the
> `facets` key (this session re-baked it surgically into `serving/` for the eval).
>
> **Win #3 — step B · retrieval quality — BUILT + verified + adversarially reviewed (uncommitted, 2026-07-01).**
> Three retrieval levers in `src/llm_frontend.py` (all reading the ALREADY-baked genome data — **no serving/export/
> streamlit change**, so no re-bake is needed for step B):
>   • **`resolve_mood(phrase)` (Engine-2 vibe/affect)** — a curated `MOOD_TABLE`/`MOOD_ALIASES` (~14 moods; every
>     output tag verified in the serving genome vocab; OOV synonyms like "cozy"/"uplifting"/"mature" mapped onto
>     in-vocab tags) routes a free `mood`/`vibe` slot → genome tags, feeding the Mode-2 anchors + a **separate**
>     soft re-rank term (`MOOD_RERANK_LAMBDA`, its own knob). Kept apart from the subject `genome_tags` on purpose:
>     folding both into one mean re-rank let the tone axis out-rank the subject ("feel-good movie about cooking"
>     surfaced feel-good non-food films); two additive terms fixes it, and a mood only drives the anchors when it
>     is the SOLE vibe.
>   • **`require_genome_tags` HARD floor** (`GENOME_HARD_FLOOR=0.35`) — gates the pool to films that clear the floor
>     on **every** required tag (AND, mirroring `require_genres`/`require_people` — NOT the average, so "set in Paris
>     during WWII" drops Schindler's List for ~0 Paris relevance). OOV/absent → no floor (graceful, like the year gate).
>   • **Mode-1.5 title-genome injection** — a PURE-TITLE request (liked title, no genome_tags AND no mood) injects the
>     title's own most-relevant genome tags as SOFT re-rank tags (acclaim/production-meta tags stoplisted, so
>     "masterpiece"/"imdb top 250"/"remake" don't re-summon the era-neighbours), killing the co-watch drift (a 1994
>     Pulp Fiction seed no longer pulls Forrest Gump / Toy Story).
> **A 4-lens adversarial review (5 confirmed findings) then hardened it**, all fixed: the floor now ANDs per-tag
> (was a mean → a WWII film could clear a Paris require); two vacuous `rank_above` assertions dropped (min_genome
> carries the floor) and the harness `rank_above` change reverted; a vacuous Mode-1.5 case (Seventh Seal — no base
> drift to suppress) replaced with a **differential-verified** Amélie case (fails with the lever off); the thriller
> Drama cap (tuned-to-observed) dropped with `mood="thriller"` documented as load-bearing genre-as-vibe routing; 6
> inert stoplist entries removed.
> **Final: REGRESSION 52/62 green** (the original 34 stay green; +18 newly-built mood/genome/max_runtime/Mode-1.5
> cases all green — incl. a new multi-tag AND case; the 10 remaining are pre-existing genre-coherence cases → step C).
> **SPEC 8/13** (remaining fails need `exclude_mood`/`require_country`, or are Christmas's genre-purity/Gremlins beyond
> the genome floor — none are step B). Snapshot: `retrieval_eval/report_facets_stepB.txt`. Files: `src/llm_frontend.py`
> + `retrieval_eval/eval_cases.json` (the eval harness `tools/llm_frontend_eval.py` is **net-unchanged** — a `rank_above`
> tweak was added then reverted in review; `src/export.py`/`streamlit_app.py`/`serving/` untouched).
>
> **Win #4 — step C · resolvers/routing + genre-pool degradation — DONE + adversarially reviewed + committed (2026-07-02).**
> Two threads. (1) **Engine-1a membership facets:** `resolve_facet(phrase,kind)` → country (`production_countries`,
> ISO + region rollups Scandinavian/East-Asian/European), language (`original_language`, ISO; Chinese→[zh,cn]), and
> format/attribute (8 curated top-distribution keywords: black-and-white / woman-director / based-on-a-book / …).
> `build_facet_store` emits `movieId_to_countries` (99.2%) / `_language` (99.4%) / `_attributes` (26.3%); the whole dict
> re-bakes into `serving/feature_store.pt` at export. Country/language gate **per-film-absent → DROP** (explicit
> membership demand) but **whole-table-missing → SKIP** (graceful, when `facets=None`). (2) **Genre-pool degradation,
> 3 orthogonal levers** (`src/llm_frontend.py`, attribution-clean via a turn-off sweep): L1 genre-affinity re-rank
> (under-supplied `liked_genres`), a co-genre diversity cap (Drama swamping War/Romance pools), and signature-MMR on a
> near-empty OR-fan (sci-fi-noir). **Key correction (user-confirmed, review-driven): INTERSECTION-FIRST** — a dual-genre
> "X and Y" ask means the intersection ("both in the same film" / "a horror comedy" → Shaun of the Dead), so a healthy
> X∩Y pool stays strict AND (10/10 real both-genre films) and only a near-empty intersection (Sci-Fi∩Film-Noir = 2)
> OR-fans; the eval's dual-genre assertions were rewritten from the mix-mandating `max_genre<=7` to a faithful
> `both_genres` (intersection) metric. See backlog **C** below.
>
> **CONTINUE AT → API-wiring phase (needs a key) + F2 (Engine-1c location / Engine-2 vibe over the existing anchors):**
> the retrieval/filter side is now built and green; the next non-trivial work is (a) wiring the hosted Haiku extractor
> to EMIT these slots (`require_country`/`require_language`/`require_attributes`/`mood`, plus the intersection-vs-mix
> signal so "both in the same film" resolves to strict AND from a real extraction cue, not a heuristic), (b) the
> Streamlit Ask tab, and (c) F2's location hierarchy + mood-anchor routing. See **Phasing (F2/F3)** and the backlog.
>
> ---
>
> ### ▶ Ask-AI holes measurement — DONE (2026-07-01)
> The "Ask-AI holes" run is **complete**: 500 realistic prompts (50 archetypes × 10) were oracle-recommended
> and coverage-tagged on **Sonnet** (batch 00 = 100 reused Opus records; batches 01–04 = 400 fresh Sonnet,
> run in cost-controlled sets of 10). Results are folded into the new subsection
> **"Ask-AI holes — measured results"** at the end of this Expansion II. **Headline:** the plan's biggest
> assumed data gaps (content-rating, runtime, franchise) are **not** data gaps — they're already in the
> scrape at 98–100% coverage and become cheap Engine-1 structured facets. See that subsection for the full
> axis table, the reversal, and the priority-ranked action list.
>
> **Durable artifacts** (`docs/llm_frontend/validation/ask_ai_holes/`): `harvest_500_prompts.json` (prompts),
> `rec_batch_00–04.json` (the 500 tagged records), `rec_aggregate.json` (axis tally + holes), `corpus_helper.py`
> (grounding CLI). Analysis scripts in the session scratchpad (`aggregate_holes.py`).
>
> **Open question (unchanged):** whether to **commit this plan revision** — still the user's call.

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

### Ask-AI holes — measured results (500 prompts × 50 archetypes, Sonnet oracle-rec, 2026-07-01)

**Method.** 500 realistic "Ask AI" prompts (50 user archetypes × 10) were oracle-recommended by a
film-curator agent that tagged every signal it used with **coverage** (HAVE/PARTIAL/MISSING vs our real
`serving/` data) and **layer** (intake = must be extracted from the prompt / postfilter = must be verified
on candidates). 2,463 signals total. Records `ask_ai_holes/rec_batch_00–04.json`, aggregate
`rec_aggregate.json`, script `scratchpad/aggregate_holes.py`. (96% of prompts carry ≥1 non-HAVE signal, but
that headline is noise — almost every rich prompt grazes one lossy axis; the **structural** holes below are
what matter.)

**Axis coverage — prompt-level demand vs gap (of 500):**

| axis | use | gap | gap% | dominant layer | verdict |
|---|---|---|---|---|---|
| genre | 378 | 7 | 2% | intake | ✅ workhorse |
| mood_affect | 351 | 82 | 23% | both | ✅ Engine 2 (genome) |
| theme_subject | 181 | 117 | 65% | intake | ⚠️ Engine 1b keyword (recall) |
| similar_to_title | 160 | 31 | 19% | intake | ✅ existing anchor path |
| tone_maturity | 126 | 117 | 93% | both | ⚠️ Engine 2 (genome, lossy) |
| popularity_obscurity | 125 | 23 | 18% | postfilter | ✅ ranking |
| nationality + language | 111 | 15 | 14% | postfilter | ✅ Engine 1a (planned) |
| era_period + decade | 104 | 12 | 12% | intake | ✅ year native |
| subgenre | 101 | 82 | 81% | intake | ⚠️ Engine 1b keyword |
| pacing | 94 | 91 | 97% | both | ❗ weak genome proxy only |
| person_director + actor | 86 | 7 | 8% | intake | ✅ people store (built) |
| occasion_audience | 81 | 81 | 100% | intake | ❗ NEW intent class |
| character_type | 64 | 59 | 92% | intake | ⚠️ Engine 1b/2 partial |
| visual_style | 58 | 53 | 91% | intake | ❗ director-anchor + honest limit |
| setting_location | 57 | 51 | 89% | intake | ⚠️ Engine 1c (planned) |
| awards_acclaim | 55 | 53 | 96% | postfilter | 🟢 vote_average on disk + awards(data) |
| ending_type | 47 | 45 | 96% | postfilter | ⚠️ Engine 2 + negation postfilter |
| content_rating | 40 | 40 | 100% | postfilter | 🟢 release_dates **on disk** |
| runtime | 34 | 34 | 100% | postfilter | 🟢 tmdb.runtime **on disk** |
| franchise_universe | 29 | 29 | 100% | intake | 🟢 belongs_to_collection **on disk** |
| streaming_availability | 11 | 11 | 100% | — | ⛔ out of scope (dynamic) |
| person_composer | 8 | 8 | 100% | intake | 🟢 credits crew **on disk** |
| representation_diversity | 8 | 8 | 100% | intake | 🟡 dir-gender(planned)+lgbt kw / limit |
| cross_media | 7 | 7 | 100% | intake | 🟡 source-kw cheap / seed-map hard |

**⚑ The reversal (most important finding).** The plan's stance *"keep content-rating/runtime unsupported —
no clean data"* is **wrong on the data.** Verifying the scrape (`llm_features/cache/scraped/*.json`), the top
pure-MISSING, highest-demand axes are **already on disk at 98–100% coverage** — wiring, not extraction:
- **runtime** `tmdb.runtime` — **100%**.
- **content rating** `details_raw.release_dates` (US certification) — **98%**.
- **franchise** `details_raw.belongs_to_collection` (id+name) — present for **all 2,062 franchise-entry films**
  (22% of corpus; the rest are genuinely standalone → a clean membership signal, not a gap).
- **composer** `credits_raw.crew[job='Original Music Composer'].id` — **89%**.
- **budget/revenue** `details_raw.budget/revenue` — 74% (indie-vs-blockbuster scale).
- **vote_average / vote_count** — **100%, and already baked into `serving/`** (currently unused for ranking).

So five of the highest-demand "holes" collapse into **Engine 1, cheap** — the same membership / numeric-filter
machinery, no LLM, no new scrape.

**A · Engine 1 — new structured facets (CHEAP; data on disk; pull into F1).** Reuse `_passes_constraints` +
candidate maps, baked into `serving/` at export:
1. **content_rating** — parse US cert from `release_dates`; postfilter `require_rating`/`exclude_rating≥R` +
   a "kid-safe" (G/PG) flag. 40/500, 37 postfilter — the single biggest deferred data hole, and it's free.
   Serves the whole content-sensitivity / family / "clean" / "no sex-or-gore" cluster.
2. **runtime** — wire `tmdb.runtime`; numeric filter like year (`max_runtime`/`min_runtime`). 34/500, 31
   postfilter. Serves runtime-constrained (in-flight, bedtime, "under 90", "no filler").
3. **franchise/collection** — `belongs_to_collection.id` membership → "the X films", **franchise EXCLUSION**
   ("no sequels", "skip the MCU"), watch-order. 29/500. Generalizes the people-store id-membership directly.
4. **composer** — extend the people store with the composer crew id. "Scored by Hans Zimmer" + soundtrack
   pulls. 8/500 but trivial.
5. **vote_average quality-floor** — already in `serving/`; expose as a postfilter/ranking gate for "actually
   good / not trash / critically loved". Covers most of awards_acclaim (55/500) cheaply; true awards
   (Oscar/Palme) stay a data-build → defer.
6. **budget scale** (optional) — indie/blockbuster attribute.

**B · New INTAKE slots the extractor isn't taught (schema/prompt; F1–F2).**
- **occasion / audience** (81/500, 100% gap, intake-dominant — the largest omitted intent class). Add an
  `occasion` slot (date-night, kids, party, background, solo-unwind); resolve via a routing table to a
  **composition** of existing signals (kids → animation/family genre + rating≤PG + short runtime; party →
  popular + comedy/upbeat mood; date-night → romance/dramedy + mid-runtime). A routing problem, like affect —
  not a data axis.
- **exclusion generalization** — `exclude_*` beyond genres/people: mood ("nothing sad"), ending ("no downer"),
  content ("no sex/gore" → rating), franchise ("no sequels"). The two-tower can't hard-subtract, so **exclusion
  is postfilter-only**; where the excluded axis is itself MISSING, say so rather than silently failing.
- **quality_floor** (soft) — "actually good" → vote_average gate.
- **surprise / no-preference** — cold-start "just pick something": route to popularity **+ diversity (MMR) or a
  seeded shuffle.** The holes flag that "surprise" currently collapses to the popularity prior with no
  serendipity or diversity objective (no MMR to guarantee the shortlist even differs).

**C · Route-to-Engine-2 (genome/anchor; PARTIAL, soft-only, honest limits).** pacing (91 gap),
plot_structure (59), ending_type (45), character_type (59), tone_maturity (117), some visual_style — genome
carries partial proxies (slow/fast-paced, twist-ending/nonlinear/mindfuck, happy-ending,
anti-hero/strong-female-lead, dark/gritty). Soft anchors, never hard filters; surface that they're
approximate. **pacing** is the weakest (mostly MISSING even in genome) — set expectations, don't promise it.

**D · Genuinely unservable / out-of-scope (state honestly; build nothing).** streaming availability
(dynamic/licensing); recency past ~2023 (corpus boundary — no Barbie/Oppenheimer/2024+); cross-media *seed*
from a TV show/game (title→movie-space mapping — though "based on a video game" *source* is a cheap keyword);
"so-bad-it's-good"/camp (vote_average conflates boring-bad with fun-bad); cast ethnicity / fine
representation; per-scene content grading ("one scary scene for an 8-yo"); background/second-screen
"plot-light" axis; set-in vs filmed-in. The ≥200-rating corpus floor also caps true obscurity (deep cuts fall
below it — "hidden gem" is only *relative*).

**E · Composition & thin pools (confirms the plan's hardest problem).** The compositional archetypes stack 4–6
constraints (3–5 non-HAVE each); thin intersections recur (chess <5 films, Italian giallo absent, Marrakech
sparse). Rule: **when degrading a thin/empty intersection, drop the un-computable or MISSING hard constraint
first** (you can't enforce what you don't have) — keep the highest-confidence membership facet + soft anchors,
never fall silently to empty/popularity.

**Priority (demand × cheapness):**
1. **content_rating + runtime + vote_average-floor** — highest-demand pure holes, all free, all postfilter. Do first.
2. **franchise/collection** (incl. exclusion) — free, high-salience, reuses the people-store id-membership.
3. **occasion intake + routing** — largest intent class; no data, pure schema/routing.
4. **composer** — trivial people-store extension.
5. Engine-2 routing (pacing/ending/plot/character/tone) — soft, honest limits.
6. Everything in **D** — document as out-of-scope; don't build.

**Phasing update:** pull the cheap structured facets (content_rating, runtime, franchise, composer,
vote_average-floor) into **F1** beside country/language/format — same "wire existing data" move, and they carry
*more* demand than some originally-planned keyword facets. Occasion-routing + exclusion-generalization join
**F2** with vibe/affect. Nothing here disturbs the Engine-3 demotion — it stays demoted.

### Cross-check deltas (6-lens blind adversarial synthesis, 2026-07-01)

A 6-lens Sonnet pass (missing-data / partial-lossy / intake-gaps / postfilter-gaps / unhandled-classes /
composition-thin) re-derived holes from the same evidence **blind to the writeup above** (`synthesis_6lens.json`).
It **independently reached the reversal** — content-rating via `release_dates`, unwired `runtime`, franchise via
`belongs_to_collection`, composer via `job=Original Music Composer` all flagged CHEAP without being told they're
on disk — and confirmed occasion (largest new intake class), exclusion-is-postfilter-only, pacing-weakest, and
the composition degradation rule. Beyond confirmation, it sharpened the plan on these points:

1. **Constraint-priority 3-tier (the sharpest add — operationalizes composition degradation).** Intake should
   tag every extracted slot **required** (has data, dealbreaker) / **preferred** (has data, soft) /
   **cannot_enforce** (no data yet: pacing, ending, rating-until-built). Postfilter applies required as hard
   filters, preferred as boosts, cannot_enforce as LLM-judgment-only; when a pool goes thin it relaxes
   cannot_enforce first, then preferred by *lowest data-coverage first*, **never dropping genre/people last.**
   This turns "drop the un-computable hard constraint first" into a concrete schema mechanism.
2. **Pacing is cheaper than framed — a derived genome scalar, not just an honest limit.** *Verified:* the genome
   vocab has `slow paced`, `fast paced`, `atmospheric`, `slow`. Derive a slow−fast scalar at build, wire as a
   soft rank signal + a `pacing` intake slot. The 97% gap is because it's **unwired**, not absent → Engine-2
   route, cheap. (Still soft-only, never a hard filter.)
3. **Negative-affect exclusion GATE (mechanism for Engine 2).** genome carries both poles; add a postfilter
   hard-exclude when `genome[sad|tense|dark] > threshold` AND an `exclude_mood` slot is set, with intensity
   banding (low/med/high) for "a *little* sad". Config-knob threshold, tuned on the comfort-watch canary. This
   is the concrete form of "exclusion is postfilter-only."
4. **New intake slots the schema still lacks:** `exclude_titles` (named-title negation — "not X or Y", "skip the
   ones I named", "no sequels I've seen" → remove by movieId); `liked_shows` / cross-media (decompose a TV/game
   reference into genome+keyword anchors via the **existing** mood→synthesized-anchor path — do *not* attempt a
   corpus title match; this **downgrades** the difficulty the writeup assigned cross-media); `serendipity_mode`
   (zero-anchor "surprise me" — all three engines need ≥1 non-null slot, and empty extraction currently risks
   *fabricated* anchors, so build an explicit popularity-stratified diversity draw + optional one follow-up Q).
5. **subgenre + theme_subject are primarily INTAKE-EMISSION failures, fixable now** (73/82 and 97/117 gaps at
   intake). Teach the extractor to canonicalize subgenre→TMDB-keyword clusters and emit ≥3-keyword OR-fans
   (`space opera`/`heist`/`found footage` already exist as keywords) — a prompt/schema change, not new data.
6. **Location: drop `production_countries` as a setting proxy.** City-level "set in X" → `require_keywords=[city]`,
   never `require_country`; country-level "French films" → `require_country`. Sharpens Engine 1c's intake routing.
7. **Awards has a cheap middle tier I'd dismissed:** a free **Academy Awards CSV** (Kaggle/IMDB) joined on
   title+year → `oscar_wins`/`nominations`/`best_picture` resolves the top awards cases; vote_average stays the
   base, Metacritic/RT (OMDb) is the optional expensive tier.
8. **Ending-type: an optional, *narrowly-scoped* dedicated classification** — a bounded one-time Haiku pass over
   the already-scraped plot `overview` → `{happy|bittersweet|sad|ambiguous, has_twist}` (~$0.25–3 for 9k). This
   is a justified exception to the Engine-3 demotion **only because it's closed-label classification, not an
   undirected prop sweep**; keyword-bootstrap (`twist ending`/`happy ending`) covers ~20% first. Offer, don't assume.

**Not adopted (noted for the record):** OMDb for content-rating (our verified `release_dates` on disk beats it);
a full visual-style LLM pass (stays Engine-3-demoted — only B&W via `color_info` is cheap); corpus refresh /
sub-200-rating canon-extension list (a larger project, out of scope for this plan).

### Non-API build campaign — deterministic eval + genome re-rank (2026-07-01)

**Goal:** build + measure every retrieval / filter / data / resolver feature that needs **no API key** (hosted
Haiku extraction) and **no Streamlit tab** — verified by feeding hand-authored extraction JSON straight into
`recommend()`. The API is only the last-mile free-text→JSON step; everything else is buildable + measurable now.

**Ruler (built):** `tools/llm_frontend_eval.py` — deterministic no-API regression+spec eval (the residual-listed
portfolio artifact). Reads `docs/llm_frontend/validation/retrieval_eval/eval_cases.json` — **70 cases** authored by a
10-agent Sonnet sweep (one per failure mode), each = utterance → extraction JSON → machine-checkable assertions
(`contains_genre` / `max_genre` / `excludes_genre` / `*_title_substr` / `all|none_have_person` / `rank_above` /
`min_genome`). Split **REGRESSION** (`needs_feature=none`, must stay green) vs **SPEC** (expected-fail until the named
feature lands — the eval measures build progress). Snapshots: `retrieval_eval/baseline_report.txt` (λ=0),
`report_rerank_lambda0.5.txt`.

**Diagnosis (overturned residual #1's proposed fix).** The anchor failures are NOT "dilution fixed by boosting the
soft-genre signal" — measured, `liked_genres` barely pulls genre and can make results *worse*. Real mechanisms:
(1) item-embedding **era/co-watch drift** — correct western anchors still rank 80s-action neighbours first (→ 0
westerns); (2) **tag→arthouse skew** — `intimate`/`heartfelt` peak on Bergman/Tarkovsky.

**Win #1 (landed, uncommitted): soft genome re-rank ("Source A").** In `src/llm_frontend.py`: after cosine scoring,
`score += GENOME_RERANK_LAMBDA * mean_genome_relevance(genome_tags ∪ require_genome_tags)`. Additive (cosine is
signed), soft (never a hard filter — the pool never empties). Swept λ → **0.5** best: **+12 cases / 0 regressions**,
plateaus after (regression 18→23, spec 18→25). Fixes the western/Pulp-Fiction era-drift, unlocks setting facets
(courtroom/time-travel/Paris-class via `require_genome_tags`), helps mood routing. **Not** fixed: romance→arthouse
tag-skew (separate — abstract mood tags are inherently arthouse-heavy).

**Ranked non-API backlog / where to continue:**
- **A · F1 structured facets — ✅ DONE (uncommitted, 2026-07-01).** content-rating (US MPAA cert from
  `release_dates`, 82% non-empty US cert), runtime (`tmdb.runtime`, 99.4%), franchise (`belongs_to_collection`,
  22% = 2,062 films; require/exclude, incl. a curated `FRANCHISE_UNIVERSE_ALIASES` map so "skip the MCU/DCEU"
  resolves across per-series collections), composer (`crew[Original Music Composer]`, folded into the people
  store — e.g. Hans Zimmer pid 947 / 112 credits), and a `min_vote_average` quality floor (`tmdb.vote_average`,
  99.4%). Built: extended `llm_features/build_facet_store.py` to emit the tables; `src/export.py` already bakes
  the whole facet_store dict (**no export change**); extended `_passes_constraints` (`require_max_rating` /
  `max_runtime`/`min_runtime` / `require_franchise`/`exclude_franchise` / `min_vote_average`; composer =
  `require_composers`/`exclude_composers` → `require_people` extension) with absent-metadata-passes semantics;
  gave the eval faithful facet-membership assertion types. Verified via `tools/llm_frontend_eval.py`:
  **facet-correctness 100% (11/11, 0 violations); 9/11 SPEC green; 0 regressions.** Remaining 2 = max_runtime
  cases whose runtime filter is correct but genre retrieval lags (a step-B lever). Not built here: schema/prompt
  slots for the hosted extractor (deferred to the API-wiring phase — this is the non-API campaign) and
  country/language/studio facets (a sibling F1 resolver — step C `resolve_facet`).
- **B · Retrieval quality — ✅ DONE (uncommitted, 2026-07-01).** Three levers in `src/llm_frontend.py`, all over the
  already-baked genome data (**no serving/export/streamlit change**): (1) `resolve_mood(phrase)` — curated
  `MOOD_TABLE`/`MOOD_ALIASES` (~14 moods, all output tags in-vocab) → a `mood`/`vibe` slot feeds the anchors + a
  **separate** soft re-rank term (`MOOD_RERANK_LAMBDA`), kept apart from subject `genome_tags` so tone doesn't out-rank
  subject; (2) `require_genome_tags` **HARD floor** (`GENOME_HARD_FLOOR=0.35`) with **AND** semantics (each required tag
  clears the floor, mirroring `require_genres` — not the mean); (3) **Mode-1.5 title-genome injection** — a pure-title
  request (no genome_tags, no mood) injects the liked title's own discriminative genome tags (acclaim/production-meta
  stoplisted) as soft re-rank tags to kill co-watch/era drift. 4-lens adversarial review (5 confirmed findings) fixed.
  Verified via `tools/llm_frontend_eval.py`: **REGRESSION 52/62 green** (0 new regressions; +18 built mood/genome/
  max_runtime/Mode-1.5 cases incl. a multi-tag AND case); **SPEC 8/13**. Snapshot `retrieval_eval/report_facets_stepB.txt`.
  See the step-B "Win #3" block up in the RESUME-HERE callout for the full detail.
- **C · Resolvers/routing + genre-pool degradation — ✅ DONE + committed (2026-07-02).** `resolve_facet`
  (country/language/format — Engine-1a, from `production_countries`/`original_language` + a curated attribute-keyword
  vocab), `build_facet_store` emitting the three new tables (whole dict re-bakes into `serving/`), and the three
  genre-pool levers (L1 genre re-rank / co-genre cap / near-empty-pool signature-MMR). **INTERSECTION-FIRST** replaced
  the original "OR-fan/never-collapse" framing after the review showed the utterances demand the intersection ("both in
  the same film" / "a horror comedy"): a healthy X∩Y pool stays strict AND (real horror-comedies / action-comedies);
  only a near-empty intersection (Sci-Fi∩Film-Noir = 2) OR-fans. **REGRESSION 69/69, SPEC 6/9** (snapshot
  `report_facets_stepC.txt`). The eval's dual-genre assertions moved from mix-mandating `max_genre<=7` to a faithful
  `both_genres` (intersection) metric. Adversarial review (4 lenses → verify): 5 confirmed findings, all addressed
  (intersection-first; facet-absent-empties-pool guard; oracle set-compare). The remaining SPEC fails are NOT step C —
  `exclude_mood` (2 HOLE cases) + Christmas genre-purity (needs `require_genome_tags` tuning), i.e. F2/F3 territory.

**Still to do (needs an API key or Streamlit — out of the non-API campaign):**
- **API-wiring:** teach the hosted Haiku extractor to EMIT the new slots (`require_country`/`require_language`/
  `require_attributes`/`mood`) AND an explicit intersection-vs-mix signal, so "action comedy, both in the same film" →
  strict AND from a real extraction cue rather than the current pool-size heuristic. Budget a judged pass on this
  schema reopening (see "Caveats — Schema reopening").
- **F2 — Engine-1c location + Engine-2 vibe:** the `"city, region"` location hierarchy rollup and mood/tone anchor
  routing over the existing embedding. Both reuse machinery already built; measure with the same ruler.
- **F3 — composition validation + (optional) Engine-3 backfill:** a judged pass on multi-constraint asks; only then,
  if a gap remains, the narrow curated-object / no-keyword-movie plot backfill.

**Recommended next step:** wire the hosted extractor (API-wiring) — the retrieval/filter substrate is now built and
green, so the highest-leverage remaining work is making the free-text→JSON step actually populate these slots + the
Streamlit Ask tab, turning the whole non-API substrate into a usable product. F2 (location/vibe) is the natural
non-API follow-on if staying key-free.

**Committed (this campaign):** Phase 0/1 + step A + step B (commits `5fbadea`/`cbd0257`/`4ac377d`/`9b8beab`); step C
(2026-07-02) touches `src/llm_frontend.py`, `llm_features/build_facet_store.py`, `tools/llm_frontend_eval.py`,
`docs/llm_frontend/validation/retrieval_eval/` (cases + `report_facets_stepC.txt`), and `serving/feature_store.pt`
(re-baked with the new facet tables — `main.py export <PROD α=0.5 ckpt>` reproduces it for deploy). A canary eyeball
on the new country/language/format + intersection behavior is still worth doing before the next Streamlit deploy.

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
