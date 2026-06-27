# LLM Conversational Front-End for Two-Tower Recommender

## Feature Summary

Add a natural-language input layer on top of an existing deployed two-tower recommender (start with the Movie or Steam model). The user types a free-form request in plain English. A small, fast hosted LLM parses that request into the structured input the trained two-tower model expects. The two-tower model does the actual retrieval. The user sees recommendations from the trained model — never raw LLM output.

**Core principle:** The LLM is the interface, not the recommender. It translates human language into model input. It never decides what to recommend. The trained two-tower model does all the actual recommendation work.

This is the production-correct pattern for LLMs in recommendation systems: classical model for fast, cheap, large-catalog retrieval; LLM for natural-language understanding at the edge.

---

## Why This Architecture (Context For The Implementer)

This is a deliberate demonstration of a production pattern, NOT a claim that the demo needs an LLM at this scale. At ~5,000 items and low query volume, you could feed the catalog to an LLM directly. The two-tower model earns its place only at production scale — catalogs too large for any context window, millisecond latency budgets, billions of requests. The architecture demonstrates understanding of WHEN this pattern becomes necessary.

Keep this framing in the README. Do not claim the architecture "saves money" at demo scale — it doesn't, and any sharp reader will catch that. Frame it as demonstrating the pattern that scales.

---

## Iterative Rollout (v1 → v1.5 → v2)

Ship the smallest honest cut first, then add capability. Each phase is independently demoable and independently a stronger portfolio story than the last.

**v1 — Conversational input over the constraints the serving artifacts already support.**
- Develop and validate the extraction prompt entirely inside Claude Code, with NO hosted API and NO Streamlit wiring (see "Testing In Claude Code Before Any API"). This is the first thing built.
- Mode 1 (resolve mentioned titles) + Mode 2 (mood → genome tags → anchor movies), both wired into the existing `src/inference.py:build_user_embedding(...)` path.
- Post-filters limited to **what `serving/feature_store.pt` already carries**: `movieId_to_year` (date ranges) and `movieId_to_genres` (genre include/exclude). No director, no content rating yet.
- Once the prompt is dialed in against the example inputs, wire the hosted Haiku API into Streamlit as a new tab.

**v1.5 — Richer post-retrieval filtering (the "Post-Retrieval Filtering Layer" section below).**
- Genome-tag semantic facets: the LLM names a closed-vocab tag (e.g. `paris`); the system enforces a calibrated relevance floor or re-ranks by it. Uses data already in `serving/`.
- Scraped-data structured facet store: bake a compact `serving/filter_store.pt` (director/cast, certification, runtime, language, country, franchise, exact date) from the already-scraped TMDB+Wikipedia dump via `src/export.py`. Genuinely new serving data; never a model feature.
- Together these enable the "no Nolan films" / "family-friendly only" / "actually set in Paris" / "under 2 hours" / "Korean cinema" class of constraint that v1's year+genre filters can't express.

**v2 — Future extensions (out of scope until v1/v1.5 land).**
- Broader/fuzzier LLM-extracted facets in the filter store (setting-from-prose, tone, sub-genre) where structured fields don't reach.
- LLM-generated explanations shown back to the user (changes the security profile — see "Critical Design Constraint").
- Multi-turn conversation / conversational memory.
- Replicate the pattern on the Steam model.

The rest of this document describes the full target. The phase tags above govern what actually ships when.

---

## The Two Modes The LLM Operates In

The LLM's extraction job falls into two modes depending on what the user provides:

**Mode 1 — Resolution.** User mentions actual item titles ("I liked Inception and Interstellar"). The LLM resolves those titles to items in the catalog and extracts attribute/genre hints. Retrieval is grounded in real seed items. In this repo Mode 1 is a thin wrapper over the existing inference path: the LLM is just choosing the movies the user would otherwise have clicked in the manual UI.

**Mode 2 — Synthesis.** User describes taste/mood/constraints with no titles ("something slow and atmospheric and melancholy"). The LLM must construct a synthetic query. **Use the repo's existing genome-anchor mechanism, not a hand-built query vector.** The app already does this: a genome tag resolves to its top-k representative *real* movies (`USER_TYPE_TO_GENOME_TAGS` → anchor movies, `tools/persona_tools.py`, mirrored in `src/evaluate.py`), injected as likes at weight 1.0. So Mode 2 is: LLM maps the mood description to **genome tag names drawn from the closed 1,128-tag vocabulary** (`feature_store.pt['genome_tag_names']`), and the existing anchor machinery turns those tags into real seed movies. This is strictly better than synthesizing a raw content vector for two reasons: it keeps the dominant ID-pool channel of the user tower (128 of 196 concat dims) populated and **in-distribution** (the repo's own lesson — `probe_genome_tag`'s note that "real movies avoid OOD inputs"), and it reuses tested code instead of a new untested inference path.

Because the genome vocabulary is closed, the LLM must **select tag names from the provided list**, not free-generate mood words. Hand it the vocabulary (or the curated middle-frequency subset the genome map already uses) as part of the extraction prompt.

Most real queries are a mix of both modes.

---

## What The LLM Extracts (Structured Output)

The LLM's output is a structured object that maps directly onto `src/inference.py:build_user_embedding(...)`, whose signature already is the schema:

```
build_user_embedding(model, fs,
    liked_titles_with_weights,   # Mode 1 likes + Mode 2 genome anchors, [(title, weight), ...]
    disliked_titles,             # Mode 1 dislikes
    ts_inference,                # timestamp bin (use most-recent, like the canaries)
    liked_genres=(),             # genre hints
    disliked_genres=())          # genre exclusions the tower can express
```

So the extraction object is conceptually:

- **liked_items**: titles the user mentioned positively → resolved to catalog IDs/titles
- **disliked_items**: titles the user mentioned negatively → resolved to catalog IDs/titles
- **genome_tags** (Mode 2): mood/style descriptors mapped to names from the closed genome vocabulary → expanded to anchor movies, appended to `liked_titles_with_weights`
- **liked_genres / disliked_genres**: genre hints the user tower can express directly
- **hard_constraints**: things the model can't express but a post-filter can — year ranges and genre include/exclude in **v1**; director-exclusion / content-rating in **v1.5**

Most of the "model-input construction" the implementer might expect to write already exists. The new code is: the extraction call, title resolution/fuzzy-match, mapping genome tags → anchors (an existing helper), and the post-retrieval constraint filter.

---

## Example User Inputs (For Designing The Extraction Prompt)

These illustrate the range the extraction prompt must handle (tagged by the phase that can fully satisfy them):

1. "Something funny but smart, like Hot Fuzz or The Grand Budapest Hotel" → titles + attribute hints **(v1)**
2. "Something to watch with my 8-year-old that won't bore me to death" → no titles, constraint-driven, needs synthesis; "family-friendly" filter is **(v1.5)**, the mood synthesis is **(v1)**
3. "I liked Inception and Interstellar but I'm sick of Nolan, show me other smart sci-fi" → titles + hard exclusion; the Nolan exclusion is **(v1.5)**
4. "Slow, atmospheric, melancholy, maybe European, the kind of film where not much happens but it sticks with you" → pure mood, no titles, Mode 2 synthesis **(v1)**
5. "I want competitive multiplayer games like CS but less toxic" (Steam) → title + attribute + soft constraint **(v2 — Steam model)**

The extraction prompt should be tested against examples like these before scaling.

---

## Architecture Flow

```
User types natural-language request
        ↓
LLM extraction call (hosted API, small fast model)
  - resolve mentioned titles to catalog IDs (fuzzy-match against the catalog)
  - synthesize a Mode 2 query: mood → genome tag names (closed vocab) → anchor movies
  - extract genre/attribute hints
  - extract hard constraints
  - return STRUCTURED output (enforced schema)
        ↓
App code builds the user embedding via build_user_embedding(...)
  - liked titles + genome-anchor movies → likes; disliked titles → dislikes
        ↓
Two-tower model retrieves top-N candidates (fast, the actual ML)
        ↓
Post-filter applies hard constraints
  - v1: year ranges, genre include/exclude (metadata already in serving)
  - v1.5: excluded directors, content-rating limits (new serving metadata)
        ↓
User sees recommendations from the trained model
  (NOT raw LLM output)
```

---

## Critical Design Constraint: LLM Output Is Never User-Visible

The user types a prompt and sees recommendations. They never see the LLM's response. The LLM's output is consumed internally to build model input and is discarded after.

This is a deliberate security and architecture decision:

- **No free-chatbot abuse vector.** Because the LLM output is never shown to the user, nobody can use the demo as a free general-purpose LLM. There's no channel to read what the LLM produced. This eliminates the main abuse concern for public LLM-backed demos.
- **Reinforces the architecture story.** The LLM is plumbing, not the product. Keeping its output invisible makes that concrete.

Because of this, elaborate system-prompt constraints to "keep the LLM on task" are unnecessary. The architecture itself constrains use.

(Note: v2's LLM-generated explanations would re-introduce a user-visible channel and change this profile. That's why explanations are deferred, not in v1.)

---

## Testing In Claude Code Before Any API

**Goal: fully tune the extraction prompt and validate end-to-end recommendation quality before writing a single line of API integration or Streamlit wiring.** No hosted API, no secret, no `requests` call — all of that is wasted effort until the prompt produces good model input.

The loop runs entirely inside this Claude Code session:

1. **Claude Code drafts the extraction system prompt + JSON schema** (the `build_user_embedding` schema above, plus the closed genome-tag vocabulary or its curated subset).
2. **Claude Code spawns a Haiku subagent** (the `Agent` tool with `model: haiku`), hands it the draft system prompt + one test utterance from the example list, and the subagent returns the structured JSON. Using Haiku here is deliberate: it's the **same model family** as the hosted Haiku we'd call in v1's API step, so a prompt tuned against this subagent transfers directly to the API with no re-tuning.
3. **Claude Code feeds that JSON to the trained model locally** via a small test harness — a standalone script under `tools/` (e.g. `tools/llm_frontend_probe.py`, mirroring `tools/persona_tools.py`) that takes the extraction JSON, calls `build_user_embedding(...)`, ranks against `serving/movie_embeddings.pt`, applies the v1 post-filters, and prints the top-N titles. Short MPS inference like this is fine to run directly.
4. **Claude Code outputs the recommendations to the user** alongside the utterance and the extracted JSON, so the user can eyeball: did the LLM extract the right intent, and did the model return sensible movies?
5. **Iterate on the prompt** against the full example set (titles, pure mood, mixed, constraint-driven) until extraction and recommendations are consistently good. Watch specifically for: title-resolution misses, genome-tag choices that don't match the mood, and Mode 2 results collapsing to generic popularity.

Only after the prompt is locked in does v1 proceed to wire the real hosted Haiku call into Streamlit. The harness script stays in the repo as a reproducible test/QA tool and as a portfolio artifact showing the prompt was validated against the model, not just eyeballed.

### v1 Build Handoff — what the in-repo phase locked in

The Claude Code test loop is done; these decisions and artifacts are the build spec for the
remaining v1 work (hosted Haiku API + Streamlit tab). Build against these, don't re-derive them.

**Artifacts that already exist and are validated:**
- `tools/llm_frontend_probe.py` — serving/-only harness: `resolve_title` (fuzzy resolution),
  `anchors_for` (genome-tag → anchor movies), `recommend` (full pipeline), `_passes_constraints`
  (year+genre post-filter), `_build_serving_model` (mirrors `streamlit_app.py`).
- `tools/llm_frontend_prompt.py` — `build_system_prompt()` + `build_schema()` (closed genome +
  genre vocabs injected from `feature_store.pt`).

**Reuse, don't rebuild.** Factor the resolution / anchor / model-input / ranking / post-filter
logic out of `tools/llm_frontend_probe.py` into a shared `src/` module (e.g.
`src/llm_frontend.py`) that **both** the harness and `streamlit_app.py` import. The Streamlit
tab must NOT import the `tools/` script, and must reuse the existing `@st.cache_resource`
`load_artifacts()` model rather than reloading serving artifacts.

**Locked-in policy + constants (chosen by evidence in the loop):**
- Subordinated hybrid: pure Mode 2 (no named titles) → 5 anchors/tag at weight 1.0; with named
  titles → ≤1 anchor/tag, cap 3 total, weight 0.5 (named titles must dominate the embedding).
- Like weight 2.0, dislike −2.0; most-recent timestamp bin; rank the full corpus, post-filter,
  then take top-N. Empty extraction → popularity fallback (`popularity_ordered_titles`).
- Title resolution: stdlib `difflib` (rapidfuzz not installed); normalize by stripping the
  "(year)" suffix and inverting MovieLens "Name, The" → "the name"; break ties by popularity.
  The prompt instructs specific-title-with-year because vague "Lord of the Rings" fuzzy-matches
  the 1978 animated film, not Fellowship.

**Concrete hosted call:** model `claude-haiku-4-5` (same family as the test-loop subagent, so the
tuned prompt transfers); structured output via tool use — pass `build_schema()` as the tool
`input_schema` and force that tool with `tool_choice`; `max_tokens ≈ 300`; a per-session call
counter in `st.session_state` for the rate limit. Parse the forced tool-call input as the
extraction object — same shape the harness already consumes.

**Streamlit tab:** a new tab beside the manual interface; text input + submit; spinner during the
API call; render results as the existing poster cards. Never surface the raw LLM JSON to the end
user (optionally behind a debug expander for the portfolio narrative).

---

## Post-Retrieval Filtering Layer (v1.5 → v2)

v1 retrieves with the trained model and applies only the post-filters whose metadata is
already in `serving/` (year, genre). This section is the principled generalization of that
post-filter step, and it rests on one distinction worth making explicit:

- **Model features** must be embedded, must generalize, cost training + retraining, and live
  baked into the towers. Expensive, conservative, few.
- **Filter features** are pure metadata, never touch the model, and are applied
  deterministically *after* retrieval. Cheap, arbitrarily rich, zero retraining.

The two-tower model retrieves by taste-proximity; it has no structured field for "directed by
Nolan", "rated PG", "under 2 hours", or "actually set in Paris". Those are filter features.
The LLM's job stays pure translation: it names *which* facet to filter on; the system owns the
enforcement. The model still does all the recommending. There are two complementary facet
sources.

### Source A — Genome-tag semantic facets (soft scores; data already in `serving/`)

Every movie carries a 1,128-dim genome relevance vector (`movieId_to_genome_tag_context`, 0–1
per tag). The LLM emits `require_genome_tags: ["paris"]`; the system keeps/boosts movies by
that tag's relevance. This directly fixes the limitation that surfaced in testing: for "movies
set in Paris", the model retrieved French cinema generally (La Dolce Vita, Amarcord, Hello
Dolly leaked in), because it can't enforce a setting. The genome `paris` score separates
cleanly — Midnight in Paris 0.95, 2 Days in Paris 0.99, Cléo 0.99, Amélie 0.82, Play Time 0.79
vs. La Dolce Vita 0.08, Amarcord 0.08, Hello Dolly 0.06 — so a ~0.5 floor keeps the real Paris
films and drops the imposters (214 corpus movies clear 0.5; ample pool).

Design rules (these are what make it work, not "the LLM invents a filter"):
- **LLM names the tag; the system owns the threshold.** Never ask a small model to guess a
  cutoff for a score distribution it can't see. The floor must be **per-tag calibrated** (a
  percentile of that tag's score distribution), not a global constant — `paris` is clean at
  0.5, but a sparse or fuzzy tag's distribution is totally different.
- **Complement to the Mode-2 anchors, not a replacement — and route the two kinds of tag.**
  Concrete settings/attributes (`paris`, `courtroom`, `based on a true story`, `time travel`)
  → **hard genome filter**. Moods/vibes (`atmospheric`, `melancholy`) → **soft anchors**
  (existing Mode-2 behavior). Mis-routing — hard-filtering on `atmospheric` — guts the
  candidate pool. Teaching this split reliably is the main new prompt risk and needs its own
  schema field + test cases.
- **Hard filter on a soft score is brittle; prefer a re-rank by default.** Genome relevance is
  noisy ML/crowd relevance, not a label. Default to a soft re-rank (multiply cosine by genome
  relevance) which degrades gracefully; reserve a hard relevance floor for emphatic "*only*
  movies set in Paris". Filter over the whole ranked corpus (the harness already does), and on
  an empty/too-small pool relax the floor or drop the facet and **say so**.
- **Coverage ceiling.** Only 1,128 tags. `paris` exists; `tokyo` may not. The LLM (holding the
  vocab) can only request tags that exist, so it degrades gracefully, but many constraints
  won't be expressible this way. The honest caveat: a genome-tag filter means "strongly
  associated with Paris", not a guaranteed "filmed in Paris" — clean for well-populated tags,
  fuzzy for the long tail.

### Source B — Scraped-data structured facet store (hard labels; new `serving/filter_store.pt`)

The project already scrapes the full raw TMDB + Wikipedia record per movie (the LLM-feature
pipeline feeds only a discriminative subset to the model and drops the rest). That dropped
long tail is a free source of **factual** filter features the genome vocabulary can't express:
director, full cast, writers, certification/content rating, runtime, original language,
production country, franchise/collection, exact release date, budget/revenue. None of these
should ever be a model feature — but a post-filter wants exactly them. They are real labels
(director == "Christopher Nolan"), so they make clean **hard** filters, unlike genome's soft
scores. This is the right home for the v1.5 constraint class ("no Nolan", "PG-only", "under 2
hours", "Korean cinema", "not a sequel", "released after 2010").

Build shape:
- ETL: raw scrape → clean per-movie facets → a compact `serving/filter_store.pt` baked by
  `src/export.py` (the raw dump and `data/` are absent on Streamlit Cloud, so the facets must
  ship inside `serving/`, like the LLM buffer already does).
- **Use the right source per facet.** Structured TMDB fields are clean → hard filters. But
  "set in Paris" lives in Wikipedia *prose* — extracting it is its own LLM/NER job, so prefer
  the genome `paris` tag (Source A) as the cheap proxy rather than re-deriving setting from
  text. Don't reinvent what genome/genres already give (themes, broad genre); the scraped
  data's marginal value is the factual fields.
- **Scope discipline.** Bake only the dozen facets people actually phrase in natural language,
  not 50 nobody asks for. It is a point-in-time snapshot (fine for a demo; note it in the
  README).

### Division of labor

| Constraint kind | Source | Mechanism |
|---|---|---|
| Mood / vibe ("atmospheric", "melancholy") | genome tags | soft **anchors** (Mode 2, existing) |
| Semantic facet / setting-feel ("paris", "courtroom", "heist") | genome tags | calibrated re-rank / hard floor |
| Hard factual ("no Nolan", "PG", "<2h", "Korean", "not a sequel") | scraped facet store | exact **hard filter** |
| Year / genre | already in `feature_store.pt` | hard filter (v1) |

Framing for the README/portfolio: this is how production search/recsys separates the **ranker**
(compact learned retrieval) from the **facet/filter layer** (rich, cheap, deterministic). The
LLM maps natural language onto whichever facet fits; the trained model still retrieves; the
filter is a deterministic post-step. It strengthens "LLM as interface, not recommender" rather
than blurring it.

---

## LLM Choice and API Setup

(Applies once the in-repo prompt is validated and v1 moves to the hosted API.)

**Use a small, fast, cheap hosted model.** Intent parsing does not need a frontier model. Options: Claude Haiku, GPT-4o-mini, or Gemini Flash. The task is parsing one short message into structured JSON — small models do this perfectly at ~10-20× lower cost than frontier models. Prefer **Claude Haiku** here: it's the model family the prompt was tuned against in the Claude Code test loop, and it keeps the stack consistent with the repo's existing LLM-feature work.

**Cost is negligible.** A typical extraction call is ~350 input tokens + ~150 output tokens ≈ 500 tokens, roughly $0.0001-0.0005 per query. Even thousands of demo users cost under a dollar total.

**API key handling:**
- Store the API key as a Streamlit secret (set in the Streamlit Community Cloud dashboard, never committed to the repo).
- Access via `st.secrets[...]` in the app.
- Add the secrets file pattern to `.gitignore` so a local secrets file can never be committed.
- Document in the README that the key is required to run locally and how to set it.

**In the README, note the deliberate choice of a small model** — it reinforces cost-awareness and the "LLM for interface, classical model for retrieval" story. The actual recommendation comes from the trained two-tower model, not the LLM.

---

## Protections To Implement (Cost & Reliability, Not Abuse)

Because the LLM output isn't user-visible, the protections needed are minimal and exist for cost control and pipeline reliability, not abuse prevention:

1. **Structured output / schema enforcement (REQUIRED).** Use the provider's structured-output mode (Claude tool use, OpenAI structured outputs) so the LLM always returns parseable JSON with the expected fields. This prevents the model-input construction from crashing on malformed or hallucinated output. This is the single most important protection — not for security, but so the pipeline doesn't break.

2. **max_tokens cap (REQUIRED).** Cap output at ~300 tokens. Extraction JSON never needs more. Prevents a weird input from triggering an expensive long generation and forces concise output.

3. **Session rate limit (RECOMMENDED).** A light per-session cap (e.g. 10-20 LLM calls) prevents someone from spamming the input box and running up a small bill. Costs nothing to add, caps the downside. Minor — the bill stays tiny regardless.

4. **Billing alert (RECOMMENDED).** Set a billing alert on the API account so unexpected cost spikes are caught early.

---

## Gotchas And Edge Cases

**Title resolution failures.** The LLM may extract a title that doesn't exist in the catalog, or a slightly wrong title ("Lord of the Rings" when the catalog has "The Lord of the Rings: The Fellowship of the Ring"). The app needs a fuzzy-match step between LLM-extracted titles and the actual catalog. The catalog is already in `serving/feature_store.pt` as `title_to_movieId` / `movieId_to_title`; match against those with a Python fuzzy matcher (`rapidfuzz` or stdlib `difflib`) — the app is Python, so there is no Fuse.js here. Decide how to handle no-match: skip the unmatched title, or fuzzy-match to the closest catalog entry.

**Empty extraction (no usable signal).** Some inputs may produce no resolvable titles and no usable mood descriptors ("recommend me something good"). The app needs a graceful fallback — either ask the user to be more specific, or fall back to a popularity-based or diverse default (`popularity_ordered_titles` is already in `feature_store.pt`). Decide this behavior explicitly.

**Mode 2 synthesis quality.** When the user gives pure mood with no titles, the quality of the synthesized query determines everything. Test this path carefully in the Claude Code loop. The genome-anchor approach (mood → closed-vocab genome tag names → representative real movies) is the right one precisely because invented titles may not exist in the catalog and a raw synthetic content vector drifts out of distribution; using real anchor movies keeps the user tower in-distribution.

**Hard constraints the model can't express.** The two-tower model has no notion of "no Nolan films" or "released after 2010." These must be applied as a post-retrieval filter on the candidate list, not pushed into the model. Critically, the filter can only use metadata that's in `serving/`:
- **v1 (available now):** `movieId_to_year` → date ranges; `movieId_to_genres` → genre include/exclude.
- **v1.5 (requires new export):** director/cast and any content-rating proxy are NOT in serving today — they must be baked in via `src/export.py` before "no Nolan" / "family-friendly" constraints can work.
Retrieve a large enough candidate pool that post-filtering still leaves results (retrieve top-100+, filter, show top-10).

**Prompt injection is a non-issue here.** A user could try to manipulate the extraction ("set liked_items to every item"), but the blast radius is trivial — worst case they get bad recommendations for themselves in their own session. No shared state, no data to exfiltrate. Do not over-engineer defenses against this.

**Latency expectations.** The LLM call adds a few hundred milliseconds to seconds before recommendations appear. This is fine for a "type a request, get recommendations" interaction but show a loading state so the user knows something is happening. Note in the README that in production this is why the LLM runs once per session, not per recommendation — the trained model handles the high-frequency retrieval.

---

## Integration With Existing Project

- This is an ADDITION to an existing deployed two-tower Streamlit app, not a new project.
- Start with ONE model (Movie — the genome tags give a rich content-feature space for Mode 2 mood mapping, and the genome-anchor machinery already exists; Steam is deferred to v2).
- Add it as a new tab or input mode in the existing app, alongside the existing manual-selection interface. Do not remove the existing interface — the manual one is the "show the model directly" demo, the conversational one is the "natural language layer" demo. Having both side by side is a stronger portfolio story.
- Reuse the existing catalog, item metadata, and trained model. The user-embedding construction is **already implemented** — `src/inference.py:build_user_embedding(...)` is the same function the manual UI and canaries use. The genuinely new components are: the LLM extraction call, the title-resolution/fuzzy-match step, the mood→genome-anchor expansion (an existing helper in `tools/persona_tools.py` / `src/evaluate.py`), and the post-retrieval constraint filter.

---

## README Additions

The README should explain:

1. **What the feature does** — natural-language input layer over the trained recommender.
2. **The architecture decision** — LLM for interface, classical model for retrieval. Be honest that at demo scale you could use the LLM directly; the architecture demonstrates the production pattern that scales.
3. **Why a small model** — intent parsing doesn't need a frontier model; cost-awareness.
4. **That LLM output is never user-visible** — it's consumed internally to build model input.
5. **How to set the API key** for local runs (Streamlit secret).
6. **The iterative rollout** — v1 (title + mood + year/genre constraints), v1.5 (director/rating constraints), v2 (explanations, multi-turn, Steam), so a reader sees the roadmap and the honest scoping.

---

## Success Criteria

Phased to match the rollout.

**v1 complete when:**
1. The extraction prompt is validated end-to-end inside Claude Code (Haiku subagent → JSON → model → recommendations) across the example inputs, before any API wiring.
2. User can type a free-form natural-language request and receive recommendations from the trained two-tower model.
3. Title-mention requests (Mode 1) correctly resolve to catalog items and produce relevant recommendations.
4. Pure-mood requests (Mode 2) produce sensible recommendations via genome-tag → anchor-movie mapping.
5. Year-range and genre include/exclude constraints are correctly applied as post-filters.
6. Structured output enforcement means the pipeline never crashes on malformed LLM output.
7. API key is stored as a Streamlit secret, never committed.
8. Session rate limit and max_tokens cap are in place.
9. The existing manual-selection interface still works alongside the new conversational one.
10. README documents the architecture honestly, including the "LLM as interface, not recommender" framing and the rollout phases.

**v1.5 complete when:**
11. Director/cast (and any content-rating proxy) are baked into `serving/` via `export.py`.
12. Director-exclusion and content-rating/family-friendly constraints are correctly applied as post-filters.

---

## Out Of Scope

- **(v1)** Director-exclusion and content-rating/family-friendly post-filters — deferred to v1.5 because the metadata isn't in `serving/` yet.
- **(v2)** Showing LLM-generated text/explanations back to the user (keeping LLM output invisible preserves the v1 security profile; explanations change it).
- Using the LLM to re-rank or recommend directly (that defeats the architecture's purpose).
- Loading a local/open LLM in the app (won't fit Streamlit free-tier resources — hosted API only).
- **(v2)** Multi-turn conversation (single-shot request → recommendations is the v1 scope; conversational memory is a future extension).
- **(v2)** Replicating this across all four models at once (do the Movie model well first).
