# HANDOFF — Plot-derived NARRATIVE-DIMENSION feature (unshackled from genome)

_Paste this whole file back to a fresh session. It is self-contained. Written 2026-07-02 to survive a `/clear`._

## TL;DR
Build a NEW feature layer extracted from each film's **Wikipedia plot** that captures **story-level
narrative dynamics** (archetype / central conflict / thematic question) — the one dimension that genome
tags + TMDB keywords + the 11 structured facets all miss. Unlike the old LLM-132 vector (which was
deliberately shackled to genome's 132 axes), this uses a **fresh vocabulary**.

**HARD CONSTRAINT (the reason this is a handoff): the per-movie extraction over ~9k films MUST be done
by SONNET SUBAGENTS, never by the main agent.** The main agent (you) ONLY orchestrates: samples films,
authors/tunes the extraction prompt, launches the Workflow fan-out with `model: 'sonnet'`, aggregates
results, builds the storage + retrieval plumbing, and runs evals. You do not extract movies yourself.

## Where we are (all COMMITTED + PUSHED to main, front-end only, no retrain)
Recent LLM front-end hardening from a systematic bug sweep:
- `a8e42fc` — require_country reads **origin_country** (nationality, kills the co-financier leak: Cliffhanger/
  Scott Pilgrim out of "japanese movies") + **require_genome_tags seeds anchors** (fixes "movies about racing"
  collapsing to popularity → Star Wars; now F&F/Rush).
- `b54f258` — mood-router **nostalgic** + comparative `-er` forms; bare **marvel/dc** franchise aliases.
- `afbb1ad` — thin-axis **require_genome_tags ranks by subject relevance** not cosine
  (`REQUIRE_GT_RERANK_LAMBDA=2.0`): "movies about chess" now surfaces Seventh Seal/Fresh, drops X-Men /
  the Chess-Records homonym.
- `0fa52c7` — **require_min_rating** content-rating FLOOR (mirror of the ceiling): "R-rated comedies"
  no longer returns Shrek.

Ruler (`python tools/llm_frontend_eval.py`): **REGRESSION 78/78, SPEC 6/7** (the 1 SPEC fail is a
pre-existing Christmas/Gremlins genre-purity gap, unrelated). Cases live in
`docs/llm_frontend/validation/retrieval_eval/eval_cases.json`.

## Why this feature (the verdict from a plot analysis of 9 diverse films)
Genome + keywords cover concrete content (objects/settings/topics) densely — and keywords are actually
CLEANER than genome's top signal, which is **award/meta-polluted** (Rocky's top-5 genome tags are 4 Oscars +
`sports`; `boxing` is 13th). But the **plot encodes story SHAPE and CENTRAL DYNAMIC that cut across surface
genre**, and nothing we have represents it. Concrete proof from the sample:

| Film | genome/keyword surface | narrative_archetype | central_dynamic |
|---|---|---|---|
| Ocean's Eleven | heist, caper, casino | assemble-the-team | crew vs. the mark |
| Rocky | boxing, underdog | underdog-one-shot | self-worth vs. the odds |
| Groundhog Day | time-loop, comedy | trapped→transformation | man vs. own cynicism |
| **Jaws** | shark, suspense | monster-terrorizes-town | **individual vs. institution (cover-up)** |
| **A Few Good Men** | courtroom, military | investigation-uncovers-coverup | **lone lawyer vs. chain of command** |
| **Spotlight** | journalism, boston | investigation-uncovers-coverup | **small team vs. powerful institution** |
| 127 Hours | survival, trapped | trapped-survival-ordeal | man vs. nature & own limits |
| Sixth Sense | ghost, twist | rug-pull revelation | helper haunted by past failure |
| Whiplash | jazz, mentor, obsession | obsessive-pursuit | protégé vs. abusive mentor |

The payoff clusters genome SCATTERS: **"individuals expose an institutional cover-up"** links Spotlight +
A Few Good Men + Jaws (three totally different genome regions); **"obsessive pursuit at great cost"** links
Whiplash + 127 Hours; **"trapped → transformation"** links Groundhog Day + 127 Hours. A user who loved
Spotlight and wants "more of that story" gets nothing pointing to A Few Good Men today. THAT is the hole.

## Plan
### Step 1 — VALIDATION PROTOTYPE first (do NOT skip; it decides go/no-go)
Sample ~30–40 diverse films (span genre/era/tone). Sonnet subagents extract the schema (below) from each
`wikipedia.plot`. Then YOU (orchestrator) cluster by the extracted tags and compare to genome cosine
neighborhoods. **GO** only if the new dimension produces semantically-coherent, genome-ORTHOGONAL clusters
(e.g. archetype `investigation-uncovers-coverup` actually groups Spotlight+A Few Good Men+Jaws, and genome
cosine does NOT). If genome already implicitly captures it → ABANDON and instead build the deferred
**keyword content facet** (higher raw ROI for concrete queries). Report the cluster evidence to the user
before any 9k run.

### Step 2 — FULL RUN (only if the prototype validates)
Sonnet-subagent fan-out over all ~9,300 scraped films → the narrative schema per film. Store, wire retrieval,
add eval cases, keep ruler ≥ green.

## THE EXTRACTION PROMPT — hand this to each Sonnet subagent (one or a batch of films per agent)
Give the subagent, per film: `title`, `year`, `genres` (light context only), and the `wikipedia.plot` text.
Instruct it to force a `StructuredOutput` call against this schema (use the Workflow `schema` option):

```
You are extracting the STORY STRUCTURE of a film for a recommender, from its plot summary ONLY.
Goal: capture the abstract SHAPE of the story so films with the SAME kind of story cluster together
EVEN ACROSS different genres/settings. Do NOT describe surface content (objects, place, era, cast) —
genome tags and keywords already cover that. Abstract UP to the narrative pattern.

Base everything ONLY on the PLOT text provided; ignore reviews, awards, box office, trivia. If the plot
is missing or too thin (< ~40 words), set "insufficient_plot": true and leave the tag arrays empty.

Prefer the controlled vocabulary below so films cluster consistently. If NOTHING fits, you may coin a new
tag prefixed with "new:" (e.g. "new:body-swap") — keep coined tags short, lowercase, hyphenated, and GENERIC
(reusable across films), never film-specific.

CONTROLLED VOCAB (seed — extend via "new:" sparingly):
  narrative_archetype (pick 1–2): assemble-the-team | heist | underdog-one-shot | rise-and-fall |
    obsessive-pursuit | trapped-survival | investigation-uncovers-coverup | whodunit | rug-pull-twist |
    revenge-quest | redemption-arc | coming-of-age | fish-out-of-water | forbidden-love | road-trip |
    chosen-one | monster-terrorizes-community | siege-last-stand | trapped-to-transformation |
    slow-descent | wrongly-accused | con-long-game | survival-vs-nature | mentor-and-protege
  central_dynamic (pick 1–2): individual-vs-institution | man-vs-self | man-vs-nature | mentor-vs-protege |
    oppressor-vs-oppressed | betrayal-among-allies | forbidden-love | outsider-vs-community |
    order-vs-chaos | duty-vs-desire | hunter-and-hunted | family-fracture-and-repair
  protagonist_type (pick 1): reluctant-hero | dogged-investigator | obsessive-genius |
    everyman-in-over-his-head | anti-hero | underdog | seasoned-pro-last-job | naif-loses-innocence |
    haunted-by-the-past

Return EXACTLY this JSON (StructuredOutput):
  {
    "insufficient_plot": <bool>,
    "narrative_archetype": [<=2 tags],
    "central_dynamic": [<=2 tags],
    "protagonist_type": [<=1 tag],
    "thematic_question": "<one short genre-neutral question the film poses, e.g. 'is greatness worth the cost to one's humanity?'>",
    "story_engine": "<ONE genre-neutral sentence naming the story shape, e.g. 'a small group risks everything to expose a powerful institution's cover-up'>"
  }
```
Notes for you (orchestrator) when authoring the Workflow:
- `agent(prompt, { model: 'sonnet', schema: NARRATIVE_SCHEMA, label: 'extract:<mid>' })` — force Sonnet.
- Batch a handful of films per subagent (e.g. 5–10) to amortize, OR one-per-agent for the prototype.
  Cap concurrency is automatic. For 9k, loop in chunks; `.filter(Boolean)` dead agents.
- After each round, aggregate "new:" tags; if a coined tag recurs (≥~5 films), promote it into the controlled
  vocab and note it, so the vocabulary converges instead of sprawling. (loop-until-vocab-stable.)
- Reuse the Stage-2 fan-out recipe already proven for the LLM-132 extraction — see memory
  `project_llm_extraction_stage2` for the exact Workflow pattern + the Stage-3-merge gotcha.

## Data + infra (facts)
- Plot: `llm_features/cache/scraped/{movieId}.json` → `['wikipedia']['plot']` (rich narrative, ~96.5% cov);
  `['tmdb']['overview']` (short, ~100% cov, fallback); `['tmdb']['genres']`, `title`, `year`.
- ~9,366 scrape files; serving corpus = 9,375 `top_movies`. `data/ml-32m/movies.csv` = movieId,title,genres.
- Serving: `serving/feature_store.pt` (weights_only=False). Facets under `fs['facets']`. Genome under
  `fs['movieId_to_genome_tag_context']` / `fs['genome_tag_names']` (dict id→name) / `fs['genome_tag_to_i']`.
- Probe pipeline (no API): `from tools.llm_frontend_probe import Serving; from src.llm_frontend import recommend`.

## Storage + retrieval (orchestrator's build, AFTER extraction validates)
- STORAGE options — decide during the prototype: (a) sparse TAG MEMBERSHIP (interpretable, mirrors keywords/
  facets; bake `fs['facets']['movieId_to_narrative']` in `build_facet_store.py`-style + re-bake serving),
  or (b) a small EMBEDDING of the tag-set/story_engine (denser, fuzzier; new buffer). Start with (a) — the
  controlled tags are the whole point (clustering + a `require_archetype`/soft-anchor path).
- RETRIEVAL — mirror existing patterns in `src/llm_frontend.py`: soft ANCHORS from archetype/dynamic tags
  (like `anchors_for` over genome), and/or a hard membership gate. Add extractor routing in
  `src/llm_frontend_prompt.py` so a query like "movies where a small group takes on a corrupt institution"
  → `central_dynamic: individual-vs-institution` / `narrative_archetype: investigation-uncovers-coverup`.
- EVAL: add cases to `docs/llm_frontend/validation/retrieval_eval/eval_cases.json`; keep
  `tools/llm_frontend_eval.py` REGRESSION green (currently 78/78). A good regression: a `story_engine`/archetype
  query pulls the cross-genre cluster (Spotlight→A Few Good Men) that genome cosine does not.

## Conventions (this repo)
- Front-end changes need NO retrain; validate with the ruler + direct `recommend()` probes.
- Never commit+push in one command unless the user says so; the user has been approving each fix and asking
  to commit+push per-fix. Keep serving re-bakes to when a facet table actually changes (80MB LFS-warned file).
- The user's steer this session: prioritize UNCOVERING THE RIGHT MOVIES over weak filter features; do NOT
  sacrifice other evals for a secondary feature.

## Deferred (don't lose these — from the feature-richness audit)
- **Keyword content facet** (HIGH ROI, medium): curated ~50–100 thin/homonym concepts (chess/submarine/boxing/
  dinosaur) → `movieId_to_keyword_concepts`, hard boolean pre-filter. Verified: `chess` keyword = 11 clean films,
  no Chess-Records homonym. This is the FALLBACK if the narrative prototype doesn't validate.
- **Intensity channel** (HIGH ROI, medium): "very gory"/"extremely sad" are byte-identical to "gory"/"sad" today.
  Add a `strength` field to the extraction schema + map to a GENOME_HARD_FLOOR bump / λ×2; loosen the prompt's
  intensity-word suppression (`src/llm_frontend_prompt.py` ~L220-226).
- Studio facet (A24/Pixar/Ghibli), vote_count reliability floor, plot-embedding index (do keyword facet first).
- Explicitly LOW-ROI / skip: runtime "feels long", "no sad ending", budget buckets, popularity, reception text.

## First actions for the fresh session
1. Re-read this file + memory `project_facet_store_plan` and `project_llm_extraction_stage2`.
2. Confirm ruler still 78/78 (`python tools/llm_frontend_eval.py`).
3. Author the prototype Workflow: pick ~35 diverse films, Sonnet-subagent extraction (schema above),
   aggregate, cluster, compare vs genome cosine. Report go/no-go to the user BEFORE any 9k run.
