# LLM-Generated Item Features vs Human-Curated Genome Tags

## Goal

Test whether LLM-generated item features can match or beat the human-curated MovieLens genome tags as content signal in a two-tower recommendation model.

**Research question:** Can modern LLMs replace expensive human curation for content features in recommendation systems?

**Deliverable:** Three trained two-tower models on MovieLens 32M — one using genome tags (current prod), one using LLM-generated features from scraped web content, and a no-content-slot floor baseline. Quantified ablation of which produces better recommendations, with the floor baseline measuring how much either content source adds at all.

This is a **parallel experiment** within the existing Movie Recommender repo, not a new repo or project.

---

## Why This Matters

The MovieLens genome tags are 1128-dimensional relevance scores manually curated by tagging the MovieLens community. They are unusually high-quality and unusually expensive to produce. Most real-world recommendation problems don't have this kind of pre-curated content data.

If LLM-generated features can match genome tags, that's a meaningful result: LLMs can substitute for expensive human curation in production recommendation systems. If they can't, that's also meaningful: hand-curated content data still beats automated extraction for this task.

Either outcome is a defensible portfolio finding. The experimental design is what matters, not which result you get.

---

## Non-Goals (Do Not Do These)

- Do NOT replace the entire item tower with LLM-generated features only as a baseline — the comparison must be apples-to-apples (same architecture, same training, same eval, only the content feature source changes)
- Do NOT use LLM embeddings as raw 1536-dim vectors and call it a day — the experiment needs structured extraction comparable to genome tag structure
- Do NOT add LLM features ON TOP of genome tags — that's a different experiment (does more help?) and dilutes the core question
- Do NOT use one monolithic prompt with 150+ dimensions — see Stage 2 for why
- Do NOT hand-invent the dimension list — derive it from the top-discriminability genome tags (`data/top_genome_tags_by_discriminability.csv`) so the LLM and genome measure the same axes (Stage 2)
- Do NOT ask the LLM for subjective aesthetics (`visually stunning`, `cinematography`, `neon_aesthetic`) from a plot synopsis — it will hallucinate; visual dims are factual-medium-only unless backed by a scraped critic quote (Stage 2 group 6)
- Do NOT use freeform LLM completion — structured output / JSON mode is non-negotiable
- Do NOT scrape web content without rate limiting, caching, and respect for source terms of service
- Do NOT spend more than $200 total on LLM inference costs — hard upper bound

---

## Repo Structure

Add to existing repo: `Movie-Recommender-System-PyTorch-TwoTower-Model`

```
Movie-Recommender-System-PyTorch-TwoTower-Model/
├── (existing src/, data/, etc unchanged)
├── llm_features/                       ← new directory, isolated
│   ├── derive_schema.py                ← top-discriminability genome tags → deduped, bucketed dimension list
│   ├── scrape.py                       ← web scraping per movie (incl. objective prestige indicators)
│   ├── llm_extract.py                  ← grouped LLM feature extraction
│   ├── merge_extractions.py            ← combine grouped outputs per movie
│   ├── build_features.py               ← convert merged JSON → tensor
│   ├── prompts.py                      ← 6 grouped prompts as code (versioned)
│   ├── schemas.py                      ← Pydantic schemas (genome-derived) for structured output
│   └── cache/
│       ├── scraped/                    ← raw scraped content per movie
│       │   └── {movieId}.json
│       ├── llm_groups/                 ← per-group LLM responses
│       │   ├── themes/{movieId}.json
│       │   ├── tone/{movieId}.json
│       │   ├── narrative/{movieId}.json
│       │   ├── visual/{movieId}.json
│       │   ├── audience/{movieId}.json
│       │   └── emotional/{movieId}.json
│       └── llm_merged/                 ← merged per-movie feature dicts
│           └── {movieId}.json
├── data/
│   ├── llm_experiment_movies_phase1.json  ← filtered ~4-5k movie list (Phase 1)
│   └── llm_features_v1.pt              ← (n_movies, feature_dim) tensor
│       (fills the renamed content_context_buffer slot; see repo_cleanup_plan Tier 1.1)
├── saved_models/
│   ├── best_mse_genome_*.pth           ← Model A (genome features)
│   ├── best_mse_llm_*.pth              ← Model B (LLM features)
│   └── best_mse_nocontent_*.pth        ← Model C (no content slot, floor baseline)
└── results/
    └── llm_vs_genome_ablation.md       ← results writeup
```

---

## Two-Phase Approach

This experiment runs in **two phases** to validate the approach cheaply before committing to full-corpus extraction.

- **Phase 1 — Reduced Corpus Pilot:** Filter the corpus down to ~4-5k movies, run the full pipeline (scrape → extract → merge → train all three models → eval) on that reduced set, and check whether the approach works. All three models (genome, LLM, and no-content floor) are trained on the reduced corpus — do NOT compare against the existing full-corpus prod model in Phase 1; that's apples-to-oranges.
- **Decision Gate:** Pause. The user reviews Phase 1 results and decides whether they justify the additional Phase 2 cost.
- **Phase 2 — Full Corpus:** Only if Phase 1 looks good. Scale scraping/extraction to the full 9,375-movie corpus and train **three fresh models** (genome, LLM, no-content floor) on the full corpus at `alpha=0`. Do NOT reuse the existing deployed prod checkpoint as a baseline — it was trained with `alpha=0.5`, which confounds the comparison on the exact axis (popularity) this ablation holds constant.

The stages below are written for Phase 1 (reduced corpus). The "Phase 2: Full Corpus" section near the end describes the scale-up deltas. The Decision Gate between them is mandatory.

---

## Stage 0: Corpus Filtering (Phase 1)

> **Prerequisite:** Complete Tiers 0–1 of [`repo_cleanup_plan.md`](repo_cleanup_plan.md) before scraping — they make the repo able to train more than one model variant (swappable content slot, corpus-namespaced artifacts) and fix latent bugs (Model C would otherwise KeyError on config resolution). Tier 2 (serving: export, Streamlit comparison tab, re-export, deploy) is **gated on this experiment's Decision Gate** and only runs on a positive result.

**Goal:** Validate the experimental approach cheaply before committing to full-corpus extraction.

### `llm_features/filter_corpus.py`

Apply a higher rating-count threshold to filter the MovieLens corpus down to **~4-5k movies** (vs. the full 9,375).

- Target: keep movies with at least **100-200 ratings** — tune the threshold until you hit the ~4-5k target size.
- Save the filtered movie list to `data/llm_experiment_movies_phase1.json`.
- Document the exact threshold and resulting movie count in CLAUDE.md.

All Phase 1 stages (scrape, extract, train) operate on this filtered list, not the full corpus.

### Phase 1 success criteria

After Phase 1, you should be able to answer:

1. Does the grouped extraction pipeline work without manual intervention?
2. Do the LLM features pass the similarity sanity checks (Toy Story 1/2/3 similar, etc)?
3. Does Model B (LLM features) train without issues?
4. Is the quantitative comparison meaningful — clear winner, clear loser, or roughly equivalent?
5. Do the canary user comparisons reveal interesting qualitative differences?

If all of these are yes, proceed to the Decision Gate and then Phase 2. If any are no, fix the problem before scaling.

---

## Stage 1: Web Scraping Per Movie

> **Phase 1:** Scrape only the filtered ~4-5k movies from `data/llm_experiment_movies_phase1.json`, not all 9,375.

### `llm_features/scrape.py`

Collect raw web content per movie that the LLM will reason over.

**Sources (in priority order):**

1. **TMDB** — overview, tagline, genres, cast, director, writer, production companies, runtime (free API, generous limits)
2. **IMDB** — plot summary, plot synopsis, keywords, taglines, MPAA rating
3. **Wikipedia** — full plot section, reception, themes
4. **Reddit** (optional) — top posts from r/movies, r/TrueFilm mentioning the movie

**Objective prestige/reception indicators (required — feeds Stage 2 group 5).** Genome's most discriminative tags include crowd-prestige signals (`criterion`, `imdb top 250`, `oscar (best …)`) that an LLM cannot infer from a plot. To level this *factually* rather than ban it, scrape the verifiable versions:
- **Oscar wins / nominations** (Wikipedia "Accolades" / awards section, or an awards source)
- **Criterion Collection status** (criterion.com spine list / boolean)
- **IMDb rating + vote count** and **box-office scale** (TMDB/IMDb)
- **Major-festival awards** (Palme d'Or, etc.) where available

These are mapped to schema fields factually (e.g. `oscar_winner`, `criterion`) — no hallucination. Cache alongside the other scraped fields.

**Per-movie output:** JSON file with all scraped fields, including the prestige indicators.

### Practical guidelines
- **TMDB-first is the default LLM input.** Structured TMDB fields (overview, tagline, genres, cast, director) are the canonical input to extraction — they keep the prompt small and cost-predictable. Other sources are supplementary.
- **Truncate Wikipedia hard.** Full Wikipedia plots average 1,500–3,500 words. Feeding them raw blows the token budget (see Cost Budget — output dominates, but uncapped input still ~5×'s the bill). Cap Wikipedia text to the first ~1,000–1,500 characters, OR pre-summarize long plots with a cheap model (e.g. Haiku) before the extraction call. Do NOT feed raw full plots to the extraction model.
- Cache aggressively — scrape ONCE per movie, never re-scrape
- Rate limit conservatively — TMDB is 40 req/10s; respect IMDB's robots.txt
- Skip movies where scraping fails — 90% coverage is fine
- Print summary at end: how many movies have full data, how many partial, how many failed

### Estimated cost & time
Zero cost. Phase 1: ~1-2 hours for the filtered ~4-5k movies if rate-limited properly. Phase 2: a similar amount for the remaining long tail (full corpus ~9,375).

---

## Stage 2: LLM Feature Extraction (Grouped + Structured)

> **Phase 1:** Run extraction only on the filtered ~4-5k movies (~27k calls). See the Cost Budget for the full input/output breakdown — output tokens dominate. Even the lean TMDB-only path lands near the **$100 Phase 1 cap**, so the input-discipline (Stage 1) and model-choice test (below) are not optional.

### Critical design decision: grouped prompts with structured output

A single prompt asking the LLM to extract 150+ dimensions in one pass suffers from a documented failure mode: the LLM gives careful values to the first 20-30 dimensions, then defaults to 0.5 or mirrors structure for the rest. This is the "lost in the middle" / response fatigue problem.

**Mitigation: split into 6 focused prompts of ~25-35 dimensions each, each enforced by structured output schema.**

### Derive the schema from genome discriminability — do NOT hand-invent dimensions

**The dimension list is not invented from scratch.** Inventing ~145 dims (the original draft of this section) creates a feature-axis mismatch: the LLM and genome would measure different things, making "LLM ≈ genome" unmeasurable. Instead, **derive the LLM schema from the top ~120–150 most discriminative genome tags** in `data/top_genome_tags_by_discriminability.csv` (ranked by `std_score` — high std = high separating power). Same semantic axes as genome, different extraction method (crowd-curation vs LLM). That is the clean, measurable experiment.

**Derivation procedure (`llm_features/derive_schema.py`):**
1. Take the top ~150 tags by `std_score`.
2. **Dedup near-synonyms.** The genome vocab is redundant — e.g. `based on a book` / `adapted from:book` / `based on book` are one concept; `sci-fi` / `scifi` / `science fiction` / `sci fi` are four spellings of one. Merge to one dimension each, or you waste the budget on duplicates.
3. Drop tags that are not derivable from scraped content AND not factually scrapeable (pure crowd-sentiment — see the reception note below).
4. Bucket the survivors into 6 focused Pydantic schemas (~20–30 dims each, preserving the anti-fatigue rationale).

### Six grouped extraction calls per movie (genome-derived)

1. **THEMES & PLOT** (~25): murder, obsession, loneliness, revenge, relationships, betrayal, survival, redemption, coming_of_age, isolation, corruption, conspiracy, … — content-derivable from plot text.

2. **TONE & MOOD** (~25): tense, bleak, weird, cerebral, melancholic, enigmatic, intimate, atmospheric, dark_humor, suspenseful, … — inferable from text, calibration-sensitive.

3. **SETTING & ERA** (~20): world_war_ii, 1980s, new_york, japan, medieval, space, cold_war, small_town, … — factually extractable; this family was missing from the original draft despite being highly discriminative.

4. **PROVENANCE & STRUCTURE** (~25): based_on_a_book, sequel, franchise, remake, biographical, based_on_true_story, nonlinear, twist_ending, ensemble_cast, … — factually extractable from metadata.

5. **FACTUAL RECEPTION & PRESTIGE** (~20): oscar_winner, criterion, imdb_top_250, classic, box_office_scale, … — **mapped from scraped objective indicators (Stage 1), not inferred from plot.** Kept as its **own group so it is separately ablatable** (see reception note). This is the family the LLM otherwise structurally cannot reach.

6. **VISUAL MEDIUM (rescoped — factual only)** (~12): animation, black_and_white, cgi_heavy, stop_motion, live_action, computer_animation, … — **factual medium descriptors only.** Do NOT include subjective aesthetics (`visually stunning`, `cinematography`, `neon_aesthetic`) — those require watching the film and will be hallucinated from a synopsis. Include a subjective visual dim ONLY if a scraped critic quote explicitly contains it.

Total: ~120–145 dimensions across 6 calls, every dimension traceable to a high-discriminability genome tag.

> **Reception/prestige — the validity threat, and how groups 5 handles it.** Genome's most discriminative tags include `criterion` (#10), `cinematography` (#23), `masterpiece` (#25), `imdb top 250` (#47) — crowd judgments, not content. Giving genome that signal while denying it to the LLM is unfair. **Group 5 levels the *factual* slice** by scraping objective indicators (Oscar wins/noms, Criterion status, IMDb rating, box-office scale) so the LLM maps them factually with no hallucination. **But a residue remains genome-only:** pure-sentiment tags (`masterpiece`, `great acting`, `predictable`, `overrated`, `so bad it's funny`) cannot be made objective and stay a genome advantage — disclose this in Limitations. **Caution:** box-office / IMDb-rating are quasi-popularity signals; injecting them into the content slot blurs the content-vs-collaborative line and sits in tension with the `alpha=0` "no popularity" stance. Because group 5 is a separate schema, report eval **with and without it** so its contribution is isolable.

### Structured output is non-negotiable

Every call must use the LLM provider's structured output mode:
- **Claude**: tool use with strict JSON schema validation
- **GPT**: structured outputs with Pydantic models or strict mode

This guarantees:
- Every named field appears in output (no silent dropping)
- All values are floats in [0.0, 1.0] (no string responses, no out-of-range)
- JSON is always parseable (no malformed output)

### `llm_features/schemas.py`

One Pydantic schema per group:

```python
from pydantic import BaseModel, Field

class ThemesFeatures(BaseModel):
    redemption: float = Field(ge=0.0, le=1.0)
    revenge: float = Field(ge=0.0, le=1.0)
    love: float = Field(ge=0.0, le=1.0)
    # ... ~22 more fields
    
    class Config:
        extra = "forbid"  # reject any unexpected fields

# Repeat for ToneFeatures, NarrativeFeatures, etc.
```

### `llm_features/prompts.py`

Six prompts, each laser-focused on one category:

```python
THEMES_PROMPT = """You are extracting thematic content features from a movie for a recommendation system.

For the movie below, assign a relevance score (0.0 to 1.0) for each theme:
- 0.0 means the theme is definitely not present
- 1.0 means the theme is extremely prominent
- Use the full 0.0-1.0 range, not just 0.0/0.5/1.0

Themes to score:
[list of ~25 theme dimensions]

Be thoughtful and use the range. Do not default to 0.5 for uncertainty — make a calibrated estimate.

Movie information:
{movie_content}
"""
```

Repeat for each of the 6 groups.

### Per-movie extraction flow

```python
def extract_features(movie_id, scraped_content):
    formatted = format_for_prompt(scraped_content)
    
    for group_name, prompt_template, schema in GROUPS:
        cache_path = f"cache/llm_groups/{group_name}/{movie_id}.json"
        if exists(cache_path):
            continue
        
        prompt = prompt_template.format(movie_content=formatted)
        response = llm_client.complete(
            prompt=prompt,
            response_schema=schema,  # structured output enforcement
        )
        
        validated = schema.validate(response)
        save_to_cache(cache_path, validated)
```

### LLM choice — test, don't assume
The prior assumption was "use Claude Sonnet, the cost difference vs a small model is negligible." **That is wrong here, because output dominates the bill** (structured float extraction is output-heavy, and Sonnet output is $15/MTok vs Haiku ~$4/MTok). So model choice is a real cost lever, not a rounding error.

**Decide it empirically in the validation pass below:** run the same 20 spot-check movies through both Claude Sonnet and Haiku 4.5, compare calibration/quality, and pick the cheapest model that passes the sanity checks. Do NOT lock Sonnet in by default.

**Prompt caching:** the 6 group prompts are static except the per-movie content block. Caching the static prefix cuts *input* cost if movies are batched by group within the 5-min cache TTL. Worth doing — but note it does not touch the dominant *output* cost, so it can't rescue the budget on its own.

### Estimated cost
See the consolidated Cost Budget section — input and output are priced separately there because output is the binding constraint.

### Estimated time
~56k LLM calls. With reasonable parallelism (10-20 concurrent): ~3-4 hours. Cache every response so re-runs are free.

### Validation pass before scaling
Before running on all 9,375 movies, spot-check 20 random movies:

- Within each group, are values calibrated (full 0-1 range, not all 0.5)?
- Do values match intuition (Mad Max should score high on "gritty" and "action_heavy")?
- Do similar movies produce similar features (Toy Story 1 and 2)?
- Do cross-group features avoid contradiction (not 0.9 "uplifting" AND 0.9 "devastating")?
- **Model bake-off:** run the 20 movies through both Sonnet and Haiku 4.5. If Haiku passes the calibration/intuition checks, use it — it cuts output cost ~4×. Record the decision and the per-call token counts (measured, not estimated) so the Cost Budget can be re-confirmed before scaling.

If quality is poor, revise prompts/schemas **before** spending the full LLM budget.

---

## Stage 3: Merge Group Outputs

### `llm_features/merge_extractions.py`

Combine the 6 per-group JSON files into one per-movie feature dictionary:

```python
def merge_per_movie(movie_id):
    merged = {}
    for group_name in GROUPS:
        group_data = load(f"cache/llm_groups/{group_name}/{movie_id}.json")
        merged.update(group_data)
    
    save(f"cache/llm_merged/{movie_id}.json", merged)
    return merged
```

### Cross-group consistency check (critical)

After merging, run consistency validation across the corpus:

```python
def validate_consistency():
    for movie_id in all_movies:
        merged = load_merged(movie_id)
        
        if merged['uplifting'] > 0.8 and merged['devastating'] > 0.8:
            log_warning(f"Movie {movie_id}: contradictory emotional profile")
        
        if merged['family_friendly'] > 0.8 and merged['mature_themes'] > 0.8:
            log_warning(f"Movie {movie_id}: contradictory audience targeting")
        
        if merged['animated'] > 0.8 and merged['practical_effects'] > 0.8:
            log_warning(f"Movie {movie_id}: contradictory visual style")
    
    print(f"Flagged {n_warnings} movies for review")
```

Contradictions over a few percent of the corpus suggest a problem with extraction setup, not just LLM noise. Investigate before training.

---

## Stage 4: Build Feature Tensor

### `llm_features/build_features.py`

Convert merged JSON files into a tensor matching the structure of the genome content buffer (renamed `content_context_buffer` after repo_cleanup_plan Tier 1.1).

```python
def build_llm_feature_tensor(merged_dir, top_movies, output_path):
    """
    Build (n_movies + 1, feature_dim) tensor.
    Row i = features for movie at corpus index i.
    Last row = padding (zeros).
    """
    # 1. Determine canonical feature order from schemas
    #    Save as feature_names.json for reproducibility
    
    # 2. For each movie in top_movies (in order):
    #    - Load llm_merged/{movieId}.json
    #    - Extract values in canonical feature order
    #    - Missing → 0.0
    
    # 3. Stack into tensor, add padding row
    #    Save to data/llm_features_v1.pt
    
    # 4. Print summary:
    #    - Coverage, feature distribution, sanity similarities
```

### Validation: similarity check
Before training, compute cosine similarity between known-similar movies:
- Toy Story 1, 2, 3 should be highly similar
- The Godfather Part I and II should be highly similar
- Random movie pairs should be much less similar

If similarities don't look right, extraction or aggregation is broken. Fix before training.

---

## Stage 5: Train Three Models

Train three two-tower models with **identical hyperparameters**. The only difference is what fills the content slot: genome (A), LLM (B), or nothing (C).

> **Phase 1:** Retrain all three models on the reduced ~4-5k corpus. Do NOT compare against the existing full-corpus prod model — that's apples-to-oranges. Models A, B, and C must use identical hyperparameters and the same reduced corpus.

### Model A — Genome
**Always retrained fresh at `alpha=0`, never the prod checkpoint.** **Phase 1:** train on the filtered ~4-5k corpus. **Phase 2:** train on the full 9,375-movie corpus. The existing deployed prod checkpoint was trained with `alpha=0.5`, so reusing it as a baseline would confound the comparison on popularity — the one axis this ablation must hold constant. Cost of retraining is $0 compute.

### Model B — LLM features
Same architecture as Model A, but the content slot is filled with `llm_features_*.pt` instead of genome. **Phase 1:** train on the reduced ~4-5k corpus. **Phase 2:** train on the full 9,375-movie corpus.

### Model C — No content slot (floor baseline)
Same architecture as A and B, but with the **content slot removed entirely** — no genome buffer, no LLM buffer (`content_feature_source = None`). This is the floor: it measures how much *any* content feature adds over the model's remaining signals.

**Critical — remove ONLY the slot under test.** Keep every other item feature identical to A and B: genre one-hot, tag vector, item ID embedding, year. Do NOT also strip genre/tag — that would make C a different model, not a matched control. The only difference between A/B/C is what fills the content slot (genome / LLM / nothing).

**Run C in BOTH phases.** (This revises an earlier "Phase 1 only" note.) Because Phase 1 filters to popular movies, the floor sits artificially close to A/B there — content features earn their lift on the long tail, which Phase 1 removes (see Stage 6 and the Decision Gate). The floor's most informative signal therefore only appears on the **full corpus**, so Model C must be retrained in Phase 2. Cost: $0 (no extraction, no compute cost).

```python
config = {
    'content_feature_source': 'llm',  # 'genome' | 'llm' | None
    ...
}
```

### Critical: identical training setup
Same:
- Train/val split (same random seed)
- Optimizer (Adam, same lr)
- Batch size, epochs, loss, negative sampling, evaluation protocol
- **Menon popularity correction `alpha = 0` for all models, both phases** — NO popularity debiasing anywhere in this experiment. This ablation is purely about genome tags vs. LLM features; introducing popularity correction would confound the comparison. Do not set `alpha=0.5` (the prod default) here.

**Only the content slot varies (genome / LLM / nothing).**

---

## Stage 6: Evaluation

### Quantitative metrics

| Metric | No content (Model C) | Genome (Model A) | LLM (Model B) | A−C | B−C |
|---|---|---|---|---|---|
| Hit@10 | ? | ? | ? | ? | ? |
| NDCG@10 | ? | ? | ? | ? | ? |
| MRR | ? | ? | ? | ? | ? |
| Recall@10 | ? | ? | ? | ? | ? |

Model C is the floor. The A−C and B−C columns are the content-feature lift — how much genome and LLM each add over no content slot. The headline comparison is still A vs. B, but the lift columns make it interpretable.

> **⚠️ Phase 1 selection bias — read before interpreting.** Phase 1 filters to popular movies (100–200+ ratings). Content features deliver their largest lift on long-tail / cold-start items, where collaborative filtering has little interaction data — exactly the items Phase 1 removes. So expect the **A−C and B−C lift columns to be compressed in Phase 1**: the floor (C) will sit unusually close to A and B. **The primary Phase 1 success criterion is whether Model B (LLM) matches or closely approaches Model A (genome) on this popular split — NOT the absolute lift over the floor.** The lift-over-floor story is told properly in Phase 2 on the full corpus, which includes the tail. Do not read a small Phase 1 lift as "content features don't matter."

### Long-tail split (Phase 2)
On the full corpus, additionally report metrics **restricted to long-tail movies** (low rating count). This is where content features are expected to matter most and where the genome-vs-LLM question is most consequential. Call it out explicitly — a model that ties on the head but wins on the tail is the interesting outcome.

### Qualitative comparison via canary users
3-5 canary user profiles. For each, top-10 from both models side by side. The interesting cases are where they disagree.

### Feature-level analysis
For 10 random movies, compare LLM feature vector to genome feature vector. Document where they agree and diverge.

---

## Stage 7: Streamlit Demo Extension

> **Deferred — gated on the Decision Gate and a positive result.** This is Tier 2 deploy work in [`repo_cleanup_plan.md`](repo_cleanup_plan.md). The experiment's metrics and canary comparisons do not depend on it; skip it entirely unless LLM features clearly justify deploying. The live demo runs on the existing frozen `serving/` artifacts and is unaffected by the experiment.

Add a fourth tab to the existing app:

**Tab: Feature Source Comparison**
- User builds watchlist or picks canary profile
- App shows top-10 from both models side by side
- Highlight movies that appear in only one of the two lists
- Brief explanation of the experiment

---

## Stage 8: Writeup

### `results/llm_vs_genome_ablation.md`

Structure:

1. **Hypothesis** — Can LLM-generated content features match human-curated genome tags?
2. **Setup** — Same architecture, same training, only content feature source differs; LLM schema derived from the top-discriminability genome tags so both measure the same axes
3. **Results** — Quantitative table + qualitative comparisons; **report eval with and without the factual-prestige group (5)** so its contribution is isolable
4. **Findings** — Honest interpretation
5. **Limitations** — Single dataset, single LLM, possible training-data contamination, **residual reception asymmetry** (pure-sentiment genome tags like `masterpiece` / `great acting` / `predictable` cannot be made objective and remain a genome-only advantage), **prestige-as-popularity leakage** (scraped box-office / IMDb-rating in group 5 are quasi-popularity signals that blur the content-vs-collaborative line, in tension with the `alpha=0` stance), and **Phase 1 popular-movie selection bias** (content lift is structurally suppressed on the popular split)
6. **What this means for production** — Where this pattern fits in real systems

2000-3000 words. Second-most-important deliverable after the code.

---

## Decision Gate Between Phases

Do NOT automatically proceed from Phase 1 to Phase 2. After Phase 1, **pause and ask the user to review**:

- The quantitative comparison table (all three models, reduced corpus)
- 3-5 canary user qualitative comparisons
- Any cross-group consistency warnings from the LLM extraction
- Total cost spent so far

The user decides whether the Phase 1 results justify the additional Phase 2 cost. If Phase 1 shows LLM features dramatically underperform genome tags (B clearly worse than A on the popular split), Phase 2 may not be worth running — the finding is clear enough already.

> **Do NOT gate on lift-over-floor.** Because Phase 1 is biased toward popular movies (Stage 6 warning), a *small* A−C / B−C lift is expected and is **not** evidence that content features don't matter — that signal is structurally suppressed until the full corpus. The gate is **B vs. A on the popular split**: if LLM matches/approaches genome there, that justifies Phase 2 regardless of how compressed the floor looks. The case where a weak floor lift *does* justify stopping is only when B also clearly trails A.

---

## Phase 2: Full Corpus (Only If Phase 1 Looks Good)

**Goal:** Validate the Phase 1 finding on the full MovieLens corpus, including the long tail that Phase 1 structurally excluded.

### Phase 2 work

1. **Scale up scraping** — scrape the remaining ~4-5k movies that weren't in Phase 1 (the long tail). Reuse the Phase 1 cache for movies already scraped.

2. **Scale up LLM extraction** — run grouped extraction on the remaining movies. Reuse the Phase 1 cache (per-movie, per-group responses are already saved).

3. **Train all three models fresh on the full corpus at `alpha=0`** — Model A (genome), Model B (LLM), Model C (no-content floor), identical hyperparameters, only the content slot differs. **Do NOT reuse the deployed prod checkpoint** — it was trained with `alpha=0.5` and would confound the comparison on popularity. Retraining is $0 compute.

4. **Compare A vs. B vs. C on the full corpus** — this is the real comparison, with the long tail present and popularity held constant across all three.

### Phase 2 success criteria

- All three models trained fresh on the full corpus at `alpha=0`
- Quantitative comparison A vs. B vs. C on the full corpus, popularity held constant
- **Long-tail split reported explicitly** — does LLM still match/beat genome on less popular movies, where content features matter most? This is the question Phase 1 could not answer.

### Phase 2 cost

- Incremental LLM extraction for the additional ~4-5k movies: **~$60-110 (Sonnet) / ~$35-65 (Haiku)** — model-dependent, see the consolidated Cost Budget.
- Total experiment cost (Phase 1 + Phase 2): **~$125-220 (Sonnet) / ~$75-130 (Haiku)**.

---

## Updated README

```markdown
## LLM-Generated Features Ablation

This repo contains THREE trained models on MovieLens 32M:
1. **Prod (genome tags)** — MovieLens's curated 1128-dimension genome features
2. **LLM (extracted features)** — ~145 dimensions of LLM-extracted structured content
   features from scraped IMDB/TMDB/Wikipedia content
3. **No content slot (floor baseline)** — same model with the content slot removed,
   measuring how much either content source adds at all

Identical architecture, identical training, only the content slot differs.

**Finding:** [Fill in after experiment]

See [results/llm_vs_genome_ablation.md](results/llm_vs_genome_ablation.md).
```

---

## Implementation Order (Strict)

### Phase 1 — Reduced Corpus Pilot

1. **Stage 0** — Filter corpus to ~4-5k movies, save `data/llm_experiment_movies_phase1.json`
2. **Stage 1 (test)** — Build scraper, scrape 10 movies
3. **Verify Stage 1** — Inspect 10 movies' content quality
4. **Stage 1 (Phase 1)** — Scrape the filtered ~4-5k movies (incl. objective prestige indicators)
5. **Stage 2 schema** — Derive the dimension list from `top_genome_tags_by_discriminability.csv` (top ~150, dedup synonyms, bucket into 6 groups); define 6 Pydantic schemas, draft 6 prompts
6. **Stage 2 (test)** — Run grouped extraction on 5 movies; **Sonnet-vs-Haiku bake-off**
7. **Verify Stage 2** — Calibration check, similarity check, no defaulting to 0.5
8. **Stage 2 (Phase 1)** — Run grouped extraction on the filtered ~4-5k movies
9. **Stage 3** — Merge groups, run cross-group consistency validation
10. **Stage 4** — Build feature tensor, similarity sanity check
11. **Verify Stage 4** — Confirm Toy Story 1/2/3 similar in LLM feature space
12. **Stage 5 (Phase 1)** — Train all three models (genome + LLM + no-content floor) on the reduced corpus
13. **Stage 6** — Run evaluation, generate results
14. **Stage 8** — Write up Phase 1 findings honestly

### Decision Gate

15. **Pause** — Present Phase 1 results to the user. The user decides whether to proceed to Phase 2.

### Phase 2 — Full Corpus (only if approved)

16. **Stage 1 (Phase 2)** — Scrape the remaining long-tail movies (reuse Phase 1 cache)
17. **Stage 2 (Phase 2)** — Extract features on the remaining movies (reuse Phase 1 cache)
18. **Stage 4 (Phase 2)** — Rebuild full-corpus feature tensor
19. **Stage 5 (Phase 2)** — Train all three models (A genome, B LLM, C floor) fresh on full corpus at `alpha=0`; no prod checkpoint reuse
20. **Stage 6 (Phase 2)** — Evaluate A vs. B vs. C, report the long-tail split explicitly
21. **Stage 8** — Finalize writeup
22. **Stage 7 (deferred)** — Streamlit comparison tab — only if the result justifies deploying (Tier 2 / decision gate)

Do not move to stage N+1 until stage N is verified. Do not cross the Decision Gate without user approval.

---

## Cost Budget

**Output tokens dominate.** Structured float extraction produces output-heavy calls, and at Claude Sonnet pricing output is $15/MTok vs $3/MTok input (5×). Input and output must be budgeted separately — a blended "~600 tokens/call" estimate hides the binding constraint.

### Phase 1 — reduced corpus (~4,500 movies × 6 groups = ~27,000 calls)

Worked at Sonnet pricing ($3/MTok in, $15/MTok out):

| Input regime | In tok/call | Out tok/call | Input $ | Output $ | **Total** |
|---|---|---|---|---|---|
| **TMDB-only / truncated (default)** | ~600 | ~150 | $48.60 | $60.75 | **~$109** |
| Full raw Wikipedia plots (DO NOT) | ~2,500 | ~150 | $202.50 | $60.75 | **~$263** |

**Even the lean default path (~$109) is already at/over the $100 Phase 1 target.** Two consequences:
- The Stage 1 input discipline (TMDB-first, truncate Wikipedia) is mandatory, not advisory — the raw-Wikipedia path is ~2.4× over.
- Output is the binding constraint and input-truncation alone can't fix it. The real output lever is the **model bake-off** (Stage 2): Haiku 4.5 output is ~$4/MTok vs Sonnet $15/MTok — if Haiku passes the calibration checks, Phase 1 output cost drops from ~$61 to ~$16 and the whole phase lands comfortably under $100. **The numbers above assume Sonnet; re-confirm with measured token counts after the bake-off.**

Note: these are API pay-as-you-go costs, billed against API credits — not a flat monthly subscription cap.

### Full budget

| Item | Estimated Cost |
|---|---|
| Web scraping (both phases) | $0 |
| LLM extraction — Phase 1 (~27k calls, Sonnet, lean input) | ~$109 (or ~$64 on Haiku) |
| LLM extraction — Phase 2 (incremental, remaining ~4-5k movies) | ~$60-110 (model-dependent) |
| Training compute (all models, both phases) | $0 |
| **Total (Phase 1 + Phase 2)** | **~$125-220 (Sonnet) / ~$75-130 (Haiku)** |

**Phase 1 hard cap: $100** — meeting it likely requires Haiku (confirmed via bake-off) and/or tighter input. Overall hard upper bound: $200. If approaching $175 across both phases mid-extraction, stop and revise.

---

## Critical Warnings

1. **Cache everything.** Per-group, per-movie LLM responses are cached individually. Re-calling for a movie+group combo that already has a cached response is a bug.

2. **Spot-check before scaling.** Validate 10-20 movies' output before running the full pipeline. The cost of running 56k LLM calls on a broken prompt is much higher than time validating.

3. **Structured output is non-negotiable.** Free-form responses will silently corrupt the feature tensor. Every call must enforce a JSON schema with float-range validation.

4. **Six groups, not one monolithic prompt.** A single 150-dimension prompt suffers from "lost in the middle" — LLM defaults late dimensions to 0.5. Grouped extraction is the mitigation.

5. **Cross-group consistency check is mandatory.** Contradictions between groups (movie being both "uplifting" and "devastating" at 0.9+) suggest extraction quality issues. Investigate before training.

6. **Identical training is the experiment.** If anything about the training setup differs between Model A and Model B, the comparison is meaningless. Lock down hyperparameters, random seeds, data splits before training either.

7. **The result is whatever the result is.** Do not retroactively modify LLM prompts to make features look better after seeing eval. That's p-hacking. If LLM loses, document it honestly.

8. **Watch for data leakage.** The LLM was trained on web content that may include MovieLens-related discussion. Acknowledge in writeup limitations rather than pretending it doesn't exist.

9. **Respect rate limits and terms of service.** TMDB and IMDB have specific limits. Don't get IP-banned mid-experiment.

10. **Structured features, not raw embeddings.** Dumping OpenAI's text-embedding-3 vectors as features is the lazy version. The experiment requires genome-comparable structured features.

---

## Success Criteria

The experiment is complete when:

1. All three models trained with identical hyperparameters, only the content slot differs
2. Quantitative comparison table filled in
3. Qualitative canary user comparison for 3-5 profiles
4. Cross-group consistency validation passed (or documented if not)
5. Streamlit demo updated with side-by-side tab
6. Writeup documents finding honestly
7. README updated to reflect the three-model nature

The result itself doesn't determine success — experimental rigor does.

---

## Out of Scope

- Replacing genome features entirely in prod
- Multi-LLM comparison
- LLM-generated user features
- LLM at serving time (offline only)
- Combining LLM + genome features in single model
- Replicating for Books or Steam recommenders
- Fine-tuning an LLM on movie content

Save for future experiments if initial result is interesting.
