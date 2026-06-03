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
│   ├── scrape.py                       ← web scraping per movie
│   ├── llm_extract.py                  ← grouped LLM feature extraction
│   ├── merge_extractions.py            ← combine grouped outputs per movie
│   ├── build_features.py               ← convert merged JSON → tensor
│   ├── prompts.py                      ← 6 grouped prompts as code (versioned)
│   ├── schemas.py                      ← Pydantic schemas for structured output
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
│       (parallel to existing genome_context_buffer)
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
- **Phase 2 — Full Corpus:** Only if Phase 1 looks good. Scale scraping/extraction to the full 9,375-movie corpus, train the LLM model on the full corpus, and compare directly against the existing deployed prod (genome) model.

The stages below are written for Phase 1 (reduced corpus). The "Phase 2: Full Corpus" section near the end describes the scale-up deltas. The Decision Gate between them is mandatory.

---

## Stage 0: Corpus Filtering (Phase 1)

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

**Per-movie output:** JSON file with all scraped fields.

### Practical guidelines
- Use TMDB API first — gets you 80% of what you need
- Cache aggressively — scrape ONCE per movie, never re-scrape
- Rate limit conservatively — TMDB is 40 req/10s; respect IMDB's robots.txt
- Skip movies where scraping fails — 90% coverage is fine
- Print summary at end: how many movies have full data, how many partial, how many failed

### Estimated cost & time
Zero cost. Phase 1: ~1-2 hours for the filtered ~4-5k movies if rate-limited properly. Phase 2: a similar amount for the remaining long tail (full corpus ~9,375).

---

## Stage 2: LLM Feature Extraction (Grouped + Structured)

> **Phase 1:** Run extraction only on the filtered ~4-5k movies. Cost: 4-5k movies × 6 groups × ~600 tokens ≈ **$50-70**. Hard cap for Phase 1: **$100**.

### Critical design decision: grouped prompts with structured output

A single prompt asking the LLM to extract 150+ dimensions in one pass suffers from a documented failure mode: the LLM gives careful values to the first 20-30 dimensions, then defaults to 0.5 or mirrors structure for the rest. This is the "lost in the middle" / response fatigue problem.

**Mitigation: split into 6 focused prompts of ~25-35 dimensions each, each enforced by structured output schema.**

### Six grouped extraction calls per movie

1. **THEMES** (~25 dims): redemption, revenge, love, betrayal, identity, family, friendship, sacrifice, survival, justice, ambition, isolation, coming_of_age, mortality, freedom, power, addiction, faith, war, immigrant_experience, [...]

2. **TONE** (~20 dims): dark, light, intense, contemplative, whimsical, gritty, romantic, comedic, suspenseful, melancholic, hopeful, nihilistic, satirical, earnest, [...]

3. **NARRATIVE_STRUCTURE** (~30 dims): linear, non_linear, ensemble_cast, single_protagonist, anthology, twist_ending, ambiguous_ending, character_study, plot_driven, dialogue_heavy, action_heavy, [...]

4. **VISUAL_STYLE** (~25 dims): stylized, naturalistic, animated, practical_effects, cgi_heavy, period_accurate, neon_aesthetic, gritty_realism, [...]

5. **AUDIENCE** (~20 dims): mainstream, arthouse, family_friendly, mature_themes, accessible, challenging, niche_appeal, [...]

6. **EMOTIONAL_PROFILE** (~25 dims): uplifting, devastating, thought_provoking, escapist, cathartic, unsettling, comforting, [...]

Total: ~145 dimensions across 6 calls per movie.

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

### LLM choice
**Claude Sonnet** or **GPT-4o**. Don't use smaller models (Haiku, GPT-4-mini) — quality difference matters and cost difference is small.

### Estimated cost
6 calls × 9,375 movies × ~600 tokens per call ≈ ~34M tokens. At Claude Sonnet pricing: **$100-150 total**. Within $200 hard cap, but flag if approaching $150.

### Estimated time
~56k LLM calls. With reasonable parallelism (10-20 concurrent): ~3-4 hours. Cache every response so re-runs are free.

### Validation pass before scaling
Before running on all 9,375 movies, spot-check 20 random movies:

- Within each group, are values calibrated (full 0-1 range, not all 0.5)?
- Do values match intuition (Mad Max should score high on "gritty" and "action_heavy")?
- Do similar movies produce similar features (Toy Story 1 and 2)?
- Do cross-group features avoid contradiction (not 0.9 "uplifting" AND 0.9 "devastating")?

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

Convert merged JSON files into a tensor matching the structure of `genome_context_buffer`.

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
**Phase 1:** Retrain on the filtered ~4-5k corpus (same split) so the comparison is corpus-matched. **Phase 2:** Use the existing deployed prod (genome) checkpoint directly — at full corpus it is the right baseline and does not need retraining.

### Model B — LLM features
Same architecture as Model A, but `genome_context_buffer` is replaced with `llm_features_v1.pt`. **Phase 1:** train on the reduced ~4-5k corpus. **Phase 2:** train on the full 9,375-movie corpus.

### Model C — No content slot (floor baseline)
Same architecture as A and B, but with the **genome/LLM slot removed entirely** — no genome buffer, no LLM buffer. This is the floor: it measures how much *any* content feature adds over the model's remaining signals.

**Critical — remove ONLY the slot under test.** Keep every other item feature identical to A and B: genre one-hot, tag vector, item ID embedding, year. Do NOT also strip genre/tag — that would make C a different model, not a matched control. The only difference between A/B/C is what fills the content slot (genome / LLM / nothing).

**Phase 1 only.** Model C is a floor baseline; once the floor is established on the reduced corpus there's no need to retrain it on the full corpus. Cost: $0 (no extraction).

```python
config = {
    'content_features': 'llm',  # or 'genome'
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

### Qualitative comparison via canary users
3-5 canary user profiles. For each, top-10 from both models side by side. The interesting cases are where they disagree.

### Feature-level analysis
For 10 random movies, compare LLM feature vector to genome feature vector. Document where they agree and diverge.

---

## Stage 7: Streamlit Demo Extension

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
2. **Setup** — Same architecture, same training, only content feature source differs
3. **Results** — Quantitative table + qualitative comparisons
4. **Findings** — Honest interpretation
5. **Limitations** — Single dataset, single LLM, possible training-data contamination
6. **What this means for production** — Where this pattern fits in real systems

2000-3000 words. Second-most-important deliverable after the code.

---

## Decision Gate Between Phases

Do NOT automatically proceed from Phase 1 to Phase 2. After Phase 1, **pause and ask the user to review**:

- The quantitative comparison table (all three models, reduced corpus)
- 3-5 canary user qualitative comparisons
- Any cross-group consistency warnings from the LLM extraction
- Total cost spent so far

The user decides whether the Phase 1 results justify the additional Phase 2 cost. If Phase 1 shows LLM features dramatically underperform genome tags, Phase 2 may not be worth running — the finding is clear enough already.

---

## Phase 2: Full Corpus (Only If Phase 1 Looks Good)

**Goal:** Validate the Phase 1 finding on the full MovieLens corpus and against the existing deployed prod model.

### Phase 2 work

1. **Scale up scraping** — scrape the remaining ~4-5k movies that weren't in Phase 1 (the long tail). Reuse the Phase 1 cache for movies already scraped.

2. **Scale up LLM extraction** — run grouped extraction on the remaining movies. Reuse the Phase 1 cache (per-movie, per-group responses are already saved).

3. **Train Model B (full corpus)** — train the LLM-feature model on the full 9,375-movie corpus using identical hyperparameters to the existing prod genome model.

4. **Compare against existing prod model directly** — at full corpus, the existing prod (genome) model is the right baseline. No need to retrain it.

### Phase 2 success criteria

- LLM-feature model on full corpus trained successfully
- Quantitative comparison against existing prod (genome) model on full corpus
- Long-tail performance specifically called out — does LLM still match/beat on less popular movies?
- Streamlit demo updated to use the full-corpus models

### Phase 2 cost

- Incremental LLM extraction: **~$50-80** for the additional ~4-5k movies
- Total experiment cost (Phase 1 + Phase 2): **~$100-150**

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
4. **Stage 1 (Phase 1)** — Scrape the filtered ~4-5k movies
5. **Stage 2 setup** — Define 6 schemas, draft 6 prompts
6. **Stage 2 (test)** — Run grouped extraction on 5 movies
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
19. **Stage 5 (Phase 2)** — Train Model B on full corpus; use existing prod genome model as baseline
20. **Stage 6 (Phase 2)** — Evaluate vs. prod, call out long-tail performance
21. **Stage 7** — Add Streamlit tab (full-corpus models)
22. **Stage 8** — Finalize writeup

Do not move to stage N+1 until stage N is verified. Do not cross the Decision Gate without user approval.

---

## Cost Budget

| Item | Estimated Cost |
|---|---|
| Web scraping (both phases) | $0 |
| LLM extraction — Phase 1 (6 groups × ~4-5k movies) | $50-70 |
| LLM extraction — Phase 2 (incremental, remaining ~4-5k movies) | $50-80 |
| Training compute | $0 |
| **Total (Phase 1 + Phase 2)** | **~$100-150** |

**Phase 1 hard cap: $100.** Overall hard upper bound: $200. If approaching $175 across both phases mid-extraction, stop and revise prompts.

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
