# Phased Approach: Instructions for Claude Code

Modify the existing `llm_vs_genome_ablation_plan.md` to use a two-phase approach.

## Phase 1: Reduced Corpus Pilot (Do This First)

**Goal:** Validate the experimental approach cheaply before committing to full-corpus extraction.

### Changes from original plan

**Add a Stage 0 — Corpus Filtering** before Stage 1:
- Apply a higher rating count threshold to filter the MovieLens corpus down to ~4-5k movies
- Target: keep movies with at least 100-200 ratings (tune until you hit the target size)
- Save the filtered movie list to `data/llm_experiment_movies_phase1.json`
- Document the exact threshold and resulting movie count in CLAUDE.md

**Modify Stage 1 (Scraping)** to only scrape the filtered ~4-5k movies, not all 9,375.

**Modify Stage 2 (LLM Extraction)** to only run on the filtered ~4-5k movies.

**Modify Stage 5 (Train Models)** to retrain BOTH models on the reduced corpus:
- Model A (genome features) — retrain on the same filtered corpus
- Model B (LLM features) — train on the same filtered corpus
- Identical hyperparameters between A and B
- Do NOT compare against the existing full-corpus prod model — that's apples-to-oranges

**Update cost budget:**
- LLM extraction on 4-5k movies × 6 groups × ~600 tokens = ~$50-70
- Hard cap: $100 for Phase 1

### Phase 1 success criteria

After Phase 1, you should be able to answer:

1. Does the grouped extraction pipeline work without manual intervention?
2. Do the LLM features pass the similarity sanity checks (Toy Story 1/2/3 similar, etc)?
3. Does Model B (LLM features) train without issues?
4. Is the quantitative comparison meaningful — clear winner, clear loser, or roughly equivalent?
5. Do the canary user comparisons reveal interesting qualitative differences?

If all of these are yes, proceed to Phase 2. If any are no, fix the problem before scaling.

---

## Phase 2: Full Corpus (Only If Phase 1 Looks Good)

**Goal:** Validate the Phase 1 finding on the full MovieLens corpus and against the existing deployed prod model.

### Phase 2 work

1. **Scale up scraping** — scrape the remaining ~4-5k movies that weren't in Phase 1 (the long tail). Reuse Phase 1 cache for movies already scraped.

2. **Scale up LLM extraction** — run grouped extraction on the remaining movies. Reuse Phase 1 cache.

3. **Train Model B (full corpus)** — train LLM-feature model on the full 9,375-movie corpus using identical hyperparameters to the existing prod genome model.

4. **Compare against existing prod model directly** — at full corpus, the existing prod (genome) model is the right baseline. No need to retrain it.

### Phase 2 cost
- Incremental LLM extraction: ~$50-80 for the additional ~4-5k movies
- Total experiment cost (Phase 1 + Phase 2): ~$100-150

### Phase 2 success criteria
- LLM-feature model on full corpus trained successfully
- Quantitative comparison against existing prod (genome) model on full corpus
- Long-tail performance specifically called out — does LLM still match/beat on less popular movies?
- Streamlit demo updated to use the full-corpus models

---

## Decision Gate Between Phases

Do NOT automatically proceed from Phase 1 to Phase 2. After Phase 1, pause and ask the user to review:

- The quantitative comparison table
- 3-5 canary user qualitative comparisons
- Any cross-group consistency warnings from the LLM extraction
- Total cost spent so far

The user decides whether the Phase 1 results justify the additional Phase 2 cost. If Phase 1 shows LLM features dramatically underperform genome tags, Phase 2 may not be worth running — the finding is clear enough already.

---

## Implementation Order

1. Read the existing `llm_vs_genome_ablation_plan.md`
2. Update the plan to clearly separate Phase 1 and Phase 2 sections
3. Add the Stage 0 corpus filtering step at the start of Phase 1
4. Update cost estimates and hard caps to reflect the phased approach
5. Add the decision gate between phases
6. Save the updated plan as `llm_vs_genome_ablation_plan.md` (overwrite the original)

Do not start any actual implementation work (scraping, extraction, training) until the user confirms the updated plan is correct.
