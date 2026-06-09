# LLM-Generated Item Features vs Human-Curated Genome Tags — Experiment Record

> **What this is.** The reproduction record for the completed LLM-vs-genome ablation — the
> durable lab notebook, not the original build plan. The narrative writeup is
> [`docs/llm_vs_genome_ablation.md`](../llm_vs_genome_ablation.md); grounded genome construction
> facts are in [`docs/movielens_genome_features_info.md`](../movielens_genome_features_info.md).
> This file keeps what those two don't: the authoritative model→checkpoint map, the exact eval
> methodology, the full result tables (**including Phase 1, which the writeup omits**), and the cost
> budget. The original stage-by-stage build plan, two-phase decision gate, implementation order,
> success criteria, and critical warnings were removed once the experiment shipped (2026-06-08) —
> recover them from git history (`git log --follow -p` on this file) if ever needed.

## Goal

Test whether LLM-generated item features can match or beat the human-curated MovieLens genome tags as content signal in a two-tower recommendation model.

**Research question:** Can modern LLMs replace expensive human curation for content features in recommendation systems?

**Deliverable:** Trained two-tower models on MovieLens 32M — one using genome tags (then-prod), one using LLM-generated features from scraped web content, and a no-content-slot floor baseline. Quantified ablation of which produces better recommendations, with the floor baseline measuring how much either content source adds at all. (A 4th arm, D = genome + LLM, was added later as a secondary "does *more* help?" check.)

This was a **parallel experiment** within the existing Movie Recommender repo, not a new repo or project.

## Setup — what makes it a fair test

Four arms, **identical** two-tower architecture, hyperparameters, train/val split, loss (full softmax), and 150k training steps. The **only** difference is what fills the content slot:

- **A — genome:** 1,128-dim genome scores (`FEATURE_TOWERS=genome`)
- **B — LLM:** 132 LLM-extracted dims (`FEATURE_TOWERS=llm`)
- **C — none:** content slot removed — the floor (`FEATURE_TOWERS=none`)
- **D — both:** genome + LLM, two parallel content sub-tower families concatenated pre-projection (`FEATURE_TOWERS=both`)

**Menon popularity-correction α = 0 for every arm, both phases** — no popularity debiasing anywhere, so it cannot confound the genome-vs-LLM comparison (prod's α=0.5 is deliberately *not* used here). The LLM schema is **derived from the top-discriminability genome tags** (`data/top_genome_tags_by_discriminability.csv`), so both spaces sit on the **same 132 axes** — each LLM dim records its source genome tag(s) in `data/llm_schema_dimensions.json`. Config resolves towers from state_dict weight keys, not filenames (`src/checkpoint.py:load_checkpoint`).

Two corpora: **Phase 1** pilot = 4,461 popular movies (> 1,000 raw ratings); **Phase 2** = full 9,375-movie corpus (> 200 ratings), which adds the long tail where content features earn their keep.

## Pipeline & artifacts

Code lives in `llm_features/` (run end-to-end once, then cached per movie+group):
`filter_corpus.py` (Phase 1 list) → `scrape.py` (TMDB-first: overview / tagline / genres / cast / director / writers / keywords, + Wikipedia plot + **factual** prestige indicators: Oscar wins/noms, Criterion status, box-office scale) → `derive_schema.py` (top genome tags → deduped, 6-bucketed dimension list) → `cc_extract.py` / `batch_extract.py` (six **grouped structured-output** calls per movie: themes / tone / setting / provenance / factual-reception / factual-visual) → `merge_extractions.py` → `build_features.py` → tensor.

- **Extractor:** Claude Sonnet via Claude Code (see "LLM choice" below).
- **Feature tensor:** `data/llm_features_claude-code-sonnet_v1.pt` — `9376 × 132`, 9,360 non-zero rows (9,366 of 9,375 corpus movies scraped successfully).
- **Scrape discipline:** store raw on disk, truncate **only at feed time** (`format_for_prompt`) — keeps the durable cache policy-free and the input token bill down (see Cost Budget; raw Wikipedia plots ≈5× the input cost).
- **Grouping rationale:** a single 130-dim prompt hits "lost in the middle" and defaults late dims to 0.5; six focused ~20–30-dim calls don't. Visual + prestige groups are **factual-only** (animation / b&w / Oscar-winner — yes; "visually stunning" from a synopsis — no). Reception is its own group so it is separately ablatable.
- **Validation gates (before training):** per-group 0–1 calibration (no defaulting to 0.5), known-similar similarity (Toy Story 1/2/3, Godfather I/II), and a cross-group consistency check (flag contradictory profiles, e.g. uplifting+devastating).
- **Analysis / figures:** `feature_level_analysis.py` (the 132 shared-axis correlations) and `make_figures.py` (writeup figures → `results/figures/`, committed copies in `docs/figures/`, correlation cache `feature_agreement_r.json`).

### LLM choice — Sonnet via Claude Code (bake-off dropped)
**Decision (2026-06): extraction was run with Claude Sonnet through Claude Code, on a flat-rate Max subscription — marginal API cost ~$0.** The originally-planned Sonnet-vs-Haiku bake-off is **dropped.** Its entire rationale was pay-as-you-go cost control (output dominates the bill — Sonnet output $15/MTok vs Haiku ~$4/MTok — so a cheaper model that passed calibration would have cut the bill ~4×). That premise no longer holds: the features were generated under a subscription already in hand, at no marginal API cost, so there is nothing to optimize by switching models. Running the bake-off would *itself* cost $100+ in API credits to answer a **secondary** question ("could a smaller model also do it?") that the core LLM-vs-genome comparison does not need.

**Consequence for the writeup (Limitations).** This is a **single-LLM** study — Claude Sonnet via Claude Code → `data/llm_features_claude-code-sonnet_v1.pt`. The paper can claim *Sonnet-class extraction matches genome*; it **cannot** claim a smaller/cheaper model would. State the cost honestly as "amortized under an existing flat-rate coding subscription, ~$0 marginal," **not** as a per-call API figure. The pay-as-you-go pricing analysis is retained in the Cost Budget only as a *reproduction estimate* for anyone without such a subscription.

**Prompt caching:** the 6 group prompts are static except the per-movie content block. Caching the static prefix cuts *input* cost if movies are batched by group within the 5-min cache TTL. Worth doing — but note it does not touch the dominant *output* cost, so it can't rescue the budget on its own.

## Canonical model → checkpoint map (authoritative)

All four checkpoints (and their `eval_results/`/`canary_results/` files) were **renamed on 2026-06-07 to the current explicit-tower-families naming scheme** (commit `7e5b34a`), so the filenames below are now self-consistent. A/B/C were originally trained under a legacy scheme (`best_softmax_v2[_llm|_nocontent]_…`); see "History" below. This table is the single source of truth for which checkpoint is which model; the scattered per-table references elsewhere all point here.

**Phase 2 — full corpus, α=0, 150k steps, val-MRR selection, trained 2026-06-07:**

| Model | Content slot | `FEATURE_TOWERS` | Best checkpoint (`saved_models/`) | MRR\* |
|---|---|---|---|---|
| **A** | genome tags | `genome` | `best_softmax_genome_tags_popularity_alpha_0_20260607_101027.pth` | 0.1146 |
| **B** | LLM features | `llm` | `best_softmax_llm_features_popularity_alpha_0_20260607_105646.pth` | 0.1157 |
| **C** | none (floor) | `none` | `best_softmax_popularity_alpha_0_20260607_112755.pth` | 0.1143 |
| **D** | genome + LLM | `both` | `best_softmax_genome_tags_llm_features_popularity_alpha_0_20260607_131924.pth` | 0.1154 |

\*Whole-corpus MRR, canonical eval (all 19,134 val users, n=382,138; matches the Phase 2 result tables below). Per-model `eval_results/` and `canary_results/` files share the same stem.

**Safe-rename notes:** `src/checkpoint.py:load_checkpoint` resolves towers from the *state_dict weight keys*, not the filename, so renaming a `.pth` is safe. The one caveat: A and B are "content-era" checkpoints (generic `item_content_tower` keys) whose genome-vs-llm identity is read from the `_config.json` **sidecar** (legacy `content_feature_source` key) — the sidecar must keep the same stem, so the rename moved `.pth` + `_config.json` together. A/B/C sidecar JSON still contains the legacy `content_feature_source` key internally (only the filenames changed); the loader reads both keys, so this is harmless.

**History (why git/older artifacts show different names):**
- **Model A's old filename had no "genome".** Pre-rename, genome was the *default* content source and left unmarked (`best_softmax_v2_…`); only non-default arms got a modifier (`_llm`, `_nocontent`). The `_v2` token and the unmarked-genome default are now gone — names compose tower families (`genome_tags`, `llm_features`) like D always did.
- Old → new: A `best_softmax_v2_…101027` → `best_softmax_genome_tags_…101027`; B `best_softmax_v2_llm_…105646` → `best_softmax_llm_features_…105646`; C `best_softmax_v2_nocontent_…112755` → `best_softmax_…112755`. D (`…_genome_tags_llm_features_…131924`) was already correct.

**Phase 1 — reduced corpus (`_phase1` suffix), α=0:** see the Phase 1 checkpoint list in the results below (separate val-loss vs val-MRR selection sets). Those Phase 1 checkpoints were **not** renamed (still `best_softmax_v2[_llm|_nocontent]_…_phase1_…`).

## Eval methodology & results

Rollback protocol throughout: for each held-out val user, context = history[0..j-1], target = history[j], up to 20 chronological positions per user. Recall@10 = Hit@10 (one held-out target per example).

**Phase 1 — reduced corpus (4,461 movies, > 1,000 raw ratings; α=0; rollback protocol, n=99,846 over 5,000 val users, seed 42).**

> **Checkpoint-selection criterion was changed mid-experiment — both result sets are reported below.** Originally `best_path` was saved at **minimum validation softmax cross-entropy** (a ranking *surrogate*). It is now saved at **maximum validation MRR** (the reported metric), computed on raw dot products over the same 8,192-example val subset — see `_val_ranking_metrics` in `src/train.py`. **Future runs and the paper use val-MRR selection (Table 2); Table 1 is retained because the change is itself a finding** (see "Selection-criterion effect"). NDCG@10 / Recall@10 included; Recall@10 = Hit@10 here (rollback holds out a single target per example, so they coincide).

**Table 1 — val-loss (min CE) checkpoint selection**

| Metric | No content (C) | Genome (A) | LLM (B) |
|---|---|---|---|
| Hit@1 | 0.0575 | 0.0586 | 0.0580 |
| Hit@5 | 0.1542 | 0.1558 | 0.1556 |
| Hit@10 | 0.2232 | 0.2250 | 0.2254 |
| Hit@20 | 0.3151 | 0.3168 | 0.3166 |
| Hit@50 | 0.4693 | 0.4708 | 0.4719 |
| NDCG@10 | 0.1290 | 0.1303 | 0.1301 |
| MRR | 0.1150 | **0.1161** | 0.1157 |

**Table 2 — val-MRR (max MRR) checkpoint selection — canonical**

| Metric | No content (C) | Genome (A) | LLM (B) | A−C | B−C |
|---|---|---|---|---|---|
| Hit@1 | 0.0580 | 0.0596 | **0.0612** | +0.0016 | +0.0032 |
| Hit@5 | 0.1547 | 0.1578 | **0.1600** | +0.0031 | +0.0053 |
| Hit@10 | 0.2250 | 0.2277 | **0.2284** | +0.0027 (+1.2%) | +0.0034 (+1.5%) |
| Hit@20 | 0.3149 | **0.3199** | 0.3195 | +0.0050 | +0.0046 |
| Hit@50 | 0.4682 | **0.4763** | 0.4744 | +0.0081 | +0.0062 |
| NDCG@10 | 0.1300 | 0.1322 | **0.1334** | +0.0022 (+1.7%) | +0.0034 (+2.6%) |
| MRR | 0.1157 | 0.1178 | **0.1192** | +0.0021 (+1.8%) | +0.0035 (+3.0%) |

**Selection-criterion effect (MRR, val-loss → val-MRR):** C 0.1150 → 0.1157 (+0.6%); A 0.1161 → 0.1178 (+1.5%); B 0.1157 → **0.1192 (+3.0%)**. The gain is *non-uniform* — val-loss differentially penalized the LLM model (its min-CE checkpoint sat well off its max-MRR checkpoint). **The criterion flips the headline verdict:** under val-loss, A > B by +0.0004 MRR (genome nominally ahead, within noise); under val-MRR, **B > A by +0.0014 MRR (+1.2%)**, with B leading on every metric except Hit@20 (tie). Lesson for the paper: model-selection metric must match the evaluation metric, or the surrogate can reverse a close comparison.

**Phase 1 verdict:** With correct (val-MRR) selection, **LLM features match and slightly exceed genome** on the popular split (B 0.1192 vs A 0.1178 MRR), and the content-feature lift over the no-content floor is real for both (A−C +1.8%, B−C +3.0% MRR). The Phase 1 success criterion (B matches/approaches A) is **exceeded**.

Checkpoints (all α=0, phase1, content slot is the only difference):
- **val-MRR (Table 2):** A = `best_softmax_v2_popularity_alpha_0_phase1_20260606_114000.pth`, B = `best_softmax_v2_llm_popularity_alpha_0_phase1_20260606_131452.pth`, C = `best_softmax_v2_nocontent_popularity_alpha_0_phase1_20260606_143158.pth`
- **val-loss (Table 1):** A = `…_20260604_210405.pth`, B = `…_llm_…_20260605_213508.pth`, C = `…_nocontent_…_20260604_214306.pth`
- Raw per-K outputs (K up to 250) under `eval_results/`. Recorded 2026-06-06.

Model C is the floor. The A−C and B−C columns are the content-feature lift — how much genome and LLM each add over no content slot. The headline comparison is A vs. B; the lift columns make it interpretable.

**Why the genome lift (A−C ≈ +1%) is small here — and expected.** This is measured on the **reduced Phase 1 corpus**, which keeps only popular head movies (> 1,000 ratings). For those items collaborative filtering already has abundant interaction signal to learn a good item embedding, so the content slot adds little on top — the ID-embedding pools carry most of the weight. Content features earn their keep where interactions are **sparse**: long-tail / cold-start items the model has barely seen, where there is little CF signal to leverage and the content vector is most of what distinguishes the item. Phase 1 structurally removes exactly those items, so the floor (C) sits unusually close to A. The lift is positive and consistent across every K (genome does help), just compressed; the real lift-over-floor story is told on the **full corpus in Phase 2**, where the tail is present.

**Phase 2 — full corpus (9,375 movies; α=0; rollback protocol; val-MRR checkpoint selection). Recorded 2026-06-07.**

> **Eval methodology (Phase 2).** The canonical run uses **all 19,134 val users** (`EVAL_N_USERS=19134`, vs the default 5,000) → **n=382,138** rollback examples, giving the long-tail tiers ~3.8× more signal. Whole-corpus numbers are ~1% lower than a 5,000-user run but the C < A < B ordering is identical. Popularity tiers below are by the *target* movie's **raw `ratings.csv` count** — the same basis that defines the corpus (`> 200`) and the Phase 1 threshold (`> 1000`); counts cached at `data/corpus_raw_rating_counts.npy`. The **HEAD** tier (`> 1000`) is byte-for-byte the Phase 1 corpus (4,461 movies; verified zero symmetric difference vs `data/llm_experiment_movies_phase1.json`), so **TAIL** (`≤ 1000`) is exactly the long tail Phase 1 excluded. **Q1–Q4** are equal-movie-count population quartiles (Q1 rarest → Q4 most popular). Implemented in `src/offline_eval.py` (`_build_tiers`, `_corpus_raw_rating_counts`); full per-K outputs for every tier under `eval_results/`.

| Metric | No content (C) | Genome (A) | LLM (B) | A−C | B−C |
|---|---|---|---|---|---|
| Hit@1 | 0.0577 | 0.0576 | **0.0585** | −0.0001 | +0.0008 |
| Hit@5 | 0.1536 | 0.1538 | **0.1552** | +0.0002 | +0.0016 |
| Hit@10 | 0.2213 | 0.2229 | **0.2240** | +0.0016 | +0.0027 |
| Hit@20 | 0.3101 | 0.3131 | **0.3140** | +0.0030 | +0.0039 |
| Hit@50 | 0.4611 | 0.4642 | **0.4656** | +0.0031 | +0.0045 |
| NDCG@10 | 0.1283 | 0.1288 | **0.1300** | +0.0005 | +0.0017 |
| MRR | 0.1143 | 0.1146 | **0.1157** | +0.0003 | +0.0014 |

**Phase 2 whole-corpus verdict:** **B (LLM) leads every metric** — B−A = +0.0011 MRR (+0.97%), B−C = +0.0014 (+1.2%), A−C = +0.0003 (+0.3%). The Phase 1 finding (LLM matches/slightly beats genome) **holds and firms up with the long tail present**: B now leads on *every* metric (Phase 1 had a Hit@20 tie). The aggregate lift-over-floor is small because the rollback target distribution is popularity-skewed (Q4 popular movies are 90% of examples) — the lift lives in the long-tail split below, not the aggregate.

Checkpoints (all α=0, full corpus, content slot is the only difference; trained 2026-06-07):
- A genome = `best_softmax_genome_tags_popularity_alpha_0_20260607_101027.pth`
- B llm = `best_softmax_llm_features_popularity_alpha_0_20260607_105646.pth`
- C nocontent = `best_softmax_popularity_alpha_0_20260607_112755.pth`

### Long-tail split (Phase 2)

On the full corpus we report metrics **restricted by the target movie's popularity tier** — this is where content features matter most and where the genome-vs-LLM question is most consequential.

**MRR by tier** (example count in parens; all three models share the same examples):

| Tier (n) | C | A | B | A−C | B−C | B−A |
|---|---|---|---|---|---|---|
| Whole corpus (382,138) | 0.1143 | 0.1146 | **0.1157** | +0.0003 | +0.0014 | +0.0011 |
| HEAD > 1k (369,486) | 0.1181 | 0.1184 | **0.1195** | +0.0003 | +0.0014 | +0.0011 |
| Q4 popular (343,906) | 0.1259 | 0.1260 | **0.1273** | +0.0001 | +0.0014 | +0.0013 |
| Q3 mid (26,923) | 0.0129 | **0.0148** | 0.0144 | +0.0019 | +0.0015 | −0.0004 |
| Q2 mid (8,049) | 0.0032 | **0.0038** | 0.0037 | +0.0006 | +0.0005 | −0.0001 |
| Q1 rarest (3,260) | 0.0012 | 0.0014 | 0.0014 | +0.0002 | +0.0002 | ±0.0000 |
| **TAIL ≤ 1k (12,652)** | 0.0028 | **0.0033** | 0.0032 | +0.0005 | +0.0004 | −0.0001 |

**Tail-tier recall** (Hit@50 and Hit@250 — tail tiers have few top-rank hits, so deeper-K recall carries more signal than MRR there):

| Tier | C@50 | A@50 | B@50 | C@250 | A@250 | B@250 |
|---|---|---|---|---|---|---|
| Q3 mid | 0.1134 | **0.1251** | 0.1223 | 0.3420 | **0.3608** | 0.3529 |
| Q2 mid | 0.0226 | 0.0287 | **0.0297** | 0.1442 | 0.1536 | **0.1568** |
| Q1 rarest | 0.0034 | **0.0061** | 0.0052 | 0.0463 | **0.0604** | 0.0580 |
| TAIL ≤ 1k | 0.0192 | **0.0249** | 0.0241 | 0.1243 | 0.1368 | **0.1378** |

**Findings:**

1. **Lift-over-floor thesis — confirmed on solid n.** Content's advantage over the no-content floor is ~0% on the popular head and grows steeply toward the tail. Hit@250 A−C: Q4 +0.0010 → Q3 +0.0188 → Q1 +0.0141 → TAIL +0.0125; relative MRR lift on TAIL ≈ +18% (A−C) / +14% (B−C) vs ~0.1% on Q4. Both content sources earn their keep exactly where collaborative signal is sparse — the story Phase 1 structurally could not tell.

2. **B beats A overall, driven entirely by the popular head.** Q4 (90% of examples) carries B−A ≈ +0.0013 MRR, stable across K. The whole-corpus and HEAD numbers are essentially the Q4 result.

3. **The key tail result: LLM does NOT collapse on rare movies — it matches genome.** On the deep tail (Q1, TAIL) A and B are statistically tied (MRR gaps ≤ 0.0001; Hit@250 splits favor each once: TAIL → B, Q1 → A). Even where content matters most, LLM features hold even with human-curated genome. This is the consequential, positive result for "can LLMs replace human curation," and it is the question Phase 1 could not answer.

4. **Genome keeps a small, consistent edge in the *mid*-tail (Q3, n=26,923).** Genome leads B on MRR (0.0148 vs 0.0144), Hit@50 (0.1251 vs 0.1223) and Hit@250 (0.3608 vs 0.3529) — ~2–5% relative, the one place the A > B gap is consistent across K rather than noise. Q3 spans 907–2,857 ratings (straddling the Phase 1 boundary), i.e. "moderately popular," not cold-start.

**Phase 2 verdict (paper headline):** LLM-extracted features **match human-curated genome on the long tail and slightly beat it overall**; genome retains only a marginal mid-tail (Q3) advantage. The "expensive human curation can be replaced by LLM extraction" thesis holds — including under the long-tail stress test Phase 1 structurally could not run. Caveat for the writeup: even at n=382,138 the deep-tail tiers are small (Q1 n=3,260; TAIL n=12,652) because the rollback target distribution is popularity-skewed (Q4 = 90% of examples), so deep-tail A-vs-B differences ≤ 0.0001 MRR should be read as ties, not rankings.

### Qualitative comparison via canary users

**Phase 2, full corpus.** Top-10 from genome (A) vs LLM (B) for all ~19 canary personas (`ts_max` bin), saved to `canary_results/best_softmax_genome_tags_popularity_alpha_0_20260607_101027.txt` (A) and `…_llm_features_…_20260607_105646.txt` (B). The 5 most illustrative disagreements:

| Persona (liked) | Genome (A) leans | LLM (B) leans |
|---|---|---|
| **Sci-Fi** (2001, Solaris, Contact) | cerebral/arthouse SF — Brazil, Gattaca, Forbidden Planet, A.I. | popcorn/blockbuster SF — Fifth Element, T2, Total Recall, Jurassic Park |
| **Crime** (Sicario, Narc, Am. Gangster) | drifts to finance/thriller — Big Short, Margin Call, Fight Club | **nails modern gritty crime** — Sicario: Day of the Soldado, Hell or High Water, End of Watch |
| **Western** (True Grit, spaghetti westerns) | **tight western canon** — Searchers, Liberty Valance, Rio Bravo, Outlaw Josey Wales | drifts to war epics + off-genre — Patton, Dirty Dozen, Braveheart, Batman, Silence of the Lambs |
| **Arthouse** (The Lobster, Antichrist) | **pure slow-burn/world arthouse** — Stalker, In the Mood for Love, Chungking Express, Blue Velvet | popular "smart prestige" — Fight Club, Usual Suspects, American Beauty, Spirited Away |
| **Horror** (TCM '03, Wrong Turn, Emily Rose) | 90s Scream-era slashers — Scream 2/3, Ring, Saw, Ghost Ship | **era-faithful 2000s gore** — Saw II/IV/V/VI, House of Wax, TCM: The Beginning |

**Finding — the two content sources give the model different "personalities":**
- **Genome (A) → niche sub-genre / canon purity.** Cleaner Western, Arthouse, cerebral Sci-Fi, Anime (B leaks *Dark Knight*/*Usual Suspects* into Anime), and the eclectic "Nick's" taste (A surfaces Hot Fuzz / Shaun of the Dead / Children of Men; B defaults to Harry Potter / Star Wars).
- **LLM (B) → era- and modern-subgenre matching** (2000s-gore Horror, 2010s gritty Crime) but **drifts to popular blockbusters more readily** on niche genres (Godfather II/III into Action; war epics into Western).

This complements the quantitative result: B's edge is on the popular/modern head (era-matching pays off there), genome holds the niche tail-canon better — consistent with B winning overall but genome keeping a mid-tail edge. Both show the expected **α=0 popularity drift** on the hardest drift-test personas (WW2, Fantasy); B also re-surfaces an Indian film (*Rang De Basanti*) for WW2 — the documented early-checkpoint failure mode.

> **Seed-sensitivity caveat.** The canary disagreements are **qualitative color, not a headline finding** — they depend on the chosen seed movies per persona. A later deployment-tuning pass found the original Horror seeds (Blair Witch / TCM '03 / Emily Rose / Wrong Turn) were a weak, incoherent taste profile that *manufactured* apparent popularity drift; strengthening them with acclaimed 2000s horror (28 Days Later, The Descent, Dawn of the Dead, Session 9, [REC]) made every arm recommend 100% on-genre. The quantitative tier metrics, not the canary, carry the conclusion.

### Feature-level analysis

Because the LLM schema was derived from the top-discriminability genome tags, every LLM dim records its source genome tag(s) (`data/llm_schema_dimensions.json`) — so the two spaces line up on the *same axes*. Reproducible via `llm_features/feature_level_analysis.py`.

**Corpus-level shared-axis agreement.** For each of the 132 LLM dims, Pearson correlation across all 9,375 movies between the LLM score and its mapped genome tag score(s):

- **mean r = 0.598, median 0.608; 99/132 dims at r ≥ 0.5, none below 0.1.** Strong evidence the LLM and genome measure the same axes — validating the design intent.
- **By group:** visual **0.70**, setting **0.68**, provenance **0.64** agree highest; themes 0.56, tone 0.56; **reception lowest at 0.42.**
- **Best axes (factual/objective):** vampires 0.94, documentary 0.89, animated 0.88, anime 0.88, western_frontier 0.86, world_war_ii 0.86, time_travel 0.85, sequel, musical, aliens, biographical, based_on_true_story.
- **Worst axes (subjective / crowd-sentiment):** imdb_top_250 **0.16**, criterion **0.18**, palme_dor 0.27, oscar_technical 0.33 (all reception/prestige), plus subjective tone — weird 0.36, enigmatic 0.42, nostalgic 0.38 — and abstract themes — redemption 0.38, greed 0.38, social_commentary 0.38, nonlinear 0.40.

**Per-movie (10 spot-checks).** Genome and LLM **agree strongly on each film's core identity** — e.g. *Godfather* (both: gangster/crime/mafia + oscar/imdb-top-250), *Saw* (both: horror/serial-killer/twist-ending/gory), *Schindler's List* (both: holocaust/WWII/based-on-true-story/oscar), *Spirited Away* (both: anime/animated/japan/fairy-tale), *Sicario* (both: tense/crime/violent/dark). They **diverge** in three consistent ways:
1. **Genome carries crowd-prestige & auteur signal the LLM structurally lacks** — "masterpiece", "imdb top 250", and director/composer names (lynch, miyazaki, coppola, ennio morricone). The LLM proxies prestige with its own oscar_winner/classic/afi_recognized dims, but those agree only weakly with genome's crowd versions (imdb_top_250 r=0.16, criterion 0.18).
2. **Genome has finer niche sub-genre granularity** — *Good, the Bad and the Ugly*: genome "spaghetti western" + "ennio morricone" + "civil war" vs the LLM's coarser "western_frontier". This finer aesthetic vocabulary is exactly why genome nails the Western / Arthouse canon in the canary.
3. **LLM contributes clean plot-derived facts** genome buries or omits — *2001* "artificial_intelligence"/"existentialism", *Die Hard* "based_on_book"/"eighties", *Sicario* "hitman"/"conspiracy"/"corruption".

**Synthesis (ties the experiment together).** The spaces are strongly aligned on factual/genre axes (mean r 0.60) → both place a movie in the right broad genre, which is why B matches A on the bulk metrics. Genome's advantage is concentrated in the **low-agreement axes** — subjective aesthetics, niche sub-genre granularity, and crowd-prestige — precisely where it wins the niche-canon canary personas and holds its mid-tail edge. The LLM, in turn, adds accurate plot facts and excels at era / modern-subgenre matching. This is the mechanistic explanation for the headline result: **LLM extraction reproduces nearly all of genome's content signal on the axes an LLM can reach from text, and the residual genome-only advantage is the crowd-sentiment / fine-aesthetic slice flagged as the validity threat** (the reception/prestige asymmetry — see the writeup's Limitations).

### Model D — genome + LLM combined (secondary arm; out of core scope)

> Combining both content sources in one model is a **non-goal** of the core genome-vs-LLM comparison (it answers a different question — "does *more* help?"). Run as a 4th arm with the same data so the answer is on record; kept as a marked side-section (§8) in the writeup. Model D adds the LLM-feature sub-towers *alongside* the genome-tag sub-towers (two parallel families, concatenated pre-projection) — `FEATURE_TOWERS=both`. Checkpoint `best_softmax_genome_tags_llm_features_popularity_alpha_0_20260607_131924.pth` (α=0, full corpus). Same eval protocol (n=382,138, all 19,134 val users); A/B/C re-run with the current code for an apples-to-apples 4-arm set.

**Whole-corpus (n=382,138):**

| Metric | C (none) | A (genome) | B (llm) | **D (genome+llm)** | D−A | D−B |
|---|---|---|---|---|---|---|
| Hit@10 | 0.2213 | 0.2229 | 0.2240 | **0.2243** | +0.0014 | +0.0003 |
| Hit@50 | 0.4611 | 0.4642 | **0.4656** | 0.4655 | +0.0013 | −0.0001 |
| NDCG@10 | 0.1283 | 0.1288 | **0.1300** | 0.1298 | +0.0010 | −0.0002 |
| MRR | 0.1143 | 0.1146 | **0.1157** | 0.1154 | +0.0008 | −0.0003 |

**Deep-tail recall (Hit@250) — the only place D adds anything:**

| Tier | C | A | B | **D** | best |
|---|---|---|---|---|---|
| Q3 mid | 0.3420 | 0.3608 | 0.3529 | **0.3615** | D |
| Q2 mid | 0.1442 | 0.1536 | 0.1568 | **0.1577** | D |
| Q1 rarest | 0.0463 | **0.0604** | 0.0580 | 0.0592 | A |
| TAIL ≤1k | 0.1243 | 0.1368 | 0.1378 | **0.1390** | D |

**Canary (single-run, qualitative; A/B/D side-by-side in `canary_results/`):** D is *strong* on Sci-Fi (Metropolis, Brazil, Forbidden Planet + Alien, Terminator) and Arthouse (Stalker, Blue Velvet, **Twin Peaks: FWWM**, Solaris), decent on Crime (Scarface, RocknRolla) and Horror (less Saw-sequel spam than B) — but **regresses on Western** (2 westerns + Jaws/Alien/Life of Brian; genome-A had 5). It does not cleanly fuse the two; the second tower can dilute genome's niche pull.

**Finding (does combining help?): no clear additive benefit.** On top-rank metrics D ≈ B (the better single source) and does not beat it — the two sources are largely redundant (consistent with the r=0.60 feature agreement: little independent signal to stack). The lone gain is a small deep-tail recall bump (Hit@250 best in 3/4 tail tiers). At α=0, D is **not** a clear upgrade over B or A, and its Western canary regression is a real demo concern, so D does not enter a prod comparison with a quantitative edge. (The shipped prod is a separate α=0.5 fine-tune of the D architecture — a portfolio-motivated swap, verified on par with the prior genome-only prod, not a metrics upgrade; see CLAUDE.md "Current State".)

## Cost budget (reproduction estimate)

**Actual cost (as run): ~$0 marginal.** Extraction was performed with Claude Sonnet through Claude Code on a flat-rate Max subscription, so full-corpus feature generation (`data/llm_features_claude-code-sonnet_v1.pt`) incurred **no marginal API charges** — it was amortized under a subscription already in hand. Web scraping and training compute are $0 too. The pay-as-you-go analysis below is retained only as a **reproduction estimate** for anyone extracting via the metered API instead (and as the original justification for the now-dropped model bake-off). Report the cost in the writeup as subscription-amortized / ~$0 marginal — *not* as a per-call API figure.

**Compute actually consumed (subscription — rough estimate).** The extraction ran in the **first week of June 2026** and used **~84% of the weekly Sonnet quota on a Claude Max 5× plan**. Converting that to tokens is approximate, with wide bounds:
- Max 5× baseline ≈ **140–280 Sonnet-hours/week**; with the early-June "50% more usage" promo active, ≈ **210–420 hours**. At 84% consumed → **~176–353 active Sonnet-hours**.
- At a typical coding-session rate of **~50k–100k tokens/active-hour**, that is **≈ 9–35M Sonnet tokens** consumed (order-of-magnitude only).

Caveats for the writeup: the bounds are wide (the hour→token conversion is a heuristic), and the 84% weekly quota also covered *other* development that week, so 9–35M is an **upper bound** on extraction-specific tokens, not a measured count. It is also not directly comparable to the metered-API line items below — those count only extraction prompt+completion tokens, whereas the subscription meter also absorbs Claude Code's agent overhead (tool calls, file reads) and benefits from prompt caching. The marginal **dollar** cost was still ~$0 (flat-rate); treat 9–35M as the order-of-magnitude **compute footprint** for an efficiency framing.

**Reproduction estimate (metered API).** **Output tokens dominate.** Structured float extraction produces output-heavy calls, and at Claude Sonnet pricing output is $15/MTok vs $3/MTok input (5×). Input and output must be budgeted separately — a blended "~600 tokens/call" estimate hides the binding constraint.

Phase 1 — reduced corpus (~4,500 movies × 6 groups = ~27,000 calls), at Sonnet pricing ($3/MTok in, $15/MTok out):

| Input regime | In tok/call | Out tok/call | Input $ | Output $ | **Total** |
|---|---|---|---|---|---|
| **TMDB-only / truncated (default)** | ~600 | ~150 | $48.60 | $60.75 | **~$109** |
| Full raw Wikipedia plots (DO NOT) | ~2,500 | ~150 | $202.50 | $60.75 | **~$263** |

For the metered path the feed-time input discipline (TMDB-first, and truncate Wikipedia before the extraction call — stored raw, capped in `format_for_prompt`) is mandatory, not advisory — the raw-Wikipedia path is ~2.4× over.

Full budget (metered-API reproduction only):

| Item | Estimated Cost |
|---|---|
| Web scraping (both phases) | $0 |
| LLM extraction — Phase 1 (~27k calls, Sonnet, lean input) | ~$109 |
| LLM extraction — Phase 2 (incremental, remaining ~4-5k movies) | ~$60-110 |
| Training compute (all models, both phases) | $0 |
| **Total (Phase 1 + Phase 2), metered API** | **~$170-220** |

**As actually run (Claude Code subscription), marginal API cost was ~$0.** The metered figures above only bind if you reproduce this without a flat-rate subscription.
