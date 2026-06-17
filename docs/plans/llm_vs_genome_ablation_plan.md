# LLM-Generated Item Features vs Human-Curated Genome Tags — Experiment Record

> **What this is.** The reproduction record for the completed LLM-vs-genome ablation — the
> durable lab notebook, not the original build plan. The narrative writeup is
> [`docs/llm_vs_genome_ablation/llm_vs_genome_ablation.md`](../llm_vs_genome_ablation/llm_vs_genome_ablation.md); grounded genome construction
> facts are in [`docs/llm_vs_genome_ablation/movielens_genome_features_info.md`](../llm_vs_genome_ablation/movielens_genome_features_info.md).
> This file keeps what those two don't: the authoritative model→checkpoint map, the exact eval
> methodology, the full result tables (**including Phase 1, which the writeup omits**), and the cost
> budget. The original stage-by-stage build plan, two-phase decision gate, implementation order,
> success criteria, and critical warnings were removed once the experiment shipped (2026-06-08) —
> recover them from git history (`git log --follow -p` on this file) if ever needed.

## Goal

Test whether LLM-generated item features can match or beat the human-curated MovieLens genome tags as content signal in a two-tower recommendation model.

**Research question:** Can modern LLMs replace expensive human curation for content features in recommendation systems?

**Deliverable:** Trained two-tower models on MovieLens 32M — one using genome tags (then-prod), one using LLM-generated features from scraped web content, and a no-content-slot floor baseline. Quantified ablation of which produces better recommendations, with the floor baseline measuring how much either content source adds at all. (A 4th arm, D = genome + LLM, was added as a secondary "does *more* help?" check.)

**Two experiments, in the order the writeup presents them** (`docs/llm_vs_genome_ablation/llm_vs_genome_ablation.md`):

1. **Primary — stripped "CF-base" models** (`BASE_TOWERS=idonly`): item ID embedding + a single implicit history pool + the content slot, and nothing else. This is the realistic / universal setting — what an implicit-feedback recommender with no curated metadata actually has. It is where the content question is cleanly answerable, so it leads.
2. **Follow-up — rich feature models** (default `BASE_TOWERS=all`): the same three content arms re-run with MovieLens's curated genre + 306 user tags + year + the rating-derived 4-pool history added back, to ask whether genome / LLM still help once the model is metadata-rich. (This was historically run *first*; the stripped experiment came after we noticed how much MovieLens-specific privilege the rich model baked in. The writeup and these tables present them in logical, not chronological, order.)

This was a **parallel experiment** within the existing Movie Recommender repo, not a new repo or project.

## Setup — what makes it a fair test

Within each experiment, the arms are **identical** in two-tower architecture, hyperparameters, train/val split, and loss (full softmax). The **only** difference is what fills the content slot:

- **genome:** 1,128-dim genome scores (`FEATURE_TOWERS=genome`) — arm A (rich) / A′ (stripped)
- **LLM:** 132 LLM-extracted dims (`FEATURE_TOWERS=llm`) — arm B / B′
- **none:** content slot removed — the floor (`FEATURE_TOWERS=none`) — arm C / C′
- **both:** genome + LLM concatenated pre-projection (`FEATURE_TOWERS=both`) — arm D (rich follow-up only, secondary)

The two model settings are selected by `BASE_TOWERS`: **`idonly`** (stripped/primary — ID + single history pool + content slot) vs **`all`** (rich follow-up — adds genre/tag/year/4-pool). A stripped arm's prime (C′/A′/B′) marks the `idonly` model.

**Low-variance protocol (all numbers below).** Seeded (`SEED=42`), **160k** training steps, best checkpoint selected on a **100k-example** val-MRR subset — which cut run-to-run selection noise enough that *best ≈ step_150000* everywhere. **An earlier 150k-step / 8,192-val-subset run was higher-variance**; its tables are retained below under "superseded" banners because one of its headline findings (a tail lift) did not replicate. **Menon popularity-correction α = 0 for every arm** — no popularity debiasing anywhere, so it cannot confound the genome-vs-LLM comparison (prod's α=0.5 is deliberately *not* used here). The LLM schema is **derived from the top-discriminability genome tags** (`data/top_genome_tags_by_discriminability.csv`), so both spaces sit on the **same 132 axes** — each LLM dim records its source genome tag(s) in `data/llm_schema_dimensions.json`. Config resolves towers *and* `base_towers` from state_dict weight keys, not filenames (`src/checkpoint.py:load_checkpoint`).

Two corpora: **Phase 1** pilot = 4,461 popular movies (> 1,000 raw ratings); **Phase 2** = full 9,375-movie corpus (> 200 ratings).

## Pipeline & artifacts

Code lives in `llm_features/` (run end-to-end once, then cached per movie+group):
`filter_corpus.py` (Phase 1 list) → `scrape.py` (TMDB-first: overview / tagline / genres / cast / director / writers / keywords, + Wikipedia plot + **factual** prestige indicators: Oscar wins/noms, Criterion status, box-office scale) → `derive_schema.py` (top genome tags → deduped, 6-bucketed dimension list) → `cc_extract.py` / `batch_extract.py` (six **grouped structured-output** calls per movie: themes / tone / setting / provenance / factual-reception / factual-visual) → `merge_extractions.py` → `build_features.py` → tensor.

- **Extractor:** Claude Sonnet via Claude Code (see "LLM choice" below).
- **Feature tensor:** `data/llm_features_claude-code-sonnet_v1.pt` — `9376 × 132`, 9,360 non-zero rows (9,366 of 9,375 corpus movies scraped successfully).
- **Scrape discipline:** store raw on disk, truncate **only at feed time** (`format_for_prompt`) — keeps the durable cache policy-free and the input token bill down (see Cost Budget; raw Wikipedia plots ≈5× the input cost).
- **Grouping rationale:** a single 130-dim prompt hits "lost in the middle" and defaults late dims to 0.5; six focused ~20–30-dim calls don't. Visual + prestige groups are **factual-only** (animation / b&w / Oscar-winner — yes; "visually stunning" from a synopsis — no). Reception is its own group so it is separately ablatable.
- **Validation gates (before training):** per-group 0–1 calibration (no defaulting to 0.5), known-similar similarity (Toy Story 1/2/3, Godfather I/II), and a cross-group consistency check (flag contradictory profiles, e.g. uplifting+devastating).
- **Analysis / figures:** `feature_level_analysis.py` (the 132 shared-axis correlations) and `make_figures.py` (writeup figures → `tools/results/figures/`, committed copies in `docs/llm_vs_genome_ablation/figures/`, correlation cache `feature_agreement_r.json`).

### LLM choice — Sonnet via Claude Code (bake-off dropped)
**Decision (2026-06): extraction was run with Claude Sonnet through Claude Code, on a flat-rate Max subscription — marginal API cost ~$0.** The originally-planned Sonnet-vs-Haiku bake-off is **dropped.** Its entire rationale was pay-as-you-go cost control (output dominates the bill — Sonnet output $15/MTok vs Haiku ~$4/MTok — so a cheaper model that passed calibration would have cut the bill ~4×). That premise no longer holds: the features were generated under a subscription already in hand, at no marginal API cost, so there is nothing to optimize by switching models. Running the bake-off would *itself* cost $100+ in API credits to answer a **secondary** question ("could a smaller model also do it?") that the core LLM-vs-genome comparison does not need.

**Consequence for the writeup (Limitations).** This is a **single-LLM** study — Claude Sonnet via Claude Code → `data/llm_features_claude-code-sonnet_v1.pt`. The paper can claim *Sonnet-class extraction matches genome*; it **cannot** claim a smaller/cheaper model would. State the cost honestly as "amortized under an existing flat-rate coding subscription, ~$0 marginal," **not** as a per-call API figure. The pay-as-you-go pricing analysis is retained in the Cost Budget only as a *reproduction estimate* for anyone without such a subscription.

**Prompt caching:** the 6 group prompts are static except the per-movie content block. Caching the static prefix cuts *input* cost if movies are batched by group within the 5-min cache TTL. Worth doing — but note it does not touch the dominant *output* cost, so it can't rescue the budget on its own.

## Canonical model → checkpoint map (authoritative)

Single source of truth for which checkpoint is which model. All numbers are whole-corpus MRR,
canonical eval (all 19,134 val users, n=382,138 for full; 5,000 users, n=99,846 for phase1),
**low-variance protocol** (seeded, 160k steps).

**Experiment 1 (primary) — stripped CF-base, `BASE_TOWERS=idonly`, α=0, trained 2026-06-10:**

| Model | Content | Full checkpoint (`saved_models/`) | Full MRR | Phase 1 MRR |
|---|---|---|---|---|
| **C′** | none | `best_softmax_idonly_popularity_alpha_0_20260610_081552.pth` | 0.1121 | 0.1133 |
| **A′** | genome | `best_softmax_idonly_genome_tags_popularity_alpha_0_20260610_091949.pth` | 0.1148 | 0.1158 |
| **B′** | LLM | `best_softmax_idonly_llm_features_popularity_alpha_0_20260610_095940.pth` | **0.1155** | **0.1165** |

(Phase 1 stems: `best_softmax_idonly[_genome_tags|_llm_features]_…_phase1_20260610_{104200,110324,114004}.pth`.)

**Experiment 2 (follow-up) — rich feature models, `BASE_TOWERS=all`, α=0, trained 2026-06-09:**

| Model | Content | `FEATURE_TOWERS` | Full checkpoint (`saved_models/`) | Full MRR | Phase 1 MRR |
|---|---|---|---|---|---|
| **C** | none | `none` | `best_softmax_popularity_alpha_0_20260609_212446.pth` | 0.1174 | 0.1162 |
| **A** | genome | `genome` | `best_softmax_genome_tags_popularity_alpha_0_20260609_195854.pth` | 0.1144 | 0.1151 |
| **B** | LLM | `llm` | `best_softmax_llm_features_popularity_alpha_0_20260609_204904.pth` | **0.1176** | **0.1180** |

(Phase 1 stems: `best_softmax[_genome_tags|_llm_features]_…_phase1_20260609_{183844,162905,171724}.pth`. B trained on the data-fix-corrected LLM tensor — see the "Data-integrity fix" note below.)

**Model D (genome + LLM, `FEATURE_TOWERS=both`) — secondary, high-variance only.** `best_softmax_genome_tags_llm_features_popularity_alpha_0_20260607_131924.pth`, MRR 0.1154 — the old 150k-step / 8,192-val run; **not** re-trained under the low-variance protocol, so it is not directly comparable to the A/B/C numbers above. Prod is a separate α=0.5 fine-tune of this architecture (see Current State / CLAUDE.md).

**Config resolution does not depend on the filename.** `src/checkpoint.py:load_checkpoint` resolves the content towers *and* `base_towers` from the *state_dict weight keys*, so renaming a `.pth` is safe and a stripped model rebuilds from weights alone. The low-variance and stripped checkpoints use the explicit `item_genome_tag_tower` / `item_llm_feature_tower` keys natively; only the legacy "content-era" checkpoints (the old 06-07 A/B/D) use generic `item_content_tower` keys disambiguated by the `_config.json` sidecar (must keep the same stem).

> **Superseded high-variance run (2026-06-07, 150k steps / 8,192-val).** The earlier rich A/B/C/D numbers (A 0.1146, B 0.1165, C 0.1143, D 0.1154) and the Phase 1 val-loss-vs-val-MRR selection tables in "Eval methodology" below are from that higher-variance run. They are kept for the record but are **not** the canonical numbers — the low-variance re-run above is. Where a finding from the old run did not replicate (notably the tail lift), it is flagged inline.

## Eval methodology & results (Experiment 2, rich — superseded high-variance run)

> **This whole section is the original high-variance rich-experiment run** (2026-06-07/09, 150k
> steps, 8,192-example val subset). It is the **follow-up** experiment (rich feature models), kept
> for the record. The **canonical** numbers are the low-variance re-run in the Canonical map above;
> the **primary** experiment (stripped CF-base) is in "Experiment 1 — stripped CF-base" below. One
> headline here — the tail lift — **did not replicate** under the low-variance protocol; it is
> retracted inline in the Long-tail split. Read this section as provenance, not canon.

Rollback protocol throughout: for each held-out val user, context = history[0..j-1], target = history[j], up to 20 chronological positions per user. Recall@10 = Hit@10 (one held-out target per example).

**Phase 1 — reduced corpus (4,461 movies, > 1,000 raw ratings; α=0; rollback protocol, n=99,846 over 5,000 val users, seed 42).**

> ⚠️ **Pre-fix — predates the 2026-06-09 feature-data fix.** These Phase 1 B numbers were trained on the *phase1* LLM tensor before 141 wrong-movie extractions were corrected (see the data-fix note in the Phase 2 section). phase1 B was **not** re-trained on the rebuilt `_phase1.pt`, so the Phase 1 B column below is stale; the corrected single-source result is the **Phase 2** table. (A/C are unaffected — they use no LLM features.)

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

> **Eval methodology (Phase 2).** The canonical run uses **all 19,134 val users** (now `main.py eval`'s default; `EVAL_N_USERS` caps it down for smoke runs) → **n=382,138** rollback examples, giving the long-tail tiers ~3.8× more signal. Whole-corpus numbers are ~1% lower than a 5,000-user run but the C < A < B ordering is identical. Popularity tiers below are by the *target* movie's **raw `ratings.csv` count** — the same basis that defines the corpus (`> 200`) and the Phase 1 threshold (`> 1000`); counts cached at `data/corpus_raw_rating_counts.npy`. The **HEAD** tier (`> 1000`) is byte-for-byte the Phase 1 corpus (4,461 movies; verified zero symmetric difference vs `data/llm_experiment_movies_phase1.json`), so **TAIL** (`≤ 1000`) is exactly the long tail Phase 1 excluded. **Q1–Q4** are equal-movie-count population quartiles (Q1 rarest → Q4 most popular). Implemented in `src/offline_eval.py` (`_build_tiers`, `_corpus_raw_rating_counts`); full per-K outputs for every tier under `eval_results/`.

| Metric | No content (C) | Genome (A) | LLM (B) | A−C | B−C |
|---|---|---|---|---|---|
| Hit@1 | 0.0577 | 0.0576 | **0.0591** | −0.0001 | +0.0014 |
| Hit@5 | 0.1536 | 0.1538 | **0.1565** | +0.0002 | +0.0029 |
| Hit@10 | 0.2213 | 0.2229 | **0.2254** | +0.0016 | +0.0041 |
| Hit@20 | 0.3101 | 0.3131 | **0.3152** | +0.0030 | +0.0051 |
| Hit@50 | 0.4611 | 0.4642 | **0.4665** | +0.0031 | +0.0054 |
| NDCG@10 | 0.1283 | 0.1288 | **0.1309** | +0.0005 | +0.0026 |
| MRR | 0.1143 | 0.1146 | **0.1165** | +0.0003 | +0.0022 |

**Phase 2 whole-corpus verdict:** **B (LLM) leads every metric** — B−A = +0.0019 MRR (+1.7%), B−C = +0.0022 (+1.9%), A−C = +0.0003 (+0.3%). The Phase 1 finding (LLM matches/slightly beats genome) **holds and firms up with the long tail present**: B now leads on *every* metric (Phase 1 had a Hit@20 tie). The aggregate lift-over-floor is small because the rollback target distribution is popularity-skewed (Q4 popular movies are 90% of examples) — the lift lives in the long-tail split below, not the aggregate.

Checkpoints (α=0, full corpus, content slot is the only difference; A/C trained 2026-06-07, **B re-trained 2026-06-09 on the corrected LLM feature tensor**):
- A genome = `best_softmax_genome_tags_popularity_alpha_0_20260607_101027.pth`
- B llm = `best_softmax_llm_features_popularity_alpha_0_20260609_121604.pth`
- C nocontent = `best_softmax_popularity_alpha_0_20260607_112755.pth`

> **Data-integrity fix (2026-06-09).** An audit found **141 movies (~1.5%)** whose LLM extractions had been saved under the **wrong movieId** (a save-side misalignment in the extraction fan-out — e.g. The Matrix scored as a school romance). They were re-extracted (1 movie/agent, genre-guarded `ingest()`) and the tensor rebuilt; **B was re-trained on the corrected tensor → 0.1165** (was 0.1157 on the corrupt tensor, +0.0008, concentrated in the popular head). **A** (genome) and **C** (no content) don't use LLM features, so their 06-07 numbers are unchanged and remain directly comparable. The Phase 2 tables below reflect the corrected B; the **Phase 1 tables above predate the fix** (phase1 B was not re-trained on the rebuilt `_phase1.pt`).

### Long-tail split (Phase 2)

> ⚠️ **RETRACTED under the low-variance re-run — the tail lift was a high-variance artifact.**
> Finding 1 below ("content earns its keep on the tail, A−C ≈ +18% relative") **does not
> replicate.** On the canonical low-variance rich run, the deep tail is a dead heat: TAIL MRR is
> ~0.0031 for all three arms, and TAIL Hit@250 has genome *at or below* the floor — **C 0.1329,
> A 0.1321, B 0.1360** (A−C = −0.0008; Q1 Hit@250 even more so, A 0.0497 vs C 0.0543). The rich
> model is a near-null *everywhere*, tail included — content is redundant with the curated
> genre/tags (this is exactly what the stripped Experiment 1 isolates). The high-variance tier
> table below is kept for provenance; **the canonical finding is: the measurable content lift is on
> the popular head, in the stripped setting — not the cold tail** (at MovieLens's ≥200-rating floor
> the deep tail is too sparse to resolve, and true cold-start can't be benchmarked against the
> genome at all).

On the full corpus we report metrics **restricted by the target movie's popularity tier** — this is where content features matter most and where the genome-vs-LLM question is most consequential.

**MRR by tier** (example count in parens; all three models share the same examples):

| Tier (n) | C | A | B | A−C | B−C | B−A |
|---|---|---|---|---|---|---|
| Whole corpus (382,138) | 0.1143 | 0.1146 | **0.1165** | +0.0003 | +0.0022 | +0.0019 |
| HEAD > 1k (369,486) | 0.1181 | 0.1184 | **0.1204** | +0.0003 | +0.0023 | +0.0020 |
| Q4 popular (343,906) | 0.1259 | 0.1260 | **0.1282** | +0.0001 | +0.0023 | +0.0022 |
| Q3 mid (26,923) | 0.0129 | **0.0148** | 0.0143 | +0.0019 | +0.0014 | −0.0005 |
| Q2 mid (8,049) | 0.0032 | **0.0038** | 0.0036 | +0.0006 | +0.0004 | −0.0002 |
| Q1 rarest (3,260) | 0.0012 | 0.0014 | 0.0014 | +0.0002 | +0.0002 | ±0.0000 |
| **TAIL ≤ 1k (12,652)** | 0.0028 | **0.0033** | 0.0031 | +0.0005 | +0.0003 | −0.0002 |

**Tail-tier recall** (Hit@50 and Hit@250 — tail tiers have few top-rank hits, so deeper-K recall carries more signal than MRR there):

| Tier | C@50 | A@50 | B@50 | C@250 | A@250 | B@250 |
|---|---|---|---|---|---|---|
| Q3 mid | 0.1134 | **0.1251** | 0.1211 | 0.3420 | **0.3608** | 0.3558 |
| Q2 mid | 0.0226 | **0.0287** | 0.0278 | 0.1442 | 0.1536 | **0.1542** |
| Q1 rarest | 0.0034 | **0.0061** | 0.0058 | 0.0463 | **0.0604** | 0.0589 |
| TAIL ≤ 1k | 0.0192 | **0.0249** | 0.0241 | 0.1243 | **0.1368** | 0.1367 |

**Findings:**

1. **Lift-over-floor thesis — confirmed on solid n.** Content's advantage over the no-content floor is ~0% on the popular head and grows steeply toward the tail. Hit@250 A−C: Q4 +0.0010 → Q3 +0.0188 → Q1 +0.0141 → TAIL +0.0125; relative MRR lift on TAIL ≈ +18% (A−C) / +14% (B−C) vs ~0.1% on Q4. Both content sources earn their keep exactly where collaborative signal is sparse — the story Phase 1 structurally could not tell.

2. **B beats A overall, driven entirely by the popular head.** Q4 (90% of examples) carries B−A ≈ +0.0022 MRR, stable across K. The whole-corpus and HEAD numbers are essentially the Q4 result.

3. **The key tail result: LLM does NOT collapse on rare movies — it matches genome.** On the deep tail (Q1, TAIL) A and B are statistically tied (MRR gaps ≤ 0.0002; Hit@250 gaps ≤ 0.0015, with TAIL a dead heat — A 0.1368 vs B 0.1367). Even where content matters most, LLM features hold even with human-curated genome. This is the consequential, positive result for "can LLMs replace human curation," and it is the question Phase 1 could not answer.

4. **Genome keeps a small, consistent edge in the *mid*-tail (Q3, n=26,923).** Genome leads B on MRR (0.0148 vs 0.0143), Hit@50 (0.1251 vs 0.1211) and Hit@250 (0.3608 vs 0.3558) — ~2–5% relative, the one place the A > B gap is consistent across K rather than noise. Q3 spans 907–2,857 ratings (straddling the Phase 1 boundary), i.e. "moderately popular," not cold-start.

**Phase 2 verdict (high-variance run — superseded).** This run read "LLM matches genome on the tail and slightly beats it overall; genome keeps a marginal Q3 edge." Under the low-variance re-run the **rich** experiment is a near-null everywhere (genome below the floor, tail a dead heat — see the retraction banner above), so this verdict no longer stands as written. **Canonical verdict** (low-variance, both experiments): in the *rich* model content is redundant (genome below floor, LLM ties it); in the *stripped* model both content sources clear the floor and **LLM ≥ genome**, the LLM being the less redundant source (substitution ladder). Full statement in "Experiment 1 — stripped CF-base" below.

### Qualitative comparison via canary users

**Phase 2, full corpus.** Top-10 from genome (A) vs LLM (B) for all ~19 canary personas (`ts_max` bin), saved to `canary_results/best_softmax_genome_tags_popularity_alpha_0_20260607_101027.txt` (A) and `…_llm_features_…_20260609_121604.txt` (B — re-run on the corrected tensor). The 5 most illustrative disagreements (below) were curated from the **pre-fix** B canary (`…_20260607_105646`) and their **specific titles are now stale** — a spot-check of the post-fix B canary (saved at the new stem) shows the persona *personalities* still hold (B → era/modern, blockbuster-leaning, drifts off niche canon; A → canon-pure), but the exact movies differ (e.g. Western B now drifts to Godfather/Goodfellas/Apocalypse Now rather than Patton/Dirty Dozen). Read the table as illustrating the **personalities**, not B's literal current output; regenerate from the new canary if title-accuracy is needed:

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

**Synthesis (ties the experiment together).** The spaces are strongly aligned on factual/genre axes (mean r 0.60) → both place a movie in the right broad genre, which is why B matches A on the bulk metrics. Genome's *exclusive* axes are the **low-agreement** ones — subjective aesthetics, niche sub-genre granularity, and crowd-prestige — which show up qualitatively in the niche-canon canary personas. But those axes are mostly crowd-sentiment that tracks popularity, not content, so they **don't translate into a ranking win** (under the low-variance run genome does not beat the LLM, or even the floor, on any tier). This is the mechanistic explanation for the headline result: **LLM extraction reproduces nearly all of genome's content signal on the axes an LLM can reach from text** (mean r 0.60), so B matches/edges A; what genome holds exclusively is the crowd-sentiment / fine-aesthetic slice that barely moves ranking (the reception/prestige asymmetry — see the writeup's Limitations).

### Model D — genome + LLM combined (secondary arm; out of core scope)

> Combining both content sources in one model is a **non-goal** of the core genome-vs-LLM comparison (it answers a different question — "does *more* help?"). Run as a 4th arm with the same data so the answer is on record; documented here only (the reframed writeup omits D entirely). Model D adds the LLM-feature sub-towers *alongside* the genome-tag sub-towers (two parallel families, concatenated pre-projection) — `FEATURE_TOWERS=both`. Checkpoint `best_softmax_genome_tags_llm_features_popularity_alpha_0_20260607_131924.pth` (α=0, full corpus). Same eval protocol (n=382,138, all 19,134 val users); A/B/C re-run with the current code for an apples-to-apples 4-arm set.

> ⚠️ **High-variance only — D was not carried into the low-variance re-run.** The tables in this section are from the original 06-07 high-variance 4-arm set (their A/B/C match the superseded numbers, not the canonical low-variance ones). D was deliberately **excluded** from the low-variance re-run (the re-run covered only the single-source A/B/C arms it needed); there is no low-variance D. So read D's numbers only *within* this high-variance set, as a directional "does combining help?" check — not against the canonical Experiment 1/2 numbers. The qualitative finding (D ≈ better single source, no clear additive benefit) is what carries forward.

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

### Experiment 1 — stripped CF-base (primary; isolating the content slot)

> **This is the PRIMARY experiment** (the writeup leads with it; `docs/llm_vs_genome_ablation/llm_vs_genome_ablation.md`
> §4). It is placed after the rich follow-up here only for document history — the rich run was
> built first; logically the stripped experiment is first (see the Goal's two-experiment framing).
> **Status (2026-06-10):** full-corpus + phase1 arms done + evaluated, canonical low-variance
> protocol (seeded, 160k steps, 100k-example val subset). Compared against the matched low-variance
> rich-base (Experiment 2 in the Canonical map), so the strip is cleanly isolated from any method
> change.

**Why.** Arm **C ("floor")** is not a true floor: it still carries the always-on **genre
one-hot + 306 user tags + year + 4-pool** user history — content/interaction signal that
*overlaps* with what genome/LLM encode. So "genome adds ≈0 over C" (and, under the low-variance
re-run, genome sits *below* C: A−C = −0.0030) cannot be distinguished from "genome is
**redundant** with the genre+tags already present." Phase B removes the confound: strip the base
towers and re-ask the content question against a genuine collaborative-filtering floor, where a
content feature has to carry the signal alone.

**Method.** A new `BASE_TOWERS=idonly` mode (`src/model.py`, gated exactly like the content
slots) strips, as a block: item/user **genre**, item **tag**, **year**, **timestamp** towers
(+ buffers), and collapses the 4-pool user history to the **single full-history sum pool** (drops
the liked/disliked/rating-weighted pools — all rating-derived). What remains is the barest
two-tower CF model:
- user = `LayerNorm(Σ watched-item ID embeddings)`  [+ content context, if a slot is on]
- item = `ID embedding`  [+ content vector, if a slot is on]

Three arms, identical but for the content slot; α=0, 160k steps, seed 42, full corpus:
- **C′** — ID pool only (`BASE_TOWERS=idonly FEATURE_TOWERS=none`): pure CF floor
- **A′** — ID pool + genome (`…FEATURE_TOWERS=genome`)
- **B′** — ID pool + LLM (`…FEATURE_TOWERS=llm`)

Stripped concat dims: C′ user 32 / item 32; A′, B′ user 64 / item 64 (vs rich 196 / 96). Config
resolves `base_towers` from state_dict keys (`src/checkpoint.py`), so the loader rebuilds a
stripped model from weights alone. Same canonical eval (all 19,134 val users, n=382,138).

Checkpoints (full corpus):
- C′ `best_softmax_idonly_popularity_alpha_0_20260610_081552.pth`
- A′ `best_softmax_idonly_genome_tags_popularity_alpha_0_20260610_091949.pth`
- B′ `best_softmax_idonly_llm_features_popularity_alpha_0_20260610_095940.pth`

**Full-corpus results (n=382,138), MRR by tier:**

| Tier (n) | C′ floor | A′ genome | B′ llm | A′−C′ | B′−C′ | B′−A′ |
|---|---|---|---|---|---|---|
| Whole (382,138) | 0.1121 | 0.1148 | **0.1155** | +0.0027 | +0.0034 | +0.0007 |
| HEAD > 1k (369,486) | 0.1159 | 0.1186 | **0.1193** | +0.0027 | +0.0034 | +0.0007 |
| Q4 popular (343,906) | 0.1234 | 0.1264 | **0.1271** | +0.0030 | +0.0037 | +0.0007 |
| Q3 mid (26,923) | 0.0133 | 0.0140 | **0.0142** | +0.0007 | +0.0009 | +0.0002 |
| Q2 mid (8,049) | 0.0034 | 0.0034 | **0.0037** | ±0.0000 | +0.0003 | +0.0003 |
| TAIL ≤ 1k (12,652) | 0.0029 | 0.0030 | **0.0033** | +0.0001 | +0.0004 | +0.0003 |
| Q1 rarest (3,260) | 0.0011 | 0.0013 | **0.0014** | +0.0002 | +0.0003 | +0.0001 |

Ordering is **C′ < A′ < B′ on every tier.**

**Phase 1 results (reduced corpus, head-only; default 5,000 val users, n=99,846), whole MRR.**
Phase 1 is all > 1,000-rating head movies, so Whole = HEAD and the deep-tail tiers are degenerate;
it isolates the content question on the head only. Checkpoints `best_softmax_idonly[_genome_tags|
_llm_features]_…_phase1_2026061{0}_…`.

| Arm | C′ floor | A′ genome | B′ llm | A′−C′ | B′−C′ | B′−A′ |
|---|---|---|---|---|---|---|
| Whole = HEAD (99,846) | 0.1133 | 0.1158 | **0.1165** | +0.0025 | +0.0032 | +0.0007 |

Same ordering **C′ < A′ < B′** as the full corpus. Phase 1 reads ~1% higher in absolute terms
(5,000-user protocol vs 19,134), but the comparison is internally matched.

**Substitution ladder — drop from the matched (low-variance) rich-base arm, whole MRR.**

| Arm | full rich-base | full stripped | full Δ | phase1 rich-base | phase1 stripped | phase1 Δ |
|---|---|---|---|---|---|---|
| C (floor) | 0.1174 | 0.1121 | **−0.0053** | 0.1162 | 0.1133 | **−0.0029** |
| B (LLM) | 0.1176 | 0.1155 | **−0.0021** | 0.1180 | 0.1165 | **−0.0015** |
| A (genome) | 0.1144 | 0.1148 | **+0.0004** | 0.1151 | 0.1158 | **+0.0007** |

(Full HEAD shows the same as full Whole: C −0.0054, B −0.0022, A +0.0004.) **The ladder
replicates on both corpora** — genome gains/holds, LLM loses a little, the floor loses the most.

**Findings.**

1. **The genome lift flips sign once isolated — on both corpora.** Full rich base: A−C = −0.0030
   (genome net-negative on top of genre/tag/year); stripped A′−C′ = **+0.0027 (+2.4%)**,
   B′−C′ = **+0.0034 (+3.0%)**. Phase 1 replicates the flip on a fully independent corpus:
   A−C = −0.0011 → A′−C′ = **+0.0025**. Both content sources clear the *true* CF floor — the
   rich-base near-null was **redundancy** with the curated genre/tags, not content being worthless.

2. **Substitution ladder — the sharpest evidence, and it replicates on both corpora.** How much
   each arm *lost* to the strip — full: genome **+0.0004 (nothing)**, LLM **−0.0021 (a little)**,
   floor **−0.0053 (the most)**; phase1: genome **+0.0007**, LLM **−0.0015**, floor **−0.0029** —
   same ordering both times. Genome fully reconstructs genre/tags from its own vectors (genome tags
   ≈ curated genre/tags rebadged) → complete substitute; the LLM only *partially* backfills them →
   it is a partly **orthogonal** basis (plot/tone/theme/cast) that can't fully stand in for the
   curated metadata; the floor has nothing to fall back on. Substitutability genome > LLM > none is
   direct evidence that the **LLM features overlap less with cheap curated metadata — i.e. carry
   more genuinely additive signal.**

3. **B′ > A′ on every tier, both corpora** (full: whole +0.0007, HEAD +0.0007, TAIL +0.0003,
   Q1–Q4 all ≥ 0; phase1: +0.0007). It replicates the rich-base direction (B > A there too, by
   more). Across two corpora and two training regimes, LLM ≥ genome on every tier.

4. **Lift is on the head, not the cold tail.** TAIL stays ≈0.003 for all three (B′ nominally
   highest); on 12.6k sparse examples it is barely resolvable, and MovieLens "tail" is still
   ≥200 ratings. The content separation lives in Q4/Q3.

**Honest caveats.**
- **Single seed; gaps near the noise floor.** B′−A′ = +0.0007 is well inside the ±0.003–0.004
  run-to-run noise established by the low-variance re-run; on its own it is not significant. The
  weight is in the *consistency* and *replication*, not any one gap: B′ ≥ A′ on every tier of
  *both* corpora, the sign-flip holds on both, and the substitution ladder reproduces on both —
  a cross-corpus replication is stronger evidence than two more seeds on one corpus would be,
  though a multi-seed run per arm would still firm up the exact magnitudes.
- **Stripped ≠ better.** The best stripped arm (B′ 0.1155) is still below the best rich arm
  (B 0.1176). The base towers add aggregate value; we strip to *isolate the content question*,
  not to improve the model.
- **The stripped floor also drops rating polarity (liked/disliked/weighted pools), which is
  interaction-derived — every real system has it.** So C′ is a deliberately *weak* CF baseline,
  which **inflates** the measured content lift. The true lift depends on CF strength: the two
  experiments **bracket** it — ≈0 with full rich metadata, +2–3% against minimal CF — and a
  realistic no-curated-metadata system (CF with rating polarity, no genre/tags) sits between,
  closer to the stripped end. A "keep-4-pool, strip-only-curated-metadata" arm would pin it
  exactly; noted as a follow-up, not run (approach (a): caveat in prose, don't add the arm).

**Framing for the writeup (the practitioner / CTO read).** MovieLens is a *privileged* dataset:
clean professional **genre labels** on every title plus a massive **crowd-tagging** effort (the
genome is built from millions of tag applications). Almost no real catalog has that. The team
deciding *"is it worth building content features?"* has interactions + whatever it can
scrape/LLM-extract — the **stripped** regime, not the rich one. So the two experiments answer two
different questions:
- **Experiment 1 (rich):** *"I'm MovieLens-like and already have genome + tags — add more
  content?"* → No, it's redundant.
- **Experiment 2 (stripped):** *"I have interactions but no free curated metadata — build content
  features?"* → Yes: modest but real (+2–3% MRR over pure CF), LLM-extracted ≥ genome-style, with
  the LLM features measurably *less* redundant.

The second question is the one most teams actually face. And **C is not a "no-content" baseline in
Experiment 1** — it is a *rich-metadata* baseline minus one slot; the writeup must state what C
contains so "floor beat genome" cannot be misread as "content features don't help."

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
