# $200 vs $200k: Item Content Features With an LLM Instead of Hand-Labeled Tagging

*A controlled two-tower recommender system ablation on MovieLens 32M.*

> *A hand-curated tag genome and LLM-extracted features are the same idea one generation apart: turn cheap signals into a dense, per-item content matrix. The first took fifteen years of community data to build; the second needs only an item's public web text — synopsis, cast, plot, scraped and run through an LLM — and built the whole corpus in a day.*

## TL;DR

Every recommender needs a dense content vector per item — and the gold-standard source, MovieLens's hand-curated **tag genome**, rests on fifteen years of community folksonomy. The feasible alternative — scrape each item's own text and **extract structured features with an LLM** — covered all **\~9,375 corpus movies** (MovieLens's 200+-rating slice) in **a single day**, at near-zero marginal cost. In a matched three-arm ablation (same model, same training, only the content tags change — **MovieLens genome tags vs LLM feature tags vs no content tags**), the cheap LLM features come out **statistically tied with the gold-standard genome** — and on genome's *own* taxonomy, a deliberate handicap — at MRR 0.1157 vs 0.1146. The claim isn't "better," it's *as good as*.

![Recommendations from the deployed app for a 'pixar animation' genome-tag query](figures/pixar_movies.jpeg)

*The deployed recommender these features power. A user's taste vector is built entirely from movie content — the curated genome tags plus the web-scraped, LLM-extracted features this post compares — with no user-ID embedding.*

## 1. The content-feature problem, and the two ways to solve it

A two-tower recommender represents each item as an embedding. Collaborative signal — who watched what — carries most of the weight for popular items, but it runs dry exactly where you need help: the long tail, and brand-new items nobody has interacted with yet. That's what a **content vector** is for — a dense, per-item descriptor of what the item *is*, independent of who has touched it. The question is where it comes from.

**Option 1 — the genome.** The MovieLens **tag genome** is the gold standard: a dense matrix of `rel(movie, tag) ∈ [0,1]` — **1,128 tags × 16,376 movies (≈18M scores)** in this project's MovieLens 32M data (9,734 movies in GroupLens's original 2014 release), with every cell populated. But note that **16,376**: it's all GroupLens ever scored — only **\~19% of MovieLens 32M's 87,585 titles**. The genome simply does not exist for the sparser \~71k (a structural limit we return to in the limitations below); it is dense *within* its coverage but covers only the well-tagged head. And the part most people get wrong: it was built **not** by mass human labeling, but from a small **50,203-judgment survey of 676 volunteers** (1–5 scale; a fraction of a percent of the matrix; plus an 85-user pilot), with the rest filled by machine learning (a `glmer` regression, MAE 0.211) over features mined from a **large pre-existing folksonomy** — 186,000 users, 17M ratings, 246,000 tag applications accrued since 1997 — plus crawled IMDb reviews.

**Option 2 — LLM extraction.** Scrape the item's public text (TMDB synopsis, cast, Wikipedia plot) and ask an LLM to score it against a fixed tag taxonomy, with structured (JSON-schema) output. The content vector falls out of the item's own description.

**The catch that sets up the experiment.** The genome's inputs — a mature folksonomy, crawled reviews, a relevance survey — are exactly what a growing company *doesn't* have, and they're all zero for an item added a minute ago. So the honest question isn't "which is better in the abstract." It's: **the LLM path is the one most teams can actually take — is its content quality good enough?**

## 2. Why it's a fair fight

To compare content *sources* and nothing else, everything else is held fixed: the same two-tower architecture, the same training recipe, and the same evaluation. Only the content slot changes:

- **A — MovieLens genome tags:** the 1,128-dim genome scores fill the content tower.
- **B — LLM feature tags:** 132 LLM-extracted dims fill it instead.
- **C — no content tags:** the content slot is removed entirely — the **floor**, what the model scores on its ID-embedding history pools, genre, tags, and year alone.

(Throughout, **A / B / C** are shorthand for these three — genome / LLM / no-content.)

One more rigor move — and it's a deliberate **handicap on the LLM, not the genome**: the LLM schema is **derived from genome's own top-discriminability tags**, not hand-invented. Both spaces measure the *same axes* — otherwise "LLM ≈ genome" would be unmeasurable — but confining the LLM to genome's taxonomy makes the question the sharpest one there is: *can LLM extraction match the curated genome on the genome's home turf, tag for tag?* The LLM is graded only on axes GroupLens chose, and earns no credit for discriminative patterns it could surface from the scraped text but that genome never tagged. (132 dims, each stored alongside the source genome tag it was derived from.)

This also answers the obvious objection — *"a greenfield team has no taxonomy to extract against."* It doesn't need one: an LLM can **draft** the taxonomy too, clustering recurring themes across thousands of plots. We could have let it mine extra tags it found discriminative in the scrape data; we deliberately didn't, to keep the fight fair. The genome dependency is a self-imposed rule of *this experiment*, not a requirement of the *method* — so if anything, the tie **understates** what an unconstrained LLM pipeline could do.

Worth flagging one real asymmetry: genome feeds **1,128** raw dims into the content tower and the LLM only **132**, but both get squeezed down to the same 32 — so genome takes the harder compression. You'd think that hurts genome, but shrinking a big sparse vector into a small dense one is routine, and it usually helps the model find the signal rather than memorize individual cells. If it tips the result either way, it isn't toward the LLM.

**Where this sits.** Manufacturing item features with an LLM isn't new — it's an active 2023–25 direction (KAR, ONCE/GENRE at WSDM '24, LLMRec, RLMRec all feed LLM-derived signal into recommenders, KAR with a reported production A/B gain at Huawei). What's new here is the *comparand*: rather than scoring LLM features against a weak or absent baseline, this pits them head-to-head against the **gold-standard human-curated genome, on the same axes**, with a no-content floor (C) to calibrate the lift. The question isn't "do LLM features help" — it's "how close to hand-curation does the feasible option get."

## 3. The cheap pipeline

The number a team lead will care about first: **the entire corpus was scraped and feature-extracted in a single day** — one engineer, no annotation team. Because each item is scored independently, the work is parallel; hundreds of model-hours of extraction fan out across concurrent calls and finish inside a day. Set that against the genome's binding input — fifteen years of accrued community data, or the weeks-to-months of dedicated new tagging effort it would take to bootstrap even a rough substitute.

For each of the \~9,375 corpus movies (9,366 scraped successfully): pull TMDB first — overview, tagline, genres, top-billed cast, director, writers, keywords — supplemented by Wikipedia plot and **factual** prestige indicators (Oscar wins/noms, Criterion status, box-office scale). Then run **six grouped structured-output extraction calls** — themes, tone, setting/era, provenance/structure, factual reception/prestige, visual medium — \~20–30 dimensions each, every call enforced by a JSON schema. Grouping is deliberate: a single 130-dim prompt hits "lost in the middle" and defaults late dimensions to 0.5; six focused calls don't.

Honest design calls are baked in. Structured output is non-negotiable — free-form silently corrupts the tensor. The visual and prestige groups are **factual-only**: animation, black-and-white, Oscar-winner, yes; "visually stunning" hallucinated from a synopsis, no. Reception/prestige is its own group so it can be ablated separately. The extractor was **Claude Sonnet via Claude Code**.

**What comes out — one movie's fingerprint.** Nothing below is hand-picked or tuned; it's the raw six-call output for *Alien* (1979), top scores per group:

| Group | Top extracted features (score 0–1) |
|---|---|
| Themes & plot | `survival` 1.0, `betrayal` 0.7, `mortality` 0.7 |
| Tone & mood | `tense` 1.0, `dark` 0.9, `atmospheric` 0.9, `creepy` 0.9, `scary` 0.9 |
| Setting, era & sub-genre | `space` 1.0, `aliens` 1.0, `monster` 0.9, `future` 0.7 |
| Provenance & structure | `franchise` 0.8, `twist_ending` 0.7 |
| Factual reception & prestige | `oscar_technical` 1.0, `classic` 0.9, `cult_classic` 0.6 |
| Visual medium | `cgi_heavy` 0.3 |

Every score reads right, and the two *factual* groups behave: `oscar_technical` is a true 1.0 (Alien won the Academy Award for Best Visual Effects), while `cgi_heavy` stays low because the film's effects are practical, not computer-generated — exactly the factual-only discipline the visual and prestige groups are built for.

## 4. Does it work?

The pilot (Phase 1) ran the whole pipeline on the 4,461 popular movies (>1,000 ratings); the full run (Phase 2) added the long tail and retrained all three arms fresh on the full corpus. That corpus is **9,375 movies — already a heavily filtered slice of MovieLens 32M's 87,585 titles**, keeping only movies with more than 200 ratings (\~11% of the catalog). Phase 2 is the real test — it has the tail, where content features earn their keep. Canonical eval: rollback protocol, all 19,134 validation users, **n = 382,138** rollback examples (random Hit@250 baseline = 2.7%, so these models are doing real work).

**Phase 2, whole corpus:**

| Metric | C — no content | A — genome tags | B — LLM tags |
|---|---|---|---|
| Hit@5 | 0.1536 | 0.1538 | **0.1552** |
| Hit@10 | 0.2213 | 0.2229 | **0.2240** |
| Hit@50 | 0.4611 | 0.4642 | **0.4656** |
| NDCG@10 | 0.1283 | 0.1288 | **0.1300** |
| MRR | 0.1143 | 0.1146 | **0.1157** |

**B numerically leads every metric — but by a margin that doesn't survive scrutiny.** B−A = +0.0011 MRR (+0.97%); B−C = +0.0014 (+1.2%). Each arm is a single training run, with no seed ensemble and no confidence intervals, so a sub-1% gap sits inside run-to-run noise — read A and B as **tied**, not ranked (the methodological aside below shows how readily the sign flips). What is *not* noise: both content arms clear the no-content floor C. That lift over the floor looks small only because 90% of rollback targets are popular (Q4) movies where collaborative signal already dominates — the content story lives in the tail.

**MRR by popularity tier:**

| Tier (n) | C — no content | A — genome | B — LLM | B−A |
|---|---|---|---|---|
| Q4 popular (343,906) | 0.1259 | 0.1260 | **0.1273** | +0.0013 |
| Q3 mid (26,923) | 0.0129 | **0.0148** | 0.0144 | −0.0004 |
| TAIL ≤1k (12,652) | 0.0028 | **0.0033** | 0.0032 | −0.0001 |

![Genome and LLM lift over the no-content floor across popularity tiers](figures/fig1_tier_lift.png)

*Figure 1. Left: MRR by popularity quartile (log scale) — ranking quality drops \~100× from the popular head to the rarest tier, because collaborative signal thins out. Right: relative MRR lift over the no-content floor (C) — content adds \~0% on popular movies but +12–19% on the rare tiers, and the LLM (B) tracks the genome (A) rather than collapsing.*

Three things to read off this:

1. **Content earns its keep on the tail.** The Hit@250 lift over the floor (A−C) is \~0 on Q4 but **+0.0125 on the tail** — a relative MRR lift of roughly +18%. Both content sources help exactly where collaborative filtering is starved — the story Phase 1 structurally couldn't tell.
2. **The overall gap is a popular-head effect — and a suspect one.** B's whole-corpus lead over A is almost entirely Q4 (90% of examples; B−A ≈ +0.0013 stably across K; the whole-corpus number is essentially the Q4 result). That's backwards for a *content* feature, which should matter *least* where collaborative signal is strongest — and it lines up with a known leak: the LLM's reception group carries scraped box-office / IMDb-rating signal (**prestige-as-popularity leakage**, §7), which helps precisely on the popular head. The likeliest read is that the head "edge" is the LLM smuggling a popularity feature, not better content — one more reason to score A and B as tied on content quality. Isolating it cleanly would mean re-training B with the reception group ablated (§7).
3. **The LLM does *not* collapse on rare movies.** On the deep tail (Q1, TAIL) A and B are statistically tied — MRR gaps ≤ 0.0001, which should be read as ties, not rankings. Even where content matters most, LLM features hold even with the human-curated genome. *That* is the consequential result for "can LLMs replace human curation."

**A methodological aside:** the A-vs-B winner even flips with the checkpoint-selection rule — genome leads by +0.0004 MRR if you select on validation loss, the LLM by +0.0014 if you select on validation MRR — itself the tell that the two are tied, not ranked.

## 5. Why it works — what each source actually knows

Because both spaces sit on the same axes, we can correlate them directly. For each of the 132 LLM dims, the Pearson r against its mapped genome tag(s), across all 9,375 movies:

**Mean r = 0.598, median 0.608; 99 of 132 dims at r ≥ 0.5, none below 0.1.** The two are measuring the same thing. By group: visual **0.70**, setting 0.68, provenance 0.64 agree highest; themes and tone 0.56; **reception lowest at 0.42**.

- **Best axes (factual):** vampires 0.94, documentary 0.89, animated/anime 0.88, western 0.86, WWII 0.86, time-travel 0.85.
- **Worst axes (crowd-sentiment):** imdb_top_250 **0.16**, criterion **0.18**, palme_dor 0.27.

![Distribution of the 132 per-dimension genome-vs-LLM Pearson correlations](figures/fig2_agreement_hist.png)

*Figure 2. Per-dimension agreement between each LLM feature and its source genome tag, across all 9,375 movies (mean r = 0.60; 99 of 132 dims at r ≥ 0.5, none below 0.1). Agreement is highest on factual axes — genre, era, medium — and lowest on crowd-prestige axes (imdb top 250, criterion): exactly the slice an LLM can't read from a synopsis.*

That split *is* the mechanism. The LLM reproduces nearly all of genome's signal on the axes it can reach from text — genre, setting, provenance, factual medium — which is why B matches A on the bulk metrics. Genome's residual advantage is concentrated precisely where agreement is lowest: **crowd-prestige** ("masterpiece," "imdb top 250"), **fine niche sub-genre** granularity (*The Good, the Bad and the Ugly*: genome's "spaghetti western" + "ennio morricone" vs the LLM's coarser "western"), and **subjective aesthetics**. The LLM, in turn, contributes clean plot facts genome buries — "artificial_intelligence" for *2001*, "based_on_book" for *Die Hard*, "hitman"/"conspiracy" for *Sicario*.

**Qualitative color (seed-dependent — not a headline).** Top-10s for canary personas show the two sources give the model different *personalities*: genome leans niche-canon-pure (tight Western, slow-burn Arthouse, cerebral Sci-Fi), the LLM leans era- and modern-subgenre matching (2000s-gore Horror, 2010s gritty Crime) but drifts to blockbusters more readily on niche genres. Five illustrative disagreements:

| Persona | Genome (A) leans | LLM (B) leans |
|---|---|---|
| Sci-Fi | cerebral — Brazil, Gattaca, Forbidden Planet | popcorn — Fifth Element, T2, Total Recall |
| Crime | drifts to finance — Big Short, Margin Call | nails gritty — Sicario 2, Hell or High Water |
| Western | tight canon — Searchers, Rio Bravo | drifts to war epics — Patton, Braveheart |
| Arthouse | slow-burn — Stalker, In the Mood for Love | prestige — Fight Club, American Beauty |
| Horror | 90s slashers — Scream 2/3, Ring | 2000s gore — Saw II/IV/V, House of Wax |

Treat this strictly as color — the tier metrics, not the canary, carry the conclusion.

## 6. The payoff: feasibility, speed, cost

Here's where "good enough" cashes out. The point was never that the LLM features are 1% better — it's that they're the option a real team can actually build. Dimension by dimension (every replication dollar figure below is a labeled **estimate** — the genome papers publish none):

| Dimension | Genome (GroupLens) | LLM extraction (this repo) |
|---|---|---|
| Human labeling | 50,203 judgments (676 volunteers); **\~$5k–23k (central \~$10k)** to buy as crowd labor today *(est)* | **Zero** |
| Binding prerequisite | A \~15-year folksonomy (186k users / 17M ratings / 246k tag applications, since 1997) + crawled IMDb reviews — accrues with usage, not buyable quickly | The item's own text — exists day one |
| Specialist engineering | LSI/SVD features + rating-affinity + text-mining + a 6-model regression bake-off; **weeks–months** *(est)* | Scraper + schema derivation + 6 grouped prompts; **a few engineer-days** |
| Direct $ (this corpus) | \~$10k survey + tens of $k labor, *only if you already own the folksonomy*; **\~$85–210k** to hand-curate without one *(est)* | **\~$0 marginal as run**; **\~$170–220** if reproduced on the metered API (Sonnet $3/$15) |
| Wall-clock to build, full corpus | Accreted over \~15 years of community use | **\~1 day** for all \~9,375 items — independent calls, fanned out in parallel |
| Time-to-first-vector, new item | Effectively **never** until the crowd tags it | **Sub-hour**, zero-shot from text |
| Maintenance / new domain | Re-survey + re-crawl + re-train; new domain = redo everything | Parallel API calls per item |
| Quality (this ablation) | A: MRR 0.1146 | B: 0.1157 — **statistically tied** (sub-1% gap, within run-to-run noise) |

Three legs:

1. **Feasibility / build-vs-build.** The genome needs a mature folksonomy + specialist research; the LLM needs only item text. The LLM side is cheap but **not "$0"** — full-corpus extraction ran in a single day, consuming \~84% of one week's Sonnet quota on a Claude Max plan (≈$0 marginal under the subscription; \~$170–220 if reproduced on the metered API). Direct-dollar savings run roughly 1–3 orders of magnitude — but the durable claim is **feasibility, not price**: for a company without a folksonomy, the genome path isn't expensive, it's *unavailable*.
2. **Speed / cold-start.** Time-to-first-content-vector for a brand-new item: **sub-hour** (scrape + six calls) versus the genome's **effectively never** — its input features are all zero until the crowd tags the item.
3. **Cold-start bootstrapping (enabled-by, *not measured here*).** A sub-hour content vector lets you compute content-space nearest neighbors and seed a new item into the traffic of users who already get its most-similar items — warming up its collaborative embedding on far fewer impressions (cf. DropoutNet, NeurIPS 2017). This experiment didn't build or measure that; but it validates the premise it rests on — the r≈0.60 agreement and the Toy Story / Godfather similarity checks show the content NN is meaningful. Mind the axis, though: NN-seeding is a *rich-content-vs-no-content* benefit, so genome enables it too; it's **LLM-specific only at true cold start**, where genome doesn't exist to NN on in the first place.

**"Neural ≠ cheaper" sidebar.** The 2021 genome refresh (TagDL, a PyTorch MLP) bought \~2.6% MAE and changed *none* of the data prerequisites — same survey, same folksonomy. And the genome team's own 2026 cross-domain paper hit exactly the prerequisite wall extending to Amazon — *"the absence of … item-tag ratings and tag applications"* — had to reuse old survey labels, and took a measured accuracy hit. The strongest evidence that the prerequisite is binding comes from the people who built the genome.

## 7. Limitations

Limitations, stated plainly:

- **No significance testing — read A vs B as a tie.** Each arm is a single training run; no seed ensembles, no confidence intervals, no significance test. A sub-1% MRR gap (B−A = +0.97%) is inside run-to-run noise, and the §4 checkpoint-selection flip shows the sign of the gap is not even stable. The only ordering that survives is content (A, B) **>** no-content (C); A vs B is a statistical tie, not a ranking.
- **Two separate filterings, easy to conflate — and we filtered neither of the genome ones.** *(1) Our corpus:* all three arms train and evaluate on the **9,375 movies with more than 200 ratings** — *our* cutoff on MovieLens 32M's **87,585** titles, chosen for clean collaborative signal and tractable extraction. Both sources fully cover it (the genome scores all 9,375), so the head-to-head is scale-matched and fair, and the cost comparison above compares like for like. *(2) The genome's own ceiling:* **GroupLens never published genome scores for all 87,585 titles** — the file we downloaded from them covers exactly **16,376 movies (\~19% of the catalog)**, and that limit is *theirs*, not our doing: the other **71,209 (\~81%)** are too sparsely tagged and rated for their model to score at all. So the eval — restricted to the popular 9,375 that *both* sources cover well — can't even see the gap that matters most: across \~81% of the catalog the genome simply **does not exist**, while the LLM produces a vector for any title from its text alone (\~$1.6–2k metered for the whole catalog, *est*). And even the "TAIL ≤1k" tier above is movies with 200–1,000 ratings, *not* genuine cold-start. So the offline numbers almost certainly **understate** the real-world feasibility gap.
- **A true cold-start head-to-head is structurally impossible.** Cold start is exactly where an LLM content vector should pay off (§6) — but it cannot be benchmarked *against the genome*, because the genome doesn't exist there: it covers only the well-tagged head (the 16,376 titles above), so across the \~71k-title tail there is no genome arm to train or compare. "Retrain the ablation on the tail" isn't an experiment we declined — there is no genome signal to train arm A on. The most one could measure is an *LLM-only* hold-out-and-NN-seed simulation (a separate experiment, not run here).
- **Single LLM.** Claude Sonnet only, no cheaper-model bake-off. This supports "Sonnet-class extraction matches genome," **not** "any cheap model would."
- **The shared taxonomy is genome's — by design, to handicap the LLM (§2), not because the method needs one.** We held the LLM to genome's own tags so the match is tag-for-tag on genome's home turf. The honest residual caveat is narrow: we *showed* an LLM can fill a curated taxonomy to genome quality; an LLM *drafting* a richer taxonomy from scratch — which the scrape data would support, and which a greenfield team would do — is argued in §2, not separately benchmarked here.
- **Movies are a text-rich, easy case.** Every item here ships with a Wikipedia plot, TMDB cast, and reviews — the extractor never had to work from thin text. Whether it still tracks a gold-standard signal on items with three sentences of description is unmeasured; "Sonnet-class extraction matches genome on text-rich movies" does **not** automatically extend to text-poor domains.
- **Cost is amortized, not zero.** The \~$0 is *marginal dollars* under a flat-rate subscription, not a per-call API figure.
- **Crowd-sentiment is a scope choice, not a hard limit.** Genome's pure-sentiment tags ("masterpiece," "predictable," "overrated") are the axis the LLM trails on (r≈0.16–0.18) — but not because an LLM can't read sentiment. We deliberately confined scraping to Wikipedia + TMDB; pointed at critic and audience reviews, the same pipeline recovers reception signal too. It's a boundary we drew here, not an inherent genome-only advantage.
- **Fine-grained niche taxonomy.** Where genome keeps a real, if narrow, edge: the sub-genre and auteur detail the coarse LLM tags miss — *spaghetti western* + *ennio morricone* where the LLM lands only on *western* (§5). It shows up on the tight Western and Arthouse canon.
- **Prestige-as-popularity leakage.** Scraped box-office / IMDb-rating in the reception group are quasi-popularity signals, in mild tension with a comparison meant to isolate *content* — which is why that group is separately ablatable.
- **Possible training-data contamination — and it cuts toward the LLM.** MovieLens is one of the most-discussed datasets online and the genome tags are public, so the extractor may be partly *reciting* genome-adjacent knowledge rather than reading it off the synopsis we fed it. That would inflate the r≈0.60 agreement (and the tie) specifically on this much-discussed corpus, and need not transfer to a novel, undiscussed catalog. Untested here; a fresh-catalog replication is the clean check.

## 8. Takeaway

For a team without a pre-existing folksonomy — which is most teams, and every new product — **LLM extraction is the pragmatic default**, and it recovers nearly all of the genome's content signal on the axes reachable from text. The residual genome edge is the crowd-sentiment / fine-aesthetic slice you only get from a community you may not have.

The deeper point: the genome and the LLM are two generations of the *same idea* — propagate a dense content matrix from cheap signals. GroupLens propagated a 50,000-judgment survey across millions of (movie, tag) cells with regression, standing on fifteen years of community data. The LLM generation propagates from the item's own text — and in doing so removes the community, the folksonomy dependence, and the cold-start wall. It trades a thin slice of crowd-curated nuance for the ability to run on day one, for any item, at any company. For the content-feature problem most teams actually have, that's the trade you want.

---

*Sources & notes: results come from a held-out rollback evaluation — all 19,134 validation users, 382,138 ranking examples. Genome-construction facts are drawn from GroupLens's tag-genome work (Vig, Sen & Riedl, 2010/2012) and its later cost/feasibility line (TagDL, SIGIR '21; book genome, CHIIR '22; cross-domain genome, CHIIR '26), with cost anchors from public MTurk/Prolific and Pandora figures. Every dollar figure is a labeled estimate — the genome papers publish none. Full code, data pipeline, and per-tier eval outputs are in the repository.*
