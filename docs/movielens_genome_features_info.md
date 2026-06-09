# The MovieLens Tag Genome: How It Was Built, and What It Would Take a Company to Replicate

> **Sourcing note (read first).** Every *construction* figure in this document is quoted from the
> primary source — the GroupLens technical report **"Computing the Tag Genome"** (Jesse Vig, Shilad
> Sen, John Riedl; Minneapolis, Sept 2010), published as **Vig, Sen & Riedl, "The Tag Genome:
> Encoding Community Knowledge to Support Novel Interaction," *ACM TiiS* 2(3), 2012** — or from the
> [GroupLens tag-genome dataset README](https://files.grouplens.org/datasets/tag-genome/README.html).
> Section numbers (§) refer to the 2010 report unless tagged **TiiS** (the expanded 2012 journal).
> **The papers report no dollar cost and no labor-hour figures** (verified by an exhaustive keyword
> sweep across both papers + README + release blog). Every cost/effort number for *replicating* the
> approach is therefore explicitly labeled an **estimate**, never presented as fact. Earlier drafts
> cited unsourced figures (~212k judgments, ~294 hours, ~$10.6k labels, ~$50–150k engineering,
> "Ridge + SVD") that did not survive a check against the paper; they have been removed. The dollar
> **anchors** that now appear (e.g. ~$10k to buy the survey, ~$85–210k for hand-curation) are
> *replication estimates* derived from present-day crowd-labor rates (MTurk/Prolific 2026) and
> published Pandora hand-curation figures — labeled as estimates, and distinct from the genome
> papers' own (still absent) cost reporting.

---

## What the Tag Genome is

The Tag Genome is a dense matrix of **tag relevance** scores. For each (movie, tag) pair it gives a
continuous value `rel(i, t) ∈ [0, 1]` — how strongly movie *i* exhibits the property named by tag
*t*. The paper's own examples (§2): `rel(Reservoir Dogs, violent) = 0.98`,
`rel(The Usual Suspects, violent) = 0.65`, `rel(A Cinderella Story, violent) = 0.03`.

The original 2014 standalone dataset spans **1,128 tags × 9,734 movies** ≈ **11 million relevance
scores**, every cell populated ([README](https://files.grouplens.org/datasets/tag-genome/README.html));
the genome **shipped with the larger MovieLens ratings datasets covers more** — **16,376 movies ×
1,128 tags ≈ 18.5M scores** in this project's MovieLens-32M data. But note the ceiling: **GroupLens
never scored the whole catalog.** That 16,376 is **only ~19% of MovieLens-32M's 87,585 titles** — the
other ~71k are too sparsely tagged/rated for their model to reach (the cold-start wall, below), so the
genome *does not exist* for them. (This is GroupLens's own coverage limit, baked into the dataset you
download — not a downstream filtering step on your end.) The genome is dense *within* its coverage but
covers only the well-tagged head. That density is the whole point: unlike raw user tags — which are
sparse, binary, and uneven — every *covered* movie has a calibrated score for every tag.

---

## How it was actually built (and how it was *not*)

~11M cells makes exhaustive human labeling a non-starter, and **GroupLens did not hand-label it.**
They labeled a tiny, high-quality slice and propagated the rest with machine learning.

**1 — A small gold-standard survey.** An in-platform survey asked MovieLens users to rate tag-movie
relevance on a **1 ("does not apply at all") to 5 ("applies very strongly")** scale. Tags had to be
applied by ≥10 users; movies had to have ≥100 ratings; each round showed a user 8 pages of one tag ×
6 movies (§4.1). A separate **85-user / 3,304-rating pilot** (50 tags) trained the stratified-sampling
model first. The entire human-labeled ground truth was:

> **676 users · 50,203 (item, tag) relevance ratings** (§4.1), linearly rescaled to 0–1.

That is **~0.46% of the 11M-cell matrix** — on the order of a couple of hours of effort per
volunteer (≈74 ratings each, *estimate*). The "expensive human curation" story is **not** a
giant-labeling story; the labeling was small.

**2 — Features mined from the existing community.** The other ~99.5% of cells were *predicted* by
regression, from features that are all reductions of content the MovieLens community had **already**
produced (§4.2):
- **Tag signals** — `tag-count`, `tag-applied`, and `tag-lsi-sim` (latent semantic indexing — an SVD
  over the item×tag matrix — to capture related tags).
- **Rating signals** — `avg-rating` and `rating-sim` (affinity between a tag and an item's rating
  pattern).
- **Text-review signals** — `text-freq` and `text-lsi-sim`, computed from **user-written movie
  reviews crawled from IMDb** (TiiS §4.2; ~35k words/movie on average).

**3 — Regression to fill the matrix.** Six regression models were compared (3 linear, 3
generalized-linear, in hierarchical/per-tag/single variants), 10-fold cross-validated in R. The
**generalized-linear hierarchical model (`glmer`) won, MAE 0.211** (2010 report Table 1 — the TiiS
2012 journal reports this only in a figure). `tag-count` and `tag-share` were dropped in feature
selection (no lift over `tag-applied`). That model scores all ~11M pairs.

**The prerequisite that's easy to miss.** Every input feature is a *reduction of a pre-existing
folksonomy*. At the time the genome was computed, MovieLens had (2010 report §3): **operation since
1997, 186,000 users, 17 million ratings, and 5,375 users who had applied 31,325 distinct tags across
246,000 tag applications** — plus crawled IMDb reviews. (The expanded TiiS 2012 journal, a build ~2
years later, reports the larger 198,000 users / 18M ratings / 6,166 taggers / 29,581 tags / 273,000
applications.) No folksonomy → no features → no genome.

**Who built it.** An NSF-funded academic research group (grants IIS 03-24851, 05-34420, 09-64695,
09-64697; §5) — i.e., specialist researcher-time, not a data-entry job. GroupLens has since updated
the *predictor* — the 2021 tag-genome dataset swaps the `glmer` regression for **TagDL**, a PyTorch
MLP ([Kotkov, Maslov & Neovius, SIGIR '21](https://doi.org/10.1145/3404835.3463019)) — for a ~2.6%
MAE gain that changed **none** of the data prerequisites: same tag applications, same ratings, same
crawled reviews, same human survey as training ground truth. "Newer/neural genome" ≠ "cheaper
genome." The **recipe is unchanged** — small human survey + features mined from a large folksonomy +
ML propagation — and so is its dependence on pre-existing community data.

---

## The framing that matters: you need content features for a recommender — which path is open to you?

Drop the "it cost $100k" mythology: the paper reports no cost, and the labeling itself was tiny
(~50k volunteer judgments). The decision a real team faces is **build-vs-build** — to get a dense,
genome-style content vector for *every* item, which path can you actually take?

### Path A — Replicate the GroupLens genome

*What you must already have or build (all figures **estimates**; the paper gives none):*
- **A mature interaction corpus** — hundreds of thousands of users, millions of ratings, hundreds of
  thousands of tag applications. GroupLens had 15 years of MovieLens. **A growing company does not**,
  and this cannot be bought quickly — it is the binding constraint.
- **Crawled per-item web reviews** for the text features.
- **A relevance survey** — ~50k+ judgments. GroupLens got them *free* from a passionate volunteer
  research community; a company without one must pay for them. *Estimate to buy the same 50,203
  judgments as crowd labor today:* **~$5k–$23k** (central **~$10k**) — single 1–5 Likert ratings are
  among the cheapest microtasks (~$0.10–$0.45 per usable judgment after 3–5× redundancy + gold
  checks, at MTurk/Prolific 2026 rates). Small, by design — it was a *seed* for ML, not exhaustive
  labeling.
- **Specialist ML engineering** — feature construction (LSI/SVD, rating-affinity, text mining) plus
  fitting and cross-validating the regression. Weeks-to-months of skilled time.
- **Ongoing maintenance** — re-crawl, re-survey, and re-train as the catalog grows; **a new domain
  means redoing the whole exercise** (the [book genome](https://grouplens.org/datasets/book-genome/)
  needed a fresh Goodreads folksonomy + a new 986-user / 145,825-rating MTurk survey).

*Realistic effort (estimate):* dominated by specialist engineering time (weeks-to-months → readily
tens of thousands of dollars of skilled labor) **and gated on already owning a large folksonomy.**
And it has a hard floor: for a **brand-new item, or a new company with no interaction history, the
input features are all zero — so it produces nothing usable** (the cold-start wall, below). *If you
have no folksonomy at all,* the only genome-style route is brute-force expert hand-curation
(Pandora-style, ~$9–$22/item *estimate*) — **~$85k–$210k just to tag this repo's ~9,375-movie corpus
once**, the very model the genome was designed to avoid.

### Path B — Scrape + LLM extraction (this repo's approach)

*What you need:*
- **The item's own text** — title and synopsis — which you always have, even for an item added one
  minute ago.
- **A scraper** for public metadata (TMDB / Wikipedia): a few engineer-days.
- **An LLM with structured (JSON-schema) output** to score the item against a fixed tag taxonomy.

*Realistic cost (grounded in this repo's actual run — see
[`docs/plans/llm_vs_genome_ablation_plan.md`](plans/llm_vs_genome_ablation_plan.md), Cost Budget):*
scraping is free; extraction for the full ~9.4k-movie corpus is **low-hundreds-of-dollars** of
metered LLM inference (and **~$0 marginal** when amortized under a flat-rate subscription — as it was
actually run, it consumed ~84% of one week's Sonnet quota on a Claude Max 5× plan, June 2026); the
core pipeline is **a few engineer-days**. It runs **on day one, for every item, including brand-new
ones.**

### The verdict

For a growing company, Path B isn't merely cheaper — **Path A is frequently *infeasible*,** because
its inputs (a large, mature folksonomy) don't exist yet. Path B needs only what you always have: the
item's own text. That reframes the whole comparison from a price tag to a **feasibility** question —
and it is the entire motivation for the experiment in this repo: *given that the LLM path is the one
most teams can actually take, how close is its content quality to the gold-standard human-curated
genome?*

---

## The cold-start wall (why Path A can't be rescued by money)

Even with unlimited budget, Path A fails for the items a growing catalog cares about most: brand-new
ones. Every genome input feature is a reduction of interaction history, so for a brand-new item:
- `tag-count` / `tag-applied` = 0 (no one has tagged it),
- rating features = 0 (no one has rated it),
- text-review features ≈ 0 (few or no reviews yet).

With all-zero inputs the regression can only emit an uninformative prior — useless for ranking. The
genome is **structurally a warm-catalog asset.** The LLM path, operating on the item's own text, is
genuinely zero-shot: a complete content vector the moment the item exists.

**The genome authors say so themselves.** Extending the recipe to Amazon, GroupLens' own 2026
cross-domain paper ([Kotkov et al., CHIIR '26](https://doi.org/10.1145/3786304.3787950)) hit exactly
this wall — *"the absence of … item-tag ratings and tag applications"* — and had to transfer old
survey labels onto matched items, drop the folksonomy features, and accept a **measured accuracy
hit** on the sparser data. The strongest evidence that the prerequisite is binding comes from the
people who built the genome.

*Caveat carried into the writeup:* the item still needs a model training pass to earn its
collaborative ID embedding before it can be served well — what's instant is the **content vector**,
which a new genome row cannot provide *at all*. Retraining is required either way, so it does not
separate the two approaches; *availability of a content vector* does.

---

## Build-vs-build summary

| | **Genome (GroupLens recipe)** | **LLM extraction (this repo)** |
|---|---|---|
| **Core input** | A large pre-existing folksonomy (tags, ratings, reviews) | The item's own text (title, synopsis) |
| **Works for a new company?** | No — needs years of community data first | Yes — needs only item text |
| **Works for a brand-new item (cold start)?** | No — all input features are zero | Yes — zero-shot from text |
| **Human labeling** | One-time gold standard: 50,203 ratings / 676 users (§4.1); ~$10k to buy today *(est)* | None |
| **Specialist effort** *(estimate)* | Weeks–months: feature engineering + regression + crawl | A few engineer-days |
| **Direct cost** *(estimate)* | ~$10k survey + tens of $k skilled labor, gated on owning a folksonomy; ~$85–210k to hand-curate this corpus without one | *Actual:* low-hundreds-$ metered (≈$0 marginal under subscription) |
| **Scales with catalog?** | Re-crawl + re-survey + re-train; bounded by folksonomy growth | Parallel API calls per new item |
| **New domain (e.g. books)?** | Redo everything — new folksonomy + new survey | Same pipeline, different text |
| **Time to first vector, new item** | Effectively never (until it accrues interactions) | Sub-hour |

*Construction figures: Vig, Sen & Riedl, "Computing the Tag Genome" (GroupLens tech report, 2010;
ACM TiiS 2012) and the [tag-genome dataset README](https://files.grouplens.org/datasets/tag-genome/README.html).
Predictor lineage: [TagDL (SIGIR '21)](https://doi.org/10.1145/3404835.3463019),
[book genome (CHIIR '22)](https://doi.org/10.1145/3498366.3505833),
[cross-domain (CHIIR '26)](https://doi.org/10.1145/3786304.3787950). Replication cost/effort entries
are labeled estimates — the genome papers report none; dollar anchors derive from 2026 MTurk/Prolific
microtask rates and published Pandora hand-curation figures. LLM-path costs are this repo's own run.*
