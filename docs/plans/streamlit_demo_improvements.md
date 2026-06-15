# Streamlit Demo Improvements — candidate features

**Audience:** ML engineers / CTOs evaluating recsys mastery (incl. live interview demos).
**Design principle (from the α toggle):** take a *recsys concept the interviewer will probe* and
make it **manipulable in the browser**, so the visitor *sees* the mechanism move — not a static
result. Every idea below is graded on that axis: how strong a signal it sends, and how cheaply it
ships against the artifacts we already export.

## What the app already signals well

- **Cold-start-free serving** (no user-ID embedding; taste built from a few likes) — the headline.
- **α=0.5 vs α=0 trained-twin toggle** — popularity-bias literacy, made interactive.
- **Similar tab 2×2** (genome vs LLM × learned vs raw) — the ablation thesis, made tactile.
- **Item-tower probes** (Genres / Genome) — "I can interrogate what the model encodes."
- **Serving story** in About — precompute-item-tower / one-matmul / device-agnostic export.

Gap: the app is strong on *retrieval mechanics and feature provenance*, but light on the
**beyond-accuracy** axis (diversity / novelty / calibration / explanation) — which is exactly
where senior recsys interviewers push hardest. Most ideas below fill that gap.

---

## Tier 1 — highest signal-per-effort, ships from existing artifacts

### 1. Diversity ⇄ relevance slider (MMR re-ranking)
**Concept:** Maximal Marginal Relevance — `score = λ·rel(i) − (1−λ)·max_{j∈chosen} sim(i,j)`.
The canonical post-hoc diversification every recsys IC knows.
**Control:** a `λ` slider on Recommend/Examples (and Similar). λ=1 → pure relevance (today's
output); slide down → intra-list redundancy gets penalized and near-duplicates drop out.
**Signal:** "beyond-accuracy" fluency + the insight that retrieval relevance ≠ a good *list*. This
is the single closest analog to the α toggle and the highest-value add.
**Feasibility:** trivial — greedy MMR over the top-N candidates using the 128-dim `all_embs` we
already load (candidate-vs-chosen cosine). No retraining, no new artifact. ~30 lines.
**Effort:** S.

### 2. "Why this?" per-recommendation explanation
**Concept:** feature-/neighbor-based explanation — *"recommended because it resembles X you liked
and shares tags a, b."*
**Control:** under each result card, a one-line attribution: the seed movie with the highest
cosine to this rec, plus the genome tags they share (intersection of each one's top-k genome
context).
**Signal:** explainability is the #2 thing interviewers raise after accuracy; turning an opaque
dot-product into a human reason reads as product maturity. Also makes every card more clickable.
**Feasibility:** from artifacts already loaded — `all_embs` (nearest seed) + `genome_tag_names` /
`movieId_to_genome_tag_context` (shared tags). No model change. The 4-pool tower is nonlinear so
this is a *faithful approximation* (neighbor attribution), not exact Shapley — label it honestly
as "most similar to your picks," which is what it is.
**Effort:** S–M.

### 3. Beyond-accuracy metrics strip
**Concept:** quantify a *list*, not just rank it — popularity percentile, intra-list diversity
(ILD), genre coverage, catalog coverage, novelty.
**Control:** a compact metrics row above any result feed (Recommend / Examples / Similar).
**Signal:** "I measure what matters past NDCG." It's also the **connective tissue** that makes the
other interactive knobs legible: flip α off → watch avg-popularity-percentile spike; pull the MMR
slider → watch ILD climb and genre coverage widen. The knobs become *demonstrations* instead of
vibes.
**Feasibility:** popularity percentile from `popularity_ordered_titles` rank (already exported);
ILD / coverage from `all_embs` + `movieId_to_genres`. (Optional: bake raw `mid_counts` into
`feature_store` at export for a true popularity-percentile rather than rank — a 3-line export
change.)
**Effort:** S–M.

---

## Tier 2 — high "wow," small new artifact or more UI

### 4. 2D map of the learned catalog
**Concept:** the embedding space *has structure* — show it.
**Control:** a precomputed UMAP/t-SNE projection of all ~9,375 item embeddings, colored by genre,
with the visitor's **taste vector** and their **recommendations** overlaid as highlighted points.
Hover = title/genres.
**Signal:** the strongest *visual* proof to a non-IC (CTO) that the model learned a coherent space;
to an IC it shows you can diagnose a space, not just score in it. Pairs beautifully with α/MMR —
watch the recommendation cloud tighten or spread.
**Feasibility:** compute 2D coords **offline at export** (add `item_coords_2d` to `feature_store`,
~1 line of UMAP + store an (N,2) array), render with `st.plotly_chart` (WebGL scatter handles 9k
points fine). The user vector projects via the same fitted transform (store the reducer, or
nearest-neighbor-place it).
**Effort:** M (offline projection + one new tab).

### 5. Taste blending / interpolation
**Concept:** the user is a *vector you can do algebra on* — the deepest implication of the
no-user-ID design.
**Control:** pick two personas (or two movie sets) A and B and a slider t; show recommendations for
`normalize((1−t)·u_A + t·u_B)`. Watch the list morph from one taste into the other.
**Signal:** demonstrates the representation is continuous and composable — a genuinely memorable
"I get embeddings" moment, and directly downstream of the architecture's central choice.
**Feasibility:** build the two user embeddings (existing `build_user_embedding`), lerp, renormalize,
re-score. No new artifact. SLERP optional but lerp+renorm is fine for unit vectors.
**Effort:** M.

### 6. Calibrated recommendations toggle (Steck 2018)
**Concept:** *calibration* — the genre mix of recs should track the genre mix of your history; pure
relevance often collapses to the user's single dominant genre.
**Control:** two small bar charts (your-likes genre distribution vs your-recs distribution) + a KL
number; optional toggle re-ranks to minimize that divergence.
**Signal:** name-drops a citable Netflix result (Harald Steck, RecSys'18) and shows you think about
*distributional* quality, not just top-1 relevance. Complements diversity without duplicating it.
**Feasibility:** genre histograms from `movieId_to_genres`; greedy calibrated re-rank mirrors the
MMR machinery. No model change.
**Effort:** M.

---

## Tier 3 — systems/scaling signal, optional

### 7. ANN vs exact retrieval
**Concept:** the demo serves with an *exact* full-catalog matmul (honest at 9k items); production
uses ANN. Show the tradeoff.
**Control:** a small benchmark (build a FAISS/HNSW index over `all_embs`) reporting recall@k vs
latency speedup vs exact — live, or as a static About-tab figure.
**Signal:** separates "trained a model" from "would ship it at catalog scale" — the scaling literacy
CTOs screen for. Naturally extends the existing serving narrative.
**Feasibility:** new dep (`faiss-cpu` / `hnswlib`); index builds in-process at load. Keep it small
or precompute the numbers to protect the free-tier cold start.
**Effort:** M–L.

### 8. Cold-start sensitivity curve
**Concept:** quantify the "embed you from a few likes" claim — how fast does the list stabilize as
seeds go 1 → N?
**Control:** add likes one at a time and plot a stability/metrics curve (e.g. rank correlation of
the top-K vs the previous step), or just re-rank live as seeds accumulate.
**Signal:** turns the headline claim into a measured curve. Educational and on-brand.
**Feasibility:** existing inference path, looped over prefixes. No new artifact.
**Effort:** S–M.

### 9. Quick wins
- **Negative steering in Recommend.** The inference path + disliked pool already support dislikes
  (Examples uses them); expose a "Movies you dislike" multiselect on the Recommend tab. Effort: S.
- **Latency readout.** "Scored 9,375 movies in X ms on a free CPU box." Time the matmul; one caption.
  Reinforces the serving story for ~5 lines. Effort: XS.

---

## What to skip / why

- **Temperature toggle** — like α, temperature is *training-time only*; a live toggle would need a
  third trained twin for marginal incremental signal over α. Not worth the artifact.
- **User-ID / session personalization** — intentionally absent (the whole thesis). Don't add; the
  About tab already frames it as a deliberate limitation.
- **A full embedding-projector clone** — the 2D map (#4) captures 90% of the value; a TensorBoard-
  style projector is scope creep for a single tab.

---

## Recommended sequence

1. **Metrics strip (#3)** first — it's the measurement layer the others demo *against*.
2. **MMR diversity slider (#1)** — highest-signal interactive knob; reads against the strip.
3. **"Why this?" (#2)** — cheapest perceived-quality jump; every card benefits.
4. **2D map (#4)** — the CTO-facing "wow," once the export gains a coords artifact.
5. **Taste blending (#5)** / **calibration (#6)** — pick one as the next conceptual showpiece.

Items 1–3 ship from today's `serving/` bundle with zero retraining and no re-export; 4 needs a
one-line export addition (`item_coords_2d`). None require touching the trained model.

---

## Sources (web grounding)

- Beyond-accuracy objectives (diversity / novelty / serendipity / fairness):
  [Fairness and Diversity in Recommender Systems: A Survey](https://arxiv.org/pdf/2307.04644)
- MMR re-ranking: [Carbonell & Goldstein, The Use of MMR for Reranking](https://www.researchgate.net/publication/2269571_The_Use_of_MMR_Diversity-Based_Reranking_for_Reordering_Documents_and_Producing_Summaries);
  [Re-ranking Based Diversification: A Unifying View](https://arxiv.org/pdf/1906.11285)
- Calibration: [A Framework and Decision Protocol to Calibrate Recommender Systems](https://arxiv.org/pdf/2204.03706)
- Explainability / feature- and neighbor-based explanations:
  [Hands-on Explainable Recommender Systems (RecSys'22 tutorial)](https://explainablerecsys.github.io/recsys2022/);
  [Using LLMs to Build Explainable Recommender Systems](https://towardsai.net/p/l/using-llms-to-build-explainable-recommender-systems)
- Two-tower retrieval + ANN serving:
  [The Two-Tower Model: A Deep Dive (Shaped)](https://www.shaped.ai/blog/the-two-tower-model-for-recommendation-systems-a-deep-dive);
  [Two Tower Model Architecture (reachsumit)](https://blog.reachsumit.com/posts/2023/03/two-tower-model/);
  [ANN-Benchmarks](https://arxiv.org/pdf/1807.05614)
- Popularity bias / softmax: [Test-Time Embedding Normalization for Popularity Bias](https://arxiv.org/pdf/2308.11288);
  [Sampled Softmax Powering Great Recommendations](https://medium.com/better-ml/sampled-softmax-powering-great-recommedations-f875659c2cd8)
- Embedding visualization of recsys spaces (t-SNE/2D clustering of learned embeddings): referenced
  in the popularity-bias survey work above.
