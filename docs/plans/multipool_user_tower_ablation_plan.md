# Architectural Ablation: Combating Signal Dilution in Two-Tower Retrieval via Multi-Pool User Dynamics

> **What this is.** The build plan + experiment record for the multi-pool user-tower ablation.
> It isolates the contribution of *partitioning user history into semantic, rating-derived
> channels* against a plain mean/sum-pool baseline — the user-tower sibling of the
> [LLM-vs-genome](./llm_vs_genome_ablation_plan.md) study (which instead varies the *item* content
> slot). Result tables are **templates to fill after training** — do not cite numbers from them
> until the runs are in and the user has verified the eval.

## Thesis

Standard two-tower retrieval pools a user's *entire* interaction history into one vector — mean or
sum over every watched-item embedding — treating a 1★ hate-watch identically to a 5★ favourite.
That flattens intent and **dilutes the target vector**: positive and negative feedback are averaged
together, dragging the user point toward clusters the user actively dislikes. This ablation tests
whether splitting history into explicit **behavioral channels** (full / liked / disliked /
rating-weighted / last-liked), each with its own pool and LayerNorm, fused by the user
projection, **preserves high-fidelity preference signal and improves retrieval** (MRR / Recall@K) —
at **zero inference-time cost**, because the user tower still emits a single d-dim vector and the
item tower is untouched.

**Research question.** Does partitioning explicit user history into rating-derived pools beat
single-pool aggregation on retrieval metrics, and which channels carry the lift?

## TL;DR — what we learned *(all 12 arms: idonly base, 200k · α=0 · seed 42, full eval n=382,138)*

We tried many ways to pool a user's history. The result is clean — and mostly *not* what the thesis
predicted:

- **Recency is the whole game — the single most-recently *watched* item.** A user tower that sees
  *only* the last watched movie (arm 10, MRR **0.1099**) nearly matches one that sums the entire
  50-item history (arm 1, **0.1133**). Fusing the two — `full,last_watched` (arm 11, **0.1386**) — is
  the big win, **+22% over the full-history floor.**
- **Recency beats valence.** The most-recent *watched* item (any rating) is a far stronger next-item
  query than the most-recent *liked* item: arm 11 > arm 7 (`full,last_liked`) and arm 10 > arm 9 on
  *every* popularity tier. The like-filter throws away the immediately-preceding watch — the single
  most predictive signal — whenever it was rated below the user's mean.
- **The rating-channel thesis was wrong.** We expected splitting history into liked / disliked /
  rating-weighted pools to sharpen the user vector. It didn't — arms 2/3/5/8 all sit within the
  ±0.003–0.004 noise floor of the plain `full` pool, and rating-*weighted*-only (arm 4) actively
  collapses (−0.04, gradient starvation). Full-history pooling already summarizes taste; the valence
  channels are redundant with it.
- **Diminishing returns past the last item.** Adding the *2nd*-to-last watch as its own input
  (arm 12, **0.1431**) is the top number but only +0.0045 over arm 11 — at the noise floor. The
  signal is overwhelmingly first-order: one recent item carries it.
- **Head vs tail.** The seed-only arms own the long tail (arm 10 TAIL 0.0128, ~4× the full pool),
  because the popularity-biased `full` pool drags rare targets headward. So `full,last_watched`
  (11/12) wins the head metric; `last_watched`-alone (10) wins the tail.

**Why it matters.** Arm 10 is effectively a learned first-order Markov chain (FMC); arm 11 is
FPMC/Fossil (long-term implicit-user preference + short-term last-item transition). The ablation
independently walked the **FMC → FPMC → SASRec** ladder, and the order-2 diminishing return (arm 12)
is precisely why you stop hand-building positional pools and adopt a model that *learns* the recency
weighting — the motivation for the sister SASRec/Transformer repo. **Honest caveat:** the rollback
protocol makes `last_watched` the item immediately before the target, so the recency arms partly
exploit an immediate-predecessor→successor signal amplified by same-session MovieLens rating bursts.
Single-seed, so the *large* effects (recency arms, the weighted collapse) are solid; the *marginal*
ones (arm 12 vs 11, the individual rating channels) are noise-limited.

## Fixed setup — what makes it a clean test

Every arm is **identical** except for *which user-history pools are active*. To isolate the pooling
question we strip everything else to the bone — this is exactly the LLM-ablation `BASE_TOWERS=idonly
FEATURE_TOWERS=none` CF-base:

- **Item tower: ID embedding only.** No genre/tag/year towers, no genome/LLM content slot
  (`BASE_TOWERS=idonly`, `FEATURE_TOWERS=none`). The item side is `item_embedding_tower(lookup)`
  and nothing else — so any metric movement traces to the *user* tower.
- **User tower: pooling only.** No genre-affinity tower, no genome/LLM watched-context, no
  timestamp tower (all off under `idonly`). The user vector is built *purely* from history pools.
- **α = 0 for every arm.** No Menon popularity correction anywhere (it is a deployment knob, never
  a comparison knob — see CLAUDE.md). Cleanest variant comparison; matches all ablation work.
- **Full offline eval for every arm.** All 19,134 val users (n≈382,138), rollback protocol.
  `main.py eval` runs the full corpus by default (`EVAL_N_USERS` only caps it *down*, for smoke
  runs). Training uses the sampled val-MRR subset
  for checkpoint selection; the *reported* numbers are always full-corpus.
- **Training protocol (identical across all 12 arms; only `USER_POOLS` varies):** seed 42,
  **200k steps**, LR 0.001 cosine→1e-4, batch 512, temp 0.1, α=0, val-MRR checkpoint selection on a
  100k-example subset, full corpus. **Note:** the 200k schedule deepens the anneal vs the LLM
  ablation's 160k, so these arms are internally comparable to each other but **not** directly
  comparable to the 160k C′/A′/B′ figures — **every arm here, including Arm 1, is trained fresh at
  200k.**

## The arms

Twelve models, each `BASE_TOWERS=idonly FEATURE_TOWERS=none`, varying only `USER_POOLS` (new knob,
see Implementation). Each active pool contributes 32 dims (`item_movieId_embedding_size`) to the
user concat, projected `→256→128→L2-norm` as usual.

| # | Model | `USER_POOLS` | Pools | User concat dim | Status |
|---|---|---|---|---|---|
| **1** | Only history | `full` | full | 32 | New — retrain at 200k (architecturally ≡ LLM-ablation **C′**) |
| **2** | History + likes | `full,liked` | full, liked | 64 | New |
| **3** | History + likes + dislikes | `full,liked,disliked` | full, liked, disliked | 96 | New |
| **4** | Only rating-weighted | `weighted` | weighted | 32 | New |
| **5** | 4-pool | `full,liked,disliked,weighted` | full, liked, disliked, weighted | 128 | New |
| **6** | 4-pool + last-liked (**final**) | `full,liked,disliked,weighted,last_liked` | + last_liked | 160 | New |
| **7** | History + last-liked *(optional)* | `full,last_liked` | full, last_liked | 64 | New |
| **8** | History + weighted *(new)* | `full,weighted` | full, weighted | 64 | New |
| **9** | Last-liked only *(new)* | `last_liked` | last_liked | 32 | New |
| **10** | Last-watched only *(new)* | `last_watched` | last_watched | 32 | New |
| **11** | History + last-watched *(new)* | `full,last_watched` | full, last_watched | 64 | New |
| **12** | History + last-watched + 2nd-last *(new)* | `full,last_watched,second_to_last_watched` | full + last_watched + 2nd-to-last — **3 distinct inputs, separate weights (not a pool over the last two)** | 96 | New |

**Reconciling the thesis prose with the arm list.** The narrative frames "the 4 pools" as
liked / disliked / weighted / recency; the arm list (authoritative here) defines the **4-pool**
model as full + liked + disliked + weighted, with **last-liked as a 5th** signal added in arm 6.
This doc uses the arm-list definition throughout.

**"Last-liked" = a 5th pool, not the timestamp tower.** It is the ID embedding of the single
most-recent *positively-rated* history item (the *rightmost non-pad position with `debiased_rating>0`*
— see [Implementation](#implementation)), with its own LayerNorm — a short-term-context channel that
fuses recency with valence (the most-recent item the user actually *liked*, not a recent hate-watch),
living *inside* the pool system and orthogonal to the `idonly` strip. (Alternatives considered:
(a) the raw most-recent item regardless of rating — rejected, a recent low-rated watch is a noisy
session signal; (b) reuse the `watch_month` `timestamp_embedding_tower` — rejected, it encodes *when*
the user is watching, not *what* they last liked, and is stripped by `idonly`, muddying the
isolation. If a calendar-recency arm is wanted later, add it as arm 8.)

**Arm rationale — what each isolates:**
- **1 vs 2:** does adding a clean "affinity" (liked) channel beat lumping likes into the full pool?
- **2 vs 3:** does an explicit "avoidance" (disliked) channel — negative space — add signal the
  projection can *subtract*?
- **4 (weighted-only):** can a *single* rating-weighted pool recover most of the multi-pool lift on
  its own? The weighted sum already encodes valence (negative coefficients for disliked items), so
  4 is the cheapest "valence-aware" model — a strong, parsimonious baseline to beat.
- **1 vs 4:** unweighted-vs-weighted, same items, one pool each — the purest "does rating valence
  matter at all" contrast.
- **5:** the full partition — all rating-derived channels together (this is the prod user tower's
  pooling block, in isolation).
- **6 (final):** add immediate session context. Expected best.
- **7 (optional):** isolates the marginal value of the last-*liked* item on top of plain history,
  without the full rating-pool partition — the minimal recency×valence test (it still uses valence,
  via the like-filter on a single item, but not the liked/disliked/weighted pools).
- **8 (full + weighted):** the direct complement to arm 4 — adds the rating-weighted pool *on top of*
  the dense full-history pool. Arm 4 showed weighted-*alone* underfits: the debiased weighting
  starves the user-side gradient (most ratings sit near the user mean → near-zero weight, so the
  shared item embedding learns from only a few extreme-rated items per user — grad_norm ~0.8 vs
  arm 1's ~3.9, MRR plateauing ~0.067). Arm 8 tests whether the same valence signal becomes
  *additive* once a full-strength `full` pool carries the dense learning signal. **1 vs 8** isolates
  the marginal value of valence-weighting as an overlay; **5 vs 8** shows how much of the 4-pool lift
  is the weighted pool vs the explicit liked/disliked channels.
- **9 (last-liked only):** the extreme isolation — the model is blind to *everything* except the
  single most-recent liked movie (`_last_liked_ids` → one ID embedding → β fallback for users with no
  likes). It quantifies the recency×valence leak the rollback protocol rewards (see Honest caveats):
  the target is the chronologically next item, and `last_liked` is its most-recent positive
  predecessor. **9 vs 1** is the decisive contrast — if one liked movie rivals or beats the entire
  full-history pool, then much of every richer arm's score is really this "next ≈ last-liked" recency
  signal, not broad taste modeling; **6 vs 5** already measures last-liked's marginal lift *on top of*
  the 4-pool, but arm 9 bounds how much it carries *alone*.
- **10 (last-watched only) & 11 (full + last-watched):** the valence-free siblings of arms 9 and 7.
  `last_watched` is the most-recent item *regardless of rating* (rightmost non-pad), so it includes a
  recent hate-watch that the `last_liked` filter skips — the alternative this plan originally rejected
  as "a noisy session signal." **10 vs 9** and **11 vs 7** isolate the *value of the like-filter*: if
  last-watched ≈ last-liked, recency alone carries the lift (valence on the recency channel is noise);
  if last-watched < last-liked, the like-filter is doing real work.
- **12 (full + last-watched + 2nd-to-last-watched):** sibling of arm 11 with the second-most-recent
  watch added as a **separate input channel** — its own LayerNorm and its own columns in the
  projection matrix, **not summed/pooled with the last watch**. The two recent items stay
  position-distinct (the model can weight "most recent" vs "2nd-most recent" differently), making this
  a hand-built 2nd-order Markov term, not an order-agnostic bag of the last two. **12 vs 11** asks whether
  the recency signal extends past the single last item: if 12 ≈ 11, the lift is purely first-order
  (only the immediately-preceding watch matters — the clean FMC story); if 12 > 11 meaningfully,
  recency decays gradually and there's a case for modeling the fuller sequence (the motivation for a
  sequential/Transformer model). The rollback protocol inflates this the same way it does
  `last_watched` — 2nd-to-last in eval is `history[j-2]`, boosted by same-session co-occurrence.

## Relationship to the LLM-vs-genome ablation

Arm 1 is **architecturally** the LLM ablation's floor **C′** (`BASE_TOWERS=idonly FEATURE_TOWERS=none`):
a single full-history sum pool over the ID embedding, ID-only item tower. But C′ was trained at
160k steps (`best_softmax_idonly_popularity_alpha_0_20260610_081552.pth`, whole-corpus MRR 0.1121 at
160k) — under this campaign's **200k** protocol Arm 1 is retrained fresh, so C′'s 0.1121 is a
reference point, not the matched baseline. The two studies sit on the *same CF-base* and pull in
orthogonal directions:

- **LLM-vs-genome** holds the user tower fixed (single pool) and varies the **item content slot**
  (none / genome / LLM) → "does a content feature help, and which?"
- **This study** holds the item tower fixed (ID-only) and varies the **user pooling structure** →
  "does partitioning history by rating valence help, and which channels?"

Both answer the practitioner's question from the same minimal-CF starting point, so the deltas are
comparable in magnitude and the writeups reference one shared floor.

## Architecture

Reference — the canonical 4-pool user tower (`src/model.py:264–297`). Two kinds of channel feed the
user vector. **Sum pools** (full/liked/disliked/weighted) sum over the **raw** 32-dim
`item_embedding_lookup` (shared with the item tower). **Single-item slots** (last_liked, last_watched,
second_to_last_watched) look up exactly *one* item's embedding — no sum. Every channel has its own
LayerNorm and is **concatenated** into the user vector (never added to another channel), so each
lands in its own columns of the projection matrix — i.e. its own learned weights. In particular
arm 12's `last_watched` and `second_to_last_watched` are **two distinct inputs with separate weights,
NOT a pool over the last-two embeddings** — the model can weight "most recent" and "2nd-most recent"
differently:

```
pool_full              = LayerNorm( Σ  emb[history] )                         # always on
pool_liked             = LayerNorm( Σ  emb[history where debiased_rating>0] )
pool_disliked          = LayerNorm( Σ  emb[history where debiased_rating<0] )
pool_weighted          = LayerNorm( Σ (emb[history] · debiased_rating) )      # valence-weighted
last_liked   (slot)    = LayerNorm( emb[history last non-pad, debiased_rating>0] )  # single most-recent liked item
last_watched (slot)    = LayerNorm( emb[history last non-pad] )                     # single most-recent item, any rating
second_to_last (slot)  = LayerNorm( emb[history 2nd-from-last non-pad] )            # the watch BEFORE last — its OWN input/weights
concat(active channels) → Linear(256) → ReLU → Linear(128) → L2-normalize → user_emb (128)
```

`debiased_rating` is `rating − user_mean` (`dataset.py:399`), already stored as `X_hist_ratings`
(right-aligned, pad=0.0). Liked/disliked are pre-split right-aligned index tensors
(`X_hist_liked` / `X_hist_disliked`, `dataset.py:415–422`). **Every signal these pools need already
exists in the cached softmax 7-tuple — no Stage-1/2/3 rebuild.**

### The signal-dilution geometry (why valence channels help)

Take a concrete user. **Alice** likes 7 indie dramas (rated 4.5★, embeddings ≈ unit vector **d**)
and was dragged to 3 action blockbusters she rated 1★ (embeddings ≈ unit vector **a**), with the
two clusters roughly orthogonal (**d**·**a** ≈ 0). Her debiased mean is
`(7·4.5 + 3·1)/10 = 3.45`, so liked items carry weight `+1.05`, disliked `−2.45`.

**Single mean/sum pool (the baseline, arm 1).** The raw pool is `0.7d + 0.3a`; L2-normalized:

```
u ≈ 0.919·d + 0.394·a          (the disliked cluster pulls u 23° toward action)
```

Retrieval scores by cosine `u·eₜ`. An indie target (eₜ≈d) scores **0.919**; an action target
(eₜ≈a) scores **0.394** — *43% of the top score, earned purely by movies she hated*. With a deep
catalog that is enough to leak blockbusters into her top-K. **The dislikes have polluted the query.**

**Rating-weighted pool (arm 4 / inside arm 5).** The code weights each embedding by the debiased
rating before summing (`model.py:277–278`):

```
raw = 7·(+1.05)·d + 3·(−2.45)·a = 7.35·d − 7.35·a   →   u ≈ 0.707·d − 0.707·a
```

Now the action cluster has a **negative** projection (`u·a ≈ −0.707`): blockbusters are demoted
*below neutral*, a ~1.1-unit swing from the +0.394 the mean pool gave them. Valence is preserved,
not averaged away.

**Explicit liked/disliked channels (arms 3, 5).** The projection sees `pool_liked ≈ d` and
`pool_disliked ≈ a` as *separate* inputs and can learn `u ∝ d − λ·a` directly — actively pushing
the query away from the avoidance cluster rather than toward it. This is the multi-gate version of
the same repulsion the weighted pool gets implicitly; with both present the model can route
valence two independent ways.

The thesis: **partitioning by valence converts dilution into directional signal.** Arms 4 and 5 are
the two mechanisms (implicit weighting vs explicit gating); the results matrix measures whether the
geometry shows up in MRR/Recall.

## Handling sparsity in production

Real users have ragged history — no dislikes, brand-new profiles, one liked movie. The pooling
design degrades gracefully:

- **Empty pools → a learned null vector, not a crash.** A user with no disliked items has an
  all-padding `X_hist_disliked` row. `item_embedding_lookup` has `padding_idx = top_movies_len`
  (`model.py:126–128`), so pad rows embed to **zero**; the sum is the zero vector; `LayerNorm(0)`
  returns the layer's learned bias **β** — a constant "no-dislikes" signal the projection can use.
  No masking arithmetic, no NaN.
- **Weighted pool is divide-safe.** The user-context helper clamps the weight normalizer at
  `1e-6` (`model.py:260`); the weighted *pool* is a plain weighted sum (no division), so an
  all-neutral history yields a zero vector → β, same clean fallback.
- **Last-liked is resolved alignment-independently.** The training dataset right-aligns history
  (`dataset.py:412`) but the offline-eval path *left*-aligns it (`pad_history_batch`,
  `dataset.py:178`) — so a naive `[:, -1]` would be a pad index at eval time and silently corrupt
  the metric. The pool instead gathers the **rightmost non-pad position with `debiased_rating>0`**
  (`model.py:_last_liked_ids`), which is the most-recent *liked* item under *both* layouts (both
  order history oldest→newest). A user with no liked items (or a cold all-pad serving row) has no
  such position → zero → β, a benign default — the same graceful fallback as the empty liked/disliked
  pools.
- **No retraining to represent a new user.** Consistent with the project's core design (CLAUDE.md):
  any user is built at inference from a few liked movies — adding pools keeps that property, since
  each pool is a pure function of the input history tensors.

The right-alignment + `padding_idx` convention *is* the mask: zero-padding at the left, real items
at the right, pad rows contributing zero by construction.

## Inference efficiency — zero added latency

The architectural cost of multi-pool lives entirely in **feature prep + the user forward pass**,
both already trivial, and **nothing on the retrieval hot path changes**:

- **Item tower is untouched and pre-computed.** All corpus item embeddings are baked to
  `serving/movie_embeddings.pt` at export; the ANN index is built once, offline. Pools only alter
  the *user* tower.
- **The user tower still emits one d-dim vector.** Whether 1 pool (32-d concat) or 5 (160-d
  concat), the projection compresses to the same **128-d** query. Dot-product / ANN-search cost is
  **identical** — the query dimensionality downstream of the projection never changes.
- **Pool construction is O(history length).** Splitting liked/disliked is a couple of boolean
  gathers; the last-liked pool is a single masked gather. Negligible next to the embedding lookups the
  baseline already does.
- **No new serving data for arms 1–5.** `serving/feature_store.pt` already carries the
  liked/disliked/rating tensors (prod is a 4-pool model). Arm 6's last-liked needs only the
  existing history + rating tensors (the rightmost liked position) — no new artifact.

The feature-engineering pipeline gets slightly richer; **inference does not.** This is the
production headline: high-fidelity preference modeling for free at serving time.

## Training protocol

Seed 42, **200k steps**, val-MRR selection, α=0, full corpus — identical across all 12 arms (only
`USER_POOLS` varies). **Reuse the cached softmax dataset** — no `preprocess`/`features`/`dataset`
re-run; only the model config changes. Run each arm in the user's own terminal (never a
Claude-spawned background job — those run ~10× slower; see CLAUDE.md / memory):

```bash
# Arm 1 — retrained fresh at 200k (architecturally ≡ C′; the 160k C′ is NOT protocol-matched).
USER_POOLS=full                                BASE_TOWERS=idonly FEATURE_TOWERS=none python main.py train softmax   # 1
USER_POOLS=full,liked                          BASE_TOWERS=idonly FEATURE_TOWERS=none python main.py train softmax   # 2
USER_POOLS=full,liked,disliked                 BASE_TOWERS=idonly FEATURE_TOWERS=none python main.py train softmax   # 3
USER_POOLS=weighted                            BASE_TOWERS=idonly FEATURE_TOWERS=none python main.py train softmax   # 4
USER_POOLS=full,liked,disliked,weighted        BASE_TOWERS=idonly FEATURE_TOWERS=none python main.py train softmax   # 5
USER_POOLS=full,liked,disliked,weighted,last_liked  BASE_TOWERS=idonly FEATURE_TOWERS=none python main.py train softmax   # 6
USER_POOLS=full,last_liked                          BASE_TOWERS=idonly FEATURE_TOWERS=none python main.py train softmax   # 7
USER_POOLS=full,weighted                            BASE_TOWERS=idonly FEATURE_TOWERS=none python main.py train softmax   # 8
USER_POOLS=last_liked                               BASE_TOWERS=idonly FEATURE_TOWERS=none python main.py train softmax   # 9
USER_POOLS=last_watched                             BASE_TOWERS=idonly FEATURE_TOWERS=none python main.py train softmax   # 10
USER_POOLS=full,last_watched                        BASE_TOWERS=idonly FEATURE_TOWERS=none python main.py train softmax   # 11
USER_POOLS=full,last_watched,second_to_last_watched BASE_TOWERS=idonly FEATURE_TOWERS=none python main.py train softmax   # 12
```

Checkpoints should carry a pool-describing stem so the 12 arms don't collide in `saved_models/` or
`eval_results/` — e.g. `best_softmax_idonly_pools_<full-liked-disliked-weighted-last_liked>_alpha_0_<ts>.pth`.
(See the Implementation checklist — the stem must encode the pool set; config itself is re-resolved
from `hist_*_norm` weight keys at load, so renaming stays safe.)

## Eval protocol

Full-corpus offline eval for every arm — **all 19,134 val users**, which is now `main.py eval`'s
default. Avoid a small-user run: setting `EVAL_N_USERS` low overwrites the canonical
`eval_results/<stem>.txt` by stem (see memory).

```bash
python main.py eval saved_models/<stem>.pth      # full eval (all val users) by default → eval_results/<stem>.txt
```

Report Recall/Hit@K, NDCG@10, MRR @ K∈{1,5,10,20,50,...}, plus the popularity-tier split
(HEAD / Q1–Q4 / TAIL) the LLM ablation uses, so head-vs-tail behavior of each pool is visible.
Canary users are **optional qualitative color** here (the thesis is metric-driven); if run, use the
standard personas and read them as personality, not headline.

## Results matrix *(filled 2026-06-16; arms 10–12 added 2026-06-17 — 200k · α=0 · seed 42 · full eval, all 19,134 val users / n=382,138)*

**Whole-corpus MRR (n≈382,138), Δ vs arm 1 (single full pool):**

| # | Model | `USER_POOLS` | MRR | Recall@10 | Recall@50 | NDCG@10 | ΔMRR vs 1 |
|---|---|---|---|---|---|---|---|
| 1 | Only history | `full` | 0.1133 | 0.2189 | 0.4590 | 0.1270 | — (baseline) |
| 2 | + likes | `full,liked` | 0.1141 | 0.2208 | 0.4602 | 0.1280 | +0.0008 |
| 3 | + dislikes | `full,liked,disliked` | 0.1133 | 0.2204 | 0.4593 | 0.1274 | 0.0000 |
| 4 | Weighted only | `weighted` | 0.0708 | 0.1476 | 0.3484 | 0.0790 | −0.0425 |
| 5 | 4-pool | `full,liked,disliked,weighted` | 0.1132 | 0.2217 | 0.4613 | 0.1276 | −0.0001 |
| 6 | 4-pool + last-liked | `…,last_liked` | 0.1227 | 0.2417 | 0.4938 | 0.1391 | +0.0094 |
| 7 | history + last-liked | `full,last_liked` | 0.1236 | 0.2431 | 0.4936 | 0.1402 | +0.0103 |
| 8 | history + weighted | `full,weighted` | 0.1130 | 0.2199 | 0.4596 | 0.1270 | −0.0003 |
| 9 | last-liked only | `last_liked` | 0.0805 | 0.1688 | 0.3655 | 0.0917 | −0.0328 |
| 10 | last-watched only | `last_watched` | 0.1099 | 0.2100 | 0.4066 | 0.1241 | −0.0034 |
| 11 | history + last-watched | `full,last_watched` | 0.1386 | 0.2637 | 0.5129 | 0.1566 | +0.0253 |
| **12** | **history + last-watched + 2nd-last** | `full,last_watched,second_to_last_watched` | **0.1431** | **0.2724** | **0.5274** | **0.1619** | **+0.0298** |

(Arm 1 at 200k = 0.1133 vs the *160k* C′'s 0.1121 — consistent, so Arm 1 is the protocol-matched
floor. Arm 12 (`full,last_watched,second_to_last_watched`) is the new best — **0.1431, +0.0298 vs
arm 1** — but it edges arm 11 (`full,last_watched`, 0.1386) by only **+0.0045**, right at the
±0.003–0.004 single-seed noise floor: adding the *2nd*-to-last watch as its own input buys a
marginal, near-noise bump, so the signal is overwhelmingly **first-order** — the single last item.
`last_watched` also beats `last_liked` everywhere — arm 11 > arm 7 (`full,last_liked`, 0.1236) by
+0.0150, and arm 10 (`last_watched` only, 0.1099) > arm 9 (`last_liked` only, 0.0805) by +0.0294,
nearly matching the full pool (0.1133) with one movie. Recency dominates valence. Past the noise
floor, arms 2/3/5/8 stay indistinguishable from Arm 1 — likes/dislikes/weighted add nothing on a
dense `full` pool. Caveat: in the rollback protocol the most-recent watch is the item immediately
*before* the target, so arms 10/11/12 lean on an immediate-predecessor→successor signal — strong here
partly because consecutive MovieLens ratings are often same-session.)

**MRR by popularity tier** (does partitioning help more on the head or the tail?):

| Tier (n) | 1 full | 2 +liked | 3 +disliked | 4 weighted | 5 4-pool | 6 +lastL | 7 full+lastL | 8 full+wt | 9 lastL-only | 10 lastW-only | 11 full+lastW | 12 full+lastW+2nd |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Whole (382,138) | 0.1133 | 0.1141 | 0.1133 | 0.0708 | 0.1132 | 0.1227 | 0.1236 | 0.1130 | 0.0805 | 0.1099 | 0.1386 | **0.1431** |
| HEAD >1k (369,486) | 0.1171 | 0.1179 | 0.1171 | 0.0732 | 0.1169 | 0.1268 | 0.1277 | 0.1168 | 0.0831 | 0.1132 | 0.1431 | **0.1478** |
| Q4 popular (343,906) | 0.1247 | 0.1255 | 0.1247 | 0.0782 | 0.1245 | 0.1348 | 0.1358 | 0.1244 | 0.0881 | 0.1198 | 0.1520 | **0.1568** |
| Q3 mid (26,923) | 0.0142 | 0.0140 | 0.0147 | 0.0051 | 0.0152 | 0.0188 | 0.0182 | 0.0133 | 0.0152 | **0.0251** | 0.0232 | **0.0251** |
| TAIL ≤1k (12,652) | 0.0032 | 0.0033 | 0.0031 | 0.0015 | 0.0036 | 0.0044 | 0.0046 | 0.0029 | 0.0070 | **0.0128** | 0.0061 | 0.0070 |

(Whole-corpus MRR is ~90% Q4 by example count — 343,906/382,138 rollbacks — so it tracks Q4 and is
effectively a head metric. Two patterns: (1) `last_watched` dominates `last_liked` tier-for-tier —
arm 10 beats arm 9, and arm 11 beats arm 7, on *every* tier; arm 12 (full + last + 2nd-to-last) adds
~+0.005 over arm 11 on the head tiers (HEAD/Q4/Whole) and ties arm 10 for the best Q3, confirming the
2nd-to-last watch adds only a sliver beyond the last. (2) The **tail** still favors the seed-only
arms: arm 10 (`last_watched` only) posts the best TAIL by far (0.0128, ~2× arm 9, ~4× plain `full`),
while every arm carrying the `full` pool (7/11/12) sits at ~0.005–0.007 — the popularity-biased full
pool pulls rare targets toward the head. So arm 12 owns the head metric; arm 10 owns the tail. 160k
C′ reference tiers: Whole 0.1121 / HEAD 0.1159 / Q4 0.1234 / Q3 0.0133 / TAIL 0.0029.)

## Hypotheses vs. outcomes

- **Predicted:** monotone-ish lift 1 → 2 → 3 → 5, each rating channel adding a little, with the
  disliked channel and weighted pool carrying the most (as *signed* information the full pool can't
  represent). **✗ Refuted.** 1→2→3→5 is flat — all ~0.113, inside the noise floor; the rating
  channels add nothing on a dense `full` pool. The "recency on top" half held, but the lift comes
  from the *last-item* channel, and it's `last_watched` (arms 11/12), not the originally-planned
  `last_liked` (arm 6), that wins.
- **Predicted:** arm 4 (weighted-only) is the dark horse — might match arm 5 at a quarter the concat
  width. **✗ Refuted, badly.** Arm 4 is the *worst* arm (0.0708, −0.04): debiased rating-weighting
  starves the user-side gradient (most ratings sit near the user mean → near-zero weight), so the
  pool underfits. One signed pool does **not** substitute for the dense `full` pool.
- **Predicted:** lift concentrates on the head (Q4). **✓ Confirmed, with a twist.** Whole-corpus MRR
  tracks Q4 and that's where the arms separate — but the *seed-only* recency arms (9, 10) flip the
  tail, beating the full-pool arms there because they don't inherit the full pool's popularity bias.

## Honest caveats

- **Single seed (42); gaps may be inside the noise floor.** The LLM ablation established ±0.003–0.004
  run-to-run noise at the matched 160k protocol (expect comparable at 200k). Pool deltas could land
  inside it — so the conclusion rests on *consistency across tiers and the monotone ordering across
  arms*, not any single gap. All arms share seed 42 (no multi-seed averaging — the ordering, not a
  single-arm magnitude, carries the result).
- **The `idonly` base is a deliberately weak CF floor** (absolute MRR ~0.11). It drops the very
  rating-polarity pools we're studying from the *baseline* on purpose, to isolate them — so the
  measured lift is an **upper-ish bound** vs a richer base where genre/tags partly proxy the same
  signal. State this: the result is "partitioning helps *on a pure-CF base*," and the rich-base
  prod already ships the 4-pool, so the architecture's value is established separately.
- **The winning arms exploit recency the rollback protocol rewards.** The target is the
  chronologically next item, and `last_watched` is the item *immediately before* it — so arms 10/11/12
  lean on an immediate-predecessor→successor signal that's unusually strong in offline sequential eval
  (consecutive MovieLens ratings are often same-session). This is *why* `last_watched` beat
  `last_liked`: the like-filter forfeits that immediate predecessor whenever it was below the user's
  mean. Honest framing — part of the recency lift is a protocol/session effect; the production value
  depends on whether "what the user just watched" is known and session-correlated at serve time.
- **Not a prod comparison.** We strip to `idonly` to isolate pooling, not to beat prod. The recency
  arms (11/12, ~0.14 whole-corpus) do exceed the rich prod model's reported rollback MRR (0.1123),
  but that's confounded — arms are `idonly`/α=0/200k while prod is the rich `both`/`all` base at
  α=0.5, and α=0.5 deliberately trades offline MRR for less popularity drift — so it is **not** a
  clean "stripped beats rich" result. What it *does* suggest: `last_watched` is a large enough lever
  that adding it to the rich base would likely help (untested here; prod isn't slated for a retrain).

## Implementation

**Status: implemented, smoke-tested, and fully trained + evaluated (all 12 arms — see the Results matrix).** Three independent, fully
backward-compatible knobs were added; with no new env var the model is byte-identical to before
(same `state_dict` keys, same dims, no `hist_last_liked_norm`). The coarse `FEATURE_TOWERS`/`BASE_TOWERS`
selectors now only set *defaults*; the fine-grained knobs override them per side:

- **`USER_POOLS`** ⊆ `{full, liked, disliked, weighted, last_liked, last_watched, second_to_last_watched}` — which history pools the user tower sums.
- **`USER_FEATURES`** ⊆ `{genre, genome, llm, timestamp}` — user-side **non-pool** context towers.
- **`ITEM_FEATURES`** ⊆ `{genre, tag, genome, llm, year}` — item-tower features besides the always-on ID embedding.

(`USER_FEATURES`/`ITEM_FEATURES` were added alongside `USER_POOLS` so any user-side context tower or
item feature can be toggled independently — they generalize this experiment's "item ID-only,
user pooling-only" fixed setup into reusable knobs. For the 12 arms here, all three reduce to
`BASE_TOWERS=idonly FEATURE_TOWERS=none` + the varying `USER_POOLS`.)

What changed (all gated exactly like `feature_towers`/`base_towers`; surgical):

1. **`src/model.py`** — split the shared `has_genome`/`has_genre`/`has_llm` flags into per-side
   `has_user_*`/`has_item_*` (single source of truth); added `user_pools`/`user_features`/`item_features`
   constructor args (`None` → legacy set derived from `feature_towers`+`base_towers` via the new
   `default_user_pools`/`default_user_features`/`default_item_features` helpers); per-pool LayerNorms
   incl. `hist_last_liked_norm`; `n_pools = len(user_pools)`; canonical-order concat in `user_embedding`/
   `item_embedding`. Derived `has_genome`/`has_llm`/… aliases kept for external consumers
   (`export.py`, `tools/`). The `last_liked` pool uses `_last_liked_ids` (rightmost non-pad with
   `debiased_rating>0`) — correct under both the train (right-aligned) and eval (left-aligned) layouts.
2. **`src/train.py`** — `get_config` parses/validates the three env knobs into canonical-order
   lists (fail-fast on typos); `build_model` computes per-side buffer needs (genome/llm shared;
   genre/tag/year item-only) and threads the sets through; `print_model_summary` prints the
   effective pools/features; the checkpoint stem appends `pools-`/`uf-`/`if-` tokens **only when a
   set deviates from its default**, so existing runs keep byte-identical filenames and the 12 arms
   get distinct, self-describing stems.
3. **`src/checkpoint.py`** — `resolve_config_from_state_dict` resolves `user_pools` from the
   `hist_*_norm` keys and per-side features from the `item_*`/`user_*` tower keys, so any checkpoint
   (old prod, idonly, content-era-remapped, or new fine-grained) rebuilds from weights alone and
   `load_state_dict(strict=True)` at eval/export succeeds.
4. **`src/evaluate.py`** — `_setup`/`build_movie_embeddings` use the per-side flags (so eval works
   for asymmetric configs too). **No dataset rebuild** — every pool signal comes from the
   existing softmax 7-tuple. **`export.py`/`streamlit_app.py` untouched** (per project rule): the
   `None`-defaults make the prod `both`/`all` serving model byte-identical.

**Verification done (shapes/imports/round-trip only — not metrics):** a 60-check smoke suite passed —
default == legacy `state_dict` keys (no `hist_last_liked_norm`); all pool arms + asymmetric + weighted-only
build, forward, and backprop to `(B,128)` with correct concat dims; the recency-slot index helpers
(`_last_liked_ids`/`_last_watched_ids`/`_second_to_last_watched_ids`) are correct on
right/left/no-likes/single-item/all-pad inputs; save→resolve→`strict` load round-trips for prod-like, C′, weighted-only,
arm6, and asymmetric checkpoints; bad/empty env knobs raise. A 5-dimension adversarial review
(backward-compat, resolver, concat-order, buffer-needs, stem/consumers) found **zero issues**. All 12 arms have since been trained and full-eval'd; the verified numbers
are in the Results matrix above.
