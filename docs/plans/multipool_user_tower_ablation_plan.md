# User-Tower Pooling Ablation: Which History Representation Drives Two-Tower Retrieval?

> **What this is.** The experiment record for the multi-pool user-tower ablation — the user-tower
> sibling of the [LLM-vs-genome](./llm_vs_genome_ablation_plan.md) study (which varies the *item*
> content slot instead). Every arm is identical except for **which user-history pools are active**;
> we measure offline retrieval (MRR / Recall / NDCG) to see which pooling structure carries the lift.

**Research question.** Holding the item tower fixed (ID-embedding only), how should a user's watch
history be pooled into the user vector — a single full-history sum pool, rating-valence channels
(liked / disliked / rating-weighted), explicit recency channels (last item), or some combination —
and which of these actually moves retrieval metrics?

## TL;DR — *(all 12 arms: `idonly` base, 200k · α=0 · seed 42, full eval n=382,138)*

We pooled a user's watch history **12 different ways**, holding everything else fixed. One factor
explained almost all the variance in MRR: **whether the user tower can see the single
most-recently-watched item.**

- **The lever is one recent item.** `full,last_watched` (arm 11, **0.1386**) beats the plain
  full-history pool (arm 1, **0.1133**) by **+22%**. Adding the 2nd-to-last watch as its own channel
  (arm 12, **0.1431**) is the top score but only **+0.0045** over arm 11 — at the noise floor. The
  signal is **first-order**: one recent item carries it.
- **A single recent item ≈ the entire history summed into one embedding.** `last_watched` alone — one
  movie's embedding — (arm 10, **0.1099**) lands within 3% of arm 1 (**0.1133**), where all ~50
  history items are summed into a single pooled embedding before projection.
- **Recency beats valence-filtered recency.** `last_watched` > `last_liked` on *every* tier: arm 11
  (0.1386) > arm 7 `full,last_liked` (0.1236); arm 10 (0.1099) > arm 9 `last_liked`-only (0.0805).
  The like-filter discards the immediately-preceding watch whenever it was rated below the user's mean
  — and that's the most predictive item.
- **Rating-valence channels added nothing.** liked / disliked / weighted pools (arms 2/3/5/8) all sit
  at **~0.113**, inside the ±0.003–0.004 single-seed noise floor of the plain `full` pool. And
  rating-**weighted-only** (arm 4) is the *worst* arm (**0.0708**, −0.0425): one debiased-weighted
  pool starves the user-side gradient and underfits — it cannot substitute for a dense full pool.
- **Head and tail diverge.** Whole-corpus MRR is effectively a head metric (~90% of rollbacks are
  Q4-popular). The `full`-pool arms own the head; the **seed-only arms own the tail** — arm 10 posts
  the best TAIL by far (**0.0128**, ~4× the full pool) because it doesn't inherit the full pool's
  popularity bias. So arm 12 wins the headline number; arm 10 wins the long tail.

**Why it matters.** Arm 10 is a learned first-order Markov chain (FMC); arm 11 is FPMC/Fossil
(long-term implicit-user preference + short-term last-item transition); arm 12's diminishing return
shows the recency signal is overwhelmingly first-order. The full-history pool's ceiling is that it's a
**sum, not a sequence** — one projection over a pre-summed bag, with no per-item identity or order. A
model that keeps each history item's own embedding and *learns* how to weight them (recency,
relevance) — i.e. self-attention / a **transformer (SASRec)** — should recover the signal the sum
throws away. The ablation independently walked the **FMC → FPMC → SASRec** ladder; the order-2
diminishing return is exactly the cue to stop hand-building positional pools and adopt a sequential
model — the motivation for the sister SASRec/Transformer repo.

## Setup — what makes it a clean test

Every arm is **identical** except for `USER_POOLS`. We strip everything else to isolate the pooling
question (this is the LLM-ablation `BASE_TOWERS=idonly FEATURE_TOWERS=none` CF-base):

- **Item tower: ID embedding only.** No genre/tag/year towers, no genome/LLM content slot. So any
  metric movement traces to the *user* tower.
- **User tower: pooling only.** No genre-affinity, genome/LLM watched-context, or timestamp towers.
  The user vector is built purely from history pools.
- **α = 0 for every arm.** No Menon popularity correction (a deployment knob, never a comparison knob
  — see CLAUDE.md). Cleanest variant comparison.
- **Training (identical across all 12 arms; only `USER_POOLS` varies):** seed 42, **200k steps**, LR
  0.001 cosine→1e-4, batch 512, temp 0.1, α=0; val-MRR checkpoint selection on a 100k-example subset.
  Reuses the cached softmax 7-tuple — every pool signal already exists in it, **no
  preprocess/features/dataset rebuild**.
- **Eval: full corpus, every arm** — all 19,134 val users (n≈382,138), rollback protocol. Reported
  numbers are always full-corpus.

> **Note on comparability.** The 200k schedule deepens the anneal vs the LLM ablation's 160k, so these
> arms are internally comparable to each other but **not** directly comparable to the 160k C′/A′/B′
> figures. Arm 1 is retrained fresh at 200k as the protocol-matched floor.

## The arms

Twelve models, each `BASE_TOWERS=idonly FEATURE_TOWERS=none`, varying only `USER_POOLS`. Each active
pool contributes 32 dims to the user concat, projected `→256→128→L2-norm`.

| # | Model | `USER_POOLS` | Pools | Concat dim | Isolates |
|---|---|---|---|---|---|
| 1 | Only history | `full` | full | 32 | baseline single sum pool (≡ LLM-ablation C′) |
| 2 | + likes | `full,liked` | full, liked | 64 | does a clean affinity channel beat lumping likes into `full`? |
| 3 | + dislikes | `full,liked,disliked` | full, liked, disliked | 96 | does an explicit avoidance channel add subtractable signal? |
| 4 | Weighted only | `weighted` | weighted | 32 | can one rating-weighted pool recover the lift alone? |
| 5 | 4-pool | `full,liked,disliked,weighted` | full, liked, disliked, weighted | 128 | the full rating partition (= prod pooling block, isolated) |
| 6 | 4-pool + last-liked | `…,last_liked` | + last_liked | 160 | add immediate liked-session context on the 4-pool |
| 7 | history + last-liked | `full,last_liked` | full, last_liked | 64 | minimal recency×valence: last *liked* item on plain history |
| 8 | history + weighted | `full,weighted` | full, weighted | 64 | valence-weighting as an overlay on a dense `full` pool |
| 9 | last-liked only | `last_liked` | last_liked | 32 | the most-recent liked item, alone |
| 10 | last-watched only | `last_watched` | last_watched | 32 | the most-recent item (any rating), alone — FMC |
| 11 | history + last-watched | `full,last_watched` | full, last_watched | 64 | recency channel on plain history — FPMC |
| 12 | history + last-watched + 2nd-last | `full,last_watched,second_to_last_watched` | full + last + 2nd-last (3 distinct inputs, separate weights) | 96 | does recency extend past the single last item? |

**Key contrasts:** **1 vs 4** = does rating valence matter at all (unweighted vs weighted, one pool
each); **1→2→3→5** = does the rating partition add monotone lift; **5 vs 8** = how much of any 4-pool
lift is the weighted pool vs liked/disliked; **9 vs 1** = does one liked movie rival the whole
history; **10 vs 9** and **11 vs 7** = the value of the like-filter (recency vs recency×valence);
**12 vs 11** = is recency first-order or does it decay gradually.

## Architecture

Two kinds of channel feed the user vector. **Sum pools** (full/liked/disliked/weighted) sum over the
raw 32-d `item_embedding_lookup` (shared with the item tower) — collapsing all history into one
vector. **Single-item slots** (last_liked / last_watched / second_to_last) look up exactly *one*
item's embedding — no sum. Every channel has its own LayerNorm and is **concatenated** (never added),
so each lands in its own columns of the projection. In arm 12, `last_watched` and
`second_to_last_watched` are **two distinct inputs with separate weights, NOT a pool over the last
two** — the model can weight "most recent" vs "2nd-most recent" differently.

```
pool_full      = LayerNorm( Σ  emb[history] )                                    # sum pool, always-on baseline
pool_liked     = LayerNorm( Σ  emb[history where debiased_rating>0] )
pool_disliked  = LayerNorm( Σ  emb[history where debiased_rating<0] )
pool_weighted  = LayerNorm( Σ (emb[history] · debiased_rating) )                 # valence-weighted sum
last_liked     = LayerNorm( emb[rightmost non-pad item, debiased_rating>0] )     # single most-recent liked item
last_watched   = LayerNorm( emb[rightmost non-pad item] )                        # single most-recent item, any rating
second_to_last = LayerNorm( emb[2nd-from-rightmost non-pad item] )               # the watch BEFORE last — own input/weights
concat(active channels) → Linear(256) → ReLU → Linear(128) → L2-normalize → user_emb (128)
```

`debiased_rating` is `rating − user_mean` (`dataset.py:399`), stored as `X_hist_ratings`. Liked/disliked
are pre-split right-aligned index tensors (`X_hist_liked`/`X_hist_disliked`, `dataset.py:415–422`).
The single-item slots gather the **rightmost non-pad position** (with the `debiased_rating>0` filter
for `last_liked`) — correct under both the train (right-aligned) and eval (left-aligned) layouts.
Empty pools / cold users embed to zero → `LayerNorm(0)` returns the learned bias **β**, a graceful
"no-signal" fallback. No new serving artifact: `serving/feature_store.pt` already carries the
history+rating tensors every pool needs.

## Results

Run each arm in the user's own terminal (never a Claude-spawned background job). Checkpoints carry a
pool-describing stem so the 12 arms don't collide; config is re-resolved from the `hist_*_norm` weight
keys at load, so renaming stays safe.

```bash
USER_POOLS=full                                     BASE_TOWERS=idonly FEATURE_TOWERS=none python main.py train softmax   # 1
USER_POOLS=full,liked                               BASE_TOWERS=idonly FEATURE_TOWERS=none python main.py train softmax   # 2
USER_POOLS=full,liked,disliked                      BASE_TOWERS=idonly FEATURE_TOWERS=none python main.py train softmax   # 3
USER_POOLS=weighted                                 BASE_TOWERS=idonly FEATURE_TOWERS=none python main.py train softmax   # 4
USER_POOLS=full,liked,disliked,weighted             BASE_TOWERS=idonly FEATURE_TOWERS=none python main.py train softmax   # 5
USER_POOLS=full,liked,disliked,weighted,last_liked  BASE_TOWERS=idonly FEATURE_TOWERS=none python main.py train softmax   # 6
USER_POOLS=full,last_liked                          BASE_TOWERS=idonly FEATURE_TOWERS=none python main.py train softmax   # 7
USER_POOLS=full,weighted                            BASE_TOWERS=idonly FEATURE_TOWERS=none python main.py train softmax   # 8
USER_POOLS=last_liked                               BASE_TOWERS=idonly FEATURE_TOWERS=none python main.py train softmax   # 9
USER_POOLS=last_watched                             BASE_TOWERS=idonly FEATURE_TOWERS=none python main.py train softmax   # 10
USER_POOLS=full,last_watched                        BASE_TOWERS=idonly FEATURE_TOWERS=none python main.py train softmax   # 11
USER_POOLS=full,last_watched,second_to_last_watched BASE_TOWERS=idonly FEATURE_TOWERS=none python main.py train softmax   # 12

python main.py eval saved_models/<stem>.pth   # full eval (all val users) by default → eval_results/<stem>.txt
```

### Whole-corpus (n≈382,138), Δ vs arm 1

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

Arm 1 at 200k (0.1133) is consistent with the 160k C′ floor (0.1121), confirming it as the
protocol-matched baseline. Arm 12 is the best number but edges arm 11 by only +0.0045 — at the
±0.003–0.004 single-seed noise floor, so the recency signal is overwhelmingly **first-order**.
`last_watched` beats `last_liked` everywhere (arm 11 > arm 7 by +0.0150; arm 10 > arm 9 by +0.0294).
Arms 2/3/5/8 stay indistinguishable from arm 1 — likes/dislikes/weighted add nothing on a dense `full`
pool.

### MRR by popularity tier

| Tier (n) | 1 full | 2 +liked | 3 +disliked | 4 weighted | 5 4-pool | 6 +lastL | 7 full+lastL | 8 full+wt | 9 lastL-only | 10 lastW-only | 11 full+lastW | 12 full+lastW+2nd |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Whole (382,138) | 0.1133 | 0.1141 | 0.1133 | 0.0708 | 0.1132 | 0.1227 | 0.1236 | 0.1130 | 0.0805 | 0.1099 | 0.1386 | **0.1431** |
| HEAD >1k (369,486) | 0.1171 | 0.1179 | 0.1171 | 0.0732 | 0.1169 | 0.1268 | 0.1277 | 0.1168 | 0.0831 | 0.1132 | 0.1431 | **0.1478** |
| Q4 popular (343,906) | 0.1247 | 0.1255 | 0.1247 | 0.0782 | 0.1245 | 0.1348 | 0.1358 | 0.1244 | 0.0881 | 0.1198 | 0.1520 | **0.1568** |
| Q3 mid (26,923) | 0.0142 | 0.0140 | 0.0147 | 0.0051 | 0.0152 | 0.0188 | 0.0182 | 0.0133 | 0.0152 | **0.0251** | 0.0232 | **0.0251** |
| TAIL ≤1k (12,652) | 0.0032 | 0.0033 | 0.0031 | 0.0015 | 0.0036 | 0.0044 | 0.0046 | 0.0029 | 0.0070 | **0.0128** | 0.0061 | 0.0070 |

Whole-corpus MRR is ~90% Q4 by example count (343,906/382,138), so it tracks the head. Two patterns:
(1) `last_watched` dominates `last_liked` tier-for-tier (arm 10 > arm 9, arm 11 > arm 7 on *every*
tier); (2) the **tail** flips to the seed-only arms — arm 10 posts the best TAIL (0.0128, ~4× plain
`full`), while every arm carrying the `full` pool sits at ~0.005–0.007, because the popularity-biased
full pool drags rare targets headward. So arm 12 owns the head metric; arm 10 owns the tail. *160k C′
reference tiers:* Whole 0.1121 / HEAD 0.1159 / Q4 0.1234 / Q3 0.0133 / TAIL 0.0029.

## Caveats

- **Single seed (42).** The LLM ablation established ±0.003–0.004 run-to-run noise at the matched
  protocol. The *large* effects (the recency lift, the weighted-only collapse) are well outside it;
  the *marginal* ones (arm 12 vs 11, the individual rating channels) are inside it. The conclusion
  rests on consistency across tiers and the ordering across arms, not any single gap.
- **The `idonly` base is a deliberately weak CF floor** (absolute MRR ~0.11). It drops the genre/tag
  context that would partly proxy the same signal, so the measured pooling lift is an upper-ish bound
  vs a richer base. The result is "pooling structure helps *on a pure-CF base*."
- **The winning arms exploit recency the rollback protocol rewards.** The target is the
  chronologically next item, and `last_watched` is the item *immediately before* it — so arms 10/11/12
  lean on an immediate-predecessor→successor signal that's unusually strong in offline sequential eval
  (consecutive MovieLens ratings are often same-session). This is *why* `last_watched` beat
  `last_liked`. Production value depends on whether "what the user just watched" is known and
  session-correlated at serve time.
- **Not a prod comparison.** The recency arms (~0.14 whole-corpus) exceed the rich prod model's
  reported rollback MRR (0.1123), but that's confounded — arms are `idonly`/α=0/200k while prod is the
  rich `both`/`all` base at α=0.5 (which deliberately trades offline MRR for less popularity drift).
  What it *does* suggest: `last_watched` is a large enough lever that adding it to the rich base would
  likely help (untested here).

## Implementation

Three backward-compatible knobs override the coarse `BASE_TOWERS`/`FEATURE_TOWERS` selectors per side;
with no new env var the model is byte-identical to before (same `state_dict` keys, no
`hist_last_liked_norm`):

- **`USER_POOLS`** ⊆ `{full, liked, disliked, weighted, last_liked, last_watched, second_to_last_watched}` — user-tower history pools.
- **`USER_FEATURES`** ⊆ `{genre, genome, llm, timestamp}` — user-side non-pool context towers.
- **`ITEM_FEATURES`** ⊆ `{genre, tag, genome, llm, year}` — item-tower features besides the always-on ID embedding.

For the 12 arms, all three reduce to `BASE_TOWERS=idonly FEATURE_TOWERS=none` + the varying
`USER_POOLS`. Touched: `src/model.py` (per-side `has_user_*`/`has_item_*` flags, per-pool LayerNorms,
canonical-order concat, `_last_liked_ids`/`_last_watched_ids`/`_second_to_last_watched_ids` slot
helpers correct under both layouts); `src/train.py` (`get_config` parse/validate, `build_model` buffer
needs, pool-describing checkpoint stem only when a set deviates from default); `src/checkpoint.py`
(`resolve_config_from_state_dict` rebuilds the pool set from `hist_*_norm` keys so `strict` load
succeeds for any checkpoint); `src/evaluate.py` (per-side flags). **No dataset rebuild** — every pool
signal comes from the existing softmax 7-tuple. **`export.py`/`streamlit_app.py` untouched**: the
`None`-defaults keep the prod `both`/`all` serving model byte-identical.

Verified (shapes/imports/round-trip, not metrics): a 60-check smoke suite — default == legacy
`state_dict` keys; all pool arms build/forward/backprop to `(B,128)` with correct concat dims; the
slot index helpers correct on right/left/no-likes/single-item/all-pad inputs; save→resolve→`strict`
load round-trips; bad/empty env knobs raise. All 12 arms have since been trained and full-eval'd — the
numbers are in the Results matrix above.

## Relationship to the LLM-vs-genome ablation

Both studies sit on the same minimal CF-base (`BASE_TOWERS=idonly FEATURE_TOWERS=none`) and pull in
orthogonal directions:

- **LLM-vs-genome** holds the user tower fixed (single pool) and varies the **item content slot**
  (none / genome / LLM) → "does a content feature help, and which?"
- **This study** holds the item tower fixed (ID-only) and varies the **user pooling structure** →
  "which history representation helps?"

Arm 1 is *architecturally* the LLM ablation's C′ floor, but C′ trained at 160k
(`best_softmax_idonly_popularity_alpha_0_20260610_081552.pth`, whole-corpus MRR 0.1121); arm 1 is
retrained fresh at 200k, so C′ is a reference point, not the matched baseline.
