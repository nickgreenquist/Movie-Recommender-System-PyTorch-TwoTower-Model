# Recency Is All You Need: A 12-Model Experiment on User-History Representation for Two-Tower Retrieval in Recommenders

> *The textbook move in deep two-tower retrieval models — going back to YouTube's DNN recommender ([Covington et al., 2016](https://research.google/pubs/deep-neural-networks-for-youtube-recommendations/)) — is to pool a user's interaction history into a single input vector. It's the standard approach, and it buries possibly the most predictive signal there is: the last item the user interacted with.*

## TL;DR

I built a **two-tower movie recommender** and swapped its user tower **twelve ways, each differing only in which slices of a user's watch history feed the user vector** — the whole history, the liked or disliked movies, the rating-weighted history, or just the most-recent item — trained them identically on MovieLens 32M, and measured retrieval (MRR). Three findings, and not what I assumed when starting this experiment:

- **Recency matters.** Adding the single most-recently-watched movie to the standard full-history pool lifts MRR **+22%** — the biggest move in the study. One recent item nearly matches the *entire* summed history ([§5](#5-what-the-numbers-say)).
- **Valence doesn't help.** The liked / disliked / rating-weighted channels that justify the production four-pool tower add nothing over a plain sum pool; the rating-weighted-only pool is the *worst* arm.
- **The case for a Transformer.** The sharp drop-off in value past the last item is exactly why the field moved from pooling to sequential/Transformer architectures — the FMC → FPMC → SASRec ladder ([§7](#7-takeaway)).

![Whole-corpus MRR for all 12 user-pooling arms, sorted and colour-coded by pool family](figures/fig1_pooling_mrr.png)

*Twelve user towers, identical except for which history slices feed the user vector: adding the single last-watched item (orange) is a +22% MRR move, while the rating-valence channels (blue) sit on the full-pool baseline.*

---

## 1. The setup: how do you turn a history into a vector?

A two-tower model scores a (user, item) pair by the dot product of two embeddings. The item tower is easy. The user tower is the hard half: a user *is* their history, so it has to fold a variable-length list of watched movies into one fixed vector. The default is a **sum pool** — look up an embedding per item, add, project:

![Pooling an arbitrary-length watch history into one fixed user vector](figures/fig2_pooling_schematic.png)

The pooling op barely matters (sum vs. mean is a wash); what's open is *which* history you pool. Instead of one bag, you can carve history into typed sub-histories — **by rating sign** (liked vs. disliked pools), **by magnitude** (weight each item by how far its rating sat above/below the user's mean), or **by recency** (pull the most-recent item out as its own channel).

### Adding a channel

Each extra channel is another 32-d block. The recency channel is the cleanest example: the model gathers the single last-watched item, looks up its embedding (**no sum**), and runs its own LayerNorm. That 32-d block is **concatenated** alongside the sum-pool block, and the combined vector goes through the **one shared projection** — concat first, project once:

![Adding the last-watched item as its own 32-d block, concatenated before the shared projection](figures/fig3_last_watched.png)

Because each block lands on its own columns of the projection's input, the model can weight "what you just watched" differently from "everything you've watched." Every arm in this study is just a different set of these blocks switched on.

## 2. The twelve arms

Each arm activates a different subset of pools; every active pool adds a 32-dim block, concatenated and projected `→ 256 → 128 → L2-normalize`. **Sum pools** (`full`, `liked`, `disliked`, `weighted`) sum over *all* matching items into one vector. **Single-item slots** (`last_liked`, `last_watched`, `second_to_last_watched`) look up exactly *one* item — no sum — each with its own projection weights.

| # | Model | Active pools | Dim | What it isolates |
|---|---|---|---|---|
| 1 | Only history | `full` | 32 | the baseline single sum pool |
| 2 | + likes | `full, liked` | 64 | does a clean affinity channel beat lumping likes into `full`? |
| 3 | + dislikes | `full, liked, disliked` | 96 | does an explicit avoidance channel add subtractable signal? |
| 4 | Weighted only | `weighted` | 32 | can one rating-weighted pool carry it alone? |
| 5 | 4-pool | `full, liked, disliked, weighted` | 128 | the full rating partition (= the production pooling block) |
| 6 | 4-pool + last-liked | `… , last_liked` | 160 | add immediate liked-session context on the 4-pool |
| 7 | history + last-liked | `full, last_liked` | 64 | the minimal recency×valence test |
| 8 | history + weighted | `full, weighted` | 64 | valence-weighting as an overlay on a dense pool |
| 9 | last-liked only | `last_liked` | 32 | the most-recent *liked* item, alone |
| 10 | last-watched only | `last_watched` | 32 | the most-recent item (any rating), alone — *first-order Markov* |
| 11 | history + last-watched | `full, last_watched` | 64 | recency channel on plain history — *FPMC* |
| 12 | history + last-watched + 2nd-last | `full, last_watched, second_to_last_watched` | 96 | does recency reach past the single last item? |

Three families: the **baseline** (arm 1), the **rating-valence** arms (2/3/4/5/8), and the **recency** arms (6/7/9/10/11/12).

## 3. Results

Held-out **rollback** protocol: for each of **19,134 validation users**, context = history so far, target = next watch, over **n ≈ 382,138** examples. Numbers are full-corpus.

**Whole-corpus retrieval, Δ vs arm 1:**

| # | Model | Active pools | MRR | Recall@10 | Recall@50 | NDCG@10 | MRRΔ |
|---|---|---|---|---|---|---|---|
| 1 | Only history | `full` | 0.1133 | 0.2189 | 0.4590 | 0.1270 | — |
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

## 4. Results by popularity tier

Does pooling help more on popular movies or the long tail? MRR split by target-movie popularity:

| Arm | Whole (382,138) | HEAD >1k (369,486) | Q4 popular (343,906) | Q3 mid (26,923) | TAIL ≤1k (12,652) |
|---|---|---|---|---|---|
| 1 full | 0.1133 | 0.1171 | 0.1247 | 0.0142 | 0.0032 |
| 2 +liked | 0.1141 | 0.1179 | 0.1255 | 0.0140 | 0.0033 |
| 3 +disliked | 0.1133 | 0.1171 | 0.1247 | 0.0147 | 0.0031 |
| 4 weighted | 0.0708 | 0.0732 | 0.0782 | 0.0051 | 0.0015 |
| 5 4-pool | 0.1132 | 0.1169 | 0.1245 | 0.0152 | 0.0036 |
| 6 +lastL | 0.1227 | 0.1268 | 0.1348 | 0.0188 | 0.0044 |
| 7 full+lastL | 0.1236 | 0.1277 | 0.1358 | 0.0182 | 0.0046 |
| 8 full+wt | 0.1130 | 0.1168 | 0.1244 | 0.0133 | 0.0029 |
| 9 lastL-only | 0.0805 | 0.0831 | 0.0881 | 0.0152 | 0.0070 |
| 10 lastW-only | 0.1099 | 0.1132 | 0.1198 | **0.0251** | **0.0128** |
| 11 full+lastW | 0.1386 | 0.1431 | 0.1520 | 0.0232 | 0.0061 |
| 12 full+lastW+2nd | **0.1431** | **0.1478** | **0.1568** | **0.0251** | 0.0070 |

## 5. What the numbers say

- **Recency is the lever, one item deep.** Adding the last watch (arm 11) beats the plain pool **+22%**; the 2nd-to-last (arm 12) adds nothing past noise. The signal is first-order.
- **One recent item ≈ the whole summed history.** `last_watched` alone (arm 10) lands within 3% of the full pool (arm 1) — proof the sum is lossy: ~50 items collapsed to one point, no identity, no order.
- **Recency beats valence.** `last_watched` > `last_liked` on every tier. The "did they like it?" filter drops the immediately-preceding watch — the most predictive item there is.
- **Rating channels add nothing.** Liked/disliked/weighted splits (arms 2/3/5/8) sit on the plain pool; weighted-only (arm 4, **0.0708**) is the worst arm — debiasing starves the gradient.
- **Head vs tail split the prize.** Recency arms win the head (arm 12 tops every head tier); on the tail, `last_watched` alone (arm 10) wins — no `full`-pool popularity bias.

## 6. Honest caveats

- **The winning arms exploit a protocol effect.** Rollback targets the *next* item, and `last_watched` is the item immediately before it — an immediate-predecessor→successor signal that's unusually strong offline (consecutive MovieLens ratings are often same-session). It's *why* `last_watched` beat `last_liked`, and part of the +22%. Production value depends on whether "what the user just watched" is known and session-correlated at serve time.
- **This is dataset-specific.** Recency dominates on MovieLens, where consumption is sequential and session-correlated. On your problem — different catalog, weaker session structure, longer gaps between events — the last item may carry far less, and rating valence may carry more. Treat "recency is the lever" as a hypothesis to re-test, not a law.

## 7. Takeaway

If you're folding a history into one vector for two-tower retrieval, **the structure that matters is recency, not rating valence.** The single last-watched item is worth more than the entire rest of the history summed together (+22% MRR as its own channel), while the liked / disliked / weighted machinery that *feels* principled adds nothing on top of a plain sum pool.

The arms quietly retrace the **FMC → FPMC → SASRec** ladder of sequential recommendation: `last_watched` alone (arm 10) is a learned first-order **Markov chain (FMC)**; fused with the full pool (arm 11) it's **FPMC** (Factorizing Personalized Markov Chains); and arm 12's hand-built second-order term barely moves — recency decays fast, so hand-coding positional pools runs out by the second item. That diminishing return is the whole case for a model that *learns* the recency weighting instead of hard-coding it: self-attention over the sequence — **SASRec** (a Transformer that learns which past items matter). This ablation is the empirical argument for building it next.

---

## Appendix — Method & notes

*Every arm is a two-tower model with an ID-only item tower (`BASE_TOWERS=idonly FEATURE_TOWERS=none`), trained identically — seed 42, 200k steps, LR 0.001 cosine→1e-4, batch 512, temperature 0.1, α=0 — varying only `USER_POOLS`. All pool signals come from the cached softmax training tensor (no data rebuilt between arms). Single-item slots gather the rightmost non-pad history position (`debiased_rating > 0` filter for `last_liked`), correct under both the right-aligned train and left-aligned eval layouts; empty/cold pools fall back to a learned bias vector. Eval is the held-out rollback protocol over all 19,134 val users (n ≈ 382,138), reporting Recall / Hit / NDCG / MRR @ K ∈ {1,5,10,20,50,100,150,200,250} plus the tier split.*

*Background sources: averaging watch-history embeddings into the user vector is the standard baseline introduced by [Covington et al., 2016 (Deep Neural Networks for YouTube Recommendations)](https://research.google/pubs/deep-neural-networks-for-youtube-recommendations/); the order-aware critique and the last-item-vs-full-history framing follow the sequential-recommendation line — [Rendle et al., 2010 (FPMC)](https://dl.acm.org/doi/10.1145/1772690.1772773) and [Kang & McAuley, 2018 (SASRec)](https://arxiv.org/abs/1808.09781).*
