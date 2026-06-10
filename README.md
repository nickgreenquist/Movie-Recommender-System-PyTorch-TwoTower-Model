# 🎬 Movie Recommender — PyTorch Two-Tower Neural Network

> A deep two-tower recommender trained on **MovieLens 32M** that recommends movies to **any** user — including ones it has never seen — from nothing but a handful of films they like.

[![▶ Live Demo](https://img.shields.io/badge/▶_Live_Demo-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://movie-recommender-system-two-tower-model.streamlit.app/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org)
[![Dataset](https://img.shields.io/badge/Dataset-MovieLens_32M-FBBC04)](https://grouplens.org/datasets/movielens/32m/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

### 👉 **[Try the live demo](https://movie-recommender-system-two-tower-model.streamlit.app/)**

<p align="center">
  <img src="assets/demo-examples-scifi.png" alt="Sci-Fi Lover recommendations — the model returns deep-cut classics like Stalker, Forbidden Planet, and Silent Running, not just blockbusters" width="820">
</p>

<p align="center"><em>Tell it you love <strong>2001</strong>, <strong>Solaris</strong>, and <strong>Contact</strong> → it returns deep-cut classics (Stalker, Forbidden Planet, Soylent Green), not the IMDb Top 10.</em></p>

---

## Why this project is different

Almost every recommender tutorial — and most production systems — give each user a **learned ID embedding**. That works, but it has a hard limitation: **you can only recommend to users the model was trained on.** A brand-new user has no embedding, so you're stuck retraining, fine-tuning, or faking it with a "similar user" proxy.

**This model has no user-ID embedding at all.** A user is represented purely as a function of their *taste signals* — the movies they've watched, what they liked vs. disliked, their genre affinities, and the semantic "texture" of their history. The network learns to embed **features of the user, not the user's identity.**

The payoff: the same frozen model serves recommendations for *anyone* — including the visitor poking at the live demo right now, who didn't exist when the model was trained. **No retraining. No user-level cold-start.** Give it three movies you like and it builds your taste vector on the fly.

## ✨ Highlights

- 🧊 **Cold-start-free by design** — no user-ID lookup; new users are embedded from a few liked movies at inference time.
- 🏗️ **Two-tower architecture** — separate user & item towers project into a shared 128-dim space; recommendation = cosine similarity. Item embeddings are precomputed once, so scoring the whole catalog is a single matrix multiply.
- 🧮 **Full-softmax training** with **logit-adjusted popularity correction** ([Menon et al., ICLR 2021](https://arxiv.org/abs/2007.07314)) — **8.7× higher MRR** than the MSE baseline, with popular blockbusters surfacing *only when relevant*.
- 🧬 **Two content fingerprints per movie, fused in the item tower:** the dataset's **1,128 curated genome tags** *and* **132 features I built myself** — web-scraped from TMDB + Wikipedia and scored 0–1 by an LLM.
- 💰 **A cost lesson, measured**: self-built features from scraped web text + an LLM (**~half a day, no annotators**) **match** MovieLens's expensive human-curated genome tags head-to-head — so a company can bootstrap content features instead of waiting on a tagging pipeline. ([see the ablation](#-results))
- 🚀 **Live, interactive Streamlit app** with poster grids, pre-built taste personas, "similar movies," and an embedding-space explorer.
- 🍏 **Apple Silicon (MPS) accelerated** training/eval (~145 it/s), with device-agnostic CPU export for cloud serving.

---

## 🧠 How it works

### Two-tower architecture

Both towers consume different inputs but emit an **L2-normalized 128-dim vector**. Because both are unit-norm, their dot product *is* cosine similarity — so a single `user · item` product ranks the whole corpus.

```
USER TOWER (no user ID — built from taste signals)
  sum_pool(item_emb[full history])            → pool_full      (32)  ┐
  sum_pool(item_emb[liked history])           → pool_liked     (32)  │ 4 history pools,
  sum_pool(item_emb[disliked history])        → pool_disliked  (32)  │ shared item-ID table
  rating_weighted_sum(item_emb[history])      → pool_weighted  (32)  ┘
  user_genome_tower(avg genome over history)  → genome_ctx     (32)
  user_llm_tower(avg LLM feats over history)  → llm_ctx        (32)
  user_genre_tower(per-genre affinity)        → genre_emb      (32)
  timestamp_tower(watch month)                → ts_emb          (4)
        concat(228) → Linear(256) → ReLU → Linear(128) → L2-norm → user_emb (128)

ITEM TOWER
  item_embedding_tower(movie_id)              → item_id_emb    (32)  ← shared with user pools
  item_genome_tag_tower(1,128 genome scores)  → genome_emb     (32)
  item_llm_feature_tower(132 LLM features)    → llm_emb        (32)
  item_genre_tower(genre one-hot)             → genre_emb       (8)
  item_tag_tower(user-applied tags)           → tag_emb        (16)
  year_embedding_tower(release year)          → year_emb        (8)
        concat(128) → Linear(256) → ReLU → Linear(128) → L2-norm → item_emb (128)

SCORE = dot(user_emb, item_emb)   # = cosine similarity (both unit-norm)
```

**The shared item-ID embedding is the key trick.** The same 32-dim table embeds the *target movie's* identity **and** pools the user's watch history (full / liked / disliked / rating-weighted). This forces user history and item identity into the *same* space — a movie you liked pulls your user vector straight toward that movie; a disliked movie pushes it away.

### Content fingerprints: curated *and* self-built

The item tower describes each movie from several content signals. Two are **rich, dense semantic fingerprints** — and they're the swappable slot the [ablation](#-results) studies:

| Source | Dims | Origin |
|---|---|---|
| **Genome tags** | 1,128 | Ships with MovieLens — ML-derived relevance scores per (movie, tag). |
| **LLM features** | 132 | **Built end-to-end in this repo** — scraped from TMDB + Wikipedia, then scored 0–1 by an LLM across six groups: themes & plot (32), tone & mood (26), setting/era/sub-genre (32), provenance & structure (24), reception & prestige (11), visual medium (7). |

The item tower also folds in three lighter content signals: **genre** (20-way one-hot), **user-applied tags** (306 community tags — present, but far sparser and noisier than the genome scores), and **release year**.

The deployed model includes **both** rich towers (`FEATURE_TOWERS=both`) for maximum feature robustness — though, as the [ablation](#-results) shows, *either source alone matches the other*. The self-built LLM pipeline (scrape → LLM-extract → tensor) lives in [`llm_features/`](llm_features/).

### Popularity-bias correction (Menon et al., 2021)

Full softmax has a structural bias: popular movies appear in *every* batch as hard negatives **and** as frequent positives, which drags their embeddings toward the "average" user — making them look relevant to everyone. The fix is the **logit-adjusted loss**: add `α · log(interaction_count)` to each item's logit *during training only*. Popular items get a free score boost, so the model no longer needs to inflate their embeddings to win — and they shrink back to where they belong. At inference, scoring uses **raw dot products** (no post-hoc correction).

`α = 0.5` is the deployed setting — the sweet spot where genre discrimination stays sharp and blockbusters surface only when they actually fit the taste.

### Training at a glance

| | |
|---|---|
| **Dataset** | MovieLens 32M — ~33M ratings, ~200K users, ~86K movies |
| **Corpus** | movies with 200+ ratings (~9,375) · users with 20–500 ratings |
| **Loss** | full softmax cross-entropy over all corpus items + Menon logit adjustment |
| **Optimizer** | Adam, lr=1e-3, batch=512, temperature=0.1, 150K steps |
| **Split** | user-level 90/10 — no user appears in both train and val |
| **Protocol** | *rollback* — for each watch event, context = all prior watches, target = the next watch |
| **Output** | unit-norm 128-dim embeddings; cosine-similarity ranking |

---

## 📊 Results

**The modeling win.** Switching from an MSE rating-regression baseline to full-softmax retrieval with popularity correction is dramatic — **~8.7× MRR** and ~8× Hit Rate@10:

| Metric | MSE baseline | **Two-Tower Softmax (α=0.5)** |
|---|---|---|
| Hit Rate@1 | 0.43% | **5.99%** |
| Hit Rate@5 | 1.68% | **15.49%** |
| Hit Rate@10 | 2.70% | **22.06%** |
| Hit Rate@20 | 4.26% | **30.44%** |
| Hit Rate@50 | 7.36% | **44.38%** |
| **MRR** | 0.0133 | **0.1153** |

<sub>Held-out *rollback* eval: for each held-out user, context = prior watches, target = next watch.</sub>

**The ablation — the result I actually care about: cheap, self-built features _match_ expensive curated ones.** MovieLens's genome tags are a premium asset — 1,128 relevance scores per movie, reverse-engineered by GroupLens from **~212,000 human tag-survey ratings plus years of platform-scale community tagging**, run through a dedicated ML pipeline. A new company has none of that. So I asked the practical question: **can you bootstrap comparable content features from nothing but public web text + an LLM, in an afternoon?**

I scraped TMDB + Wikipedia for every movie, had an LLM extract 132 features, and trained four otherwise-identical models (same architecture, hyperparameters, α=0, full corpus) that differ *only* in what fills the content slot — evaluated over **382,138 rollbacks** (all 19,134 val users):

| Content signal | `FEATURE_TOWERS` | How it's produced | MRR |
|---|---|---|---|
| **LLM features (self-built)** | `llm` | scraped web text + an LLM · **~half a day** | **0.1165** |
| Genome tags (MovieLens) | `genome` | 212K human survey ratings + years of community tagging + an ML pipeline | 0.1146 |
| Genome + LLM | `both` | both content sources fused | 0.1154 |
| None (floor) | `none` | no content tower | 0.1143 |

**The takeaway: the self-built LLM features land in a dead heat with the expensive curated genome tags** — 0.1165 vs. 0.1146, a tie within noise, and fusing both adds nothing. The goal was never to *beat* genome tags; it was to show you can reach the same modeling power **without** the human annotators, the active-community folksonomy, or the years of data behind them. For a company that's the headline: **don't wait on a tagging pipeline or hire labelers — bootstrap content features from web text + an LLM and unlock the same lift in an afternoon.**

> *Why the live demo runs the `both` model:* fusing the two sources gave no measurable lift in the ablation, but I wanted the most feature-robust model behind the public demo, so the deployed checkpoint carries both towers. The finding stands on the single-source arms. Full write-up: [`docs/plans/llm_vs_genome_ablation_plan.md`](docs/plans/llm_vs_genome_ablation_plan.md).

---

## 🕹️ The app

Pick a few movies you love and the model builds your taste vector and ranks the whole catalog — live:

<p align="center">
  <img src="assets/demo-recommend.png" alt="Interactive Recommend tab — selecting Toy Story, Finding Nemo, and Monsters Inc. returns Shrek 2, The Incredibles, WALL·E, and other family/animation picks" width="820">
</p>

The [Streamlit app](https://movie-recommender-system-two-tower-model.streamlit.app/) has six tabs:

- **Recommend** — pick your favorite movies (and optionally nudge with genome tags) → ranked poster grid.
- **Examples** — pre-built taste personas (Sci-Fi, Horror, Comedy, Heist, …) to see the model's range instantly.
- **Similar** — nearest neighbors of any movie in the 128-dim embedding space.
- **Genres** / **Genome** — probe what the *item tower* learned about genres and genome tags.
- **About** — the full architecture, training, and popularity-correction write-up.

---

## 🚀 Quickstart

### Run the app locally

The trained, exported model lives in [`serving/`](serving/) and is committed — so the app runs with **no training and no dataset download**:

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

### Reproduce the model from scratch

Download [MovieLens 32M](https://grouplens.org/datasets/movielens/32m/) into `data/ml-32m/`, then run the pipeline. Stages 1–3 cache to disk, so you only re-run what changed.

```bash
python main.py preprocess   # raw CSVs        → base parquets
python main.py features     # base parquets   → engineered features
python main.py dataset      # features        → training tensors
python main.py train        # full-softmax training (canonical)
python main.py eval         # Recall@K, NDCG@K, Hit Rate@K, MRR
python main.py canary       # sanity-check recs for taste personas
python main.py export       # write serving/ artifacts for the app
```

<details>
<summary><strong>CLI reference & the content-source ablation</strong></summary>

```bash
# Most commands accept an optional checkpoint path (defaults to the most recent):
python main.py eval   saved_models/<checkpoint>.pth
python main.py canary saved_models/<checkpoint>.pth
python main.py probe  saved_models/<checkpoint>.pth      # genre / tag / genome embedding probes
python main.py export saved_models/<checkpoint>.pth

# Fetch TMDB poster URLs used by the app (run once; resumable):
TMDB_API_KEY=your_key python main.py posters
```

The four ablation arms are selected with the `FEATURE_TOWERS` env var (identical model, only the content slot differs):

```bash
FEATURE_TOWERS=genome python main.py train   # Model A — curated genome tags
FEATURE_TOWERS=llm    python main.py train   # Model B — self-built LLM features
FEATURE_TOWERS=none   python main.py train   # Model C — no content tower (floor)
FEATURE_TOWERS=both   python main.py train   # Model D — both, fused (deployed)
```

The self-built LLM-feature pipeline (scrape → schema → extract → merge → tensor) lives in [`llm_features/`](llm_features/).
</details>

---

## 🗂️ Project structure

```
.
├── main.py                 # CLI entry point — preprocess │ features │ dataset │ train │ eval │ export …
├── streamlit_app.py        # the live demo app
├── src/
│   ├── preprocess.py       # raw MovieLens CSVs → base parquets
│   ├── features.py         # feature engineering (genre affinity, tags, genome, timestamps)
│   ├── dataset.py          # rollback training-example builder (softmax tuples)
│   ├── model.py            # the two-tower MovieRecommender (4-pool user tower, swappable content towers)
│   ├── train.py            # full-softmax training loop + Menon popularity correction
│   ├── evaluate.py         # canary recommendations + embedding probes
│   ├── offline_eval.py     # Recall@K, NDCG@K, Hit Rate@K, MRR
│   ├── inference.py        # build a user vector from a few liked movies → rank the catalog
│   ├── export.py           # bake checkpoint → device-agnostic serving/ artifacts
│   └── checkpoint.py       # config-from-state_dict loader (no separate config file needed)
├── llm_features/           # self-built content pipeline: scrape → LLM-extract → tensor
├── serving/                # exported artifacts the app loads (model, embeddings, feature store, posters)
├── docs/plans/             # the LLM-vs-genome ablation write-up
└── tests/                  # model shape tests
```

## 🛠️ Tech stack

**PyTorch** (model, training, MPS acceleration) · **Streamlit** (interactive app) · **pandas** / **NumPy** (data pipeline) · **TMDB** + **Wikipedia** (scraped sources for the LLM feature pipeline) · **MovieLens 32M** (training data).

## 📚 Dataset & acknowledgements

- Training data: [MovieLens 32M](https://grouplens.org/datasets/movielens/32m/) by [GroupLens](https://grouplens.org/) (not redistributed here — download separately).
- Popularity correction: Menon et al., [*Long-tail learning via logit adjustment*](https://arxiv.org/abs/2007.07314), ICLR 2021.
- Movie posters via [The Movie Database (TMDB)](https://www.themoviedb.org/).

This is one of a trio of two-tower recommenders I built across domains:
[📚 Books](https://book-recommender-system-two-tower-model.streamlit.app/) ·
[🎮 Games](https://game-recommender-system-two-tower-model.streamlit.app/) ·
🎬 Movies (this repo)

## 📝 License

[MIT](LICENSE) © Nick Greenquist
</content>
