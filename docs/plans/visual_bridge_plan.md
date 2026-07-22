# Visual Bridge: A Cross-Modal Poster→Item-Space Projection Head

> **What this is.** The project plan for extending the trained two-tower recommender with a *visual
> bridge* — a small from-scratch projection head `P` that maps a movie poster's frozen vision
> embedding into the model's learned 128-dim item space. One engine (`P`) unlocks one **measured**
> cold-start result plus two honestly-labeled visual demos. Grounded in the existing repo: it reuses
> the trained towers, the baked `E_item` matrix, and the `src/offline_eval.py` rollback harness rather
> than rebuilding any of them.

**Research question.** A brand-new film with zero ratings has no collaborative-filtering vector — the
item tower can't place it. Can a poster alone predict where that film *would* land in the learned item
space? Concretely: **retrain the base tower with a set of items' interactions removed entirely** (their
`E_item` genuinely does not exist), train `P: siglip(poster) → E_item` on the surviving warm items, then
substitute `P(poster)` for each cold item and measure how much retrieval performance is recovered vs. a
gets-nothing floor, a popularity prior, and the model's own matched-tier warm ceiling.

## Framing — one measured headline + demos that ride the same engine

This is the non-negotiable design principle from the brief, preserved verbatim in scope:

- **Component 1 (cold-start injector) is the ONE rigorous measured result** — a single honest
  Recall@K / NDCG@K number in the same style as the other repos' headlines. Even modest recovery is a
  legitimate result; the number tells the truth either way.
- **Components 2 & 3 are demos that ride `P`.** Component 2 (mood-board) carries a *secondary* measured
  number (vision-only retrieval vs. CF). Component 3 (artwork personalization) is **qualitative and
  explicitly labeled as not measured** — there is no ground-truth poster-click data, so it is
  illustrative, and its only claim to rigor is that it rides `P`, which *is* measured via Component 1.
- **Do not dress up the qualitative demo as a measured result.** This plan keeps that line bright.

**Narrative for the portfolio.** "I build two-tower retrieval; here's the same architecture with a
cross-modal visual bridge." SigLIP is itself a two-tower contrastive model — structurally the same
object as these recommenders, just image↔text instead of user↔item. `P` welds the two two-towers
together at the item vector. This extends the existing body of work rather than opening a disconnected
track.

---

## The engine — build `P` once, everything rides on it

### Regression target: the 128-dim L2-normed `E_item` (recommended)

The item tower's final output is `F.normalize(item_projection(concat), p=2, dim=1)` → a **128-dim unit
vector** (`src/model.py:505`). The full corpus matrix is one call away — `model.full_item_embedding()`
→ `[9375, 128]` (`src/model.py:507`) — and is already baked per-movie as `MOVIE_EMBEDDING_COMBINED`
in `serving/movie_embeddings.pt`. **This is `P`'s regression target.** It is the exact space the app
ranks in (`user·item` cosine, single matmul at `streamlit_app.py:333`), so a poster regressed here can
be scored by the same matmul with zero new inference plumbing.

Two properties to respect:
- It is **unit-normalized** → train `P` with a **cosine loss** (`1 - cos(P(x), E_item)`), and
  L2-normalize `P`'s output. Plain MSE on unit vectors is acceptable but cosine is the honest objective.
- It is **not pure CF** — it already fuses genome + LLM + year content. So `P` regresses onto a
  *content-informed* CF vector, not a pristine collaborative signal. State this plainly (§Caveats).

**Alternative target (ablation, not v1 default): the 32-dim pure-CF lookup.**
`model.item_embedding_lookup.weight[:pad_idx]` → `[9375, 32]` is the *pure* collaborative ID embedding
(learned only from co-occurrence, no content leakage — `src/model.py:236`). Regressing here is the
"truest" pixels→CF bridge, but it lives in a different space than the 128-dim scoring vectors, so it
needs its own projection to rank. Keep as a one-line ablation arm; do not block v1 on it.

### Frozen vision encoder — a fixed dependency, never trained

- **SigLIP-2, ViT-SO400M** (e.g. `google/siglip2-so400m-patch16-384` via `transformers`, or
  `ViT-SO400M-16-SigLIP2-384` via `open_clip`). Image embedding dim ≈ **1152**. Frozen — treat it like
  a database, never fine-tune it. DINOv2 stays an optional swap if fine-grained visual detail is ever
  needed; not v1.
- **Runs offline only.** Like `umap-learn` today (an export-time-only extra, per `requirements.txt`),
  the encoder is **never imported by the deployed app**. All poster embeddings are precomputed offline
  and baked as tensors into `serving/`. Streamlit Cloud never loads SigLIP.

### `P` — a small from-scratch MLP (the part worth owning)

```
PosterProjectionHead(nn.Module):
  Linear(vision_dim≈1152 → hidden≈512) → ReLU → Dropout(p≈0.3)
  Linear(hidden → 128) → F.normalize(p=2, dim=1)
```

From-scratch in PyTorch, consistent with the rest of the repos. Loss `1 - cos`. This is a
stripped-down **DropoutNet-flavored content→CF-space regression**. It is small enough (~KB state_dict)
that the deployed app *can* run it live at request time (torch is already a dep) — which is what powers
Components 2 & 3's live centroid/variant scoring without ever touching SigLIP. Save `serving/vision_proj.pth`
plus a JSON sidecar (vision encoder id, vision_dim, hidden, dropout, train split seed) mirroring the
per-checkpoint sidecar convention.

**Known failure mode to guard against — regression-to-mean collapse.** DropoutNet-style heads love to
collapse toward the global mean item vector. The mandatory sanity baseline is "predict the global mean
`E_item`": `P` must beat it by a clear margin on held-out warm items before Component 1 is even run.

---

## Data pipeline

All new heavy artifacts land under `serving/` (deploy contract: the app reads `serving/` alone, never
`data/` — `.gitignore` line 1). Image *bytes* are a gitignored working cache, never committed.

| Stage | New module | Reads | Writes | Notes |
|---|---|---|---|---|
| 1. Poster URLs | *(exists)* `src/fetch_posters.py` | `links.csv`, TMDB | `serving/posters.json` | Already done — 9,371 `{movieId: w342_url}`. |
| 2. Poster **bytes** | `src/fetch_poster_images.py` (new) | `serving/posters.json` | `data/poster_images/<movieId>.jpg` (gitignored cache) | Copy `fetch_posters.py`'s 0.25s throttle + checkpointed resume + skip-existing loop. **No on-disk image cache exists today — write this fresh.** |
| 3. Poster **variants** | `src/fetch_poster_variants.py` (new) | `serving/tmdb_ids.json`, TMDB | `serving/poster_variants.json` = `{movieId: [url,…]}` + bytes cache | Hits `GET /3/movie/{tmdb_id}/images` → `posters[]` array (today's fetcher only reads the single primary `poster_path`). Component 3 only. |
| 4. SigLIP embeddings | `src/vision_encoder.py` + a `main.py vision-embed` cmd | image bytes cache | `serving/poster_embeddings.pt` `[9371,1152] fp16` (+ id list); `serving/variant_embeddings.pt` | Frozen encoder, batched on MPS/CPU. fp16 to keep size ~21 MB (feature_store is already 85 MB). |
| 5. Train `P` | `src/vision_bridge.py` + `main.py vision-train` | `poster_embeddings.pt`, `E_item` | `serving/vision_proj.pth` + sidecar | Warm-split pairs only (see Component 1). |

**Join integrity (a real risk, flag early).** The corpus is **9,375** items (`fs.top_movies`), but
`posters.json`/`tmdb_ids.json` have **9,371** — a handful lack a valid `tmdbId`, and some poster values
are `""` (film on TMDB, no poster) or `null` (transient error). Every stage must key by `str(movieId)`
via `posters.get(...)`, tolerate the 4-item gap, and log coverage. Items with no poster are simply
absent from `P`'s train set and from any vision-only retrieval — never silently zero-filled.

**Corpus join path (canonical):** `E_item` row `i` ↔ `fs.top_movies[i]` (movieId) ↔
`tmdb_ids.json[str(movieId)]` (tmdbId) ↔ `posters.json[str(movieId)]` (URL) ↔ image bytes. `P`'s
training pairs are `(siglip(poster_i), E_item[i])` for every corpus row `i` with a poster.

---

## Component 1 — Cold-start poster injector (the measured headline)

**The claim we're earning: true zero-interaction cold start.** Retrains are in scope (this heads toward
a blog post), so the headline is the *strict* version — not a frozen-tower proxy. We retrain the base
two-tower with a set of "cold" items' interactions **removed entirely**, so those items' `E_item`
genuinely does not exist; `P` must supply it from the poster alone. This earns the unqualified
"brand-new film" statement rather than "can a poster predict an already-learned vector."

### The cold split as a corpus variant (house-consistent namespacing)

Implement the holdout the same way the repo already namespaces experiments: through the `CORPUS` /
`corpus_suffix()` machinery (CLAUDE.md's `CORPUS=phase1` knob). A new knob — call it
`COLD_FOLD=<seed/fold id>` — selects a cold-item id set and suffixes every derived artifact
(`dataset_softmax_*_coldfoldK.pt`, `movie_interaction_counts_*_coldfoldK.npy`,
`best_*_coldfoldK.pth`) so nothing collides with prod. The masking lives in `src/dataset.py`:
- **Cold items never appear as a training target** — dropped from `label_movieId` / softmax targets.
- **Cold items map to `pad_idx` in every user-history context** — so no gradient ever reaches their
  `item_embedding_lookup` row; the embedding-table geometry stays prod-identical (rows exist, just
  untrained), which keeps `P`-injection at eval a drop-in row swap.
- Val-user split (`get_val_users`) is unchanged, so cold-target rollback examples are held out honestly.

Per CLAUDE.md this is dataset-logic that "needs retraining to validate" → I write the masking + smoke
-test shapes, then **hand you the exact `COLD_FOLD=… python main.py dataset && … train` command**; I
never launch the softmax trainer myself.

### Eval — a thin fork of the existing rollback harness

`src/offline_eval.py` already does almost all of it: builds `all_embs` (`_build_emb_matrix :46`), scores
every target against the full corpus with one matmul (`:187`), computes **Recall@K (≡ HitRate@K for
single-relevant-item retrieval), NDCG@K, MRR** (`_metrics_from_ranks :110`), and **already stratifies by
popularity** — `_build_tiers` (`:82`) gives HEAD/TAIL at 1,000 raw ratings plus quartiles Q1(rarest)…Q4.
The "lift on the low-interaction tail" the brief asks for is a free byproduct.

`src/vision_eval.py` forks `_run_rollback_eval`: load the **warm-only** checkpoint, swap the cold rows of
`all_embs` for `P(poster)`, restrict the example mask to **cold-target** rollbacks, and report the arms
below. Writes a **distinct stem** (`eval_results/vision_coldstart_coldfoldK.txt`) — the same
CLAUDE.md discipline as the `EVAL_N_USERS` clobbering warning; it must never overwrite the prod stem.

**Arms (the honest comparison), all scored under the *same* warm-only model:**
1. **Floor — cold item gets nothing.** Cold rows = zero / untrained-init vector → near-0. Sanity floor.
2. **Popularity / mean-vector prior.** Cold rows = the global mean `E_item`. The bar `P` must beat
   (also the collapse check — if `P` ≈ this, it collapsed to the mean).
3. **`P(poster)` injection — the headline arm.**
4. **Matched-tier warm ceiling.** The warm-only model's *native* retrieval quality on WARM targets at
   the same popularity tier — an apples-to-apples, within-model ceiling (no cross-model comparison).
5. **Secondary "if it had ratings" reference (optional):** the full/prod model's retrieval of those same
   cold items. Cross-recipe, so labeled as a soft reference, not the ceiling.

**Headline metric:** *fraction of the matched-tier warm ceiling recovered by the poster*, whole-corpus
and on the TAIL tier. One honest number, reported whatever it is.

### The ablation matrix (this is the blog)

Multiple runs are in scope, so Component 1 is run as a proper ablation rather than a single number:

| Axis | Arms | Cost | Why it's interesting |
|---|---|---|---|
| **Cold split** | single 80/20 (MVP) → **5-fold** (blog) | 1 → 5 base retrains | 5-fold gives every item a cold turn → a corpus-wide cold-start number, not one lucky split. |
| **Base-model seed** | 42 (+1,+2 if compute allows) | ×N retrains | Error bars vs. the ±0.003–0.004 single-seed noise floor the existing ablation docs cite. |
| **`P` target space** | 128-dim combined `E_item` **(default)** vs. 32-dim pure-CF lookup | cheap (P is tiny) | Does the poster predict the *scoring* vector or the *pure collaborative* vector better? |
| **`P` seed** | 42/1/2 | trivial (minutes each) | `P` variance is separable from base-model variance. |
| **Method** *(stretch arm)* | standalone regression head `P` **(default)** vs. **CF-dropout fusion** | +1 retrain | The DropoutNet-proper alternative: feed the vision buffer into the item tower and randomly zero the CF row during training so the tower *learns* to fall back to vision. A regression-head-vs-fusion comparison is itself a blog-worthy result. (Shares the §Stretch item-sub-tower plumbing.) |

Recommended sequencing: land the single 80/20 split end-to-end first (proves the pipeline + one number),
then expand to 5-fold + seeds for the publishable version. `P`-target and method axes are add-on arms,
not gates.

---

## Component 2 — Taste mood-board / "your aesthetic" (cool + a secondary number)

**Demo.** User multi-selects films they like → average their **poster SigLIP embeddings** into a visual
centroid → retrieve nearest films → render a mood-board of "what your taste looks like." (This absorbs
the standalone multi-select artwork-recommender UX into the same engine.)

**Measured core (second honest number).** Hold out some of a user's liked movies; from the *remaining*
liked movies' poster centroid, can we retrieve the held-out liked ones? Report **vision-only Recall@K
vs. the CF model** on the identical held-out-liked task, reusing the val-user split. Three retrieval
spaces to compare, which yields a clean "vision alone is X% of CF" statement:
1. **Pure vision:** raw-SigLIP centroid vs. raw-SigLIP poster embeddings (no `P`).
2. **Bridged:** `P(centroid)` vs. `E_item` (averaging happens in SigLIP space *then* projects — order
   matters because `P` is nonlinear, so the app must bake raw SigLIP embeddings, not pre-projected).
3. **CF reference:** the real user tower (`build_user_embedding`, `src/inference.py:27`).

Reuses the canary persona pattern verbatim — seed titles → embedding → matmul over item matrix → top-K
excluding seeds (`src/evaluate.py:328`).

**Streamlit:** new `tab_moodboard(art, posters, tmdb_ids)`; multiselect over `fs['title_to_movieId']`,
live centroid in-browser (raw SigLIP embeddings baked in `serving/`), render via the existing
`_show_results` / `_poster_div` / `.poster-grid` path (`streamlit_app.py:407, 367, 1844`).

---

## Component 3 — Artwork personalization, Netflix-style (qualitative demo, NOT measured)

**Demo.** For a given user, rank a film's poster **variants** by predicted appeal — score each variant
`P(siglip(variant))` against the current `E_user` — and show "the poster this system would show *you*."

**Honestly labeled.** No ground-truth poster-click data exists, so this is **illustrative, not a
measured result.** Its rigor is inherited: it rides `P`, which is measured in Component 1. The UI copy
and the About-tab writeup must say this in plain words — no implied metric.

**Depends on** stage-3 variant fetch (`/movie/{id}/images`) + `serving/variant_embeddings.pt`. New
`tab_artwork`; for a picked film, render its variants sorted by `P(variant)·E_user`, side-by-side with
the "generic" (most-popular) variant.

**Live-demo caveat.** A true "upload a poster of a brand-new movie and inject it" demo needs SigLIP at
request time, which can't run on Streamlit Cloud. Options: (a) bake embeddings for a small curated set
of genuinely-new/held-out films and present that as the cold-start showcase (honest: a canned set); (b)
keep live upload as a local-only demo. The *measured proof* of cold-start value is Component 1's eval,
not the live widget — so this is a presentation choice, not a rigor gap.

---

## Optional stretch — pooled-variant item sub-tower → tail lift (in-repo, not a blocker)

Pool a film's variants into a richer visual item vector and **fuse it into the item tower** as a new
sub-tower, then measure NDCG@K lift stratified by popularity decile. The architecture already has the
exact template: the LLM feature tower ingests a per-item external buffer
(`model.llm_feature_buffer[emb_idx]` → `item_llm_feature_tower`, `src/model.py:227,497`). A vision
sub-tower is the same shape — register `vision_feature_buffer`, add `item_vision_tower`, widen
`item_concat_dim`, extend `feature_towers`, retrain, re-export (the buffer bakes into
`feature_store.pt` just like `llm_feature_buffer`).

**Gated and caveated.** This needs a full softmax retrain (user runs it) + a re-export, and per CLAUDE.md
I stop at code + smoke test. **Caveat to encode (from the brief):** poster variants are correlated
marketing art (shared palette, title treatment) → less independent signal than true scene stills. A
flat result is *ambiguous* ("vision doesn't help the tail" vs. "variants too redundant"), not proof
vision is useless. If promising-but-limited, real stills become a sequel. Do not over-invest in v1.

---

## File / module structure (new)

```
src/
  fetch_poster_images.py   # download poster bytes → gitignored cache (throttle/resume from fetch_posters.py)
  fetch_poster_variants.py # /movie/{id}/images → serving/poster_variants.json (Component 3)
  vision_encoder.py        # frozen SigLIP-2 loader + batched image→embedding (OFFLINE-ONLY import)
  vision_bridge.py         # PosterProjectionHead (from scratch) + train loop + save P + sidecar
  vision_eval.py           # Component 1 cold-start eval + Component 2 vision-only eval (forks offline_eval)
main.py                    # + subcommands: poster-images, poster-variants, vision-embed, vision-train, vision-eval
streamlit_app.py           # + tab_moodboard (C2), tab_artwork (C3); reuse _show_results/_poster_div
serving/                   # + poster_embeddings.pt, variant_embeddings.pt, poster_variants.json, vision_proj.pth
tests/
  test_vision_bridge.py    # P shape/normalization; join-coverage assertions; mean-baseline sanity (CPU-only)
docs/plans/visual_bridge_plan.md   # this file
docs/visual_bridge/        # narrative writeup + linkedin_post.txt (per house convention) once results land
```

## Dependencies

- **Offline / export-time only** (never installed on the serving host — document alongside `umap-learn`
  in `requirements.txt` comments): `open_clip_torch` *or* `transformers`+`timm` for SigLIP-2 SO400M,
  `pillow`, `requests` (already used by the poster fetcher).
- **Deployed app gains nothing heavy.** It loads baked tensors + the tiny `vision_proj.pth`; `torch` is
  already a dependency. No SigLIP, no image decoding, at request time.
- **Storage budget:** `poster_embeddings.pt` fp16 ≈ 21 MB; variants multiply that — watch Streamlit
  Cloud's repo/artifact limits (feature_store is already 85 MB). Mitigations: fp16, and store raw SigLIP
  only where averaging-before-projection requires it.

## Milestones (base retrains are user-run; timeline is my-hands time, retrain wall-clock is separate)

- **M0 — Data (wk 1).** `fetch_poster_images`, `vision-embed`, bake `poster_embeddings.pt`. Gate:
  coverage report (9,371/9,375), join integrity asserted in a test.
- **M1 — Cold split + warm-only retrain (wk 1–2).** `COLD_FOLD` masking in `src/dataset.py`; smoke-test
  shapes; **hand you the `COLD_FOLD=0` dataset+train command.** You run train → canary → eval; canary is
  the go/no-go that the warm-only model didn't regress on warm personas. *(Repeat per fold/seed for the
  ablation — cheap on my side, real wall-clock on yours.)*
- **M2 — The engine (wk 2).** `PosterProjectionHead` + train `P` on the warm-only model's warm-item
  pairs (`P` training is minutes, I can run it in-session). Gate: `P` beats the global-mean baseline on
  held-out warm items by a clear cosine margin (collapse check).
- **M3 — Component 1, the headline (wk 2–3).** `vision_coldstart` eval: floor / popularity / `P` /
  matched-tier-warm-ceiling arms, tail stratified, distinct eval stem. **Gate: one honest cold-start
  number.** Then widen to 5-fold + seeds for the publishable table.
- **M4 — Component 2 (wk 3).** Vision-only mood-board eval (second number) + `tab_moodboard`.
- **M5 — Component 3 (wk 3–4).** Variant fetch + `tab_artwork`, honestly labeled.
- **M6 — Writeup (wk 4).** `docs/visual_bridge/` narrative + LinkedIn counterpart (house convention).
- **M7 — Stretch (optional).** CF-dropout fusion arm / pooled-variant item sub-tower; hand retrain to
  you; tail-lift eval. Shares the item-sub-tower plumbing with the Component-1 "method" ablation axis.

## Risks & caveats (encode these)

1. **Cold-mask correctness** — the headline is only honest if cold items are *fully* absent from
   training (not a target, mapped to `pad_idx` in every context). A leak (cold item surviving in one
   context pool) silently inflates the number. Assert zero cold gradient in a test before any retrain.
2. **`P` regression-to-mean collapse** — mandatory global-mean baseline gate (arm 2) before M3.
3. **Cross-model comparison hygiene** — the ceiling is the *within-model* matched-tier warm arm, not the
   prod model (different recipe/corpus). The prod-model number is a soft reference only, labeled as such.
4. **Join / coverage gap** (9,371 vs 9,375; `""`/`null` posters) — key by movieId, log coverage, never
   zero-fill.
5. **Retrain cost / variance** — 5-fold × seeds is real wall-clock on your machine; sequence single-split
   first so a headline lands before the full grid. Report seed variance against the ±0.003–0.004 noise floor.
6. **Deploy artifact size** — fp16, cap variant counts.
7. **Variant redundancy** (stretch) — correlated marketing art; flat result is ambiguous, not disproof.
8. **Eval-artifact clobbering** — distinct eval stem; never a small-`n` run over the canonical stem.
9. **Content leakage in the target** — `E_item` already fuses genome+LLM; `P` regresses a
   content-informed vector, not pure CF. Stated, not hidden. (32-dim pure-CF target is the ablation axis.)
10. **Don't touch prod** — `streamlit_app.py` scoring path, `src/export.py`, and the prod checkpoint stay
    untouched until Component 1's number is in hand (house rule: verify good before export/app changes).

## Consistency with existing design

- **Item-side only.** The visual bridge adds nothing to the user side — the "no user-ID embedding" taste
  design (CLAUDE.md core choice) is untouched. `P` places *items*; users stay embeddable-from-a-few-likes.
- **Same scoring primitive.** Cold-start injection and both demos rank with the same single
  `user·item` cosine matmul the app already uses — no HNSW, no new inference path (consistent with the
  "full scoring is a single matmul" note in CLAUDE.md's Do-NOT-revisit list).
- **Offline-heavy / serving-light**, exactly like the LLM feature pipeline and UMAP: compute offline,
  bake tensors, deploy thin.
```
