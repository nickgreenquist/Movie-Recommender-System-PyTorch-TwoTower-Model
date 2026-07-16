"""
Movie Recommender — Streamlit app.

Run locally:  streamlit run streamlit_app.py
Requires:     serving/model.pth
              serving/movie_embeddings.pt
              serving/feature_store.pt
              serving/posters.json   (optional — fetch with: TMDB_API_KEY=... python main.py posters)

Generate serving/ with: python main.py export
"""
import datetime
import html
import json
import os
import threading
import urllib.parse
from typing import NamedTuple

import numpy as np
import pandas as pd
import plotly.colors as pcolors
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
import torch
import torch.nn.functional as F

from src.evaluate import (
    USER_TYPE_TO_FAVORITE_MOVIES,
    USER_TYPE_TO_DISLIKED_MOVIES,
    USER_TYPE_TO_GENOME_TAGS,
)
from src.inference import build_user_embedding
from src.llm_frontend import build_frontend_context, build_serving_model, recommend
from src.llm_frontend_extraction import extract_query
from src.model import MovieRecommender

EXAMPLE_PROFILES = list(USER_TYPE_TO_FAVORITE_MOVIES.keys())

# Rating values — mirror evaluate.py canary constants (genre-override weights live in
# src/inference.py: LIKED_GENRE_VALUE / DISLIKED_GENRE_VALUE)
_LIKED_MOVIE    =  2.0
_DISLIKED_MOVIE = -2.0
# Genome anchor movies are synthetic/implicit — we use more of them (5 per tag) to push
# the model toward the selected tags, but each one carries half the weight of an explicitly
# chosen favorite movie. A user saying "this is my favorite" is a stronger signal than
# an anchor we added under the hood.
_ANCHOR_MOVIE   =  1.0
_ANCHORS_PER_TAG = 5

_PAGE_SIZE     = 20     # movies per page
_TOTAL_RESULTS = 60     # total movies to fetch (3 pages)

# Conversational ("Ask") tab — natural language → LLM intent extraction → the shared recommend()
# pipeline. The LLM (Claude Haiku) only parses intent; the trained model still does all retrieval.
# Two light call caps keep the demo's hosted-LLM bill negligible (plan §Protections): per browser
# session (resets on refresh) and per calendar day across ALL visitors (shared st.cache_resource
# counter, resets on app restart/redeploy). Both are polite in-app friction — the Anthropic
# Console spend limit on the key's workspace is the actual bill protection.
_LLM_SESSION_CAP = 20   # max hosted extraction calls per browser session
_LLM_DAILY_CAP   = 60   # max hosted extraction calls per day, all sessions combined

# Recommend / Examples popularity-correction A/B — the two separately-trained models the visitor
# switches between. Prod bakes the Menon α=0.5 logit adjustment into training; the α=0 twin is a
# distinct model trained with NO correction, surfaced so the visitor can watch popular titles take
# over once the correction is removed. The correction is train-time only (inference is plain dot
# products either way), so "Off" loads a different model — it is not a runtime knob. These labels
# double as the segmented-control options; the first is the default (prod).
_REC_MODEL_DEFAULT_LABEL  = 'On (α = 0.5)'
_REC_MODEL_NO_ALPHA_LABEL = 'Off (α = 0)'

# ── Map tab ────────────────────────────────────────────────────────────────────
# Coloring rule: a movie has several genres, so the map colors each point by its FIRST-listed
# (MovieLens-canonical "primary") genre. Deterministic and one-trace-per-genre, which is what
# makes the legend click-to-isolate. _MAP_NO_GENRE is muted grey — it isn't a real genre.
_MAP_NO_GENRE      = '(no genres listed)'
_MAP_NO_GENRE_HEX  = '#6b7280'
_MAP_BASE_SIZE     = 3       # catalog point size in the 3D cloud (px)
_MAP_BASE_OPACITY  = 0.80    # catalog point opacity with no movie picked
_MAP_DIM_OPACITY   = 0.28    # dimmed catalog opacity once a pick is on, so the overlay reads on top
_MAP_NEIGHBORS     = 25      # genome-space nearest neighbors highlighted for the picked movie
_MAP_HEIGHT        = 680     # plot height (px)
# Genre highlight (genre pills): every genre stays on the map, but a selected genre's points grow
# and the unselected genres fade to faint context so the selection pops.
_MAP_HL_SIZE       = 6       # a highlighted genre's point size
_MAP_HL_DIM_OPACITY = 0.12   # unselected genres' opacity while any genre is highlighted
# Genome-tag highlight (genome pills, below the genre pills): same pop-and-dim behavior, but the
# selection cross-cuts genre — a tag lights up the movies that very clearly carry it (genome
# relevance ≥ _MAP_GENOME_MIN_RELEVANCE), wherever they sit in the cloud, instead of a whole
# primary-genre bucket.
# A genome-tag pill highlights only movies that VERY clearly carry the tag — relevance ≥ this floor,
# never a fixed top-N. The genre pills cluster tightly because they key on a movie's single PRIMARY
# genre; a strict relevance floor is the genome analogue (a loose top-N lit up too many weakly-tagged
# movies and smeared the highlight across the whole cloud).
_MAP_GENOME_MIN_RELEVANCE = 0.8
# Hand-picked from the MIDDLE of the genome-frequency distribution — deliberately NOT the "head" tags
# that thousands of movies score high on (those fill the entire cloud and never cluster), but
# specific themes only a few dozen movies definitively carry, so each lights up a tight, legible
# region. Five are intentionally from the TAIL (rare auteur/niche labels the model groups by content
# alone, with no director or style input: kurosawa, studio ghibli, tarantino, grindhouse, giant
# robots); 007 is here by request (the Bond films). Validated for above-floor cluster tightness in
# the baked 3D space. Ordered by family (auteurs → style/era → sci-fi → horror →
# action/international → crime/war) so the pill row scans cleanly.
_MAP_GENOME_TAGS = [
    'kurosawa', 'studio ghibli', 'tarantino', 'silent', 'dreamlike', 'psychedelic',
    'cyberpunk', 'space opera', 'giant robots', 'dinosaurs', 'slasher', 'grindhouse',
    'haunted house', 'zombies', 'samurai', 'wuxia', 'bollywood', 'spaghetti western',
    'harry potter', 'marvel', '007', 'film noir', 'courtroom drama', 'holocaust',
    'based on a video game',
]
# Overlay markers — WAY bigger than the catalog points so the pick and its neighbors pop.
_MAP_PICK_SIZE     = 15      # the single picked movie (between the neighbor size and the old 20)
_MAP_NEIGHBOR_SIZE = 11      # its top genome-space neighbors
_MAP_PICK_HEX      = '#ffffff'   # white — your pick
_MAP_PICK_EDGE     = '#ffd23f'   # gold ring around your pick
_MAP_NEIGHBOR_HEX  = '#19e0ff'   # cyan — its genome-space neighbors


# ── Startup ───────────────────────────────────────────────────────────────────

_REQUIRED_ARTIFACTS = (
    'serving/feature_store.pt',
    'serving/movie_embeddings.pt',
    'serving/model.pth',
)

_TMDB_IDS_FILE = 'serving/tmdb_ids.json'


class Artifacts(NamedTuple):
    """Everything the tabs need, loaded once and cached. Named fields so the load order
    can't drift out of sync with the call sites."""
    model:           MovieRecommender   # prod (α=0.5) — also powers Genres / Genome / Similar tabs
    fs:              dict
    me:              dict
    all_ids:         list
    all_embs:        torch.Tensor
    rec_models:      dict          # Recommend/Examples A/B: {label: (model, item-emb matrix)}, prod first
    all_norm_genre:  torch.Tensor
    all_norm_genome: torch.Tensor
    sim_spaces:      dict          # Similar tab's 2x2: {source: {representation: norm_matrix}}
    ts_inference:    torch.Tensor
    frontend_ctx:    object        # src.llm_frontend.FrontendContext — powers the conversational Ask tab
    posters:         dict
    tmdb_ids:        dict
    map_coords:      object        # (N, 3) float32 baked 3D projection, all_ids order — None on old bundles
    map_reducer:     object        # name of the offline reducer (provenance caption) — None on old bundles


def _load_tmdb_ids(fs):
    """movieId → tmdbId map (string keys, like posters.json) for the TMDB card links.

    Reads serving/tmdb_ids.json when present — the committed artifact the deployed app
    uses. Locally, if the json is missing but data/ml-32m/links.csv exists, builds the
    corpus subset once and saves it (commit the json to ship links to Streamlit Cloud).
    Returns {} when neither exists — cards then fall back to a TMDB title search.
    """
    if os.path.exists(_TMDB_IDS_FILE):
        with open(_TMDB_IDS_FILE) as f:
            return json.load(f)
    links_path = 'data/ml-32m/links.csv'
    if os.path.exists(links_path):
        links = pd.read_csv(links_path, dtype={'tmdbId': 'Int64'}).dropna(subset=['tmdbId'])
        links = links[links['movieId'].isin(set(fs['top_movies']))]
        tmdb_ids = {str(int(row.movieId)): int(row.tmdbId) for row in links.itertuples()}
        with open(_TMDB_IDS_FILE, 'w') as f:
            json.dump(tmdb_ids, f)
        return tmdb_ids
    return {}


# _build_serving_model moved to src/llm_frontend.py (build_serving_model) so the deployed app and
# the conversational front-end reconstruct the serving model from identical code — imported above.


@st.cache_resource
def load_artifacts() -> Artifacts:
    missing = [p for p in _REQUIRED_ARTIFACTS if not os.path.exists(p)]
    if missing:
        st.error(
            "Serving artifacts are missing:\n\n"
            + "\n".join(f"- `{p}`" for p in missing)
            + "\n\nGenerate them with `python main.py export`."
        )
        st.stop()

    fs  = torch.load('serving/feature_store.pt', weights_only=False)
    me  = torch.load('serving/movie_embeddings.pt', weights_only=False)
    cfg = fs['model_config']

    model = build_serving_model(fs, cfg)
    state_dict = torch.load('serving/model.pth', weights_only=True)
    # The semantic-feature/context buffers are non-persistent (rebuilt in build_serving_model);
    # drop any an older serving/model.pth still carries so strict load works on old and new artifacts.
    for buf in ('genome_context_buffer', 'content_context_buffer', 'llm_feature_buffer'):
        state_dict.pop(buf, None)
    model.load_state_dict(state_dict)
    model.eval()

    all_ids  = list(me.keys())
    all_embs = torch.cat([me[m]['MOVIE_EMBEDDING_COMBINED'] for m in all_ids], dim=0)

    # ── Popularity-correction A/B: prod (α=0.5) + optional α=0 twin ──────────────────────────────
    # Both models share this feature_store/config — identical architecture, vocabs and feature
    # buffers; only the trained weights differ (α is train-time only). The twin's artifacts
    # (model_no_alpha.pth + movie_embeddings_no_alpha.pt) are written by
    # `python main.py export <ckpt> --variant no_alpha`; absent them, only prod loads and the
    # Recommend/Examples tabs render no toggle. rec_models maps the segmented-control label →
    # (model, item-embedding matrix); dict insertion order makes prod the default.
    rec_models = {_REC_MODEL_DEFAULT_LABEL: (model, all_embs)}
    na_model_path = 'serving/model_no_alpha.pth'
    na_emb_path   = 'serving/movie_embeddings_no_alpha.pt'
    if os.path.exists(na_model_path) and os.path.exists(na_emb_path):
        na_model = build_serving_model(fs, cfg)
        na_sd    = torch.load(na_model_path, weights_only=True)
        for buf in ('genome_context_buffer', 'content_context_buffer', 'llm_feature_buffer'):
            na_sd.pop(buf, None)
        na_model.load_state_dict(na_sd)
        na_model.eval()
        na_me       = torch.load(na_emb_path, weights_only=False)
        na_all_embs = torch.cat([na_me[m]['MOVIE_EMBEDDING_COMBINED'] for m in all_ids], dim=0)
        rec_models[_REC_MODEL_NO_ALPHA_LABEL] = (na_model, na_all_embs)

    # Per-sub-tower embedding matrices, normalized once at load so the Genres / Genome /
    # Similar tabs rank the whole corpus with a single matmul instead of a per-movie cosine loop.
    all_genre_embs  = torch.cat([me[m]['MOVIE_GENRE_EMBEDDING'].view(1, -1) for m in all_ids], dim=0)
    all_norm_genre  = F.normalize(all_genre_embs, dim=1)
    all_genome_embs = torch.cat([me[m]['MOVIE_GENOME_TAG_EMBEDDING'].view(1, -1) for m in all_ids], dim=0)
    all_norm_genome = F.normalize(all_genome_embs, dim=1)

    # ── Similar-tab content spaces: a 2×2 of {feature source} × {representation} ──────────────
    # source  = Genome (curated 1,128-dim relevance) | LLM (132-dim extracted features)
    # rep     = Learned embedding (32-dim item-tower projection, trained end-to-end) |
    #           Raw features (direct cosine over the un-projected vectors, a content baseline)
    # Each cell is an L2-normalized (n_movies, d) matrix; the tab ranks with one matmul and reads
    # the seed row by index. The "raw" matrices need no model — they come straight from the
    # feature store, so they exist even on artifacts predating the LLM embedding export.
    all_norm_genome_raw = F.normalize(torch.stack(
        [torch.tensor(fs['movieId_to_genome_tag_context'][m], dtype=torch.float32) for m in all_ids]), dim=1)
    sim_spaces = {'Genome Tags': {'Learned embedding': all_norm_genome, 'Raw features': all_norm_genome_raw}}

    # LLM source — only when the learned LLM embedding is in the artifacts (re-exported from a model
    # with the LLM tower). Its raw matrix is the baked feature buffer, indexed in corpus order.
    if 'MOVIE_LLM_FEATURE_EMBEDDING' in me[all_ids[0]]:
        all_norm_llm = F.normalize(
            torch.cat([me[m]['MOVIE_LLM_FEATURE_EMBEDDING'].view(1, -1) for m in all_ids], dim=0), dim=1)
        emb_i = fs['item_emb_movieId_to_i']
        all_norm_llm_raw = F.normalize(torch.stack(
            [fs['llm_feature_buffer'][emb_i[m]].float() for m in all_ids]), dim=1)
        sim_spaces['LLM Features'] = {'Learned embedding': all_norm_llm, 'Raw features': all_norm_llm_raw}

    # Most-recent timestamp bin — used for all inference (canary approach)
    ts_inference = torch.bucketize(
        torch.tensor([float(fs['timestamp_bins'][-1].item())]),
        fs['timestamp_bins'], right=False,
    )

    # Conversational ("Ask") tab: the natural-language pipeline reuses this exact loaded model +
    # corpus. Build the shared FrontendContext once here (cheap lookups over fs) so the tab never
    # reloads serving/ — it ranks against the same prod model the manual Recommend tab uses.
    # fs['facets'] (people facet store, baked by export) powers people filters ("Tom Hanks movies");
    # absent on an old serving artifact → recommend() reports people unresolved and drops them.
    frontend_ctx = build_frontend_context(
        model, fs, all_ids, all_embs, ts_inference, facets=fs.get('facets'))

    poster_path = 'serving/posters.json'
    if os.path.exists(poster_path):
        with open(poster_path) as f:
            posters = json.load(f)
    else:
        posters = {}

    tmdb_ids = _load_tmdb_ids(fs)

    # ── Map tab coordinates ──────────────────────────────────────────────────────────────────
    # The 3D projection is baked at export (src/export.py:_project_embeddings_3d). Realign it
    # to all_ids order via the row→movieId map that ships beside it, so the coords stay matched to
    # all_embs even if the two artifacts were ever written in different orders. Absent on bundles
    # exported before this feature — map_coords stays None and the Map tab shows a re-export notice.
    map_coords  = None
    map_reducer = None
    coords_3d   = fs.get('item_coords_3d')
    if coords_3d is not None:
        coords_3d   = np.asarray(coords_3d, dtype=np.float32)
        coord_mids  = fs.get('item_coords_movie_ids')
        if coord_mids is not None and list(coord_mids) != all_ids:
            row_of     = {m: i for i, m in enumerate(coord_mids)}
            map_coords = np.stack([coords_3d[row_of[m]] for m in all_ids]).astype(np.float32)
        else:
            map_coords = coords_3d
        map_reducer = fs.get('item_coords_reducer', '3D projection')

    return Artifacts(
        model=model, fs=fs, me=me, all_ids=all_ids, all_embs=all_embs, rec_models=rec_models,
        all_norm_genre=all_norm_genre, all_norm_genome=all_norm_genome, sim_spaces=sim_spaces,
        ts_inference=ts_inference, frontend_ctx=frontend_ctx, posters=posters, tmdb_ids=tmdb_ids,
        map_coords=map_coords, map_reducer=map_reducer,
    )



# ── Inference helpers ─────────────────────────────────────────────────────────

def _build_user_embedding(model, fs, liked_titles_with_weights, disliked_titles,
                           liked_genres, disliked_genres, ts_inference):
    """
    Build a user embedding via model.user_embedding() — identical path to canary.
    Delegates to the shared inference path (src/inference.py), which both this app and
    the canary eval use so the user tower is fed identically.

    liked_titles_with_weights: list of (title, weight) tuples.
      Explicit liked movies use _LIKED_MOVIE; genome anchors use _ANCHOR_MOVIE.
    disliked_titles: flat list of title strings (always weighted _DISLIKED_MOVIE).
    """
    return build_user_embedding(
        model, fs, liked_titles_with_weights, disliked_titles, ts_inference,
        liked_genres=liked_genres, disliked_genres=disliked_genres,
        disliked_movie_value=_DISLIKED_MOVIE,
    )


def _build_genome_i_to_name(fs):
    return {fs['genome_tag_to_i'][tid]: fs['genome_tag_names'][tid] for tid in fs['genome_tag_to_i']}


def _top_genome_tags(mid, fs, i_to_name, n=5):
    ctx     = fs['movieId_to_genome_tag_context'][mid]
    top_idx = sorted(range(len(ctx)), key=lambda i: -ctx[i])[:n]
    return ', '.join(i_to_name[i] for i in top_idx)


def _score_movies(user_emb, all_ids, all_embs, fs, exclude_titles, top_n=_TOTAL_RESULTS):
    """Dot-product score all movies, filter seeds, return top-n as a DataFrame."""
    raw_scores = (all_embs @ user_emb.T).squeeze(-1)
    exclude    = set(exclude_titles)
    i_to_name  = _build_genome_i_to_name(fs)
    rows = []
    for i in raw_scores.argsort(descending=True).tolist():
        mid   = all_ids[i]
        title = fs['movieId_to_title'][mid]
        if title in exclude:
            continue
        rows.append({
            'Title':           title,
            'Genres':          ', '.join(fs['movieId_to_genres'][mid]),
            'Top Genome Tags': _top_genome_tags(mid, fs, i_to_name),
        })
        if len(rows) >= top_n:
            break
    return pd.DataFrame(rows)


# ── TMDB links ────────────────────────────────────────────────────────────────

def _tmdb_url(mid, tmdb_ids, title: str) -> str:
    """TMDB movie page when the id is known; otherwise a TMDB title search — a handful
    of corpus movies have no (or an invalid) tmdbId in links.csv. Search strips the
    trailing ' (year)' from the corpus title so TMDB matches on the bare name."""
    tmdb_id = tmdb_ids.get(str(mid)) if mid else None
    if tmdb_id:
        return f"https://www.themoviedb.org/movie/{tmdb_id}"
    query = urllib.parse.quote_plus(title.rsplit(' (', 1)[0])
    return f"https://www.themoviedb.org/search?query={query}"


def _poster_div(poster_url: str, link_url: str) -> str:
    """Poster rendered as a background-image div wrapped in an <a> (opens the TMDB page
    in a new tab), NOT an st.image: Streamlit images can't be made into links, and a
    background div degrades to the dark 🎬 placeholder tile when the poster is missing
    instead of a broken-image icon. TMDB w342 posters are 2:3 → fix that aspect ratio so
    the tile fills the card width. The .cover-link hover cue lives in the global style
    block at app init."""
    image       = f"background-image:url(\"{poster_url}\");" if poster_url else ""
    placeholder = "" if poster_url else "🎬"
    div = (f"<div style='width:100%;aspect-ratio:2/3;border-radius:6px;"
           f"background-color:#1e1e1e;background-position:center;background-size:cover;"
           f"display:flex;align-items:center;justify-content:center;font-size:2rem;{image}'>"
           f"{placeholder}</div>")
    return (f"<a class='cover-link' href='{link_url}' target='_blank' rel='noopener' "
            f"style='display:block'>{div}</a>")


# ── Results feed (show-more) ──────────────────────────────────────────────────

def _store_results(df, result_key: str) -> None:
    """Save a results DataFrame to session state and reset the feed to the first page."""
    st.session_state[f'{result_key}_df']    = df
    st.session_state[f'{result_key}_shown'] = _PAGE_SIZE


def _show_more_button(state_key: str, shown: int, total: int) -> None:
    """Centered 'Show more' button that reveals the next _PAGE_SIZE results BELOW the ones
    already on screen (cumulative append, not page replacement). Reads/writes
    st.session_state[f'{state_key}_shown'] and reruns. Appending — rather than swapping
    pages — means the browser keeps its scroll position and the new movies slot in just
    under where the button was, so no scroll-to-top juggling is needed. No-op once
    everything is shown."""
    if shown >= total:
        return
    _, mid, _ = st.columns([2, 1, 2])
    if mid.button('Show more', use_container_width=True, key=f'{state_key}_more'):
        st.session_state[f'{state_key}_shown'] = shown + _PAGE_SIZE
        st.rerun()


def _show_results(result_key: str, posters, fs, tmdb_ids) -> None:
    """Render stored results as a growing feed: the first _PAGE_SIZE movies plus a
    'Show more' button that appends _PAGE_SIZE more each click. Posters link to the
    movie's TMDB page; titles stay plain captions."""
    df = st.session_state.get(f'{result_key}_df')
    if df is None or df.empty:
        return

    shown   = st.session_state.get(f'{result_key}_shown', _PAGE_SIZE)
    page_df = df.iloc[:shown]

    if not posters:
        st.dataframe(page_df, use_container_width=True, hide_index=True)
    else:
        titles = page_df['Title'].tolist()
        # One continuous CSS grid (.poster-grid in the global style block: 5-across on desktop,
        # 3-across on phones) — NOT st.columns(5) per set of 5, which the phone layout would
        # wrap into a lopsided 3+2 within every set. The keyed container (class
        # st-key-<key>_grid) survives as the results-scroll target.
        cells = []
        for title in titles:
            clean_title = title.replace('  ◀ ANCHOR', '').replace('  ◀ anchor', '')
            mid  = fs['title_to_movieId'].get(clean_title)
            url  = (posters.get(str(mid)) or '') if mid else ''
            link = _tmdb_url(mid, tmdb_ids, clean_title)
            cells.append("<div>" + _poster_div(url, link) +
                         f"<div class='poster-caption'>{html.escape(title)}</div></div>")
        with st.container(key=f'{result_key}_grid'):
            st.html("<div class='poster-grid'>" + "".join(cells) + "</div>")

    _show_more_button(result_key, shown, len(df))


# ── Popularity-correction selector (Recommend + Examples) ─────────────────────

def _active_rec_model(rec_models, key):
    """Resolve the currently-selected popularity-correction model from session state (default =
    prod), WITHOUT rendering anything, so the recommend logic has a (model, item-embeddings) pair
    even on runs where the A/B control isn't drawn (e.g. before any results exist). The control
    itself is rendered separately by _render_rec_model_toggle and gated to runs that already have
    results on screen. Returns (variant_label, model, all_embs)."""
    labels  = list(rec_models)
    variant = st.session_state.get(key, labels[0])
    if variant not in rec_models:
        variant = labels[0]
    model, all_embs = rec_models[variant]
    return variant, model, all_embs


def _render_rec_model_toggle(rec_models, key):
    """Draw the popularity-correction A/B control (segmented control matching the Similar tab's
    toggle aesthetic, plus an explanatory caption when the twin is active). No-op when only prod is
    loaded. Call sites gate this on results already being on screen, so a toggle with nothing to
    act on never clutters the empty initial state. Framed as two trained models — the Menon
    correction is baked into training, so "Off" loads the separately-trained α=0 twin rather than
    flipping an inference knob; on change Streamlit reruns and _active_rec_model picks up the new
    selection (the recompute keys it off the variant)."""
    labels = list(rec_models)
    if len(labels) == 1:
        return
    choice = st.segmented_control(
        "Popularity correction", labels, default=labels[0], selection_mode="single",
        required=True, key=key,
        help=("Two separately-trained models. The Menon α correction is applied during training "
              "only — inference is plain dot products either way — so this loads a different "
              "model, not a runtime knob. α=0.5 keeps recommendations on-taste; α=0 lets popular "
              "titles dominate.")) or labels[0]
    if choice != labels[0]:
        st.caption("⚠️ Popularity correction **off** — watch popular blockbusters crowd out niche, "
                   "on-genre picks. This is a failure mode common in recommender models, where the "
                   "model learns to 'play it safe' by often recommending popular movies.")
    else:
        st.caption("Popularity correction **on** — a training-time logit adjustment (α = 0.5) "
                   "counteracts the popularity bias the model would otherwise learn from the "
                   "ratings data. Flip it off to see the difference.")


# ── Tab: Recommend ────────────────────────────────────────────────────────────

def tab_recommend(rec_models, fs, all_ids, ts_inference, posters, tmdb_ids):
    st.caption(
        "Select movies you love and optionally refine with genome tags. "
        "The model builds your taste embedding from the movies' content — curated genome tags plus "
        "web-scraped, LLM-extracted features — and scores every movie in the corpus."
    )
    all_titles = fs['popularity_ordered_titles']

    # Flag handling must happen before widgets are instantiated
    if st.session_state.pop('_clear_rec', False):
        for key in ('rec_liked', 'rec_genome_tags'):
            st.session_state[key] = []
        for key in ('rec_df', 'rec_shown', 'rec_query', 'rec_seed_key',
                    'rec_alpha', '_rec_scroll'):
            st.session_state.pop(key, None)

    liked_titles = st.multiselect("Favorite Movies", all_titles, key='rec_liked',
                                  max_selections=30)

    with st.expander("Refine by Genome Tags (optional)"):
        st.caption(
            "Select content descriptors — tones, themes, settings, cultural touchstones "
            "(e.g. 'atmospheric', 'cyberpunk', 'world war ii'). "
            "The 5 most representative movies for these tags will be added as implicit likes."
        )
        genome_tag_names     = sorted(fs['genome_tag_names'][tid] for tid in fs['genome_tag_names'])
        selected_genome_tags = st.multiselect("Genome tags", genome_tag_names, key='rec_genome_tags',
                                              max_selections=10)

    # Resolve the active model silently here (needed below to score). The A/B toggle itself is
    # rendered further down, only once results are on screen — see _render_rec_model_toggle.
    variant, model, all_embs = _active_rec_model(rec_models, 'rec_alpha')

    st.markdown("""
        <style>
        div[data-testid="stButton"] > button[kind="primary"] {
            display: block;
            margin: 1rem auto;
            padding: 0.6rem 1.5rem;
            font-size: 1.05rem;
            font-weight: 600;
            white-space: nowrap;   /* keep the CTA on one line inside its column */
        }
        /* The one-shot scroll helper (components.html, height 0) still costs a flex-gap slot;
           collapse its element container so it never adds blank space above the results. */
        div[data-testid="stElementContainer"]:has(iframe[height="0"]) { display: none; }
        /* Scroll targets for the one-shot results scroll — keep them clear of the toolbar. */
        .st-key-rec_alpha, .st-key-rec_grid { scroll-margin-top: 2rem; }
        </style>
    """, unsafe_allow_html=True)
    _, btn_col, clear_col = st.columns([2, 1, 2], vertical_alignment="center")
    # Clear All renders only once there is something to clear — on the empty form it read as
    # clutter and competed visually with the primary CTA.
    has_state = bool(liked_titles or selected_genome_tags
                     or st.session_state.get('rec_df') is not None)
    if has_state and clear_col.button("Clear All"):
        st.session_state['_clear_rec'] = True
        st.rerun()
    if btn_col.button("Get Recommendations", type="primary", use_container_width=True):
        if not liked_titles and not selected_genome_tags:
            st.warning("Select at least one movie or genome tag.")
        else:
            anchor_tag_title_pairs = []
            if selected_genome_tags:
                name_to_idx = {
                    fs['genome_tag_names'][tid]: fs['genome_tag_to_i'][tid]
                    for tid in fs['genome_tag_to_i']
                }
                seen_titles = set(liked_titles)  # exclude explicit likes from anchors
                for tag in selected_genome_tags:
                    if tag not in name_to_idx:
                        continue
                    tag_idx     = name_to_idx[tag]
                    sorted_mids = sorted(
                        fs['top_movies'],
                        key=lambda mid: float(fs['movieId_to_genome_tag_context'][mid][tag_idx]),
                        reverse=True,
                    )
                    count = 0
                    for mid in sorted_mids:
                        if count >= _ANCHORS_PER_TAG:
                            break
                        title = fs['movieId_to_title'][mid]
                        if title not in seen_titles:
                            anchor_tag_title_pairs.append((tag, title))
                            seen_titles.add(title)
                            count += 1
            # Store the model-independent query (likes + anchors). The score block below (re)runs it
            # against whichever α variant is selected, so flipping the toggle re-ranks the SAME
            # query immediately — no need to re-press the button.
            anchors_by_tag = {}
            for tag, title in anchor_tag_title_pairs:
                anchors_by_tag.setdefault(tag, []).append(title)
            # Captions are committed WITH the query so they always describe the results on
            # screen, even after the user edits the form without re-running. Wording matches
            # the Examples tab ("Because you like…"), keeping the two tabs' results blocks twins.
            st.session_state['rec_query'] = {
                'liked_with_weights': (
                    [(t, _LIKED_MOVIE)  for t in liked_titles] +
                    [(t, _ANCHOR_MOVIE) for _, t in anchor_tag_title_pairs]
                ),
                'exclude_titles': liked_titles,
                'liked_caption': (
                    "Because you like these movies: " + ", ".join(liked_titles)
                    if liked_titles else None),
                'tags_caption': (
                    "Because you like these genome tags: " + ", ".join(selected_genome_tags)
                    if selected_genome_tags else None),
                # grouped per tag ("cyberpunk: A, B, C · dystopia: X, Y") — repeating the tag
                # on every anchor made the caption twice as long for no information
                'anchor_caption': (
                    "Genome anchors — " + " · ".join(
                        f"{tag}: " + ", ".join(titles)
                        for tag, titles in anchors_by_tag.items())
                    if anchor_tag_title_pairs else None),
            }

    # Recompute when the committed query OR the α variant changes (user tower + item embeddings
    # both differ between the two models, so the whole scoring is redone per variant).
    query = st.session_state.get('rec_query')
    if query is not None:
        cache_key = (tuple(query['liked_with_weights']), variant)
        if st.session_state.get('rec_seed_key') != cache_key:
            with torch.no_grad():
                user_emb = _build_user_embedding(
                    model, fs, query['liked_with_weights'], [],
                    [], [], ts_inference,
                )
            df = _score_movies(user_emb, all_ids, all_embs, fs,
                               exclude_titles=query['exclude_titles'])
            _store_results(df, 'rec')
            st.session_state['rec_seed_key'] = cache_key
            st.session_state['_rec_scroll']  = True   # fresh results → scroll them into view

    if 'rec_df' in st.session_state:
        _render_rec_model_toggle(rec_models, 'rec_alpha')
        for cap in ('liked_caption', 'tags_caption', 'anchor_caption'):
            if query and query.get(cap):
                st.caption(query[cap])
        # One-shot smooth scroll to the fresh results, which render below the fold. The script
        # runs in components.html's same-origin iframe, so it can reach the parent document.
        # Target: the α-toggle's keyed container (top of the results section), falling back to
        # the poster grid when only one model is loaded and the toggle doesn't render.
        if st.session_state.pop('_rec_scroll', False):
            components.html(
                "<script>const d = window.parent.document;"
                "(d.querySelector('.st-key-rec_alpha') || d.querySelector('.st-key-rec_grid'))"
                "?.scrollIntoView({behavior: 'smooth', block: 'start'});</script>",
                height=0,
            )
    _show_results('rec', posters, fs, tmdb_ids)


# ── Tab: Recommend (Examples) ─────────────────────────────────────────────────

def tab_recommend_examples(rec_models, fs, all_ids, ts_inference, posters, tmdb_ids):
    st.caption("Select a pre-built user profile to see what the model recommends for that taste.")
    selected_profile = st.selectbox(
        "Profile",
        options=[None] + EXAMPLE_PROFILES,
        format_func=lambda x: "Choose a profile..." if x is None else x,
        label_visibility="collapsed",
    )

    if not selected_profile:
        st.session_state.pop('examples_profile', None)
        return

    fav_movies  = USER_TYPE_TO_FAVORITE_MOVIES[selected_profile]
    dis_movies  = USER_TYPE_TO_DISLIKED_MOVIES[selected_profile]
    genome_tags = USER_TYPE_TO_GENOME_TAGS.get(selected_profile, [])

    # Debug: report any fav_movies that didn't resolve to a corpus title
    missing = [t for t in fav_movies if t not in fs['title_to_movieId']]
    if missing:
        st.warning("⚠️ Not found in corpus (check title format): " + ", ".join(missing))

    # A profile is selected (results follow below), so the toggle always renders here — its empty
    # state was never the problem the gating addresses.
    _render_rec_model_toggle(rec_models, 'examples_alpha')
    variant, model, all_embs = _active_rec_model(rec_models, 'examples_alpha')

    if st.session_state.get('examples_profile') != (selected_profile, variant):
        # For genome-driven profiles, compute anchor movies from tags
        anchor_tag_title_pairs = []
        if genome_tags:
            name_to_idx = {
                fs['genome_tag_names'][tid]: fs['genome_tag_to_i'][tid]
                for tid in fs['genome_tag_to_i']
            }
            seen_titles = set(fav_movies)  # exclude fav movies from anchors, matches canary
            for tag in genome_tags:
                if tag not in name_to_idx:
                    continue
                tag_idx = name_to_idx[tag]
                sorted_mids = sorted(
                    fs['top_movies'],
                    key=lambda mid: float(fs['movieId_to_genome_tag_context'][mid][tag_idx]),
                    reverse=True,
                )
                count = 0
                for mid in sorted_mids:
                    if count >= _ANCHORS_PER_TAG:
                        break
                    title = fs['movieId_to_title'][mid]
                    if title not in seen_titles:
                        anchor_tag_title_pairs.append((tag, title))
                        seen_titles.add(title)
                        count += 1

        liked_with_weights = (
            [(t, _LIKED_MOVIE)  for t in fav_movies] +
            [(t, _ANCHOR_MOVIE) for _, t in anchor_tag_title_pairs]
        )

        with torch.no_grad():
            user_emb = _build_user_embedding(
                model, fs, liked_with_weights, dis_movies,
                [], [], ts_inference,
            )
        df = _score_movies(user_emb, all_ids, all_embs, fs,
                           exclude_titles=fav_movies + dis_movies +
                                          [t for _, t in anchor_tag_title_pairs])
        _store_results(df, 'examples')
        if anchor_tag_title_pairs:
            # same grouped format as the Recommend tab ("tag: A, B, C · tag: X, Y")
            anchors_by_tag = {}
            for tag, title in anchor_tag_title_pairs:
                anchors_by_tag.setdefault(tag, []).append(title)
            st.session_state['examples_anchor_caption'] = (
                "Genome anchors — " + " · ".join(
                    f"{tag}: " + ", ".join(titles)
                    for tag, titles in anchors_by_tag.items()))
        else:
            st.session_state.pop('examples_anchor_caption', None)
        st.session_state['examples_profile'] = (selected_profile, variant)

    # No "Recommendations for: <profile>" heading — the selectbox above already names the
    # profile, and the "Because you like…" captions carry the seeds (matches the Recommend
    # tab, which is likewise heading-less).
    if fav_movies:
        st.caption("Because you like these movies: " + ", ".join(fav_movies))
    if genome_tags:
        st.caption("Because you like these genome tags: " + ", ".join(genome_tags))
    anchor_caption = st.session_state.get('examples_anchor_caption')
    if anchor_caption:
        st.caption(anchor_caption)
    _show_results('examples', posters, fs, tmdb_ids)


# ── Tab: Ask (natural language → LLM extraction → trained model) ──────────────

def _anthropic_api_key():
    """The Anthropic key for the conversational tab: Streamlit secret first (the deployed path,
    set in the Community Cloud dashboard), then ANTHROPIC_API_KEY from the environment (convenient
    for local runs). None when neither is set — the tab then shows a setup notice and never calls
    the API. Wrapped in try/except because st.secrets raises when no secrets.toml exists locally."""
    try:
        key = st.secrets.get('ANTHROPIC_API_KEY')
    except Exception:
        key = None
    return key or os.environ.get('ANTHROPIC_API_KEY')


@st.cache_resource
def _llm_daily_budget():
    """The global daily call counter behind _LLM_DAILY_CAP. st.cache_resource hands every session
    in the container the SAME dict, so this counts extraction calls across all visitors (unlike
    st.session_state, which is per-browser). 'date' rolls the count over at midnight; the lock
    guards concurrent sessions. Resets on app restart/redeploy — acceptable, the Console spend
    limit is the hard backstop."""
    return {'lock': threading.Lock(), 'date': None, 'count': 0}


@st.cache_resource
def _load_ask_examples():
    """serving/ask_examples.json — the pre-generated example boards (9 theme roots, each with 5–6
    related-prompt children, built by tools/gen_ask_examples.py through the same extract→recommend pipeline the
    live path runs). A theme-card / riffing-chip click replays a stored report with zero API cost,
    so the tour works with no key, off the daily budget, and with deterministic boards for live
    demos. Returns None when the artifact is absent — the cards simply don't render. Ids missing
    from 'examples' (a partially regenerated artifact) are filtered out rather than crashing."""
    try:
        with open('serving/ask_examples.json') as f:
            data = json.load(f)
    except Exception:
        return None
    known = data.get('examples', {})
    data['roots'] = [r for r in data.get('roots', []) if r in known]
    data['tree']  = {r: [c for c in kids if c in known]
                     for r, kids in data.get('tree', {}).items()}
    return data if data['roots'] else None


def _report_to_df(report, fs):
    """Adapt src.llm_frontend.recommend()'s report into the poster-feed DataFrame the other tabs
    use (Title / Genres / Top Genome Tags), so the conversational results render as the same cards."""
    i_to_name = _build_genome_i_to_name(fs)
    rows = []
    for title, genres, _year, _score in report['recs']:
        mid = fs['title_to_movieId'].get(title)
        rows.append({
            'Title':           title,
            'Genres':          ', '.join(genres),
            'Top Genome Tags': _top_genome_tags(mid, fs, i_to_name) if mid is not None else '',
        })
    return pd.DataFrame(rows)


def _badges(names, color):
    """Markdown badge chips (`:color-badge[…]`) for a list of labels. Square brackets would
    terminate the directive early, so they are folded to parens."""
    return " ".join(f":{color}-badge[{str(n).replace('[', '(').replace(']', ')')}]" for n in names)


def _render_ask_debug(report):
    """A collapsed expander showing how the request was interpreted — the structured fields the LLM
    produced and what they resolved to. The plan keeps raw LLM output out of the default UI (no
    free-chatbot channel); this panel is opt-in, for the portfolio narrative of how the layer works.
    It leads with the pipeline's own intent chips (the human-readable echo of every parsed slot);
    the raw extraction JSON sits behind a toggle so the default view reads as product, not debug."""
    ex  = report['extraction']
    res = report['resolution']
    with st.expander("Under the hood — what the model was told (normally hidden)"):
        st.caption(
            "The LLM only fills in the structured fields below; the trained two-tower model does the "
            "actual retrieval. This panel is here for transparency — end users never see LLM output."
        )
        if report.get('intent_chips'):
            st.markdown("**Understood as:** " + _badges(report['intent_chips'], 'blue'))
        liked    = [c for _, c, _ in res['liked'] if c]
        disliked = [c for _, c, _ in res['disliked'] if c]
        dropped  = [r for r, c, _ in res['liked'] + res['disliked'] if not c]
        if liked:
            st.markdown("**Resolved likes:** " + ", ".join(liked))
        if disliked:
            st.markdown("**Resolved dislikes:** " + ", ".join(disliked))
        if report['anchors']:
            st.markdown(f"**Mood anchors** (weight {report['anchor_weight']}): "
                        + _badges(report['anchors'], 'gray'))
        if dropped:
            st.markdown("**Titles not matched in the catalog (dropped):** " + ", ".join(dropped))
        if report['fallback']:
            st.markdown("_No usable taste signal extracted — showing popular titles._")
        if st.toggle("Show raw extraction (consumed internally)", key='ask_raw_extraction'):
            st.json(ex)


def _ask_clear_results():
    """Wipe the active board — report, title, riffing row, poster feed — returning the tab
    to the landing view (theme cards + search bar). on_click for the results view's
    '← Back' escape hatch."""
    for k in ('ask_report', 'ask_title', 'ask_theme', 'ask_df', 'ask_shown'):
        st.session_state.pop(k, None)


def _ask_open_board(examples, fs, example_id):
    """Activate one pre-generated example board (zero API cost): on_click for the landing's
    theme cards and the workhorse of the riffing chips' on_change. Writes the same session keys
    the live path writes — the results view can't tell canned from live — plus 'ask_theme', the
    root whose riffing row to show: the id itself for a root, its parent for a leaf, so the
    sibling chips stay on screen while one of them is active."""
    entry = examples['examples'][example_id]
    st.session_state['ask_report'] = entry['report']
    st.session_state['ask_title']  = entry['query']    # heading echoes the full prompt
    theme = (example_id if example_id in examples['tree'] else
             next((r for r, kids in examples['tree'].items() if example_id in kids), None))
    st.session_state['ask_theme'] = theme
    if theme:
        # Seed the riffing row's selection (legal — callbacks run before widgets instantiate)
        # so a leaf opened straight from the landing grid renders lit among its siblings; a
        # root leaves the row unlit. Via the riff on_change this writes the value the widget
        # already holds — a no-op.
        st.session_state[f'ask_riff_{theme}'] = example_id if example_id != theme else None
    _store_results(_report_to_df(entry['report'], fs), 'ask')


def _on_ask_riff_change(examples, fs, theme):
    """on_change for the riffing chips: swap boards in place, staying on the results view. A
    chip toggled OFF falls back to the theme's own headline board rather than an empty panel."""
    leaf = st.session_state.get(f'ask_riff_{theme}')
    _ask_open_board(examples, fs, leaf or theme)


def _ask_title(query):
    """Results-heading text for a LIVE request: the prompt itself, whitespace-collapsed and capped
    at a word boundary so a long sentence stays a tidy heading that never cuts mid-title
    ("…Hachi: A Do…"). Canned chips use _ask_pill_text (the query, truncated) instead."""
    q = " ".join((query or "").split())
    if len(q) <= 80:
        return q
    return q[:80].rsplit(' ', 1)[0] + "…"


# 'More:' riffing chips show the query itself (never a separate summary that could drift out of
# sync with the prompt), truncated to keep the row tidy. 48 ≈ the median leaf-query length, so
# most chips read in full and only the long anchored ones get an ellipsis.
_ASK_PILL_MAXLEN = 48


def _ask_pill_text(query):
    """A 'More:' chip's text: the query, whitespace-collapsed and cut at a word boundary to
    _ASK_PILL_MAXLEN. It is a PREFIX of the real prompt (and of the board title the chip loads),
    so the chip and the prompt can never say different things — the ellipsis only caps width."""
    q = " ".join((query or "").split())
    if len(q) <= _ASK_PILL_MAXLEN:
        return q
    return q[:_ASK_PILL_MAXLEN].rsplit(' ', 1)[0] + "…"


_ASK_CATALOG_NOTE = ("Disclaimer: catalog is MovieLens 32M (movies with 200+ ratings) — coverage "
                     "effectively ends around 2019, so most films after that simply aren't in the "
                     "dataset.")

# The nine roots (r1–r9) now fill the landing grid as a clean 3×3 on their own. Earlier, with
# only seven roots, two leaf boards were promoted here to round the grid out to 9; adding the
# Sharks (r8) and Sports (r9) roots retired that stopgap.
_ASK_LANDING_EXTRAS = ()

# Invitation copy, not a canned example: nudges the query shapes that work (a vibe, an
# era, named favorites) without pushing one specific board.
_ASK_PLACEHOLDER = "What are you in the mood for? A vibe, an era, a few films you love…"


def _ask_handle_live(art, fs, utterance):
    """One live request: budget gates → hosted Haiku extraction → recommend(). The tab
    renders one shared search bar over both views (chat_input is a trigger widget — it
    returns the submitted text once and None on every other rerun, so this fires exactly
    once per submit); notices render at the call site. On success, writes the same session
    keys a card writes and reruns straight into the (fresh) results view."""
    if not utterance:
        return
    api_key = _anthropic_api_key()
    calls   = st.session_state.get('ask_calls', 0)
    budget  = _llm_daily_budget()
    with budget['lock']:   # roll the day + read together; check-then-increment races
        today = datetime.date.today().isoformat()   # are tolerable (friction, not a
        if budget['date'] != today:                 # security boundary)
            budget['date'], budget['count'] = today, 0
        daily_used = budget['count']
    if api_key is None:
        st.info(
            "The conversational tab needs an Anthropic API key. Set `ANTHROPIC_API_KEY` in "
            "`.streamlit/secrets.toml` (or the environment) — see the README. The manual "
            "**Recommend** tab works without a key."
        )
    elif calls >= _LLM_SESSION_CAP:
        st.warning(
            f"Per-session limit reached ({_LLM_SESSION_CAP} requests). Refresh to start over "
            "— this cap keeps the demo's API cost negligible."
        )
    elif daily_used >= _LLM_DAILY_CAP:
        st.warning(
            "Today's demo budget is used up — the **Recommend** tab works without the LLM, "
            "or come back tomorrow."
        )
    else:
        extraction = None
        try:
            with st.spinner("Understanding your request…"):
                extraction = extract_query(utterance, fs, api_key=api_key)
        except Exception as exc:  # surface, never crash the app
            st.error(f"Couldn't reach the language model — try again in a moment. "
                     f"({type(exc).__name__})")
        if extraction is not None:
            st.session_state['ask_calls'] = calls + 1
            with budget['lock']:
                budget['count'] += 1
            report = recommend(art.frontend_ctx, extraction, top_n=_TOTAL_RESULTS)
            _store_results(_report_to_df(report, fs), 'ask')
            st.session_state['ask_report'] = report
            st.session_state['ask_title']  = _ask_title(utterance)   # heading echoes the prompt
            st.session_state['ask_theme']  = None                    # live boards have no riff row
            st.rerun()   # this run already drew the current view — repaint as the fresh results


def _ask_landing(fs, examples):
    """The Ask tab's first page: heading + subtitle and a 3-across grid of theme cards (one
    per example root plus the promoted leaves, each card its full prompt — a click opens the
    frozen board through _ask_open_board). The floating search bar and the footer are shared
    with the results view and rendered by tab_ask."""
    st.markdown("## Ask")
    # Subtitle = the payoff (what you get back); the search-bar placeholder covers what to
    # type — the two deliberately don't repeat each other (YouTube's Ask split).
    st.markdown(
        "Describe a movie mood in plain English — get an instant board of matching films."
    )
    if examples:
        # Roots first, then the promoted leaves (skipping any id a partially regenerated
        # artifact dropped). One st.columns row per 3 cards (not one st.columns(3) filled
        # column-major): the desktop grid is identical either way, but on phones the rows
        # stack in reading order instead of column-by-column.
        cards = examples['roots'] + [i for i in _ASK_LANDING_EXTRAS
                                     if i in examples['examples']]
        with st.container(key='ask_cards'):
            for row in range(0, len(cards), 3):
                cols = st.columns(3)
                for col, card in zip(cols, cards[row:row + 3]):
                    col.button(
                        examples['examples'][card]['query'], key=f'ask_card_{card}',
                        use_container_width=True, on_click=_ask_open_board,
                        args=(examples, fs, card),
                    )


def _ask_results(fs, examples, posters, tmdb_ids):
    """The board view: '← Back' top-left (the escape hatch to the landing), the prompt as
    the board title, and — canned boards only — a 'More:' chip row of the theme's sibling
    boards, inline on one line. Each chip is a standalone board swapped in place at zero
    API cost (deliberate drifts off the theme, not refinements); live prompts carry no
    theme, so no row. Below: the unchanged results stack (relaxed-constraint notice,
    'under the hood' expander, poster feed). The floating search bar and the footer are
    shared with the landing and rendered by tab_ask, so a new search can start from
    anywhere in the feed."""
    st.button("← Back", key='ask_back', on_click=_ask_clear_results)
    st.markdown(f"## {st.session_state.get('ask_title', '')}")
    theme = st.session_state.get('ask_theme')
    if examples and examples['tree'].get(theme):
        st.pills(
            # Chip text is the query itself (truncated) — a 'More:' pill reads as a PREFIX of the
            # exact prompt it loads (and the board title it becomes), so it can never drift out of
            # sync the way a separate short label did. The `label` field is no longer surfaced.
            "More:", examples['tree'][theme], selection_mode="single",
            format_func=lambda i: _ask_pill_text(examples['examples'][i]['query']),
            key=f'ask_riff_{theme}', on_change=_on_ask_riff_change,
            args=(examples, fs, theme),
        )
    report  = st.session_state['ask_report']
    relaxed = report.get('relaxed_constraints') or []
    if relaxed:
        labels = {'require_attributes': 'format', 'require_genome_tags': 'vibe/setting',
                  'require_genres': 'genre', 'require_keyword_concepts': 'topic'}
        dropped = ", ".join(labels.get(r, r) for r in relaxed)
        st.info(f"No exact matches for every constraint — showing the closest titles, with these "
                f"relaxed: **{dropped}**. Identity filters (people, franchise, rating, year) were kept.")
    _render_ask_debug(report)
    _show_results('ask', posters, fs, tmdb_ids)


def tab_ask(art, posters, tmdb_ids):
    """Two views, switched on whether a board is active (st.session_state['ask_report']): the
    LANDING (theme-card grid) and the RESULTS view (back button, prompt echo, 'More:' chips,
    poster grid) — an in-tab page pair, YouTube-"Ask"-style, with ONE search bar floating
    pinned to the viewport bottom over both. Cards and chips replay frozen
    serving/ask_examples.json boards for free; only the typed path calls the hosted LLM.
    '← Back' returns to the first page; the bar starts a fresh live board from either view."""
    fs = art.fs
    st.markdown("""
        <style>
        /* Landing theme cards: filled rounded tiles, tall enough that the grid lands as even
           rows, prompt text left-aligned and wrapping. color-mix over currentColor gives a
           subtle fill that tracks the theme (dark AND light) without hardcoding either. */
        .st-key-ask_cards { gap: 0.75rem; }
        .st-key-ask_cards button {
            min-height: 4.25rem;
            border-radius: 1rem;
            padding: 0.6rem 1rem;
            border: none;
            background: color-mix(in srgb, currentColor 8%, transparent);
        }
        .st-key-ask_cards button:hover {
            background: color-mix(in srgb, currentColor 14%, transparent);
        }
        /* Left-align the prompt text: Streamlit's generated button styles center the label
           through a wrapper-div → span → markdown-container chain (and win the specificity
           coin-toss), so force every layer full-width and left-aligned. */
        .st-key-ask_cards button,
        .st-key-ask_cards button * { text-align: left !important; }
        .st-key-ask_cards button > div,
        .st-key-ask_cards button span,
        .st-key-ask_cards button div[data-testid="stMarkdownContainer"] {
            display: block;
            width: 100%;
        }
        .st-key-ask_cards button p { font-size: 1.02rem; }
        /* The one search bar (both views): floats pinned to the viewport bottom,
           YouTube-style. Legal inside st.tabs: the bar lives in the Ask tab panel, and
           hidden panels hide their fixed children — so it vanishes whenever another tab
           is active. */
        .st-key-ask_search {
            position: fixed;
            bottom: 1rem;
            left: 50%;
            transform: translateX(-50%);
            width: min(44rem, calc(100vw - 2rem));
            z-index: 100;
        }
        /* Taller, more visible composer: room for ~3 lines, placeholder up top. (No
           align-items here — stChatInput is a COLUMN flex; flex-end would collapse and
           right-shove its content column.) */
        .st-key-ask_search [data-testid="stChatInput"] {
            min-height: 6.5rem;
            position: relative;
            border-radius: 1.25rem;
            box-shadow: 0 4px 24px rgba(0, 0, 0, 0.45);
        }
        .st-key-ask_search [data-testid="stChatInput"] textarea {
            min-height: 4.75rem;
            padding-right: 3.5rem;   /* text never runs under the corner-pinned arrow */
        }
        /* Reserve room so the cards / last poster row / footer scroll clear of the bar. */
        [data-baseweb="tab-panel"]:has(.st-key-ask_search) { padding-bottom: 10rem; }
        /* The send arrow: bigger, pinned to the box's bottom-right corner (its native flex
           row hugs the top of the tall composer, so flex alignment can't reach the bottom). */
        .st-key-ask_search [data-testid="stChatInputSubmitButton"] {
            width: 2.75rem;
            height: 2.75rem;
            position: absolute;
            right: 0.5rem;
            bottom: 0.5rem;
        }
        .st-key-ask_search [data-testid="stChatInputSubmitButton"] svg {
            width: 1.75rem;
            height: 1.75rem;
        }
        /* 'More:' riffing row: label and chips inline on one line (wrapping on phones). */
        [class*="st-key-ask_riff_"] [data-testid="stButtonGroup"] {
            display: flex;
            flex-direction: row;
            align-items: center;
            column-gap: 0.6rem;
        }
        [class*="st-key-ask_riff_"] [data-testid="stWidgetLabel"] { flex: 0 0 auto; }
        [class*="st-key-ask_riff_"] [data-testid="stWidgetLabel"] p {
            font-size: 1.1rem;
            font-weight: 600;
        }
        /* Results view: '← Back' reads better a notch larger than a default button. */
        .st-key-ask_back button { font-size: 1.05rem; padding: 0.45rem 1rem; }
        </style>
    """, unsafe_allow_html=True)
    examples = _load_ask_examples()
    if st.session_state.get('ask_report') is None:
        _ask_landing(fs, examples)
    else:
        _ask_results(fs, examples, posters, tmdb_ids)
    # ── Search bar (Enter to submit; no button) ───────────────────────────────
    # One st.chat_input for both views, floating pinned to the viewport bottom (the
    # position:fixed lives on the keyed container — this DOM slot only decides where its
    # budget/error notices flow: below, in normal flow above the footer). Submitting starts
    # a fresh live board wherever the user is.
    with st.container(key='ask_search'):
        utterance = st.chat_input(_ASK_PLACEHOLDER, key='ask_chat')
    _ask_handle_live(art, fs, utterance)
    st.caption(_ASK_CATALOG_NOTE)   # footer disclaimer, both views


# ── Tab: Similar ──────────────────────────────────────────────────────────────

def tab_similar(fs, all_ids, sim_spaces, posters, tmdb_ids):
    # "More like this" exposed as a 2×2 the visitor switches between — feature source ×
    # representation — so the learned two-tower embedding can be compared directly against a raw
    # content-cosine baseline on the same seed. sim_spaces (built in load_artifacts) maps
    # {source: {representation: L2-normalized (n_movies, d) matrix}}; each cell ranks with one matmul.
    st.caption(
        "Find a movie's nearest neighbours by cosine similarity. Switch the **feature source** "
        "(Genome tags vs LLM features) and **representation** (the learned two-tower embedding vs "
        "raw feature cosine) to compare how each ranks the same movie."
    )

    selected = st.selectbox(
        "Movie to find similar titles for",
        options=[None] + fs['popularity_ordered_titles'],
        format_func=lambda x: "Choose a movie..." if x is None else x,
        key='sim_title', label_visibility="collapsed",
    )

    if not selected:
        for key in ('sim_seed_key', 'sim_df', 'sim_shown'):
            st.session_state.pop(key, None)
        return

    # 2×2 toggles, side by side to save vertical space. required + default keep one option always
    # selected (a click switches, never clears). Source shows only when LLM is available;
    # representation is the same set for every source, so precompute it. Rendered only once a movie
    # is picked (below the seed guard), so the empty initial state isn't cluttered by toggles with
    # nothing to act on.
    sources  = list(sim_spaces)
    reps     = list(next(iter(sim_spaces.values())))
    src_help = ("Genome Tags — curated 1,128-dim MovieLens genome relevance scores. "
                "LLM Features — 132 content features extracted from each film by an LLM.")
    rep_help = ("Learned embedding — the 32-dim item-tower projection trained end-to-end. "
                "Raw features — direct cosine over the un-projected feature vectors (no learning).")

    # Both controls in one keyed container; CSS (.st-key-sim_toggle_row) flexes them into a single
    # content-width row, tightly bound with a gap — not st.columns, which splits the full width
    # (huge desktop gap) and stacks with big margins on mobile.
    with st.container(key='sim_toggle_row'):
        if len(sources) > 1:
            source = st.segmented_control(
                "Feature source", sources, default=sources[0], selection_mode="single",
                required=True, key='sim_source', help=src_help) or sources[0]
        else:
            source = sources[0]
        rep = st.segmented_control(
            "Representation", reps, default=reps[0], selection_mode="single",
            required=True, key='sim_rep', help=rep_help) or reps[0]
    all_norm = sim_spaces[source][rep]

    # Recompute when the seed movie or either toggle changes — results appear immediately on
    # change, no button press.
    cache_key = (selected, source, rep)
    if st.session_state.get('sim_seed_key') != cache_key:
        row_of = {m: i for i, m in enumerate(all_ids)}
        mid    = fs['title_to_movieId'].get(selected)
        if mid not in row_of:
            st.error(f"'{selected}' not in corpus.")
            return

        with torch.no_grad():
            # Seed vector is the corpus row of the chosen matrix (already L2-normalized).
            seed_norm = all_norm[row_of[mid]:row_of[mid] + 1]
            sims      = (all_norm @ seed_norm.T).squeeze(-1)

        i_to_name = _build_genome_i_to_name(fs)
        rows = []
        for idx in sims.argsort(descending=True).tolist():
            candidate = all_ids[idx]
            if candidate == mid:
                continue
            rows.append({
                'Title':           fs['movieId_to_title'][candidate],
                'Genres':          ', '.join(fs['movieId_to_genres'][candidate]),
                'Top Genome Tags': _top_genome_tags(candidate, fs, i_to_name),
            })
            if len(rows) >= _TOTAL_RESULTS:
                break
        _store_results(pd.DataFrame(rows), 'sim')
        st.session_state['sim_seed_key'] = cache_key

    st.subheader(f"Similar to: {selected}")
    _show_results('sim', posters, fs, tmdb_ids)


# ── Tab: Explore Genres ───────────────────────────────────────────────────────

def tab_explore_genres(model, fs, all_ids, all_norm_genre, posters, tmdb_ids):
    st.subheader("Explore Genre Item Tower Embeddings")
    st.caption(
        "Queries the item genre embedding space directly — finds movies whose "
        "genre embedding best matches the selected genres."
    )
    genres          = fs['genres_ordered']
    selected_genres = st.multiselect("Genres", genres, key='explore_genre')
    if st.button("Explore", key='btn_genre'):
        if not selected_genres:
            st.warning("Select at least one genre.")
        else:
            ctx = [0.0] * len(genres)
            for g in selected_genres:
                ctx[fs['genre_to_i'][g]] = 1.0
            with torch.no_grad():
                query      = model.item_genre_tower(torch.tensor([ctx])).view(-1)
                query_norm = F.normalize(query.unsqueeze(0), dim=1)
                sims       = (all_norm_genre @ query_norm.T).squeeze(-1)
            i_to_name = _build_genome_i_to_name(fs)
            rows = []
            for idx in sims.argsort(descending=True).tolist():
                mid = all_ids[idx]
                rows.append({
                    'Title':           fs['movieId_to_title'][mid],
                    'Genres':          ', '.join(fs['movieId_to_genres'][mid]),
                    'Top Genome Tags': _top_genome_tags(mid, fs, i_to_name),
                })
                if len(rows) >= _TOTAL_RESULTS:
                    break
            _store_results(pd.DataFrame(rows), 'genres')

    _show_results('genres', posters, fs, tmdb_ids)


# ── Tab: Explore Genome Tags ──────────────────────────────────────────────────

def tab_explore_genome(me, fs, all_ids, all_norm_genome, posters, tmdb_ids):
    st.subheader("Explore Genome Tag Item Tower Embeddings")
    st.caption(
        "Select a genome tag to describe what you're looking for — a genre, tone, theme, "
        "setting, time period, plot element, or cultural touchstone "
        "(e.g. 'atmospheric', 'cyberpunk', 'world war ii', 'studio ghibli'). "
        f"The model anchors on the {_ANCHORS_PER_TAG} most representative movies for that tag, "
        "then finds similar movies in the genome embedding space."
    )
    genome_tag_names = sorted(fs['genome_tag_names'][tid] for tid in fs['genome_tag_names'])
    selected_tag     = st.selectbox(
        "Genome tag",
        options=[None] + genome_tag_names,
        format_func=lambda x: "Choose a genome tag..." if x is None else x,
        key='explore_genome', label_visibility="collapsed",
    )

    if not selected_tag:
        for key in ('genome_active', 'genome_df', 'genome_shown', 'genome_anchor_caption'):
            st.session_state.pop(key, None)
        return

    # Recompute only when the selection changes — results appear immediately on
    # selection, no button press.
    if st.session_state.get('genome_active') != selected_tag:
        name_to_idx = {
            fs['genome_tag_names'][tid]: fs['genome_tag_to_i'][tid]
            for tid in fs['genome_tag_to_i']
        }
        if selected_tag not in name_to_idx:
            st.warning("Genome tag did not match the vocabulary.")
            return
        tag_idx     = name_to_idx[selected_tag]
        sorted_mids = sorted(
            fs['top_movies'],
            key=lambda mid: float(fs['movieId_to_genome_tag_context'][mid][tag_idx]),
            reverse=True,
        )
        anchor_tag_title_pairs = []
        seen_titles = set()
        for mid in sorted_mids:
            if len(anchor_tag_title_pairs) >= _ANCHORS_PER_TAG:
                break
            title = fs['movieId_to_title'][mid]
            if title not in seen_titles:
                anchor_tag_title_pairs.append((selected_tag, mid, title))
                seen_titles.add(title)

        anchor_mids = [mid for _, mid, _ in anchor_tag_title_pairs]
        anchor_set  = set(anchor_mids)
        with torch.no_grad():
            query_emb  = torch.stack([
                me[m]['MOVIE_GENOME_TAG_EMBEDDING'].view(-1) for m in anchor_mids
            ]).mean(dim=0)
            query_norm = F.normalize(query_emb.unsqueeze(0), dim=1)
            sims       = (all_norm_genome @ query_norm.T).squeeze(-1)
        i_to_name = _build_genome_i_to_name(fs)
        rows = []
        for idx in sims.argsort(descending=True).tolist():
            mid = all_ids[idx]
            rows.append({
                'Title':           fs['movieId_to_title'][mid] + ('  ◀ ANCHOR' if mid in anchor_set else ''),
                'Genres':          ', '.join(fs['movieId_to_genres'][mid]),
                'Top Genome Tags': _top_genome_tags(mid, fs, i_to_name),
            })
            if len(rows) >= _TOTAL_RESULTS:
                break
        _store_results(pd.DataFrame(rows), 'genome')
        st.session_state['genome_anchor_caption'] = (
            "Genome anchors — "
            + " · ".join(f"{tag}: {title}" for tag, _, title in anchor_tag_title_pairs)
        )
        st.session_state['genome_active'] = selected_tag

    caption = st.session_state.get('genome_anchor_caption')
    if caption:
        st.caption(caption)
    _show_results('genome', posters, fs, tmdb_ids)


# ── Tab: Map ───────────────────────────────────────────────────────────────────

def _genre_color_map(genres_ordered):
    """Stable genre → hex color. A bright qualitative palette (Light24) reads well on the dark
    canvas; the synthetic '(no genres listed)' bucket gets a muted grey so it never competes with
    a real genre for attention. Colors are assigned in genres_ordered order so they're identical
    across reruns (the legend never reshuffles)."""
    palette = pcolors.qualitative.Light24
    cmap, ci = {}, 0
    for g in genres_ordered:
        if g == _MAP_NO_GENRE:
            cmap[g] = _MAP_NO_GENRE_HEX
        else:
            cmap[g] = palette[ci % len(palette)]
            ci += 1
    return cmap


@st.cache_resource
def _map_base_layers():
    """Per-genre Scatter3d-ready arrays for the catalog cloud — built once, on first Map-tab view.

    Buckets every catalog movie into ONE layer by its first-listed (primary) genre, so each genre
    is an independently toggleable trace driven by the genre filter. Each layer carries its 3D xyz
    plus pre-rendered hover customdata (title / full genre list / top genome tags). Movies with no
    genre are dropped entirely — '(no genres listed)' is not a real genre and never appears on the
    map. Cached as a resource: it's static for the process, and the per-request neighbor overlay is
    layered on top at render time without ever recomputing this. Lazy (not built in load_artifacts)
    so visitors who never open the tab don't pay for it at startup."""
    art = load_artifacts()
    fs, all_ids, coords = art.fs, art.all_ids, art.map_coords
    color_map = _genre_color_map(fs['genres_ordered'])
    i_to_name = _build_genome_i_to_name(fs)

    buckets = {g: [] for g in fs['genres_ordered']}
    for i, mid in enumerate(all_ids):
        genres = fs['movieId_to_genres'].get(mid) or [_MAP_NO_GENRE]
        primary = genres[0] if genres[0] in buckets else _MAP_NO_GENRE
        buckets[primary].append(i)

    layers = []
    for g in fs['genres_ordered']:
        if g == _MAP_NO_GENRE:        # not a real genre — keep it off the map entirely
            continue
        idx = buckets[g]
        if not idx:
            continue
        xyz = coords[idx]
        customdata = np.array([
            [fs['movieId_to_title'][all_ids[i]],
             ', '.join(fs['movieId_to_genres'][all_ids[i]]),
             _top_genome_tags(all_ids[i], fs, i_to_name)]
            for i in idx
        ], dtype=object)
        layers.append({'name': g, 'color': color_map[g], 'rows': idx,
                       'x': xyz[:, 0], 'y': xyz[:, 1], 'z': xyz[:, 2], 'customdata': customdata})
    return layers


@st.cache_resource
def _map_genome_highlight_rows():
    """For each genome-tag pill, the set of all_ids ROW indices it highlights on the map — every
    movie whose relevance for that tag clears _MAP_GENOME_MIN_RELEVANCE (so only movies that very
    clearly carry the tag light up, the way a genre pill keys on the single primary genre). Returns
    an ordered {tag_name: frozenset(rows)} so the pill row stays in _MAP_GENOME_TAGS order; tags
    missing from the vocabulary are skipped (defensive). Built once: the per-render highlight is then
    a set union over the picked pills, keyed by the same all_ids rows the base layers store."""
    art = load_artifacts()
    fs, all_ids = art.fs, art.all_ids
    name_to_i = {fs['genome_tag_names'][tid]: fs['genome_tag_to_i'][tid] for tid in fs['genome_tag_to_i']}
    G = np.stack([np.asarray(fs['movieId_to_genome_tag_context'][mid], dtype=np.float32) for mid in all_ids])
    out = {}
    for tag in _MAP_GENOME_TAGS:
        col = name_to_i.get(tag)
        if col is None:
            continue
        rows = np.where(G[:, col] >= _MAP_GENOME_MIN_RELEVANCE)[0]
        out[tag] = frozenset(int(r) for r in rows)
    return out


def _genome_neighbors_overlay(art, selected_title, k=_MAP_NEIGHBORS):
    """Spotlight a picked movie and its k nearest neighbors IN GENOME-TAG SPACE on the map.

    This tab is a *visualizer of the genome content space*, not a recommender: neighbors are ranked
    by cosine over the item tower's genome-tag projection (art.all_norm_genome) — the exact ranking
    the Similar tab's "Genome Tags / Learned embedding" uses — NOT through the user/retrieval tower.
    So the highlighted points are literally the nearest movies in the space the map is drawn from,
    which is why they sit right next to the pick. Returns {'pick': (xy, title),
    'neighbors': (xy, titles)}, or None if the title isn't in the corpus."""
    fs, all_ids, coords = art.fs, art.all_ids, art.map_coords
    row_of = {m: i for i, m in enumerate(all_ids)}
    mid    = fs['title_to_movieId'].get(selected_title)
    if mid not in row_of:
        return None
    i = row_of[mid]
    with torch.no_grad():
        sims = (art.all_norm_genome @ art.all_norm_genome[i:i + 1].T).squeeze(-1)
    nbr_rows   = [j for j in torch.argsort(sims, descending=True).tolist() if j != i][:k]
    nbr_titles = [fs['movieId_to_title'][all_ids[j]] for j in nbr_rows]
    return {
        'pick':      (coords[i], selected_title),
        'neighbors': (coords[nbr_rows], nbr_titles),
    }


def _map_figure(layers, highlight_genres, highlight_rows=None, overlay=None):
    """Assemble the 3D point cloud: per-genre Scatter3d traces + optional neighbor overlay.

    Every genre is always drawn. Two highlighters drive emphasis, both with the same pop-and-dim
    feel: `highlight_genres` (genre pills) lights up whole primary-genre buckets, while
    `highlight_rows` (genome-tag pills) is a set of all_ids ROW indices that cross-cuts genre — the
    movies that very clearly carry the picked tags. A point is emphasized if its genre is selected OR its row is in
    highlight_rows; emphasized points grow (_MAP_HL_SIZE) and stay bright, the rest fade to faint
    context (_MAP_HL_DIM_OPACITY). Nothing is added or removed, only emphasized; the Plotly legend is
    off (the pills beside the chart drive it). When any highlight is active a genre layer is split
    into its on/off point subsets so opacity (a per-trace property on Scatter3d) can differ within a
    genre — the genome pills need within-genre emphasis that whole-trace opacity can't express.

    Every movie fills a true volumetric position (its baked 3D genome-space coordinate — all three
    axes carry structure, so this preserves more of the high-dim neighbourhood than a flat 2D map,
    and a sphere SURFACE would not — that's only 2D). Points are Scatter3d (WebGL gl3d) — ~9k stay
    smooth on the free CPU tier. Base points dim when a movie is picked so the big pick/neighbor
    markers read on top. Axes are hidden — only relative position (the clusters) is meaningful — and
    aspectmode='data' keeps the cloud's true proportions as you spin it."""
    base_opacity = _MAP_DIM_OPACITY if overlay else _MAP_BASE_OPACITY
    highlight_rows = highlight_rows or set()
    highlighting = bool(highlight_genres) or bool(highlight_rows)
    hover = ("<b>%{customdata[0]}</b><br>%{customdata[1]}"
             "<br><span style='color:#9aa3ad'>%{customdata[2]}</span><extra></extra>")

    fig = go.Figure()

    def _add_subset(layer, mask, size, opacity):
        # One Scatter3d trace over a boolean subset of a genre layer (mask=None → whole layer).
        sel = slice(None) if mask is None else mask
        fig.add_trace(go.Scatter3d(
            x=layer['x'][sel], y=layer['y'][sel], z=layer['z'][sel], name=layer['name'],
            mode='markers',
            marker=dict(size=size, color=layer['color'], opacity=opacity),
            customdata=layer['customdata'][sel], hovertemplate=hover,
        ))

    for layer in layers:
        if not highlighting:
            _add_subset(layer, None, _MAP_BASE_SIZE, base_opacity)
            continue
        if layer['name'] in highlight_genres:
            on = np.ones(len(layer['rows']), dtype=bool)   # whole genre selected → all points pop
        else:
            on = np.fromiter((r in highlight_rows for r in layer['rows']),
                             dtype=bool, count=len(layer['rows']))
        if on.any():
            _add_subset(layer, on, _MAP_HL_SIZE, base_opacity)
        if (~on).any():
            _add_subset(layer, ~on, _MAP_BASE_SIZE, _MAP_HL_DIM_OPACITY)

    annotations = []
    if overlay:
        # In-graph legend for the overlay markers — kept in the plot's blank top-left rather than
        # below the chart (where it sits out of view). Only shown once a movie is picked.
        annotations.append(dict(
            xref='paper', yref='paper', x=0.01, y=0.99, xanchor='left', yanchor='top',
            showarrow=False, align='left',
            text=(f"<span style='color:{_MAP_PICK_HEX}'>⬤</span> your pick<br>"
                  f"<span style='color:{_MAP_NEIGHBOR_HEX}'>◆</span> "
                  f"{_MAP_NEIGHBORS} nearest genome-tag neighbors"),
            font=dict(size=12, color='#c8cdd4'),
            bgcolor='rgba(14,17,23,0.55)', bordercolor='#3a3f47', borderwidth=1, borderpad=6,
        ))
        nbr_xyz, nbr_titles = overlay['neighbors']
        fig.add_trace(go.Scatter3d(
            x=nbr_xyz[:, 0], y=nbr_xyz[:, 1], z=nbr_xyz[:, 2], name='Top neighbors', mode='markers',
            marker=dict(size=_MAP_NEIGHBOR_SIZE, color=_MAP_NEIGHBOR_HEX, symbol='diamond',
                        line=dict(width=1, color='#06343b')),
            customdata=np.array(nbr_titles, dtype=object).reshape(-1, 1),
            hovertemplate="<b>%{customdata[0]}</b><br>genome-space neighbor<extra></extra>",
        ))
        pick_xyz, pick_title = overlay['pick']
        fig.add_trace(go.Scatter3d(
            x=[float(pick_xyz[0])], y=[float(pick_xyz[1])], z=[float(pick_xyz[2])],
            name='Your pick', mode='markers',
            marker=dict(size=_MAP_PICK_SIZE, color=_MAP_PICK_HEX, symbol='circle',
                        line=dict(width=2, color=_MAP_PICK_EDGE)),
            customdata=np.array([[pick_title]], dtype=object),
            hovertemplate="<b>%{customdata[0]}</b><br>your pick<extra></extra>",
        ))

    hidden_axis = dict(visible=False, showbackground=False, showgrid=False, zeroline=False,
                       showticklabels=False, title='')
    fig.update_layout(
        template='plotly_dark',
        height=_MAP_HEIGHT,
        margin=dict(l=2, r=2, t=8, b=2),   # a couple px so the framing border isn't clipped
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=False,   # genre highlighting lives in the Streamlit pills beside the chart
        annotations=annotations,
        # Thin border around the plot 'world' — makes it obvious where a drag/scroll spins the cloud
        # vs. where it scrolls the page (the transparent canvas gives no edge cue otherwise).
        shapes=[dict(type='rect', xref='paper', yref='paper', x0=0, y0=0, x1=1, y1=1,
                     line=dict(color='#3a3f47', width=1), fillcolor='rgba(0,0,0,0)')],
        hoverlabel=dict(bgcolor='#0e1117', bordercolor='#444444', font=dict(size=12)),
        scene=dict(
            xaxis=hidden_axis, yaxis=hidden_axis, zaxis=hidden_axis,
            aspectmode='data', bgcolor='rgba(0,0,0,0)',
            # 'orbit' = free trackball rotation in EVERY direction (up/down AND side-to-side). The
            # plotly default, 'turntable', locks the up-axis so vertical drags barely tilt — that's
            # the unusable feel. center pinned at the origin + pan removed (config below) means a
            # drag only SPINS the cloud in place; it never slides around the screen.
            dragmode='orbit',
            # eye pulled in (was 1.6/1.6/1.0) so the cloud fills the framed box on first load
            camera=dict(eye=dict(x=1.0, y=1.0, z=0.63), center=dict(x=0, y=0, z=0)),
        ),
    )
    return fig


def _render_map_filters(col, all_genres, genome_rows_by_tag):
    """The highlighters beside the cloud — they replace the old Plotly legend (Streamlit can't drive
    a Plotly legend's clicks, so they could never reach the app). Two stacked multi-select pill
    groups: 'genre' lights up whole primary-genre buckets; 'genome tag' (below it) lights up the
    movies that very clearly carry a content tag, which cross-cuts genre. Both behave identically —
    pick some to make their points pop while the rest fade back; pick none and the whole cloud shows
    normally. Returns (highlight_genres set, highlight_rows set of all_ids row indices)."""
    with col:
        st.caption("**Highlight genre**")
        genres = st.pills("Genres", options=all_genres, selection_mode="multi",
                          key='map_genre_pills', label_visibility="collapsed")
        st.caption("**Highlight genome tag**")
        tags = st.pills("Genome tags", options=list(genome_rows_by_tag.keys()),
                        selection_mode="multi", key='map_genome_pills', label_visibility="collapsed")
    highlight_rows = set().union(*(genome_rows_by_tag[t] for t in tags)) if tags else set()
    return set(genres or []), highlight_rows


def tab_map(art):
    st.subheader("Map of the learned movie space")

    # Graceful degradation: bundles exported before this feature have no baked coordinates. Keep
    # the tab present (stable layout) but show a re-export notice instead of crashing.
    if art.map_coords is None:
        st.info(
            "This serving bundle predates the 3D map — the projection isn't baked in yet.\n\n"
            "Re-export to enable it: `python main.py export <checkpoint>` "
            "(`pip install umap-learn` first for the best projection)."
        )
        return

    st.caption(
        "All ~9,375 catalog movies in a **true 3D projection** of the item tower's learned "
        "genome-tag **content space** (the same one the *Similar* tab ranks in). Colored by primary "
        "genre — but genre was never an input, so the clustering is the model's own. **Drag to "
        "rotate, scroll to zoom, hover** for a title and its top genome tags; use the **genre** or "
        "**genome-tag pills** to light up a theme, or **pick a movie below** to spotlight its "
        "nearest neighbors."
    )

    selected = st.selectbox(
        "Movie to place on the map",
        options=[None] + art.fs['popularity_ordered_titles'],
        format_func=lambda x: "Choose a movie..." if x is None else x,
        key='map_movie', label_visibility="collapsed",
        help="Highlights this movie and its nearest neighbors in the genome-tag content space — "
             "the same ranking as the Similar tab's Genome / Learned-embedding option.")

    overlay = _genome_neighbors_overlay(art, selected) if selected else None

    layers     = _map_base_layers()
    all_genres = [layer['name'] for layer in layers]      # real genres present, in cloud order
    genome_rows_by_tag = _map_genome_highlight_rows()

    chart_col, filter_col = st.columns([4, 1], gap="medium")
    highlight_genres, highlight_rows = _render_map_filters(filter_col, all_genres, genome_rows_by_tag)

    fig = _map_figure(layers, highlight_genres, highlight_rows, overlay)
    # scrollZoom for wheel/pinch zoom; drop 'pan3d' so the cloud can't be dragged off-center —
    # left-drag is locked to orbit-rotation, so the user only ever spins it to find their movies.
    # (Click-to-select isn't wired: Streamlit's Plotly click handler only emits a selection for
    # sunburst/treemap points — it bails on a scatter3d point — so a node click can't reach Python.
    # The dropdown above is the selection path; hover shows a point's title and tags.)
    chart_cfg = {'displaylogo': False, 'scrollZoom': True,
                 'modeBarButtonsToRemove': ['pan3d', 'resetCameraLastSave3d']}
    with chart_col:
        st.plotly_chart(fig, use_container_width=True, theme=None, key='map_chart', config=chart_cfg)


# ── Tab: About ───────────────────────────────────────────────────────────────

def tab_about():
    # Left-aligned column that fills the page like the rest of the app, capped at
    # a readable max-width in CSS (.st-key-about) so prose lines don't run too long
    # on a wide monitor. width="stretch" fills the parent, so this is full-width on
    # mobile (where the parent is narrower than the cap, so it never bites) and a
    # wide, page-aligned column on desktop — no centered offset from the
    # left-aligned title and tab bar above it.
    with st.container(width="stretch", key="about"):
        _REPO_URL = "https://github.com/nickgreenquist/Movie-Recommender-System-PyTorch-TwoTower-Model"

        st.header("About this demo")
        st.markdown(
            "The live app embeds **you** — a user it has never seen — from a handful of movies you like, "
            "then ranks the whole catalog against your taste. This page walks through how that happens at "
            "serving time."
        )
        st.info(
            f"📖 **Full project write-up → [GitHub README]({_REPO_URL})**\n\n"
            "Architecture diagram, training recipe, offline metrics, and the *\\$200-vs-\\$200k* "
            "LLM-vs-genome feature experiment."
        )

        st.header("The model")
        st.markdown(
            "A **PyTorch two-tower network** trained on MovieLens 32M. A *user tower* and an *item tower* "
            "each emit an **L2-normalized 128-dim embedding**; because both are unit-norm, their dot "
            "product *is* cosine similarity — so a single `user · item` product ranks the whole catalog. "
            "Three choices drive what you see in the app:"
        )
        st.markdown(
            "- **No user-ID embedding.** A user is represented purely from taste signals — watch history "
            "(full / liked / disliked / rating-weighted pools over a shared item-ID table), genre "
            "affinity, content texture, and timestamp. That is the choice that lets the demo embed "
            "*you* — a brand-new user — on the fly from a few liked movies, with **no retraining and no "
            "user-level cold start**.\n"
            "- **Two content fingerprints per movie**, fused in the item tower: MovieLens's 1,128 curated "
            "**genome tags** *and* **132 features I built myself** (web-scraped from TMDB + Wikipedia, "
            "scored 0–1 by an LLM).\n"
            "- **Full-softmax training** with **Menon et al. (2021) logit-adjusted popularity correction** "
            "(α=0.5), so popular titles surface only when they actually fit the taste."
        )
        st.caption(
            f"Full architecture diagram, per-component tables, and training details: [README]({_REPO_URL})."
        )

        st.header("How the demo serves a recommendation")
        st.markdown(
            "The two-tower split is what makes this cheap to serve. Only the **user tower** runs per "
            "request; every movie's item vector is computed **once, offline**. A recommendation is three "
            "steps:"
        )
        st.markdown(
            "1. **Build your taste vector.** Your picks pass through the user tower → one 128-dim "
            "unit-norm vector. No ID lookup, no fine-tuning — built fresh from the signal you give it.\n"
            "2. **Score the catalog in a single matmul.** Item embeddings for all ~9,375 movies are "
            "precomputed and stacked into one matrix, so scoring the entire catalog is one matrix–vector "
            "product — both sides unit-norm, so every score is a cosine similarity.\n"
            "3. **Rank & filter.** Argsort the scores, drop the movies you already picked, return the top "
            "of the list."
        )
        st.code(
            "# the entire request-time scoring step — one matmul over the precomputed item matrix\n"
            "raw_scores = (all_embs @ user_emb.T).squeeze(-1)   # (n_movies,) cosine sims\n"
            "ranking    = raw_scores.argsort(descending=True)   # whole catalog, ranked",
            language="python",
        )
        st.markdown(
            "There are no per-candidate model calls and no approximate-nearest-neighbor index — the "
            "catalog is small enough that an **exact full-catalog matmul runs sub-second on Streamlit "
            "Cloud's free CPU tier**. Because the item embeddings are precomputed, the per-request cost "
            "is one cheap user-tower pass plus one dense matmul."
        )

        st.header("What ships to the server")
        st.markdown(
            "The app is fully self-contained: it loads a small `serving/` bundle and **never reads the "
            "training data**. The user tower is rebuilt from these artifacts alone."
        )
        st.markdown("""
| Artifact | What it is |
|---|---|
| `model.pth` | ~2 MB weights-only state_dict — the trained two-tower network |
| `movie_embeddings.pt` | precomputed 128-dim item embeddings for the whole catalog; the item tower never runs at request time |
| `feature_store.pt` | vocabs, index maps, model config + baked content-feature buffers, so the user tower rebuilds with no `data/` directory |
| `posters.json` / `tmdb_ids.json` | TMDB poster URLs + id map for the result cards |
| `ask_examples.json` | the *Ask* tab's pre-generated example boards — frozen LLM extractions + results, computed offline through the same retrieval path as a live query |
""", unsafe_allow_html=True)
        st.markdown(
            "Two serving details worth noting:\n"
            "- **Device-agnostic export.** The model trains on Apple-Silicon **MPS** but exports with "
            "`map_location='cpu'`, so it loads on a cloud box with no GPU. (MPS-saved tensors crash on an "
            "MPS-less host — the export step is what handles that.)\n"
            "- **Loaded once, cached per process** (`@st.cache_resource`). The model, the precomputed item "
            "matrix, and the normalized sub-spaces the Similar / Genres / Genome tabs rank against are "
            "built a single time, not on every interaction."
        )

        st.header("Two models behind one switch")
        st.markdown(
            "The **Popularity correction** toggle on the *Recommend* and *Examples* tabs isn't a runtime "
            "setting — the Menon correction is applied **only during training** (inference is plain dot "
            "products either way). Turning it **Off** instead **swaps in a second, separately-trained α=0 "
            "model**: both checkpoints *and* both precomputed item matrices are loaded and cached, and the "
            "switch re-scores against the other one."
        )
        st.markdown(
            "**Try it.** Flip it Off and watch popular blockbusters crowd out the niche, on-genre picks — "
            "a *Fantasy* taste collapses toward Star Wars / Star Trek, a *Heist* taste toward generic "
            "action tentpoles. That is exactly the failure mode the α=0.5 model is trained to fix."
        )

        st.header("The Ask tab — LLM as interface, not recommender")
        st.markdown(
            "The landing tab adds a natural-language layer on top of the same retrieval path. A "
            "small, fast LLM (**Claude Haiku**) parses your sentence into the structured input the "
            "user tower already consumes — liked/disliked titles, mood as genome-tag anchors, hard "
            "year / genre / people / topic filters — via a **forced tool call against a JSON "
            "schema**, so malformed output can never break the pipeline. The trained two-tower "
            "model still does all the retrieval; the LLM's output is consumed internally and never "
            "shown (the *Under the hood* expander on each result reveals the parsed fields, for "
            "the curious).\n"
            "- **The suggested-prompt boards are pre-generated.** Every theme card and riffing "
            "chip replays a board computed offline through the same extract → recommend pipeline "
            "as a live query — instant, deterministic, and **no API key needed**.\n"
            "- **Typed queries are budgeted.** Free-form requests call the hosted LLM behind "
            "per-session and global daily caps, which keeps the demo's API cost negligible."
        )

        st.header("What each tab demonstrates")
        st.markdown("""
| Tab | What it shows |
|---|---|
| **Ask** | The landing tab: plain-English requests, parsed by a small LLM into the model's input — with an instant, pre-generated example-board tour. See the section above. |
| **Recommend** | Build a taste vector from your *own* picks (optionally nudged with genome tags) → ranked poster grid. Cold-start-free serving in action. |
| **Examples** | Pre-built taste personas (Sci-Fi, Horror, Heist, …) to see the model's range without typing anything. |
| **Similar** | Nearest neighbors of any movie in the 128-dim space — with a learned-embedding vs. raw-feature toggle, over genome or LLM content. |
| **Genres** / **Genome** | Probe what the *item tower itself* encodes — query its genre and genome-tag sub-spaces directly. |
| **Map** | The whole catalog as an interactive **3D point cloud** — a true volumetric projection of the item tower's learned *content* embeddings, colored by genre — the model's thematic structure, made visible. Pick any movie to spotlight it and its nearest neighbors in genome-tag space. |
""", unsafe_allow_html=True)

        st.header("Limitations of the live demo")
        st.markdown("""
- **No user ID** — personalization is limited to the signals you provide here; there is no long-term profile.
- **~9,375-movie catalog** — only films with 200+ MovieLens ratings are served; very obscure titles aren't in the index.
- **Fixed timestamp** — every visitor is scored at the most-recent time bin, so the temporal signal is effectively constant in the app.
- **LLM content features are model-estimated** — thinly-documented films get weaker (or all-zero) feature vectors.
""")

        st.divider()
        st.info(
            f"📖 **Want the whole story? → [GitHub README]({_REPO_URL})**\n\n"
            "The complete architecture, the self-built LLM-feature pipeline and its head-to-head "
            "experiment against the curated genome tags, full training details, and offline metrics."
        )


# ── Layout ────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")
st.markdown("""
    <style>
    /* Keep the tab bar on one horizontally-scrollable line on mobile.
       Scope to the baseweb tab-list only — `stTabs > div:first-child` wraps the
       tab panels too in newer Streamlit, leaking white-space:nowrap into all
       tab content (unwrappable paragraphs running off-screen). */
    div[data-testid="stTabs"] div[data-baseweb="tab-list"] {
        overflow-x: auto;
        white-space: nowrap;
        flex-wrap: nowrap;
    }
    /* Prevent any content from causing horizontal page overflow on mobile; pull the whole
       page up — Streamlit's default ~6rem top padding wastes a third of the first screen.
       (stMainBlockContainer is the current testid; .main .block-container kept for older builds.) */
    .main .block-container,
    div[data-testid="stMainBlockContainer"] {
        overflow-x: hidden;
        max-width: 100%;
        padding-top: 2.5rem;
    }
    table {
        display: block;
        overflow-x: auto;
        max-width: 100%;
    }
    div[data-testid="stDataFrame"] {
        overflow-x: auto;
        max-width: 100%;
    }
    div[data-testid="stCaptionContainer"] p {
        word-break: break-word;
        white-space: normal;
    }
    /* Help (?) tooltips: cap the width so a long help string doesn't balloon the
       popover to ~670px, which the positioner then shoves flush against the
       viewport's left edge (no breathing room). A compact box re-anchors under
       the ? icon with normal spacing. */
    div[data-testid="stTooltipContent"] { max-width: 320px; }
    /* Similar tab: lay the two segmented-control toggles (Feature source / Representation) in a
       single content-width row, tightly bound with a gap — instead of st.columns, which splits the
       full width (huge desktop gap) and stacks with large margins on mobile. flex-wrap lets them
       drop to a second line on a narrow phone with only a small row-gap (not Streamlit's tall
       default stack). */
    .st-key-sim_toggle_row {           /* the keyed class sits on the flex stVerticalBlock itself */
        flex-direction: row;
        flex-wrap: wrap;
        gap: 0.4rem 2rem;
        align-items: flex-end;
    }
    .st-key-sim_toggle_row > div[data-testid="stElementContainer"] {
        width: auto;
        flex: 0 0 auto;
    }
    a.cover-link { transition: filter .15s ease, transform .15s ease; cursor: pointer; }
    a.cover-link:hover { filter: brightness(1.12); transform: scale(1.02); }
    /* Results poster grid (_show_results): one CONTINUOUS grid so posters flow across
       page-size boundaries — 5-across on desktop, 3-across on phones. (st.columns(5) per
       set of 5 wrapped into a lopsided 3+2 within every set on phones.) */
    .poster-grid {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 1rem;
    }
    .poster-caption {
        margin-top: 0.25rem;
        font-size: 0.8rem;
        line-height: 1.35;
        opacity: 0.6;               /* match st.caption's faded look, whatever the theme */
        word-break: break-word;
    }
    @media (max-width: 640px) {
        .poster-grid {
            grid-template-columns: repeat(3, 1fr);
            gap: 0.6rem;
        }
    }
    /* About tab: a wide, page-aligned column. The container is width="stretch" so it
       fills the content area; this cap keeps prose lines readable on a wide monitor.
       On a phone the parent is narrower than the cap, so the column stays full-width. */
    .st-key-about { max-width: 1040px; }
    /* About tab tables: render as real tables filling the column — the global
       display:block rule shrink-wraps them, letting the first column hog width
       while the prose columns crush into tall slivers. On phones, keep a
       minimum table width and scroll inside the markdown wrapper instead. */
    .st-key-about div[data-testid="stMarkdownContainer"] { overflow-x: auto; }
    .st-key-about table { display: table; width: 100%; }
    @media (max-width: 640px) {
        /* all About tables are now 2-column (artifacts, tab guide) — they fit a
           phone screen as-is, so just nudge the font down, no min-width scroll */
        .st-key-about table { font-size: 0.8rem; }
        /* shrink the serving-score code snippet so its commented lines need less
           horizontal scroll on a phone; on the wide desktop column it fits at the
           normal code size, so this shrink is mobile-only */
        .st-key-about pre code { font-size: 0.72rem; }
    }
    </style>
""", unsafe_allow_html=True)
# Slim header: h2 instead of st.title's h1 and a single credit line, so the tab bar and
# actual content sit higher on every view (the old block cost ~250px, wrapping on phones).
st.markdown("## Movie Recommender")
art = load_artifacts()

st.markdown(
    "<small>Two-Tower neural network · Built with "
    "<a href='https://grouplens.org/datasets/movielens/32m/' target='_blank'>MovieLens 32M</a>"
    " and <a href='https://pytorch.org' target='_blank'>PyTorch</a> · "
    "Code: <a href='https://github.com/nickgreenquist/Movie-Recommender-System-PyTorch-TwoTower-Model' target='_blank'>GitHub</a></small>",
    unsafe_allow_html=True,
)

# Ask leads: its pre-computed pill boards put posters on screen in one click (and showcase
# the LLM front-end + two-tower together), where Recommend's cold start is an empty form.
ask_tab, recommend_tab, examples_tab, similar_tab, genres_tab, genome_tab, map_tab, about_tab = st.tabs(
    ["Ask", "Recommend", "Examples", "Similar", "Genres", "Genome", "Map", "About"]
)

with recommend_tab:
    tab_recommend(art.rec_models, art.fs, art.all_ids, art.ts_inference,
                  art.posters, art.tmdb_ids)

with ask_tab:
    tab_ask(art, art.posters, art.tmdb_ids)

with examples_tab:
    tab_recommend_examples(art.rec_models, art.fs, art.all_ids, art.ts_inference,
                           art.posters, art.tmdb_ids)

with similar_tab:
    tab_similar(art.fs, art.all_ids, art.sim_spaces, art.posters, art.tmdb_ids)

with genres_tab:
    tab_explore_genres(art.model, art.fs, art.all_ids, art.all_norm_genre,
                       art.posters, art.tmdb_ids)

with genome_tab:
    tab_explore_genome(art.me, art.fs, art.all_ids, art.all_norm_genome,
                       art.posters, art.tmdb_ids)

with map_tab:
    tab_map(art)

with about_tab:
    tab_about()
