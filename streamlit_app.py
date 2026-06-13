"""
Movie Recommender — Streamlit app.

Run locally:  streamlit run app.py
Requires:     serving/model.pth
              serving/movie_embeddings.pt
              serving/feature_store.pt
              serving/posters.json   (optional — fetch with: TMDB_API_KEY=... python main.py posters)

Generate serving/ with: python main.py export
"""
import importlib
import json
import os
import urllib.parse
from typing import NamedTuple

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F

import src.evaluate
importlib.reload(src.evaluate)
from src.evaluate import (
    USER_TYPE_TO_FAVORITE_MOVIES,
    USER_TYPE_TO_DISLIKED_MOVIES,
    USER_TYPE_TO_GENOME_TAGS,
)
from src.inference import build_user_embedding
from src.model import MovieRecommender

EXAMPLE_PROFILES = list(USER_TYPE_TO_FAVORITE_MOVIES.keys())

# Rating values — mirror evaluate.py canary constants
_LIKED_MOVIE    =  2.0
_DISLIKED_MOVIE = -2.0
_LIKED_GENRE    =  4.0
_DISLIKED_GENRE = -2.0
# Genome anchor movies are synthetic/implicit — we use more of them (5 per tag) to push
# the model toward the selected tags, but each one carries half the weight of an explicitly
# chosen favorite movie. A user saying "this is my favorite" is a stronger signal than
# an anchor we added under the hood.
_ANCHOR_MOVIE   =  1.0
_ANCHORS_PER_TAG = 5

_POSTER_COLS   = 5      # poster grid columns
_PAGE_SIZE     = 20     # movies per page
_TOTAL_RESULTS = 60     # total movies to fetch (3 pages)

# Recommend / Examples popularity-correction A/B — the two separately-trained models the visitor
# switches between. Prod bakes the Menon α=0.5 logit adjustment into training; the α=0 twin is a
# distinct model trained with NO correction, surfaced so the visitor can watch popular titles take
# over once the correction is removed. The correction is train-time only (inference is plain dot
# products either way), so "Off" loads a different model — it is not a runtime knob. These labels
# double as the segmented-control options; the first is the default (prod).
_REC_MODEL_DEFAULT_LABEL  = 'On (α = 0.5)'
_REC_MODEL_NO_ALPHA_LABEL = 'Off (α = 0)'


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
    posters:         dict
    tmdb_ids:        dict


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


def _build_serving_model(fs, cfg):
    """Reconstruct the serving MovieRecommender from the self-contained serving/ artifacts.

    Mirrors src.train.build_model's MovieRecommender call, but sources every non-persistent
    buffer from serving/ rather than the (gitignored, Streamlit-Cloud-absent) data/ dir: the
    genome buffer is restacked from movieId_to_genome_tag_context, and the LLM buffer is read
    from llm_feature_buffer baked into feature_store.pt at export time. We can't call build_model
    directly here precisely because it loads data/llm_features_*.pt. feature_towers
    ('genome' | 'llm' | 'both' | None) — from the exported model_config — selects which
    semantic-feature towers are built.
    """
    feature_towers = cfg.get('feature_towers', cfg.get('content_feature_source', 'genome'))
    has_genome = feature_towers in ('genome', 'both')
    has_llm    = feature_towers in ('llm', 'both')

    top_movies = fs['top_movies']

    genome_context_buffer = None
    genome_tags_len       = len(fs['genome_tag_ids'])
    if has_genome:
        genome_matrix = np.array(
            [fs['movieId_to_genome_tag_context'][mid] for mid in top_movies], dtype=np.float32)
        pad_row = np.zeros((1, genome_matrix.shape[1]), dtype=np.float32)
        genome_context_buffer = torch.from_numpy(np.vstack([genome_matrix, pad_row]))
        genome_tags_len = genome_matrix.shape[1]

    llm_feature_buffer = None
    llm_feature_len    = None
    if has_llm:
        llm_feature_buffer = fs['llm_feature_buffer']
        llm_feature_len    = llm_feature_buffer.shape[1]

    return MovieRecommender(
        genres_len=len(fs['genres_ordered']),
        tags_len=len(fs['tags_ordered']),
        genome_tags_len=genome_tags_len,
        top_movies_len=len(top_movies),
        all_years_len=len(fs['years_ordered']),
        timestamp_num_bins=fs['timestamp_num_bins'],
        user_context_size=fs['user_context_size'],
        feature_towers=feature_towers,
        genome_context_buffer=genome_context_buffer,
        llm_feature_buffer=llm_feature_buffer,
        llm_feature_len=llm_feature_len,
        item_genre_embedding_size=cfg['item_genre_embedding_size'],
        item_tag_embedding_size=cfg['item_tag_embedding_size'],
        item_genome_embedding_size=cfg.get('item_genome_embedding_size', 32),
        item_llm_embedding_size=cfg.get('item_llm_embedding_size', 32),
        item_movieId_embedding_size=cfg['item_movieId_embedding_size'],
        item_year_embedding_size=cfg['item_year_embedding_size'],
        user_genre_embedding_size=cfg['user_genre_embedding_size'],
        timestamp_feature_embedding_size=cfg['timestamp_feature_embedding_size'],
        user_genome_embedding_size=cfg.get('user_genome_embedding_size', 32),
        user_llm_embedding_size=cfg.get('user_llm_embedding_size', 32),
        proj_hidden=cfg.get('proj_hidden', 256),
        output_dim=cfg.get('output_dim', 128),
    )


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

    model = _build_serving_model(fs, cfg)
    state_dict = torch.load('serving/model.pth', weights_only=True)
    # The semantic-feature/context buffers are non-persistent (rebuilt in _build_serving_model);
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
        na_model = _build_serving_model(fs, cfg)
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

    poster_path = 'serving/posters.json'
    if os.path.exists(poster_path):
        with open(poster_path) as f:
            posters = json.load(f)
    else:
        posters = {}

    tmdb_ids = _load_tmdb_ids(fs)

    return Artifacts(
        model=model, fs=fs, me=me, all_ids=all_ids, all_embs=all_embs, rec_models=rec_models,
        all_norm_genre=all_norm_genre, all_norm_genome=all_norm_genome, sim_spaces=sim_spaces,
        ts_inference=ts_inference, posters=posters, tmdb_ids=tmdb_ids,
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
        for row_start in range(0, len(titles), _POSTER_COLS):
            row_titles = titles[row_start:row_start + _POSTER_COLS]
            cols = st.columns(_POSTER_COLS)
            for col, title in zip(cols, row_titles):
                clean_title = title.replace('  ◀ ANCHOR', '').replace('  ◀ anchor', '')
                mid  = fs['title_to_movieId'].get(clean_title)
                url  = (posters.get(str(mid)) or '') if mid else ''
                link = _tmdb_url(mid, tmdb_ids, clean_title)
                with col:
                    st.html(_poster_div(url, link))
                    st.caption(title)

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


# ── Tab: Recommend ────────────────────────────────────────────────────────────

def tab_recommend(rec_models, fs, all_ids, ts_inference, posters, tmdb_ids):
    st.caption(
        "Select movies you love and optionally refine with genome tags. "
        "The model builds your taste embedding from the movies' content — curated genome tags plus "
        "web-scraped, LLM-extracted features — and scores every movie in the corpus. "
        "To learn how the model works, see the About tab."
    )
    all_titles = fs['popularity_ordered_titles']

    # Flag handling must happen before widgets are instantiated
    if st.session_state.pop('_clear_rec', False):
        for key in ('rec_liked', 'rec_genome_tags'):
            st.session_state[key] = []
        for key in ('rec_df', 'rec_shown', 'rec_anchor_caption', 'rec_query', 'rec_seed_key',
                    'rec_alpha'):
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
        div[data-testid="stButton"] > button[kind="secondary"] {
            display: block;
            margin: 1rem auto;
            padding: 0.75rem 3rem;
            font-size: 1.2rem;
            font-weight: 600;
        }
        </style>
    """, unsafe_allow_html=True)
    _, btn_col, clear_col = st.columns([2, 1, 2])
    if clear_col.button("Clear All", use_container_width=False):
        st.session_state['_clear_rec'] = True
        st.rerun()
    if btn_col.button("Get Recommendations", use_container_width=True):
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
            st.session_state['rec_query'] = {
                'liked_with_weights': (
                    [(t, _LIKED_MOVIE)  for t in liked_titles] +
                    [(t, _ANCHOR_MOVIE) for _, t in anchor_tag_title_pairs]
                ),
                'exclude_titles': liked_titles,
                'anchor_caption': (
                    "Genome anchors — " + " · ".join(
                        f"{tag}: {title}" for tag, title in anchor_tag_title_pairs)
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
            if query['anchor_caption']:
                st.session_state['rec_anchor_caption'] = query['anchor_caption']
            else:
                st.session_state.pop('rec_anchor_caption', None)
            st.session_state['rec_seed_key'] = cache_key

    if 'rec_df' in st.session_state:
        _render_rec_model_toggle(rec_models, 'rec_alpha')
        caption = st.session_state.get('rec_anchor_caption')
        if caption:
            st.caption(caption)
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
            st.session_state['examples_anchor_caption'] = "Genome anchors: " + ", ".join(
                title for _, title in anchor_tag_title_pairs
            )
        else:
            st.session_state.pop('examples_anchor_caption', None)
        st.session_state['examples_profile'] = (selected_profile, variant)

    st.subheader(f"Recommendations for: {selected_profile}")
    if fav_movies:
        st.caption("Because you like these movies: " + ", ".join(fav_movies))
    if genome_tags:
        st.caption("Because you like these genome tags: " + ", ".join(genome_tags))
    anchor_caption = st.session_state.get('examples_anchor_caption')
    if anchor_caption:
        st.caption(anchor_caption)
    _show_results('examples', posters, fs, tmdb_ids)


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

        st.header("What each tab demonstrates")
        st.markdown("""
| Tab | What it shows |
|---|---|
| **Recommend** | Build a taste vector from your *own* picks (optionally nudged with genome tags) → ranked poster grid. Cold-start-free serving in action. |
| **Examples** | Pre-built taste personas (Sci-Fi, Horror, Heist, …) to see the model's range without typing anything. |
| **Similar** | Nearest neighbors of any movie in the 128-dim space — with a learned-embedding vs. raw-feature toggle, over genome or LLM content. |
| **Genres** / **Genome** | Probe what the *item tower itself* encodes — query its genre and genome-tag sub-spaces directly. |
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

st.set_page_config(page_title="Movie Recommender", layout="wide")
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
    /* Prevent any content from causing horizontal page overflow on mobile */
    .main .block-container {
        overflow-x: hidden;
        max-width: 100%;
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
st.title("Movie Recommender")
art = load_artifacts()

st.markdown(
    "<small>Two-Tower neural network · Built with "
    "<a href='https://grouplens.org/datasets/movielens/32m/' target='_blank'>MovieLens 32M</a>"
    " and <a href='https://pytorch.org' target='_blank'>PyTorch</a><br>"
    "Code: <a href='https://github.com/nickgreenquist/Movie-Recommender-System-PyTorch-TwoTower-Model' target='_blank'>GitHub</a></small>",
    unsafe_allow_html=True,
)

recommend_tab, examples_tab, similar_tab, genres_tab, genome_tab, about_tab = st.tabs(
    ["Recommend", "Examples", "Similar", "Genres", "Genome", "About"]
)

with recommend_tab:
    tab_recommend(art.rec_models, art.fs, art.all_ids, art.ts_inference,
                  art.posters, art.tmdb_ids)

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

with about_tab:
    tab_about()
