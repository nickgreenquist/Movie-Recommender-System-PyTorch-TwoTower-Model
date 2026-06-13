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
    model:           MovieRecommender
    fs:              dict
    me:              dict
    all_ids:         list
    all_embs:        torch.Tensor
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
        model=model, fs=fs, me=me, all_ids=all_ids, all_embs=all_embs,
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


# ── Tab: Recommend ────────────────────────────────────────────────────────────

def tab_recommend(model, fs, all_ids, all_embs, ts_inference, posters, tmdb_ids):
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
        for key in ('rec_df', 'rec_shown', 'rec_anchor_caption'):
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
            liked_with_weights = (
                [(t, _LIKED_MOVIE)  for t in liked_titles] +
                [(t, _ANCHOR_MOVIE) for _, t in anchor_tag_title_pairs]
            )
            with torch.no_grad():
                user_emb = _build_user_embedding(
                    model, fs, liked_with_weights, [],
                    [], [], ts_inference,
                )
            df = _score_movies(user_emb, all_ids, all_embs, fs,
                               exclude_titles=liked_titles)
            _store_results(df, 'rec')
            if anchor_tag_title_pairs:
                st.session_state['rec_anchor_caption'] = "Genome anchors — " + " · ".join(
                    f"{tag}: {title}" for tag, title in anchor_tag_title_pairs
                )
            else:
                st.session_state.pop('rec_anchor_caption', None)

    if 'rec_df' in st.session_state:
        caption = st.session_state.get('rec_anchor_caption')
        if caption:
            st.caption(caption)
    _show_results('rec', posters, fs, tmdb_ids)


# ── Tab: Recommend (Examples) ─────────────────────────────────────────────────

def tab_recommend_examples(model, fs, all_ids, all_embs, ts_inference, posters, tmdb_ids):
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

    if st.session_state.get('examples_profile') != selected_profile:
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
        st.session_state['examples_profile'] = selected_profile

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
        "Find a movie's nearest neighbours by cosine similarity in a chosen content space. "
        "**Feature source** — curated MovieLens **Genome** tags (1,128 relevance scores) or "
        "**LLM**-extracted features (132 dims). **Representation** — the **learned** item-tower "
        "projection (32-dim, trained end-to-end inside the two-tower model) or the **raw** feature "
        "vectors (direct cosine, a content-based baseline). The learned projection is optimized for "
        "user→item retrieval, so comparing it against raw shows what the model's embedding captures "
        "beyond — and trades off against — literal content."
    )

    selected = st.selectbox(
        "Movie to find similar titles for",
        options=[None] + fs['popularity_ordered_titles'],
        format_func=lambda x: "Choose a movie..." if x is None else x,
        key='sim_title', label_visibility="collapsed",
    )

    # 2×2 toggles, side by side to save vertical space. required + default keep one option always
    # selected (a click switches, never clears). Source shows only when LLM is available;
    # representation is the same set for every source, so precompute it.
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

    if not selected:
        for key in ('sim_seed_key', 'sim_df', 'sim_shown'):
            st.session_state.pop(key, None)
        return

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
    # Single centered readable-width column. width caps at the parent width on
    # narrow screens, so this is full-width on mobile and ~75 chars/line on
    # desktop — unlike the old st.columns([1, 1]) hack, which scaled with
    # monitor width. The outer container centers the fixed-width inner one.
    with st.container(horizontal_alignment="center"), st.container(width=760, key="about"):
        st.header("What is this?")
        st.markdown(
            "A **v3 PyTorch two-tower neural network** trained on the MovieLens 32M dataset. "
            "Both towers output L2-normalized 128-dim embeddings; cosine similarity between them ranks every movie in the corpus."
        )
        st.markdown(
            "The model is trained with **full softmax cross-entropy** (every corpus item as a negative) "
            "and **Menon et al. (2021) logit-adjusted loss** (α=0.5) to correct for popularity bias — "
            "popular movies get a log-count boost during training so their embeddings don't dominate inference. "
            "At inference, raw dot products are used with no post-hoc correction."
        )
        st.markdown(
            "Each movie carries **two independent content fingerprints**, fused in the item tower: the "
            "dataset's curated **genome tags** (1,128 ML-derived relevance scores) and a set of **132 "
            "features I produced myself** — scraped from TMDB + Wikipedia, then scored 0–1 by an LLM across "
            "themes, tone, setting/era, provenance, prestige, and visual medium. The deployed model uses "
            "**both** (`feature_towers='both'`); each source gets its own item-side and user-side sub-tower. "
            "See *The LLM Content Features* below."
        )

        st.subheader("The core design choice: no user ID")
        st.markdown("Most recommender systems embed a unique ID for every user in the training set.")
        st.markdown("This works, but has a fundamental limitation: **inference is only possible for users the model has already seen.**")
        st.markdown("If a new user signs up, you have no embedding for them. Your options are:")
        st.markdown("""
- Retrain the entire model
- Partially fine-tune the new user in with a few gradient steps
- Find an existing user who seems similar and use their embedding as a proxy
""")
        st.markdown("This model takes a different approach. **There is no user ID embedding.**")
        st.markdown("Instead, every user is represented as a function of their taste signals — watch history, genre affinity, content texture, and timestamp.")
        st.markdown("The model learns to embed *features of the user*, not the user themselves.")
        st.markdown("This means the model can generate recommendations for **any user** as long as you can provide even a small amount of signal: a few movies they liked.")
        st.markdown("No retraining required. No cold-start problem at the user level. The same trained model works in production for users who never existed when the model was trained.")

        st.code("""\
User Tower (4-pool):
  sum_pool(item_embedding_lookup[full_history])            →  pool_full      (32)  [LayerNorm]
  sum_pool(item_embedding_lookup[liked_history])           →  pool_liked     (32)  [LayerNorm]
  sum_pool(item_embedding_lookup[disliked_history])        →  pool_disliked  (32)  [LayerNorm]
  rating_weighted_sum(item_embedding_lookup[full_history]) →  pool_weighted  (32)  [LayerNorm]
  user_genome_context_tower(rating_weighted_avg(genome[history]))  →  genome_ctx  (32)
  user_llm_feature_tower(rating_weighted_avg(llm[history]))         →  llm_ctx     (32)
  user_genre_tower([avg_rating_per_genre | watch_frac])    →  genre_emb      (32)
  timestamp_embedding_tower(watch_month)                   →  ts_emb          (4)
  concat (228) → Linear(256) → ReLU → Linear(128) → L2-normalize → user_emb (128)

Item Tower:
  item_embedding_tower(movie_id)        →  item_id_emb      (32)  [shared lookup with all 4 user pools]
  item_genome_tag_tower(genome_scores)  →  item_genome_emb  (32)
  item_llm_feature_tower(llm_features)  →  item_llm_emb     (32)
  item_genre_tower(genre_onehot)        →  item_genre_emb    (8)
  item_tag_tower(tag_vector)            →  item_tag_emb     (16)
  year_embedding_tower(release_year)    →  year_emb          (8)
  concat (128) → Linear(256) → ReLU → Linear(128) → L2-normalize → item_emb (128)

Prediction: dot_product(user_emb, item_emb) = cosine similarity (both L2-normalized)
""", language=None)

        st.header("User Tower")
        st.markdown(
            "Each component encodes a different aspect of taste into a fixed-size vector. "
            "All eight outputs are concatenated into a 228-dim vector, then passed through a "
            "projection MLP (Linear 256 → ReLU → Linear 128) to produce the final 128-dim user embedding."
        )
        st.markdown("""
| Component | Output | Input | What it learns |
|---|---|---|---|
| Full History Sum Pool | 32-dim | All watched movies (unweighted item ID embeddings) | Collaborative signal — every watched film shapes the user space |
| Liked Sum Pool | 32-dim | Movies with positive debiased rating | Positive taste — pulls toward movies you actively liked |
| Disliked Sum Pool | 32-dim | Movies with negative debiased rating | Negative taste — pushes away from movies you disliked |
| Rating-Weighted Sum Pool | 32-dim | All watched movies weighted by debiased rating | Combined sentiment — liked films pull, disliked films push |
| user_genre_tower | 32-dim | Avg rating per genre + watch fraction per genre | Genre affinity — how strongly you lean toward<br>or away from each of the 20 broad genre categories |
| timestamp_embedding_tower | 4-dim | Month bin of most recent watch activity | Temporal context —<br>captures era-based taste shifts |
| Genome Context Tower | 32-dim | Rating-weighted avg of raw 1,128-dim genome scores<br>across all watched movies | Overall content taste fingerprint — a dense summary<br>of which genome tags define your taste profile |
| LLM Feature Context Tower | 32-dim | Rating-weighted avg of the 132-dim LLM feature vector<br>across all watched movies | Self-built content fingerprint — themes, tone, setting,<br>prestige & medium distilled by an LLM from scraped text |
| **Projection MLP** | **128-dim** | concat(228) → Linear(256) → ReLU → Linear(128) | Cross-feature interactions — learns how history,<br>genre, timing, and genome + LLM content taste combine |
""", unsafe_allow_html=True)

        st.header("Item Tower")
        st.markdown(
            "Each movie is encoded from six independent signals. "
            "All six outputs are concatenated into a 128-dim vector, then passed through a "
            "projection MLP (Linear 256 → ReLU → Linear 128) to produce the final 128-dim item embedding."
        )
        st.markdown("""
| Component | Output | Input | What it learns |
|---|---|---|---|
| item_embedding_tower | 32-dim | Movie ID (shared lookup with all four user history pools) | Collaborative identity — a learned fingerprint<br>for each movie based on who watches it together |
| item_genome_tag_tower | 32-dim | 1,128 ML-derived genome relevance scores | Content texture — the film's vibe, themes,<br>and tone in a dense semantic space |
| item_llm_feature_tower | 32-dim | 132 LLM-extracted features (0–1), scored from<br>scraped TMDB + Wikipedia text | Self-built content texture — themes, tone, setting,<br>era, provenance, prestige & visual medium |
| item_genre_tower | 8-dim | 20-dim genre one-hot vector | Broad genre positioning |
| item_tag_tower | 16-dim | 306 user-applied tag counts (normalized) | Crowd-sourced descriptors —<br>how the community collectively labels the film |
| year_embedding_tower | 8-dim | Release year | Era — captures stylistic and<br>cultural shifts across decades |
| **Projection MLP** | **128-dim** | concat(128) → Linear(256) → ReLU → Linear(128) | Cross-feature interactions — learns how identity,<br>genome + LLM content, genre, tags, and era combine |
""", unsafe_allow_html=True)

        st.header("Shared Embeddings")
        st.markdown(
            "One component is shared between the user and item towers — same weights, same embedding space:"
        )
        st.markdown("""
**item_embedding_lookup** — The same 32-dim embedding table is used for the target movie's ID
*and* for all four user history pools (full, liked, disliked, rating-weighted).

This forces all four user history representations and the item's identity into the same
space: a movie you liked pulls your user embedding directly toward that movie's embedding,
while a disliked movie pushes it away.
""")

        st.header("The LLM Content Features")
        st.markdown(
            "The genome tags ship with MovieLens. The second content source is **mine**: for every movie "
            "in the corpus I scraped TMDB + Wikipedia, then had an LLM score **132 features** in [0, 1] "
            "across six groups — **themes & plot** (32), **tone & mood** (26), **setting / era / sub-genre** "
            "(32), **provenance & structure** (24), **reception & prestige** (11), and **visual medium** (7). "
            "Each movie becomes a 132-dim dense vector that feeds its own item-side and user-side sub-tower, "
            "exactly parallel to the genome towers."
        )
        st.markdown(
            "The deployed model fuses **both** sources (`feature_towers='both'`) — a deliberate, "
            "portfolio-motivated choice: one model combining the dataset's curated tags with content "
            "signals I built end-to-end. On the held-out rollback eval it lands **on par** with the "
            "genome-only model — MRR 0.1123 vs 0.1144 over 382,138 rollbacks, with identical deep-tail "
            "MRR (0.0159) — trading a negligible aggregate cost for the fused-feature story."
        )

        st.header("Training")
        st.markdown("""
- **Dataset:** MovieLens 32M — ~33M ratings from ~200K users across ~86K movies
- **Corpus:** filtered to movies with 200+ ratings (~9,375 movies) and users with 20–500 ratings
- **Content features:** two per-movie semantic sources fused in the item tower — 1,128 curated genome relevance scores and 132 self-produced LLM features (scraped from TMDB + Wikipedia, scored 0–1 by an LLM)
- **Loss:** Full softmax cross-entropy over all ~9,375 corpus items, with **Menon et al. (2021) logit-adjusted loss (α=0.5)** for popularity bias correction
- **Optimizer:** Adam, lr=0.001, batch size 512, temperature=0.1
- **Steps:** 150,000
- **Split:** user-based 90/10 — 90% of users are exclusively in train, 10% exclusively in val; no user appears in both
- **Training protocol:** Rollback — for each watch event, context = all prior watches (chronological), target = next watch. Up to 20 examples per user. The model learns to predict what a user will like *next*, at every stage of their watch history.
- **L2 normalization:** both towers output unit-norm vectors; dot product = cosine similarity
- **Inference:** raw dot products, no post-hoc popularity correction needed
""")

        st.header("Popularity Bias Correction")
        st.markdown(
            "Full softmax has a structural popularity bias: popular movies dominate every training batch "
            "as hard negatives, suppressing their embeddings — but they're also frequent positive targets, "
            "pulling their embeddings up. The net effect: popular-item embeddings are pushed toward the "
            "average user, making them appear relevant to everyone."
        )
        st.markdown(
            "The fix is the **logit-adjusted loss** from "
            "[Menon et al., \"Long-tail learning via logit adjustment\" (ICLR 2021)](https://arxiv.org/abs/2007.07314): "
            "add `α · log(interaction_count)` to each item's logit before softmax during training. "
            "Popular items get a free score boost, so the model doesn't need to push their embeddings up "
            "to make them easy positives — their embeddings naturally shrink. "
            "At inference, raw dot products are used with no post-hoc correction needed."
        )
        st.markdown("""
| α | Effect |
|---|---|
| 0.0 | No correction — popular items dominate every recommendation regardless of taste |
| **0.5 (this model)** | **Balanced — genre discrimination sharp, popular items appear only when relevant** |
| 1.0 | Over-corrected — suppresses popular items so hard that obscure/low-quality content surfaces |
""")
        st.markdown("""
| Metric | MSE baseline | **v3 Softmax α=0.5 (4-pool, genome)** |
|---|---|---|
| Hit Rate@1 | 0.43% | **5.99%** |
| Hit Rate@5 | 1.68% | **15.49%** |
| Hit Rate@10 | 2.70% | **22.06%** |
| Hit Rate@20 | 4.26% | **30.44%** |
| Hit Rate@50 | 7.36% | **44.38%** |
| MRR | 0.0133 | **0.1153** |

*Rollback eval protocol: for each held-out user, context = all prior watches, target = next watch.*
""")

        st.header("What We Tried")
        st.markdown(
            "The previous user tower had two pools: a rating-weighted avg pool over item ID embeddings, "
            "and a rating-weighted avg pool over genome tag embeddings for each watched movie. "
            "The genome pool was expensive (passing 50 movies per user through a Linear(1128→32) each step) "
            "and conflated positive and negative watches. We replaced it with four sum pools over the same "
            "32-dim item ID embedding: full history, liked-only, disliked-only, and rating-weighted. "
            "Each pool has its own LayerNorm. The genome context tower (a single pass of the user's "
            "rating-weighted average raw genome vector through Linear(1128→32)) was kept."
        )
        st.markdown("""
| Metric | Previous (avg pool + genome pool) | **v3 4-pool (genome)** |
|---|---|---|
| Hit Rate@1 | 4.14% | **5.99%** |
| Hit Rate@5 | 11.73% | **15.49%** |
| Hit Rate@10 | 17.49% | **22.06%** |
| Hit Rate@20 | 25.00% | **30.44%** |
| Hit Rate@50 | 37.80% | **44.38%** |
| MRR | 0.0878 | **0.1153** |

*Rollback eval protocol. +31% MRR.*
""")
        st.markdown(
            "Separating liked and disliked history into independent pools gives the model explicit positive "
            "and negative taste signals. The full and rating-weighted pools handle the general collaborative "
            "signal. All four share the same item ID embedding table, so they live in the same space as the "
            "item tower — a liked movie directly pulls the user embedding toward that item."
        )

        st.header("Limitations")
        st.markdown("""
- No user ID — personalization is limited to the signals you provide in the app
- 9,375-movie corpus — films with fewer than 200 ratings are not included
- War genre can drift toward prestige drama due to genome overlap with the "acclaimed serious film" cluster
- The timestamp tower is a weak signal in the app — all users receive the most recent timestamp bin
- The LLM content features are estimated by an LLM from scraped text — thinly-documented films get weaker (or all-zero) feature vectors
""")


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
    /* About tab: shrink the architecture diagram so its ~96-char lines fit the
       760px readable column without horizontal scroll (still scrolls on phones) */
    .st-key-about pre code { font-size: 0.72rem; }
    /* About tab tables: render as real tables filling the column — the global
       display:block rule shrink-wraps them, letting the first column hog width
       while the prose columns crush into tall slivers. On phones, keep a
       minimum table width and scroll inside the markdown wrapper instead. */
    .st-key-about div[data-testid="stMarkdownContainer"] { overflow-x: auto; }
    .st-key-about table { display: table; width: 100%; }
    @media (max-width: 640px) {
        .st-key-about table { font-size: 0.8rem; }
        /* only the wide 4-column tower tables need to scroll — the 2/3-column
           alpha and metrics tables fit a phone screen as-is */
        .st-key-about table:has(td:nth-child(4)) { min-width: 560px; }
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
    tab_recommend(art.model, art.fs, art.all_ids, art.all_embs, art.ts_inference,
                  art.posters, art.tmdb_ids)

with examples_tab:
    tab_recommend_examples(art.model, art.fs, art.all_ids, art.all_embs, art.ts_inference,
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
