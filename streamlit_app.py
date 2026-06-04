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

@st.cache_resource
def load_artifacts():
    fs  = torch.load('serving/feature_store.pt', weights_only=False)
    me  = torch.load('serving/movie_embeddings.pt', weights_only=False)
    cfg = fs['model_config']

    top_movies = fs['top_movies']
    genome_matrix = np.array(
        [fs['movieId_to_genome_tag_context'][mid] for mid in top_movies],
        dtype=np.float32,
    )
    pad_row = np.zeros((1, genome_matrix.shape[1]), dtype=np.float32)
    genome_context_buffer = torch.from_numpy(np.vstack([genome_matrix, pad_row]))

    model = MovieRecommender(
        genres_len=len(fs['genres_ordered']),
        tags_len=len(fs['tags_ordered']),
        genome_tags_len=len(fs['genome_tag_ids']),
        top_movies_len=len(top_movies),
        all_years_len=len(fs['years_ordered']),
        timestamp_num_bins=fs['timestamp_num_bins'],
        user_context_size=fs['user_context_size'],
        genome_context_buffer=genome_context_buffer,
        item_genre_embedding_size=cfg['item_genre_embedding_size'],
        item_tag_embedding_size=cfg['item_tag_embedding_size'],
        item_genome_tag_embedding_size=cfg['item_genome_tag_embedding_size'],
        item_movieId_embedding_size=cfg['item_movieId_embedding_size'],
        item_year_embedding_size=cfg['item_year_embedding_size'],
        user_genre_embedding_size=cfg['user_genre_embedding_size'],
        timestamp_feature_embedding_size=cfg['timestamp_feature_embedding_size'],
        user_genome_context_embedding_size=cfg.get('user_genome_context_embedding_size', 32),
        proj_hidden=cfg.get('proj_hidden', None),
        output_dim=cfg.get('output_dim', 128),
    )
    model.load_state_dict(torch.load('serving/model.pth', weights_only=True))
    model.eval()

    all_ids  = list(me.keys())
    all_embs = torch.cat([me[m]['MOVIE_EMBEDDING_COMBINED'] for m in all_ids], dim=0)
    all_norm = F.normalize(all_embs, dim=1)

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

    return model, fs, me, all_ids, all_embs, all_norm, ts_inference, posters



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


# ── Paginated results grid ────────────────────────────────────────────────────

def _store_results(df, result_key: str) -> None:
    """Save a results DataFrame to session state and reset to page 0."""
    st.session_state[f'{result_key}_df']   = df
    st.session_state[f'{result_key}_page'] = 0


def _show_results(result_key: str, posters, fs) -> None:
    """Render the current page of stored results with Prev / Next navigation."""
    df = st.session_state.get(f'{result_key}_df')
    if df is None or df.empty:
        return

    page        = st.session_state.get(f'{result_key}_page', 0)
    total_pages = max(1, (len(df) + _PAGE_SIZE - 1) // _PAGE_SIZE)
    page_df     = df.iloc[page * _PAGE_SIZE:(page + 1) * _PAGE_SIZE]

    if not posters:
        st.dataframe(page_df, use_container_width=True, hide_index=True)
    else:
        titles = page_df['Title'].tolist()
        for row_start in range(0, len(titles), _POSTER_COLS):
            row_titles = titles[row_start:row_start + _POSTER_COLS]
            cols = st.columns(_POSTER_COLS)
            for col, title in zip(cols, row_titles):
                clean_title = title.replace('  ◀ ANCHOR', '').replace('  ◀ anchor', '')
                mid = fs['title_to_movieId'].get(clean_title)
                url = posters.get(str(mid), '') if mid else ''
                with col:
                    if url:
                        st.image(url, use_container_width=True)
                    else:
                        st.markdown(
                            "<div style='background:#1e1e1e;border-radius:6px;aspect-ratio:2/3;"
                            "display:flex;align-items:center;justify-content:center;"
                            "font-size:2rem;'>🎬</div>",
                            unsafe_allow_html=True,
                        )
                    st.caption(title)

    if total_pages > 1:
        _, prev_col, info_col, next_col, _ = st.columns([3, 1, 1, 1, 3])
        if prev_col.button('← Prev', disabled=(page == 0), key=f'{result_key}_prev'):
            st.session_state[f'{result_key}_page'] = page - 1
            st.rerun()
        info_col.markdown(
            f"<div style='text-align:center;padding-top:0.4rem'>{page + 1} / {total_pages}</div>",
            unsafe_allow_html=True,
        )
        if next_col.button('Next →', disabled=(page >= total_pages - 1), key=f'{result_key}_next'):
            st.session_state[f'{result_key}_page'] = page + 1
            st.rerun()


# ── Tab: Recommend ────────────────────────────────────────────────────────────

def tab_recommend(model, fs, all_ids, all_embs, ts_inference, posters):
    st.caption(
        "Select movies you love and optionally refine with genome tags. "
        "The model builds your taste embedding from the movies' content and scores every movie in the corpus. "
        "To learn how the model works, see the About tab."
    )
    all_titles = fs['popularity_ordered_titles']

    # Flag handling must happen before widgets are instantiated
    if st.session_state.pop('_clear_rec', False):
        for key in ('rec_liked', 'rec_genome_tags'):
            st.session_state[key] = []
        for key in ('rec_df', 'rec_page', 'rec_anchor_caption'):
            st.session_state.pop(key, None)

    profile = st.session_state.pop('_load_profile', None)
    if profile:
        st.session_state['rec_liked'] = USER_TYPE_TO_FAVORITE_MOVIES[profile]

    liked_titles = st.multiselect("Favorite Movies", all_titles, key='rec_liked')

    with st.expander("Refine by Genome Tags (optional)"):
        st.caption(
            "Select content descriptors — tones, themes, settings, cultural touchstones "
            "(e.g. 'atmospheric', 'cyberpunk', 'world war ii'). "
            "The 5 most representative movies for these tags will be added as implicit likes."
        )
        genome_tag_names     = sorted(fs['genome_tag_names'][tid] for tid in fs['genome_tag_names'])
        selected_genome_tags = st.multiselect("Genome tags", genome_tag_names, key='rec_genome_tags')

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
    _show_results('rec', posters, fs)


# ── Tab: Recommend (Examples) ─────────────────────────────────────────────────

def tab_recommend_examples(model, fs, all_ids, all_embs, ts_inference, posters):
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
    _show_results('examples', posters, fs)


# ── Tab: Similar ──────────────────────────────────────────────────────────────

def tab_similar(me, fs, all_ids, all_norm, posters):
    st.caption(
        "Each movie is represented by a single combined embedding — the concatenation of "
        "its genre tower, tag tower, genome tag tower, movieId embedding, and year embedding. "
        "This tab finds the movies whose combined embedding is most similar (by cosine similarity) "
        "to the selected seed movie."
    )
    all_titles  = fs['popularity_ordered_titles']
    selections = st.multiselect("Movie", all_titles, key='sim_title')

    if st.button("Find Similar Movies"):
        if not selections:
            st.warning("Select a movie.")
        else:
            for old_title in st.session_state.get('sim_active_titles', []):
                st.session_state.pop(f'sim_{old_title}_df', None)
                st.session_state.pop(f'sim_{old_title}_page', None)
            active_titles = []
            for title in selections:
                mid = fs['title_to_movieId'].get(title)
                if mid not in me:
                    st.error(f"'{title}' not in corpus.")
                    continue

                with torch.no_grad():
                    seed_norm = F.normalize(me[mid]['MOVIE_EMBEDDING_COMBINED'], dim=1)
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
                _store_results(pd.DataFrame(rows), f'sim_{title}')
                active_titles.append(title)
            st.session_state['sim_active_titles'] = active_titles

    for title in selections:
        if f'sim_{title}_df' in st.session_state:
            st.subheader(f"Similar to: {title}")
            _show_results(f'sim_{title}', posters, fs)


# ── Tab: Explore Genres ───────────────────────────────────────────────────────

def tab_explore_genres(model, me, fs, posters):
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
                query = model.item_genre_tower(torch.tensor([ctx])).view(-1)
            sims = {
                mid: F.cosine_similarity(
                    query.unsqueeze(0),
                    me[mid]['MOVIE_GENRE_EMBEDDING'].view(-1).unsqueeze(0),
                ).item()
                for mid in fs['top_movies']
            }
            i_to_name = _build_genome_i_to_name(fs)
            rows = [
                {
                    'Title':           fs['movieId_to_title'][mid],
                    'Genres':          ', '.join(fs['movieId_to_genres'][mid]),
                    'Top Genome Tags': _top_genome_tags(mid, fs, i_to_name),
                }
                for mid, s in sorted(sims.items(), key=lambda x: x[1], reverse=True)[:_TOTAL_RESULTS]
            ]
            _store_results(pd.DataFrame(rows), 'genres')

    _show_results('genres', posters, fs)


# ── Tab: Explore Genome Tags ──────────────────────────────────────────────────

def tab_explore_genome(model, me, fs, posters):
    st.subheader("Explore Genome Tag Item Tower Embeddings")
    st.caption(
        "Select genome tags to describe what you're looking for — genres, tones, themes, "
        "settings, time periods, plot elements, or cultural touchstones "
        "(e.g. 'atmospheric', 'cyberpunk', 'world war ii', 'studio ghibli'). "
        "The model anchors on the 5 most representative movies for those tags, "
        "then finds similar movies in the genome embedding space."
    )
    genome_tag_names = sorted(fs['genome_tag_names'][tid] for tid in fs['genome_tag_names'])
    selected_tags    = st.multiselect("Genome tags", genome_tag_names, key='explore_genome')
    if st.button("Explore", key='btn_genome'):
        if not selected_tags:
            st.warning("Select at least one genome tag.")
        else:
            name_to_idx = {
                fs['genome_tag_names'][tid]: fs['genome_tag_to_i'][tid]
                for tid in fs['genome_tag_to_i']
            }
            anchor_tag_title_pairs = []
            seen_titles = set()
            for tag in selected_tags:
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
                        anchor_tag_title_pairs.append((tag, mid, title))
                        seen_titles.add(title)
                        count += 1

            if not anchor_tag_title_pairs:
                st.warning("No genome tags matched the vocabulary.")
            else:
                anchor_mids = [mid for _, mid, _ in anchor_tag_title_pairs]
                anchor_set  = set(anchor_mids)
                query_emb   = torch.stack([
                    me[m]['MOVIE_GENOME_TAG_EMBEDDING'].view(-1) for m in anchor_mids
                ]).mean(dim=0)

                sims = {
                    mid: F.cosine_similarity(
                        query_emb.unsqueeze(0),
                        me[mid]['MOVIE_GENOME_TAG_EMBEDDING'].view(-1).unsqueeze(0),
                    ).item()
                    for mid in fs['top_movies']
                }
                i_to_name = _build_genome_i_to_name(fs)
                rows = [
                    {
                        'Title':           fs['movieId_to_title'][mid] + ('  ◀ ANCHOR' if mid in anchor_set else ''),
                        'Genres':          ', '.join(fs['movieId_to_genres'][mid]),
                        'Top Genome Tags': _top_genome_tags(mid, fs, i_to_name),
                    }
                    for mid, s in sorted(sims.items(), key=lambda x: x[1], reverse=True)[:_TOTAL_RESULTS]
                ]
                _store_results(pd.DataFrame(rows), 'genome')
                st.session_state['genome_anchor_caption'] = (
                    "Genome anchors — "
                    + " · ".join(f"{tag}: {title}" for tag, _, title in anchor_tag_title_pairs)
                )

    if 'genome_df' in st.session_state:
        caption = st.session_state.get('genome_anchor_caption')
        if caption:
            st.caption(caption)
    _show_results('genome', posters, fs)


# ── Tab: About ───────────────────────────────────────────────────────────────

def tab_about():
    col, _ = st.columns([1, 1])
    with col:
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
  user_genre_tower([avg_rating_per_genre | watch_frac])    →  genre_emb      (32)
  timestamp_embedding_tower(watch_month)                   →  ts_emb          (4)
  concat (196) → Linear(256) → ReLU → Linear(128) → L2-normalize → user_emb (128)

Item Tower:
  item_embedding_tower(movie_id)        →  item_id_emb      (32)  [shared lookup with all 4 user pools]
  item_genome_tag_tower(genome_scores)  →  item_genome_emb  (32)
  item_genre_tower(genre_onehot)        →  item_genre_emb    (8)
  item_tag_tower(tag_vector)            →  item_tag_emb     (16)
  year_embedding_tower(release_year)    →  year_emb          (8)
  concat (96) → Linear(256) → ReLU → Linear(128) → L2-normalize → item_emb (128)

Prediction: dot_product(user_emb, item_emb) = cosine similarity (both L2-normalized)
""", language=None)

    col, _ = st.columns([1, 1])
    with col:
        st.header("User Tower")
        st.markdown(
            "Each component encodes a different aspect of taste into a fixed-size vector. "
            "All seven outputs are concatenated into a 196-dim vector, then passed through a "
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
| **Projection MLP** | **128-dim** | concat(196) → Linear(256) → ReLU → Linear(128) | Cross-feature interactions — learns how history,<br>content, genre, timing, and genome taste combine |
""", unsafe_allow_html=True)

        st.header("Item Tower")
        st.markdown(
            "Each movie is encoded from five independent signals. "
            "All five outputs are concatenated into a 96-dim vector, then passed through a "
            "projection MLP (Linear 256 → ReLU → Linear 128) to produce the final 128-dim item embedding."
        )
        st.markdown("""
| Component | Output | Input | What it learns |
|---|---|---|---|
| item_embedding_tower | 32-dim | Movie ID (shared lookup with all four user history pools) | Collaborative identity — a learned fingerprint<br>for each movie based on who watches it together |
| item_genome_tag_tower | 32-dim | 1,128 ML-derived relevance scores | Content texture — the film's vibe, themes,<br>and tone in a dense semantic space |
| item_genre_tower | 8-dim | 20-dim genre one-hot vector | Broad genre positioning |
| item_tag_tower | 16-dim | 306 user-applied tag counts (normalized) | Crowd-sourced descriptors —<br>how the community collectively labels the film |
| year_embedding_tower | 8-dim | Release year | Era — captures stylistic and<br>cultural shifts across decades |
| **Projection MLP** | **128-dim** | concat(96) → Linear(256) → ReLU → Linear(128) | Cross-feature interactions — learns how identity,<br>content, genre, tags, and era combine |
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

        st.header("Training")
        st.markdown("""
- **Dataset:** MovieLens 32M — ~33M ratings from ~200K users across ~86K movies
- **Corpus:** filtered to movies with 200+ ratings (~9,375 movies) and users with 20–500 ratings
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
| Metric | MSE baseline | **v3 Softmax α=0.5 (this model)** |
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
| Metric | Previous (avg pool + genome pool) | **This model (4-pool)** |
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
""")


# ── Layout ────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Movie Recommender", layout="wide")
st.markdown("""
    <style>
    div[data-testid="stTabs"] > div:first-child {
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
    </style>
""", unsafe_allow_html=True)
st.title("Movie Recommender")
model, fs, me, all_ids, all_embs, all_norm, ts_inference, posters = load_artifacts()

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
    tab_recommend(model, fs, all_ids, all_embs, ts_inference, posters)

with examples_tab:
    tab_recommend_examples(model, fs, all_ids, all_embs, ts_inference, posters)

with similar_tab:
    tab_similar(me, fs, all_ids, all_norm, posters)

with genres_tab:
    tab_explore_genres(model, me, fs, posters)

with genome_tab:
    tab_explore_genome(model, me, fs, posters)

with about_tab:
    tab_about()
