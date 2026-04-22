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
    USER_TYPE_TO_FAVORITE_GENRES,
    USER_TYPE_TO_WORST_GENRES,
    USER_TYPE_TO_FAVORITE_MOVIES,
    USER_TYPE_TO_DISLIKED_MOVIES,
    USER_TYPE_TO_GENOME_TAGS,
)
from src.model import MovieRecommender

EXAMPLE_PROFILES = [k for k in USER_TYPE_TO_FAVORITE_GENRES
                    if k not in ('Myself', 'Fantasy Lover', 'War Movie Lover', 'Sci-Fi Lover')]

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
        use_user_genome_pool=cfg.get('use_user_genome_pool', True),
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
    Build a combined user embedding from title and genre signals.
    liked_titles_with_weights: list of (title, weight) tuples.
      Explicit liked movies use _LIKED_MOVIE; genome anchors use _ANCHOR_MOVIE.
    disliked_titles: flat list of title strings (always weighted _DISLIKED_MOVIE).
    """
    n_genres = len(fs['genres_ordered'])
    ctx = [0.0] * (2 * n_genres)

    # Derive genre context from liked/disliked movies — ensures ctx is never
    # all-zeros when the user picks movies but skips the genre selectors.
    all_titles_with_weights = (
        list(liked_titles_with_weights) +
        [(t, _DISLIKED_MOVIE) for t in disliked_titles]
    )
    genre_rating_sum   = {}
    genre_movie_count  = {}
    total_movies       = 0
    for t, w in all_titles_with_weights:
        mid = fs['title_to_movieId'].get(t)
        if mid is None:
            continue
        total_movies += 1
        for g in fs['movieId_to_genres'].get(mid, []):
            genre_rating_sum[g]  = genre_rating_sum.get(g, 0.0)  + w
            genre_movie_count[g] = genre_movie_count.get(g, 0)   + 1

    for g, rsum in genre_rating_sum.items():
        avg_r = rsum / genre_movie_count[g]
        frac  = genre_movie_count[g] / max(total_movies, 1)
        if g in fs['user_context_genre_avg_rating_to_i']:
            ctx[fs['user_context_genre_avg_rating_to_i'][g]]   = avg_r
        if g in fs['user_context_genre_watch_count_to_i']:
            ctx[fs['user_context_genre_watch_count_to_i'][g]]  = frac

    # Explicit genre selections override — stronger signal than movie-derived estimates
    for g in liked_genres:
        if g in fs['user_context_genre_avg_rating_to_i']:
            ctx[fs['user_context_genre_avg_rating_to_i'][g]] = _LIKED_GENRE
        if g in fs['user_context_genre_watch_count_to_i']:
            ctx[fs['user_context_genre_watch_count_to_i'][g]] = 1.0 / max(len(liked_genres), 1)
    for g in disliked_genres:
        if g in fs['user_context_genre_avg_rating_to_i']:
            ctx[fs['user_context_genre_avg_rating_to_i'][g]] = _DISLIKED_GENRE

    liked_hist = [
        (fs['item_emb_movieId_to_i'][fs['title_to_movieId'][t]], w)
        for t, w in liked_titles_with_weights
        if t in fs['title_to_movieId'] and fs['title_to_movieId'][t] in fs['item_emb_movieId_to_i']
    ]
    dis_hist = [
        (fs['item_emb_movieId_to_i'][fs['title_to_movieId'][t]], _DISLIKED_MOVIE)
        for t in disliked_titles
        if t in fs['title_to_movieId'] and fs['title_to_movieId'][t] in fs['item_emb_movieId_to_i']
    ]
    history = liked_hist + dis_hist
    ratings = [h[1] for h in liked_hist] + [_DISLIKED_MOVIE] * len(dis_hist)

    if history:
        hist_ids    = torch.tensor([h[0] for h in history], dtype=torch.long).unsqueeze(0)
        hist_wts    = torch.tensor([ratings], dtype=torch.float)
        hist_embs   = model.item_embedding_lookup(hist_ids)
        wt_sum      = hist_wts.unsqueeze(-1).abs().sum(dim=1).clamp(min=1e-6)
        history_emb = (hist_embs * hist_wts.unsqueeze(-1)).sum(dim=1) / wt_sum
    else:
        history_emb = torch.zeros(1, model.item_embedding_lookup.embedding_dim)

    genre_emb = model.user_genre_tower(torch.tensor([ctx]))
    ts_emb    = model.timestamp_embedding_tower(model.timestamp_embedding_lookup(ts_inference))

    if model.use_user_genome_pool:
        # Genome pooling — mirrors history pooling in content space (shared tower)
        genome_contexts = []
        genome_weights  = []
        for t, w in list(liked_titles_with_weights) + [(t, _DISLIKED_MOVIE) for t in disliked_titles]:
            mid = fs['title_to_movieId'].get(t)
            if mid and mid in fs['movieId_to_genome_tag_context']:
                genome_contexts.append(fs['movieId_to_genome_tag_context'][mid])
                genome_weights.append(w)

        if genome_contexts:
            gc_tensor  = torch.tensor(np.array(genome_contexts, dtype=np.float32))
            ge_embs    = model.item_genome_tag_tower(gc_tensor)
            wts        = torch.tensor(genome_weights)
            wt_sum_g   = wts.abs().sum().clamp(min=1e-6)
            genome_emb = (ge_embs * wts.unsqueeze(-1)).sum(dim=0, keepdim=True) / wt_sum_g
        else:
            genome_emb = torch.zeros(1, model.item_genome_tag_tower[0].out_features)

        return torch.cat([history_emb, genome_emb, genre_emb, ts_emb], dim=1)
    else:
        return torch.cat([history_emb, genre_emb, ts_emb], dim=1)


def _build_genome_i_to_name(fs):
    return {fs['genome_tag_to_i'][tid]: fs['genome_tag_names'][tid] for tid in fs['genome_tag_to_i']}


def _top_genome_tags(mid, fs, i_to_name, n=5):
    ctx     = fs['movieId_to_genome_tag_context'][mid]
    top_idx = sorted(range(len(ctx)), key=lambda i: -ctx[i])[:n]
    return ', '.join(i_to_name[i] for i in top_idx)


def _score_movies(user_emb, all_ids, all_embs, fs, exclude_titles, top_n=20):
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


_POSTER_COLS = 5

def _show_results(df, posters, fs):
    """Display recommendation results as a poster grid, falling back to a table if no posters."""
    if not posters:
        st.dataframe(df, use_container_width=True, hide_index=True)
        return

    titles = df['Title'].tolist()
    for row_start in range(0, len(titles), _POSTER_COLS):
        row_titles = titles[row_start:row_start + _POSTER_COLS]
        cols = st.columns(_POSTER_COLS)
        for col, title in zip(cols, row_titles):
            mid = fs['title_to_movieId'].get(title)
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
            return
        anchor_tag_title_pairs = []
        if selected_genome_tags:
            name_to_idx = {
                fs['genome_tag_names'][tid]: fs['genome_tag_to_i'][tid]
                for tid in fs['genome_tag_to_i']
            }
            seen_titles = set()
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
        if anchor_tag_title_pairs:
            st.caption("Genome anchors — " + " · ".join(
                f"{tag}: {title}" for tag, title in anchor_tag_title_pairs
            ))
        df = _score_movies(user_emb, all_ids, all_embs, fs,
                           exclude_titles=liked_titles)
        _show_results(df, posters, fs)


# ── Tab: Recommend (Examples) ─────────────────────────────────────────────────

def tab_recommend_examples(model, fs, all_ids, all_embs, ts_inference, posters):
    st.caption("Select a pre-built user profile to see what the model recommends for that taste.")
    selected_profile = st.selectbox(
        "Profile",
        options=[None] + EXAMPLE_PROFILES,
        format_func=lambda x: "Choose a profile..." if x is None else x,
        label_visibility="collapsed",
    )

    if selected_profile:
        fav_movies   = USER_TYPE_TO_FAVORITE_MOVIES[selected_profile]
        dis_movies   = USER_TYPE_TO_DISLIKED_MOVIES[selected_profile]
        fav_genres   = USER_TYPE_TO_FAVORITE_GENRES[selected_profile]
        worst_genres = USER_TYPE_TO_WORST_GENRES[selected_profile]
        genome_tags  = USER_TYPE_TO_GENOME_TAGS.get(selected_profile, [])

        # For genome-driven profiles, compute anchor movies from tags
        anchor_tag_title_pairs = []
        if genome_tags:
            name_to_idx = {
                fs['genome_tag_names'][tid]: fs['genome_tag_to_i'][tid]
                for tid in fs['genome_tag_to_i']
            }
            seen_titles = set()
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

        # Debug: report any fav_movies that didn't resolve to a corpus title
        missing = [t for t in fav_movies if t not in fs['title_to_movieId']]
        if missing:
            st.warning("⚠️ Not found in corpus (check title format): " + ", ".join(missing))

        with torch.no_grad():
            user_emb = _build_user_embedding(
                model, fs, liked_with_weights, dis_movies,
                fav_genres, worst_genres, ts_inference,
            )
        df = _score_movies(user_emb, all_ids, all_embs, fs,
                           exclude_titles=fav_movies + dis_movies +
                                          [t for _, t in anchor_tag_title_pairs])
        st.subheader(f"Recommendations for: {selected_profile}")
        if fav_movies:
            st.caption("Because you like these movies: " + ", ".join(fav_movies))
        if fav_genres:
            st.caption("Because you like these genres: " + ", ".join(fav_genres))
        if genome_tags:
            st.caption("Because you like these genome tags: " + ", ".join(genome_tags))
        _show_results(df, posters, fs)


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
            return
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
                    'Score':           f"{sims[idx].item():.3f}",
                })
                if len(rows) >= 20:
                    break
            st.subheader(f"Similar to: {title}")
            _show_results(pd.DataFrame(rows), posters, fs)


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
            return
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
                'Score':           f'{s:.4f}',
            }
            for mid, s in sorted(sims.items(), key=lambda x: x[1], reverse=True)[:20]
        ]
        _show_results(pd.DataFrame(rows), posters, fs)


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
            return

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
                'Score':           f'{s:.4f}',
            }
            for mid, s in sorted(sims.items(), key=lambda x: x[1], reverse=True)[:20]
        ]
        st.caption(
            "Genome anchors — "
            + " · ".join(f"{tag}: {title}" for tag, _, title in anchor_tag_title_pairs)
        )
        _show_results(pd.DataFrame(rows), posters, fs)


# ── Tab: About ───────────────────────────────────────────────────────────────

def tab_about():
    col, _ = st.columns([1, 1])
    with col:
        st.header("What is this?")
        st.markdown(
            "A PyTorch two-tower neural network trained on the MovieLens 32M dataset. "
            "A dot product of the user and item embeddings predicts a de-biased rating."
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

    st.image('diagram.png')

    col, _ = st.columns([1, 1])
    with col:
        st.header("User Tower")
        st.markdown(
            "Each component encodes a different aspect of taste into a fixed-size vector. "
            "All four are concatenated into a single 110-dim user embedding."
        )
        st.markdown("""
| Component | Input | What it learns |
|---|---|---|
| Rating-Weighted Avg Pool | Watch history — movie IDs weighted by your ratings | Collaborative taste — liked movies pull the user<br>toward similar items in embedding space |
| Rating-Weighted Genome Pool | Genome scores for each watched movie,<br>passed through the shared genome tower | Content texture — the kinds of films you like<br>(atmospheric, cerebral, gritty, etc.) weighted by how much you liked them |
| user_genre_tower | Avg rating per genre + watch fraction per genre | Genre affinity — how strongly you lean toward<br>or away from each of the 20 broad genre categories |
| timestamp_embedding_tower | Month bin of most recent watch activity | Temporal context —<br>captures era-based taste shifts |
""", unsafe_allow_html=True)

        st.header("Item Tower")
        st.markdown(
            "Each movie is encoded from five independent signals into a single 110-dim item embedding."
        )
        st.markdown("""
| Component | Input | What it learns |
|---|---|---|
| item_embedding_tower | Movie ID (shared lookup with user history pool) | Collaborative identity — a learned fingerprint<br>for each movie based on who watches it together |
| item_genome_tag_tower | 1,128 ML-derived relevance scores<br>(shared tower with user genome pool) | Content texture — the film's vibe, themes,<br>and tone in a dense semantic space |
| item_genre_tower | 20-dim genre one-hot vector | Broad genre positioning |
| item_tag_tower | 306 user-applied tag counts (normalized) | Crowd-sourced descriptors —<br>how the community collectively labels the film |
| year_embedding_tower | Release year | Era — captures stylistic and<br>cultural shifts across decades |
""", unsafe_allow_html=True)

        st.header("Shared Embeddings")
        st.markdown(
            "Two components are shared between the user and item towers — same weights, same embedding space:"
        )
        st.markdown("""
**item_embedding_lookup** — The same embedding table is used for the target movie's ID
*and* for each movie in the user's watch history pool.

This forces the user's history representation and the item's identity into the same
space: a movie you liked pulls your user embedding directly toward that movie's embedding.

**item_genome_tag_tower** — The same linear layer processes genome scores for the target
movie *and* for each watched movie in the user's genome pool.

This ensures your content taste vector and the item's content vector are directly
comparable via dot product.
""")

        st.header("Training")
        st.markdown("""
- **Dataset:** MovieLens 32M — ~33M ratings from ~200K users across ~86K movies
- **Corpus:** filtered to movies with 200+ ratings (~9,375 movies) and users with 20–500 ratings
- **Loss:** MSE on de-biased ratings (raw rating minus user mean)
- **Optimizer:** SGD, lr=0.005, momentum=0.9, batch size 64
- **Steps:** 150,000
- **Split:** user-based 90/10 — 90% of users are exclusively in train, 10% exclusively
  in val; no user appears in both
- **History context:** Within each user, ratings are sorted by timestamp. The earliest 90% form the watch history context.
- **Labels:** The latest 10% are the prediction labels — the model never uses future watches as context when predicting earlier ones
""")

        st.header("Why MSE and not Softmax?")
        st.markdown(
            "In-batch negatives softmax (the YouTube DNN approach) works well on user-driven datasets like Goodreads, "
            "where interactions reflect genuine independent preference."
        )
        st.markdown(
            "MovieLens is different — users rate movies they were *shown* by the platform's own recommender, "
            "so the interaction data has a strong popularity-driven structure baked in."
        )
        st.markdown(
            "Softmax suppresses popular items (they appear as frequent in-batch negatives), "
            "but on MovieLens popular movies are genuinely what most users are watching. "
            "MSE on explicit ratings avoids this — it learns directly from how much each user liked each film."
        )
        st.markdown("""
| Metric | **MSE (this model)** | Softmax |
|---|---|---|
| Hit Rate@1 | **1.12%** | 0.44% |
| Hit Rate@5 | **4.86%** | 1.82% |
| Hit Rate@10 | **8.36%** | 3.54% |
| Hit Rate@20 | **13.60%** | 6.32% |
| Hit Rate@50 | **24.54%** | 12.64% |
| Recall@10 | **0.0140** | 0.0057 |
| NDCG@10 | **0.0133** | 0.0056 |
| MRR | **0.0381** | 0.0183 |

*Evaluated on 5,000 held-out users, leave-one-out protocol. Random Hit Rate@50 baseline: 0.53%.*
""")

        st.header("Limitations")
        st.markdown("""
- No user ID — personalization is limited to the signals you provide in the app
- 9,375-movie corpus — films with fewer than 200 ratings are not included
- War and Sci-Fi genres can bleed into prestige/arthouse due to overlapping genome signals
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
