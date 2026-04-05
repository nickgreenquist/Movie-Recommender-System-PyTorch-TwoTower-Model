"""
Movie Recommender — Streamlit app.

Run locally:  streamlit run app.py
Requires:     serving/model.pth
              serving/movie_embeddings.pt
              serving/feature_store.pt

Generate serving/ with: python main.py export
"""
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F

from src.evaluate import (
    USER_TYPE_TO_FAVORITE_GENRES,
    USER_TYPE_TO_WORST_GENRES,
    USER_TYPE_TO_FAVORITE_MOVIES,
    USER_TYPE_TO_DISLIKED_MOVIES,
)
from src.model import MovieRecommender

EXAMPLE_PROFILES = [k for k in USER_TYPE_TO_FAVORITE_GENRES
                    if k not in ('Myself', 'Fantasy Lover', 'War Movie Lover', 'Sci-Fi Lover')]

# Rating values — mirror evaluate.py canary constants
_LIKED_MOVIE    =  2.0
_DISLIKED_MOVIE = -2.0
_LIKED_GENRE    =  4.0
_DISLIKED_GENRE = -2.0


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

    return model, fs, me, all_ids, all_embs, all_norm, ts_inference


# ── Inference helpers ─────────────────────────────────────────────────────────

def _build_user_embedding(model, fs, liked_titles, disliked_titles,
                           liked_genres, disliked_genres, ts_inference):
    """
    Build a combined user embedding from title and genre signals.
    Direct port of evaluate.py:_build_user_embedding.
    Title → movieId lookup happens here; callers pass raw title strings.
    """
    n_genres = len(fs['genres_ordered'])
    ctx = [0.0] * (2 * n_genres)

    for g in liked_genres:
        if g in fs['user_context_genre_avg_rating_to_i']:
            ctx[fs['user_context_genre_avg_rating_to_i'][g]] = _LIKED_GENRE
        if g in fs['user_context_genre_watch_count_to_i']:
            ctx[fs['user_context_genre_watch_count_to_i'][g]] = 1.0 / max(len(liked_genres), 1)
    for g in disliked_genres:
        if g in fs['user_context_genre_avg_rating_to_i']:
            ctx[fs['user_context_genre_avg_rating_to_i'][g]] = _DISLIKED_GENRE

    liked_hist = [
        (fs['item_emb_movieId_to_i'][fs['title_to_movieId'][t]], _LIKED_MOVIE)
        for t in liked_titles
        if t in fs['title_to_movieId'] and fs['title_to_movieId'][t] in fs['item_emb_movieId_to_i']
    ]
    dis_hist = [
        (fs['item_emb_movieId_to_i'][fs['title_to_movieId'][t]], _DISLIKED_MOVIE)
        for t in disliked_titles
        if t in fs['title_to_movieId'] and fs['title_to_movieId'][t] in fs['item_emb_movieId_to_i']
    ]
    history = liked_hist + dis_hist
    ratings = [_LIKED_MOVIE] * len(liked_hist) + [_DISLIKED_MOVIE] * len(dis_hist)

    if history:
        hist_ids    = torch.tensor([h[0] for h in history], dtype=torch.long).unsqueeze(0)
        hist_wts    = torch.tensor([ratings], dtype=torch.float)
        hist_embs   = model.item_embedding_lookup(hist_ids)
        wt_sum      = hist_wts.unsqueeze(-1).abs().sum(dim=1).clamp(min=1e-6)
        history_emb = (hist_embs * hist_wts.unsqueeze(-1)).sum(dim=1) / wt_sum
    else:
        history_emb = torch.zeros(1, model.item_embedding_lookup.embedding_dim)

    # Genome pooling — mirrors history pooling in content space (shared tower)
    genome_contexts = []
    genome_weights  = []
    for t, w in zip(liked_titles + disliked_titles,
                    [_LIKED_MOVIE] * len(liked_titles) + [_DISLIKED_MOVIE] * len(disliked_titles)):
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

    genre_emb = model.user_genre_tower(torch.tensor([ctx]))
    ts_emb    = model.timestamp_embedding_tower(model.timestamp_embedding_lookup(ts_inference))
    return torch.cat([history_emb, genome_emb, genre_emb, ts_emb], dim=1)


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


# ── Tab: Recommend ────────────────────────────────────────────────────────────

def tab_recommend(model, fs, all_ids, all_embs, ts_inference):
    st.caption(
        "Your movie and genre selections are combined into a user tower embedding that is scored "
        "against every movie in the corpus. The more signal you provide, the sharper the recommendations. "
        "To learn how the model works, see the About tab."
    )
    all_titles = fs['popularity_ordered_titles']
    genres     = fs['genres_ordered']

    # Flag handling must happen before widgets are instantiated
    if st.session_state.pop('_clear_rec', False):
        for key in ('rec_liked', 'rec_disliked', 'rec_liked_genres', 'rec_disliked_genres', 'rec_genome_tags'):
            st.session_state[key] = []

    profile = st.session_state.pop('_load_profile', None)
    if profile:
        st.session_state['rec_liked']           = USER_TYPE_TO_FAVORITE_MOVIES[profile]
        st.session_state['rec_disliked']        = USER_TYPE_TO_DISLIKED_MOVIES[profile]
        st.session_state['rec_liked_genres']    = USER_TYPE_TO_FAVORITE_GENRES[profile]
        st.session_state['rec_disliked_genres'] = USER_TYPE_TO_WORST_GENRES[profile]

    col1, col2 = st.columns(2)
    liked_titles    = col1.multiselect("Favorite Movies 😊", all_titles, key='rec_liked')
    disliked_titles = col2.multiselect("Least Favorite Movies 🤮", all_titles, key='rec_disliked')

    col3, col4 = st.columns(2)
    liked_genres    = col3.multiselect("Favorite Genres 😊", genres, key='rec_liked_genres')
    disliked_genres = col4.multiselect("Least Favorite Genres 🤮", genres, key='rec_disliked_genres')

    with st.expander("Refine by Genome Tags (optional)"):

        st.caption(
            "Select content descriptors — tones, themes, settings, cultural touchstones "
            "(e.g. 'atmospheric', 'cyberpunk', 'world war ii'). "
            "The 3 most representative movies for these tags will be added as implicit likes."
        )
        genome_tag_names     = sorted(fs['genome_tag_names'][tid] for tid in fs['genome_tag_names'])
        selected_genome_tags = st.multiselect("Genome tags", genome_tag_names, key='rec_genome_tags')

    with st.expander("Example Profiles — click one to pre-fill the form"):
        cols_per_row = 3
        for i in range(0, len(EXAMPLE_PROFILES), cols_per_row):
            cols = st.columns(cols_per_row)
            for col, name in zip(cols, EXAMPLE_PROFILES[i:i + cols_per_row]):
                if col.button(name, use_container_width=True):
                    st.session_state['_load_profile'] = name
                    st.rerun()

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
        if not liked_titles and not liked_genres and not selected_genome_tags:
            st.warning("Select at least one liked movie, genre, or genome tag.")
            return
        # Top ANCHORS_PER_TAG movies per selected genome tag, deduplicated across tags.
        ANCHORS_PER_TAG = 3
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
                    if count >= ANCHORS_PER_TAG:
                        break
                    title = fs['movieId_to_title'][mid]
                    if title not in seen_titles:
                        anchor_tag_title_pairs.append((tag, title))
                        seen_titles.add(title)
                        count += 1
        anchor_titles = [title for _, title in anchor_tag_title_pairs]
        with torch.no_grad():
            user_emb = _build_user_embedding(
                model, fs, liked_titles + anchor_titles, disliked_titles,
                liked_genres, disliked_genres, ts_inference,
            )
        if anchor_tag_title_pairs:
            st.caption("Genome anchors — " + " · ".join(
                f"{tag}: {title}" for tag, title in anchor_tag_title_pairs
            ))
        df = _score_movies(user_emb, all_ids, all_embs, fs,
                           exclude_titles=liked_titles + disliked_titles)
        st.dataframe(df, use_container_width=True, hide_index=True)


# ── Tab: Similar ──────────────────────────────────────────────────────────────

def tab_similar(me, fs, all_ids, all_norm):
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
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ── Tab: Explore Genres ───────────────────────────────────────────────────────

def tab_explore_genres(model, me, fs):
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
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ── Tab: Explore Genome Tags ──────────────────────────────────────────────────

def tab_explore_genome(model, me, fs):
    st.subheader("Explore Genome Tag Item Tower Embeddings")
    st.caption(
        "Select genome tags to describe what you're looking for — genres, tones, themes, "
        "settings, time periods, plot elements, or cultural touchstones "
        "(e.g. 'atmospheric', 'cyberpunk', 'world war ii', 'studio ghibli'). "
        "The model anchors on the 3 most representative movies for those tags, "
        "then finds similar movies in the genome embedding space."
    )
    genome_tag_names = sorted(fs['genome_tag_names'][tid] for tid in fs['genome_tag_names'])
    selected_tags    = st.multiselect("Genome tags", genome_tag_names, key='explore_genome')
    if st.button("Explore", key='btn_genome'):
        if not selected_tags:
            st.warning("Select at least one genome tag.")
            return

        ANCHORS_PER_TAG = 3
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
                if count >= ANCHORS_PER_TAG:
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
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ── Tab: About ───────────────────────────────────────────────────────────────

def tab_about():
    st.header("What is this?")
    st.markdown(
        "A PyTorch two-tower neural network trained on the MovieLens 32M dataset. "
        "A dot product of the user and item embeddings predicts a de-biased rating."
    )

    st.subheader("The core design choice: no user ID")
    st.markdown("""
Most recommender systems embed a unique ID for every user in the training set. This works, but has a
fundamental limitation: **inference is only possible for users the model has already seen.** If a new
user signs up, you have no embedding for them. Your options are:

- Retrain the entire model
- Partially fine-tune the new user in with a few gradient steps
- Find an existing user who seems similar and use their embedding as a proxy

This model takes a different approach. **There is no user ID embedding.** Instead, every user is
represented as a function of their taste signals — watch history, genre affinity, content texture,
and timestamp. The model learns to embed *features of the user*, not the user themselves.

This means the model can generate recommendations for **any user** as long as you can provide even a
small amount of signal: a few movies they liked, some genres they prefer. No retraining required.
No cold-start problem at the user level. The same trained model works in production for users who
never existed when the model was trained.
""")

    st.image('diagram.png')

    st.header("User Tower")
    st.markdown(
        "Each component encodes a different aspect of taste into a fixed-size vector. "
        "All four are concatenated into a single 120-dim user embedding."
    )
    st.markdown("""
| Component | Input | What it learns |
|---|---|---|
| Rating-Weighted Avg Pool | Watch history — movie IDs weighted by your ratings | Collaborative taste — liked movies pull the user toward similar items in embedding space |
| Rating-Weighted Genome Pool | Genome scores for each watched movie, passed through the shared genome tower | Content texture of your taste — the kinds of films you like (atmospheric, cerebral, gritty, etc.) weighted by how much you liked them |
| user_genre_tower | Avg rating per genre + watch fraction per genre | Genre affinity — how strongly you lean toward or away from each of the 20 broad genre categories |
| timestamp_embedding_tower | Month bin of most recent watch activity | Temporal context — captures era-based taste shifts |
""")

    st.header("Item Tower")
    st.markdown(
        "Each movie is encoded from five independent signals into a single 120-dim item embedding."
    )
    st.markdown("""
| Component | Input | What it learns |
|---|---|---|
| item_embedding_tower | Movie ID (shared lookup with user history pool) | Collaborative identity — a learned fingerprint for each movie based on who watches it together |
| item_genome_tag_tower | 1,128 ML-derived relevance scores (shared tower with user genome pool) | Content texture — the film's vibe, themes, and tone in a dense semantic space |
| item_genre_tower | 20-dim genre one-hot vector | Broad genre positioning |
| item_tag_tower | 306 user-applied tag counts (normalized) | Crowd-sourced descriptors — how the community collectively labels the film |
| year_embedding_tower | Release year | Era — captures stylistic and cultural shifts across decades |
""")

    st.header("Shared Embeddings")
    st.markdown(
        "Two components are shared between the user and item towers — same weights, same embedding space:"
    )
    st.markdown("""
**item_embedding_lookup** — The same embedding table is used for the target movie's ID *and* for each movie in the user's watch history pool. This forces the user's history representation and the item's identity into the same space: a movie you liked pulls your user embedding directly toward that movie's embedding.

**item_genome_tag_tower** — The same linear layer processes genome scores for the target movie *and* for each watched movie in the user's genome pool. This ensures your content taste vector and the item's content vector are directly comparable via dot product.
""")

    st.header("Training")
    st.markdown("""
- **Dataset:** MovieLens 32M — ~33M ratings from ~200K users across ~86K movies
- **Corpus:** filtered to movies with 1,000+ ratings (~4,461 movies) and users with 20–500 ratings
- **Loss:** MSE on de-biased ratings (raw rating minus user mean)
- **Optimizer:** SGD, lr=0.005, momentum=0.9, batch size 64
- **Steps:** 150,000
- **Split:** user-based 90/10 — 90% of users are exclusively in train, 10% exclusively in val; no user appears in both. Within each user, ratings are sorted by timestamp: the earliest 90% form the watch history context and the latest 10% are the prediction labels, so the model never uses future watches as context when predicting earlier ones
""")

    st.header("Limitations")
    st.markdown("""
- No user ID — personalization is limited to the signals you provide in the app
- 4,461-movie corpus — films with fewer than 1,000 ratings are not included
- War and Sci-Fi genres can bleed into prestige/arthouse due to overlapping genome signals
- The timestamp tower is a weak signal in the app — all users receive the most recent timestamp bin
""")


# ── Layout ────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Movie Recommender", layout="wide")
st.title("Movie Recommender")
model, fs, me, all_ids, all_embs, all_norm, ts_inference = load_artifacts()

st.markdown(
    "<small>Two-Tower neural network · Built with <a href='https://grouplens.org/datasets/movielens/32m/' target='_blank'>MovieLens 32M</a> and <a href='https://pytorch.org' target='_blank'>PyTorch</a></small>",
    unsafe_allow_html=True,
)

recommend_tab, similar_tab, genres_tab, genome_tab, about_tab = st.tabs(
    ["Movie Recommendations for You", "Explore Similar Movies", "Explore Genres", "Explore Genome Tags", "About"]
)

with recommend_tab:
    tab_recommend(model, fs, all_ids, all_embs, ts_inference)

with similar_tab:
    tab_similar(me, fs, all_ids, all_norm)

with genres_tab:
    tab_explore_genres(model, me, fs)

with genome_tab:
    tab_explore_genome(model, me, fs)

with about_tab:
    tab_about()
