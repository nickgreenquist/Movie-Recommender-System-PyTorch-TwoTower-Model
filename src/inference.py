"""
Shared inference path: build a user embedding from liked/disliked titles.

One source of truth for the user tower across the canary eval (src/evaluate.py) and the
Streamlit app (streamlit_app.py). Both feed model.user_embedding() identically, so a
checkpoint scores a given taste profile the same everywhere — previously this logic was
copy-pasted in both places and kept in sync by hand.

FeatureStore access is abstracted because callers pass either the dataclass (attribute
access, src.dataset.FeatureStore) or the exported serving dict (item access, built in
src/export.py). Both expose the same field names.
"""
import torch


# Explicit-genre override weights (Streamlit only — canary passes no genres, so these are
# inert there).
LIKED_GENRE_VALUE    =  4.0
DISLIKED_GENRE_VALUE = -2.0


def _fs_get(fs, key):
    """Read a FeatureStore field from either the dataclass (attr) or the serving dict (item)."""
    return fs[key] if isinstance(fs, dict) else getattr(fs, key)


def build_user_embedding(model, fs, liked_titles_with_weights, disliked_titles,
                         ts_inference, liked_genres=(), disliked_genres=(),
                         disliked_movie_value=-2.0):
    """
    Build the combined user embedding via model.user_embedding(). Returns (1, output_dim).

    liked_titles_with_weights: list[(title, weight)] — explicit likes plus genome anchors.
    disliked_titles:           list[title] — each weighted disliked_movie_value.
    liked_genres/disliked_genres: optional explicit genre overrides (empty for canary, so
        canary output is unchanged).
    Tensors are placed on the model's device; for the CPU serving model this is a no-op.
    """
    genres_ordered        = _fs_get(fs, 'genres_ordered')
    title_to_movieId      = _fs_get(fs, 'title_to_movieId')
    movieId_to_genres     = _fs_get(fs, 'movieId_to_genres')
    avg_rating_to_i       = _fs_get(fs, 'user_context_genre_avg_rating_to_i')
    watch_count_to_i      = _fs_get(fs, 'user_context_genre_watch_count_to_i')
    item_emb_movieId_to_i = _fs_get(fs, 'item_emb_movieId_to_i')

    # ── Genre context ────────────────────────────────────────────────────────
    n_genres = len(genres_ordered)
    ctx = [0.0] * (2 * n_genres)

    all_titles_with_weights = (
        list(liked_titles_with_weights) +
        [(t, disliked_movie_value) for t in disliked_titles]
    )
    genre_rating_sum  = {}
    genre_movie_count = {}
    for t, w in all_titles_with_weights:
        mid = title_to_movieId.get(t)
        if mid is None:
            continue
        for g in movieId_to_genres.get(mid, []):
            genre_rating_sum[g]  = genre_rating_sum.get(g, 0.0) + w
            genre_movie_count[g] = genre_movie_count.get(g, 0)  + 1

    total_assign = sum(genre_movie_count.values())  # total genre-movie pairs, matches training
    for g, rsum in genre_rating_sum.items():
        avg_r = rsum / genre_movie_count[g]
        frac  = genre_movie_count[g] / max(total_assign, 1)
        if g in avg_rating_to_i:
            ctx[avg_rating_to_i[g]]  = avg_r
        if g in watch_count_to_i:
            ctx[watch_count_to_i[g]] = frac

    # Explicit genre selections override — stronger signal than movie-derived estimates
    for g in liked_genres:
        if g in avg_rating_to_i:
            ctx[avg_rating_to_i[g]]  = LIKED_GENRE_VALUE
        if g in watch_count_to_i:
            ctx[watch_count_to_i[g]] = 1.0 / max(len(liked_genres), 1)
    for g in disliked_genres:
        if g in avg_rating_to_i:
            ctx[avg_rating_to_i[g]]  = DISLIKED_GENRE_VALUE

    # ── Watch history ────────────────────────────────────────────────────────
    liked_hist = [
        (item_emb_movieId_to_i[title_to_movieId[t]], w)
        for t, w in liked_titles_with_weights
        if t in title_to_movieId and title_to_movieId[t] in item_emb_movieId_to_i
    ]
    dis_hist = [
        (item_emb_movieId_to_i[title_to_movieId[t]], disliked_movie_value)
        for t in disliked_titles
        if t in title_to_movieId and title_to_movieId[t] in item_emb_movieId_to_i
    ]
    history = liked_hist + dis_hist
    ratings = [h[1] for h in liked_hist] + [disliked_movie_value] * len(dis_hist)

    device = next(model.parameters()).device
    if history:
        liked_ids    = [h[0] for h in liked_hist]
        disliked_ids = [h[0] for h in dis_hist]
        hist_ids   = torch.tensor([[h[0] for h in history]], dtype=torch.long).to(device)
        hist_wts   = torch.tensor([ratings],                 dtype=torch.float).to(device)
        liked_t    = torch.tensor([liked_ids],    dtype=torch.long).to(device) if liked_ids    else torch.tensor([[model.pad_idx]], dtype=torch.long).to(device)
        disliked_t = torch.tensor([disliked_ids], dtype=torch.long).to(device) if disliked_ids else torch.tensor([[model.pad_idx]], dtype=torch.long).to(device)
    else:
        hist_ids   = torch.tensor([[model.pad_idx]], dtype=torch.long).to(device)
        hist_wts   = torch.tensor([[0.0]], dtype=torch.float).to(device)
        liked_t    = torch.tensor([[model.pad_idx]], dtype=torch.long).to(device)
        disliked_t = torch.tensor([[model.pad_idx]], dtype=torch.long).to(device)

    X_inf = torch.tensor([ctx]).to(device)
    return model.user_embedding(X_inf, hist_ids, liked_t, disliked_t, hist_wts, ts_inference)
