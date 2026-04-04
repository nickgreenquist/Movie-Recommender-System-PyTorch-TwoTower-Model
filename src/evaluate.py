"""
Inference, canary user evaluation, and embedding probes.

Usage:
    python main.py evaluate
"""
import glob
import os
from itertools import zip_longest

import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.dataset import FeatureStore
from src.model import MovieRecommender
from src.train import build_model, get_config


# ── Canary user definitions ───────────────────────────────────────────────────

USER_TYPE_TO_FAVORITE_GENRES = {
    'Fantasy Lover':          ['Fantasy'],
    "Children's Movie Lover": ['Children'],
    'Horror Lover':           ['Horror'],
    'Sci-Fi Lover':           ['Sci-Fi'],
    'Comedy Lover':           ['Comedy'],
    'Romance Lover':          ['Romance'],
    'War Movie Lover':        ['War'],
    'Thriller Lover':         ['Thriller'],
    'Crime Lover':            ['Crime'],
    'Myself': ['Fantasy', 'War', 'Horror', 'Drama', 'Action'],
}

USER_TYPE_TO_WORST_GENRES = {
    'Fantasy Lover':          ['Horror', 'Children'],
    "Children's Movie Lover": ['Horror', 'Romance', 'Drama', 'Action'],
    'Horror Lover':           ['Children'],
    'Sci-Fi Lover':           ['Romance', 'Children'],
    'Comedy Lover':           ['Children'],
    'Romance Lover':          ['Children', 'Horror'],
    'War Movie Lover':        ['Children'],
    'Thriller Lover':         ['Children', 'Comedy'],
    'Crime Lover':            ['Children', 'Fantasy', 'Romance', 'Comedy'],
    'Myself':                 ['Romance'],
}

USER_TYPE_TO_FAVORITE_MOVIES = {
    'Fantasy Lover': [
        'Lord of the Rings: The Fellowship of the Ring, The (2001)',
        'Princess Bride, The (1987)', 'Willow (1988)', 'Excalibur (1981)',
        'Labyrinth (1986)', 'Legend (1985)', 'Dark Crystal, The (1982)',
        'Fantastic Beasts and Where to Find Them (2016)',
        'Hobbit: An Unexpected Journey, The (2012)',
    ],
    "Children's Movie Lover": [
        'Toy Story 2 (1999)', 'Finding Nemo (2003)', 'Monsters, Inc. (2001)',
        'Toy Story (1995)', 'Lion King, The (1994)', 'Shrek (2001)',
        'Babe (1995)', "Charlotte's Web (1973)", 'Road to El Dorado, The (2000)',
    ],
    'Horror Lover': [
        'Blair Witch Project, The (1999)', 'Silence of the Lambs, The (1991)',
        'Shining, The (1980)', "Rosemary's Baby (1968)", 'Halloween (1978)',
        'Get Out (2017)', 'Psycho (1960)', 'Birds, The (1963)',
        'Poltergeist (1982)', 'Nightmare on Elm Street, A (1984)',
    ],
    'Sci-Fi Lover': [
        'Matrix, The (1999)', 'Terminator, The (1984)', '2001: A Space Odyssey (1968)',
        'Blade Runner (1982)', 'Arrival (2016)', 'Interstellar (2014)',
        'Blade Runner 2049 (2017)', 'Contact (1997)',
        'Ghost in the Shell (Kôkaku kidôtai) (1995)',
        'Terminator 2: Judgment Day (1991)',
        'Star Wars: Episode IV - A New Hope (1977)',
        'Independence Day (a.k.a. ID4) (1996)',
    ],
    'Comedy Lover': [
        'American Pie (1999)', 'Dumb & Dumber (Dumb and Dumber) (1994)',
        'Big Lebowski, The (1998)', 'Monty Python and the Holy Grail (1975)',
        'Airplane! (1980)', 'Spaceballs (1987)', "Ferris Bueller's Day Off (1986)",
        'Clerks (1994)', 'Office Space (1999)',
        "Monty Python's Life of Brian (1979)",
    ],
    'Romance Lover': [
        'Shakespeare in Love (1998)', 'Sense and Sensibility (1995)',
        'When Harry Met Sally... (1989)', 'Sleepless in Seattle (1993)',
        'Notting Hill (1999)', 'Four Weddings and a Funeral (1994)',
        'Annie Hall (1977)', 'Casablanca (1942)', 'Jerry Maguire (1996)',
    ],
    'War Movie Lover': [
        'Saving Private Ryan (1998)', 'Apocalypse Now (1979)',
        'Full Metal Jacket (1987)', 'Platoon (1986)',
        'Bridge on the River Kwai, The (1957)', 'Glory (1989)',
        'Deer Hunter, The (1978)', 'Paths of Glory (1957)',
        'Downfall (Untergang, Der) (2004)', '1917 (2019)',
        'Enemy at the Gates (2001)',
    ],
    'Thriller Lover': [
        'Seven (a.k.a. Se7en) (1995)', 'Memento (2000)',
        'Usual Suspects, The (1995)', 'Zodiac (2007)',
        'Fight Club (1999)', 'Game, The (1997)', 'Vertigo (1958)',
    ],
    'Crime Lover': [
        'Godfather, The (1972)', 'Goodfellas (1990)', 'Reservoir Dogs (1992)',
        'L.A. Confidential (1997)', 'Departed, The (2006)',
        'Scarface (1983)', 'Casino (1995)',
    ],
    'Myself': [
        'Lord of the Rings: The Fellowship of the Ring, The (2001)',
        'Lord of the Rings: The Return of the King, The (2003)',
        '300 (2007)', 'Saving Private Ryan (1998)', 'Kill Bill: Vol. 1 (2003)',
        'Gladiator (2000)', 'Braveheart (1995)',
    ],
}

USER_TYPE_TO_DISLIKED_MOVIES = {
    'Fantasy Lover': [
        'Get Out (2017)', 'Poltergeist (1982)', 'Nightmare on Elm Street, A (1984)',
        'Coco (2017)', 'Iron Giant, The (1999)',
    ],
    "Children's Movie Lover": [
        'Get Out (2017)', 'Nightmare on Elm Street, A (1984)',
        'Casablanca (1942)', 'Jerry Maguire (1996)', "Schindler's List (1993)",
        'Die Hard (1988)', 'Terminator 2: Judgment Day (1991)',
        'Predator (1987)', 'RoboCop (1987)',
        'First Blood (Rambo: First Blood) (1982)',
        'Fast & Furious 6 (Fast and the Furious 6, The) (2013)',
        'Furious 7 (2015)',
    ],
    'Horror Lover': [
        'Coco (2017)', 'My Neighbor Totoro (Tonari no Totoro) (1988)',
        'Iron Giant, The (1999)',
    ],
    'Sci-Fi Lover': [
        'Casablanca (1942)', 'Jerry Maguire (1996)',
        'Coco (2017)', 'My Neighbor Totoro (Tonari no Totoro) (1988)',
    ],
    'Comedy Lover': [
        'Coco (2017)', 'My Neighbor Totoro (Tonari no Totoro) (1988)',
        'Iron Giant, The (1999)',
    ],
    'Romance Lover': [
        'Get Out (2017)', 'Poltergeist (1982)',
        'Coco (2017)', 'My Neighbor Totoro (Tonari no Totoro) (1988)',
    ],
    'War Movie Lover': [
        'Coco (2017)', 'My Neighbor Totoro (Tonari no Totoro) (1988)',
        'Iron Giant, The (1999)',
        'Lord of the Rings: The Fellowship of the Ring, The (2001)',
        'Avengers, The (2012)',
        'Raiders of the Lost Ark (Indiana Jones and the Raiders of the Lost Ark) (1981)',
    ],
    'Thriller Lover': [
        'Finding Nemo (2003)', 'My Neighbor Totoro (Tonari no Totoro) (1988)',
        'Ace Ventura: Pet Detective (1994)', 'Mrs. Doubtfire (1993)',
    ],
    'Crime Lover': [
        'Finding Nemo (2003)', 'My Neighbor Totoro (Tonari no Totoro) (1988)',
        'Lord of the Rings: The Fellowship of the Ring, The (2001)',
        'Casablanca (1942)', 'Pretty Woman (1990)',
    ],
    'Myself': [],
}

VALUE_FAVORITE_GENRE_RATING = 4.0
VALUE_DISLIKED_GENRE_RATING = -2.0
VALUE_FAVORITE_MOVIE_RATING = 2.0
VALUE_DISLIKED_MOVIE_RATING = -2.0


# ── Movie embedding cache ─────────────────────────────────────────────────────

def build_movie_embeddings(model: MovieRecommender, fs: FeatureStore) -> dict:
    """
    Pre-compute all movie combined embeddings for fast recommendation scoring.
    Returns movieId → {'MOVIE_EMBEDDING_COMBINED': Tensor, ...}
    """
    model.eval()
    movieId_to_embedding = {}
    with torch.no_grad():
        for movieId in fs.top_movies:
            mid = int(movieId)
            d   = {}
            emb_idx  = torch.tensor([fs.item_emb_movieId_to_i[mid]])
            year_idx = torch.tensor([fs.year_to_i[fs.movieId_to_year[mid]]])

            d['MOVIEID_EMBEDDING']      = model.item_embedding_tower(
                                            model.item_embedding_lookup(emb_idx))
            d['MOVIE_YEAR_EMBEDDING']   = model.year_embedding_tower(
                                            model.year_embedding_lookup(year_idx))
            d['MOVIE_GENRE_EMBEDDING']  = model.item_genre_tower(
                                            torch.tensor([fs.movieId_to_genre_context[mid]]))

            item_parts = [d['MOVIE_GENRE_EMBEDDING']]
            if model.use_item_tag_tower:
                d['MOVIE_TAG_EMBEDDING'] = model.item_tag_tower(
                    torch.tensor([fs.movieId_to_tag_context[mid]]))
                item_parts.append(d['MOVIE_TAG_EMBEDDING'])
            if model.use_item_genome_tag_tower:
                d['MOVIE_GENOME_TAG_EMBEDDING'] = model.item_genome_tag_tower(
                    torch.tensor([fs.movieId_to_genome_tag_context[mid]]))
                item_parts.append(d['MOVIE_GENOME_TAG_EMBEDDING'])

            item_parts += [d['MOVIEID_EMBEDDING'], d['MOVIE_YEAR_EMBEDDING']]
            d['MOVIE_EMBEDDING_COMBINED'] = torch.cat(item_parts, dim=1)
            movieId_to_embedding[mid] = d

    return movieId_to_embedding


# ── Canary user inference ─────────────────────────────────────────────────────

def _build_user_embedding(model: MovieRecommender, fs: FeatureStore, user_type: str,
                          ts_inference: torch.Tensor) -> torch.Tensor:
    """Build the combined user embedding for a canary user type."""
    fav_genres   = USER_TYPE_TO_FAVORITE_GENRES[user_type]
    worst_genres = USER_TYPE_TO_WORST_GENRES[user_type]
    fav_movies   = USER_TYPE_TO_FAVORITE_MOVIES[user_type]
    dis_movies   = USER_TYPE_TO_DISLIKED_MOVIES[user_type]

    n_genres = len(fs.genres_ordered)

    # Genre context
    ctx = [0.0] * (2 * n_genres)
    for g in fav_genres:
        if g in fs.user_context_genre_avg_rating_to_i:
            ctx[fs.user_context_genre_avg_rating_to_i[g]] = VALUE_FAVORITE_GENRE_RATING
    for g in worst_genres:
        if g in fs.user_context_genre_avg_rating_to_i:
            ctx[fs.user_context_genre_avg_rating_to_i[g]] = VALUE_DISLIKED_GENRE_RATING
    for g in fav_genres:
        if g in fs.user_context_genre_watch_count_to_i:
            ctx[fs.user_context_genre_watch_count_to_i[g]] = 1.0 / len(fav_genres)

    # Watch history
    liked_hist = [
        (fs.item_emb_movieId_to_i[fs.title_to_movieId[t]], VALUE_FAVORITE_MOVIE_RATING)
        for t in fav_movies
        if t in fs.title_to_movieId and fs.title_to_movieId[t] in fs.item_emb_movieId_to_i
    ]
    dis_hist = [
        (fs.item_emb_movieId_to_i[fs.title_to_movieId[t]], VALUE_DISLIKED_MOVIE_RATING)
        for t in dis_movies
        if t in fs.title_to_movieId and fs.title_to_movieId[t] in fs.item_emb_movieId_to_i
    ]
    history = liked_hist + dis_hist
    ratings = ([VALUE_FAVORITE_MOVIE_RATING] * len(liked_hist) +
               [VALUE_DISLIKED_MOVIE_RATING] * len(dis_hist))

    if history:
        hist_ids  = torch.tensor([h[0] for h in history], dtype=torch.long).unsqueeze(0)
        hist_wts  = torch.tensor([ratings], dtype=torch.float)
        hist_embs = model.item_embedding_lookup(hist_ids)
        wt_sum    = hist_wts.unsqueeze(-1).abs().sum(dim=1).clamp(min=1e-6)
        history_emb = (hist_embs * hist_wts.unsqueeze(-1)).sum(dim=1) / wt_sum
    else:
        history_emb = torch.zeros(1, model.item_embedding_lookup.embedding_dim)

    X_inf      = torch.tensor([ctx])
    genre_emb  = model.user_genre_tower(X_inf)
    ts_emb     = model.timestamp_embedding_tower(model.timestamp_embedding_lookup(ts_inference))

    user_parts = [history_emb, genre_emb]
    if model.use_user_tag_tower:
        # Tag profile: avg of liked seed movies' tag vectors
        tag_ctx = [0.0] * len(fs.tags_ordered)
        n_tagged = 0
        for t in fav_movies:
            if t not in fs.title_to_movieId:
                continue
            mid = fs.title_to_movieId[t]
            if mid in fs.movieId_to_tag_context:
                for j, v in enumerate(fs.movieId_to_tag_context[mid]):
                    tag_ctx[j] += v
                n_tagged += 1
        if n_tagged > 0:
            tag_ctx = [v / n_tagged for v in tag_ctx]
        user_parts.append(model.user_tag_tower(torch.tensor([tag_ctx])))

    user_parts.append(ts_emb)
    return torch.cat(user_parts, dim=1)


def run_canary_eval(model: MovieRecommender, fs: FeatureStore,
                    movie_embeddings: dict, top_n: int = 10) -> None:
    """Run all canary users and print recommendation tables."""
    model.eval()

    # Use the most recent timestamp from the timestamp range
    ts_max_bin = torch.bucketize(
        torch.tensor([float(fs.timestamp_bins[-1].item())]),
        fs.timestamp_bins, right=False
    )

    with torch.no_grad():
        for user_type in USER_TYPE_TO_FAVORITE_GENRES:
            user_emb = _build_user_embedding(model, fs, user_type, ts_max_bin)
            fav_set  = set(USER_TYPE_TO_FAVORITE_MOVIES[user_type])
            dis_set  = set(USER_TYPE_TO_DISLIKED_MOVIES[user_type])

            scores = {}
            for movieId in tqdm(fs.top_movies, desc=user_type, leave=False):
                item_emb = movie_embeddings[movieId]['MOVIE_EMBEDDING_COMBINED']
                scores[movieId] = torch.einsum('ij,ij->i', user_emb, item_emb).item()

            recs = []
            for mid, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True):
                if len(recs) >= top_n:
                    break
                title = fs.movieId_to_title[mid]
                if title not in fav_set and title not in dis_set:
                    recs.append(title)

            liked_genres    = ', '.join(USER_TYPE_TO_FAVORITE_GENRES[user_type])
            disliked_genres = ', '.join(USER_TYPE_TO_WORST_GENRES[user_type])
            liked_movies    = USER_TYPE_TO_FAVORITE_MOVIES[user_type]
            disliked_movies = USER_TYPE_TO_DISLIKED_MOVIES[user_type]

            col_w      = min(55, max((len(t) for t in liked_movies + disliked_movies), default=20))
            rec_w      = min(55, max((len(r) for r in recs), default=20))
            title_line = f"{user_type}  |  Likes: {liked_genres}  |  Dislikes: {disliked_genres}"
            bar_w      = max(col_w * 2 + rec_w + 6, len(title_line))

            print(f"\n{'═' * bar_w}")
            print(title_line)
            print(f"{'═' * bar_w}")
            header = f"{'Liked Movies':<{col_w}}  {'Disliked Movies':<{col_w}}  Recommendations"
            print(header)
            print('─' * bar_w)
            for a, b, c in zip_longest(liked_movies, disliked_movies, recs, fillvalue=''):
                print(f"{a:<{col_w}}  {b:<{col_w}}  {c}")


# ── Embedding probes ──────────────────────────────────────────────────────────

def probe_genre(model: MovieRecommender, genre: str, movie_embeddings: dict,
                fs: FeatureStore, top_n: int = 10) -> None:
    """
    Find the most representative movies for a genre in the item genre embedding space.
    Passes a one-hot genre vector through item_genre_tower, then compares via cosine
    similarity against every movie's MOVIE_GENRE_EMBEDDING.
    """
    if genre not in fs.genre_to_i:
        print(f"Genre '{genre}' not in vocabulary.")
        return

    ctx = [0.0] * len(fs.genres_ordered)
    ctx[fs.genre_to_i[genre]] = 1.0

    with torch.no_grad():
        query_emb = model.item_genre_tower(torch.tensor([ctx])).view(-1)

    sims = {
        mid: F.cosine_similarity(
            query_emb.unsqueeze(0),
            movie_embeddings[mid]['MOVIE_GENRE_EMBEDDING'].view(-1).unsqueeze(0)
        ).item()
        for mid in fs.top_movies
    }

    print(f"\nTop-{top_n} movies for genre '{genre}':")
    for mid, sim in sorted(sims.items(), key=lambda x: x[1], reverse=True)[:top_n]:
        print(f"  {sim:.4f}  {fs.movieId_to_title[mid]}")


def probe_tag(model: MovieRecommender, tags: list, movie_embeddings: dict,
              fs: FeatureStore, top_n: int = 10) -> None:
    """
    Find the most representative movies for a tag query in the item tag embedding space.
    Passes the tag vector through item_tag_tower, then compares via cosine similarity
    against every movie's MOVIE_TAG_EMBEDDING.
    """
    assert model.use_item_tag_tower, "Item tag tower disabled."
    ctx = [0.0] * len(fs.tags_ordered)
    for tag in tags:
        if tag in fs.tag_to_i:
            ctx[fs.tag_to_i[tag]] = 1.0 / len(tags)

    with torch.no_grad():
        query_emb = model.item_tag_tower(torch.tensor([ctx], dtype=torch.float)).view(-1)

    sims = {
        mid: F.cosine_similarity(
            query_emb.unsqueeze(0),
            movie_embeddings[mid]['MOVIE_TAG_EMBEDDING'].view(-1).unsqueeze(0)
        ).item()
        for mid in fs.top_movies
    }

    print(f"\nTop-{top_n} movies for tags {tags}:")
    for mid, sim in sorted(sims.items(), key=lambda x: x[1], reverse=True)[:top_n]:
        print(f"  {sim:.4f}  {fs.movieId_to_title[mid]}")


def probe_genome_tag(model: MovieRecommender, genome_tags: list, movie_embeddings: dict,
                     fs: FeatureStore, top_n: int = 10, k_anchors: int = 3) -> None:
    """
    Find movies most similar to a genome tag query in the item genome tag embedding space.
    Finds the top-k_anchors most representative movies by raw genome score, averages their
    MOVIE_GENOME_TAG_EMBEDDING vectors as the query, then compares via cosine similarity
    against all movies' MOVIE_GENOME_TAG_EMBEDDING. Avoids synthetic OOD inputs entirely.
    """
    assert model.use_item_genome_tag_tower, "Genome tag tower disabled."
    name_to_idx = {
        fs.genome_tag_names[tid]: fs.genome_tag_to_i[tid]
        for tid in fs.genome_tag_to_i
    }

    raw_scores = {}
    for mid in fs.top_movies:
        vec = fs.movieId_to_genome_tag_context[mid]
        raw_scores[mid] = sum(vec[name_to_idx[t]] for t in genome_tags if t in name_to_idx)

    anchors   = sorted(raw_scores, key=raw_scores.get, reverse=True)[:k_anchors]
    query_emb = torch.stack([
        movie_embeddings[m]['MOVIE_GENOME_TAG_EMBEDDING'].view(-1) for m in anchors
    ]).mean(dim=0)

    print(f"\nGenome tag anchors for {genome_tags}: {[fs.movieId_to_title[m] for m in anchors]}")

    sims = {
        mid: F.cosine_similarity(
            query_emb.unsqueeze(0),
            movie_embeddings[mid]['MOVIE_GENOME_TAG_EMBEDDING'].view(-1).unsqueeze(0)
        ).item()
        for mid in fs.top_movies
    }

    anchor_set = set(anchors)
    print(f"Top-{top_n} movies:")
    for mid, sim in sorted(sims.items(), key=lambda x: x[1], reverse=True)[:top_n]:
        marker = " [seed]" if mid in anchor_set else ""
        print(f"  {sim:.4f}  {fs.movieId_to_title[mid]}{marker}")


# ── Shared setup ─────────────────────────────────────────────────────────────

def _setup(data_dir: str, checkpoint_path: str, version: str):
    """Load features, build model, load checkpoint, print dims, build embeddings."""
    from src.dataset import load_features

    # Resolve and load checkpoint first — fast, and fails before the slow features load
    config = get_config()
    if checkpoint_path is None:
        pattern    = os.path.join(config['checkpoint_dir'], 'best_checkpoint_*.pth')
        candidates = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
        if not candidates:
            print("No checkpoint found in saved_models/. Train a model first.")
            return None, None, None
        checkpoint_path = candidates[0]

    print(f"Loading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, weights_only=True)

    print("Loading features ...")
    fs = load_features(data_dir, version)

    model = build_model(config, fs)
    model.load_state_dict(state_dict)
    model.eval()

    m = model
    history_dim  = m.item_embedding_lookup.embedding_dim
    genre_dim    = m.user_genre_tower[0].out_features
    ts_dim       = m.timestamp_embedding_lookup.embedding_dim
    user_tag_dim = m.user_tag_tower[0].out_features if m.use_user_tag_tower else 0
    user_total   = history_dim + genre_dim + ts_dim + user_tag_dim

    item_genre_dim   = m.item_genre_tower[0].out_features
    item_tag_dim     = m.item_tag_tower[0].out_features        if m.use_item_tag_tower        else 0
    genome_tag_dim   = m.item_genome_tag_tower[0].out_features if m.use_item_genome_tag_tower else 0
    item_movieId_dim = m.item_embedding_tower[0].out_features
    year_dim         = m.year_embedding_tower[0].out_features
    item_total       = item_genre_dim + item_tag_dim + genome_tag_dim + item_movieId_dim + year_dim

    print(f"\n── Model dimensions ──")
    print(f"  User side:  history({history_dim}) + genre({genre_dim}) + ts({ts_dim})"
          + (f" + user_tag({user_tag_dim})" if m.use_user_tag_tower else "")
          + f"  =  {user_total}")
    print(f"  Item side:  genre({item_genre_dim})"
          + (f" + tag({item_tag_dim})"      if m.use_item_tag_tower        else "")
          + (f" + genome({genome_tag_dim})" if m.use_item_genome_tag_tower else "")
          + f" + movieId({item_movieId_dim}) + year({year_dim})  =  {item_total}")
    print(f"  Towers:  use_user_tag={m.use_user_tag_tower}  "
          f"use_item_tag={m.use_item_tag_tower}  "
          f"use_item_genome_tag={m.use_item_genome_tag_tower}")

    top_movies_len  = m.item_embedding_lookup.num_embeddings - 1
    genres_len      = m.item_genre_tower[0].in_features
    tags_len        = (m.item_tag_tower[0].in_features    if m.use_item_tag_tower
                       else m.user_tag_tower[0].in_features if m.use_user_tag_tower else "n/a")
    genome_tags_len = m.item_genome_tag_tower[0].in_features if m.use_item_genome_tag_tower else "n/a"
    all_years_len   = m.year_embedding_lookup.num_embeddings
    ts_num_bins     = m.timestamp_embedding_lookup.num_embeddings
    user_ctx_size   = m.user_genre_tower[0].in_features
    n_genres_from_ctx = user_ctx_size // 2
    print(f"\n── Required vocab sizes ──")
    print(f"  top_movies:      {top_movies_len}")
    print(f"  genres:          {genres_len}")
    print(f"  tags:            {tags_len}")
    print(f"  genome_tags:     {genome_tags_len}")
    print(f"  years:           {all_years_len}")
    print(f"  timestamp_bins:  {ts_num_bins}")
    print(f"  user_ctx_size:   {user_ctx_size}  "
          f"(avg_rating_per_genre({n_genres_from_ctx}) + watch_frac_per_genre({n_genres_from_ctx}))")

    print("\nBuilding movie embeddings ...")
    movie_embeddings = build_movie_embeddings(model, fs)

    return model, fs, movie_embeddings


# ── Orchestrators ─────────────────────────────────────────────────────────────

def run_canary(data_dir: str = 'data', checkpoint_path: str = None,
               version: str = 'v1') -> None:
    model, fs, movie_embeddings = _setup(data_dir, checkpoint_path, version)
    if model is None:
        return
    print("\n── Canary user evaluation ──")
    run_canary_eval(model, fs, movie_embeddings)


PROBE_SIMILAR_TITLES = [
    'Lord of the Rings: The Return of the King, The (2003)',
    'Star Wars: Episode IV - A New Hope (1977)',
    'Toy Story (1995)',
    'Saving Private Ryan (1998)',
    'Kill Bill: Vol. 1 (2003)',
    'American Pie (1999)',
    'Blair Witch Project, The (1999)',
    'Princess Mononoke (Mononoke-hime) (1997)',
]


def probe_similar(movie_embeddings: dict, fs: FeatureStore,
                  titles: list, top_n: int = 5) -> None:
    """
    For each query title, find the top-N most similar movies by cosine similarity
    on MOVIE_EMBEDDING_COMBINED.
    """
    # Stack all embeddings into a matrix for batched cosine similarity
    all_ids  = list(movie_embeddings.keys())
    all_embs = torch.cat([movie_embeddings[m]['MOVIE_EMBEDDING_COMBINED'] for m in all_ids], dim=0)
    # Normalize rows for cosine similarity via dot product
    all_norm = F.normalize(all_embs, dim=1)

    TRUNC = 28  # max chars per cell

    def trunc(s: str) -> str:
        return s if len(s) <= TRUNC else s[:TRUNC - 1] + '…'

    # Collect all rows first
    rows = []
    for title in titles:
        if title not in fs.title_to_movieId:
            rows.append((title, []))
            continue
        mid     = fs.title_to_movieId[title]
        query   = F.normalize(movie_embeddings[mid]['MOVIE_EMBEDDING_COMBINED'], dim=1)
        sims    = (all_norm @ query.T).squeeze(-1)
        top_idx = sims.argsort(descending=True)
        results = []
        for idx in top_idx:
            candidate = all_ids[idx.item()]
            if candidate == mid:
                continue
            results.append(fs.movieId_to_title[candidate])
            if len(results) >= top_n:
                break
        rows.append((title, results))

    seed_w = max(len(trunc(t)) for t, _ in rows)
    col_w  = TRUNC
    bar_w  = seed_w + (col_w + 3) * top_n + 2

    header = f"{'Seed':<{seed_w}}" + "".join(f"  {'#'+str(i+1):<{col_w}}" for i in range(top_n))
    print(f"\n── Most similar movies ──")
    print(header)
    print('─' * len(header))
    for title, results in rows:
        if not results:
            print(f"{trunc(title):<{seed_w}}  (not in corpus)")
            continue
        row = f"{trunc(title):<{seed_w}}"
        for t in results:
            row += f"  {trunc(t):<{col_w}}"
        print(row)


def run_probes(data_dir: str = 'data', checkpoint_path: str = None,
               version: str = 'v1') -> None:
    model, fs, movie_embeddings = _setup(data_dir, checkpoint_path, version)
    if model is None:
        return
    print("\n── Embedding probes ──")
    probe_genre(model, 'Horror', movie_embeddings, fs)
    probe_genre(model, 'Sci-Fi', movie_embeddings, fs)
    if model.use_item_tag_tower:
        probe_tag(model, ['pixar', 'animation'], movie_embeddings, fs)
    if model.use_item_genome_tag_tower:
        probe_genome_tag(model, ['horror', 'gore', 'torture'], movie_embeddings, fs)
        probe_genome_tag(model, ['martial arts', 'kung fu'], movie_embeddings, fs)
    probe_similar(movie_embeddings, fs, PROBE_SIMILAR_TITLES)


