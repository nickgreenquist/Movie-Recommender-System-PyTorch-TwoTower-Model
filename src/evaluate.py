"""
Inference, canary user evaluation, and embedding probes.

Usage:
    python main.py evaluate
"""
import glob
import os
from itertools import zip_longest

import numpy as np
import torch
import torch.nn.functional as F
from src.dataset import FeatureStore
from src.model import MovieRecommender
from src.train import build_model, get_config, get_device, print_model_summary


# ── Canary user definitions ───────────────────────────────────────────────────


USER_TYPE_TO_FAVORITE_MOVIES = {
    "Children's Movie Lover": [
        'Toy Story 2 (1999)',
        'Finding Nemo (2003)',
        'Finding Dory (2016)',
        'Madagascar (2005)'
    ],
    'Horror Lover': [
        'Blair Witch Project, The (1999)',
        'Texas Chainsaw Massacre, The (2003)',
        'Exorcism of Emily Rose, The (2005)',
        'Wrong Turn (2003)'
    ],
    'Sci-Fi Lover': [
        'Contact (1997)',
        '2010: The Year We Make Contact (1984)',
        'It Came from Outer Space (1953)',
        'Solaris (Solyaris) (1972)',
        '2001: A Space Odyssey (1968)'
    ],
    'Comedy Lover': [
        'American Pie (1999)',
        'Dumb & Dumber (Dumb and Dumber) (1994)',
        'Dodgeball: A True Underdog Story (2004)',
        'Ted 2 (2015)'
    ],
    'Romance Lover': [
        'Shakespeare in Love (1998)', 'Sense and Sensibility (1995)',
        'When Harry Met Sally... (1989)', 'Sleepless in Seattle (1993)',
        'Notting Hill (1999)', 'Pride and Prejudice (1995)', 'North & South (2004)'
    ],
    'War Movie Lover': [
        'Bridge on the River Kwai, The (1957)', 'Glory (1989)',
        'Downfall (Untergang, Der) (2004)', '1917 (2019)',
        'Enemy at the Gates (2001)',
    ],
    'Fantasy Lover': [
        'Lord of the Rings: The Fellowship of the Ring, The (2001)',
        'Dark Crystal, The (1982)',
        'Lord of the Rings: The Return of the King, The (2003)',
        'Dragonslayer (1981)',
        'Dune (1984)'
    ],
    'Crime Lover': [
        'Goodfellas (1990)',
        'Reservoir Dogs (1992)',
        'Donnie Brasco (1997)',
        'The Irishman (2019)',
        'Casino (1995)',
        'Narc (2002)'
    ],
    'Heist Lover':         [
        'Heist (2001)',
        "Ocean's Eleven (2001)",
        "Ocean's Eleven (a.k.a. Ocean's 11) (1960)",
        'The Drop (2014)',
        'Bank Job, The (2008)',
        'Italian Job, The (1969)',
        'Town, The (2010)'
    ],
    'Action Junkie':       ['Die Hard 2 (1990)', 'Rambo III (1988)', 'Under Siege (1992)'],
    'Arthouse Lover':      ['The Lobster (2015)', 'Antichrist (2009)'],
    'Superhero Lover':     ['Guardians of the Galaxy (2014)', 'Iron Man 3 (2013)', 'Avengers: Age of Ultron (2015)', 'Ant-Man and the Wasp: Quantumania (2023)', 'Aquaman (2018)', 'Captain America: Civil War (2016)'],
    'WW2 Lover':           ['Stalingrad (1993)', 'Run Silent Run Deep (1958)', 'Great Escape, The (1963)', 'Band of Brothers (2001)'],
    'Western Lover':       [
        'True Grit (1969)',
        'High Plains Drifter (1973)',
        'Cool Hand Luke (1967)',
        'Wild Bill (1995)',
        'Wyatt Earp (1994)',
        'Fistful of Dollars, A (Per un pugno di dollari) (1964)',
        'Wild Bunch, The (1969)',
        'For a Few Dollars More (Per qualche dollaro in più) (1965)',
        "Once Upon a Time in the West (C'era una volta il West) (1968)",

    ],
    'Anime Lover':         [
        'Princess Mononoke (Mononoke-hime) (1997)',
        'Berserk: The Golden Age Arc 2 - The Battle for Doldrey (2012)',
        'Ponyo (Gake no ue no Ponyo) (2008)',
        'Animatrix, The (2003)',
        'Cowboy Bebop: The Movie (Cowboy Bebop: Tengoku no Tobira) (2001)'
    ],
    'Martial Arts Lover':  [
        'Ong-Bak: The Thai Warrior (Ong Bak) (2003)',
        "Jet Li\'s Fearless (Huo Yuan Jia) (2006)",
        'Protector, The (a.k.a. Warrior King) (Tom yum goong) (2005)',
        'Unleashed (Danny the Dog) (2005)',
        'The Raid 2: Berandal (2014)',
        "Project A ('A' gai waak) (1983)",
        'Shaolin Soccer (Siu lam juk kau) (2001)'
    ],
    "Nick's Recommendations": [
        'Lord of the Rings: The Fellowship of the Ring, The (2001)',
        'Lord of the Rings: The Return of the King, The (2003)',
        '300 (2007)',
        'Kill Bill: Vol. 1 (2003)',
        'Lost in Translation (2003)',
        'Enter the Dragon (1973)',
        'Casino Royale (2006)',
        'Before Sunrise (1995)',
        'Old Boy (2003)',
        'Idiocracy (2006)',
        'Parasite (2019)',
        '28 Days Later (2002)',
        'Saving Private Ryan (1998)',
        'Ip Man (2008)'
    ],
}

USER_TYPE_TO_DISLIKED_MOVIES = {
    "Children's Movie Lover": [],
    'Horror Lover':           [],
    'Sci-Fi Lover':           [],
    'Comedy Lover':           [],
    'Romance Lover':          [],
    'War Movie Lover':        [],
    'Fantasy Lover':          [],
    'Crime Lover':            [],
    'Heist Lover':            [],
    'Action Junkie':          [],
    'Arthouse Lover':         [],
    'Superhero Lover':        [],
    'WW2 Lover':              [],
    'Western Lover':          [],
    'Anime Lover':            [
        'MirrorMask (2005)'
    ],
    'Martial Arts Lover':     [],
    "Nick's Recommendations": [
        'Planet Terror (2007)',
        'Twilight (2008)'
    ],
}

USER_TYPE_TO_GENOME_TAGS = {
    'Crime Lover':           ['crime', 'gangs'],
    'Heist Lover':           ['heist'],
    'Action Junkie':         ['explosions', 'adrenaline'],
    'Arthouse Lover':        ['art house', 'slow burn'],
    'Superhero Lover':       ['superhero', 'superheroes'],
    'WW2 Lover':             ['world war ii', 'wwii'],
    'Western Lover':         ['spaghetti western'],
    'Anime Lover':           ['studio ghibli'],
    'Martial Arts Lover':    ['kung fu'],
    "Nick's Recommendations": []
}

VALUE_FAVORITE_MOVIE_RATING = 2.0
VALUE_DISLIKED_MOVIE_RATING = -2.0
VALUE_ANCHOR_MOVIE_RATING   = 1.0
ANCHORS_PER_TAG             = 5


# ── Movie embedding cache ─────────────────────────────────────────────────────

def build_movie_embeddings(model: MovieRecommender, fs: FeatureStore) -> dict:
    """
    Pre-compute all movie embeddings in a single batched forward pass.
    Returns movieId → {'MOVIE_EMBEDDING_COMBINED': Tensor, ...}
    """
    model.eval()
    device  = next(model.parameters()).device
    all_mids = [int(m) for m in fs.top_movies]

    with torch.no_grad():
        emb_idx  = torch.tensor([fs.item_emb_movieId_to_i[m]          for m in all_mids], dtype=torch.long).to(device)
        year_idx = torch.tensor([fs.year_to_i[fs.movieId_to_year[m]]  for m in all_mids], dtype=torch.long).to(device)
        genre_t  = torch.tensor([fs.movieId_to_genre_context[m]       for m in all_mids], dtype=torch.float32).to(device)
        tag_t    = torch.tensor([fs.movieId_to_tag_context[m]         for m in all_mids], dtype=torch.float32).to(device)
        genome_t = torch.tensor([fs.movieId_to_genome_tag_context[m]  for m in all_mids], dtype=torch.float32).to(device)

        id_embs      = model.item_embedding_tower(model.item_embedding_lookup(emb_idx))
        year_embs    = model.year_embedding_tower(model.year_embedding_lookup(year_idx))
        genre_embs   = model.item_genre_tower(genre_t)
        tag_embs     = model.item_tag_tower(tag_t)
        genome_embs  = model.item_genome_tag_tower(genome_t)
        combined     = model.item_embedding(genre_t, tag_t, genome_t, year_idx, emb_idx)

    movieId_to_embedding = {}
    for i, mid in enumerate(all_mids):
        movieId_to_embedding[mid] = {
            'MOVIEID_EMBEDDING':          id_embs[i:i+1],
            'MOVIE_YEAR_EMBEDDING':       year_embs[i:i+1],
            'MOVIE_GENRE_EMBEDDING':      genre_embs[i:i+1],
            'MOVIE_TAG_EMBEDDING':        tag_embs[i:i+1],
            'MOVIE_GENOME_TAG_EMBEDDING': genome_embs[i:i+1],
            'MOVIE_EMBEDDING_COMBINED':   combined[i:i+1],
        }
    return movieId_to_embedding


# ── Canary user inference ─────────────────────────────────────────────────────

def _get_anchor_titles(fs: FeatureStore, genome_tags: list, exclude: set) -> list:
    """Return up to ANCHORS_PER_TAG top movies per genome tag, skipping titles in exclude."""
    name_to_idx = {fs.genome_tag_names[tid]: fs.genome_tag_to_i[tid] for tid in fs.genome_tag_to_i}
    anchor_titles = []
    seen = set(exclude)
    for tag in genome_tags:
        if tag not in name_to_idx:
            continue
        tag_idx = name_to_idx[tag]
        sorted_mids = sorted(
            fs.top_movies,
            key=lambda mid: float(fs.movieId_to_genome_tag_context[mid][tag_idx]),
            reverse=True,
        )
        count = 0
        for mid in sorted_mids:
            if count >= ANCHORS_PER_TAG:
                break
            title = fs.movieId_to_title[mid]
            if title not in seen:
                anchor_titles.append(title)
                seen.add(title)
                count += 1
    return anchor_titles


def _build_user_embedding(model: MovieRecommender, fs: FeatureStore, user_type: str,
                          ts_inference: torch.Tensor) -> torch.Tensor:
    """Build the combined user embedding for a canary user type. Mirrors website logic."""
    fav_movies   = USER_TYPE_TO_FAVORITE_MOVIES[user_type]
    dis_movies   = USER_TYPE_TO_DISLIKED_MOVIES[user_type]
    genome_tags  = USER_TYPE_TO_GENOME_TAGS.get(user_type, [])

    # Genome anchor movies — top ANCHORS_PER_TAG movies per genome tag
    anchor_titles = _get_anchor_titles(fs, genome_tags, exclude=set(fav_movies))

    liked_with_weights = (
        [(t, VALUE_FAVORITE_MOVIE_RATING) for t in fav_movies] +
        [(t, VALUE_ANCHOR_MOVIE_RATING)   for t in anchor_titles]
    )
    # Genre context — derived from fav/disliked movies
    n_genres = len(fs.genres_ordered)
    ctx = [0.0] * (2 * n_genres)
    genre_rating_sum  = {}
    genre_movie_count = {}
    total_movies = 0
    for t, w in liked_with_weights + [(t, VALUE_DISLIKED_MOVIE_RATING) for t in dis_movies]:
        mid = fs.title_to_movieId.get(t)
        if mid is None:
            continue
        total_movies += 1
        for g in fs.movieId_to_genres.get(mid, []):
            genre_rating_sum[g]  = genre_rating_sum.get(g, 0.0) + w
            genre_movie_count[g] = genre_movie_count.get(g, 0)  + 1
    for g, rsum in genre_rating_sum.items():
        avg_r = rsum / genre_movie_count[g]
        frac  = genre_movie_count[g] / max(total_movies, 1)
        if g in fs.user_context_genre_avg_rating_to_i:
            ctx[fs.user_context_genre_avg_rating_to_i[g]]  = avg_r
        if g in fs.user_context_genre_watch_count_to_i:
            ctx[fs.user_context_genre_watch_count_to_i[g]] = frac
    # Watch history
    liked_hist = [
        (fs.item_emb_movieId_to_i[fs.title_to_movieId[t]], w)
        for t, w in liked_with_weights
        if t in fs.title_to_movieId and fs.title_to_movieId[t] in fs.item_emb_movieId_to_i
    ]
    dis_hist = [
        (fs.item_emb_movieId_to_i[fs.title_to_movieId[t]], VALUE_DISLIKED_MOVIE_RATING)
        for t in dis_movies
        if t in fs.title_to_movieId and fs.title_to_movieId[t] in fs.item_emb_movieId_to_i
    ]
    history = liked_hist + dis_hist
    ratings = [h[1] for h in liked_hist] + [VALUE_DISLIKED_MOVIE_RATING] * len(dis_hist)

    device = next(model.parameters()).device
    if history:
        hist_ids = torch.tensor([[h[0] for h in history]], dtype=torch.long).to(device)
        hist_wts = torch.tensor([ratings], dtype=torch.float).to(device)
    else:
        hist_ids = torch.tensor([[model.pad_idx]], dtype=torch.long).to(device)
        hist_wts = torch.tensor([[0.0]], dtype=torch.float).to(device)

    X_inf = torch.tensor([ctx]).to(device)
    return model.user_embedding(X_inf, hist_ids, hist_wts, ts_inference)


def run_canary_eval(model: MovieRecommender, fs: FeatureStore,
                    movie_embeddings: dict, all_ids: list, all_embs: torch.Tensor,
                    top_n: int = 10) -> None:
    """Run all canary users and print recommendation tables."""
    model.eval()
    device = next(model.parameters()).device

    # Use the most recent timestamp from the timestamp range
    ts_max_bin = torch.bucketize(
        torch.tensor([float(fs.timestamp_bins[-1].item())]),
        fs.timestamp_bins, right=False
    ).to(device)
    all_embs = all_embs.to(device)

    with torch.no_grad():
        for user_type in USER_TYPE_TO_FAVORITE_MOVIES:
            user_emb    = _build_user_embedding(model, fs, user_type, ts_max_bin)
            fav_movies  = USER_TYPE_TO_FAVORITE_MOVIES[user_type]
            dis_movies  = USER_TYPE_TO_DISLIKED_MOVIES[user_type]
            genome_tags = USER_TYPE_TO_GENOME_TAGS.get(user_type, [])
            anchor_titles = _get_anchor_titles(fs, genome_tags, exclude=set(fav_movies))
            exclude_set = set(fav_movies) | set(dis_movies) | set(anchor_titles)

            raw_scores  = (all_embs @ user_emb.T).squeeze(-1)
            sorted_idxs = torch.argsort(raw_scores, descending=True).tolist()

            recs = []
            for idx in sorted_idxs:
                if len(recs) >= top_n:
                    break
                title = fs.movieId_to_title[all_ids[idx]]
                if title not in exclude_set:
                    recs.append(title)

            col_w      = min(55, max((len(t) for t in fav_movies), default=20))
            rec_w      = min(55, max((len(r) for r in recs), default=20))
            title_line = user_type
            if genome_tags:
                title_line += f"  |  Genome: {', '.join(genome_tags)}"
            bar_w      = max(col_w + rec_w + 4, len(title_line))

            print(f"\n{'═' * bar_w}")
            print(title_line)
            print(f"{'═' * bar_w}")
            if dis_movies:
                print(f"Disliked: {', '.join(dis_movies)}")
            if anchor_titles:
                print(f"Anchors:  {', '.join(anchor_titles[:5])}")
            print()
            header = f"{'Liked Movies':<{col_w}}  Recommendations"
            print(header)
            print('─' * bar_w)
            for a, b in zip_longest(fav_movies, recs, fillvalue=''):
                print(f"{a:<{col_w}}  {b}")


# ── Genome context probe ─────────────────────────────────────────────────────

def build_user_genome_context(user_type: str, fs: FeatureStore) -> 'np.ndarray':
    """
    Build a rating-weighted genome context vector for a canary user.
    Mirrors the proposed user genome feature: for each watched movie, weight
    its 1128-dim genome score vector by the user's rating, sum, normalise by
    total absolute weight.  Returns a (genome_tags_len,) float32 array.
    """
    fav_movies    = USER_TYPE_TO_FAVORITE_MOVIES[user_type]
    dis_movies    = USER_TYPE_TO_DISLIKED_MOVIES[user_type]
    genome_tags   = USER_TYPE_TO_GENOME_TAGS.get(user_type, [])
    anchor_titles = _get_anchor_titles(fs, genome_tags, exclude=set(fav_movies))

    all_movies_with_weights = (
        [(t, VALUE_FAVORITE_MOVIE_RATING) for t in fav_movies] +
        [(t, VALUE_ANCHOR_MOVIE_RATING)   for t in anchor_titles] +
        [(t, VALUE_DISLIKED_MOVIE_RATING) for t in dis_movies]
    )

    genome_len   = len(next(iter(fs.movieId_to_genome_tag_context.values())))
    weighted_sum = np.zeros(genome_len, dtype=np.float32)
    weight_total = 0.0

    for title, weight in all_movies_with_weights:
        mid = fs.title_to_movieId.get(title)
        if mid is None or mid not in fs.movieId_to_genome_tag_context:
            continue
        weighted_sum += weight * np.array(fs.movieId_to_genome_tag_context[mid], dtype=np.float32)
        weight_total += abs(weight)

    if weight_total > 0:
        weighted_sum /= weight_total
    return weighted_sum


def probe_genome_context(fs: FeatureStore, top_n: int = 15) -> None:
    """Print the top genome tags for each canary user to validate the feature."""
    # index → tag name
    idx_to_name = {i: fs.genome_tag_names[tid] for tid, i in fs.genome_tag_to_i.items()}

    print('\n' + '═' * 70)
    print('GENOME CONTEXT PROBE — top genome tags per canary user')
    print('═' * 70)

    for user_type in USER_TYPE_TO_FAVORITE_MOVIES:
        ctx = build_user_genome_context(user_type, fs)
        top_indices = np.argsort(ctx)[::-1][:top_n]

        fav_movies  = USER_TYPE_TO_FAVORITE_MOVIES[user_type]
        dis_movies  = USER_TYPE_TO_DISLIKED_MOVIES[user_type]
        genre_tags  = USER_TYPE_TO_GENOME_TAGS.get(user_type, [])
        anchor_titles = _get_anchor_titles(fs, genre_tags, exclude=set(fav_movies))

        print(f'\n── {user_type} ' + '─' * max(0, 50 - len(user_type)))
        print(f'  Liked:   {", ".join(fav_movies[:4])}{"..." if len(fav_movies) > 4 else ""}')
        if dis_movies:
            print(f'  Disliked: {", ".join(dis_movies[:3])}')
        if anchor_titles:
            print(f'  Anchors: {", ".join(anchor_titles[:4])}{"..." if len(anchor_titles) > 4 else ""}')
        print(f'  {"Rank":<4}  {"Genome Tag":<30}  Score')
        print(f'  {"────":<4}  {"──────────":<30}  ─────')
        for rank, idx in enumerate(top_indices, 1):
            print(f'  {rank:<4}  {idx_to_name[idx]:<30}  {ctx[idx]:.4f}')


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

    device = next(model.parameters()).device
    with torch.no_grad():
        query_emb = model.item_genre_tower(torch.tensor([ctx]).to(device)).view(-1)

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
    ctx = [0.0] * len(fs.tags_ordered)
    for tag in tags:
        if tag in fs.tag_to_i:
            ctx[fs.tag_to_i[tag]] = 1.0 / len(tags)

    device = next(model.parameters()).device
    with torch.no_grad():
        query_emb = model.item_tag_tower(torch.tensor([ctx], dtype=torch.float).to(device)).view(-1)

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
    # Detect checkpoint type from filename to pick the right config
    def _resolve_config(path):
        sd  = torch.load(path, weights_only=True)
        cfg = get_config()

        item_id_dim = sd['item_embedding_lookup.weight'].shape[1]
        ts_dim      = sd['timestamp_embedding_lookup.weight'].shape[1]
        genre_dim   = sd['user_genre_tower.0.weight'].shape[0]
        genome_dim  = sd['item_genome_tag_tower.0.weight'].shape[0]
        cfg['item_movieId_embedding_size']      = item_id_dim
        cfg['user_genre_embedding_size']        = genre_dim
        cfg['timestamp_feature_embedding_size'] = ts_dim
        cfg['item_genre_embedding_size']        = sd['item_genre_tower.0.weight'].shape[0]
        cfg['item_tag_embedding_size']          = sd['item_tag_tower.0.weight'].shape[0]
        cfg['item_genome_tag_embedding_size']   = genome_dim
        cfg['item_year_embedding_size']         = sd['year_embedding_lookup.weight'].shape[1]

        cfg['user_genome_context_embedding_size'] = sd['user_genome_context_tower.0.weight'].shape[0]

        cfg['proj_hidden'] = sd['user_projection.0.weight'].shape[0]
        cfg['output_dim']  = sd['user_projection.2.weight'].shape[0]
        return cfg

    # Auto-detect most recent checkpoint if none specified
    _tmp_config = get_config()
    if checkpoint_path is None:
        checkpoint_dir = _tmp_config['checkpoint_dir']
        candidates = sorted(
            glob.glob(os.path.join(checkpoint_dir, 'best_mse_*.pth')),
            key=os.path.getmtime, reverse=True,
        )
        if not candidates:
            print("No checkpoint found in saved_models/. Train a model first.")
            return None, None, None
        checkpoint_path = candidates[0]

    config = _resolve_config(checkpoint_path)

    print(f"Loading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, weights_only=True)

    print("Loading features ...")
    fs = load_features(data_dir, version)

    model = build_model(config, fs)
    model.load_state_dict(state_dict)
    device = get_device()
    model = model.to(device)
    model.eval()
    print_model_summary(model)

    m = model
    top_movies_len    = m.item_embedding_lookup.num_embeddings - 1
    genres_len        = m.item_genre_tower[0].in_features
    tags_len          = m.item_tag_tower[0].in_features
    genome_tags_len   = m.item_genome_tag_tower[0].in_features
    all_years_len     = m.year_embedding_lookup.num_embeddings
    ts_num_bins       = m.timestamp_embedding_lookup.num_embeddings
    user_ctx_size     = m.user_genre_tower[0].in_features
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

    print("Precomputing embedding matrices ...")
    all_ids    = list(movie_embeddings.keys())
    all_embs   = torch.cat([movie_embeddings[m]['MOVIE_EMBEDDING_COMBINED'] for m in all_ids], dim=0)
    all_norm   = F.normalize(all_embs, dim=1)
    all_id_embs = torch.cat([movie_embeddings[m]['MOVIEID_EMBEDDING'] for m in all_ids], dim=0)
    all_id_norm = F.normalize(all_id_embs, dim=1)

    return model, fs, movie_embeddings, all_ids, all_embs, all_norm, all_id_norm, checkpoint_path


# ── Orchestrators ─────────────────────────────────────────────────────────────

def run_canary(data_dir: str = 'data', checkpoint_path: str = None,
               version: str = 'v1') -> None:
    import contextlib, io, sys
    model, fs, movie_embeddings, all_ids, all_embs, all_norm, all_id_norm, checkpoint_path = _setup(data_dir, checkpoint_path, version)
    if model is None:
        return
    print("\n── Canary user evaluation ──")

    real_stdout = sys.stdout
    buf = io.StringIO()

    class _Tee:
        def write(self, data): real_stdout.write(data); buf.write(data)
        def flush(self):       real_stdout.flush();     buf.flush()

    with contextlib.redirect_stdout(_Tee()):
        run_canary_eval(model, fs, movie_embeddings, all_ids, all_embs)

    os.makedirs('canary_results', exist_ok=True)
    stem     = os.path.splitext(os.path.basename(checkpoint_path))[0]
    out_path = os.path.join('canary_results', f'{stem}.txt')
    with open(out_path, 'w') as f:
        f.write(buf.getvalue())
    print(f"\n  → canary saved to {out_path}")


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
                  all_ids: list, all_norm: torch.Tensor, all_id_norm: torch.Tensor,
                  titles: list, top_n: int = 5) -> None:
    """
    For each query title, find the top-N most similar movies by cosine similarity,
    using two embedding spaces:
      1. MOVIE_EMBEDDING_COMBINED — full model embedding (content + CF signal)
      2. MOVIEID_EMBEDDING        — item ID embedding only (pure CF signal)
    """
    TRUNC = 28  # max chars per cell

    def trunc(s: str) -> str:
        return s if len(s) <= TRUNC else s[:TRUNC - 1] + '…'

    def top_n_for(norm_matrix, emb_key):
        rows = []
        for title in titles:
            if title not in fs.title_to_movieId:
                rows.append((title, []))
                continue
            mid     = fs.title_to_movieId[title]
            query   = F.normalize(movie_embeddings[mid][emb_key], dim=1)
            sims    = (norm_matrix @ query.T).squeeze(-1)
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
        return rows

    def print_table(label, rows):
        seed_w = max(len(trunc(t)) for t, _ in rows)
        col_w  = TRUNC
        header = f"{'Seed':<{seed_w}}" + "".join(f"  {'#'+str(i+1):<{col_w}}" for i in range(top_n))
        print(f"\n── {label} ──")
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

    print_table("Most similar — combined embedding", top_n_for(all_norm,    'MOVIE_EMBEDDING_COMBINED'))
    print_table("Most similar — item ID embedding",  top_n_for(all_id_norm, 'MOVIEID_EMBEDDING'))



def run_probes(data_dir: str = 'data', checkpoint_path: str = None,
               version: str = 'v1') -> None:
    model, fs, movie_embeddings, all_ids, all_embs, all_norm, all_id_norm, checkpoint_path = _setup(data_dir, checkpoint_path, version)
    if model is None:
        return
    print("\n── Embedding probes ──")
    probe_genre(model, 'Horror', movie_embeddings, fs)
    probe_genre(model, 'Sci-Fi', movie_embeddings, fs)
    probe_tag(model, ['pixar', 'animation'], movie_embeddings, fs)
    probe_genome_tag(model, ['horror', 'gore', 'torture'], movie_embeddings, fs)
    probe_genome_tag(model, ['martial arts', 'kung fu'], movie_embeddings, fs)
    probe_similar(movie_embeddings, fs, all_ids, all_norm, all_id_norm, PROBE_SIMILAR_TITLES)


