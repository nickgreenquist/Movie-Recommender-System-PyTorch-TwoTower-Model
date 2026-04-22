"""
Precompute movie poster URLs from TMDB API.

Run once:
    TMDB_API_KEY=your_key python main.py posters

Saves serving/posters.json: {"<movieId>": "<poster_url>", ...}
Empty string means the movie has no poster on TMDB.
Safe to interrupt and resume — skips already-fetched entries.

Get a free API key at: https://www.themoviedb.org/settings/api
"""
import json
import os
import time

import pandas as pd
import requests

TMDB_IMAGE_BASE = 'https://image.tmdb.org/t/p/w342'
POSTER_FILE = 'serving/posters.json'


def run_fetch_posters(data_dir: str = 'data') -> None:
    api_key = os.environ.get('TMDB_API_KEY', '').strip()
    if not api_key:
        print("Error: TMDB_API_KEY environment variable not set.")
        print("  Get a free key at: https://www.themoviedb.org/settings/api")
        return

    import torch
    fs = torch.load('serving/feature_store.pt', weights_only=False)
    corpus_ids = set(fs['top_movies'])

    links_path = os.path.join(data_dir, 'ml-32m', 'links.csv')
    links = pd.read_csv(links_path, dtype={'tmdbId': 'Int64'})
    links = links.dropna(subset=['tmdbId'])
    links = links[links['movieId'].isin(corpus_ids)]
    tmdb_map = {int(row.movieId): int(row.tmdbId) for row in links.itertuples()}

    os.makedirs('serving', exist_ok=True)
    if os.path.exists(POSTER_FILE):
        with open(POSTER_FILE) as f:
            posters = json.load(f)
        print(f"Resuming — {len(posters)} entries already cached.")
    else:
        posters = {}

    movie_ids = list(tmdb_map.keys())
    n = len(movie_ids)
    found = skipped = no_poster = errors = 0

    print(f"Fetching posters for {n} corpus movies (0.25s/request ≈ {n * 0.25 / 60:.0f} min) ...")
    from tqdm import tqdm
    for i, mid in enumerate(tqdm(movie_ids, desc="Fetching posters")):
        key = str(mid)
        if posters.get(key, '').startswith('http'):
            skipped += 1
            continue

        tmdb_id = tmdb_map[mid]
        try:
            resp = requests.get(
                f'https://api.themoviedb.org/3/movie/{tmdb_id}',
                params={'api_key': api_key},
                timeout=10,
            )
            if resp.ok:
                path = resp.json().get('poster_path')
                if path:
                    posters[key] = TMDB_IMAGE_BASE + path
                    found += 1
                else:
                    posters[key] = ''
                    no_poster += 1
            else:
                posters[key] = None  # transient error — retry on next run
                errors += 1
                if resp.status_code != 404:
                    tqdm.write(f"  HTTP {resp.status_code} for tmdbId={tmdb_id} (movieId={mid})")
        except Exception as e:
            posters[key] = None  # transient error — retry on next run
            errors += 1
            tqdm.write(f"  Error for tmdbId={tmdb_id}: {e}")

        if (i + 1) % 200 == 0:
            with open(POSTER_FILE, 'w') as f:
                json.dump(posters, f)

        time.sleep(0.25)

    with open(POSTER_FILE, 'w') as f:
        json.dump(posters, f)

    total = found + no_poster + errors
    print(f"\nDone. {found}/{total} posters found, {no_poster} no poster, {errors} errors, {skipped} skipped.")
    print(f"Saved to {POSTER_FILE}")
