"""
Stage 1 — Per-Movie Web Scraping  (LLM-vs-genome ablation)

Collects the raw web content the LLM will later reason over (Stage 2), one JSON
file per movie cached to llm_features/cache/scraped/{movieId}.json. The CORPUS env
selects the movie list — 'full' (default; the whole >200-ratings corpus from
data/base_movies.parquet, ~9,375 movies) or 'phase1' (the reduced >1000 list,
data/llm_experiment_movies_phase1.json, 4,461 movies). The cache is movieId-keyed
and shared across corpora, so a 'full' run scrapes only what phase1 hasn't already
cached. See docs/plans/llm_vs_genome_ablation_plan.md.

Source stack is deliberately lean — TMDB + Wikipedia, no OMDb / IMDB-page
scraping / prestige indicators (those are deferred to before the full run):

  TMDB (canonical LLM input) — curated convenience fields (overview, tagline,
      genres, full billed cast, director, writers, production companies, runtime,
      release date, vote average/count) PLUS the full raw API responses
      (details_raw, credits_raw) — nothing projected away. Two calls/movie:
      /movie/{id} (details, with append_to_response=keywords,release_dates,
      external_ids) + /movie/{id}/credits.
  Wikipedia (supplementary) — the full plaintext article extract (all sections),
      plus parsed Plot + Reception convenience fields, stored UNTRUNCATED. The
      page is resolved precisely via Wikidata from the imdbId (property P345),
      falling back to a title+year search.

Collection and truncation are deliberately SEPARATE steps. Scraping is the
expensive do-it-once stage (rate-limited, network/source dependent), so we store
as much as the sources give us and never throw data away here. The Stage 2 token
budget (raw Wikipedia plots ~2.4x the bill — ~$263 vs ~$109, see plan Cost
Budget) binds on what we FEED the LLM, not what we STORE on disk — so char/section
limits are applied in Stage 2's prompt assembly (format_for_prompt), where they
can be tuned or A/B'd in the model bake-off without ever re-scraping. Disk is
cheap (~tens of KB/movie); re-scraping 4,461 movies is not.

Caching is aggressive — a movie with an existing cache file is skipped (pass
--force to re-scrape anyway). Movies that yield no usable content at all
('failed') are NOT cached so transient outages retry on the next run; partials
(TMDB but no Wikipedia) ARE cached, since a missing Wikipedia page is usually
permanent, not transient.

Usage (standalone — not part of the main.py pipeline CLI):
    CORPUS=full   TMDB_API_KEY=your_key python llm_features/scrape.py   # whole corpus (default)
    CORPUS=phase1 TMDB_API_KEY=your_key python llm_features/scrape.py   # reduced phase1 set
    TMDB_API_KEY=your_key python llm_features/scrape.py 10              # first 10 (test)
    TMDB_API_KEY=your_key python llm_features/scrape.py 10 --force      # re-scrape, ignore cache

The optional integer arg scrapes only the first N movies; --force (-f) re-scrapes
even already-cached movies — use it when the cache schema changes mid-iteration.
The first 10 are low movieIds (Toy Story=1, Jumanji=2, …) — easy to eyeball for
the Stage 1 quality check. Get a free TMDB key at
https://www.themoviedb.org/settings/api.
"""
import json
import os
import re
import sys
import time

import pandas as pd
import requests


# ── Paths ────────────────────────────────────────────────────────────────────

REPO_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHASE1      = os.path.join(REPO_ROOT, 'data', 'llm_experiment_movies_phase1.json')
FULL_MOVIES = os.path.join(REPO_ROOT, 'data', 'base_movies.parquet')
LINKS       = os.path.join(REPO_ROOT, 'data', 'ml-32m', 'links.csv')
CACHE_DIR   = os.path.join(REPO_ROOT, 'llm_features', 'cache', 'scraped')


# ── Endpoints / client ───────────────────────────────────────────────────────

TMDB_API     = 'https://api.themoviedb.org/3'
WIKI_API     = 'https://en.wikipedia.org/w/api.php'
WIKIDATA_API = 'https://www.wikidata.org/w/api.php'

# Wikimedia's API etiquette requires a descriptive User-Agent identifying the
# tool and a contact; requests without one can be throttled or blocked.
USER_AGENT = ('MovieRecommenderResearch/1.0 (LLM-vs-genome ablation; '
              'nickgreenquist@gmail.com)')

# TMDB allows ~40 req/10s. 0.25s/request matches src/fetch_posters.py and keeps
# us comfortably under the cap even with two TMDB calls per movie. Wikipedia /
# Wikidata are separate hosts and uncapped, but we stay polite with a light gap.
TMDB_SLEEP = 0.25
WIKI_SLEEP = 0.10


# ── Scrape knobs ─────────────────────────────────────────────────────────────

# Nothing is truncated at scrape time — see the module docstring. These section
# headers feed the parsed Plot/Reception convenience fields only; full_extract
# always keeps every section. Tried in PRIORITY order (find_section sweeps all
# sections for each term before moving on), substring match, lower-cased. For
# reception we want critical sentiment, NOT box office: most articles nest '===
# Box office ===' before '=== Critical response ===' under a '== Reception =='
# parent, so a parent-first match would lead the convenience field with gross
# numbers (a quasi-popularity signal the plan flags as leakage). Preferring the
# critical subsection, falling back to the parent, fixes that.
PLOT_SECTION_TITLES      = ('plot', 'synopsis', 'premise')
RECEPTION_SECTION_TITLES = ('critical reception', 'critical response', 'reception')


# ── HTTP helpers ─────────────────────────────────────────────────────────────

def build_session() -> requests.Session:
    """Shared session carrying the Wikimedia-required User-Agent on every call."""
    session = requests.Session()
    session.headers.update({'User-Agent': USER_AGENT})
    return session


def tmdb_get(session: requests.Session, path: str, api_key: str, extra_params=None):
    """
    GET a TMDB v3 endpoint, returning parsed JSON or None on any failure. Sleeps
    TMDB_SLEEP after every request (success or not) to honour the rate limit.
    extra_params is merged into the query string (e.g. append_to_response).
    """
    params = {'api_key': api_key}
    if extra_params:
        params.update(extra_params)
    try:
        resp = session.get(f'{TMDB_API}{path}', params=params, timeout=10)
        data = resp.json() if resp.ok else None
    except Exception:
        data = None
    time.sleep(TMDB_SLEEP)
    return data


def wiki_get(session: requests.Session, url: str, params: dict):
    """GET a MediaWiki API (Wikipedia or Wikidata), returning JSON or None."""
    try:
        resp = session.get(url, params=params, timeout=10)
        data = resp.json() if resp.ok else None
    except Exception:
        data = None
    time.sleep(WIKI_SLEEP)
    return data


# ── TMDB ─────────────────────────────────────────────────────────────────────

def fetch_tmdb(session: requests.Session, tmdb_id: int, api_key: str):
    """
    Fetch one movie's TMDB content from two calls and return BOTH a curated set of
    convenience fields AND the full raw API responses (details_raw, credits_raw)
    — nothing is projected away at scrape time (mirrors Wikipedia's full_extract).
    The details call rides append_to_response to also pull keywords, release_dates
    (MPAA / per-country certifications) and external_ids in the SAME request for
    free. Returns None if details can't be fetched (movie unusable); a missing
    credits call degrades to empty cast/crew rather than failing.
    """
    details = tmdb_get(session, f'/movie/{tmdb_id}', api_key,
                       {'append_to_response': 'keywords,release_dates,external_ids'})
    if details is None:
        return None
    credits = tmdb_get(session, f'/movie/{tmdb_id}/credits', api_key) or {}

    # Full billed cast, ordered by billing — feed time picks the top-N it needs.
    cast = [c['name'] for c in sorted(credits.get('cast', []),
                                      key=lambda c: c.get('order', 1_000_000))]
    crew = credits.get('crew', [])
    director = [c['name'] for c in crew if c.get('job') == 'Director']
    writers  = _dedup([c['name'] for c in crew
                       if c.get('job') in ('Writer', 'Screenplay')])

    return {
        'title':                details.get('title'),
        'overview':             details.get('overview') or '',
        'tagline':              details.get('tagline') or '',
        'genres':               [g['name'] for g in details.get('genres', [])],
        'cast':                 cast,
        'director':             director,
        'writers':              writers,
        'production_companies': [c['name'] for c in
                                 details.get('production_companies', [])],
        'runtime':              details.get('runtime'),
        'release_date':         details.get('release_date') or '',
        'vote_average':         details.get('vote_average'),
        'vote_count':           details.get('vote_count'),
        'details_raw':          details,    # full /movie/{id} response (+appended)
        'credits_raw':          credits,    # full /movie/{id}/credits response
    }


def _dedup(items: list) -> list:
    """Drop duplicates from a list while preserving first-seen order."""
    seen, out = set(), []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _year_of(release_date: str):
    """Pull the 4-digit year from a 'YYYY-MM-DD' release date, or None."""
    if release_date and len(release_date) >= 4 and release_date[:4].isdigit():
        return int(release_date[:4])
    return None


# ── Wikipedia ────────────────────────────────────────────────────────────────

def wikidata_enwiki_title(session: requests.Session, imdb_id: str):
    """
    Resolve the English Wikipedia page title for a film from its IMDb ID, via
    Wikidata. Two hops: CirrusSearch for the item carrying P345 (IMDb ID) ==
    imdb_id, then read that item's enwiki sitelink. Precise — no title/year
    disambiguation guesswork. Returns None if no item or no enwiki page exists.
    """
    hits = wiki_get(session, WIKIDATA_API, {
        'action':   'query',
        'list':     'search',
        'srsearch': f'haswbstatement:P345={imdb_id}',
        'srlimit':  1,
        'format':   'json',
    })
    results = (hits or {}).get('query', {}).get('search', [])
    if not results:
        return None
    qid = results[0]['title']

    entity = wiki_get(session, WIKIDATA_API, {
        'action':     'wbgetentities',
        'ids':        qid,
        'props':      'sitelinks',
        'sitefilter': 'enwiki',
        'format':     'json',
    })
    sitelinks = (entity or {}).get('entities', {}).get(qid, {}).get('sitelinks', {})
    enwiki = sitelinks.get('enwiki')
    return enwiki['title'] if enwiki else None


def wikipedia_search_title(session: requests.Session, title: str, year):
    """
    Fallback page resolution: full-text search '<title> <year> film' and take the
    top hit. Less precise than the Wikidata path (can land on a disambiguation or
    the wrong adaptation), used only when the imdbId lookup comes up empty.
    """
    query = f'{title} {year} film' if year else f'{title} film'
    hits = wiki_get(session, WIKI_API, {
        'action':   'query',
        'list':     'search',
        'srsearch': query,
        'srlimit':  1,
        'format':   'json',
    })
    results = (hits or {}).get('query', {}).get('search', [])
    return results[0]['title'] if results else None


def wikipedia_extract(session: requests.Session, page_title: str):
    """
    Fetch the page's plain-text extract with wiki-style section headers retained
    (exsectionformat=wiki), so Plot/Reception can be sliced out downstream.
    Follows redirects; returns the extract string or None.
    """
    data = wiki_get(session, WIKI_API, {
        'action':         'query',
        'prop':           'extracts',
        'explaintext':    1,
        'exsectionformat': 'wiki',
        'redirects':      1,
        'titles':         page_title,
        'format':         'json',
    })
    pages = (data or {}).get('query', {}).get('pages', {})
    for page in pages.values():
        if 'extract' in page:
            return page['extract']
    return None


def parse_sections(text: str) -> list:
    """
    Locate every '== Header ==' (levels 2-6) in a wiki-format plaintext extract.
    Returns a list of {level, title (lower), start, h_end} in document order.
    """
    sections = []
    for m in re.finditer(r'(?m)^(={2,6})[ \t]*(.+?)[ \t]*\1[ \t]*$', text):
        sections.append({
            'level':  len(m.group(1)),
            'title':  m.group(2).strip().lower(),
            'start':  m.start(),
            'h_end':  m.end(),
        })
    return sections


def find_section(text: str, sections: list, wanted: tuple) -> str:
    """
    Return the body of the best-matching section, trying each term in `wanted` in
    PRIORITY order (term 0 across all sections, then term 1, …) rather than
    document order — so a specific '=== Critical response ===' subsection wins
    over the generic '== Reception ==' parent that precedes it. The body spans
    until the next header of the same-or-higher level, so a parent match still
    includes its subsections. '' if no term matches.
    """
    for term in wanted:
        for i, sec in enumerate(sections):
            if term in sec['title']:
                level = sec['level']
                end = len(text)
                for nxt in sections[i + 1:]:
                    if nxt['level'] <= level:
                        end = nxt['start']
                        break
                body = text[sec['h_end']:end].strip()
                # Flatten retained inner subsection headers ('=== Box office ==='
                # → 'Box office:') so the body reads cleanly as LLM input.
                return re.sub(r'(?m)^={2,6}[ \t]*(.+?)[ \t]*={2,6}[ \t]*$', r'\1:', body)
    return ''


def _clean(text: str) -> str:
    """Collapse runs of whitespace/newlines to single spaces for clean reading."""
    return ' '.join(text.split())


def fetch_wikipedia(session: requests.Session, imdb_id, title, year):
    """
    Resolve the movie's Wikipedia page (Wikidata-via-imdbId first, title+year
    search fallback) and return its content UNTRUNCATED: the full plaintext
    extract (all sections) plus parsed Plot + Reception convenience fields.
    Truncation is deferred to Stage 2 feed time — we keep everything on disk so
    section/char limits can change without re-scraping. None only if the page
    can't be resolved or the API returns no extract.
    """
    page_title = wikidata_enwiki_title(session, imdb_id) if imdb_id else None
    if not page_title and title:
        page_title = wikipedia_search_title(session, title, year)
    if not page_title:
        return None

    extract = wikipedia_extract(session, page_title)
    if not extract:
        return None

    sections = parse_sections(extract)
    return {
        'title':        page_title,
        'url':          'https://en.wikipedia.org/wiki/' + page_title.replace(' ', '_'),
        'plot':         _clean(find_section(extract, sections, PLOT_SECTION_TITLES)),
        'reception':    _clean(find_section(extract, sections, RECEPTION_SECTION_TITLES)),
        'full_extract': extract,
    }


# ── Per-movie orchestration ──────────────────────────────────────────────────

def scrape_movie(session, movie_id, tmdb_id, imdb_id, api_key) -> dict:
    """
    Scrape one movie across both sources and classify the result:
      full    — TMDB content AND a Wikipedia extract present
      partial — exactly one source yielded usable content
      failed  — neither did (movie unusable; caller leaves it uncached)
    """
    tmdb = fetch_tmdb(session, tmdb_id, api_key) if tmdb_id else None
    title = tmdb.get('title') if tmdb else None
    year  = _year_of(tmdb['release_date']) if tmdb else None
    wiki  = fetch_wikipedia(session, imdb_id, title, year) if (imdb_id or title) else None

    tmdb_ok = bool(tmdb and (tmdb.get('overview') or tmdb.get('genres')))
    wiki_ok = bool(wiki)   # non-None only when a full_extract was stored
    status = ('full'    if tmdb_ok and wiki_ok else
              'partial' if tmdb_ok or  wiki_ok else
              'failed')

    return {
        'movieId':   movie_id,
        'tmdbId':    tmdb_id,
        'imdbId':    imdb_id,
        'title':     title,
        'year':      year,
        'status':    status,
        'sources':   {'tmdb': tmdb_ok, 'wikipedia': wiki_ok},
        'tmdb':      tmdb,
        'wikipedia': wiki,
    }


# ── Orchestrator ─────────────────────────────────────────────────────────────

def run(limit=None, force=False) -> None:
    api_key = os.environ.get('TMDB_API_KEY', '').strip()
    if not api_key:
        print("Error: TMDB_API_KEY environment variable not set.")
        print("  Get a free key at: https://www.themoviedb.org/settings/api")
        return

    # Corpus selects the movie list (mirrors src/corpus.py): 'full' (default) = the whole
    # >200-ratings corpus from base_movies.parquet; 'phase1' = the reduced >1000 list. The scrape
    # cache is movieId-keyed and shared across corpora, so a 'full' run skips everything already
    # cached under phase1 and scrapes only the remaining tail.
    corpus = os.environ.get('CORPUS', 'full')
    if corpus not in ('full', 'phase1'):
        raise ValueError(f"Unknown CORPUS={corpus!r}; expected 'full' or 'phase1'")
    if corpus == 'phase1':
        with open(PHASE1) as f:
            movie_ids = json.load(f)['movie_ids']
    else:
        movie_ids = pd.read_parquet(FULL_MOVIES)['movieId'].astype(int).tolist()
    print(f"Corpus: {corpus}  ({len(movie_ids):,} movies in list)")

    links = pd.read_csv(LINKS, dtype={'tmdbId': 'Int64'})
    tmdb_map = {int(r.movieId): int(r.tmdbId)
                for r in links.dropna(subset=['tmdbId']).itertuples()}
    imdb_map = {int(r.movieId): f"tt{int(r.imdbId):07d}" for r in links.itertuples()}

    os.makedirs(CACHE_DIR, exist_ok=True)
    if limit is not None:
        movie_ids = movie_ids[:limit]
    n = len(movie_ids)
    mode = '  (--force: ignoring cache)' if force else ''
    print(f"Scraping {n} movies (TMDB + Wikipedia) → {CACHE_DIR}{mode}\n")

    session = build_session()
    full = partial = failed = skipped = 0

    for i, mid in enumerate(movie_ids):
        cache_path = os.path.join(CACHE_DIR, f'{mid}.json')
        if os.path.exists(cache_path) and not force:
            skipped += 1
            print(f"[{i + 1:>4}/{n}] movieId {mid:<6} skipped (cached)")
            continue

        record = scrape_movie(session, mid, tmdb_map.get(mid), imdb_map.get(mid), api_key)
        if record['status'] == 'full':
            full += 1
        elif record['status'] == 'partial':
            partial += 1
        else:
            failed += 1

        if record['status'] != 'failed':
            with open(cache_path, 'w') as f:
                json.dump(record, f, indent=2, ensure_ascii=False)

        title = record['title'] or '(no TMDB)'
        wiki = record['wikipedia']
        detail = (f"plot {len(wiki['plot'])}c recep {len(wiki['reception'])}c "
                  f"full {len(wiki['full_extract'])}c" if wiki else '')
        print(f"[{i + 1:>4}/{n}] movieId {mid:<6} {title[:38]:<38} "
              f"tmdb{'✓' if record['sources']['tmdb'] else '✗'} "
              f"wiki{'✓' if record['sources']['wikipedia'] else '✗'}  "
              f"→ {record['status']:<7} {detail}")

    print(f"\n✓ Scraped {full + partial} movies → {CACHE_DIR}")
    print(f"   full:    {full}")
    print(f"   partial: {partial}")
    print(f"   failed:  {failed}  (not cached — will retry next run)")
    print(f"   skipped: {skipped}  (already cached)")


if __name__ == '__main__':
    argv = sys.argv[1:]
    limit = next((int(a) for a in argv if a.isdigit()), None)
    force = any(a in ('--force', '-f') for a in argv)
    run(limit, force)
