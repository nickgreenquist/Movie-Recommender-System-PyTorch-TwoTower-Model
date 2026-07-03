"""
Scraped-Facet Store builder — people + structured F1 facets from the TMDB scrape (v1.5).

PURPOSE
    Distill the facet tables that let the LLM conversational front-end serve requests the two-tower
    model has no concept of: people ("Tom Hanks movies", "directed by Sofia Coppola", "scored by
    Hans Zimmer") AND the cheap structured "F1" facets already sitting in the scrape — US content
    rating, runtime, franchise/collection membership, and the TMDB vote_average quality signal.
    These are the `unsupported` request class the extraction prompt currently drops, leaving an
    empty query that falls back to popularity. See docs/llm_frontend/facet_store_plan.md
    (Expansion II, "Non-API build campaign" step A) and the residual list in
    docs/llm_frontend/validation/llm_frontend_haiku_validation.md.

    All facets come entirely from the TMDB metadata we already scraped for the LLM-feature pipeline
    (llm_features/scrape.py) but never wired in. This builder reads that cache and emits compact,
    int-keyed lookups; nothing here needs the model or an LLM call.

INPUT  (build-time only — local, gitignored, absent on Streamlit Cloud)
    llm_features/cache/scraped/{movieId}.json — one record per corpus movie (~9,366 of ~9,375).
    People — canonical TMDB person IDs live under tmdb.credits_raw, NOT the convenience tmdb.cast/
    director/writers fields (those are name-only strings):
      • actors    — credits_raw.cast[], billing-ordered by .order, capped to the top-N billed.
      • directors — credits_raw.crew[] where job == 'Director'.
      • writers   — credits_raw.crew[] where department == 'Writing'.
      • composers — credits_raw.crew[] where job == 'Original Music Composer' (folded into the
                    people store so "scored by X" resolves through the same require_people path).
    Keying by .id (not name) makes resolution unambiguous and splits same-name people for free.
    Structured facets — from the tmdb sub-dict: content rating from details_raw.release_dates
    (US MPAA certification), runtime from tmdb.runtime, franchise from
    details_raw.belongs_to_collection, and vote_average/vote_count from tmdb.

OUTPUT  (llm_features/cache/facet_store.pt — a build artifact; export bakes the SAME dict into
    serving/feature_store.pt['facets'] at export time, since the deployed app loads only serving/)
    movieId_to_people       : {mid: {'actors':[pid…], 'directors':[pid…], 'writers':[pid…],
                              'composers':[pid…]}} — actors capped to top-N billed (cameos pollute
                              "X movies"); directors/writers/composers uncapped. Filter unions all.
    person_id_to_name       : {pid: 'Tom Hanks'}            display + reverse lookup (all roles)
    person_name_to_ids      : {normalized_name: [pid…]}     BILLED (actor/director/writer) resolution;
                              pre-sorted by in-corpus film count desc (head on a collision)
    composer_name_to_ids    : {normalized_name: [pid…]}     COMPOSER-only resolution (separate namespace
                              so "movies with X" never shadows to a same-named composer)
    person_id_to_film_count : {pid: int}                    in-corpus catalog size; tie-break + display
    movieId_to_content_rating: {mid: 'G'|'PG'|'PG-13'|'R'|'NC-17'}   US MPAA cert (require_max_rating)
    movieId_to_runtime      : {mid: int}                    minutes (max_runtime/min_runtime filter)
    movieId_to_collection   : {mid: {'id': int, 'name': str}}   TMDB franchise (require/exclude_franchise)
    movieId_to_vote         : {mid: {'average': float, 'count': int}}   quality floor (min_vote_average)
    movieId_to_countries    : {mid: ['US','FR'…]}   ORIGIN (nationality) ISO 3166-1 codes, from
                                                    origin_country not production_countries (require_country, step C)
    movieId_to_language     : {mid: 'fr'}            original_language ISO 639-1 (require_language, step C)
    movieId_to_attributes   : {mid: ['black and white'…]}   curated format/attribute keys (require_attributes)
    movieId_to_keyword_concepts: {mid: ['chess','heist'…]}  curated KEYWORD_CONCEPTS content facet
                              (require/exclude_keyword_concepts hard boolean pre-filter)
    franchise_universe_aliases: {normalized_universe_phrase: [collection-name substring…]}   a small
                              curated table so "skip the MCU" (no single TMDB collection) resolves
    meta                    : build knobs + coverage counts

    All keys are int movieId / int pid. _norm_name is imported from src.llm_frontend so the
    build-time keys and the inference-time lookup normalize identically.

Usage (standalone — not part of the main.py pipeline CLI):
    python llm_features/build_facet_store.py            # build + save + deterministic spot-check
"""
import glob
import json
import os
import sys
from collections import Counter, defaultdict

import torch

# Repo root on sys.path so `from src...` resolves when run from anywhere.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.llm_frontend import (  # noqa: E402  (single normalization / format+keyword-vocab source)
    _norm_name, resolve_person, FORMAT_ATTR_KEYWORDS, KEYWORD_CONCEPTS)

# Reverse of FORMAT_ATTR_KEYWORDS ({canonical attr: [TMDB keyword name…]}) → {tmdb keyword (lower):
# canonical attr}, so a film's keyword list can be scanned to the compact attribute membership the
# format facet gates on. EXACT lowercase keyword match only (never substring — the resolver and the
# TMDB tagger both use the whole clean keyword; see facet_store_plan overturned #4).
_TMDB_KEYWORD_TO_ATTR = {
    kw.lower(): attr for attr, kws in FORMAT_ATTR_KEYWORDS.items() for kw in kws}

# Reverse of KEYWORD_CONCEPTS ({canonical concept: [TMDB keyword name…]}) → {tmdb keyword (lower):
# [concept…]}. Unlike the format map a keyword may belong to >1 concept ('casino heist' → casino + heist),
# so the value is a list. Same EXACT-lowercase discipline (the allow-lists are hand-curated to exclude homonyms).
_TMDB_KEYWORD_TO_CONCEPTS = {}
for _concept, _kws in KEYWORD_CONCEPTS.items():
    for _kw in _kws:
        _TMDB_KEYWORD_TO_CONCEPTS.setdefault(_kw.lower(), []).append(_concept)


# ── Constants ────────────────────────────────────────────────────────────────

# Top-N billed actors per film. A design knob (plan §"Top-N billed cutoff"): beyond ~10 is bit
# parts / cameos, and "Tom Hanks movies" wants films he leads, not one-scene voices. Start 10.
TOP_N_ACTORS = 10

# Valid US MPAA certifications, low→high restriction (used to pick a film's cert from the many
# noisy release_dates entries — most of which carry an empty certification).
MPAA_RATINGS = ('G', 'PG', 'PG-13', 'R', 'NC-17')

# Curated franchise "universe" aliases → member collection-name tokens. TMDB models franchises as
# per-series collections ("The Avengers Collection", "Batman Collection", …), NOT per-universe, so a
# request to exclude a whole cinematic universe ("skip the MCU", "nothing from the DCEU") has no
# single collection to match. This small explicit table expands a universe phrase into the member
# collection-name token-phrases that make it up; `_franchise_match` (src/llm_frontend.py) tests each
# as a CONTIGUOUS whole-word token subsequence of a collection name (never a raw substring — that
# over-matches: 'saw'⊂'chainsaw', 'the ring'⊂'the lord of the rings').
#
# This is a deliberately BROAD franchise-family heuristic, not a precise universe roster — two known
# limitations (universe membership is genuinely fuzzy and can't be read off collection names):
#   • Over-inclusive: a name family is grouped whole. "no DC" drops Nolan's Dark-Knight trilogy and
#     the Burton/Schumacher/Reeve Batman/Superman films too (not literally DCEU, but what a broad
#     "no Batman/Superman" intent usually wants). Entries verified against the corpus's collection
#     names; each hits only its intended family ('spider man mcu' excludes only the MCU Spider-Man
#     collection, sparing the Sony Raimi/Garfield/Spider-Verse films).
#   • Under-inclusive: a STANDALONE universe film TMDB gives no collection (The Incredible Hulk,
#     Justice League 2017, Black Widow, Black Adam) cannot be caught by any collection-name rule.
# The precise fix (curated member collection-ids + standalone movieIds) is deferred to F3; keys are
# pre-normalized (lowercase, hyphen→space) to match _norm_name.
FRANCHISE_UNIVERSE_ALIASES = {
    'marvel cinematic universe': [
        'avengers', 'iron man', 'captain america', 'thor', 'guardians of the galaxy',
        'ant man', 'doctor strange', 'black panther', 'captain marvel', 'spider man mcu',
    ],
    'dc extended universe': [
        'man of steel', 'wonder woman', 'aquaman', 'suicide squad', 'shazam',
        'batman', 'superman', 'dark knight',
    ],
}
# Common short forms resolve to the same expansion.
FRANCHISE_UNIVERSE_ALIASES['mcu']         = FRANCHISE_UNIVERSE_ALIASES['marvel cinematic universe']
FRANCHISE_UNIVERSE_ALIASES['marvel']      = FRANCHISE_UNIVERSE_ALIASES['marvel cinematic universe']
FRANCHISE_UNIVERSE_ALIASES['dceu']        = FRANCHISE_UNIVERSE_ALIASES['dc extended universe']
FRANCHISE_UNIVERSE_ALIASES['dc universe'] = FRANCHISE_UNIVERSE_ALIASES['dc extended universe']
FRANCHISE_UNIVERSE_ALIASES['dc']          = FRANCHISE_UNIVERSE_ALIASES['dc extended universe']

SCRAPED_DIR = os.path.join(REPO_ROOT, 'llm_features', 'cache', 'scraped')
OUT_PATH    = os.path.join(REPO_ROOT, 'llm_features', 'cache', 'facet_store.pt')


# ── Per-record extraction ────────────────────────────────────────────────────

def _people_from_record(record, top_n_actors=TOP_N_ACTORS):
    """Pull (role → [pid…]) and (pid → name) from one scraped record's credits_raw.

    Actors are the top-`top_n_actors` cast by billing order (.order ascending); directors are
    crew with job == 'Director'; writers are crew with department == 'Writing'; composers are crew
    with job == 'Original Music Composer'. Returns (roles, id_to_name) where roles is
    {'actors':[…],'directors':[…],'writers':[…],'composers':[…]} de-duped in billing/credit order,
    or (None, {}) if the record has no usable TMDB credits."""
    tmdb = record.get('tmdb')
    if not tmdb:
        return None, {}
    credits = tmdb.get('credits_raw') or {}
    cast = credits.get('cast') or []
    crew = credits.get('crew') or []

    id_to_name = {}

    def _collect(entries):
        pids = []
        for e in entries:
            pid, name = e.get('id'), e.get('name')
            if pid is None:
                continue
            if pid not in pids:           # de-dupe within a role (a person can hold two crew jobs)
                pids.append(pid)
            if name:
                id_to_name.setdefault(pid, name)
        return pids

    actors_sorted = sorted(cast, key=lambda c: c.get('order', 1 << 30))[:top_n_actors]
    actors    = _collect(actors_sorted)
    directors = _collect([c for c in crew if c.get('job') == 'Director'])
    writers   = _collect([c for c in crew if c.get('department') == 'Writing'])
    composers = _collect([c for c in crew if c.get('job') == 'Original Music Composer'])

    if not (actors or directors or writers or composers):
        return None, {}
    return {'actors': actors, 'directors': directors,
            'writers': writers, 'composers': composers}, id_to_name


def _us_content_rating(details_raw):
    """The film's US MPAA certification (G/PG/PG-13/R/NC-17) from TMDB release_dates, or None.

    TMDB lists many US release entries (theatrical, premiere, TV, re-release), most with an empty
    certification; take the most common valid MPAA cert across them, ties broken toward the more
    restrictive rating (a conservative ceiling filter). Toy Story → 'G'."""
    rd = details_raw.get('release_dates') or {}
    results = rd.get('results') if isinstance(rd, dict) else None
    certs = []
    for blk in results or []:
        if blk.get('iso_3166_1') != 'US':
            continue
        for e in blk.get('release_dates') or []:
            c = (e.get('certification') or '').strip().upper()
            if c in MPAA_RATINGS:
                certs.append(c)
    if not certs:
        return None
    counts = Counter(certs)
    top = max(counts.values())
    # tie → most restrictive (highest index in MPAA_RATINGS)
    return max((c for c, n in counts.items() if n == top), key=MPAA_RATINGS.index)


def _format_attrs(dr):
    """The curated format/attribute keys a film carries (Engine-1a), from details_raw.keywords.

    TMDB stores keywords as {'keywords': [{'id','name'}…]}. Scan the names for an EXACT (lowercase)
    membership in _TMDB_KEYWORD_TO_ATTR and return the de-duped set of canonical attr keys present
    ('black and white', 'woman director', 'based on book', …). Empty if the film has none of them."""
    kw = dr.get('keywords') or {}
    entries = kw.get('keywords') if isinstance(kw, dict) else (kw if isinstance(kw, list) else None)
    attrs = []
    for e in entries or []:
        name = (e.get('name') if isinstance(e, dict) else e) or ''
        attr = _TMDB_KEYWORD_TO_ATTR.get(name.strip().lower())
        if attr and attr not in attrs:
            attrs.append(attr)
    return attrs


def _keyword_concepts(dr):
    """The curated KEYWORD_CONCEPTS a film carries (chess / submarine / heist / …), from details_raw.keywords.

    Scan the keyword names for an EXACT (lowercase) membership in _TMDB_KEYWORD_TO_CONCEPTS; a keyword may map
    to >1 concept ('casino heist' → casino + heist), so accumulate the de-duped concept list. Empty if none."""
    kw = dr.get('keywords') or {}
    entries = kw.get('keywords') if isinstance(kw, dict) else (kw if isinstance(kw, list) else None)
    concepts = []
    for e in entries or []:
        name = (e.get('name') if isinstance(e, dict) else e) or ''
        for c in _TMDB_KEYWORD_TO_CONCEPTS.get(name.strip().lower(), ()):
            if c not in concepts:
                concepts.append(c)
    return concepts


def _attrs_from_record(record):
    """Pull the structured F1/step-C attribute facets from one record's tmdb sub-dict: US MPAA content
    rating (details_raw.release_dates), runtime minutes (tmdb.runtime), franchise/collection
    ({id,name} from details_raw.belongs_to_collection), the vote_average/vote_count quality signal,
    origin/nationality countries (details_raw.origin_country → ISO 3166-1 list), original language
    (details_raw.original_language), and the curated format/attribute keys. Returns a dict of only the
    fields present — a missing/unusable field is simply omitted (the filter treats its absence as 'no
    info', mirroring the year gate; format attributes are membership, so absence there means 'not that
    kind of film')."""
    tmdb = record.get('tmdb') or {}
    dr   = tmdb.get('details_raw') or {}
    out  = {}

    cert = _us_content_rating(dr)
    if cert:
        out['content_rating'] = cert

    rt = tmdb.get('runtime')
    if isinstance(rt, (int, float)) and rt > 0:
        out['runtime'] = int(rt)

    coll = dr.get('belongs_to_collection')
    if coll and coll.get('id') is not None:
        out['collection'] = {'id': int(coll['id']), 'name': coll.get('name') or ''}

    va = tmdb.get('vote_average')
    if isinstance(va, (int, float)) and va > 0:
        vc = tmdb.get('vote_count')
        out['vote'] = {'average': float(va),
                       'count':   int(vc) if isinstance(vc, (int, float)) else 0}

    # Country of ORIGIN → ISO 3166-1 alpha-2 list (nationality, require_country). We source
    # origin_country, NOT production_countries: the latter lists every co-FINANCIER, so an
    # ANY-membership require_country gate leaked US films with a sliver of foreign financing into
    # nationality queries ("japanese movies" surfaced Cliffhanger [prod FR/IT/US/JP] and Scott Pilgrim
    # [prod JP/GB/US] — both origin US). origin_country is TMDB's actual nationality field (US for
    # those, JP for genuine Japanese films), ~99% of scraped records populated. SETTING ("set in Japan") is
    # a separate genome path (require_genome_tags), untouched — see llm_frontend require_genome_tags floor.
    # Residual ceiling (inherent to TMDB origin_country + ANY-membership, ~8x smaller leak than production):
    # a few genuine legal co-productions where the queried country is a real co-origin still surface (e.g. some
    # US/DE tax-shelter films under "German"), plus rare IP-noise multi-origins (Super Mario Bros → JP) — not
    # removable without dropping true co-productions.
    oc = dr.get('origin_country')
    oc = oc if isinstance(oc, list) else []   # mirror the language line's guard (a scalar crashes the
                                              # comprehension; a bare string would char-split into ['U','S'])
    codes = [c for c in oc if isinstance(c, str) and c.strip()]
    codes = list(dict.fromkeys(codes))  # de-dupe, order-preserving
    if codes:
        out['countries'] = codes

    # Original language → ISO 639-1 (require_language / require_original_language).
    ol = dr.get('original_language')
    if isinstance(ol, str) and ol.strip():
        out['language'] = ol.strip()

    # Curated format/attribute keys (black and white / woman director / based on a book / …).
    attrs = _format_attrs(dr)
    if attrs:
        out['attributes'] = attrs

    # Curated keyword CONTENT concepts (chess / submarine / heist / …) — a hard-filter membership axis.
    concepts = _keyword_concepts(dr)
    if concepts:
        out['keyword_concepts'] = concepts
    return out


# ── Build ────────────────────────────────────────────────────────────────────

def build_facet_store(scraped_dir=SCRAPED_DIR, top_n_actors=TOP_N_ACTORS):
    """Read every scraped record and assemble the facet-store tables (see module docstring).

    person_name_to_ids lists are sorted by in-corpus film count (desc, then pid asc) so
    resolve_person can take the head on a same-name collision and get the better-covered person.
    Returns the store dict; the caller saves it (Phase 1 bakes the same tables into serving/)."""
    files = sorted(glob.glob(os.path.join(scraped_dir, '*.json')),
                   key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
    if not files:
        raise FileNotFoundError(f'no scraped JSONs under {scraped_dir} (build-time data missing)')

    movieId_to_people = {}
    person_id_to_name = {}
    pid_films         = defaultdict(set)   # pid → {mid…}, for film_count + tie-break
    billed_pids       = set()              # pids in an actor/director/writer role (the require_people namespace)
    composer_pids     = set()              # pids in a composer role (the require_composers namespace)
    movieId_to_content_rating = {}
    movieId_to_runtime        = {}
    movieId_to_collection     = {}
    movieId_to_vote           = {}
    movieId_to_countries      = {}
    movieId_to_language       = {}
    movieId_to_attributes     = {}
    movieId_to_keyword_concepts = {}
    n_no_tmdb = n_no_people = 0

    for path in files:
        with open(path) as f:
            record = json.load(f)
        mid = int(record['movieId'])

        # Structured attribute facets are independent of the credits, so collect them first — a
        # film may carry a rating/runtime/collection even if its cast/crew is unusable.
        attrs = _attrs_from_record(record)
        if 'content_rating' in attrs: movieId_to_content_rating[mid] = attrs['content_rating']
        if 'runtime'        in attrs: movieId_to_runtime[mid]        = attrs['runtime']
        if 'collection'     in attrs: movieId_to_collection[mid]     = attrs['collection']
        if 'vote'           in attrs: movieId_to_vote[mid]           = attrs['vote']
        if 'countries'      in attrs: movieId_to_countries[mid]      = attrs['countries']
        if 'language'       in attrs: movieId_to_language[mid]       = attrs['language']
        if 'attributes'     in attrs: movieId_to_attributes[mid]     = attrs['attributes']
        if 'keyword_concepts' in attrs: movieId_to_keyword_concepts[mid] = attrs['keyword_concepts']

        roles, id_to_name = _people_from_record(record, top_n_actors)
        if roles is None:
            # Disjoint buckets: no TMDB at all vs. TMDB present but no usable cast/crew.
            if record.get('tmdb') is None:
                n_no_tmdb += 1
            else:
                n_no_people += 1
            continue
        movieId_to_people[mid] = roles
        for pid, name in id_to_name.items():
            person_id_to_name.setdefault(pid, name)
        billed_role   = set(roles['actors']) | set(roles['directors']) | set(roles['writers'])
        composer_role = set(roles['composers'])
        for pid in billed_role | composer_role:
            pid_films[pid].add(mid)
        billed_pids   |= billed_role
        composer_pids |= composer_role

    person_id_to_film_count = {pid: len(mids) for pid, mids in pid_films.items()}

    # Reverse index: normalized name → [pid…], pre-sorted most-in-corpus-films first so a same-name
    # collision resolves to the better-covered person without inference-time work. Billed people
    # (actor/director/writer) and composers get SEPARATE indexes so a "movies with X" (actor intent)
    # query never resolves to a same-named composer — composers are credited on their whole uncapped
    # filmography and would otherwise out-count the (top-10-billed-capped) actor and shadow them.
    def _name_index(pids):
        idx = defaultdict(list)
        for pid in pids:
            name = person_id_to_name.get(pid)
            norm = _norm_name(name) if name else ''
            if norm:
                idx[norm].append(pid)
        return {norm: sorted(ps, key=lambda p: (-person_id_to_film_count.get(p, 0), p))
                for norm, ps in idx.items()}

    person_name_to_ids   = _name_index(billed_pids)     # require_people (actor/director/writer)
    composer_name_to_ids = _name_index(composer_pids)   # require_composers (separate namespace)

    n_total = len(files)
    n_covered = len(movieId_to_people)
    return {
        'movieId_to_people':        movieId_to_people,
        'person_id_to_name':        person_id_to_name,
        'person_name_to_ids':       person_name_to_ids,
        'composer_name_to_ids':     composer_name_to_ids,
        'person_id_to_film_count':  person_id_to_film_count,
        'movieId_to_content_rating': movieId_to_content_rating,
        'movieId_to_runtime':       movieId_to_runtime,
        'movieId_to_collection':    movieId_to_collection,
        'movieId_to_vote':          movieId_to_vote,
        'movieId_to_countries':     movieId_to_countries,
        'movieId_to_language':      movieId_to_language,
        'movieId_to_attributes':    movieId_to_attributes,
        'movieId_to_keyword_concepts': movieId_to_keyword_concepts,
        'franchise_universe_aliases': FRANCHISE_UNIVERSE_ALIASES,
        'meta': {
            'top_n_actors':     top_n_actors,
            'n_scraped_files':  n_total,
            'n_movies_covered': n_covered,
            'n_no_tmdb':        n_no_tmdb,
            'n_no_people':      n_no_people,
            'n_persons':        len(person_id_to_name),
            'n_composers':      len(composer_name_to_ids),
            'n_collisions':     sum(1 for v in person_name_to_ids.values() if len(v) > 1),
            'n_content_rating': len(movieId_to_content_rating),
            'n_runtime':        len(movieId_to_runtime),
            'n_collection':     len(movieId_to_collection),
            'n_vote':           len(movieId_to_vote),
            'n_countries':      len(movieId_to_countries),
            'n_language':       len(movieId_to_language),
            'n_attributes':     len(movieId_to_attributes),
            'n_keyword_concepts': len(movieId_to_keyword_concepts),
        },
    }


# ── Spot-check (deterministic; no LLM / no model) ────────────────────────────

def _films_for_pid(store, pid):
    """[(mid, role)…] every movie the person appears in, by scanning movieId_to_people (the
    inverse index isn't stored — compact forward tables only; this is a spot-check convenience)."""
    out = []
    for mid, roles in store['movieId_to_people'].items():
        for role in ('actors', 'directors', 'writers', 'composers'):
            if pid in roles.get(role, []):
                out.append((mid, role))
                break
    return out


def _spot_check(store, titles=None):
    """Print coverage stats and three deterministic checks: a headline actor (Tom Hanks → 31),
    a director, and an auto-found same-name collision. `titles` (mid→title) is optional display."""
    titles = titles or {}
    name_of = lambda mid: titles.get(mid, f'(movieId {mid})')
    m = store['meta']
    print('── facet store ──────────────────────────────────────────────')
    print(f"  scraped files     : {m['n_scraped_files']}")
    print(f"  movies covered    : {m['n_movies_covered']}  "
          f"({100 * m['n_movies_covered'] / m['n_scraped_files']:.1f}%)")
    print(f"  uncovered         : {m['n_no_tmdb']} no-tmdb + {m['n_no_people']} tmdb-but-no-people")
    print(f"  unique persons    : {m['n_persons']}")
    print(f"  name collisions   : {m['n_collisions']}")
    print(f"  top_n_actors      : {m['top_n_actors']}")
    n = m['n_scraped_files']
    print(f"  content ratings   : {m.get('n_content_rating', 0)}  ({100 * m.get('n_content_rating', 0) / n:.1f}%)")
    print(f"  runtimes          : {m.get('n_runtime', 0)}  ({100 * m.get('n_runtime', 0) / n:.1f}%)")
    print(f"  franchise members : {m.get('n_collection', 0)}  ({100 * m.get('n_collection', 0) / n:.1f}%)")
    print(f"  vote_average      : {m.get('n_vote', 0)}  ({100 * m.get('n_vote', 0) / n:.1f}%)")
    print(f"  countries         : {m.get('n_countries', 0)}  ({100 * m.get('n_countries', 0) / n:.1f}%)")
    print(f"  languages         : {m.get('n_language', 0)}  ({100 * m.get('n_language', 0) / n:.1f}%)")
    print(f"  format attributes : {m.get('n_attributes', 0)}  ({100 * m.get('n_attributes', 0) / n:.1f}%)")
    print(f"  keyword concepts  : {m.get('n_keyword_concepts', 0)}  ({100 * m.get('n_keyword_concepts', 0) / n:.1f}%)")
    from collections import Counter as _C
    lang_c = _C(store['movieId_to_language'].values())
    attr_c = _C(a for v in store['movieId_to_attributes'].values() for a in v)
    concept_c = _C(c for v in store['movieId_to_keyword_concepts'].values() for c in v)
    print(f"  top languages     : {lang_c.most_common(8)}")
    print(f"  attribute counts  : {dict(attr_c)}")
    print(f"  top concepts      : {concept_c.most_common(12)}")

    # Deterministic attribute spot-check on Toy Story (movieId 1): G / 81 min / Toy Story Collection.
    a = store['movieId_to_content_rating'].get(1), store['movieId_to_runtime'].get(1)
    coll = store['movieId_to_collection'].get(1)
    vote = store['movieId_to_vote'].get(1)
    print(f"\n  attrs[Toy Story]  : rating={a[0]}  runtime={a[1]}  "
          f"collection={coll['name'] if coll else None}  vote={vote['average'] if vote else None}  "
          f"countries={store['movieId_to_countries'].get(1)}  lang={store['movieId_to_language'].get(1)}")

    def report(label, name):
        pid, note = resolve_person(name, store)
        print(f"\n  {label}: {name!r} → pid={pid}  [{note}]")
        if pid is None:
            return
        disp = store['person_id_to_name'].get(pid)
        films = _films_for_pid(store, pid)
        print(f"     canonical name : {disp}   in-corpus films: "
              f"{store['person_id_to_film_count'].get(pid)}")
        for mid, role in sorted(films)[:12]:
            print(f"       · [{role[:3]}] {name_of(mid)}")
        if len(films) > 12:
            print(f"       … +{len(films) - 12} more")

    report('actor   ', 'Tom Hanks')
    report('director', 'Christopher Nolan')
    report('composer', 'Hans Zimmer')

    # Auto-find a real same-name collision (two distinct pids whose canonical names normalize
    # equal AND both spell the same) to exercise the tie-break path.
    collision = None
    for norm, pids in store['person_name_to_ids'].items():
        if len(pids) > 1:
            names = {store['person_id_to_name'][p] for p in pids}
            if len(names) == 1:        # identical spelling, different TMDB IDs — a true collision
                collision = (names.pop(), pids)
                break
    if collision:
        cname, pids = collision
        counts = [store['person_id_to_film_count'].get(p, 0) for p in pids]
        print(f"\n  collision: {cname!r} → pids {pids} films {counts}  "
              f"(resolve_person picks {pids[0]}, the most-covered)")
    else:
        print('\n  collision: none with identical spelling found')


def main():
    store = build_facet_store()
    torch.save(store, OUT_PATH)
    print(f'wrote {OUT_PATH}\n')

    # Titles for readable spot-check output, if a serving store is present (display only —
    # the facet store itself stays title-free; the export bake will use serving's title map).
    titles = {}
    fs_path = os.path.join(REPO_ROOT, 'serving', 'feature_store.pt')
    if os.path.exists(fs_path):
        fs = torch.load(fs_path, weights_only=False)
        titles = fs.get('movieId_to_title', {})
    _spot_check(store, titles)


if __name__ == '__main__':
    main()
