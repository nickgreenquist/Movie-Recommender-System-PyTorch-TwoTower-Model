"""
Scraped-Facet Store builder — people facets from the TMDB credits scrape (v1.5).

PURPOSE
    Distill the person→movie facet tables that let the LLM conversational front-end serve
    people-facet requests ("Tom Hanks movies", "directed by Sofia Coppola") — the entire
    `unsupported` request class the extraction prompt currently drops, leaving an empty query
    that falls back to popularity. See docs/llm_frontend/facet_store_plan.md (this builder is Phase 0)
    and the residual list in docs/llm_frontend/validation/llm_frontend_haiku_validation.md.

    The two-tower model has no actor/director concept; these facets come entirely from the TMDB
    credits we already scraped for the LLM-feature pipeline (llm_features/scrape.py) but never
    wired in. This builder reads that cache and emits compact, int-keyed lookups.

INPUT  (build-time only — local, gitignored, absent on Streamlit Cloud)
    llm_features/cache/scraped/{movieId}.json — one record per corpus movie (~9,366 of ~9,375).
    Canonical TMDB person IDs live under tmdb.credits_raw, NOT the convenience tmdb.cast/
    director/writers fields (those are name-only strings):
      • actors    — credits_raw.cast[], billing-ordered by .order, capped to the top-N billed.
      • directors — credits_raw.crew[] where job == 'Director'.
      • writers   — credits_raw.crew[] where department == 'Writing'.
    Keying by .id (not name) makes resolution unambiguous and splits same-name people for free.

OUTPUT  (llm_features/cache/facet_store.pt — a build artifact; Phase 1 will bake the SAME tables
    into serving/feature_store.pt at export time, since the deployed app loads only serving/)
    movieId_to_people      : {mid: {'actors':[pid…], 'directors':[pid…], 'writers':[pid…]}}
                             actors capped to top-N billed (cameos pollute "X movies");
                             directors/writers uncapped. The filter path unions all three.
    person_id_to_name      : {pid: 'Tom Hanks'}            display + reverse lookup
    person_name_to_ids     : {normalized_name: [pid…]}     resolution; pre-sorted by in-corpus
                             film count desc (so resolve_person takes the head on a collision)
    person_id_to_film_count: {pid: int}                    in-corpus catalog size; tie-break + display
    meta                   : build knobs + coverage counts

    All keys are int movieId / int pid. _norm_name is imported from src.llm_frontend so the
    build-time keys and the inference-time lookup normalize identically.

Usage (standalone — not part of the main.py pipeline CLI):
    python llm_features/build_facet_store.py            # build + save + deterministic spot-check
"""
import glob
import json
import os
import sys
from collections import defaultdict

import torch

# Repo root on sys.path so `from src...` resolves when run from anywhere.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.llm_frontend import _norm_name, resolve_person  # noqa: E402  (single normalization source)


# ── Constants ────────────────────────────────────────────────────────────────

# Top-N billed actors per film. A design knob (plan §"Top-N billed cutoff"): beyond ~10 is bit
# parts / cameos, and "Tom Hanks movies" wants films he leads, not one-scene voices. Start 10.
TOP_N_ACTORS = 10

SCRAPED_DIR = os.path.join(REPO_ROOT, 'llm_features', 'cache', 'scraped')
OUT_PATH    = os.path.join(REPO_ROOT, 'llm_features', 'cache', 'facet_store.pt')


# ── Per-record extraction ────────────────────────────────────────────────────

def _people_from_record(record, top_n_actors=TOP_N_ACTORS):
    """Pull (role → [pid…]) and (pid → name) from one scraped record's credits_raw.

    Actors are the top-`top_n_actors` cast by billing order (.order ascending); directors are
    crew with job == 'Director'; writers are crew with department == 'Writing'. Returns
    (roles, id_to_name) where roles is {'actors':[…],'directors':[…],'writers':[…]} de-duped in
    billing/credit order, or (None, {}) if the record has no usable TMDB credits."""
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

    if not (actors or directors or writers):
        return None, {}
    return {'actors': actors, 'directors': directors, 'writers': writers}, id_to_name


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
    n_no_tmdb = n_no_people = 0

    for path in files:
        with open(path) as f:
            record = json.load(f)
        mid = int(record['movieId'])
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
        for pid in set(roles['actors']) | set(roles['directors']) | set(roles['writers']):
            pid_films[pid].add(mid)

    person_id_to_film_count = {pid: len(mids) for pid, mids in pid_films.items()}

    # Reverse index: normalized name → [pid…], pre-sorted most-in-corpus-films first so a
    # same-name collision resolves to the better-covered person without inference-time work.
    name_to_ids = defaultdict(list)
    for pid, name in person_id_to_name.items():
        norm = _norm_name(name)
        if norm:
            name_to_ids[norm].append(pid)
    person_name_to_ids = {
        norm: sorted(pids, key=lambda p: (-person_id_to_film_count.get(p, 0), p))
        for norm, pids in name_to_ids.items()
    }

    n_total = len(files)
    n_covered = len(movieId_to_people)
    return {
        'movieId_to_people':       movieId_to_people,
        'person_id_to_name':       person_id_to_name,
        'person_name_to_ids':      person_name_to_ids,
        'person_id_to_film_count': person_id_to_film_count,
        'meta': {
            'top_n_actors':     top_n_actors,
            'n_scraped_files':  n_total,
            'n_movies_covered': n_covered,
            'n_no_tmdb':        n_no_tmdb,
            'n_no_people':      n_no_people,
            'n_persons':        len(person_id_to_name),
            'n_collisions':     sum(1 for v in person_name_to_ids.values() if len(v) > 1),
        },
    }


# ── Spot-check (deterministic; no LLM / no model) ────────────────────────────

def _films_for_pid(store, pid):
    """[(mid, role)…] every movie the person appears in, by scanning movieId_to_people (the
    inverse index isn't stored — compact forward tables only; this is a spot-check convenience)."""
    out = []
    for mid, roles in store['movieId_to_people'].items():
        for role in ('actors', 'directors', 'writers'):
            if pid in roles[role]:
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
