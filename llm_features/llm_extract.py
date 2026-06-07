"""
Stage 2 (extraction) — Grouped structured feature extraction  (LLM-vs-genome ablation)

Runs the six grouped extraction calls per movie against Claude, validating each call
against its Pydantic group schema (schemas.py) via structured output, and caches the
per-(group, model, movie) result. This is the spot-check / bake-off driver: the
default run is the small calibration test the plan gates the full $100 run behind —
3 recognisable movies through BOTH Sonnet 4.6 and Haiku 4.5, with calibration stats
and a measured-token cost projection so the cheaper model can be chosen empirically.
See docs/plans/llm_vs_genome_ablation_plan.md (Stage 2, "LLM choice — test, don't
assume" + "Validation pass before scaling").

Design notes:
  • Structured output via client.messages.parse(output_format=<group model>): every
    field is present and validated to float[0,1] CLIENT-SIDE by Pydantic (the wire
    schema can't bound numbers — see schemas.py). parsed_output is the validated
    model instance.
  • format_for_prompt feeds only the DISCRIMINATIVE slice of the scrape (title, year,
    genres, MPAA, studio, director, writers, top cast, keywords, production budget,
    tagline, overview, and TRUNCATED Wikipedia plot + reception + Accolades/awards) —
    NOT the full below-the-line crew or the raw full_extract. The awards block is
    sliced out of full_extract here (the reception convenience field is the
    critical-response subsection only, which omits the award facts). Truncation lives
    here, at feed time, never in the durable scrape cache (store-raw / truncate-at-feed).
  • thinking is DISABLED: output tokens dominate the bill, and structured float
    extraction needs no chain-of-thought.
  • cache_control is set on the static system prefix, but the per-group prompt is
    ~500 tokens — BELOW the 2048/4096-tok cacheable minimum — so caching is a
    no-op here (cache_read stays 0). It's harmless and future-proofs a longer prompt;
    we print the cache-read count so the no-op is visible rather than assumed.

Usage (standalone — needs ANTHROPIC_API_KEY; ~$pennies for the default 3-movie test):
    ANTHROPIC_API_KEY=sk-... python llm_features/llm_extract.py            # Toy Story, Jumanji, Pulp Fiction
    ANTHROPIC_API_KEY=sk-... python llm_features/llm_extract.py 1 2 296    # specific movieIds
    ANTHROPIC_API_KEY=sk-... python llm_features/llm_extract.py --force    # ignore cache, re-call
"""
import json
import os
import sys
import time

import anthropic

from llm_features.prompts import PROMPT_VERSION, SYSTEM_PROMPTS, user_message
from llm_features.schemas import FEATURE_ORDER, GROUPS
from llm_features.scrape import find_section, parse_sections


# ── Paths ────────────────────────────────────────────────────────────────────

REPO_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHASE1      = os.path.join(REPO_ROOT, 'data', 'llm_experiment_movies_phase1.json')
SCRAPED_DIR = os.path.join(REPO_ROOT, 'llm_features', 'cache', 'scraped')
GROUPS_DIR  = os.path.join(REPO_ROOT, 'llm_features', 'cache', 'llm_groups')

DEFAULT_TEST_MOVIES = [1, 2, 296]   # Toy Story, Jumanji, Pulp Fiction — easy to eyeball


# ── Bake-off models + pricing ────────────────────────────────────────────────

# (input $/MTok, output $/MTok) — current pay-as-you-go pricing. Output dominates;
# Haiku output is 3× cheaper than Sonnet, which is the whole reason for the bake-off.
MODELS = [
    ('claude-sonnet-4-6', 'Sonnet 4.6', 3.0, 15.0),
    ('claude-haiku-4-5',  'Haiku 4.5',  1.0,  5.0),
]

MAX_TOKENS = 1024   # a 6–28 float JSON object is small; this is generous headroom


# ── Feed-time formatting (discriminative slice only) ─────────────────────────

PLOT_CHARS      = 1500   # Wikipedia plot cap — full plots blow the token budget
RECEPTION_CHARS = 600    # critical-response cap — sentiment, not the award facts
AWARDS_CHARS    = 600    # Accolades/awards-section cap — the discrete prestige facts
MAX_CAST        = 8      # top-billed only; the long tail adds tokens, not signal


def _us_certification(details_raw: dict) -> str:
    """First non-empty US MPAA certification from TMDB release_dates, or ''."""
    for entry in (details_raw or {}).get('release_dates', {}).get('results', []):
        if entry.get('iso_3166_1') == 'US':
            for rd in entry.get('release_dates', []):
                if rd.get('certification'):
                    return rd['certification']
    return ''


def _keywords(details_raw: dict) -> list:
    """TMDB keyword names (highly discriminative for themes/setting/provenance)."""
    kw = (details_raw or {}).get('keywords', {})
    return [k['name'] for k in kw.get('keywords', kw.get('results', []))]


def _budget(details_raw: dict) -> str:
    """Production budget as '$Xm' — the factual basis for the reception group's
    big_budget dim. Gross revenue is deliberately NOT fed: 'big budget' means
    production scale, not commercial take, and gross is a quasi-popularity signal in
    tension with the alpha=0 stance (the experiment holds popularity constant)."""
    v = (details_raw or {}).get('budget') or 0
    return f"${v / 1e6:.0f}M" if v > 0 else ''


def _awards(wiki: dict) -> str:
    """
    The Wikipedia Accolades / awards-season section, sliced from the stored
    full_extract at feed time. The parsed `reception` convenience field is the
    Critical-response subsection ONLY — the discrete award facts (Oscars, Palme d'Or,
    Criterion) live in a SEPARATE Accolades section it doesn't capture, so the reception
    group's prestige dims were unscoreable without this. Reuses scrape.py's section
    parser (store-raw / slice-at-feed rule → no re-scrape; full_extract is cached).
    """
    full = (wiki or {}).get('full_extract') or ''
    if not full:
        return ''
    body = find_section(full, parse_sections(full), ('accolades', 'awards'))
    return body[:AWARDS_CHARS] + ('…' if len(body) > AWARDS_CHARS else '')


def format_for_prompt(record: dict) -> str:
    """
    Build the compact, labelled text block fed to the extraction model from a scraped
    record — only the discriminative fields, with Wikipedia text truncated. Degrades
    gracefully when TMDB is missing (the 17 Wikipedia-only partials).
    """
    tmdb = record.get('tmdb') or {}
    wiki = record.get('wikipedia') or {}
    details = tmdb.get('details_raw') or {}

    title = record.get('title') or tmdb.get('title') or '(unknown)'
    year  = record.get('year')
    lines = [f"Title: {title}" + (f" ({year})" if year else '')]

    def add(label, value):
        if value:
            lines.append(f"{label}: {value}")

    add('Genres',   ', '.join(tmdb.get('genres', [])))
    add('MPAA',     _us_certification(details))
    add('Studio',   ', '.join(tmdb.get('production_companies', [])[:4]))
    add('Director', ', '.join(tmdb.get('director', [])))
    add('Writers',  ', '.join(tmdb.get('writers', [])[:4]))
    add('Cast',     ', '.join(tmdb.get('cast', [])[:MAX_CAST]))
    add('Keywords',   ', '.join(_keywords(details)[:25]))
    add('Budget',     _budget(details))
    add('Tagline',  tmdb.get('tagline'))
    add('Overview', tmdb.get('overview'))

    plot = (wiki.get('plot') or '')[:PLOT_CHARS]
    add('Plot', plot + ('…' if wiki.get('plot', '') and len(wiki['plot']) > PLOT_CHARS else ''))
    recep = (wiki.get('reception') or '')[:RECEPTION_CHARS]
    add('Reception', recep + ('…' if wiki.get('reception', '') and len(wiki['reception']) > RECEPTION_CHARS else ''))
    add('Awards', _awards(wiki))

    return '\n'.join(lines)


# ── One grouped call ─────────────────────────────────────────────────────────

def extract_group(client, model_id: str, group: dict, content: str):
    """
    One structured-output call: the group's static system prefix (cached marker) +
    the per-movie content, validated against the group's Pydantic model. Returns
    (feature_dict, usage). Raises on API / validation failure (caller handles).
    """
    resp = client.messages.parse(
        model=model_id,
        max_tokens=MAX_TOKENS,
        thinking={'type': 'disabled'},
        system=[{
            'type':          'text',
            'text':          SYSTEM_PROMPTS[group['key']],
            'cache_control': {'type': 'ephemeral'},
        }],
        messages=[{'role': 'user', 'content': user_message(content)}],
        output_format=group['model'],
    )
    if resp.parsed_output is None:
        raise ValueError(f"no parsed output (stop_reason={resp.stop_reason})")
    return resp.parsed_output.model_dump(), resp.usage


def cache_path(group_key: str, model_id: str, movie_id: int) -> str:
    return os.path.join(GROUPS_DIR, group_key, model_id, f'{movie_id}.json')


def extract_movie(client, model_id: str, record: dict, force: bool):
    """
    Run all six groups for one movie under one model. Returns (merged_features,
    usages) where merged_features is the 116-dim dict and usages is the per-group
    usage list (cached groups contribute their stored usage). Per-group results are
    cached to cache/llm_groups/{group}/{model}/{movieId}.json.
    """
    movie_id = record['movieId']
    content = format_for_prompt(record)
    merged, usages = {}, []

    for group in GROUPS:
        path = cache_path(group['key'], model_id, movie_id)
        if os.path.exists(path) and not force:
            blob = json.load(open(path))
            merged.update(blob['features'])
            usages.append(blob.get('usage', {}))
            continue

        features, usage = extract_group(client, model_id, group, content)
        usage_d = {
            'input_tokens':                usage.input_tokens,
            'output_tokens':               usage.output_tokens,
            'cache_read_input_tokens':     getattr(usage, 'cache_read_input_tokens', 0) or 0,
            'cache_creation_input_tokens': getattr(usage, 'cache_creation_input_tokens', 0) or 0,
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump({
                'movieId':        movie_id,
                'group':          group['key'],
                'model':          model_id,
                'prompt_version': PROMPT_VERSION,
                'features':       features,
                'usage':          usage_d,
            }, f, indent=2)
        merged.update(features)
        usages.append(usage_d)

    return merged, usages


# ── Reporting ────────────────────────────────────────────────────────────────

def _cost(usages: list, in_price: float, out_price: float) -> float:
    """Dollar cost of a list of usage dicts (cache-aware, though caching is a no-op)."""
    total = 0.0
    for u in usages:
        billed_in = (u['input_tokens']
                     + u['cache_creation_input_tokens'] * 1.25
                     + u['cache_read_input_tokens'] * 0.10)
        total += billed_in / 1e6 * in_price + u['output_tokens'] / 1e6 * out_price
    return total


def _calibration(features: dict) -> dict:
    """Distribution stats over the 116 scores — the 0.5-defaulting / range-use tell."""
    vals = [features[k] for k in FEATURE_ORDER if k in features]
    n = len(vals) or 1
    return {
        'zeros':     sum(v == 0.0 for v in vals),
        'near_half': sum(0.45 <= v <= 0.55 for v in vals),   # defaulting tell
        'high':      sum(v >= 0.5 for v in vals),
        'mean':      sum(vals) / n,
        'max':       max(vals) if vals else 0.0,
    }


def _top_features(features: dict, thresh: float = 0.4, k: int = 12) -> list:
    """Top-scoring features ≥ thresh, for eyeballing intuition."""
    hits = sorted(((v, name) for name, v in features.items() if v >= thresh), reverse=True)
    return [(name, v) for v, name in hits[:k]]


def report(results: dict, movie_titles: dict) -> None:
    """results[model_id][movie_id] = (merged, usages). Print calibration + cost."""
    print("\n" + "═" * 78)
    print("BAKE-OFF RESULTS  —  per-movie top features, calibration, projected cost")
    print("═" * 78)

    for movie_id, title in movie_titles.items():
        print(f"\n■ {title}  (movieId {movie_id})")
        for model_id, label, *_ in MODELS:
            if model_id not in results or movie_id not in results[model_id]:
                continue
            merged, _ = results[model_id][movie_id]
            c = _calibration(merged)
            tops = _top_features(merged)
            print(f"   ── {label}: "
                  f"max={c['max']:.2f} mean={c['mean']:.3f} "
                  f"zeros={c['zeros']}/{len(FEATURE_ORDER)} "
                  f"≈0.5={c['near_half']} ≥0.5={c['high']}")
            print("      top: " + ', '.join(f"{n}={v:.2f}" for n, v in tops))

        # Cross-model divergence on the shared 116-vector.
        if all(movie_id in results.get(m[0], {}) for m in MODELS):
            a = results[MODELS[0][0]][movie_id][0]
            b = results[MODELS[1][0]][movie_id][0]
            mad = sum(abs(a[k] - b[k]) for k in FEATURE_ORDER) / len(FEATURE_ORDER)
            print(f"   ── Sonnet↔Haiku mean abs diff: {mad:.3f}")

    # ── Cost: measured + projected to the full Phase 1 corpus ────────────────
    n_corpus = json.load(open(PHASE1))['n_movies']
    print("\n" + "─" * 78)
    print(f"COST  (measured on this run; projected = per-movie × {n_corpus:,} Phase-1 movies)")
    for model_id, label, in_p, out_p in MODELS:
        if model_id not in results:
            continue
        all_usages, n_movies = [], 0
        for movie_id, (_, usages) in results[model_id].items():
            all_usages += usages
            n_movies += 1
        measured = _cost(all_usages, in_p, out_p)
        in_tok  = sum(u['input_tokens'] for u in all_usages)
        out_tok = sum(u['output_tokens'] for u in all_usages)
        per_movie = measured / (n_movies or 1)
        print(f"   {label:<11} {n_movies} movies × 6 groups: "
              f"in={in_tok:,} out={out_tok:,} tok  "
              f"${measured:.4f} measured  →  ${per_movie * n_corpus:,.2f} full Phase 1")


# ── Orchestrator ─────────────────────────────────────────────────────────────

def load_record(movie_id: int) -> dict:
    path = os.path.join(SCRAPED_DIR, f'{movie_id}.json')
    if not os.path.exists(path):
        return None
    return json.load(open(path))


def run(movie_ids: list, force: bool) -> None:
    api_key = os.environ.get('ANTHROPIC_API_KEY', '').strip()
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set.")
        print("  Get a key at: https://console.anthropic.com/settings/keys")
        return

    client = anthropic.Anthropic()

    records = {}
    for mid in movie_ids:
        rec = load_record(mid)
        if rec is None:
            print(f"  ⚠ movieId {mid} not scraped — skipping")
            continue
        records[mid] = rec

    movie_titles = {mid: f"{r.get('title')} ({r.get('year')})" for mid, r in records.items()}
    print(f"Extracting {len(records)} movies × {len(GROUPS)} groups × {len(MODELS)} models "
          f"= {len(records) * len(GROUPS) * len(MODELS)} calls"
          f"{'  (--force: ignoring cache)' if force else ''}\n")

    results = {}
    for model_id, label, *_ in MODELS:
        results[model_id] = {}
        for mid, rec in records.items():
            t0 = time.time()
            try:
                merged, usages = extract_movie(client, model_id, rec, force)
            except Exception as e:
                print(f"  ✗ {label:<11} {movie_titles[mid]:<28} FAILED: {e}")
                continue
            results[model_id][mid] = (merged, usages)
            out_tok = sum(u['output_tokens'] for u in usages)
            print(f"  ✓ {label:<11} {movie_titles[mid]:<28} "
                  f"{len(merged)} feats, {out_tok:>4} out-tok, {time.time() - t0:4.1f}s")

    report(results, movie_titles)


if __name__ == '__main__':
    argv = sys.argv[1:]
    force = any(a in ('--force', '-f') for a in argv)
    ids = [int(a) for a in argv if a.isdigit()] or DEFAULT_TEST_MOVIES
    run(ids, force)
