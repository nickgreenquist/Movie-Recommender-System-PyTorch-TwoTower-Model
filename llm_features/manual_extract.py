"""
Stage 2 (manual stopgap) — model-in-the-loop extraction WITHOUT the API.

A temporary bridge so the rest of the pipeline (merge → tensor → similarity check)
can be exercised before the ANTHROPIC_API_KEY / Sonnet-vs-Haiku bake-off is set up.
The SCORES below were produced by the Opus 4.8 session model reading each movie's
`format_for_prompt()` feed and applying the same six-group prompts and calibration
rules the API path uses — i.e. Claude doing the extraction by hand instead of over
the wire.

This is explicitly NOT the official result:
  • one (expensive) model, not the Sonnet-vs-Haiku bake-off the experiment needs;
  • no per-model token/cost measurement;
  • only 3 movies — it CANNOT train Model B, which needs all 4,461.
Use it only to sanity-check schema/prompt quality and the downstream plumbing.
Tagged model 'manual-opus-4-8' so its cache never mixes with real API output.

Authored scores list only the NON-ZERO features per movie (the calibration rule is
"most features are 0.0"); every other field defaults to 0.0. Each group's subset is
validated against its real Pydantic model (schemas.py) before caching, so a
mis-named or out-of-range score fails loudly rather than silently corrupting the
feature tensor.

Usage (no API key needed):
    python -m llm_features.manual_extract
"""
import json
import os

from llm_features.llm_extract import GROUPS_DIR, _calibration, _top_features, cache_path, load_record
from llm_features.prompts import PROMPT_VERSION
from llm_features.schemas import FEATURE_ORDER, GROUPS

MODEL_TAG = 'manual-opus-4-8'


# ── Authored extraction (Opus 4.8 reading each feed, per-group calibration) ───
# Only non-zero features are listed; the loader fills every other dimension with
# 0.0. Reception scores are taken ONLY from evidence present in the truncated feed
# — note Pulp Fiction's Palme d'Or / Oscars and Toy Story's Oscar nods are absent
# here because the 600-char reception slice doesn't mention them (a real
# feed-truncation limitation, not a scoring miss).

SCORES = {
    1: {  # Toy Story (1995) — Pixar CGI landmark, family comedy, universal acclaim
        'friendship': 0.90, 'family': 0.65, 'relationships': 0.35, 'redemption': 0.30,
        'betrayal': 0.25, 'survival': 0.20, 'obsession': 0.15, 'loneliness': 0.15,
        'feel_good': 0.85, 'comedic': 0.80, 'emotional': 0.45, 'fast_paced': 0.45,
        'tense': 0.25, 'quirky': 0.20, 'scary': 0.15, 'nostalgic': 0.15,
        'small_town': 0.25,
        'classic': 0.70, 'box_office_scale': 0.70, 'imdb_top_250': 0.55, 'cult_classic': 0.10,
        'animated': 1.00, 'computer_animation': 1.00, 'cgi_heavy': 0.25,
    },
    2: {  # Jumanji (1995) — VFX-heavy family adventure, based on the picture book, mixed reviews
        'survival': 0.60, 'family': 0.55, 'coming_of_age': 0.35, 'friendship': 0.30,
        'redemption': 0.30, 'loneliness': 0.25, 'destiny': 0.15,
        'fast_paced': 0.60, 'tense': 0.55, 'feel_good': 0.50, 'comedic': 0.50,
        'scary': 0.40, 'emotional': 0.35, 'atmospheric': 0.20,
        'small_town': 0.45, 'historical': 0.20,
        'based_on_book': 0.90, 'fairy_tale': 0.30,
        'box_office_scale': 0.65, 'cult_classic': 0.25, 'classic': 0.10,
        'cgi_heavy': 0.70,
    },
    296: {  # Pulp Fiction (1994) — nonlinear LA crime, neo-noir, universal acclaim
        'crime': 0.95, 'murder': 0.70, 'redemption': 0.55, 'mortality': 0.45,
        'greed': 0.30, 'betrayal': 0.30, 'addiction': 0.30, 'loneliness': 0.10,
        'dark_humor': 0.85, 'violent': 0.85, 'tense': 0.70, 'stylish': 0.70,
        'gory': 0.60, 'dark': 0.55, 'comedic': 0.50, 'disturbing': 0.40,
        'los_angeles': 0.90, 'war': 0.15,
        'nonlinear': 0.95, 'multiple_storylines': 0.90, 'independent_film': 0.45,
        'character_study': 0.25,
        'classic': 0.80, 'imdb_top_250': 0.70, 'box_office_scale': 0.55,
    },
}


# ── Validate + cache (same format the API path writes) ───────────────────────

def ingest(movie_id: int, scores: dict) -> dict:
    """
    Split a movie's flat score dict into the six groups, default missing fields to
    0.0, validate each subset against its real Pydantic model, and write per-group
    cache files identical to llm_extract's API path. Returns the merged 116-dim dict.
    """
    unknown = set(scores) - set(FEATURE_ORDER)
    if unknown:
        raise ValueError(f"movie {movie_id}: unknown feature(s) {sorted(unknown)}")

    merged = {}
    for group in GROUPS:
        subset = {name: float(scores.get(name, 0.0)) for name in group['dim_names']}
        validated = group['model'](**subset).model_dump()   # raises on range/shape error

        path = cache_path(group['key'], MODEL_TAG, movie_id)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump({
                'movieId':        movie_id,
                'group':          group['key'],
                'model':          MODEL_TAG,
                'prompt_version': PROMPT_VERSION,
                'features':       validated,
                'usage':          {},   # no tokens — hand extraction
            }, f, indent=2)
        merged.update(validated)
    return merged


def run() -> None:
    print(f"Manual extraction ({MODEL_TAG}) — {len(SCORES)} movies × {len(GROUPS)} groups "
          f"→ {os.path.relpath(GROUPS_DIR, os.path.dirname(GROUPS_DIR))}/<group>/{MODEL_TAG}/\n")
    print("⚠  Stopgap only: one model, no cost measurement, 3 movies — NOT the bake-off.\n")

    for movie_id, scores in SCORES.items():
        rec = load_record(movie_id)
        title = f"{rec.get('title')} ({rec.get('year')})" if rec else f"movieId {movie_id}"
        merged = ingest(movie_id, scores)
        c = _calibration(merged)
        print(f"■ {title}")
        print(f"   calibration: max={c['max']:.2f} mean={c['mean']:.3f} "
              f"zeros={c['zeros']}/{len(FEATURE_ORDER)} ≈0.5={c['near_half']} ≥0.5={c['high']}")
        print("   top: " + ', '.join(f"{n}={v:.2f}" for n, v in _top_features(merged, thresh=0.3, k=14)))
        print()

    print(f"✓ Wrote per-group cache for {len(SCORES)} movies under model '{MODEL_TAG}'")


if __name__ == '__main__':
    run()
