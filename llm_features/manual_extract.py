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
#
# The first three movies were the original schema-validation stopgap; the five that
# follow (741, 1587, 48394, 68237, 61240) are deliberately NON-blockbuster, sub-genre-
# stressing films added to exercise the v2 schema's restored granularity — every score
# on a dim like cyberpunk / clones / vampires / high_fantasy / hitman was inexpressible
# in the old 116-dim schema (folded into a coarse bucket or absent entirely).

SCORES = {
    1: {  # Toy Story (1995) — Pixar CGI landmark, family comedy, universal acclaim
        'friendship': 0.90, 'family': 0.65, 'relationships': 0.35, 'redemption': 0.30,
        'betrayal': 0.25, 'survival': 0.20, 'obsession': 0.15, 'loneliness': 0.15,
        'feel_good': 0.85, 'comedic': 0.80, 'emotional': 0.45, 'fast_paced': 0.45,
        'tense': 0.25, 'quirky': 0.20, 'scary': 0.15, 'nostalgic': 0.15,
        'small_town': 0.25,
        'oscar_winner': 0.40, 'oscar_nominated': 0.60, 'oscar_technical': 0.40,  # from fed Awards: Lasseter Special Achievement + 3 noms
        'classic': 0.70, 'big_budget': 0.50, 'imdb_top_250': 0.55, 'cult_classic': 0.10,
        'animated': 1.00, 'computer_animation': 1.00, 'cgi_heavy': 0.25,
    },
    2: {  # Jumanji (1995) — VFX-heavy family adventure, based on the picture book, mixed reviews
        'survival': 0.60, 'family': 0.55, 'coming_of_age': 0.35, 'friendship': 0.30,
        'redemption': 0.30, 'loneliness': 0.25, 'destiny': 0.15,
        'fast_paced': 0.60, 'tense': 0.55, 'feel_good': 0.50, 'comedic': 0.50,
        'scary': 0.40, 'emotional': 0.35, 'atmospheric': 0.20,
        'small_town': 0.45, 'historical': 0.20,
        'based_on_book': 0.90, 'fairy_tale': 0.30,
        'big_budget': 0.70, 'cult_classic': 0.25, 'classic': 0.10,
        'cgi_heavy': 0.70,
    },
    296: {  # Pulp Fiction (1994) — nonlinear LA crime, neo-noir, universal acclaim
        # v2: 'crime' (0.95) now splits into the specific sub-genres the film actually is.
        'crime': 0.85, 'hitman': 0.85, 'gangster': 0.55, 'murder': 0.65,
        'redemption': 0.55, 'mortality': 0.45,
        'greed': 0.30, 'betrayal': 0.30, 'addiction': 0.30, 'loneliness': 0.10,
        'dark_humor': 0.85, 'violent': 0.85, 'tense': 0.70, 'stylish': 0.70,
        'gory': 0.60, 'dark': 0.55, 'comedic': 0.50, 'disturbing': 0.40,
        'los_angeles': 0.90, 'war': 0.15,
        'nonlinear': 0.95, 'multiple_storylines': 0.90, 'independent_film': 0.45,
        'character_study': 0.25,
        'oscar_winner': 0.70, 'oscar_nominated': 0.85,  # from fed Awards: Best Original Screenplay win + 26 noms
        'classic': 0.80, 'imdb_top_250': 0.70, 'big_budget': 0.15,  # $8M indie — was a box-office-scale trap at 0.55
    },
    741: {  # Ghost in the Shell (1995) — cyberpunk anime, AI/transhumanism, philosophical
        'existentialism': 0.85, 'conspiracy': 0.55, 'mystery': 0.45, 'espionage': 0.45,
        'mortality': 0.45, 'secrets': 0.40, 'corruption': 0.35,
        'cerebral': 0.85, 'atmospheric': 0.75, 'stylish': 0.70, 'dark': 0.55,
        'tense': 0.55, 'reflective': 0.55, 'enigmatic': 0.55, 'violent': 0.50,
        'cyberpunk': 0.95, 'artificial_intelligence': 0.90, 'future': 0.90,
        'dystopia': 0.60, 'japan': 0.55, 'robots': 0.45,
        'based_on_comic': 0.90, 'foreign_language': 1.00,
        'animated': 1.00, 'anime': 1.00,
        'cult_classic': 0.65, 'classic': 0.45, 'imdb_top_250': 0.30,
    },
    1587: {  # Conan the Barbarian (1982) — sword & sorcery / high fantasy, revenge quest
        'revenge': 0.90, 'supernatural': 0.55, 'survival': 0.50, 'destiny': 0.45,
        'mortality': 0.45, 'murder': 0.35, 'sacrifice': 0.35, 'friendship': 0.30,
        'violent': 0.80, 'gory': 0.50, 'dark': 0.50, 'fast_paced': 0.50,
        'atmospheric': 0.45, 'stylish': 0.45, 'tense': 0.45,
        'high_fantasy': 0.90, 'wizards': 0.45, 'monster': 0.40, 'medieval': 0.30,
        'historical': 0.30,
        'based_on_book': 0.75,
        'cult_classic': 0.55, 'classic': 0.40, 'big_budget': 0.45,
    },
    48394: {  # Pan's Labyrinth (2006) — dark fairy tale set in post-civil-war fascist Spain
        'sacrifice': 0.65, 'mortality': 0.60, 'survival': 0.55, 'coming_of_age': 0.50,
        'social_commentary': 0.45, 'family': 0.45, 'destiny': 0.40, 'corruption': 0.35,
        'dark': 0.80, 'atmospheric': 0.85, 'disturbing': 0.60, 'bleak': 0.60,
        'violent': 0.65, 'surreal': 0.55, 'emotional': 0.55, 'creepy': 0.55,
        'melancholic': 0.55, 'tense': 0.55, 'gory': 0.45,
        'war': 0.65, 'historical': 0.70, 'monster': 0.55, 'high_fantasy': 0.35,
        'fairy_tale': 0.90, 'foreign_language': 1.00, 'character_study': 0.30,
        'classic': 0.40, 'cult_classic': 0.45,
    },
    68237: {  # Moon (2009) — hard sci-fi, lone astronaut, AI companion, clone twist
        'loneliness': 0.80, 'existentialism': 0.75, 'conspiracy': 0.65, 'mortality': 0.60,
        'secrets': 0.55, 'mystery': 0.50, 'corruption': 0.45, 'survival': 0.45,
        'cerebral': 0.80, 'melancholic': 0.70, 'atmospheric': 0.70, 'reflective': 0.65,
        'tense': 0.55, 'bleak': 0.50, 'intimate': 0.50, 'enigmatic': 0.45,
        'clones': 0.95, 'space': 0.90, 'artificial_intelligence': 0.70, 'future': 0.70,
        'dystopia': 0.40, 'robots': 0.20,
        'character_study': 0.65, 'twist_ending': 0.65,
        'cult_classic': 0.40, 'classic': 0.25,
    },
    61240: {  # Let the Right One In (2008) — child-vampire + bullied-boy, Swedish, 1982
        'coming_of_age': 0.70, 'friendship': 0.70, 'loneliness': 0.70, 'murder': 0.55,
        'mortality': 0.50, 'revenge': 0.45, 'romance': 0.45, 'secrets': 0.45,
        'survival': 0.35, 'betrayal': 0.20,
        'atmospheric': 0.80, 'dark': 0.70, 'melancholic': 0.70, 'intimate': 0.60,
        'creepy': 0.60, 'violent': 0.55, 'gory': 0.55, 'bleak': 0.55, 'disturbing': 0.55,
        'emotional': 0.55, 'tense': 0.50,
        'vampires': 0.95, 'eighties': 0.60, 'small_town': 0.45, 'monster': 0.30,
        'based_on_book': 0.90, 'foreign_language': 1.00, 'character_study': 0.40,
        'cult_classic': 0.50, 'classic': 0.35,
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
