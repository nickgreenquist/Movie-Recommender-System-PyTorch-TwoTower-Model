"""
Stage 2 (Claude-Code extraction) — Phase-0 feature extraction WITHOUT the API
                                    (LLM-vs-genome ablation)

A bridge that lets a Claude Code session (Sonnet — via subagents or parallel terminals)
do the six-group extraction the API path (llm_extract.py) will later do over the wire,
so a full-corpus Model B can be trained NOW, before ANTHROPIC_API_KEY / the paid bake-off
is set up. It is a DRY-RUN yielding a provisional, OPTIMISTIC Model B (Claude-Code Sonnet
WITH thinking is likely stronger than API Sonnet with thinking disabled, and it carries no
per-call cost measurement) — the official, cost-measured result still comes from the API
bake-off. See docs/plans/llm_vs_genome_ablation_plan.md.

Namespacing: writes under model tag 'claude-code-sonnet' →
cache/llm_groups/<group>/claude-code-sonnet/<movieId>.json — physically separate from the
API tags ('claude-sonnet-4-6' / 'claude-haiku-4-5'), so neither path overwrites the other
and build_features.py can pick which tag to turn into a tensor.

It reuses the EXACT artifacts of the API path — format_for_prompt (the feed), the six
grouped system prompts (prompts.SYSTEM_PROMPTS), and the Pydantic group schemas
(schemas.py) — and validates every ingested movie against those schemas. So the only thing
that differs from the API run is WHO produced the scores; the features are structurally
identical and the API result drops in later as a clean swap.

This module is the reusable CORE (feed_for / system_prompt_for / ingest / list_remaining);
the actual scoring is done by a Claude Code agent that reads a movie's feed + the six group
prompts and returns scores, which ingest() validates and caches. The orchestration (fan out
Sonnet subagents, or run parallel terminals over movie-ID ranges) lives outside this file;
the skip-if-cached `list_remaining` makes either approach resumable and collision-free.

Usage (standalone — no API / GPU):
    python -m llm_features.cc_extract --remaining        # how many movies left for the tag
    python -m llm_features.cc_extract --show 741         # one movie's full extraction task
"""
import json
import os
import sys

from llm_features.llm_extract import cache_path, format_for_prompt, load_record
from llm_features.prompts import PROMPT_VERSION, SYSTEM_PROMPTS, user_message
from llm_features.schemas import FEATURE_ORDER, GROUPS


# ── Paths / tag ───────────────────────────────────────────────────────────────

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHASE1    = os.path.join(REPO_ROOT, 'data', 'llm_experiment_movies_phase1.json')

MODEL_TAG = 'claude-code-sonnet'


# ── The extraction task (what an agent/terminal needs) ────────────────────────

def feed_for(movie_id: int):
    """The discriminative feed an extractor reads — identical to the API path. None if
    the movie was never scraped."""
    rec = load_record(movie_id)
    return format_for_prompt(rec) if rec else None


def system_prompt_for(group_key: str) -> str:
    """The exact static system prompt the API would send for a group (calibration rules
    + guidance + enumerated features with genome glosses)."""
    return SYSTEM_PROMPTS[group_key]


def user_prompt_for(feed: str) -> str:
    """The per-movie user turn wrapping the feed — identical to the API path."""
    return user_message(feed)


# ── Genre-consistency guard (save-time mismatch detector) ─────────────────────

# Maps a TMDB genre to the LLM features a correctly-extracted film of that genre must
# score > 0 on at least once. A misaligned save — one movie's scores filed under another
# movie's id (see the 2026-06 audit: The Matrix extracted as a romance) — zeroes the real
# genre's signature, so this catches the gross cross-genre swaps at write time instead of
# letting them silently corrupt the feature tensor. STRICT genres have a near-1:1 signature
# (an animated film with no animation dim, a documentary with no documentary dim) where a
# zero is almost always a mismatch, so a violation RAISES. The rest only WARN: their genre
# tag is looser (a 'Science Fiction'-tagged superhero film legitimately has no space/robots),
# so an all-zero there is often a true negative, not a swap.
GENRE_SIGNATURES = {
    'Animation':       ['animated', 'computer_animation', 'stop_motion', 'anime'],
    'Documentary':     ['documentary'],
    'Horror':          ['scary', 'creepy', 'disturbing', 'gory', 'slasher', 'zombies',
                        'vampires', 'ghosts', 'monster', 'supernatural', 'dark', 'violent'],
    'War':             ['war', 'world_war_ii', 'cold_war'],
    'Western':         ['western_frontier'],
    'Music':           ['musical'],
    'Science Fiction': ['space', 'aliens', 'future', 'dystopia', 'post_apocalyptic',
                        'time_travel', 'cyberpunk', 'artificial_intelligence', 'robots', 'clones'],
    'Crime':           ['crime', 'heist', 'gangster', 'hitman', 'murder', 'corruption'],
}
GENRE_GUARD_STRICT = ('Animation', 'Documentary')   # violation → raise; all other genres → warn


def _genre_violations(movie_id: int, merged: dict) -> tuple:
    """
    TMDB genres the scrape asserts but for which the merged extraction has ALL signature
    features at 0. Returns (strict, warn) lists of genre names — empty when the scrape isn't
    cached (no ground truth to check against). The scrape's TMDB genres are the same ones the
    extractor saw in its feed, so a contradiction here means the SCORES, not the genres, are wrong.
    """
    rec    = load_record(movie_id)
    genres = set((rec.get('tmdb') or {}).get('genres', [])) if rec else set()
    strict, warn = [], []
    for g in genres & set(GENRE_SIGNATURES):
        if not any(merged.get(f, 0) > 0 for f in GENRE_SIGNATURES[g]):
            (strict if g in GENRE_GUARD_STRICT else warn).append(g)
    return strict, warn


# ── Validate + cache (tag-aware; mirrors the API path's writer) ───────────────

def ingest(movie_id: int, scores: dict, model_tag: str = MODEL_TAG) -> dict:
    """
    Validate a flat {feature: score} dict against the six group Pydantic models, default
    any unscored dim to 0.0, and write per-group cache files identical in shape to the API
    path. Raises on an unknown field or an out-of-range score, so a malformed extraction
    fails LOUDLY rather than silently corrupting the feature tensor. Also genre-checks the
    result against the scrape (see _genre_violations) to catch movieId/scores misalignment —
    a STRICT-genre contradiction raises; a looser one warns. Returns the merged 132-dim dict.
    """
    unknown = set(scores) - set(FEATURE_ORDER)
    if unknown:
        raise ValueError(f"movie {movie_id}: unknown feature(s) {sorted(unknown)}")

    merged = {}
    for g in GROUPS:
        subset    = {name: float(scores.get(name, 0.0)) for name in g['dim_names']}
        validated = g['model'](**subset).model_dump()   # raises on range / shape error
        merged.update(validated)

    # Mismatch guard — verify the scores actually describe THIS movie before writing them.
    strict, warn = _genre_violations(movie_id, merged)
    if warn:
        print(f"⚠️  movie {movie_id}: extraction shows no signal for genre(s) {warn} — "
              f"possible movieId/scores misalignment; verify.", file=sys.stderr)
    if strict:
        raise ValueError(
            f"movie {movie_id}: extraction contradicts scrape genre(s) {strict} (all signature "
            f"features 0) — almost certainly a misaligned save. Re-check the movieId↔scores "
            f"pairing; pass the correct scores for this movie.")

    for g in GROUPS:
        validated = {name: merged[name] for name in g['dim_names']}
        path = cache_path(g['key'], model_tag, movie_id)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump({
                'movieId':        movie_id,
                'group':          g['key'],
                'model':          model_tag,
                'prompt_version': PROMPT_VERSION,
                'features':       validated,
                'usage':          {},   # no tokens — Claude-Code path, not the API
            }, f, indent=2)
    return merged


# ── Batching helpers ──────────────────────────────────────────────────────────

def phase1_movie_ids() -> list:
    return json.load(open(PHASE1))['movie_ids']


def list_remaining(model_tag: str = MODEL_TAG, movie_ids: list = None) -> list:
    """phase1 movieIds lacking a COMPLETE six-group extraction under this tag (so a
    half-done movie is re-tried, and parallel workers can take disjoint ID ranges)."""
    ids = movie_ids if movie_ids is not None else phase1_movie_ids()
    return [mid for mid in ids
            if not all(os.path.exists(cache_path(g['key'], model_tag, mid)) for g in GROUPS)]


# ── CLI (inspection only — scoring is done by an agent) ───────────────────────

def _show(movie_id: int) -> None:
    feed = feed_for(movie_id)
    if feed is None:
        print(f"movieId {movie_id} not scraped — nothing to show.")
        return
    print("═" * 78)
    print(f"FEED  (movieId {movie_id})")
    print("═" * 78)
    print(feed)
    for g in GROUPS:
        print("\n" + "═" * 78)
        print(f"GROUP: {g['key']} ({g['title']}) — score every field 0.0–1.0")
        print("═" * 78)
        print(system_prompt_for(g['key']))


if __name__ == '__main__':
    args = sys.argv[1:]
    if '--remaining' in args:
        rem = list_remaining()
        total = len(phase1_movie_ids())
        print(f"{len(rem)} / {total} phase1 movies remaining under tag '{MODEL_TAG}'")
        print(f"   first 20: {rem[:20]}")
    elif '--show' in args:
        _show(int(args[args.index('--show') + 1]))
    else:
        print(__doc__)
