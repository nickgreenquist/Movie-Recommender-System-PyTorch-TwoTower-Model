"""
Phase-0 batch extraction driver — intended to be run BY a Claude Code (Sonnet) session.

The session running this script IS the extractor: it reads the feed + prompts for one
movie at a time, scores every feature in each group, and calls --save with a JSON dict.
No API key needed; this uses Claude Code's built-in model access.

Stratified 25-movie sanity batch (not the full 4,461-movie run — see BATCH_MOVIE_IDS):
covers foreign-language, pre-1970, arthouse, horror sub-genres (zombie/slasher/psychological),
sci-fi sub-genres (AI/robots/neo-noir), crime sub-genres (gangster/neo-noir).

── Workflow for the Claude Code Sonnet session running this ────────────────────────────

  Step 1 — See what's left:
    python -m llm_features.batch_extract --remaining

  Step 2 — Show next movie's full extraction task (feed + all 6 prompts):
    python -m llm_features.batch_extract --show

  Step 3 — After reading feed + prompts, score EVERY feature listed for each group.
    Calibration rules (from the prompts):
      • 0.0 = definitely absent, 1.0 = extremely prominent, in-between = partial presence
      • Use the FULL 0.0–1.0 range — do NOT default to 0.5 when unsure
      • Most features are 0.0 for any given film — only the few that genuinely apply should be high
      • Score from the text provided; where silent on something factual, score 0.0

  Step 4 — Save (only non-zero scores needed; 0.0 is the default for omitted features):
    python -m llm_features.batch_extract --save '<movieId>' '<json>'

    Example (replace with real scores):
      python -m llm_features.batch_extract --save '6016' '{"crime": 0.90, "gangster": 0.70, ...}'

  Step 5 — Repeat from Step 2 until --remaining shows 0.

── After all 25 are done ───────────────────────────────────────────────────────────────

  python -m llm_features.batch_extract --similarity   # sanity check: sci-fi should cluster

────────────────────────────────────────────────────────────────────────────────────────
"""
import json
import os
import sys

import torch

from llm_features.cc_extract import MODEL_TAG, feed_for, ingest, system_prompt_for
from llm_features.prompts import user_message
from llm_features.schemas import FEATURE_ORDER, GROUPS


# ── Stratified batch ──────────────────────────────────────────────────────────

BATCH_MOVIE_IDS = [
    # foreign-language
    6016,   # City of God (2002, PT) — crime/gangster/favela
    7022,   # Battle Royale (2000, JA) — dystopia/survival/violence
    3503,   # Solaris (1972, RU) — cerebral sci-fi/existentialism
    2692,   # Run Lola Run (1998, DE) — nonlinear/tense
    # pre-1970
    904,    # Rear Window (1954) — mystery/suspense/Hitchcock
    750,    # Dr. Strangelove (1964) — satire/cold_war/dark_humor
    919,    # The Wizard of Oz (1939) — fantasy/fairy_tale/musical
    # arthouse / indie
    6711,   # Lost in Translation (2003) — intimate/melancholic/arthouse
    4878,   # Donnie Darko (2001) — surreal/time_travel/psychological
    3676,   # Eraserhead (1977) — surreal/disturbing/Lynch
    # horror sub-genres — stress slasher/zombie/psychological dims
    1258,   # The Shining (1980) — psychological horror, NOT slasher
    8874,   # Shaun of the Dead (2004) — zombie/comedy hybrid
    7387,   # Dawn of the Dead (1978) — zombie, Romero
    4437,   # Suspiria (1977, IT) — giallo, witches, visual horror
    3018,   # Re-Animator (1985) — horror/sci-fi, creature
    # sci-fi sub-genres — stress cyberpunk/AI/robots/clones/neo-noir dims
    1653,   # Gattaca (1997) — biopunk, clones, dystopia
    1748,   # Dark City (1998) — neo-noir sci-fi, mystery
    1240,   # The Terminator (1984) — AI/robots, time_travel
    # crime sub-genres — stress heist/gangster/hitman/neo-noir
    44761,  # Brick (2005) — neo-noir, high school
    1208,   # Apocalypse Now (1979) — war, moral_complexity
    1213,   # Goodfellas (1990) — gangster/mafia, based_on_true_story
    # additional stress tests
    2858,   # American Beauty (1999) — satire/suburban, dark_humor
    3114,   # Toy Story 2 (1999) — sequel/animated (should resemble Toy Story)
]


# ── Cache helpers ─────────────────────────────────────────────────────────────

REPO_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GROUPS_DIR = os.path.join(REPO_ROOT, 'llm_features', 'cache', 'llm_groups')


def _is_done(mid: int) -> bool:
    from llm_features.llm_extract import cache_path
    return all(os.path.exists(cache_path(g['key'], MODEL_TAG, mid)) for g in GROUPS)


def _remaining() -> list:
    return [mid for mid in BATCH_MOVIE_IDS if not _is_done(mid)]


# ── Commands ──────────────────────────────────────────────────────────────────

def cmd_remaining():
    rem = _remaining()
    done = len(BATCH_MOVIE_IDS) - len(rem)
    print(f"Batch progress: {done}/{len(BATCH_MOVIE_IDS)} done, {len(rem)} remaining")
    if rem:
        print(f"Next: movieId {rem[0]}")
        print(f"Remaining IDs: {rem}")


def cmd_show(movie_id: int = None):
    """Print the full extraction task for one movie: feed + all 6 group prompts."""
    if movie_id is None:
        rem = _remaining()
        if not rem:
            print("All done — nothing left to show.")
            return
        movie_id = rem[0]

    feed = feed_for(movie_id)
    if feed is None:
        print(f"movieId {movie_id} not scraped — cannot show task.")
        return

    print("═" * 80)
    print(f"EXTRACTION TASK — movieId {movie_id}")
    print("═" * 80)
    print(feed)

    for g in GROUPS:
        print("\n" + "═" * 80)
        print(f"GROUP: {g['key'].upper()}  ({g['title']})  —  {len(g['dim_names'])} features")
        print("═" * 80)
        print(system_prompt_for(g['key']))
        print(f"\n[User prompt for this movie:]")
        print(user_message(feed))

    print("\n" + "═" * 80)
    print("All 6 groups shown. Score every feature in each group.")
    print(f"Save with: python -m llm_features.batch_extract --save '{movie_id}' '{{\"feature\": score, ...}}'")
    print("Only non-zero scores needed. 0.0 is the default for omitted features.")
    print("═" * 80)


def cmd_save(movie_id: int, scores_json: str):
    """Validate + cache scores for one movie (mirrors manual_extract.ingest)."""
    scores = json.loads(scores_json)
    merged = ingest(movie_id, scores, model_tag=MODEL_TAG)
    nz = {k: v for k, v in merged.items() if v > 0}
    tops = sorted(nz.items(), key=lambda x: -x[1])[:12]
    rem = len(_remaining())
    print(f"✓ movieId {movie_id} saved  ({len(nz)} non-zero dims)")
    print("  top: " + ', '.join(f"{k}={v:.2f}" for k, v in tops))
    print(f"  {rem} movies remaining in batch")


def cmd_similarity():
    """Cosine similarity check over all extracted batch movies."""
    from llm_features.merge_extractions import discover_movies, merge_one
    complete, _ = discover_movies(MODEL_TAG)
    if len(complete) < 2:
        print("Need ≥2 extracted movies for similarity check.")
        return

    from llm_features.build_features import SANITY_TITLES
    import torch.nn.functional as F

    # Also include manual-opus-4-8 movies for cross-check
    manual_complete, _ = discover_movies('manual-opus-4-8')
    all_ids = sorted(set(complete) | set(manual_complete))

    vecs, labels = [], []
    for mid in all_ids:
        tag = MODEL_TAG if mid in complete else 'manual-opus-4-8'
        feats = merge_one(tag, mid)
        v = torch.tensor([feats[k] for k in FEATURE_ORDER])
        vecs.append(v)
        title = SANITY_TITLES.get(mid, str(mid))
        source = '' if mid in complete else ' (manual)'
        labels.append(f"{title}{source}")

    mat = torch.stack(vecs)
    mat = F.normalize(mat, dim=1)
    sims = mat @ mat.T

    print(f"\n── Cosine similarity ({len(all_ids)} movies, tag={MODEL_TAG}) ──")
    for i, label in enumerate(labels):
        order = torch.argsort(sims[i], descending=True).tolist()
        nbrs  = [(labels[j], float(sims[i, j])) for j in order if j != i][:3]
        print(f"  {label:<28} → " + ', '.join(f"{n}={s:.2f}" for n, s in nbrs))


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    args = sys.argv[1:]
    if not args or '--help' in args:
        print(__doc__)
    elif '--remaining' in args:
        cmd_remaining()
    elif '--show' in args:
        idx = args.index('--show')
        mid = int(args[idx + 1]) if idx + 1 < len(args) and args[idx + 1].isdigit() else None
        cmd_show(mid)
    elif '--save' in args:
        idx = args.index('--save')
        cmd_save(int(args[idx + 1]), args[idx + 2])
    elif '--similarity' in args:
        cmd_similarity()
    else:
        print(f"Unknown args: {args}\nRun with --help for usage.")
