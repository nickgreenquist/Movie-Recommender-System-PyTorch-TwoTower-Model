"""
tools/llm_frontend_probe.py — v1 test harness for the LLM conversational front-end.

PURPOSE
    Validate the natural-language → two-tower pipeline END-TO-END *before* any hosted
    API or Streamlit wiring (see docs/plans/plan.md → "Testing In Claude Code Before
    Any API"). The flow under test:

        free-text utterance
          → LLM extraction (done OUTSIDE this script, by a Haiku subagent in the
            Claude Code loop; this script only consumes the structured JSON)
          → THIS HARNESS: resolve titles, synthesize a Mode-2 query from genome tags,
            build the user embedding via the real serving model, rank the corpus,
            apply v1 post-filters (year + genre)
          → top-N recommendations FROM THE TRAINED MODEL (never raw LLM output).

    The harness is serving/-only: it loads exactly what the deployed Streamlit app
    loads (serving/feature_store.pt, serving/movie_embeddings.pt, serving/model.pth)
    and mirrors streamlit_app.py's _build_serving_model + _score_movies, so the
    recommendations here match what the app would produce for the same model input.
    It never reads data/. Short MPS/CPU inference — safe to run directly.

EXTRACTION JSON (v1 schema — what the LLM is asked to produce; all fields optional)
    {
      "liked_items":    ["Inception", "Interstellar"],   # titles → fuzzy-matched to catalog
      "disliked_items": ["The Notebook"],                # titles → fuzzy-matched to catalog
      "genome_tags":    ["atmospheric", "melancholy"],   # Mode-2 mood → anchor movies (closed vocab)
      "liked_genres":   ["Sci-Fi"],                      # soft taste signal → user genre tower
      "disliked_genres":["Horror"],                      # soft taste signal → user genre tower
      "hard_constraints": {                              # post-retrieval filters (v1: year + genre only)
        "year_min": 2010,
        "year_max": null,
        "require_genres": ["Sci-Fi"],                    # rec must contain ALL of these
        "exclude_genres": ["Horror"]                     # drop rec if it contains ANY of these
      }
    }

    Soft vs hard genre: a *taste* preference ("I like sci-fi") belongs in liked_genres
    (shapes the embedding); a *hard* filter ("only sci-fi", "no horror") belongs in
    require_genres / exclude_genres (drops candidates after retrieval). v1.5 adds
    director / content-rating constraints once that metadata is baked into serving/.

USAGE (run from repo root)
    python tools/llm_frontend_probe.py --json '{"liked_items":["Inception"]}'
    python tools/llm_frontend_probe.py --json-file /tmp/extraction.json --utterance "..."
    echo '{"genome_tags":["western"]}' | python tools/llm_frontend_probe.py
    python tools/llm_frontend_probe.py --dump-vocab        # genome tag + genre vocab for the prompt
    python tools/llm_frontend_probe.py --smoke             # self-check: load + 3 canned queries
"""

import argparse
import difflib
import json
import os
import re
import sys

import numpy as np
import torch

# Repo root on sys.path so `src` imports resolve when run from anywhere.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.inference import build_user_embedding  # noqa: E402
from src.model import MovieRecommender          # noqa: E402

# ── Constants ────────────────────────────────────────────────────────────────
# History weights mirror streamlit_app.py's rec path: explicit likes 2.0, genome
# anchors 1.0, dislikes -2.0 (the same values the manual UI and canaries use).
LIKED_MOVIE_WEIGHT    =  2.0
ANCHOR_MOVIE_WEIGHT   =  1.0
DISLIKED_MOVIE_WEIGHT = -2.0
# Anchor budget. In pure Mode 2 (no named titles) the anchors ARE the query, so we use the
# full 5/tag at full weight (mirrors src/evaluate.py:ANCHORS_PER_TAG) — this path retrieves
# beautifully. But when the user named real titles (Mode 1), the titles ARE the query and
# anchors must only nudge: empirically, even 6 weight-1.0 anchors (sum 6.0) still outweigh
# 2 likes at 2.0 (sum 4.0) and drag the result off-target. So with likes present we cap
# anchors hard AND down-weight them, keeping total anchor signal well below the likes'
# (3 × 0.5 = 1.5 ≪ 4.0), so the named titles always dominate the user embedding.
ANCHORS_PER_TAG            = 5    # pure Mode 2 — anchors carry the whole query
ANCHORS_PER_TAG_WITH_LIKES = 1    # Mode 1 — at most one anchor per distinctive vibe tag
MAX_ANCHORS_WITH_LIKES     = 3    # ...and never enough, at half weight, to overwhelm the likes
ANCHOR_WEIGHT_WITH_LIKES   = 0.5  # down-weighted vs ANCHOR_MOVIE_WEIGHT so likes dominate
CANDIDATE_POOL        = 400    # retrieve large, post-filter, then surface TOP_N
TOP_N                 = 15
FUZZY_CUTOFF          = 0.6    # difflib ratio floor for title resolution

_SERVING = os.path.join(_REPO_ROOT, 'serving')
_NON_PERSISTENT_BUFFERS = ('genome_context_buffer', 'content_context_buffer', 'llm_feature_buffer')


# ── Serving load (mirrors streamlit_app.py:_build_serving_model + load_artifacts) ──
def _build_serving_model(fs, cfg):
    """Reconstruct the serving MovieRecommender from serving/ artifacts alone.

    Verbatim mirror of streamlit_app.py:_build_serving_model — sources the genome
    buffer from movieId_to_genome_tag_context and the LLM buffer from the baked
    llm_feature_buffer, so no data/ dir is needed.
    """
    feature_towers = cfg.get('feature_towers', cfg.get('content_feature_source', 'genome'))
    has_genome = feature_towers in ('genome', 'both')
    has_llm    = feature_towers in ('llm', 'both')

    top_movies = fs['top_movies']

    genome_context_buffer = None
    genome_tags_len       = len(fs['genome_tag_ids'])
    if has_genome:
        genome_matrix = np.array(
            [fs['movieId_to_genome_tag_context'][mid] for mid in top_movies], dtype=np.float32)
        pad_row = np.zeros((1, genome_matrix.shape[1]), dtype=np.float32)
        genome_context_buffer = torch.from_numpy(np.vstack([genome_matrix, pad_row]))
        genome_tags_len = genome_matrix.shape[1]

    llm_feature_buffer = None
    llm_feature_len    = None
    if has_llm:
        llm_feature_buffer = fs['llm_feature_buffer']
        llm_feature_len    = llm_feature_buffer.shape[1]

    return MovieRecommender(
        genres_len=len(fs['genres_ordered']),
        tags_len=len(fs['tags_ordered']),
        genome_tags_len=genome_tags_len,
        top_movies_len=len(top_movies),
        all_years_len=len(fs['years_ordered']),
        timestamp_num_bins=fs['timestamp_num_bins'],
        user_context_size=fs['user_context_size'],
        feature_towers=feature_towers,
        genome_context_buffer=genome_context_buffer,
        llm_feature_buffer=llm_feature_buffer,
        llm_feature_len=llm_feature_len,
        item_genre_embedding_size=cfg['item_genre_embedding_size'],
        item_tag_embedding_size=cfg['item_tag_embedding_size'],
        item_genome_embedding_size=cfg.get('item_genome_embedding_size', 32),
        item_llm_embedding_size=cfg.get('item_llm_embedding_size', 32),
        item_movieId_embedding_size=cfg['item_movieId_embedding_size'],
        item_year_embedding_size=cfg['item_year_embedding_size'],
        user_genre_embedding_size=cfg['user_genre_embedding_size'],
        timestamp_feature_embedding_size=cfg['timestamp_feature_embedding_size'],
        user_genome_embedding_size=cfg.get('user_genome_embedding_size', 32),
        user_llm_embedding_size=cfg.get('user_llm_embedding_size', 32),
        proj_hidden=cfg.get('proj_hidden', 256),
        output_dim=cfg.get('output_dim', 128),
    )


class Serving:
    """Bundle of the loaded serving artifacts + derived lookups the harness needs."""

    def __init__(self):
        self.fs  = torch.load(os.path.join(_SERVING, 'feature_store.pt'), weights_only=False)
        self.me  = torch.load(os.path.join(_SERVING, 'movie_embeddings.pt'), weights_only=False)
        cfg = self.fs['model_config']

        self.model = _build_serving_model(self.fs, cfg)
        state_dict = torch.load(os.path.join(_SERVING, 'model.pth'), weights_only=True)
        for buf in _NON_PERSISTENT_BUFFERS:
            state_dict.pop(buf, None)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        # Item retrieval matrix — row i ↔ movieId all_ids[i] (mirror list(me.keys())).
        self.all_ids  = list(self.me.keys())
        self.all_embs = torch.cat(
            [self.me[m]['MOVIE_EMBEDDING_COMBINED'] for m in self.all_ids], dim=0)  # [N,128] L2-normed

        # Most-recent timestamp bin for inference (the canary / streamlit approach).
        bins = self.fs['timestamp_bins']
        self.ts_inference = torch.bucketize(
            torch.tensor([float(bins[-1].item())]), bins, right=False)

        # Title resolution index + popularity rank (for collision tie-break and fallback).
        self._title_index = _build_title_index(self.fs)
        self._pop_rank = {t: i for i, t in enumerate(self.fs['popularity_ordered_titles'])}

        # Genome anchor lookup: tag NAME → 0-based column in the 1128-dim genome vector.
        gn, gi = self.fs['genome_tag_names'], self.fs['genome_tag_to_i']
        self._genome_name_to_idx = {gn[tid]: gi[tid] for tid in gi}


# ── Title resolution (fuzzy-match LLM-extracted titles → catalog titles) ──────
_YEAR_SUFFIX_RE = re.compile(r'\s*\(\d{4}\)\s*$')
_ARTICLE_RE     = re.compile(
    r"^(.*),\s+(the|a|an|le|la|les|los|las|il|der|die|das|l')$", re.IGNORECASE)


def _norm_title(s):
    """Normalize a title for matching: drop "(year)", un-invert MovieLens "Name, The"
    → "the name", lowercase, strip punctuation. Handles user input ("The Matrix")
    matching the catalog's inverted, year-suffixed form ("Matrix, The (1999)")."""
    s = _YEAR_SUFFIX_RE.sub('', s).strip()
    m = _ARTICLE_RE.match(s)
    if m:
        art = m.group(2)
        sep = '' if art.lower() == "l'" else ' '
        s = f"{art}{sep}{m.group(1)}"
    s = s.lower()
    s = re.sub(r'[^a-z0-9]+', ' ', s).strip()
    return s


def _build_title_index(fs):
    """normalized-title → list of canonical catalog titles (with "(year)" suffix)."""
    index = {}
    for _mid, title in fs['movieId_to_title'].items():
        index.setdefault(_norm_title(title), []).append(title)
    return index


def resolve_title(raw, srv):
    """Resolve one LLM-extracted title to a canonical catalog title.

    Returns (canonical_title | None, note). Strategy: exact (case-insensitive,
    year-aware) → exact normalized → difflib fuzzy. On a normalized collision (or
    fuzzy tie) prefer the most popular candidate. Unresolved titles return None and
    are reported, then dropped (build_user_embedding would silently drop them too)."""
    fs = srv.fs
    # 1. Exact catalog hit (the LLM sometimes returns the full "Title (year)" string).
    if raw in fs['title_to_movieId']:
        return raw, 'exact'

    norm = _norm_title(raw)
    if not norm:
        return None, 'empty'

    # 2. Exact normalized hit (handles missing year + article inversion).
    if norm in srv._title_index:
        cands = srv._title_index[norm]
        best = min(cands, key=lambda t: srv._pop_rank.get(t, 1 << 30))
        note = 'normalized' if len(cands) == 1 else f'normalized, {len(cands)} matches → most popular'
        return best, note

    # 3. Fuzzy match on normalized keys.
    keys = list(srv._title_index.keys())
    hit = difflib.get_close_matches(norm, keys, n=1, cutoff=FUZZY_CUTOFF)
    if hit:
        ratio = difflib.SequenceMatcher(None, norm, hit[0]).ratio()
        cands = srv._title_index[hit[0]]
        best = min(cands, key=lambda t: srv._pop_rank.get(t, 1 << 30))
        return best, f'fuzzy {ratio:.2f}'

    return None, 'no match'


# ── Mode-2 synthesis: genome tag → anchor movies (mirror evaluate._get_anchor_titles) ──
def anchors_for(srv, genome_tags, exclude, per_tag=ANCHORS_PER_TAG, max_total=None):
    """Top `per_tag` representative real movies per genome tag (descending genome relevance
    over the whole corpus), globally de-duplicated against `exclude` and each other, capped
    at `max_total` anchors overall when given. Returns (anchor_titles, unresolved_tags).
    Mirrors src/evaluate.py's anchor derivation so Mode-2 reuses the tested machinery rather
    than a synthetic query vector; `per_tag`/`max_total` add the Mode-1 subordination budget."""
    fs = srv.fs
    name_to_idx = srv._genome_name_to_idx
    anchor_titles, unresolved = [], []
    seen = set(exclude)
    for tag in genome_tags:
        if max_total is not None and len(anchor_titles) >= max_total:
            break
        if tag not in name_to_idx:
            unresolved.append(tag)
            continue
        ti = name_to_idx[tag]
        ordered = sorted(
            fs['top_movies'],
            key=lambda mid: float(fs['movieId_to_genome_tag_context'][mid][ti]),
            reverse=True,
        )
        n = 0
        for mid in ordered:
            if n >= per_tag or (max_total is not None and len(anchor_titles) >= max_total):
                break
            title = fs['movieId_to_title'][mid]
            if title not in seen:
                anchor_titles.append(title)
                seen.add(title)
                n += 1
    return anchor_titles, unresolved


# ── v1 post-filter (year + genre only; director/rating deferred to v1.5) ─────
def _passes_constraints(mid, fs, hc):
    """True if movie `mid` satisfies the hard constraints. Unknown/non-numeric year
    passes the year gate (no info to filter on)."""
    if not hc:
        return True
    year_min, year_max = hc.get('year_min'), hc.get('year_max')
    if year_min is not None or year_max is not None:
        raw_year = fs['movieId_to_year'].get(mid)
        try:
            year = int(raw_year)
            if year_min is not None and year < year_min:
                return False
            if year_max is not None and year > year_max:
                return False
        except (TypeError, ValueError):
            pass  # no usable year → don't filter it out
    genres = set(fs['movieId_to_genres'].get(mid, []))
    require = hc.get('require_genres') or []
    exclude = hc.get('exclude_genres') or []
    if require and not all(g in genres for g in require):
        return False
    if exclude and any(g in genres for g in exclude):
        return False
    return True


# ── Recommend ────────────────────────────────────────────────────────────────
def recommend(srv, extraction, top_n=TOP_N):
    """Run one extraction object through the full v1 pipeline. Returns a report dict."""
    fs = srv.fs
    liked_raw    = extraction.get('liked_items') or []
    disliked_raw = extraction.get('disliked_items') or []
    genome_tags  = extraction.get('genome_tags') or []
    liked_genres = extraction.get('liked_genres') or []
    disliked_genres = extraction.get('disliked_genres') or []
    hc = extraction.get('hard_constraints') or {}

    # 1. Resolve mentioned titles (Mode 1).
    liked_resolved, disliked_resolved = [], []
    resolution_log = {'liked': [], 'disliked': []}
    for raw in liked_raw:
        canon, note = resolve_title(raw, srv)
        resolution_log['liked'].append((raw, canon, note))
        if canon:
            liked_resolved.append(canon)
    for raw in disliked_raw:
        canon, note = resolve_title(raw, srv)
        resolution_log['disliked'].append((raw, canon, note))
        if canon:
            disliked_resolved.append(canon)

    # 2. Mode-2 synthesis: genome tags → anchor movies (exclude already-named seeds).
    #    When the user named real titles (Mode 1) the anchors are subordinated so they refine
    #    rather than swamp the explicit likes; pure Mode 2 keeps the full anchor strength.
    seed_exclude = set(liked_resolved) | set(disliked_resolved)
    has_likes = bool(liked_resolved)
    per_tag      = ANCHORS_PER_TAG_WITH_LIKES if has_likes else ANCHORS_PER_TAG
    max_total    = MAX_ANCHORS_WITH_LIKES if has_likes else None
    anchor_weight = ANCHOR_WEIGHT_WITH_LIKES if has_likes else ANCHOR_MOVIE_WEIGHT
    anchors, unresolved_tags = anchors_for(srv, genome_tags, seed_exclude, per_tag, max_total)

    # 3. Assemble model input: explicit likes (2.0) + anchors (1.0, or 0.5 when subordinated),
    #    dislikes (-2.0).
    liked_with_weights = (
        [(t, LIKED_MOVIE_WEIGHT) for t in liked_resolved] +
        [(t, anchor_weight)      for t in anchors]
    )

    # Validate genre names against the model's vocabulary (surface, don't crash).
    genre_vocab = set(fs['genres_ordered'])
    unknown_genres = sorted(
        (set(liked_genres) | set(disliked_genres)
         | set(hc.get('require_genres') or []) | set(hc.get('exclude_genres') or []))
        - genre_vocab)

    # Empty-signal fallback → popularity (a sensible, diverse default).
    fallback = not liked_with_weights and not disliked_resolved \
        and not liked_genres and not disliked_genres

    # 4. Build the user embedding via the real serving model.
    with torch.no_grad():
        user_emb = build_user_embedding(
            srv.model, fs, liked_with_weights, disliked_resolved, srv.ts_inference,
            liked_genres=liked_genres, disliked_genres=disliked_genres,
            disliked_movie_value=DISLIKED_MOVIE_WEIGHT,
        )

    # 5. Score whole corpus (raw dot product == cosine; both L2-normed). No alpha/temp.
    raw_scores = (srv.all_embs @ user_emb.T).squeeze(-1)

    # 6. Rank → drop seeds by title → apply post-filters → take top_n.
    seed_titles = set(liked_resolved) | set(disliked_resolved) | set(anchors)
    recs, kept, filtered = [], 0, 0
    if fallback:
        ranked_titles = fs['popularity_ordered_titles']
        title_to_mid = fs['title_to_movieId']
        for title in ranked_titles:
            mid = title_to_mid.get(title)
            if mid is None or title in seed_titles:
                continue
            if not _passes_constraints(mid, fs, hc):
                filtered += 1
                continue
            recs.append((title, fs['movieId_to_genres'].get(mid, []), fs['movieId_to_year'].get(mid), None))
            kept += 1
            if kept >= top_n:
                break
    else:
        order = raw_scores.argsort(descending=True).tolist()
        for i in order:
            if kept >= top_n:
                break
            mid = srv.all_ids[i]
            title = fs['movieId_to_title'][mid]
            if title in seed_titles:
                continue
            if not _passes_constraints(mid, fs, hc):
                filtered += 1
                continue
            recs.append((title, fs['movieId_to_genres'].get(mid, []),
                         fs['movieId_to_year'].get(mid), float(raw_scores[i])))
            kept += 1

    return {
        'extraction': extraction,
        'resolution': resolution_log,
        'anchors': anchors,
        'anchor_weight': anchor_weight,
        'unresolved_tags': unresolved_tags,
        'unknown_genres': unknown_genres,
        'seed_count': len(seed_titles),
        'fallback': fallback,
        'filtered': filtered,
        'recs': recs,
    }


# ── Reporting ────────────────────────────────────────────────────────────────
def print_report(report, utterance=None):
    ex = report['extraction']
    print('=' * 72)
    print('LLM FRONT-END PROBE  (v1: title-resolution + genome-anchor synthesis)')
    print('=' * 72)
    if utterance:
        print(f'Utterance: {utterance}')
        print('-' * 72)
    print('Extraction (LLM output, consumed internally — never shown to the end user):')
    print(f"  liked_items     : {ex.get('liked_items') or []}")
    print(f"  disliked_items  : {ex.get('disliked_items') or []}")
    print(f"  genome_tags     : {ex.get('genome_tags') or []}")
    print(f"  liked_genres    : {ex.get('liked_genres') or []}")
    print(f"  disliked_genres : {ex.get('disliked_genres') or []}")
    print(f"  hard_constraints: {ex.get('hard_constraints') or {}}")
    print('-' * 72)

    print('Resolution:')
    for raw, canon, note in report['resolution']['liked']:
        arrow = f'→ {canon}' if canon else '→ (UNRESOLVED, dropped)'
        print(f"  like    {raw!r:<32} {arrow}  [{note}]")
    for raw, canon, note in report['resolution']['disliked']:
        arrow = f'→ {canon}' if canon else '→ (UNRESOLVED, dropped)'
        print(f"  dislike {raw!r:<32} {arrow}  [{note}]")
    if report['anchors']:
        print(f"  genome anchors (Mode-2 synthesis, weight {report['anchor_weight']}):")
        for t in report['anchors']:
            print(f"      • {t}")
    if report['unresolved_tags']:
        print(f"  ⚠ genome tags NOT in vocab (ignored): {report['unresolved_tags']}")
    if report['unknown_genres']:
        print(f"  ⚠ genre names NOT in model vocab (ignored): {report['unknown_genres']}")
    if report['fallback']:
        print('  ⚠ empty extraction → POPULARITY fallback')
    print('-' * 72)

    n = len(report['recs'])
    tag = 'popularity fallback' if report['fallback'] else 'model retrieval'
    print(f'Recommendations (top {n} after year+genre post-filter; '
          f'{report["filtered"]} candidates filtered; source: {tag}):')
    for rank, (title, genres, year, score) in enumerate(report['recs'], 1):
        sc = f'  (cos {score:+.3f})' if score is not None else ''
        print(f"  {rank:>2}. {title}  [{', '.join(genres)}]{sc}")
    print('=' * 72)


# ── Vocab dump (for building the extraction prompt) ──────────────────────────
def dump_vocab(srv):
    fs = srv.fs
    names = [fs['genome_tag_names'][tid] for tid in sorted(fs['genome_tag_names'])]
    print(f'# GENRES ({len(fs["genres_ordered"])}) — for liked_genres / require_genres / exclude_genres')
    print(json.dumps([g for g in fs['genres_ordered'] if g != '(no genres listed)']))
    print()
    print(f'# GENOME TAGS ({len(names)}) — closed vocab for genome_tags (Mode-2)')
    print(json.dumps(names))


# ── Smoke test ───────────────────────────────────────────────────────────────
_SMOKE_CASES = [
    {'liked_items': ['Inception', 'Interstellar']},                       # Mode 1
    {'genome_tags': ['western'], 'liked_genres': ['Western']},            # Mode 2
    {'liked_items': ['The Matrix'], 'genome_tags': ['post-apocalyptic'],  # mixed + constraints
     'disliked_genres': ['Horror'],
     'hard_constraints': {'year_min': 2000, 'exclude_genres': ['Horror']}},
]


def main():
    ap = argparse.ArgumentParser(description='v1 LLM front-end test harness (serving/-only).')
    ap.add_argument('--json', help='inline extraction JSON string')
    ap.add_argument('--json-file', help='path to a file containing extraction JSON')
    ap.add_argument('--utterance', help='original NL utterance, for the report header')
    ap.add_argument('--top-n', type=int, default=TOP_N)
    ap.add_argument('--dump-vocab', action='store_true', help='print genome-tag + genre vocab and exit')
    ap.add_argument('--smoke', action='store_true', help='load + run canned queries (self-check)')
    args = ap.parse_args()

    srv = Serving()

    if args.dump_vocab:
        dump_vocab(srv)
        return

    if args.smoke:
        for case in _SMOKE_CASES:
            print_report(recommend(srv, case, top_n=args.top_n))
            print()
        return

    if args.json:
        extraction = json.loads(args.json)
    elif args.json_file:
        with open(args.json_file) as f:
            extraction = json.load(f)
    else:
        extraction = json.load(sys.stdin)

    print_report(recommend(srv, extraction, top_n=args.top_n), utterance=args.utterance)


if __name__ == '__main__':
    main()
