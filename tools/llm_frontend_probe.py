"""
tools/llm_frontend_probe.py — v1 test harness for the LLM conversational front-end.

PURPOSE
    Validate the natural-language → two-tower pipeline END-TO-END *before* any hosted
    API or Streamlit wiring (see docs/llm_frontend/llm_frontend_plan.md → "Testing In Claude Code Before
    Any API"). The flow under test:

        free-text utterance
          → LLM extraction (done OUTSIDE this script, by a Haiku subagent in the
            Claude Code loop; this script only consumes the structured JSON)
          → the shared pipeline (src/llm_frontend.py), driven by this harness: resolve
            titles, synthesize a Mode-2 query from genome tags, build the user embedding
            via the real serving model, rank the corpus, apply v1 post-filters (year + genre)
          → top-N recommendations FROM THE TRAINED MODEL (never raw LLM output).

    The harness is serving/-only: it loads exactly what the deployed Streamlit app
    loads (serving/feature_store.pt, serving/movie_embeddings.pt, serving/model.pth)
    and drives the shared src/llm_frontend.py core (the same retrieval / resolution /
    anchor-synthesis / post-filter code the Streamlit conversational tab uses), so the
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
import json
import os
import sys

import torch

# Repo root on sys.path so `src` imports resolve when run from anywhere.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.llm_frontend import (  # noqa: E402
    NON_PERSISTENT_BUFFERS,
    TOP_N,
    build_frontend_context,
    build_serving_model,
    recommend,
)

_SERVING = os.path.join(_REPO_ROOT, 'serving')


# ── Serving load (mirrors streamlit_app.py:load_artifacts) ────────────────────
class Serving:
    """Load the serving/ artifacts and build the FrontendContext the shared pipeline
    (src/llm_frontend.py) consumes. Mirrors streamlit_app.py:load_artifacts minus the
    UI-only matrices — the Streamlit conversational tab will instead pass its cached
    Artifacts to build_frontend_context rather than reloading serving/ from disk."""

    def __init__(self):
        self.fs = torch.load(os.path.join(_SERVING, 'feature_store.pt'), weights_only=False)
        me      = torch.load(os.path.join(_SERVING, 'movie_embeddings.pt'), weights_only=False)
        cfg     = self.fs['model_config']

        model = build_serving_model(self.fs, cfg)
        state_dict = torch.load(os.path.join(_SERVING, 'model.pth'), weights_only=True)
        for buf in NON_PERSISTENT_BUFFERS:
            state_dict.pop(buf, None)
        model.load_state_dict(state_dict)
        model.eval()

        # Item retrieval matrix — row i ↔ movieId all_ids[i] (mirror list(me.keys())).
        all_ids  = list(me.keys())
        all_embs = torch.cat(
            [me[m]['MOVIE_EMBEDDING_COMBINED'] for m in all_ids], dim=0)  # [N,128] L2-normed

        # Most-recent timestamp bin for inference (the canary / streamlit approach).
        bins = self.fs['timestamp_bins']
        ts_inference = torch.bucketize(
            torch.tensor([float(bins[-1].item())]), bins, right=False)

        # Scraped-facet store (people tables). Prefer the baked serving copy (fs['facets'], written
        # by export); fall back to the local build artifact for dev before the bake exists. The
        # deployed app only ever sees fs['facets'] — this disk fallback is harness-only.
        facets = self.fs.get('facets')
        fp = os.path.join(_REPO_ROOT, 'llm_features', 'cache', 'facet_store.pt')
        if facets is None:
            if os.path.exists(fp):
                facets = torch.load(fp, weights_only=False)
        elif os.path.exists(fp):
            # Split-brain guard (harness-only): the baked facet tables lag the locally-rebuilt
            # store whenever KEYWORD_CONCEPTS grows between gated exports, so /trace and the probe
            # CLI would disagree with the rulers on concept membership (a require on a new concept
            # empty-pools here while passing eval). Overlay the tables that drift — the same patch
            # as mock_serving.py / llm_frontend_eval.py; serving/ stays untouched. keyword_to_movieIds
            # (the step-2 resolver's raw-keyword index) is local-only and never baked, so it always
            # rides in from the local store.
            local = torch.load(fp, map_location='cpu', weights_only=False)
            for table in ('movieId_to_keyword_concepts', 'keyword_to_movieIds'):
                if table in local:
                    facets[table] = local[table]

        self.ctx = build_frontend_context(
            model, self.fs, all_ids, all_embs, ts_inference, facets=facets)


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
    pr = report.get('people_resolution') or {}
    for key in ('require', 'exclude'):
        for raw, pid, note in pr.get(key, []):
            arrow = f'→ person #{pid}' if pid is not None else '→ (UNRESOLVED, dropped)'
            print(f"  {key+' ppl':<8}{raw!r:<32} {arrow}  [{note}]")
    tr = report.get('topic_resolution') or {}
    for key in ('require', 'exclude'):
        for raw, note, size in tr.get(key, []):
            arrow = f'→ {note} ({size} films)' if size is not None else f'→ (UNRESOLVED, dropped) [{note}]'
            print(f"  {key+' topic':<8}{raw!r:<32} {arrow}")
    if report['anchors']:
        print(f"  genome anchors (Mode-2 synthesis, weight {report['anchor_weight']}):")
        for t in report['anchors']:
            print(f"      • {t}")
    if report['unresolved_tags']:
        print(f"  ⚠ genome tags NOT in vocab (ignored): {report['unresolved_tags']}")
    if report['unknown_genres']:
        print(f"  ⚠ genre names NOT in model vocab (ignored): {report['unknown_genres']}")
    if report['fallback']:
        print('  ⚠ no embedding signal → POPULARITY ranking (within any hard constraints)')
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
    {'hard_constraints': {'require_people': ['Tom Hanks']}},              # people facet filter (Phase 1)
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
            print_report(recommend(srv.ctx, case, top_n=args.top_n))
            print()
        return

    if args.json:
        extraction = json.loads(args.json)
    elif args.json_file:
        with open(args.json_file) as f:
            extraction = json.load(f)
    else:
        extraction = json.load(sys.stdin)

    print_report(recommend(srv.ctx, extraction, top_n=args.top_n), utterance=args.utterance)


if __name__ == '__main__':
    main()
