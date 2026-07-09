"""tools/ask_live_vs_frozen.py — grade the LIVE hosted-Haiku extractor against the frozen pills.

For every Ask-tab pill in serving/ask_examples.json this runs the REAL live extraction
(src.llm_frontend_extraction.extract_query — the exact hosted call the deployed Ask tab makes) and
compares it to the committed frozen extraction (tools/ask_extractions/<id>.json, the free-form
Claude Code Haiku-subagent output that produced the curated boards). It reports, per pill and in
aggregate: require_keyword_concepts adoption (live vs frozen) and top-5 / top-10 recommendation
overlap between the live board and the frozen board.

This is the HONESTY CHECK behind the pre-gen pills: a showcaseable pill must not be "too good to be
true" — its canned recs should be reproducible on the live Ask tab. A pill with near-zero live
overlap is either (a) a live-extractor gap to fix in the prompt, or (b) a frozen board that was
hand-seeded past what the query alone can produce (then swap the leaf for one that IS hittable).

READ-ONLY: never writes serving/ask_examples.json. (Regenerating the artifact is
tools/gen_ask_examples.py — and NEVER --live, per the pinned-board convention.)

Needs an API key: ANTHROPIC_API_KEY in the env, or a bare `ANTHROPIC_API_KEY = "..."` line in
.streamlit/secrets.toml (read automatically here). ~53 sequential calls ride the prompt cache
(one write + ~52 reads), so a full run is a few cents of Haiku.

USAGE (from repo root, in the torch env)
    /opt/anaconda3/envs/pytorch_env/bin/python tools/ask_live_vs_frozen.py
    ... tools/ask_live_vs_frozen.py --only r4c1,r5c1,r6c5     # grade a subset
    ... tools/ask_live_vs_frozen.py --k 3                      # sample each pill K times (noise/rate)
    ... tools/ask_live_vs_frozen.py --cache /tmp/live_ext.json # reuse cached live extractions (K=1)
"""

import argparse
import json
import os
import re
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.llm_frontend import recommend, normalize_extraction   # noqa: E402
from src.llm_frontend_extraction import extract_query          # noqa: E402
from tools.llm_frontend_probe import Serving                   # noqa: E402

_ARTIFACT    = os.path.join(_REPO_ROOT, 'serving', 'ask_examples.json')
_EXTRACTIONS = os.path.join(_REPO_ROOT, 'tools', 'ask_extractions')
_SECRETS     = os.path.join(_REPO_ROOT, '.streamlit', 'secrets.toml')
_STRAGGLER   = 2.0   # topic-pill top-5 overlap below this = flagged for prompt work or a leaf swap


def _ensure_api_key():
    """No env key? Pull the bare ANTHROPIC_API_KEY line out of .streamlit/secrets.toml."""
    if os.environ.get('ANTHROPIC_API_KEY'):
        return
    if os.path.exists(_SECRETS):
        m = re.search(r'ANTHROPIC_API_KEY\s*=\s*"([^"]+)"', open(_SECRETS).read())
        if m:
            os.environ['ANTHROPIC_API_KEY'] = m.group(1)
            return
    sys.exit('No ANTHROPIC_API_KEY in env or .streamlit/secrets.toml — set one to run live extraction.')


def _kw(ex):
    return (ex.get('hard_constraints') or {}).get('require_keyword_concepts') or []


def _board(ctx, ex, n):
    """Top-N titles for an extraction, via the real serving pipeline (schema-flatten repaired first)."""
    rep = recommend(ctx, normalize_extraction(ex), top_n=n)
    return [r[0] for r in rep['recs']], rep['fallback']


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--only', help='comma-separated pill ids to grade (default: all)')
    ap.add_argument('--k', type=int, default=1, help='samples per pill (Haiku is non-deterministic)')
    ap.add_argument('--cache', help='path to cache/reuse live extractions as JSON (K=1 only)')
    args = ap.parse_args()

    _ensure_api_key()
    # serving_only=True: grade against serving/ ALONE (no local facet_store.pt overlay), so this
    # honesty check sees EXACTLY what the deployed Ask tab resolves — the overlay masked the
    # 2026-07-09 keyword-fallback skew (pills looked reproducible locally but fell back live).
    srv = Serving(serving_only=True)
    art = json.load(open(_ARTIFACT))
    ids = list(art['examples'])
    if args.only:
        ids = [e for e in args.only.split(',') if e in art['examples']]
    cache = {}
    if args.cache and os.path.exists(args.cache):
        cache = json.load(open(args.cache))

    rows = []
    for eid in ids:
        q = art['examples'][eid]['query']
        frozen = json.load(open(os.path.join(_EXTRACTIONS, f'{eid}.json')))
        fz5, _ = _board(srv.ctx, frozen, 5)
        fz10, _ = _board(srv.ctx, frozen, 10)
        fz5, fz10 = set(fz5), set(fz10)
        kws, o5s, o10s, live_fb = 0, [], [], False
        for i in range(args.k):
            if args.cache and args.k == 1 and eid in cache:
                live = cache[eid]
            else:
                live = extract_query(q, srv.fs)
                if args.cache and args.k == 1:
                    cache[eid] = live
                    json.dump(cache, open(args.cache, 'w'), indent=1)
            lv5, fb = _board(srv.ctx, live, 5)
            lv10, _ = _board(srv.ctx, live, 10)
            kws += 1 if _kw(live) else 0
            o5s.append(len(set(lv5) & fz5))
            o10s.append(len(set(lv10) & fz10))
            live_fb = live_fb or fb
        rows.append({
            'id': eid, 'label': art['examples'][eid]['label'],
            'frozen_kw': bool(_kw(frozen)), 'live_kw': kws, 'k': args.k,
            'ov5': sum(o5s) / len(o5s), 'ov10': sum(o10s) / len(o10s), 'fallback': live_fb,
        })

    rows.sort(key=lambda r: r['ov5'])
    print(f"{'id':6} {'label':18} froz_kw live_kw   ov5   ov10  flags")
    for r in rows:
        flags = 'FALLBACK' if r['fallback'] else ''
        print(f"{r['id']:6} {r['label'][:18]:18}   {int(r['frozen_kw'])}    {r['live_kw']:>2}/{r['k']}  "
              f"{r['ov5']:.1f}/5 {r['ov10']:>4.1f}/10  {flags}")

    m5 = sum(r['ov5'] for r in rows) / len(rows)
    m10 = sum(r['ov10'] for r in rows) / len(rows)
    stragglers = [r['id'] for r in rows if r['ov5'] < _STRAGGLER]
    print(f"\nN={len(rows)} pills, K={args.k}")
    print(f"mean live-vs-frozen top-5 overlap : {m5:.2f}/5")
    print(f"mean live-vs-frozen top-10 overlap: {m10:.2f}/10")
    print(f"stragglers (top-5 overlap < {_STRAGGLER:.0f}): {len(stragglers)}  {stragglers}")
    print("  → for each: tighten the prompt, or (if the frozen board was hand-seeded past the "
          "query) swap the leaf for a hittable pill so the pre-gen pill isn't 'too good to be true'.")


if __name__ == '__main__':
    main()
