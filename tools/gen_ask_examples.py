"""
tools/gen_ask_examples.py — pre-generate the Ask tab's example-chip boards.

Walks the hand-curated query tree (tools/ask_examples_spec.py), obtains one extraction per
query, runs each through the REAL serving pipeline (src.llm_frontend.recommend over the
exported serving/ artifacts — the exact code path the deployed Ask tab runs), and writes
serving/ask_examples.json. The Streamlit Ask tab renders these boards through the same
_report_to_df / _render_ask_debug path as live queries, so a chip click is byte-for-byte
the live experience minus the hosted LLM call.

EXTRACTION SOURCES (pick one)
    --from-extractions DIR   read pre-made extraction JSONs (DIR/<id>.json) — e.g. produced by
                             the no-key Haiku-subagent harness (the /trace pattern; same model
                             family the hosted call uses, so extractions transfer).
    --live                   call the hosted extractor (src.llm_frontend_extraction) — needs
                             ANTHROPIC_API_KEY. 42 sequential calls ride the prompt cache
                             (~1 write + 41 reads ≈ $0.10). Use to refresh the artifact after
                             a model promotion / re-export.

NORMALIZATION (applied to both sources; keeps the artifact schema-faithful)
    - Root-level hard-constraint keys are re-nested under hard_constraints ("schema-flatten",
      a known extractor drift — the pipeline ignores root-level keys, so an unrepaired flatten
      silently loses the filter and the canned board would not reflect the intended query).
    - String values for array-typed fields are split on commas into lists (schema declares
      them array[string]; small models occasionally emit a bare string).

USAGE (from repo root)
    python tools/gen_ask_examples.py --from-extractions /path/to/extractions [--report out.md]
    ANTHROPIC_API_KEY=... python tools/gen_ask_examples.py --live

Prints a per-query curation report (extraction gist, honesty flags, top-10 board) — review it
and reword/swap weak spec entries, then regenerate. --only r4c2,r5c1 regenerates a subset in
place (the artifact is read-merge-written), so a swap doesn't re-run the other 40.
"""

import argparse
import json
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.llm_frontend import TOP_N, recommend            # noqa: E402
from tools.ask_examples_spec import ROOTS, all_entries    # noqa: E402
from tools.llm_frontend_probe import Serving              # noqa: E402

_ARTIFACT = os.path.join(_REPO_ROOT, 'serving', 'ask_examples.json')
_TOP_N    = 60   # mirror streamlit_app._TOTAL_RESULTS so canned pagination matches live

# Schema-derived field typing (src.llm_frontend_prompt.build_schema): list-typed keys at each
# level, used to re-nest flattened constraints and coerce stray strings back to lists.
_TOP_LIST_KEYS = {'liked_items', 'disliked_items', 'genome_tags', 'liked_genres',
                  'disliked_genres', 'mood', 'unsupported_notes'}
_HC_LIST_KEYS  = {'require_genres', 'exclude_genres', 'require_people', 'exclude_people',
                  'require_genome_tags', 'exclude_genome_tags', 'exclude_mood',
                  'require_country', 'require_language', 'require_attributes',
                  'require_keyword_concepts', 'exclude_keyword_concepts',
                  'require_franchise', 'exclude_franchise',
                  'require_composers', 'exclude_composers'}
_HC_SCALAR_KEYS = {'year_min', 'year_max', 'require_max_rating', 'require_min_rating',
                   'max_runtime', 'min_runtime', 'min_vote_average'}


# ── extraction normalization ─────────────────────────────────────────────────
def _listify(v):
    """Coerce a stray string into the list the schema declares (split multi-phrase strings)."""
    if isinstance(v, str):
        return [p.strip() for p in v.split(',') if p.strip()]
    return v


def normalize_extraction(ex):
    """Repair known small-model drift so the stored extraction matches the schema shape the
    hosted forced-tool call produces. Returns a new dict; the input is not mutated."""
    ex = dict(ex)
    hc = dict(ex.get('hard_constraints') or {})
    for key in list(ex.keys()):                       # schema-flatten repair: root → nested
        if key in _HC_LIST_KEYS or key in _HC_SCALAR_KEYS:
            hc.setdefault(key, ex.pop(key))
    for key in _TOP_LIST_KEYS & set(ex):
        ex[key] = _listify(ex[key])
    for key in _HC_LIST_KEYS & set(hc):
        hc[key] = _listify(hc[key])
    if hc:
        ex['hard_constraints'] = hc
    return ex


# ── JSON sanitization ────────────────────────────────────────────────────────
def _jsonable(x):
    """Recursively convert a recommend() report to plain-JSON types (tuples → lists,
    tensor/numpy scalars → python). str() is a guarded last resort, not an expected path."""
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if hasattr(x, 'item'):          # torch / numpy zero-dim scalar
        return _jsonable(x.item())
    if hasattr(x, 'tolist'):        # torch / numpy array
        return _jsonable(x.tolist())
    return str(x)


# ── curation report ──────────────────────────────────────────────────────────
def _extraction_gist(ex):
    parts = []
    for k in ('liked_items', 'disliked_items', 'genome_tags', 'mood',
              'liked_genres', 'disliked_genres'):
        if ex.get(k):
            parts.append(f"{k}={ex[k]}")
    for k, v in (ex.get('hard_constraints') or {}).items():
        if v not in (None, [], ''):
            parts.append(f"hc.{k}={v}")
    return '; '.join(parts) or '(empty extraction)'


def _flags(report):
    """The honesty signals a canned example must be clean on (or consciously accept)."""
    out = []
    if report['fallback']:
        out.append('FALLBACK')
    if report['relaxed_constraints']:
        out.append(f"relaxed={report['relaxed_constraints']}")
    dropped = [raw for raw, canon, _ in
               report['resolution']['liked'] + report['resolution']['disliked'] if not canon]
    if dropped:
        out.append(f"dropped_titles={dropped}")
    pr = report.get('people_resolution') or {}
    dropped_people = [raw for bucket in ('require', 'exclude')
                      for raw, pid, _ in pr.get(bucket, []) if pid is None]
    if dropped_people:
        out.append(f"dropped_people={dropped_people}")
    if report.get('unresolved_tags'):
        out.append(f"unresolved_tags={report['unresolved_tags']}")
    if report.get('unknown_genres'):
        out.append(f"unknown_genres={report['unknown_genres']}")
    if report.get('capability_notice'):
        out.append(f"notice={report['capability_notice']!r}")
    if len(report['recs']) < _TOP_N:
        out.append(f"thin_pool={len(report['recs'])}")
    return out


def _report_lines(entry_id, label, query, extraction, report):
    lines = [f"## {entry_id} — {label}", f"query: {query}",
             f"extraction: {_extraction_gist(extraction)}"]
    flags = _flags(report)
    lines.append("flags: " + ("; ".join(flags) if flags else "clean"))
    top = [f"{t} ({y})" for t, _g, y, _s in report['recs'][:10]]
    lines.append("top10: " + " | ".join(top))
    return lines


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--from-extractions', metavar='DIR',
                     help='directory of pre-made <id>.json extraction files')
    src.add_argument('--live', action='store_true',
                     help='call the hosted extractor (needs ANTHROPIC_API_KEY)')
    ap.add_argument('--only', default=None,
                    help='comma-separated ids to (re)generate; others kept from the artifact')
    ap.add_argument('--report', default=None, help='also write the curation report to this file')
    args = ap.parse_args()

    only = set(args.only.split(',')) if args.only else None
    serving = Serving()

    # Start from the existing artifact so --only swaps merge instead of clobbering.
    artifact = {'generated_with': None, 'roots': [], 'tree': {}, 'examples': {}}
    if only and os.path.exists(_ARTIFACT):
        with open(_ARTIFACT) as f:
            artifact = json.load(f)

    artifact['generated_with'] = 'hosted-api' if args.live else 'subagent-haiku'
    artifact['roots'] = [r['id'] for r in ROOTS]
    artifact['tree']  = {r['id']: [c['id'] for c in r['children']] for r in ROOTS}

    report_lines = []
    for entry, _parent in all_entries():
        eid = entry['id']
        if only and eid not in only:
            continue
        if args.live:
            from src.llm_frontend_extraction import extract_query
            extraction = extract_query(entry['query'], serving.fs)
        else:
            path = os.path.join(args.from_extractions, f'{eid}.json')
            with open(path) as f:
                extraction = json.load(f)
        extraction = normalize_extraction(extraction)
        report = recommend(serving.ctx, extraction, top_n=_TOP_N)
        artifact['examples'][eid] = {
            'label':  entry['label'],
            'query':  entry['query'],
            'report': _jsonable(report),
        }
        report_lines.extend(_report_lines(eid, entry['label'], entry['query'],
                                          extraction, report))
        report_lines.append('')

    missing = [e['id'] for e, _ in all_entries() if e['id'] not in artifact['examples']]
    if missing:
        print(f"WARNING: artifact is missing {len(missing)} spec entries: {missing}",
              file=sys.stderr)

    with open(_ARTIFACT, 'w') as f:
        json.dump(artifact, f, indent=1)
    size_kb = os.path.getsize(_ARTIFACT) // 1024
    print('\n'.join(report_lines))
    print(f"wrote {_ARTIFACT} ({size_kb} KB, {len(artifact['examples'])} examples, "
          f"source={artifact['generated_with']})")
    if args.report:
        with open(args.report, 'w') as f:
            f.write('\n'.join(report_lines) + '\n')


if __name__ == '__main__':
    main()
