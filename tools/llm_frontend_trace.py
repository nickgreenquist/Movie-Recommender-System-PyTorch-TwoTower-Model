"""
tools/llm_frontend_trace.py — end-to-end TRACE of the LLM front-end for a single NL query.

PURPOSE
    A no-API-key dry-run of the production Streamlit "Ask" flow, for inspecting what the
    pipeline actually does with a free-text request. The extraction step is performed by a
    Haiku-class SUBAGENT in the Claude Code loop (the same cheap model family the hosted path
    feeds via build_system_prompt()), and THIS tool renders everything downstream of that
    extraction — model inputs, hard filters, and the final top-N the user would see — into a
    human-readable Markdown trace.

FLOW (what this reproduces, stage → trace section)
    free-text utterance                                                              (§1)
      → [orchestrator] Haiku subagent runs build_system_prompt() → extraction JSON   (§2)
      → recommend() over the real serving model: resolve titles/people/facets,
        synthesize Mode-2 anchors, build the user embedding, rank, post-filter       (§3, §4)
      → top-N recommendations                                                        (§5)

    The extraction subagent is the ONLY LLM call and it needs no API key (it runs in the
    Claude Code loop). recommend() is pure serving/torch. So the whole trace runs offline
    and mirrors what the deployed app would produce for the same extraction.

USAGE (run from repo root)
    # 1. Print the exact Haiku system prompt (constant across queries) to hand to a subagent:
    python tools/llm_frontend_trace.py --emit-prompt

    # 2. Render the trace from that subagent's extraction JSON:
    python tools/llm_frontend_trace.py --utterance "movies like Oldboy, nothing too gory" \
        --json '{"liked_items":["Oldboy (2003)"]}'
    echo '{"genome_tags":["western"]}' | python tools/llm_frontend_trace.py --utterance "..."

    --out FILE   write the trace to FILE (default: tools/results/traces/<ts>_<slug>.md)
    --log FILE   APPEND the trace to FILE (accumulate a session of queries in one file)
    --top-n N    how many recs to surface (default TOP_N)
    --quiet      don't echo the trace to stdout (still writes the file)
"""

import argparse
import datetime
import json
import os
import re
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.llm_frontend import (  # noqa: E402
    DISLIKED_MOVIE_WEIGHT,
    GENOME_EXCLUDE_CEILING,
    GENOME_HARD_FLOOR,
    LIKED_MOVIE_WEIGHT,
    TOP_N,
    _resolve_exclude_slots,
    recommend,
)
from src.llm_frontend_prompt import build_system_prompt  # noqa: E402
from tools.llm_frontend_probe import Serving  # noqa: E402

_TRACE_DIR = os.path.join(_REPO_ROOT, 'tools', 'results', 'traces')


# ── Markdown rendering ───────────────────────────────────────────────────────
def _fmt_kv(label, value):
    return f"- **{label}:** {value}"


def _resolution_lines(report):
    """Human-readable notes on what each named entity resolved to (or why it was dropped)."""
    out = []
    for kind in ('liked', 'disliked'):
        for raw, canon, note in report['resolution'][kind]:
            arrow = f"→ `{canon}`" if canon else "→ **(unresolved, dropped)**"
            out.append(f"  - title `{raw}` {arrow}  _[{note}]_")
    pr = report.get('people_resolution') or {}
    for bucket in ('require', 'exclude'):
        for raw, pid, note in pr.get(bucket, []):
            arrow = f"→ person #{pid}" if pid is not None else "→ **(unresolved, dropped)**"
            out.append(f"  - {bucket} person `{raw}` {arrow}  _[{note}]_")
    fr = report.get('facet_resolution') or {}
    for bucket in ('country', 'language', 'attribute'):
        for phrase, vals, note in fr.get(bucket, []):
            arrow = f"→ {vals}" if vals else "→ **(unresolved, dropped)**"
            out.append(f"  - {bucket} `{phrase}` {arrow}  _[{note}]_")
    tr = report.get('topic_resolution') or {}
    for bucket in ('require', 'exclude'):
        for phrase, note, size in tr.get(bucket, []):
            arrow = (f"→ {note} ({size} films)" if size is not None
                     else f"→ **(unresolved, dropped)**  _[{note}]_")
            out.append(f"  - {bucket} topic `{phrase}` {arrow}")
    return out


def render_trace(report, utterance, top_n):
    """Render the full pipeline trace for one query as a Markdown string (the 5 sections)."""
    ex = report['extraction']
    hc = ex.get('hard_constraints') or {}
    fallback = report['fallback']
    stamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    L = []

    L.append(f"# LLM Front-End Trace — {utterance!r}")
    L.append(f"_generated {stamp} · serving model · extraction by Haiku-class subagent (no API key)_")
    L.append("")

    # §1 User prompt
    L.append("## 1. User prompt")
    L.append(f"> {utterance}")
    L.append("")

    # §2 Extraction (the LLM's structured output)
    L.append("## 2. Extraction — Haiku subagent output")
    L.append("_(the LLM's ONLY job; consumed internally, never shown to the end user)_")
    L.append("```json")
    L.append(json.dumps(ex, indent=2, ensure_ascii=False))
    L.append("```")
    res = _resolution_lines(report)
    if res:
        L.append("**Resolution:**")
        L.extend(res)
    for warn, key in (("genome tags NOT in vocab (ignored)", 'unresolved_tags'),
                      ("genre names NOT in model vocab (ignored)", 'unknown_genres')):
        if report.get(key):
            L.append(f"  - ⚠ {warn}: {report[key]}")
    L.append("")

    # §3 Model inputs (two-tower user tower)
    L.append("## 3. Model inputs — what the two-tower user tower receives")
    if fallback:
        L.append("- **ranking source:** ⚠ POPULARITY FALLBACK — no taste signal, no user embedding built "
                 "(the corpus is ranked by popularity within the hard filters below)")
    else:
        L.append("- **ranking source:** model retrieval (user embedding → cosine over the corpus)")
        likes = [c for _, c, _ in report['resolution']['liked'] if c]
        dislikes = [c for _, c, _ in report['resolution']['disliked'] if c]
        if likes:
            L.append(_fmt_kv(f"liked seeds (weight {LIKED_MOVIE_WEIGHT:+})", ", ".join(f"`{t}`" for t in likes)))
        if dislikes:
            L.append(_fmt_kv(f"disliked seeds (weight {DISLIKED_MOVIE_WEIGHT:+})",
                             ", ".join(f"`{t}`" for t in dislikes)))
        if report['anchors']:
            L.append(_fmt_kv(f"Mode-2 genome anchors (synthesized, weight {report['anchor_weight']:+})",
                             ", ".join(f"`{t}`" for t in report['anchors'])))
        if report.get('mood_tags'):
            L.append(_fmt_kv("mood/affect tags (Engine-2, soft re-rank)", report['mood_tags']))
        if report.get('mode15_tags'):
            L.append(_fmt_kv("Mode-1.5 title-genome tags (soft re-rank)", report['mode15_tags']))
        if ex.get('liked_genres'):
            L.append(_fmt_kv("liked_genres (user genre tower)", ex['liked_genres']))
        if ex.get('disliked_genres'):
            L.append(_fmt_kv("disliked_genres (user genre tower)", ex['disliked_genres']))
        # active soft re-rank terms
        subject = list(ex.get('genome_tags') or []) + list(hc.get('require_genome_tags') or []) \
            + list(report.get('mode15_tags') or [])
        active = []
        if subject:
            active.append("genome-subject λ=0.5")
        if report.get('mood_tags'):
            active.append("mood λ=0.5")
        if ex.get('liked_genres') or hc.get('require_genres'):
            active.append("genre λ=0.3")
        if active:
            L.append(_fmt_kv("soft re-rank terms active", " · ".join(active)))
    L.append("")

    # §4 Hard filters (post-retrieval)
    L.append("## 4. Hard filters — applied as post-retrieval gates")
    f = []
    ymin, ymax = hc.get('year_min'), hc.get('year_max')
    if ymin is not None or ymax is not None:
        f.append(_fmt_kv("year", f"{ymin or '–'} … {ymax or '–'}"))
    if hc.get('require_genres'):
        f.append(_fmt_kv("require_genres (ALL)", hc['require_genres']))
    if hc.get('exclude_genres'):
        f.append(_fmt_kv("exclude_genres (ANY)", hc['exclude_genres']))
    fr = report.get('facet_resolution') or {}

    def _codes(bucket):
        return sorted({v for _, vals, _ in fr.get(bucket, []) for v in (vals or [])})
    if _codes('country'):
        f.append(_fmt_kv("require_country → codes (ANY)", _codes('country')))
    if _codes('language'):
        f.append(_fmt_kv("require_language → codes (ANY)", _codes('language')))
    if _codes('attribute'):
        f.append(_fmt_kv("require_attributes → keys (ALL)", _codes('attribute')))
    pr = report.get('people_resolution') or {}
    req_p = [pid for _, pid, _ in pr.get('require', []) if pid is not None]
    exc_p = [pid for _, pid, _ in pr.get('exclude', []) if pid is not None]
    if req_p:
        f.append(_fmt_kv("require_people → ids (ALL)", req_p))
    if exc_p:
        f.append(_fmt_kv("exclude_people → ids (ANY)", exc_p))
    for key, lbl in (('require_franchise', 'require_franchise'), ('exclude_franchise', 'exclude_franchise'),
                     ('require_max_rating', 'require_max_rating (MPAA ceiling)'),
                     ('max_runtime', 'max_runtime (min)'), ('min_runtime', 'min_runtime (min)'),
                     ('min_vote_average', 'min_vote_average (quality floor)')):
        if hc.get(key) is not None and hc.get(key) != []:
            f.append(_fmt_kv(lbl, hc[key]))
    if hc.get('require_genome_tags'):
        f.append(_fmt_kv(f"require_genome_tags — HARD floor (each ≥ {GENOME_HARD_FLOOR})",
                         hc['require_genome_tags']))
    tr = report.get('topic_resolution') or {}
    req_topics = [(p, n, s) for p, n, s in tr.get('require', []) if s is not None]
    exc_topics = [(p, n, s) for p, n, s in tr.get('exclude', []) if s is not None]
    if req_topics:
        f.append(_fmt_kv("topic pool — resolver member sets, OR across terms",
                         ", ".join(f"`{p}` → {n} ({s})" for p, n, s in req_topics)))
    if exc_topics:
        f.append(_fmt_kv("topic exclusion — resolver member sets (drop ANY)",
                         ", ".join(f"`{p}` → {n} ({s})" for p, n, s in exc_topics)))
    exclude_tags = _resolve_exclude_slots(hc)
    if exclude_tags:
        f.append(_fmt_kv(f"exclude_mood / exclude_genome_tags — anti-floor (drop if ≥ {GENOME_EXCLUDE_CEILING} "
                         f"on ANY)", exclude_tags))
    if not f:
        f.append("- _(none — no hard constraints extracted)_")
    L.extend(f)
    L.append("")
    L.append(_fmt_kv("candidates dropped by filters", report['filtered']))
    if report.get('ranked_by_similarity'):
        L.append(_fmt_kv("ranking", "item-item genome content similarity ('movies like X', Similar-tab space)"))
    if report.get('relaxed_constraints'):
        L.append(_fmt_kv("⚠ empty pool → relaxed (closest matches)", ", ".join(report['relaxed_constraints'])))
    L.append("")

    # §5 Recommendations (what the user sees)
    n = len(report['recs'])
    src = ('genome similarity' if report.get('ranked_by_similarity')
           else 'popularity fallback' if fallback else 'model retrieval')
    L.append(f"## 5. Recommendations — top {n} _(source: {src}; this is what the user sees)_")
    L.append("")
    L.append("| # | Title | Year | Genres | cos |")
    L.append("|--:|-------|-----:|--------|----:|")
    for rank, (title, genres, year, score) in enumerate(report['recs'][:top_n], 1):
        sc = f"{score:+.3f}" if score is not None else "—"
        g = ", ".join(genres) if genres else ""
        L.append(f"| {rank} | {title} | {year or '—'} | {g} | {sc} |")
    L.append("")
    return "\n".join(L)


# ── I/O ──────────────────────────────────────────────────────────────────────
def _slug(s, n=40):
    return (re.sub(r'[^a-z0-9]+', '-', (s or 'query').lower()).strip('-') or 'query')[:n]


def _read_extraction(args):
    if args.json:
        return json.loads(args.json)
    if args.json_file:
        with open(args.json_file) as fh:
            return json.load(fh)
    return json.load(sys.stdin)


def main():
    ap = argparse.ArgumentParser(description='End-to-end trace of the LLM front-end (serving/-only).')
    ap.add_argument('--emit-prompt', action='store_true',
                    help='print the exact Haiku extraction system prompt (hand it to a subagent) and exit')
    ap.add_argument('--utterance', default='(no utterance given)', help='the raw NL request, for §1')
    ap.add_argument('--json', help='inline extraction JSON string (the subagent output)')
    ap.add_argument('--json-file', help='path to a file containing extraction JSON')
    ap.add_argument('--out', help='write the trace to this file (default: tools/results/traces/<ts>_<slug>.md)')
    ap.add_argument('--log', help='APPEND the trace to this file (accumulate a session in one file)')
    ap.add_argument('--top-n', type=int, default=TOP_N)
    ap.add_argument('--quiet', action='store_true', help="don't echo the trace to stdout")
    args = ap.parse_args()

    if args.emit_prompt:
        print(build_system_prompt())
        return

    srv = Serving()
    extraction = _read_extraction(args)
    report = recommend(srv.ctx, extraction, top_n=args.top_n)
    trace = render_trace(report, args.utterance, args.top_n)

    if args.log:
        path = args.log
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, 'a') as fh:
            fh.write(trace + "\n\n---\n\n")
    else:
        os.makedirs(_TRACE_DIR, exist_ok=True)
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        path = args.out or os.path.join(_TRACE_DIR, f"{ts}_{_slug(args.utterance)}.md")
        with open(path, 'w') as fh:
            fh.write(trace + "\n")

    if not args.quiet:
        print(trace)
    print(f"\n[trace written to {os.path.relpath(path, _REPO_ROOT)}]", file=sys.stderr)


if __name__ == '__main__':
    main()
