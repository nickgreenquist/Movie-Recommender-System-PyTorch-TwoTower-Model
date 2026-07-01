"""tools/llm_frontend_eval.py — deterministic, NO-API regression + spec eval for the LLM front-end.

PURPOSE
    The v1→v5 validation showed the residual recs gap is harness-side (anchor re-weighting,
    genome/structured facets) — NOT the extraction prompt. Those fixes need a *ruler* that does
    not burn an API key on every iteration. This runner is that ruler: it feeds HAND-AUTHORED
    extraction JSON straight into recommend() over the real serving model and checks each result
    against machine-checkable assertions. No Haiku call, no Streamlit — pure retrieval logic.

    It doubles as a forward SPEC: cases whose `needs_feature` is not "none" exercise slots that
    aren't built yet (require_genome_tags, require_max_rating, …). recommend() ignores unknown
    hard-constraint keys, so those cases fail today and turn green as each feature lands — the eval
    measures build progress. The summary separates REGRESSION (needs_feature==none, must be green)
    from SPEC (expected-fail-until-built).

USAGE
    python tools/llm_frontend_eval.py [cases.json ...] [--verbose] [--only-regression]
    (default cases: docs/llm_frontend/validation/retrieval_eval/eval_cases.json)

CASE / ASSERTION FORMAT
    A case = { utterance, extraction, assertions:[…], needs_feature, note }. Assertion types are
    documented in _check_assertion below; each is evaluated over recommend()'s top-k titles + the
    serving metadata (genres in the rec tuple, people via the facet store, per-movie genome scores).
"""
import os, sys, json, re

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

from tools.llm_frontend_probe import Serving
from src.llm_frontend import recommend, resolve_person, MPAA_ORDER, _franchise_match

_DEFAULT_CASES = os.path.join(
    _REPO_ROOT, 'docs/llm_frontend/validation/retrieval_eval/eval_cases.json')
EVAL_TOP_N = 50  # ask recommend() for a deep list so rank_above / large-k assertions can resolve


# ── serving-metadata accessors (built once from the loaded feature_store) ────────────────────────
class _Meta:
    """Lookups the assertion checker needs, derived from the loaded serving feature_store + facets."""

    def __init__(self, srv):
        self.fs = srv.fs
        self.facets = srv.ctx.facets
        self.title_to_mid = self.fs['title_to_movieId']
        self.mid_to_genres = self.fs['movieId_to_genres']
        # genome tag NAME → per-movie score column, for min_genome
        self._name_to_col = {}
        gn, gi = self.fs['genome_tag_names'], self.fs['genome_tag_to_i']
        for tid, name in gn.items():
            if tid in gi:
                self._name_to_col[name.lower()] = gi[tid]
        self.mid_to_genome = self.fs.get('movieId_to_genome_tag_context')

    def genome_score(self, title, tag):
        """Relevance (0–1) of `tag` for `title`, or None if the title/tag/vector is unavailable."""
        col = self._name_to_col.get(tag.lower())
        mid = self.title_to_mid.get(title)
        if col is None or mid is None or not self.mid_to_genome:
            return None
        vec = self.mid_to_genome.get(mid)
        try:
            return float(vec[col])
        except Exception:
            return None

    def _facet(self, table, title):
        """Value of facet `table` (a movieId-keyed facets dict) for `title`, or None."""
        if not self.facets:
            return None
        mid = self.title_to_mid.get(title)
        return (self.facets.get(table) or {}).get(mid) if mid is not None else None

    def runtime(self, title):
        return self._facet('movieId_to_runtime', title)

    def content_rating(self, title):
        return self._facet('movieId_to_content_rating', title)

    def in_franchise(self, title, spec):
        """True if `title`'s TMDB collection matches the franchise `spec` (True / name / [name…]),
        using the same _franchise_match the real post-filter uses (universe aliases included)."""
        if not self.facets:
            return False
        mid = self.title_to_mid.get(title)
        coll = (self.facets.get('movieId_to_collection') or {}).get(mid) if mid is not None else None
        return _franchise_match(coll, spec, self.facets.get('franchise_universe_aliases') or {})

    def people_ids(self, title):
        """Set of TMDB person IDs (actors ∪ directors ∪ writers ∪ composers) for `title`, from the
        facet store. Composers are unioned in so all_have_person "Hans Zimmer" checks the score
        credit — the same union recommend()'s people filter uses."""
        if not self.facets:
            return None
        mid = self.title_to_mid.get(title)
        roles = self.facets['movieId_to_people'].get(mid) if mid is not None else None
        if not roles:
            return set()
        return (set(roles.get('actors', [])) | set(roles.get('directors', []))
                | set(roles.get('writers', [])) | set(roles.get('composers', [])))


def _norm(s):
    return (s or '').lower()


# ── assertion checker ────────────────────────────────────────────────────────────────────────────
def _check_assertion(a, recs, meta):
    """Evaluate one assertion over recs (list of (title, genres, year, score)). Returns (ok, detail).

    Types:
      contains_genre {genre,k,min}          excludes_genre {genre,k}        max_genre {genre,k,max}
      contains_title_substr {substr,k,min}  excludes_title_substr {substr,k}
      all_have_person {name,k}              none_have_person {name,k}
      rank_above {a,b}                      min_genome {tag,k,min_score,min}
      max_runtime {cap,k}                   max_rating {rating,k}
      excludes_franchise {spec,k}           oracle {title,facet,equals[,spec]}
    Facet-membership assertions (max_runtime / max_rating / excludes_franchise) test the F1
    structured facets directly — the plan's PRIMARY metric (a hard facet's membership IS its
    correctness), not the genre proxies the other assertions use. A film with the facet UNKNOWN
    passes, BUT an empty pool or an all-unknown top-k FAILS (an over-pruning filter must not read
    green vacuously). `oracle` is the independent ground-truth check (facet value vs a literal),
    the only assertion that doesn't consult the same filter path.
    """
    t = a.get('type')
    titles = [r[0] for r in recs]
    genres = [set(r[1] or []) for r in recs]
    k = int(a.get('k', len(recs)))

    if t == 'contains_genre':
        n = sum(1 for g in genres[:k] if a['genre'] in g)
        need = int(a.get('min', 1))
        return n >= need, f"{n} of top-{k} are {a['genre']} (need >={need})"
    if t == 'excludes_genre':
        bad = [titles[i] for i in range(min(k, len(recs))) if a['genre'] in genres[i]]
        return not bad, f"{len(bad)} {a['genre']} in top-{k}" + (f": {bad[:3]}" if bad else "")
    if t == 'max_genre':
        n = sum(1 for g in genres[:k] if a['genre'] in g)
        mx = int(a['max'])
        return n <= mx, f"{n} of top-{k} are {a['genre']} (allow <={mx})"
    if t == 'contains_title_substr':
        s = _norm(a['substr']); n = sum(1 for x in titles[:k] if s in _norm(x))
        need = int(a.get('min', 1))
        return n >= need, f"{n} of top-{k} titles contain '{a['substr']}' (need >={need})"
    if t == 'excludes_title_substr':
        s = _norm(a['substr']); hit = [x for x in titles[:k] if s in _norm(x)]
        return not hit, f"'{a['substr']}' in top-{k}: {hit[:3]}" if hit else f"'{a['substr']}' absent"
    if t in ('all_have_person', 'none_have_person'):
        if not meta.facets:
            return False, "no facet store loaded"
        pid, note = resolve_person(a['name'], meta.facets)
        if pid is None:
            return False, f"unresolved person '{a['name']}' [{note}]"
        hits = [titles[i] for i in range(min(k, len(recs))) if pid in (meta.people_ids(titles[i]) or set())]
        if t == 'all_have_person':
            miss = [titles[i] for i in range(min(k, len(recs))) if pid not in (meta.people_ids(titles[i]) or set())]
            return not miss, f"{len(hits)}/{min(k,len(recs))} have {a['name']}" + (f"; miss {miss[:3]}" if miss else "")
        return not hits, f"{a['name']} in top-{k}: {hits[:3]}" if hits else f"{a['name']} absent"
    if t == 'rank_above':
        ia = next((i for i, x in enumerate(titles) if _norm(a['a']) in _norm(x)), None)
        ib = next((i for i, x in enumerate(titles) if _norm(a['b']) in _norm(x)), None)
        if ia is None or ib is None:
            return False, f"missing in results (a@{ia}, b@{ib})"
        return ia < ib, f"'{a['a']}'@{ia} vs '{a['b']}'@{ib}"
    if t == 'min_genome':
        thr = float(a.get('min_score', 0.5)); need = int(a.get('min', 1))
        scored = [(x, meta.genome_score(x, a['tag'])) for x in titles[:k]]
        if all(s is None for _, s in scored):
            return False, f"genome tag '{a['tag']}' not in vocab / no scores"
        n = sum(1 for _, s in scored if s is not None and s >= thr)
        return n >= need, f"{n} of top-{k} carry '{a['tag']}'>={thr} (need >={need})"
    if t == 'max_runtime':
        cap = int(a['cap'])
        if not recs:
            return False, "no recs (empty pool — vacuous)"
        known = [(titles[i], meta.runtime(titles[i])) for i in range(min(k, len(recs)))]
        known = [(x, r) for x, r in known if r is not None]
        if not known:  # all-unknown can't stand in for 'filter worked' (cf. min_genome's guard)
            return False, f"no top-{k} film has a known runtime — cannot verify {cap} min cap"
        bad = [(x, r) for x, r in known if r > cap]
        return not bad, (f"{len(bad)} of top-{k} exceed {cap} min: {bad[:3]}" if bad
                         else f"all {len(known)} known-runtime top-{k} within {cap} min")
    if t == 'max_rating':
        ceil = MPAA_ORDER.get(str(a['rating']).strip().upper())
        if ceil is None:
            return False, f"unknown rating ceiling '{a['rating']}'"
        if not recs:
            return False, "no recs (empty pool — vacuous)"
        known = [(titles[i], meta.content_rating(titles[i])) for i in range(min(k, len(recs)))]
        known = [(x, c) for x, c in known if c]
        if not known:
            return False, f"no top-{k} film has a known rating — cannot verify {a['rating']} ceiling"
        bad = [(x, c) for x, c in known if MPAA_ORDER.get(c, 0) > ceil]
        return not bad, (f"{len(bad)} of top-{k} above {a['rating']}: {bad[:3]}" if bad
                         else f"all {len(known)} rated top-{k} <= {a['rating']}")
    if t == 'excludes_franchise':
        spec = a['spec']
        if not recs:
            return False, "no recs (empty pool — vacuous)"
        hit = [titles[i] for i in range(min(k, len(recs))) if meta.in_franchise(titles[i], spec)]
        return not hit, (f"{len(hit)} of top-{k} in franchise {spec!r}: {hit[:3]}" if hit
                         else f"no top-{k} in franchise {spec!r}")
    if t == 'oracle':
        # Independent ground-truth check on a KNOWN film's facet value — the one assertion that does
        # NOT read through the same filter path, so it catches a build_facet_store extraction bug
        # (wrong unit / mis-parsed cert / collection mismatch) the self-consistent facet gates miss.
        title, facet, exp = a['title'], a['facet'], a['equals']
        if facet == 'runtime':
            got = meta.runtime(title)
        elif facet == 'content_rating':
            got = meta.content_rating(title)
        elif facet == 'in_franchise':
            got = meta.in_franchise(title, a['spec'])
        else:
            return False, f"unknown oracle facet '{facet}'"
        return got == exp, f"{facet}({title!r}) = {got!r} (expect {exp!r})"
    return False, f"UNKNOWN assertion type '{t}'"


# ── runner ───────────────────────────────────────────────────────────────────────────────────────
def run_eval(cases_paths, verbose=False, only_regression=False):
    srv = Serving()
    meta = _Meta(srv)
    cases = []
    for p in cases_paths:
        cases.extend(json.load(open(p)))
    print(f"Loaded {len(cases)} cases from {len(cases_paths)} file(s). facet store: "
          f"{'baked' if srv.ctx.facets else 'ABSENT'}\n" + "=" * 84)

    buckets = {'regression': [0, 0], 'spec': [0, 0]}  # [passed, total]
    for c in cases:
        needs = (c.get('needs_feature') or 'none').strip()
        # A case whose named feature has SHIPPED carries `built: true` — it moves into the
        # must-stay-green regression bucket so a regression in a built facet fails loudly (and is
        # not skipped by --only-regression), rather than hiding as an "expected-fail-until-built" SPEC.
        bucket = 'regression' if (needs in ('', 'none') or c.get('built')) else 'spec'
        if only_regression and bucket == 'spec':
            continue
        try:
            recs = recommend(srv.ctx, c.get('extraction', {}), top_n=EVAL_TOP_N)['recs']
        except Exception as e:
            buckets[bucket][1] += 1
            print(f"[ERROR ] {c.get('utterance','')[:60]!r}: recommend() raised {type(e).__name__}: {e}")
            continue
        results = [(_check_assertion(a, recs, meta), a) for a in c.get('assertions', [])]
        ok = all(r[0][0] for r in results)
        buckets[bucket][0] += int(ok); buckets[bucket][1] += 1
        tag = 'PASS' if ok else ('SPEC-FAIL' if bucket == 'spec' else 'FAIL')
        if ok and not verbose:
            continue
        flag = '' if needs in ('', 'none') else f'  (needs: {needs})'
        print(f"[{tag:9s}] {c.get('utterance','')[:66]!r}{flag}")
        for (passed, detail), a in results:
            if not passed or verbose:
                print(f"           {'ok ' if passed else 'MISS'} {a.get('type'):22s} {detail}")
        if verbose:
            print(f"           top: {[r[0] for r in recs[:8]]}")

    print("=" * 84)
    for b in ('regression', 'spec'):
        if only_regression and b == 'spec':
            continue
        p, tot = buckets[b]
        label = 'REGRESSION (must stay green)' if b == 'regression' else 'SPEC (expected-fail until built)'
        print(f"  {label:34s} {p}/{tot} passing" + (f"  ({tot-p} not-yet-built)" if b == 'spec' and tot else ""))
    return buckets


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    flags = {a for a in sys.argv[1:] if a.startswith('--')}
    paths = args or [_DEFAULT_CASES]
    paths = [p for p in paths if os.path.exists(p)]
    if not paths:
        print(f"No case files found (looked for {args or [_DEFAULT_CASES]}).")
        sys.exit(1)
    run_eval(paths, verbose='--verbose' in flags, only_regression='--only-regression' in flags)
