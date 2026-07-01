"""Deterministic post-filter ablation on the 32 v5 subset cases (NO agents).

For each case: run recommend() with the hard post-filter ON and OFF, and measure
  - fallback?          (empty user embedding → model bypassed, recs = popularity+filter)
  - pool               (# of the ~9375 catalog movies that pass the hard filter)
  - off_violations     (of the filter-OFF top-15, how many VIOLATE the stated constraint
                        — i.e. how many wrong items the filter had to remove)
  - on/off overlap     (how much the filter changes the surfaced list)

Run: python docs/llm_frontend/validation/v5/ablate_postfilter.py
"""
import sys, os, json
sys.path.insert(0, os.getcwd())

from tools.llm_frontend_probe import Serving
import src.llm_frontend as L
from src.llm_frontend import recommend

srv = Serving()
ctx = srv.ctx
fs = ctx.fs
real_passes = L._passes_constraints                      # the real year+genre gate

def has_hard(hc):
    return bool((hc.get('require_genres') or hc.get('exclude_genres')
                 or hc.get('year_min') is not None or hc.get('year_max') is not None))

def pool_size(hc):
    return sum(1 for mid in fs['top_movies'] if real_passes(mid, fs, hc))

exts = json.load(open('docs/llm_frontend/validation/v5/extractions_v5_subset.json'))
rows = []
for c in exts:
    if c.get('error'):
        continue
    ex = c['extraction']; hc = ex.get('hard_constraints') or {}
    hh = has_hard(hc)

    L._passes_constraints = real_passes                  # ON
    on = recommend(ctx, ex, top_n=15)
    L._passes_constraints = lambda mid, fs, hc, *_: True  # OFF (*_ absorbs movieId_to_people)
    off = recommend(ctx, ex, top_n=15)
    L._passes_constraints = real_passes

    on_titles  = [t for (t, *_ ) in on['recs']]
    off_titles = [t for (t, *_ ) in off['recs']]
    off_mids   = [fs['title_to_movieId'][t] for t in off_titles if t in fs['title_to_movieId']]
    viol = sum(1 for mid in off_mids if not real_passes(mid, fs, hc)) if hh else 0
    rows.append({
        'id': c['id'], 'cat': c['cat'], 'text': c['text'][:34], 'hard': hh,
        'fallback': on['fallback'], 'filtered_on': on['filtered'],
        'pool': pool_size(hc) if hh else len(fs['top_movies']),
        'off_viol': viol, 'overlap': len(set(on_titles) & set(off_titles)),
    })

print(f"{'id':>3} {'cat':11} {'request':34} {'hard':4} {'fb':3} {'pool':>5} {'OFF-viol/15':>11} {'on∩off':>6}")
for r in sorted(rows, key=lambda r: (r['cat'], r['id'])):
    print(f"{r['id']:>3} {r['cat']:11} {r['text']:34} {str(r['hard'])[0]:4} "
          f"{str(r['fallback'])[0]:3} {r['pool']:>5} {r['off_viol']:>9}/15 {r['overlap']:>6}")

hard = [r for r in rows if r['hard']]
fb_hard = [r for r in hard if r['fallback']]
print(f"\n— {len(rows)} cases | {len(hard)} have a hard constraint | "
      f"{len(fb_hard)} of those are fallback (model bypassed: recs = popularity+filter)")
if hard:
    avg_viol = sum(r['off_viol'] for r in hard) / len(hard)
    avg_ovl  = sum(r['overlap'] for r in hard) / len(hard)
    print(f"— hard-constraint cases, filter OFF: avg {avg_viol:.1f}/15 recs VIOLATE the request; "
          f"avg on∩off overlap {avg_ovl:.1f}/15")
    thin = [r for r in hard if r['pool'] < 50]
    if thin:
        print(f"— thin pools (<50 movies pass the filter): " +
              ", ".join(f"{r['id']}({r['pool']})" for r in thin))
