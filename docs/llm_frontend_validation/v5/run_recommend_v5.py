"""Stage 2 (v5) — batch recommend() over the v5 Haiku extractions.

Mirror of v4_resume/run_recommend_v4_full.py for the v5 subset. Loads serving/ ONCE via the
validated harness Serving loader, runs the real shared pipeline (src/llm_frontend.recommend)
over every extracted case, and writes one cases_v5/case_<id>.json per case for the judge.

    python docs/llm_frontend_validation/v5/run_recommend_v5.py <SP>   # SP = the v5/ dir
"""
import sys, os, json

sp = sys.argv[1]
REPO = os.getcwd()
sys.path.insert(0, REPO)

from tools.llm_frontend_probe import Serving           # the validated serving/ loader
from src.llm_frontend import recommend

srv = Serving()                                          # load serving/ once
cases = json.load(open(os.path.join(sp, 'extractions_v5_subset.json')))
os.makedirs(os.path.join(sp, 'cases_v5'), exist_ok=True)

errs = 0
written_ids = []
for c in cases:
    if c.get('error') or c.get('extraction') is None:
        errs += 1
        print('  SKIP id', c.get('id'), '-', c.get('error', 'no extraction'))
        continue
    ex = c['extraction']
    r = recommend(srv.ctx, ex, top_n=15)
    recs = [{'title': t, 'genres': g, 'year': y, 'cos': s} for (t, g, y, s) in r['recs']]
    out = {
        'id': c['id'], 'cat': c['cat'], 'utterance': c['text'],
        'extraction': ex,
        'recs': recs,
        'resolution': r['resolution'],
        'anchors': r['anchors'],
        'anchor_weight': r['anchor_weight'],
        'unresolved_tags': r['unresolved_tags'],
        'unknown_genres': r['unknown_genres'],
        'fallback': r['fallback'],
        'filtered': r['filtered'],
    }
    json.dump(out, open(os.path.join(sp, 'cases_v5', f"case_{c['id']}.json"), 'w'), indent=1)
    written_ids.append(c['id'])

json.dump(sorted(written_ids), open(os.path.join(sp, 'case_ids_v5.json'), 'w'))
print('wrote', len(written_ids), 'cases | extraction errors:', errs)
