"""v5 subset comparison vs v1/v3/v4 on the same cases.

  python docs/llm_frontend_validation/v5/compare_v5.py <V4DIR> <V5DIR> <v5_judge_output>

V4DIR = the v4_resume dir (holds judge_summary{,_v3_full,_v4_full}.json + cases_v4_full for cats).
V5DIR = the v5 dir (holds cases_v5; receives judge_summary_v5_subset.json).
v5_judge_output = the wf_judge_v5 workflow output file (has a top-level 'result').
"""
import sys, json, glob
from collections import defaultdict
v4dir, v5dir, v5out = sys.argv[1], sys.argv[2], sys.argv[3]

def load_scores(path):
    j = json.load(open(path))
    return {s['id']: s for s in j['all_scores'] if isinstance(s.get('rr'), int)}

s1 = load_scores(f"{v4dir}/judge_summary.json")          # v1
s3 = load_scores(f"{v4dir}/judge_summary_v3_full.json")  # v3
s4 = load_scores(f"{v4dir}/judge_summary_v4_full.json")  # v4

w = json.load(open(v5out)); res = w['result']
if isinstance(res, str): res = json.loads(res)
json.dump(res, open(f"{v5dir}/judge_summary_v5_subset.json", "w"), indent=1)
s5 = {s['id']: s for s in res['all_scores'] if isinstance(s.get('rr'), int)}

# Subset = the ids v5 actually judged.
subset = sorted(s5)
# Cats from the v5 case files.
cats = {}
for f in glob.glob(f"{v5dir}/cases_v5/case_*.json"):
    c = json.load(open(f)); cats[c['id']] = c['cat']

print(f"v5 judged {len(s5)} | overlap v1={len(set(subset)&set(s1))} v3={len(set(subset)&set(s3))} v4={len(set(subset)&set(s4))}\n")

NAMES = {'ic': 'intent', 'tq': 'tags', 'rq': 'resolution', 'rr': 'recs', 'cr': 'constraints'}
def mean(d, ids, k):
    ids = [i for i in ids if i in d]
    return sum(d[i][k] for i in ids) / len(ids) if ids else float('nan')

print("AGGREGATE (v5 subset)   v1     v3     v4     v5    Δ(v5-v4)")
for k in ['ic', 'tq', 'rq', 'rr', 'cr']:
    a, b, c, e = (mean(d, subset, k) for d in (s1, s3, s4, s5))
    print(f"  {NAMES[k]:12}      {a:4.2f}   {b:4.2f}   {c:4.2f}   {e:4.2f}    {e-c:+.2f}")

print("\nrecs by CATEGORY        n   v1    v3    v4    v5    Δ(v5-v4)")
by = defaultdict(list)
for i in subset:
    by[cats.get(i, '?')].append(i)
for cat in sorted(by):
    ids = by[cat]; n = len(ids)
    a, b, c, e = (mean(d, ids, 'rr') for d in (s1, s3, s4, s5))
    print(f"  {cat:13}        {n:>2}  {a:.2f}  {b:.2f}  {c:.2f}  {e:.2f}   {e-c:+.2f}")

print("\nv4→v5 per-case recs changes (the gate's effect):")
for i in subset:
    if i in s4 and s5[i]['rr'] != s4[i]['rr']:
        arrow = 'UP  ' if s5[i]['rr'] > s4[i]['rr'] else 'DOWN'
        print(f"  [{i:>3}] {cats.get(i,''):<11} {arrow} {s4[i]['rr']}->{s5[i]['rr']}  v5:{s5[i]['mode']}")

print("\nv5 severity (subset):", {sv: sum(1 for i in subset if s5[i]['sev'] == sv)
                                   for sv in ['critical', 'major', 'minor', 'none']})
