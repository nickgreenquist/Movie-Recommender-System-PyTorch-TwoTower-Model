"""
Stage 6 — Feature-level analysis (LLM-vs-genome ablation).

Compares the LLM-extracted content features to the human-curated genome tags on
the *same semantic axes* — the schema was derived from the top-discriminability
genome tags (see derive_schema.py), so every LLM dimension records its source
genome tag(s) in data/llm_schema_dimensions.json. That provenance lets us line
the two spaces up directly instead of guessing a mapping.

Two outputs (both read-only; no model needed):

  1. Corpus-level shared-axis agreement — for each of the 132 LLM dims, take its
     source genome tag(s), and compute the Pearson correlation across all 9,375
     corpus movies between the LLM score and the (mean of the) mapped genome
     score(s). Reports the distribution, per-group means, and the best/worst
     axes. High r = LLM and genome measure the same thing on that axis.

  2. Per-movie top tags — for a handful of movies (one per canary persona +
     iconic anchors), the top-10 genome tags vs the top-10 LLM features side by
     side, to show where they agree and diverge at the item level.

Inputs (full corpus):
  data/llm_feature_names_v1.json          — 132 LLM dim names, canonical order
  data/llm_schema_dimensions.json         — per-dim source genome tag(s)
  data/llm_features_claude-code-sonnet_v1.pt — (9376, 132), last row is padding

Usage (from repo root):
    CORPUS=full python -m llm_features.feature_level_analysis
"""
import json
from collections import defaultdict

import numpy as np
import torch

from src.dataset import load_features

# Movies to spot-check: one per canary persona family + iconic anchors.
SPOT_CHECK = [
    '2001: A Space Odyssey', 'Godfather, The (1972)', 'Toy Story (1995)',
    'Mulholland Drive', 'Saw (2004)', 'Sicario (2015)', "Schindler's List",
    'Spirited Away', 'Die Hard (1988)', 'Good, the Bad and the Ugly',
]


def load_llm_artifacts():
    names = json.load(open('data/llm_feature_names_v1.json'))
    sch   = json.load(open('data/llm_schema_dimensions.json'))
    dim2genome, dim2group = {}, {}
    for grp in sch['groups']:
        for d in grp['dimensions']:
            dim2genome[d['name']] = [t['tag'] for t in d['genome_tags']]
            dim2group[d['name']]  = grp['key']
    L = torch.load('data/llm_features_claude-code-sonnet_v1.pt').numpy()[:-1]  # drop padding row
    return names, dim2genome, dim2group, L


def corpus_agreement(names, dim2genome, dim2group, L, G, gname_to_idx):
    rows = []
    for d, name in enumerate(names):
        idxs = [gname_to_idx[g] for g in dim2genome[name] if g in gname_to_idx]
        if not idxs:
            rows.append((name, dim2group[name], None)); continue
        gvec, lvec = G[:, idxs].mean(axis=1), L[:, d]
        if gvec.std() < 1e-9 or lvec.std() < 1e-9:
            rows.append((name, dim2group[name], None)); continue
        rows.append((name, dim2group[name], float(np.corrcoef(lvec, gvec)[0, 1])))

    rs = np.array([r for (_, _, r) in rows if r is not None])
    print("=" * 70)
    print("CORPUS-LEVEL SHARED-AXIS AGREEMENT  (LLM dim vs its source genome tag)")
    print("=" * 70)
    print(f"dims resolvable & with variance: {len(rs)}/132")
    print(f"mean r = {rs.mean():.3f} | median = {np.median(rs):.3f} | "
          f"min = {rs.min():.3f} | max = {rs.max():.3f}")
    print(f"r>=0.5: {(rs >= 0.5).sum()} | 0.3-0.5: {((rs >= 0.3) & (rs < 0.5)).sum()} | "
          f"0.1-0.3: {((rs >= 0.1) & (rs < 0.3)).sum()} | r<0.1: {(rs < 0.1).sum()}")

    ranked = sorted([(n, g, r) for (n, g, r) in rows if r is not None], key=lambda x: -x[2])
    print("\nTOP 15 best-agreement axes:")
    for n, g, r in ranked[:15]:  print(f"  {r:+.3f}  {n:22s} [{g}]")
    print("\nBOTTOM 15 worst-agreement axes:")
    for n, g, r in ranked[-15:]: print(f"  {r:+.3f}  {n:22s} [{g}]")

    gd = defaultdict(list)
    for (_, g, r) in rows:
        if r is not None: gd[g].append(r)
    print("\nMean agreement by group:")
    for g in ['themes', 'tone', 'setting', 'provenance', 'reception', 'visual']:
        if gd[g]: print(f"  {g:12s} mean r={np.mean(gd[g]):+.3f}  (n={len(gd[g])})")


def per_movie(names, L, G, fs, gname_at_idx):
    def resolve(q):
        ql = q.lower()
        cand = [(mid, t) for mid, t in fs.movieId_to_title.items() if ql in t.lower()]
        cand.sort(key=lambda x: len(x[1]))
        return cand[0] if cand else None

    print("\n" + "=" * 70)
    print("PER-MOVIE TOP TAGS: genome (top 10) vs LLM (top 10)")
    print("=" * 70)
    for q in SPOT_CHECK:
        res = resolve(q)
        if not res:
            print(f"\n[{q}] NOT FOUND"); continue
        mid, title = res
        ci   = fs.item_emb_movieId_to_i[mid]
        gtop = sorted(range(len(gname_at_idx)), key=lambda i: -G[ci, i])[:10]
        ltop = sorted(range(len(names)),        key=lambda d: -L[ci, d])[:10]
        print(f"\n### {title}")
        print(f"  {'GENOME (curated)':38s} | LLM (Sonnet)")
        for a, b in zip(gtop, ltop):
            print(f"  {gname_at_idx[a] + f' ({G[ci, a]:.2f})':38s} | {names[b]} ({L[ci, b]:.2f})")


def main():
    names, dim2genome, dim2group, L = load_llm_artifacts()
    fs = load_features()
    assert L.shape[0] == len(fs.top_movies), (L.shape, len(fs.top_movies))

    G = np.array([fs.movieId_to_genome_tag_context[mid] for mid in fs.top_movies], dtype=np.float32)
    gname_at_idx = [fs.genome_tag_names[tid] for tid in fs.genome_tag_ids]
    gname_to_idx = {n: i for i, n in enumerate(gname_at_idx)}

    corpus_agreement(names, dim2genome, dim2group, L, G, gname_to_idx)
    per_movie(names, L, G, fs, gname_at_idx)


if __name__ == '__main__':
    main()
