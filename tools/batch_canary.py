"""
Batch canary over arbitrary personas, scored through BOTH deployed twins at once.

The `main.py canary` path bakes its personas into src/evaluate.py and reloads the model
per checkpoint. For the popularity-correction blog we want to sweep *many* candidate
personas (new taste profiles) through the α=0 and α=0.5 prod twins and rank them by how
cleanly they show the before/after story — so this loads the 12GB features ONCE, builds
both models in the same process, and reuses the exact canary user-tower path
(seeds @2.0 + genome anchors @1.0 + disliked @-2.0) so a profile scores identically to
`main.py canary`.

Personas come from a JSON list:
    [{"name": "...", "genome_tag": "western"|["a","b"], "liked": ["Title (Year)", ...],
      "disliked": ["Title (Year)", ...]   # optional
     }, ...]

    python tools/batch_canary.py tools/personas/personas.json
        → prints side-by-side α=0 vs α=0.5 top-10 (rating-count badged) per persona
        → writes tools/results/batch_canary.json (structured, for ranking / poster wall)

Short MPS inference job — fine to run in the background.
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.checkpoint import load_checkpoint
from src.dataset import load_features
from src.evaluate import (VALUE_ANCHOR_MOVIE_RATING, VALUE_DISLIKED_MOVIE_RATING,
                          VALUE_FAVORITE_MOVIE_RATING, _get_anchor_titles,
                          build_movie_embeddings)
from src.features import FEATURES_VERSION
from src.inference import build_user_embedding
from src.train import build_model, get_device

ALPHA0_CKPT  = 'saved_models/PROD_NO_ALPHA_best_softmax_genome_tags_llm_features_popularity_alpha_0_20260613_063904.pth'
ALPHA05_CKPT = 'saved_models/PROD_best_softmax_genome_tags_llm_features_popularity_alpha_05_20260612_080719.pth'
TOP_N = 10


def _fmt(n):
    if n is None:
        return '—'
    if n >= 1_000_000:
        return f'{n/1_000_000:.1f}M'
    if n >= 1_000:
        return f'{n/1_000:.1f}k'
    return str(n)


def _load_model(ckpt, fs, device):
    config, state_dict = load_checkpoint(ckpt)
    model = build_model(config, fs)
    model.load_state_dict(state_dict)
    model = model.to(device).eval()
    embs = build_movie_embeddings(model, fs)
    all_ids  = list(embs.keys())
    all_embs = torch.cat([embs[m]['MOVIE_EMBEDDING_COMBINED'] for m in all_ids], dim=0).to(device)
    return model, all_ids, all_embs


def _recommend(model, fs, all_ids, all_embs, liked_w, disliked, exclude, ts_bin, mid_to_count):
    """Top-N (title, count) for one persona on one model — the canary scoring loop."""
    with torch.no_grad():
        ue = build_user_embedding(model, fs, liked_w, disliked, ts_bin,
                                  disliked_movie_value=VALUE_DISLIKED_MOVIE_RATING)
        scores = (all_embs @ ue.T).squeeze(-1)
        order  = torch.argsort(scores, descending=True).tolist()
    recs = []
    for idx in order:
        if len(recs) >= TOP_N:
            break
        title = fs.movieId_to_title[all_ids[idx]]
        if title not in exclude:
            recs.append((title, mid_to_count.get(int(all_ids[idx]))))
    return recs


def _median(recs):
    vals = [c for _, c in recs if c is not None]
    return int(np.median(vals)) if vals else 0


def main():
    personas_path = sys.argv[1] if len(sys.argv) > 1 else 'tools/personas/personas.json'
    personas = json.load(open(personas_path))

    device = get_device()
    print(f"Device: {device}")
    print("Loading features (once) ...")
    fs = load_features('data', FEATURES_VERSION)

    counts = np.load('data/corpus_raw_rating_counts.npy')
    top    = [int(m) for m in fs.top_movies]
    mid_to_count = {top[i]: int(counts[i]) for i in range(len(top))}

    print(f"Loading α=0 twin:   {os.path.basename(ALPHA0_CKPT)}")
    m0, ids0, embs0 = _load_model(ALPHA0_CKPT, fs, device)
    print(f"Loading α=0.5 prod: {os.path.basename(ALPHA05_CKPT)}")
    m5, ids5, embs5 = _load_model(ALPHA05_CKPT, fs, device)

    ts_bin = torch.bucketize(
        torch.tensor([float(fs.timestamp_bins[-1].item())]), fs.timestamp_bins, right=False
    ).to(device)

    out = []
    for p in personas:
        name      = p['name']
        gtags     = p['genome_tag'] if isinstance(p.get('genome_tag'), list) else ([p['genome_tag']] if p.get('genome_tag') else [])
        liked     = p['liked']
        disliked  = p.get('disliked', [])
        anchors   = _get_anchor_titles(fs, gtags, exclude=set(liked))
        liked_w   = ([(t, VALUE_FAVORITE_MOVIE_RATING) for t in liked] +
                     [(t, VALUE_ANCHOR_MOVIE_RATING)   for t in anchors])
        exclude   = set(liked) | set(disliked) | set(anchors)

        recs0 = _recommend(m0, fs, ids0, embs0, liked_w, disliked, exclude, ts_bin, mid_to_count)
        recs5 = _recommend(m5, fs, ids5, embs5, liked_w, disliked, exclude, ts_bin, mid_to_count)
        med0, med5 = _median(recs0), _median(recs5)

        # which seeds were actually found (silently-dropped seeds weaken the persona)
        missing = [t for t in liked if t not in fs.title_to_movieId]

        rec = {
            'name': name, 'genome_tag': gtags, 'liked': liked, 'disliked': disliked,
            'anchors': anchors, 'missing_seeds': missing,
            'alpha0':  [{'title': t, 'count': c} for t, c in recs0],
            'alpha05': [{'title': t, 'count': c} for t, c in recs5],
            'median_alpha0': med0, 'median_alpha05': med5,
            'pop_drop_x': round(med0 / med5, 1) if med5 else None,
        }
        out.append(rec)

        bar = '═' * 100
        print(f"\n{bar}\n{name}   |   genome: {', '.join(gtags)}"
              f"   |   median {_fmt(med0)} → {_fmt(med5)}"
              f"   ({'%.1f×' % (med0/med5) if med5 else 'n/a'} less popular)")
        if missing:
            print(f"  ⚠ dropped seeds (not in corpus): {missing}")
        print(bar)
        print(f"{'α=0  (before)':<52}  {'α=0.5  (after)':<52}")
        print('─' * 106)
        for (t0, c0), (t5, c5) in zip(recs0, recs5):
            l = f"{t0[:42]:<42} {_fmt(c0):>7}"
            r = f"{t5[:42]:<42} {_fmt(c5):>7}"
            print(f"{l}   {r}")

    os.makedirs('tools/results', exist_ok=True)
    out_path = 'tools/results/batch_canary.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n  → wrote {out_path}  ({len(out)} personas)")


if __name__ == '__main__':
    main()
