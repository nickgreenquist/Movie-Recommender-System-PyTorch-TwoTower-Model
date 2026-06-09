"""
Stage 6 — Figures for the LLM-vs-genome writeup (results/llm_vs_genome_ablation.md).

Generates two publication-quality PNGs into results/figures/:

  fig1_tier_lift.png       — content's lift over the no-content floor by popularity
                             tier. Left: absolute MRR by tier (log) — ranking quality
                             collapses on the tail. Right: relative MRR lift over the
                             floor (%) for genome (A) and LLM (B) — content earns its
                             keep exactly where collaborative signal is sparse, and the
                             LLM tracks genome rather than collapsing.
  fig2_agreement_hist.png  — distribution of the 132 per-dimension genome-vs-LLM
                             Pearson correlations (shared-axis agreement).

Figure 1 uses the recorded Phase 2 tier metrics (n=382,138, alpha=0; see the ablation
plan Stage 6 / eval_results). Figure 2 recomputes the 132 correlations with the same
logic as feature_level_analysis.py and caches them to
results/figures/feature_agreement_r.json (so re-runs skip the slow features load).

Usage (from repo root):
    CORPUS=full python -m llm_features.make_figures
"""
import json
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

FIG_DIR = 'results/figures'
R_CACHE = os.path.join(FIG_DIR, 'feature_agreement_r.json')

# floor / genome / llm — colourblind-safe
C_COLOR, A_COLOR, B_COLOR = '#9aa0a6', '#3b6db5', '#e08a3c'
INK, MUTED = '#222222', '#6b6b6b'


def set_theme():
    mpl.rcParams.update({
        'figure.dpi': 120, 'savefig.dpi': 200, 'savefig.bbox': 'tight',
        'savefig.facecolor': 'white', 'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'font.size': 11, 'font.family': 'sans-serif',
        'axes.titlesize': 13, 'axes.titleweight': 'bold', 'axes.labelsize': 11,
        'axes.edgecolor': '#444444', 'axes.linewidth': 0.9,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.axisbelow': True,
        'axes.grid': True, 'grid.color': '#e8e8e8', 'grid.linewidth': 0.9,
        'xtick.color': INK, 'ytick.color': INK, 'text.color': INK, 'axes.labelcolor': INK,
        'legend.frameon': False,
    })


# ── Figure 1: tier lift ──────────────────────────────────────────────────────
def figure1():
    # Phase 2 MRR by equal-count popularity quartile (n=382,138, alpha=0).
    tiers = ['Q4 · popular\nn=343,906', 'Q3\nn=26,923', 'Q2\nn=8,049', 'Q1 · rarest\nn=3,260']
    C = np.array([0.1259, 0.0129, 0.0032, 0.0012])
    A = np.array([0.1260, 0.0148, 0.0038, 0.0014])
    B = np.array([0.1273, 0.0144, 0.0037, 0.0014])
    x = np.arange(len(tiers))
    w = 0.26

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.8))

    # Left — absolute MRR by tier (log)
    for i, (vals, col, lab) in enumerate([(C, C_COLOR, 'No content (C)'),
                                          (A, A_COLOR, 'Genome (A)'),
                                          (B, B_COLOR, 'LLM (B)')]):
        axL.bar(x + (i - 1) * w, vals, w, color=col, label=lab, edgecolor='white', linewidth=0.6)
    axL.set_yscale('log')
    axL.set_ylim(8e-4, 0.25)
    axL.set_ylabel('MRR  (log scale)')
    axL.set_title('Ranking quality collapses on the tail', color=INK)
    axL.set_xticks(x)
    axL.set_xticklabels(tiers, fontsize=9.5)
    axL.grid(axis='x', visible=False)
    axL.legend(fontsize=9, loc='upper right')

    # Right — relative lift over the no-content floor (%)
    liftA = (A - C) / C * 100
    liftB = (B - C) / C * 100
    bA = axR.bar(x - w / 2, liftA, w, color=A_COLOR, label='Genome  (A − C)', edgecolor='white', linewidth=0.6)
    bB = axR.bar(x + w / 2, liftB, w, color=B_COLOR, label='LLM  (B − C)', edgecolor='white', linewidth=0.6)
    axR.axhline(0, color='#444444', lw=0.9)
    axR.set_ylabel('MRR lift over no-content floor (%)')
    axR.set_title('Content earns its keep on the tail', color=INK)
    axR.set_xticks(x)
    axR.set_xticklabels(tiers, fontsize=9.5)
    axR.grid(axis='x', visible=False)
    axR.set_ylim(-2, max(liftA.max(), liftB.max()) * 1.18)
    axR.legend(fontsize=9, loc='upper left')
    for bars, vals, col in [(bA, liftA, A_COLOR), (bB, liftB, B_COLOR)]:
        for rect, v in zip(bars, vals):
            axR.annotate(f'{v:+.0f}%', (rect.get_x() + rect.get_width() / 2, v),
                         textcoords='offset points', xytext=(0, 3), ha='center',
                         fontsize=8.5, fontweight='bold', color=col)

    fig.suptitle('Content features matter where collaborative signal is sparse',
                 fontsize=15, fontweight='bold', y=1.03)
    fig.text(0.5, -0.06,
             'MovieLens 32M · Phase 2 rollback eval (n = 382,138, α = 0) · '
             'tiers by target-movie rating count · same examples across arms',
             ha='center', fontsize=8.5, color=MUTED)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, 'fig1_tier_lift.png')
    fig.savefig(out)
    plt.close(fig)
    print(f'wrote {out}')


# ── Figure 2: shared-axis agreement ──────────────────────────────────────────
def compute_r():
    if os.path.exists(R_CACHE):
        print(f'using cached {R_CACHE}')
        return json.load(open(R_CACHE))
    print('computing 132 per-dim correlations (loading features — slow)…')
    from llm_features.feature_level_analysis import load_llm_artifacts
    from src.dataset import load_features

    names, dim2genome, dim2group, L = load_llm_artifacts()
    fs = load_features()
    G = np.array([fs.movieId_to_genome_tag_context[mid] for mid in fs.top_movies], dtype=np.float32)
    gname_at_idx = [fs.genome_tag_names[tid] for tid in fs.genome_tag_ids]
    gname_to_idx = {nm: i for i, nm in enumerate(gname_at_idx)}

    rows = []
    for d, name in enumerate(names):
        idxs = [gname_to_idx[g] for g in dim2genome[name] if g in gname_to_idx]
        if not idxs:
            continue
        gvec, lvec = G[:, idxs].mean(axis=1), L[:, d]
        if gvec.std() < 1e-9 or lvec.std() < 1e-9:
            continue
        rows.append({'name': name, 'group': dim2group[name],
                     'r': float(np.corrcoef(lvec, gvec)[0, 1])})
    os.makedirs(FIG_DIR, exist_ok=True)
    json.dump(rows, open(R_CACHE, 'w'), indent=2)
    print(f'wrote {R_CACHE} ({len(rows)} dims)')
    return rows


def figure2(rows):
    rs = np.array([d['r'] for d in rows])
    by_name = {d['name']: d['r'] for d in rows}
    mean_r, med_r = rs.mean(), np.median(rs)
    n_ge_half = int((rs >= 0.5).sum())

    fig, ax = plt.subplots(figsize=(9.8, 5.7))
    bins = np.linspace(0.0, 1.0, 21)
    counts, edges, patches = ax.hist(rs, bins=bins, edgecolor='white', linewidth=0.9)

    # colour each bar by its agreement (low r = red, high r = green)
    cmap, norm = mpl.colormaps['RdYlGn'], Normalize(vmin=0.1, vmax=0.95)
    for patch, lo, hi in zip(patches, edges[:-1], edges[1:]):
        patch.set_facecolor(cmap(norm((lo + hi) / 2)))

    ax.axvline(mean_r, color='#222222', lw=2.0, ls='--', label=f'mean  r = {mean_r:.2f}')
    ax.axvline(med_r, color='#888888', lw=1.6, ls=':', label=f'median  r = {med_r:.2f}')

    ymax = counts.max()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, ymax * 1.16)
    ax.set_xlabel('Pearson r  (LLM dimension vs its source genome tag, across 9,375 movies)')
    ax.set_ylabel('number of dimensions')
    ax.set_title('Genome and LLM measure the same axes', color=INK, pad=14)
    ax.grid(axis='x', visible=False)
    ax.legend(fontsize=9.5, loc='upper left')

    # stat callout (upper-left, under the legend)
    ax.text(0.025, 0.70, f'{n_ge_half} / {len(rs)} dims at r ≥ 0.5\nnone below r = 0.1',
            transform=ax.transAxes, fontsize=9.5, color=INK,
            bbox=dict(boxstyle='round,pad=0.5', fc='#f4f4f4', ec='#cccccc'))

    # label the two ends directly — the gradient already encodes low=red / high=green, so no arrows
    def lbl(name):
        return f'{name.replace("_", " ")} {by_name[name]:.2f}'
    lo_axes = [a for a in ['imdb_top_250', 'criterion'] if a in by_name]
    hi_axes = [a for a in ['vampires', 'documentary', 'animated'] if a in by_name]
    ax.text(0.05, ymax * 0.34, 'crowd-prestige axes\n' + '\n'.join(lbl(a) for a in lo_axes),
            fontsize=8.8, color='#9c3b3b', ha='left', va='bottom', linespacing=1.4)
    ax.text(0.845, ymax * 0.40, 'factual axes\n' + '\n'.join(lbl(a) for a in hi_axes),
            fontsize=8.8, color='#1b6b3a', ha='center', va='bottom', linespacing=1.4)

    fig.text(0.5, -0.04,
             'Schema derived from genome’s top-discriminability tags, so both spaces sit on the same 132 axes · '
             'feature_level_analysis.py',
             ha='center', fontsize=8.5, color=MUTED)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, 'fig2_agreement_hist.png')
    fig.savefig(out)
    plt.close(fig)
    print(f'wrote {out}')


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    set_theme()
    figure1()
    figure2(compute_r())


if __name__ == '__main__':
    main()
