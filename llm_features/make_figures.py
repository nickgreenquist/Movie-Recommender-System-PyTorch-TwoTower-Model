"""
Stage 6 — Figures for the LLM-vs-genome writeup (results/llm_vs_genome_ablation.md).

Generates two publication-quality PNGs into results/figures/:

  fig1_content_lift.png    — content's value in the universal vs the MovieLens-rich
                             setting. Left: pure-CF floor vs +genome vs +LLM (base
                             model, both corpora) — content beats the floor and C′<A′<B′
                             replicates. Right: the same content slot added to the rich
                             model (genre + user tags + year + rating pools) — the lift
                             collapses to ~0/negative as content goes redundant with the
                             curated metadata.
  fig2_agreement_hist.png  — distribution of the 132 per-dimension genome-vs-LLM
                             Pearson correlations (shared-axis agreement).

Figure 1 uses the low-variance (seeded, 160k-step) base + rich whole-corpus MRR
(α=0; see the ablation plan "Phase B" section / eval_results). Figure 2 recomputes the
132 correlations with the same logic as feature_level_analysis.py and caches them to
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


# ── Figure 1: content lift — base (universal) vs rich (MovieLens) ─────────
def figure1():
    """Two panels, whole-corpus MRR, α=0, low-variance protocol (seeded, 160k steps).

    Left  — the primary experiment: a pure-CF floor (single implicit history pool + item ID)
            vs the same model + genome / + LLM content, on both corpora. Content beats the
            floor and the ordering C′ < A′ < B′ replicates.
    Right — the same content slot added to the *rich* model (genome-era genre + user tags +
            year + rating pools): the lift collapses to ~0 / negative — content goes redundant
            with the curated metadata. The two regimes bracket the real-world answer.
    """
    os.makedirs(FIG_DIR, exist_ok=True)

    # Left — base (universal-setting) MRR, both corpora.
    corpora = ['Full corpus\nn = 382,138', 'Phase 1 (head)\nn = 99,846']
    Cs = np.array([0.1121, 0.1133])   # C′ floor (ID pool only)
    As = np.array([0.1148, 0.1158])   # A′ + genome
    Bs = np.array([0.1155, 0.1165])   # B′ + LLM
    x = np.arange(len(corpora))
    w = 0.26

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.8))

    for i, (vals, col, lab) in enumerate([(Cs, C_COLOR, 'Pure CF floor (C′)'),
                                          (As, A_COLOR, '+ genome (A′)'),
                                          (Bs, B_COLOR, '+ LLM (B′)')]):
        bars = axL.bar(x + (i - 1) * w, vals, w, color=col, label=lab,
                       edgecolor='white', linewidth=0.6)
        for rect, v in zip(bars, vals):
            axL.annotate(f'{v:.4f}', (rect.get_x() + rect.get_width() / 2, v),
                         textcoords='offset points', xytext=(0, 3), ha='center',
                         fontsize=8, color=MUTED)
    axL.set_ylim(0.108, 0.119)
    axL.set_ylabel('MRR')
    axL.set_title('Content beats pure CF — and it replicates', color=INK)
    axL.set_xticks(x)
    axL.set_xticklabels(corpora, fontsize=9.5)
    axL.grid(axis='x', visible=False)
    axL.legend(fontsize=9, loc='lower center', ncol=3, columnspacing=1.0, handletextpad=0.4)

    # Right — lift over the floor (%), base vs rich (full corpus, low-variance).
    regimes = ['Base model\n(what most teams have)', 'Rich metadata\n(MovieLens only)']
    # base: A′−C′, B′−C′ over C′=0.1121 ; rich (low-var): A−C, B−C over C=0.1174
    liftA = np.array([(0.1148 - 0.1121) / 0.1121, (0.1144 - 0.1174) / 0.1174]) * 100
    liftB = np.array([(0.1155 - 0.1121) / 0.1121, (0.1176 - 0.1174) / 0.1174]) * 100
    xr = np.arange(len(regimes))
    bA = axR.bar(xr - w / 2, liftA, w, color=A_COLOR, label='genome lift  (A − C)',
                 edgecolor='white', linewidth=0.6)
    bB = axR.bar(xr + w / 2, liftB, w, color=B_COLOR, label='LLM lift  (B − C)',
                 edgecolor='white', linewidth=0.6)
    axR.axhline(0, color='#444444', lw=0.9)
    axR.set_ylabel('MRR lift over the floor (%)')
    axR.set_title('…but it vanishes once you have the metadata', color=INK)
    axR.set_xticks(xr)
    axR.set_xticklabels(regimes, fontsize=9.5)
    axR.grid(axis='x', visible=False)
    axR.set_ylim(min(liftA.min(), liftB.min()) - 1.3, max(liftA.max(), liftB.max()) * 1.25)
    axR.legend(fontsize=9, loc='upper right')
    for bars, vals in [(bA, liftA), (bB, liftB)]:
        for rect, v in zip(bars, vals):
            axR.annotate(f'{v:+.1f}%', (rect.get_x() + rect.get_width() / 2, v),
                         textcoords='offset points', xytext=(0, 3 if v >= 0 else -11),
                         ha='center', fontsize=8.5, fontweight='bold', color=INK)

    fig.suptitle('In the setting most teams actually start from, content features earn their keep',
                 fontsize=15, fontweight='bold', y=1.03)
    fig.text(0.5, -0.06,
             'MovieLens 32M · rollback eval, α = 0, low-variance protocol · '
             "'base model' = single implicit history pool + item ID; "
             "'rich' adds curated genre + user tags + year + rating pools",
             ha='center', fontsize=8.5, color=MUTED)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, 'fig1_content_lift.png')
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
