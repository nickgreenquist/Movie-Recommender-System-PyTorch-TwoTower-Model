"""
Figure for the multipool user-tower ablation writeup
(docs/multipool_user_tower_ablation/multipool_user_tower_ablation.md).

Generates two publication-quality PNGs into the writeup's figures/ dir:

  fig1_pooling_mrr.png        — whole-corpus MRR for all 12 user-pooling arms, sorted,
                          colour-coded by pool family (baseline / rating-valence /
                          recency), with the single-full-pool baseline marked. The
                          message reads at a glance: the recency arms (full + last
                          watched) take the top; the rating-valence arms pile up on
                          the baseline (adding nothing); the impoverished single-pool
                          arms collapse at the bottom.
  fig2_pooling_schematic.png  — schematic for §1: an arbitrary-length watch history,
                          one embedding looked up per movie, summed into a single
                          fixed 32-d pooled vector, then projected to the 128-d user
                          vector. Conveys "any-length history → one input vector".
  fig3_last_watched.png       — schematic for §1: how a recency channel is added. The
                          single last-watched item is looked up (no sum) into its own
                          32-d block with its own LayerNorm, CONCATENATED alongside the
                          sum-pool block (64-d), then the combined vector goes through
                          the ONE shared user_projection. Matches src/model.py forward:
                          parts.append(...) per pool → torch.cat(parts) → user_projection.

Numbers are the whole-corpus MRR (n=382,138, α=0, seed 42, 200k steps) from the
results matrix in docs/plans/multipool_user_tower_ablation_plan.md — hardcoded here,
so the figure regenerates instantly with no model/feature load.

Usage (from repo root):
    python tools/make_multipool_figure.py
"""
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

FIG_DIR = 'docs/multipool_user_tower_ablation/figures'

# pool families — reuse the house palette (floor grey / genome blue / llm orange)
BASE_COLOR, VALENCE_COLOR, RECENCY_COLOR = '#9aa0a6', '#3b6db5', '#e08a3c'
INK, MUTED = '#222222', '#6b6b6b'

# (arm #, short label, whole-corpus MRR, family)
ARMS = [
    (1,  'full  (baseline)',                      0.1133, 'base'),
    (2,  'full + liked',                          0.1141, 'valence'),
    (3,  'full + liked + disliked',               0.1133, 'valence'),
    (4,  'weighted only',                         0.0708, 'valence'),
    (5,  'full + liked + disliked + weighted',    0.1132, 'valence'),
    (6,  '4-pool + last-liked',                   0.1227, 'recency'),
    (7,  'full + last-liked',                     0.1236, 'recency'),
    (8,  'full + weighted',                       0.1130, 'valence'),
    (9,  'last-liked only',                       0.0805, 'recency'),
    (10, 'last-watched only',                     0.1099, 'recency'),
    (11, 'full + last-watched',                   0.1386, 'recency'),
    (12, 'full + last-watched + 2nd-last',        0.1431, 'recency'),
]
BASELINE = 0.1133  # arm 1, single full sum pool
FAMILY_COLOR = {'base': BASE_COLOR, 'valence': VALENCE_COLOR, 'recency': RECENCY_COLOR}


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


def figure1():
    os.makedirs(FIG_DIR, exist_ok=True)

    rows = sorted(ARMS, key=lambda r: r[2])          # ascending → best ends up on top
    labels = [lab for (_, lab, _, _) in rows]
    vals = np.array([r[2] for r in rows])
    colors = [FAMILY_COLOR[r[3]] for r in rows]
    y = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(10, 6.6))
    bars = ax.barh(y, vals, color=colors, edgecolor='white', linewidth=0.7, height=0.74)

    # baseline reference line (single full pool)
    ax.axvline(BASELINE, color='#444444', lw=1.3, ls='--', zorder=1)
    ax.annotate('single full pool\n(baseline)', (BASELINE, len(rows) - 0.5),
                textcoords='offset points', xytext=(6, -2), ha='left', va='top',
                fontsize=8.5, color=MUTED, linespacing=1.3)

    # value labels at the bar ends
    for rect, v in zip(bars, vals):
        ax.annotate(f'{v:.4f}', (v, rect.get_y() + rect.get_height() / 2),
                    textcoords='offset points', xytext=(4, 0), ha='left', va='center',
                    fontsize=8.5, color=INK)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.set_xlim(0, 0.168)
    ax.set_xlabel('Whole-corpus MRR  (rollback eval, n = 382,138)')
    ax.set_title('Recency Is All You Need: The Evidence', color=INK, pad=12)
    ax.grid(axis='y', visible=False)

    # legend for the three families
    handles = [mpl.patches.Patch(facecolor=BASE_COLOR, label='baseline (single full pool)'),
               mpl.patches.Patch(facecolor=VALENCE_COLOR, label='rating-valence pools'),
               mpl.patches.Patch(facecolor=RECENCY_COLOR, label='recency (last-item) pools')]
    ax.legend(handles=handles, fontsize=9, loc='lower right', borderaxespad=0.8)

    # the headline callout: +22% on the recency winner
    win = next(r for r in rows if r[0] == 11)
    yi = rows.index(win)
    ax.annotate('+22% vs the full pool —\nthe single last-watched item',
                (win[2], yi), textcoords='offset points', xytext=(54, 2),
                ha='left', va='center', fontsize=9, color=RECENCY_COLOR, fontweight='bold',
                linespacing=1.3,
                arrowprops=dict(arrowstyle='-|>', color=RECENCY_COLOR, lw=1.4,
                                connectionstyle='arc3,rad=-0.2'))

    fig.text(0.5, -0.03,
             'MovieLens 32M · two-tower retrieval, ID-only item tower · α = 0, seed 42, 200k steps · '
             'only the active user-history pools differ across arms',
             ha='center', fontsize=8.5, color=MUTED)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, 'fig1_pooling_mrr.png')
    fig.savefig(out)
    plt.close(fig)
    print(f'wrote {out}')


def figure2():
    """Schematic: arbitrary-length history → per-item embedding → sum pool → project.

    Pure diagram (no data) — boxes + arrows in the house palette. The visual argument
    for §1: a variable-length list collapses into one fixed input vector via the sum.
    """
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyArrowPatch

    os.makedirs(FIG_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 56)
    ax.axis('off')

    def box(x, y, w, h, text, fc, ec=INK, fs=10, tc=INK, lw=1.1):
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle='round,pad=0.4,rounding_size=1.2',
            fc=fc, ec=ec, lw=lw, zorder=2))
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
                fontsize=fs, color=tc, zorder=3, linespacing=1.15)

    def arrow(x0, y0, x1, y1, color=MUTED, lw=1.4):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>',
                     mutation_scale=12, color=color, lw=lw, zorder=1,
                     shrinkA=2, shrinkB=2))

    # column x-positions
    x_mov, x_emb, x_sum, x_pool, x_proj, x_user = 3, 26, 49, 60, 76, 90
    mov_w, mov_h = 19, 6.2

    # ── left: a variable-length history (movies), with an ellipsis row ──────
    movies = [('Heat (1995)', 46), ('Se7en (1995)', 38),
              ('Fargo (1996)', 30), ('⋮', 22), ('last watch', 13)]
    emb_y = []
    for label, y in movies:
        if label == '⋮':
            ax.text(x_mov + mov_w / 2, y + mov_h / 2, '⋮', ha='center', va='center',
                    fontsize=15, color=MUTED)
            ax.text(x_emb + 7, y + mov_h / 2, '⋮', ha='center', va='center',
                    fontsize=15, color=MUTED)
            continue
        box(x_mov, y, mov_w, mov_h, label, '#f4f6f9', ec='#c4ccd6', fs=9.5)
        # per-item embedding chip (a little 4-cell vector)
        box(x_emb, y + 0.6, 14, mov_h - 1.2, '', '#ffffff', ec=RECENCY_COLOR, lw=1.2)
        for k in range(4):
            ax.add_patch(mpatches.Rectangle(
                (x_emb + 0.9 + k * 3.15, y + 1.4), 2.7, mov_h - 2.8,
                fc=RECENCY_COLOR, ec='none', alpha=0.32 + 0.16 * k, zorder=3))
        arrow(x_mov + mov_w, y + mov_h / 2, x_emb, y + mov_h / 2)
        emb_y.append(y + mov_h / 2)

    ax.text(x_mov + mov_w / 2, 53.5, 'watch history\n(any length N)', ha='center',
            va='center', fontsize=10.5, color=INK, fontweight='bold', linespacing=1.2)
    ax.text(x_emb + 7, 53.5, 'lookup\nembedding / item', ha='center', va='center',
            fontsize=10.5, color=INK, fontweight='bold', linespacing=1.2)

    # ── sum node ────────────────────────────────────────────────────────────
    sy = 29.5
    ax.add_patch(mpatches.Circle((x_sum, sy), 3.4, fc='#fff', ec=INK, lw=1.4, zorder=2))
    ax.text(x_sum, sy, 'Σ', ha='center', va='center', fontsize=17, color=INK, zorder=3)
    for y in emb_y:
        arrow(x_emb + 14, y, x_sum - 3.4, sy, color='#b9c1cc', lw=1.1)
    ax.text(x_sum, sy + 6.2, 'sum pool', ha='center', va='center', fontsize=10.5,
            color=INK, fontweight='bold')

    # ── pooled fixed vector (32-d) ──────────────────────────────────────────
    box(x_pool, sy - 4, 10, 8, '', '#ffffff', ec=RECENCY_COLOR, lw=1.4)
    for k in range(4):
        ax.add_patch(mpatches.Rectangle(
            (x_pool + 0.8 + k * 2.25, sy - 3), 1.9, 6,
            fc=RECENCY_COLOR, ec='none', alpha=0.30 + 0.17 * k, zorder=3))
    ax.text(x_pool + 5, sy - 6.6, 'one fixed\n32-d vector', ha='center', va='center',
            fontsize=9.5, color=INK, linespacing=1.2)
    arrow(x_sum + 3.4, sy, x_pool, sy)

    # ── projection → user vector ────────────────────────────────────────────
    box(x_proj, sy - 4.5, 11, 9, 'Linear → ReLU\n→ Linear → L2', '#eef2f7',
        ec='#c4ccd6', fs=9)
    arrow(x_pool + 10, sy, x_proj, sy)
    box(x_user, sy - 4, 9, 8, '', '#ffffff', ec=BASE_COLOR, lw=1.4)
    for k in range(4):
        ax.add_patch(mpatches.Rectangle(
            (x_user + 0.7 + k * 2.0, sy - 3), 1.7, 6,
            fc=BASE_COLOR, ec='none', alpha=0.34 + 0.15 * k, zorder=3))
    ax.text(x_user + 4.5, sy - 6.6, 'user vector\n(128-d)', ha='center', va='center',
            fontsize=9.5, color=INK, fontweight='bold', linespacing=1.2)
    arrow(x_proj + 11, sy, x_user, sy)

    ax.set_title('Any-length history → one fixed input vector', color=INK,
                 fontsize=14, pad=4, loc='center')
    fig.text(0.5, 0.015,
             'The sum collapses every watched item into a single 32-d vector before projection — '
             'no per-item identity, no order survive the Σ.',
             ha='center', fontsize=9, color=MUTED)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, 'fig2_pooling_schematic.png')
    fig.savefig(out)
    plt.close(fig)
    print(f'wrote {out}')


def figure3():
    """Schematic: how the recency channel is added (arm 11 = full + last_watched).

    Matches src/model.py user_embedding(): each active pool appends a 32-d block (its own
    LayerNorm) to `parts`; `last_watched` is a single-item lookup with NO sum; the blocks are
    torch.cat'd FIRST (→ 64-d), then a single shared user_projection maps that to the 128-d
    user vector. Drawn to make the concat-before-projection ordering unambiguous.
    """
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyArrowPatch

    os.makedirs(FIG_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 5.0))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 50)
    ax.axis('off')

    def box(x, y, w, h, text, fc, ec=INK, fs=9.5, tc=INK, lw=1.1):
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle='round,pad=0.4,rounding_size=1.2',
            fc=fc, ec=ec, lw=lw, zorder=2))
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
                fontsize=fs, color=tc, zorder=3, linespacing=1.15)

    def chip(x, y, w, h, color, ec):
        box(x, y, w, h, '', '#ffffff', ec=ec, lw=1.3)
        for k in range(4):
            ax.add_patch(mpatches.Rectangle(
                (x + 0.8 + k * (w - 1.6) / 4, y + 1.0), (w - 1.6) / 4 - 0.4, h - 2.0,
                fc=color, ec='none', alpha=0.30 + 0.17 * k, zorder=3))

    def arrow(x0, y0, x1, y1, color=MUTED, lw=1.4):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>',
                     mutation_scale=12, color=color, lw=lw, zorder=1,
                     shrinkA=2, shrinkB=2))

    y_top, y_bot = 37, 11           # full-pool branch / last-watched branch
    x_src, x_op, x_blk, x_cat, x_proj, x_user = 2, 27, 41, 58, 70, 88

    # ── top branch: full history → Σ sum pool → 32-d block ──────────────────
    box(x_src, y_top - 3.5, 21, 7, 'full history\n(N items)', '#f4f6f9', ec='#c4ccd6')
    ax.add_patch(mpatches.Circle((x_op + 2, y_top), 3.2, fc='#fff', ec=INK, lw=1.4, zorder=2))
    ax.text(x_op + 2, y_top, 'Σ', ha='center', va='center', fontsize=15, color=INK, zorder=3)
    arrow(x_src + 21, y_top, x_op - 1.2, y_top)
    ax.text(x_op + 2, y_top + 5.6, 'sum pool', ha='center', fontsize=9, color=INK, fontweight='bold')
    chip(x_blk, y_top - 3.4, 11, 6.8, RECENCY_COLOR, RECENCY_COLOR)
    arrow(x_op + 5.2, y_top, x_blk, y_top)
    ax.text(x_blk + 5.5, y_top - 5.4, 'LayerNorm\n→ 32-d', ha='center', va='top',
            fontsize=8.3, color=MUTED, linespacing=1.15)

    # ── bottom branch: last watched (1 item) → lookup (no sum) → 32-d block ──
    box(x_src, y_bot - 3.5, 21, 7, 'last watched\n(1 item)', '#fff4ea', ec=RECENCY_COLOR)
    ax.text(x_op + 2, y_bot + 2.4, 'lookup', ha='center', fontsize=9, color=INK, fontweight='bold')
    ax.text(x_op + 2, y_bot - 2.2, 'no sum', ha='center', fontsize=8.3, color='#b04a1e', style='italic')
    arrow(x_src + 21, y_bot, x_blk, y_bot)
    chip(x_blk, y_bot - 3.4, 11, 6.8, RECENCY_COLOR, RECENCY_COLOR)
    ax.text(x_blk + 5.5, y_bot - 5.4, 'LayerNorm\n→ 32-d', ha='center', va='top',
            fontsize=8.3, color=MUTED, linespacing=1.15)

    # ── concat (64-d) ───────────────────────────────────────────────────────
    ymid = (y_top + y_bot) / 2
    box(x_cat, ymid - 6.5, 8.5, 13, 'concat\n64-d', '#eef2f7', ec='#9aa6b4', fs=9.5)
    arrow(x_blk + 11, y_top, x_cat, ymid + 3.2)
    arrow(x_blk + 11, y_bot, x_cat, ymid - 3.2)

    # ── one shared projection → user vector ─────────────────────────────────
    box(x_proj, ymid - 5, 13, 10, 'user_projection\nLinear → ReLU\n→ Linear → L2', '#eef2f7',
        ec='#c4ccd6', fs=8.6)
    arrow(x_cat + 8.5, ymid, x_proj, ymid)
    chip(x_user, ymid - 3.6, 10, 7.2, BASE_COLOR, BASE_COLOR)
    ax.text(x_user + 5, ymid - 5.6, 'user vector\n128-d', ha='center', va='top',
            fontsize=8.6, color=INK, fontweight='bold', linespacing=1.15)
    arrow(x_proj + 13, ymid, x_user, ymid)

    ax.set_title('Adding a recency channel: concatenate, then project once', color=INK,
                 fontsize=14, pad=6)
    fig.text(0.5, 0.015,
             'The last-watched item is a single lookup (no Σ) with its own LayerNorm; its 32-d block is '
             'concatenated alongside the sum-pool block, and the one shared projection sees both.',
             ha='center', fontsize=9, color=MUTED)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, 'fig3_last_watched.png')
    fig.savefig(out)
    plt.close(fig)
    print(f'wrote {out}')


def main():
    set_theme()
    figure1()
    figure2()
    figure3()


if __name__ == '__main__':
    main()
