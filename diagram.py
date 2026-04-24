"""
Generates a two-tower architecture diagram.

Usage:
    python diagram.py               # saves diagram.png
    python diagram.py --show        # also opens the image
"""
import sys
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# ── Palette ───────────────────────────────────────────────────────────────────
C_USER    = '#4C8BF5'
C_ITEM    = '#F5824C'
C_COMBINE = '#34A853'
C_BG      = '#F8F9FA'
C_TEXT    = '#1A1A2E'


def draw_box(ax, cx, cy, w, h, title, subtitle, source, dim, color, title_size=9):
    patch = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.015",
        facecolor=color, edgecolor='white', linewidth=1.5, zorder=3,
    )
    ax.add_patch(patch)
    ax.text(cx, cy + h * 0.26, title, ha='center', va='center',
            fontsize=title_size, fontweight='bold', color='white', zorder=4)
    ax.text(cx, cy + h * 0.04, subtitle, ha='center', va='center',
            fontsize=7.5, color='white', alpha=0.85, style='italic', zorder=4)
    ax.text(cx, cy - h * 0.22, source, ha='center', va='center',
            fontsize=6.8, color='white', alpha=0.70, zorder=4)
    ax.text(cx + w / 2 - 0.015, cy - h / 2 + 0.010, dim,
            ha='right', va='bottom', fontsize=8, color='white',
            fontweight='bold', alpha=0.9, zorder=4)


def bracket_to_concat(ax, x, ys, y_concat, color):
    y_mid = min(ys) - 0.03
    for y in ys:
        ax.plot([x, x], [y, y_mid], color=color, lw=1.2, zorder=2)
    ax.plot([x - 0.01, x + 0.01], [y_mid, y_mid], color=color, lw=1.2, zorder=2)
    ax.annotate('', xy=(x, y_concat + 0.004), xytext=(x, y_mid),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.8), zorder=2)


# ── Figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 20))
fig.patch.set_facecolor(C_BG)
ax.set_facecolor(C_BG)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

BOX_W  = 0.30
BOX_H  = 0.095
USER_X = 0.26
ITEM_X = 0.74

# ── Title ─────────────────────────────────────────────────────────────────────
ax.text(0.5, 0.980, 'Two-Tower Movie Recommender Architecture',
        ha='center', va='top', fontsize=14, fontweight='bold', color=C_TEXT)

# ── Column headers ────────────────────────────────────────────────────────────
ax.text(USER_X, 0.950, 'USER TOWER',
        ha='center', fontsize=11, fontweight='bold', color=C_USER)
ax.text(ITEM_X, 0.950, 'ITEM TOWER',
        ha='center', fontsize=11, fontweight='bold', color=C_ITEM)

# ── Layout y positions ────────────────────────────────────────────────────────
# Shared pairs must sit at the same y:
#   ROW_A: user watch history pool  ↔  item_embedding_tower   (shared item_embedding_lookup)
#   ROW_B: user genome pool         ↔  item_genome_tag_tower  (shared item_genome_tag_tower)
ROW_A = 0.905
ROW_B = 0.795
# Row spacing: 0.110 for all rows

# ── User tower components ─────────────────────────────────────────────────────
user_comps = [
    # (title, subtitle, source, dim, y)
    ('Rating-Weighted Avg Pool',
     'watch history (indices + ratings)',
     'ratings.csv  →  sort by timestamp  →  movie indices + debiased ratings',
     '32-dim', ROW_A),

    ('Rating-Weighted Genome Pool',
     'genome ctx[history]  →  shared tower',
     'genome-scores.csv  →  buffer[history_indices]  →  (hist_len × 1,128) float32',
     '32-dim', ROW_B),

    ('user_genre_tower',
     'genre context (avg_rating, watch_frac)',
     'ratings.csv  →  avg_debiased_rating(20) + watch_frac(20) per genre',
     '32-dim', 0.685),

    ('timestamp_embedding_tower',
     'watch month bin',
     'ratings.csv timestamp  →  unix seconds  →  monthly bin index (0–1,499)',
     '4-dim', 0.575),
]
for title, sub, src, dim, y in user_comps:
    draw_box(ax, USER_X, y, BOX_W, BOX_H, title, sub, src, dim, C_USER)

USER_CONCAT_Y = 0.445
bracket_to_concat(ax, USER_X,
                  [y for *_, y in user_comps],
                  USER_CONCAT_Y, C_USER)
draw_box(ax, USER_X, USER_CONCAT_Y, BOX_W, BOX_H,
         'concat', 'history(32) + genome(32) + genre(32) + ts(4)', '', '100-dim', C_COMBINE)

USER_PROJ_Y = 0.320
ax.annotate('', xy=(USER_X, USER_PROJ_Y + BOX_H / 2 + 0.003),
            xytext=(USER_X, USER_CONCAT_Y - BOX_H / 2 - 0.003),
            arrowprops=dict(arrowstyle='->', color=C_COMBINE, lw=1.8), zorder=2)
draw_box(ax, USER_X, USER_PROJ_Y, BOX_W, BOX_H,
         'user_projection', 'projection MLP', 'Linear(256) → ReLU → Linear(128)', '128-dim', C_COMBINE)

# ── Item tower components ─────────────────────────────────────────────────────
# item_embedding_tower at ROW_A and item_genome_tag_tower at ROW_B
# so both shared annotations are clean horizontal arrows.
item_comps = [
    ('item_embedding_tower',
     'movie ID lookup',
     'movies.csv  →  integer index per top movie (0–9,374)',
     '32-dim', ROW_A),

    ('item_genome_tag_tower',
     'genome scores  (1,128 tags)',
     'genome-scores.csv  →  1,128 relevance scores (0.0–1.0) per movie',
     '32-dim', ROW_B),

    ('item_genre_tower',
     'genre one-hot  (20 genres)',
     'movies.csv genres field  →  20-dim binary one-hot vector',
     '8-dim', 0.685),

    ('item_tag_tower',
     'tag vector  (306 user-applied tags)',
     'tags.csv  →  306 tag counts normalized to sum=1',
     '16-dim', 0.575),

    ('year_embedding_tower',
     'release year lookup',
     'movies.csv title  →  extract "(year)"  →  integer index (0–191)',
     '8-dim', 0.465),
]
for title, sub, src, dim, y in item_comps:
    draw_box(ax, ITEM_X, y, BOX_W, BOX_H, title, sub, src, dim, C_ITEM)

ITEM_CONCAT_Y = 0.350
bracket_to_concat(ax, ITEM_X,
                  [y for *_, y in item_comps],
                  ITEM_CONCAT_Y, C_ITEM)
draw_box(ax, ITEM_X, ITEM_CONCAT_Y, BOX_W, BOX_H,
         'concat', 'movieId(32) + genome(32) + genre(8) + tag(16) + year(8)', '', '96-dim', C_COMBINE)

ITEM_PROJ_Y = 0.225
ax.annotate('', xy=(ITEM_X, ITEM_PROJ_Y + BOX_H / 2 + 0.003),
            xytext=(ITEM_X, ITEM_CONCAT_Y - BOX_H / 2 - 0.003),
            arrowprops=dict(arrowstyle='->', color=C_COMBINE, lw=1.8), zorder=2)
draw_box(ax, ITEM_X, ITEM_PROJ_Y, BOX_W, BOX_H,
         'item_projection', 'projection MLP', 'Linear(256) → ReLU → Linear(128)', '128-dim', C_COMBINE)

# ── Shared embedding annotations ──────────────────────────────────────────────
def shared_arrow(ax, y, label):
    ax.annotate('', xy=(ITEM_X - BOX_W / 2 - 0.01, y),
                xytext=(USER_X + BOX_W / 2 + 0.01, y),
                arrowprops=dict(arrowstyle='<->', color='#888888',
                                lw=1.2, linestyle='dashed'), zorder=2)
    ax.text(0.50, y + 0.028, label,
            ha='center', va='bottom', fontsize=7.5, color='#666666', style='italic')

shared_arrow(ax, ROW_A, 'shared  item_embedding_lookup')
shared_arrow(ax, ROW_B, 'shared  item_genome_tag_tower')

# ── Dot product ───────────────────────────────────────────────────────────────
DOT_Y = 0.090
DOT_X = 0.50

for cx, proj_y in [(USER_X, USER_PROJ_Y), (ITEM_X, ITEM_PROJ_Y)]:
    ax.annotate('', xy=(DOT_X + (0.06 if cx > 0.5 else -0.06), DOT_Y + 0.045),
                xytext=(cx, proj_y - BOX_H / 2),
                arrowprops=dict(arrowstyle='->', color=C_COMBINE, lw=2.0,
                                connectionstyle='arc3,rad=0.12'), zorder=2)

dot = plt.Circle((DOT_X, DOT_Y), 0.055, color=C_COMBINE, zorder=3)
ax.add_patch(dot)
ax.text(DOT_X, DOT_Y + 0.010, 'dot',     ha='center', va='center',
        fontsize=10, fontweight='bold', color='white', zorder=4)
ax.text(DOT_X, DOT_Y - 0.016, 'product', ha='center', va='center',
        fontsize=10, fontweight='bold', color='white', zorder=4)

ax.text(DOT_X + 0.12, DOT_Y - 0.010, '→  predicted rating (de-biased)',
        ha='left', va='center', fontsize=9, color=C_TEXT, style='italic')

plt.tight_layout()
plt.savefig('diagram.png', dpi=150, bbox_inches='tight', facecolor=C_BG)
print("Saved diagram.png")

if '--show' in sys.argv:
    plt.show()
