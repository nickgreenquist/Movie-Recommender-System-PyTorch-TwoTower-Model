"""
Similar-movies component diagnostic.

The demo "Similar" tab (streamlit_app.py:tab_similar) and probe_similar both rank
neighbors by the FULL combined item embedding — the L2-normalized item_projection output
over the concat of [genre|tag|genome|llm|id|year]. When those neighbors look weird, we
can't tell which sub-tower is responsible from the combined vector alone.

This script decomposes the item tower into four separable item-side representations and
runs the same cosine-similarity neighbor search under each, so we can see which component
drives (or wrecks) the neighborhoods:

  1. full     — model.item_embedding(idx)                          (128-dim, what prod uses)
  2. id       — item_embedding_tower(item_embedding_lookup(idx))   ( 32-dim, CF signal + proj)
  3. genome   — item_genome_tag_tower(genome_context_buffer[idx])  ( 32-dim, genome projection)
  4. llm      — item_llm_feature_tower(llm_feature_buffer[idx])    ( 32-dim, LLM projection)

Each matrix is L2-normalized row-wise, so (M @ q.T) == cosine similarity (same protocol as
probe_similar / tab_similar). For each seed we drop the seed itself and take top-N.

Output: a markdown report with one side-by-side table per seed (rank × method).

Run:  python tools/similar_movies_diagnostic.py
"""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.checkpoint import load_checkpoint
from src.dataset import load_features
from src.train import build_model


PROD_CKPT = 'saved_models/PROD_best_softmax_genome_tags_llm_features_popularity_alpha_05_20260612_080719.pth'
OUT_MD    = 'tools/results/similar_movies_component_breakdown.md'
TOP_N     = 10

SEED_TITLES = [
    'Lord of the Rings: The Return of the King, The (2003)',
    '28 Days Later (2002)',
    'Ong-Bak: The Thai Warrior (Ong Bak) (2003)',
    'Princess Mononoke (Mononoke-hime) (1997)',
    'Iron Man 3 (2013)',
    'Die Hard 2 (1990)',
    'Casino (1995)',
    '1917 (2019)',
    'Holiday, The (2006)',
    '40-Year-Old Virgin, The (2005)',
    'Texas Chainsaw Massacre, The (2003)',
    'Toy Story 2 (1999)',
    'Shrek (2001)',
    'Hangover, The (2009)',
    'Hobbit: An Unexpected Journey, The (2012)',
]

METHODS = [
    ('1. Full item tower (128d)', 'full'),
    ('2. Item-ID emb + proj (32d)', 'id'),
    ('3. Genome-tag proj (32d)', 'genome'),
    ('4. LLM-feature proj (32d)', 'llm'),
]


def _norm_key(title: str) -> str:
    """Loose key for fuzzy title resolution: lowercase, collapse whitespace."""
    return ' '.join(title.lower().split())


def resolve_title(title: str, title_to_movieId: dict) -> tuple:
    """Resolve a requested title to (movieId, canonical_title). Exact → normalized → substring."""
    if title in title_to_movieId:
        return title_to_movieId[title], title
    norm_map = {_norm_key(t): t for t in title_to_movieId}
    key = _norm_key(title)
    if key in norm_map:
        canon = norm_map[key]
        return title_to_movieId[canon], canon
    # Substring fallback: drop the trailing "(year)" and match the leading title text.
    stem = key.rsplit('(', 1)[0].strip()
    hits = [t for k, t in norm_map.items() if k.startswith(stem)]
    if len(hits) == 1:
        return title_to_movieId[hits[0]], hits[0]
    if len(hits) > 1:
        # Prefer one whose year matches, else give up and report ambiguity.
        yr = key.rsplit('(', 1)[-1] if '(' in key else ''
        for t in hits:
            if yr and yr in _norm_key(t):
                return title_to_movieId[t], t
        return None, f'AMBIGUOUS: {hits[:5]}'
    return None, None


def build_component_matrices(model, all_idxs):
    """Return {method_key: L2-normalized (n_movies, dim) tensor} for the four item-side reps."""
    model.eval()
    with torch.no_grad():
        full   = model.item_embedding(all_idxs)                                      # already L2-normed
        id_emb = model.item_embedding_tower(model.item_embedding_lookup(all_idxs))   # 32d, CF + proj
        genome = model.item_genome_tag_tower(model.genome_context_buffer[all_idxs])  # 32d
        llm    = model.item_llm_feature_tower(model.llm_feature_buffer[all_idxs])    # 32d

    return {
        'full':   F.normalize(full,   dim=1),
        'id':     F.normalize(id_emb, dim=1),
        'genome': F.normalize(genome, dim=1),
        'llm':    F.normalize(llm,    dim=1),
    }


def top_neighbors(mat, seed_row, all_ids, movieId_to_title, seed_mid, top_n):
    """Top-n (title, score) neighbors of seed_row in mat by cosine, excluding the seed itself."""
    q    = mat[seed_row:seed_row + 1]                 # (1, d), already unit-norm
    sims = (mat @ q.T).squeeze(-1)                    # (n_movies,)
    order = sims.argsort(descending=True).tolist()
    out = []
    for idx in order:
        mid = all_ids[idx]
        if mid == seed_mid:
            continue
        out.append((movieId_to_title[mid], float(sims[idx])))
        if len(out) >= top_n:
            break
    return out


def main():
    print(f"Loading checkpoint: {PROD_CKPT}")
    config, state_dict = load_checkpoint(PROD_CKPT)
    print(f"  feature_towers={config.get('feature_towers')}  base_towers={config.get('base_towers')}")

    print("Loading features ...")
    fs = load_features('data')

    model = build_model(config, fs)
    model.load_state_dict(state_dict)
    model = model.to('cpu')   # corpus is tiny; CPU keeps it simple + deterministic
    model.eval()

    assert model.has_genome and model.has_llm, "prod checkpoint must have both genome + llm towers"

    # Corpus index order == fs.top_movies order (the buffers/lookup are built in this order).
    all_ids  = [int(m) for m in fs.top_movies]
    all_idxs = torch.tensor([fs.item_emb_movieId_to_i[m] for m in all_ids], dtype=torch.long)
    movieId_to_title = fs.movieId_to_title
    # corpus index (row in each matrix) for a given movieId
    mid_to_row = {mid: i for i, mid in enumerate(all_ids)}

    print(f"Corpus: {len(all_ids)} movies. Building 4 component matrices ...")
    mats = build_component_matrices(model, all_idxs)

    # Resolve seeds.
    print("\nResolving seed titles:")
    seeds = []
    for t in SEED_TITLES:
        mid, canon = resolve_title(t, fs.title_to_movieId)
        if mid is None or mid not in mid_to_row:
            print(f"  [MISS] {t!r}  ->  {canon}")
            seeds.append((t, None, None))
            continue
        if canon != t:
            print(f"  [ok]   {t!r}  ->  {canon!r}")
        else:
            print(f"  [ok]   {t}")
        seeds.append((t, mid, canon))

    # Build report.
    lines = []
    lines.append("# Similar-Movies Component Breakdown\n")
    lines.append(f"**Checkpoint:** `{os.path.basename(PROD_CKPT)}`  ")
    lines.append(f"**Corpus:** {len(all_ids)} movies &nbsp;|&nbsp; **Top-N:** {TOP_N} &nbsp;|&nbsp; "
                 "metric: cosine over L2-normalized item-side vectors\n")
    lines.append(
        "Four item-side representations, each ranked the same way the demo ranks "
        "(`cosine`, seed excluded):\n\n"
        "1. **Full item tower (128d)** — `item_embedding(idx)`; the L2-normed projection over "
        "`[genre|tag|genome|llm|id|year]`. **This is what the demo / `probe_similar` actually use.**\n"
        "2. **Item-ID emb + proj (32d)** — `item_embedding_tower(item_embedding_lookup(idx))`; "
        "the pure collaborative-filtering signal.\n"
        "3. **Genome-tag proj (32d)** — `item_genome_tag_tower(genome_buffer[idx])`; content from "
        "the 1128-dim genome scores.\n"
        "4. **LLM-feature proj (32d)** — `item_llm_feature_tower(llm_buffer[idx])`; content from "
        "the 132-dim LLM features.\n"
    )

    for orig, mid, canon in seeds:
        lines.append(f"\n## {canon or orig}\n")
        if mid is None:
            lines.append(f"_Not found in corpus (requested as `{orig}`)._\n")
            continue
        row = mid_to_row[mid]
        per_method = {}
        for _, key in METHODS:
            per_method[key] = top_neighbors(mats[key], row, all_ids, movieId_to_title, mid, TOP_N)

        header = "| # | " + " | ".join(label for label, _ in METHODS) + " |"
        sep    = "|---|" + "|".join(["---"] * len(METHODS)) + "|"
        lines.append(header)
        lines.append(sep)
        for r in range(TOP_N):
            cells = []
            for _, key in METHODS:
                title, score = per_method[key][r]
                cells.append(f"{title} ({score:.3f})")
            lines.append(f"| {r+1} | " + " | ".join(cells) + " |")
        lines.append("")

    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
    with open(OUT_MD, 'w') as f:
        f.write("\n".join(lines))
    print(f"\n→ wrote {OUT_MD}")


if __name__ == '__main__':
    main()
