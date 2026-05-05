"""
Qualitative ranker inspection on canary users.

For each canary:
  1. Run CG on the synthetic user history → top-100 corpus candidates (ordered by CG score)
  2. Build ranker user features (same genre_ctx as CG; user_avg_rating + n_ratings are
     synthetic since canaries don't have raw rating histories — see SYNTHETIC_USER_AVG below)
  3. Run ranker on the 100 candidates → re-sort
  4. Print side-by-side top-10 (CG order vs Ranker order)

Reuses src/evaluate._build_user_embedding to mirror the production CG path exactly.
"""
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ranker.dataset import _build_item_features
from ranker.model import MLPRanker
from ranker.train import get_config, get_device
from src.dataset import load_features
from src.evaluate import (USER_TYPE_TO_DISLIKED_MOVIES,
                           USER_TYPE_TO_FAVORITE_MOVIES,
                           USER_TYPE_TO_GENOME_TAGS, _build_user_embedding,
                           _get_anchor_titles)
from src.train import build_model, get_v2_config


# Canaries don't have raw rating histories — they're synthetic "this person likes X".
# Pick reasonable proxies for ranker user features:
#   - SYNTHETIC_USER_AVG: 4.0 (canaries are enthusiasts; matches an active rater)
#   - n_ratings: count of fav + anchor movies in the canary's history
SYNTHETIC_USER_AVG     = 4.0
DEFAULT_CANARIES       = ['Horror Lover', 'Sci-Fi Lover', 'Western Lover',
                          'Heist Lover', 'Anime Lover', 'Crime Lover',
                          'Children\'s Movie Lover', 'WW2 Lover']
TOP_K_CG               = 100
TOP_N_DISPLAY_DEFAULT  = 10


def _resolve_cg_checkpoint() -> str:
    matches = sorted(glob.glob('saved_models/PROD_best_softmax_v2_popularity_alpha_*.pth'))
    if not matches:
        raise FileNotFoundError("No PROD softmax checkpoint found in saved_models/")
    return matches[-1]


def _build_synthetic_genre_ctx(user_type: str, fs) -> np.ndarray:
    """Mirror src.evaluate._build_user_embedding's genre_ctx construction. (40-dim)"""
    fav  = USER_TYPE_TO_FAVORITE_MOVIES[user_type]
    dis  = USER_TYPE_TO_DISLIKED_MOVIES[user_type]
    tags = USER_TYPE_TO_GENOME_TAGS.get(user_type, [])
    anchors = _get_anchor_titles(fs, tags, exclude=set(fav))
    weighted = ([(t, 2.0) for t in fav] +
                [(t, 1.0) for t in anchors] +
                [(t, -2.0) for t in dis])

    n_genres = len(fs.genres_ordered)
    ctx = np.zeros(2 * n_genres, dtype=np.float32)
    rsum, rcount = {}, {}
    for t, w in weighted:
        mid = fs.title_to_movieId.get(t)
        if mid is None:
            continue
        for g in fs.movieId_to_genres.get(mid, []):
            rsum[g]   = rsum.get(g, 0.0) + w
            rcount[g] = rcount.get(g, 0)  + 1
    total = sum(rcount.values())
    for g, s in rsum.items():
        if g in fs.user_context_genre_avg_rating_to_i:
            ctx[fs.user_context_genre_avg_rating_to_i[g]] = s / rcount[g]
        if g in fs.user_context_genre_watch_count_to_i:
            ctx[fs.user_context_genre_watch_count_to_i[g]] = rcount[g] / max(total, 1)
    n_history = len(fav) + len(anchors)
    return ctx, n_history


def _format_title(fs, mid: int, max_w: int) -> str:
    t = fs.movieId_to_title[mid]
    return t if len(t) <= max_w else t[:max_w - 1] + '…'


def run_canary(cg_checkpoint: str | None = None,
               ranker_checkpoint: str | None = None,
               canaries: list = None,
               data_dir: str = 'data',
               top_n: int = TOP_N_DISPLAY_DEFAULT,
               output_file: str | None = None):
    """
    Run side-by-side CG vs Ranker recommendations for a list of canary users.

    Args:
        cg_checkpoint: CG checkpoint (defaults to PROD softmax)
        ranker_checkpoint: ranker checkpoint (defaults to most recent)
        canaries: list of canary user names; defaults to DEFAULT_CANARIES
        top_n: how many recommendations to show per user
        output_file: override auto-generated save path (default: ranker/canary_results/<ckpt>.txt)
    """
    import io
    canaries = canaries or DEFAULT_CANARIES

    if ranker_checkpoint is None:
        matches = glob.glob('saved_models/ranker/ranker_mlp_*.pth')
        if not matches:
            raise FileNotFoundError("No ranker checkpoint found in saved_models/ranker/")
        ranker_checkpoint = max(matches, key=os.path.getmtime)
    cg_checkpoint = cg_checkpoint or _resolve_cg_checkpoint()

    out = io.StringIO()

    def emit(line: str = ''):
        print(line)
        out.write(line + '\n')

    if output_file is None:
        base = os.path.splitext(os.path.basename(ranker_checkpoint))[0]
        output_file = f"ranker/canary_results/{base}.txt"

    emit(f"CG checkpoint:     {cg_checkpoint}")
    emit(f"Ranker checkpoint: {ranker_checkpoint}")
    emit(f"Top-N per canary:  {top_n}")
    emit(f"Canaries:          {len(canaries)}")
    emit('')

    fs = load_features(data_dir=data_dir, version='v1')
    movie_stats = pd.read_parquet(f'{data_dir}/ranker_movie_stats.parquet')

    device = get_device()
    emit(f"Device: {device}")
    emit('')

    # CG model
    cg_cfg = get_v2_config()
    cg = build_model(cg_cfg, fs)
    cg.load_state_dict(torch.load(cg_checkpoint, weights_only=True, map_location='cpu'))
    cg.eval().to(device)
    with torch.no_grad():
        V_all = cg.full_item_embedding()  # (n_movies, 128) on device

    # Item features for ranker
    item_features = _build_item_features(fs, movie_stats).to(device)
    item_dim = item_features.shape[1]
    user_dim = 40 + 2  # genre_ctx + (avg, log1p_count)

    # Load architecture params from checkpoint config sidecar.
    cfg = get_config()
    config_path = os.path.splitext(ranker_checkpoint)[0] + '_config.json'
    n_interact = 0
    if os.path.exists(config_path):
        with open(config_path) as _f:
            saved_cfg = json.load(_f)
        n_interact = saved_cfg.get('n_interaction_features', 0)
        for k in ('hidden_dims', 'dropout', 'genome_dim', 'genome_bottleneck_dim'):
            if k in saved_cfg:
                cfg[k] = saved_cfg[k]

    # Ranker model
    ranker = MLPRanker(user_dim, item_dim + n_interact,
                        hidden_dims=cfg['hidden_dims'],
                        dropout=cfg['dropout'],
                        genome_dim=cfg['genome_dim'],
                        genome_bottleneck_dim=cfg['genome_bottleneck_dim']).to(device)
    ranker.load_state_dict(torch.load(ranker_checkpoint, weights_only=True, map_location=device))
    ranker.eval()

    # Most recent timestamp bin (canary convention from src/evaluate.py)
    ts_max_bin = torch.bucketize(
        torch.tensor([float(fs.timestamp_bins[-1].item())]),
        fs.timestamp_bins, right=False,
    ).to(device)

    for user_type in canaries:
        if user_type not in USER_TYPE_TO_FAVORITE_MOVIES:
            print(f"[skip] unknown canary: {user_type}")
            continue

        # ── 1. CG retrieval ──────────────────────────────────────────────────
        with torch.no_grad():
            user_emb = _build_user_embedding(cg, fs, user_type, ts_max_bin)  # (1, 128)
            cg_scores = (user_emb @ V_all.T).squeeze(0)                       # (n_movies,)

        # Build exclusion set (favs + dislikes + anchors) — same as src/evaluate
        fav  = USER_TYPE_TO_FAVORITE_MOVIES[user_type]
        dis  = USER_TYPE_TO_DISLIKED_MOVIES[user_type]
        tags = USER_TYPE_TO_GENOME_TAGS.get(user_type, [])
        anchors = _get_anchor_titles(fs, tags, exclude=set(fav))
        exclude_titles = set(fav) | set(dis) | set(anchors)

        # Take CG top-100 (excluding any titles in the exclusion set)
        sorted_cg = torch.argsort(cg_scores, descending=True).tolist()
        cg_top100_corpus = []
        for idx in sorted_cg:
            mid = fs.top_movies[idx]
            if fs.movieId_to_title[mid] not in exclude_titles:
                cg_top100_corpus.append(idx)
            if len(cg_top100_corpus) >= TOP_K_CG:
                break

        # ── 2. Ranker rescore on those 100 ───────────────────────────────────
        genre_ctx, n_history = _build_synthetic_genre_ctx(user_type, fs)
        user_count_log1p     = float(np.log1p(n_history))
        user_feat_np = np.concatenate([
            genre_ctx,
            [SYNTHETIC_USER_AVG, user_count_log1p],
        ]).astype(np.float32)

        cand_t      = torch.tensor(cg_top100_corpus, dtype=torch.long, device=device)  # (100,)
        item_feat_b = item_features[cand_t]                                            # (100, item_dim)
        user_feat_b = torch.from_numpy(user_feat_np).to(device).unsqueeze(0).expand(len(cand_t), -1)

        with torch.no_grad():
            if n_interact > 0:
                # CG score passthrough (cosine similarity, already computed above)
                cand_cg_scores = cg_scores[cand_t]  # (100,)

                # Genome pool for synthetic user: equal-weight avg of fav + anchor raw genome vecs
                fav_anchor_cidxs = []
                for t_name in list(fav) + list(anchors):
                    mid = fs.title_to_movieId.get(t_name)
                    if mid is not None and mid in fs.item_emb_movieId_to_i:
                        fav_anchor_cidxs.append(fs.item_emb_movieId_to_i[mid])
                if fav_anchor_cidxs:
                    hist_t   = torch.tensor(fav_anchor_cidxs, dtype=torch.long, device=device)
                    gp_raw   = cg.genome_context_buffer[hist_t].mean(dim=0, keepdim=True)
                    gp_norm  = F.normalize(gp_raw, p=2, dim=1)   # (1, genome_dim)
                else:
                    gp_norm = torch.zeros(1, cg.genome_context_buffer.shape[1], device=device)

                cand_genome  = F.normalize(cg.genome_context_buffer[cand_t], p=2, dim=1)  # (100, gd)
                genome_cos   = (gp_norm * cand_genome).sum(dim=1)                          # (100,)

                interact_b   = torch.stack([cand_cg_scores, genome_cos], dim=1)  # (100, 2)
                item_feat_b  = torch.cat([item_feat_b, interact_b], dim=1)       # (100, item_dim+2)

            ranker_scores = ranker(user_feat_b, item_feat_b)                           # (100,) logits
        rk_order = torch.argsort(ranker_scores, descending=True).tolist()
        rk_top   = [cg_top100_corpus[i] for i in rk_order[:top_n]]
        cg_top   = cg_top100_corpus[:top_n]

        # ── 3. Render side-by-side ───────────────────────────────────────────
        col_w = 50
        bar_w = col_w * 2 + 8
        title_line = user_type
        if tags:
            title_line += f"  |  Genome: {', '.join(tags)}"
        emit('')
        emit('═' * bar_w)
        emit(title_line)
        emit('═' * bar_w)
        emit(f"Liked: {', '.join(fav)}")
        emit('')
        emit(f"{'#':<3}  {f'CG top-{top_n}':<{col_w}}  {f'Ranker top-{top_n}':<{col_w}}")
        emit('─' * bar_w)
        for i in range(top_n):
            cg_t = _format_title(fs, fs.top_movies[cg_top[i]], col_w) if i < len(cg_top) else ''
            rk_t = _format_title(fs, fs.top_movies[rk_top[i]], col_w) if i < len(rk_top) else ''
            emit(f"{i+1:<3}  {cg_t:<{col_w}}  {rk_t:<{col_w}}")

    emit('')

    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(out.getvalue())
    print(f"Wrote canary results → {output_file}  ({out.getvalue().count(chr(10)):,} lines)")
    return output_file


def dump_canary(top_n: int = 20, output_file: str | None = None,
                cg_checkpoint: str | None = None,
                ranker_checkpoint: str | None = None):
    """
    Dump top-N side-by-side CG vs Ranker recommendations for ALL canaries to a file.
    Default output path: canary_results/ranker_<ranker_checkpoint_basename>.txt
    """
    if ranker_checkpoint is None:
        matches = glob.glob('saved_models/ranker/ranker_mlp_*.pth')
        if not matches:
            raise FileNotFoundError("No ranker checkpoint found in saved_models/ranker/")
        ranker_checkpoint = max(matches, key=os.path.getmtime)
    if output_file is None:
        base = os.path.splitext(os.path.basename(ranker_checkpoint))[0]
        output_file = f"ranker/canary_results/{base}.txt"

    return run_canary(
        cg_checkpoint=cg_checkpoint,
        ranker_checkpoint=ranker_checkpoint,
        canaries=list(USER_TYPE_TO_FAVORITE_MOVIES.keys()),
        top_n=top_n,
        output_file=output_file,
    )
