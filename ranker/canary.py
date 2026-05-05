"""
Qualitative ranker inspection on canary users.

For each canary:
  1. Run CG on the synthetic user history → top-100 corpus candidates (CG order)
  2. Build synthetic ranker user inputs (X_history, X_hist_ratings, ts, user_genre_ctx)
  3. Run ranker on the 100 candidates → re-sort
  4. Print side-by-side top-N

Synthetic history weights (mirror src/evaluate._build_user_embedding):
  favorites = 2.0, anchors = 1.0, disliked = -2.0
"""
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ranker.dataset import _build_item_features, compute_cross_features
from ranker.train import build_ranker, get_config, get_device
from src.dataset import MAX_HISTORY_LEN, load_features
from src.evaluate import (USER_TYPE_TO_DISLIKED_MOVIES,
                           USER_TYPE_TO_FAVORITE_MOVIES,
                           USER_TYPE_TO_GENOME_TAGS, _build_user_embedding,
                           _get_anchor_titles)
from src.train import build_model, get_v2_config


VALUE_FAVORITE_MOVIE_RATING = 2.0
VALUE_ANCHOR_MOVIE_RATING   = 1.0
VALUE_DISLIKED_MOVIE_RATING = -2.0

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


def _build_synthetic_genre_ctx(user_type: str, fs) -> tuple:
    """40-dim user_genre_ctx; mirrors src.evaluate._build_user_embedding's ctx construction."""
    fav  = USER_TYPE_TO_FAVORITE_MOVIES[user_type]
    dis  = USER_TYPE_TO_DISLIKED_MOVIES[user_type]
    tags = USER_TYPE_TO_GENOME_TAGS.get(user_type, [])
    anchors = _get_anchor_titles(fs, tags, exclude=set(fav))
    weighted = ([(t, VALUE_FAVORITE_MOVIE_RATING) for t in fav] +
                [(t, VALUE_ANCHOR_MOVIE_RATING)   for t in anchors] +
                [(t, VALUE_DISLIKED_MOVIE_RATING) for t in dis])

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
    n_history = len(fav) + len(anchors) + len(dis)
    return ctx, n_history, fav, anchors, dis


def _build_synthetic_history(fav, anchors, dis, fs, pad_idx, max_hist=MAX_HISTORY_LEN):
    """Build padded (max_hist,) corpus indices and ratings for a synthetic canary user."""
    weighted = ([(t, VALUE_FAVORITE_MOVIE_RATING) for t in fav] +
                [(t, VALUE_ANCHOR_MOVIE_RATING)   for t in anchors] +
                [(t, VALUE_DISLIKED_MOVIE_RATING) for t in dis])
    pairs = []
    for t, w in weighted:
        mid = fs.title_to_movieId.get(t)
        if mid is None:
            continue
        cidx = fs.item_emb_movieId_to_i.get(mid)
        if cidx is not None:
            pairs.append((cidx, w))
    if not pairs:
        return (np.full(max_hist, pad_idx, dtype=np.int64),
                np.zeros(max_hist, dtype=np.float32))
    pairs = pairs[-max_hist:]                  # right-aligned, take the last max_hist if too long
    ids   = np.full(max_hist, pad_idx, dtype=np.int64)
    rats  = np.zeros(max_hist, dtype=np.float32)
    take  = len(pairs)
    ids[max_hist - take:]  = [p[0] for p in pairs]
    rats[max_hist - take:] = [p[1] for p in pairs]
    return ids, rats


def _format_title(fs, mid: int, max_w: int) -> str:
    t = fs.movieId_to_title[mid]
    return t if len(t) <= max_w else t[:max_w - 1] + '…'


def run_canary(cg_checkpoint: str | None = None,
               ranker_checkpoint: str | None = None,
               canaries: list = None,
               data_dir: str = 'data',
               top_n: int = TOP_N_DISPLAY_DEFAULT,
               output_file: str | None = None):
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

    # CG model (read-only, used for retrieval only).
    cg_cfg = get_v2_config()
    cg = build_model(cg_cfg, fs)
    cg.load_state_dict(torch.load(cg_checkpoint, weights_only=True, map_location='cpu'))
    cg.eval().to(device)
    with torch.no_grad():
        V_all = cg.full_item_embedding()  # (n_movies, 128) on device

    # Item features for ranker cross-feature computation.
    item_features = _build_item_features(fs, movie_stats).to(device)
    pad_idx = item_features.shape[0]   # = len(fs.top_movies)

    # Load ranker config from sidecar, then build ranker (no CG coupling — buffers from FeatureStore).
    cfg = get_config()
    config_path = os.path.splitext(ranker_checkpoint)[0] + '_config.json'
    if os.path.exists(config_path):
        with open(config_path) as _f:
            saved_cfg = json.load(_f)
        for k in ('hidden_dims', 'dropout', 'item_id_emb_dim', 'item_genre_emb_dim',
                  'item_tag_emb_dim', 'item_genome_emb_dim', 'item_year_emb_dim',
                  'user_genre_emb_dim', 'user_genome_ctx_emb_dim', 'ts_emb_dim',
                  'n_cross_features',
                  'use_user_watch_history_genome_pool', 'use_user_genome_context'):
            if k in saved_cfg:
                cfg[k] = saved_cfg[k]
    n_cross = cfg.get('n_cross_features', 0)

    ranker = build_ranker(cfg, fs).to(device)
    ranker.load_state_dict(torch.load(ranker_checkpoint, weights_only=True, map_location=device))
    ranker.eval()

    # Most recent timestamp bin (canary convention from src/evaluate.py).
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
            user_emb  = _build_user_embedding(cg, fs, user_type, ts_max_bin)  # (1, 128)
            cg_scores = (user_emb @ V_all.T).squeeze(0)                       # (n_movies,)

        fav  = USER_TYPE_TO_FAVORITE_MOVIES[user_type]
        dis  = USER_TYPE_TO_DISLIKED_MOVIES[user_type]
        tags = USER_TYPE_TO_GENOME_TAGS.get(user_type, [])
        anchors = _get_anchor_titles(fs, tags, exclude=set(fav))
        exclude_titles = set(fav) | set(dis) | set(anchors)

        sorted_cg = torch.argsort(cg_scores, descending=True).tolist()
        cg_top100_corpus = []
        for idx in sorted_cg:
            mid = fs.top_movies[idx]
            if fs.movieId_to_title[mid] not in exclude_titles:
                cg_top100_corpus.append(idx)
            if len(cg_top100_corpus) >= TOP_K_CG:
                break

        # ── 2. Build synthetic ranker user inputs ────────────────────────────
        genre_ctx, n_history, _, _, _ = _build_synthetic_genre_ctx(user_type, fs)
        user_count_log1p = float(np.log1p(n_history))
        hist_ids, hist_rats = _build_synthetic_history(fav, anchors, dis, fs, pad_idx)

        n_cand = len(cg_top100_corpus)
        cand_t = torch.tensor(cg_top100_corpus, dtype=torch.long, device=device)  # (n_cand,)

        # User-side: one synthetic "user" per canary — compute user_embedding ONCE.
        ugc_one = torch.from_numpy(genre_ctx).to(device).unsqueeze(0)             # (1, 40)
        xh_one  = torch.from_numpy(hist_ids).to(device).unsqueeze(0)              # (1, max_hist)
        xhr_one = torch.from_numpy(hist_rats).to(device).unsqueeze(0)             # (1, max_hist)
        ts_one  = ts_max_bin                                                       # (1,)

        # Expanded per-candidate copies for cross-feature computation only.
        ugc_t = ugc_one.expand(n_cand, -1)

        # User mean year_norm from fav + anchor corpus indices.
        valid_cidxs = [c for c in hist_ids.tolist() if c < pad_idx]
        if valid_cidxs:
            valid_t = torch.tensor(valid_cidxs, dtype=torch.long, device=device)
            user_year_val = float(item_features[valid_t, 1150].mean().item())
        else:
            user_year_val = 0.5

        cand_feat       = item_features[cand_t]
        cand_genre_oh   = cand_feat[:, 1128:1148]
        cand_global_avg = cand_feat[:, 1148]
        cand_log_count  = cand_feat[:, 1149]
        cand_year_norm  = cand_feat[:, 1150]

        user_avg_t  = torch.full((n_cand,), SYNTHETIC_USER_AVG, device=device)
        user_cnt_t  = torch.full((n_cand,), user_count_log1p,    device=device)
        user_year_t = torch.full((n_cand,), user_year_val,       device=device)

        # Genome cosine for synthetic user vs each candidate (using ranker's own genome_buffer).
        if valid_cidxs:
            valid_t = torch.tensor(valid_cidxs, dtype=torch.long, device=device)
            gp_raw  = ranker.genome_buffer[valid_t].mean(dim=0, keepdim=True)
            gp_norm = torch.nn.functional.normalize(gp_raw, p=2, dim=1)
        else:
            gp_norm = torch.zeros(1, ranker.genome_buffer.shape[1], device=device)
        cand_genome  = torch.nn.functional.normalize(ranker.genome_buffer[cand_t], p=2, dim=1)
        genome_cos_t = (gp_norm * cand_genome).sum(dim=1)                      # (n_cand,)

        cross = compute_cross_features(
            ugc_t, user_avg_t, user_cnt_t, user_year_t,
            cand_genre_oh, cand_year_norm, cand_global_avg, cand_log_count,
            genome_cos_t,
        )

        with torch.no_grad():
            user_concat_one = ranker.user_embedding(ugc_one, xh_one, xhr_one, ts_one)  # (1, U)
            user_concat_exp = user_concat_one.expand(n_cand, -1)                       # cheap view
            item_concat     = ranker.item_embedding(cand_t)                            # (n_cand, I)
            ranker_scores   = ranker.score_pairs(user_concat_exp, item_concat, cross)  # (n_cand,)

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
            cg_t_str = _format_title(fs, fs.top_movies[cg_top[i]], col_w) if i < len(cg_top) else ''
            rk_t_str = _format_title(fs, fs.top_movies[rk_top[i]], col_w) if i < len(rk_top) else ''
            emit(f"{i+1:<3}  {cg_t_str:<{col_w}}  {rk_t_str:<{col_w}}")

    emit('')

    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(out.getvalue())
    print(f"Wrote canary results → {output_file}  ({out.getvalue().count(chr(10)):,} lines)")
    return output_file


def dump_canary(top_n: int = 20, output_file: str | None = None,
                cg_checkpoint: str | None = None,
                ranker_checkpoint: str | None = None):
    """Dump top-N side-by-side CG vs Ranker recommendations for ALL canaries."""
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
