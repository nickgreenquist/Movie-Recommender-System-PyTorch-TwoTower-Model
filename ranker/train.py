"""
Stage 2 — Wide & Deep ranker training loop with full v3 CG feature parity.

Architecture (see ranker/model.py):
  User concat (196): pool_full(32) + pool_liked(32) + pool_disliked(32)
                     + pool_weighted(32) + genome_ctx(32) + genre_emb(32) + ts_emb(4)
  Item concat  (96): item_genre(8) + item_tag(16) + item_genome(32)
                     + item_id(32) + year(8)
  Deep MLP (292 → [256,128,64]) → 64
  Head: cat(deep_out(64), cross_features(5)) → Linear(69, 1)

Cross features (wide bypass): genome_cosine, genre_affinity, era_gap, rating_calibration,
popularity_match — computed in dataset.sample_batch / evaluate.compute_label_ranks.

Menon α: same mechanic as before — added to logits during training, raw at inference.
Currently disabled (α=0) until ranker beats CG.
"""
import glob
import json
import os
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

from ranker.dataset import (compute_cross_features, load_splits, sample_batch,
                            CANDIDATES_PER_ROW)
from ranker.evaluate import (cg_baseline, compute_label_ranks,
                              evaluate_ndcg_mrr, hit_rates_from_ranks,
                              _ndcg_mrr_from_ranks)
from ranker.model import WideDeepRanker


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def get_config() -> dict:
    return {
        'lr':                 1e-3,
        'weight_decay':       0.0,
        'adam_eps':           1e-6,   # match CG — protects sparse embedding updates from oversized steps

        'batch_size':         4096,
        'training_steps':     150_000,
        'log_every':          2_000,
        'grad_clip':          1.0,
        'hidden_dims':        [256, 128, 64],
        'dropout':            0.0,
        'seed':               42,
        'checkpoint_dir':     'saved_models/ranker',
        'easy_neg_frac':         0.5,
        # CG-equivalent sub-tower output dims (mirror src/train.get_config defaults)
        'item_id_emb_dim':       32,
        'item_genre_emb_dim':    8,
        'item_tag_emb_dim':      16,
        'item_genome_emb_dim':   32,
        'item_year_emb_dim':     8,
        'user_genre_emb_dim':    32,
        'user_genome_ctx_emb_dim': 32,
        'ts_emb_dim':            4,
        # Wide bypass: genome_cosine + genre_affinity + era_gap + rating_cal + pop_match.
        'n_cross_features':      5,
        'popularity_alpha':      0.0,
        # Training-time eval: sample N val rows (deterministic, fixed seed) for fast logging.
        # Final eval (end of training) and evaluate_only always use the full val set.
        'n_eval_samples':        20_000,
        # Warm start from CG (Option B): copy CG's pretrained tower + LayerNorm
        # weights into the ranker before training. Ranker still owns the params and
        # fine-tunes them — no runtime CG coupling. Set to None to disable.
        # Glob matches the latest PROD softmax checkpoint (currently v3).
        'warm_start_cg_checkpoint':           'saved_models/PROD_best_softmax_v2_popularity_alpha_*.pth',
    }


def _config_path(checkpoint_path: str) -> str:
    return os.path.splitext(checkpoint_path)[0] + '_config.json'


def _save_config(config: dict, checkpoint_path: str) -> None:
    with open(_config_path(checkpoint_path), 'w') as f:
        json.dump(config, f, indent=2)


# ── Model factory ───────────────────────────────────────────────────────────

def _buffers_from_fs(fs):
    """Build the four item-feature buffers (with padding row) from a FeatureStore."""
    n_movies = len(fs.top_movies)

    genome_matrix = np.array(
        [fs.movieId_to_genome_tag_context[mid] for mid in fs.top_movies], dtype=np.float32)
    genome_buf = torch.from_numpy(np.vstack(
        [genome_matrix, np.zeros((1, genome_matrix.shape[1]), dtype=np.float32)]))

    genre_matrix = np.array(
        [fs.movieId_to_genre_context[mid] for mid in fs.top_movies], dtype=np.float32)
    genre_buf = torch.from_numpy(np.vstack(
        [genre_matrix, np.zeros((1, genre_matrix.shape[1]), dtype=np.float32)]))

    tag_matrix = np.array(
        [fs.movieId_to_tag_context[mid] for mid in fs.top_movies], dtype=np.float32)
    tag_buf = torch.from_numpy(np.vstack(
        [tag_matrix, np.zeros((1, tag_matrix.shape[1]), dtype=np.float32)]))

    year_array = np.array(
        [fs.year_to_i[fs.movieId_to_year[mid]] for mid in fs.top_movies], dtype=np.int64)
    year_buf = torch.from_numpy(np.concatenate([year_array, np.zeros((1,), dtype=np.int64)]))

    return n_movies, genome_buf, genre_buf, tag_buf, year_buf


# Map of CG state-dict keys → ranker state-dict keys for warm-start transfer.
# Only includes parameters whose semantics + shapes are equivalent between CG and ranker.
# After copy, these become ranker-owned parameters that fine-tune during training.
_CG_TO_RANKER_KEY_MAP = {
    # Item-side: shared embedding + sub-towers
    'item_embedding_lookup.weight':         'item_id_lookup.weight',
    'item_embedding_tower.0.weight':        'item_id_tower.0.weight',
    'item_embedding_tower.0.bias':          'item_id_tower.0.bias',
    'item_genre_tower.0.weight':            'item_genre_tower.0.weight',
    'item_genre_tower.0.bias':              'item_genre_tower.0.bias',
    'item_tag_tower.0.weight':              'item_tag_tower.0.weight',
    'item_tag_tower.0.bias':                'item_tag_tower.0.bias',
    'item_genome_tag_tower.0.weight':       'item_genome_tower.0.weight',
    'item_genome_tag_tower.0.bias':         'item_genome_tower.0.bias',
    'year_embedding_lookup.weight':         'year_lookup.weight',
    'year_embedding_tower.0.weight':        'year_tower.0.weight',
    'year_embedding_tower.0.bias':          'year_tower.0.bias',
    # User-side towers
    'user_genre_tower.0.weight':            'user_genre_tower.0.weight',
    'user_genre_tower.0.bias':              'user_genre_tower.0.bias',
    'timestamp_embedding_lookup.weight':    'ts_lookup.weight',
    'timestamp_embedding_tower.0.weight':   'ts_tower.0.weight',
    'timestamp_embedding_tower.0.bias':     'ts_tower.0.bias',
    'user_genome_context_tower.0.weight':   'user_genome_ctx_tower.0.weight',
    'user_genome_context_tower.0.bias':     'user_genome_ctx_tower.0.bias',
    # 4-pool LayerNorms (CG v3) — same names on both sides
    'hist_full_norm.weight':                'hist_full_norm.weight',
    'hist_full_norm.bias':                  'hist_full_norm.bias',
    'hist_liked_norm.weight':               'hist_liked_norm.weight',
    'hist_liked_norm.bias':                 'hist_liked_norm.bias',
    'hist_disliked_norm.weight':            'hist_disliked_norm.weight',
    'hist_disliked_norm.bias':              'hist_disliked_norm.bias',
    'hist_weighted_norm.weight':            'hist_weighted_norm.weight',
    'hist_weighted_norm.bias':              'hist_weighted_norm.bias',
}


def _resolve_cg_checkpoint(spec: str | None) -> str | None:
    """Resolve a glob/path/None to a concrete CG checkpoint path. None disables warm start."""
    if not spec:
        return None
    matches = sorted(glob.glob(spec))
    if not matches:
        raise FileNotFoundError(
            f"warm_start_cg_checkpoint='{spec}' matched no file. "
            f"Set to None to skip warm start."
        )
    return matches[-1]


def _warm_start_from_cg(ranker: WideDeepRanker, cg_checkpoint: str) -> None:
    """
    Initialize ranker parameters from a CG checkpoint (Option B: warm start, then fine-tune).

    Copies parameter VALUES — the ranker still owns the params and they update during training.
    No runtime CG dependency: the ranker's own (now-fine-tuned) state_dict is what gets saved.
    """
    cg_state     = torch.load(cg_checkpoint, weights_only=True, map_location='cpu')
    ranker_state = ranker.state_dict()

    transfer, missing_in_cg, shape_mismatch = {}, [], []
    for cg_key, ranker_key in _CG_TO_RANKER_KEY_MAP.items():
        if cg_key not in cg_state:
            missing_in_cg.append(cg_key)
            continue
        if ranker_key not in ranker_state:
            continue   # silently skip — ranker may have been built without this layer
        cg_t   = cg_state[cg_key]
        rnk_t  = ranker_state[ranker_key]
        if cg_t.shape != rnk_t.shape:
            shape_mismatch.append((cg_key, tuple(cg_t.shape), tuple(rnk_t.shape)))
            continue
        transfer[ranker_key] = cg_t

    # strict=False — we're only loading a subset of ranker params (the rest stay at fresh init)
    result = ranker.load_state_dict(transfer, strict=False)
    n_transferred = len(transfer)
    n_total       = len(ranker_state)

    print(f"Warm start from CG checkpoint: {os.path.basename(cg_checkpoint)}")
    print(f"  Transferred {n_transferred} parameter tensors out of {n_total} total ranker params:")
    for cg_key in sorted(_CG_TO_RANKER_KEY_MAP.keys()):
        if _CG_TO_RANKER_KEY_MAP[cg_key] in transfer:
            print(f"    ✓ {cg_key}")
    if missing_in_cg:
        print(f"  Missing from CG state_dict ({len(missing_in_cg)}): {missing_in_cg[:3]}{'...' if len(missing_in_cg) > 3 else ''}")
    if shape_mismatch:
        print(f"  Shape mismatch — NOT transferred ({len(shape_mismatch)}):")
        for cg_key, cg_shape, rnk_shape in shape_mismatch:
            print(f"    {cg_key}: CG{cg_shape} vs ranker{rnk_shape}")
    if result.unexpected_keys:
        print(f"  Unexpected keys in transfer dict (mapping bug): {result.unexpected_keys}")


def build_ranker(config: dict, fs) -> WideDeepRanker:
    n_movies, genome_buf, genre_buf, tag_buf, year_buf = _buffers_from_fs(fs)
    ranker = WideDeepRanker(
        n_movies=n_movies,
        n_genres=len(fs.genres_ordered),
        n_tags=len(fs.tags_ordered),
        n_genome_tags=len(fs.genome_tag_ids),
        n_years=len(fs.years_ordered),
        n_ts_bins=fs.timestamp_num_bins,
        user_context_size=fs.user_context_size,
        genome_buffer=genome_buf,
        genre_buffer=genre_buf,
        tag_buffer=tag_buf,
        year_buffer=year_buf,
        item_id_emb_dim=config['item_id_emb_dim'],
        item_genre_emb_dim=config['item_genre_emb_dim'],
        item_tag_emb_dim=config['item_tag_emb_dim'],
        item_genome_emb_dim=config['item_genome_emb_dim'],
        item_year_emb_dim=config['item_year_emb_dim'],
        user_genre_emb_dim=config['user_genre_emb_dim'],
        user_genome_ctx_emb_dim=config['user_genome_ctx_emb_dim'],
        ts_emb_dim=config['ts_emb_dim'],
        hidden_dims=config['hidden_dims'],
        dropout=config['dropout'],
        n_cross_features=config['n_cross_features'],
    )

    # Optional: warm start from CG (Option B — copy weights, then fine-tune; no runtime coupling).
    cg_ckpt = _resolve_cg_checkpoint(config.get('warm_start_cg_checkpoint'))
    if cg_ckpt:
        _warm_start_from_cg(ranker, cg_ckpt)
        config['warm_start_cg_checkpoint_resolved'] = cg_ckpt   # persist for sidecar

    return ranker


# ── Training ────────────────────────────────────────────────────────────────

def train(checkpoint_dir: str | None = None) -> str:
    config = get_config()
    if checkpoint_dir:
        config['checkpoint_dir'] = checkpoint_dir

    print("Loading datasets ...")
    train_ds, val_ds, fs = load_splits('data')

    device = get_device()
    print(f"\nDevice: {device}")
    train_ds.to(device)
    val_ds.to(device)

    # Deterministic eval sample for training-time logging. Fixed seed so the same val
    # rows are scored at every log step → NDCG curves are directly comparable over time.
    n_eval_samples = config.get('n_eval_samples')
    if n_eval_samples is None or n_eval_samples >= val_ds.N:
        eval_indices = None
        print(f"Train-time eval: full val set ({val_ds.N:,} rows)")
    else:
        eval_rng     = np.random.default_rng(config['seed'] + 7)
        eval_indices = eval_rng.choice(val_ds.N, size=n_eval_samples, replace=False).astype(np.int64)
        eval_indices.sort()                                    # ascending → faster sequential reads
        print(f"Train-time eval: deterministic {n_eval_samples:,}-row sample of {val_ds.N:,} val rows")

    cg_ndcg, cg_mrr = cg_baseline(val_ds, eval_indices=eval_indices)
    print(f"CG baseline (sample): NDCG@10={cg_ndcg:.4f}  MRR={cg_mrr:.4f}")
    print(f"Ranker target:        beat NDCG@10={cg_ndcg:.4f}\n")

    n_cross = config['n_cross_features']
    model = build_ranker(config, fs).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"WideDeepRanker:")
    print(f"  user_concat({model.user_concat_dim}) + item_concat({model.item_concat_dim}) "
          f"= deep_in({model.deep_in})")
    print(f"  → hidden={config['hidden_dims']} → head({config['hidden_dims'][-1]}+{n_cross}→1)")
    print(f"  ({n_params:,} trainable params)")

    # ── Menon α popularity bias ───────────────────────────────────────────────
    pop_alpha = float(config['popularity_alpha'])
    if pop_alpha > 0:
        if train_ds.movie_interaction_counts is None:
            raise RuntimeError("popularity_alpha > 0 but movie_interaction_counts_v2.npy is missing")
        counts_t       = torch.from_numpy(train_ds.movie_interaction_counts).to(device)
        popularity_bias = pop_alpha * torch.log1p(counts_t)
        print(f"Menon α: alpha={pop_alpha}  bias range=[{popularity_bias.min():.3f}, "
              f"{popularity_bias.max():.3f}]  mean={popularity_bias.mean():.3f}")
    else:
        popularity_bias = None
        print(f"Menon α: DISABLED (alpha=0)")

    optimizer = torch.optim.Adam(model.parameters(),
                                  lr=config['lr'],
                                  weight_decay=config['weight_decay'],
                                  eps=config['adam_eps'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['training_steps'], eta_min=config['lr'] * 0.01,
    )

    rng = np.random.default_rng(config['seed'])
    val_rng = np.random.default_rng(config['seed'] + 1)
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    run_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    alpha_tag = (str(pop_alpha).replace('.', '')
                 if pop_alpha != int(pop_alpha) else str(int(pop_alpha)))
    best_path = os.path.join(config['checkpoint_dir'],
                              f'ranker_mlp_alpha_{alpha_tag}_{run_ts}.pth')

    best_ndcg = -1.0
    loss_buf, grad_buf, pos_buf = [], [], []

    print(f"\nStarting training loop ({config['training_steps']:,} steps, "
          f"batch={config['batch_size']}, lr={config['lr']}) ...\n")
    start = time.time()

    from tqdm import tqdm
    pbar = tqdm(range(config['training_steps']), desc="Training")
    for step in pbar:
        model.train()
        ugc, xh, xhr, ts, cand_b, label_b, cross_b = sample_batch(
            train_ds, config['batch_size'], device, rng,
            easy_neg_frac=config['easy_neg_frac'])
        logits = model(ugc, xh, xhr, ts, cand_b, cross_b)

        if popularity_bias is not None:
            logits = logits + popularity_bias[cand_b]

        loss = F.binary_cross_entropy_with_logits(logits, label_b)

        optimizer.zero_grad()
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip']).item()
        optimizer.step()
        scheduler.step()

        loss_buf.append(loss.item())
        grad_buf.append(gn)
        pos_buf.append(label_b.sum().item() / label_b.numel())

        if step > 0 and step % config['log_every'] == 0:
            elapsed = time.time() - start
            start   = time.time()
            ndcg, mrr = evaluate_ndcg_mrr(model, val_ds, device, eval_indices=eval_indices)
            avg_loss = float(np.mean(loss_buf[-config['log_every']:]))
            avg_gn   = float(np.mean(grad_buf[-config['log_every']:]))
            avg_pos  = float(np.mean(pos_buf[-config['log_every']:]))
            cur_lr   = scheduler.get_last_lr()[0]

            # Val loss: average BCE over 8 val batches.
            model.eval()
            with torch.no_grad():
                vl_buf = []
                for _ in range(8):
                    vugc, vxh, vxhr, vts, vc, vl, vcross = sample_batch(
                        val_ds, config['batch_size'], device, val_rng,
                        easy_neg_frac=config['easy_neg_frac'])
                    vl_buf.append(F.binary_cross_entropy_with_logits(
                        model(vugc, vxh, vxhr, vts, vc, vcross), vl).item())
            val_loss = float(np.mean(vl_buf))

            improved = ndcg > best_ndcg
            tag      = " ← new best" if improved else ""
            print(f"[{step:06d}]  loss={avg_loss:.4f}  val_loss={val_loss:.4f}  "
                  f"grad={avg_gn:.3f}  pos_rate={avg_pos:.3f}  lr={cur_lr:.5f}  "
                  f"val NDCG@10={ndcg:.4f}  MRR={mrr:.4f}  "
                  f"(CG: {cg_ndcg:.4f}/{cg_mrr:.4f})  ({elapsed:.0f}s){tag}")
            pbar.set_postfix(loss=f"{avg_loss:.4f}", val_loss=f"{val_loss:.4f}",
                             val_ndcg=f"{ndcg:.4f}")
            if improved:
                best_ndcg = ndcg
                torch.save(model.state_dict(), best_path)
                _save_config(config, best_path)

    # Final eval on the same sampled set — last log point on the training curve.
    final_ndcg, final_mrr = evaluate_ndcg_mrr(model, val_ds, device, eval_indices=eval_indices)
    if final_ndcg > best_ndcg:
        best_ndcg = final_ndcg
        torch.save(model.state_dict(), best_path)
        _save_config(config, best_path)
        print(f"[final · sampled]  val NDCG@10={final_ndcg:.4f}  MRR={final_mrr:.4f}  ← new best")

    print(f"\nTraining complete.")
    print(f"  Best sampled val NDCG@10: {best_ndcg:.4f}  (CG sampled: {cg_ndcg:.4f}, "
          f"delta: {best_ndcg - cg_ndcg:+.4f})")
    print(f"  Checkpoint:               {best_path}")
    print(f"  Run `python ranker/main.py evaluate` for full-val metrics on the saved checkpoint.")
    return best_path


# ── Eval-only ───────────────────────────────────────────────────────────────

def evaluate_only(checkpoint_path: str | None = None):
    import glob as _glob
    import io as _io

    if checkpoint_path is None:
        matches = _glob.glob('saved_models/ranker/ranker_mlp_*.pth')
        if not matches:
            raise FileNotFoundError("No ranker checkpoint found in saved_models/ranker/")
        checkpoint_path = max(matches, key=os.path.getmtime)

    print(f"Loading datasets ...")
    train_ds, val_ds, fs = load_splits('data')
    device = get_device()
    print(f"Device: {device}")
    val_ds.to(device)

    config = get_config()
    cfg_path = _config_path(checkpoint_path)
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            saved = json.load(f)
        for k in ('hidden_dims', 'dropout', 'item_id_emb_dim', 'item_genre_emb_dim',
                  'item_tag_emb_dim', 'item_genome_emb_dim', 'item_year_emb_dim',
                  'user_genre_emb_dim', 'user_genome_ctx_emb_dim', 'ts_emb_dim',
                  'n_cross_features'):
            if k in saved:
                config[k] = saved[k]

    model = build_ranker(config, fs).to(device)
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True, map_location=device))
    print(f"Loaded checkpoint: {checkpoint_path}")

    n_cands = 1 + val_ds.n_neg

    raw_cg_ranks = val_ds.cg_label_rank
    cg_ranks     = np.where(raw_cg_ranks < n_cands, raw_cg_ranks, n_cands + 1)
    ranker_ranks = compute_label_ranks(model, val_ds, device)

    cg_ndcg, cg_mrr = _ndcg_mrr_from_ranks(cg_ranks)
    rk_ndcg, rk_mrr = _ndcg_mrr_from_ranks(ranker_ranks)
    cg_hits = hit_rates_from_ranks(cg_ranks)
    rk_hits = hit_rates_from_ranks(ranker_ranks)
    cg_recall_at_pool = float((raw_cg_ranks < n_cands).mean())

    buf = _io.StringIO()

    def emit(line: str = ''):
        print(line)
        buf.write(line + '\n')

    emit(f"Checkpoint: {checkpoint_path}")
    emit(f"Val rows: {len(raw_cg_ranks):,}  |  Pool size: {n_cands}")
    emit('')
    emit("══ Production-realistic metrics (E2E ceiling applied to both) ═════════")
    emit(f"{'Metric':<14} {'CG':>10} {'Ranker':>10} {'Delta':>12}")
    emit("─" * 50)
    emit(f"{'NDCG@10':<14} {cg_ndcg:>10.4f} {rk_ndcg:>10.4f} {rk_ndcg - cg_ndcg:>+12.4f}")
    emit(f"{'MRR':<14} {cg_mrr:>10.4f} {rk_mrr:>10.4f} {rk_mrr - cg_mrr:>+12.4f}")
    for k_name in cg_hits:
        cg_v, rk_v = cg_hits[k_name], rk_hits[k_name]
        emit(f"{k_name:<14} {cg_v:>10.4f} {rk_v:>10.4f} {rk_v - cg_v:>+12.4f}")
    emit("─" * 50)
    emit(f"{'Recall@'+str(n_cands):<14} {cg_recall_at_pool:>10.4f} {'(ceiling)':>10}   "
         f"← max ranker Hit@K in production")

    # ── Conditional metrics: pure reranking quality (no E2E ceiling) ──────────
    # Restrict to examples where CG retrieved the label. This isolates "given the
    # ranker gets a chance, how well does it rerank?" — the E2E ceiling drops out
    # since both CG and ranker see only labels in the 250-pool here.
    retrieved_mask = raw_cg_ranks < n_cands
    n_retrieved    = int(retrieved_mask.sum())

    if n_retrieved > 0:
        cg_ranks_sub     = raw_cg_ranks[retrieved_mask]            # all in [1, 249]
        ranker_ranks_sub = ranker_ranks[retrieved_mask]            # all in [1, 250] by construction
        cg_ndcg_sub, cg_mrr_sub  = _ndcg_mrr_from_ranks(cg_ranks_sub)
        rk_ndcg_sub, rk_mrr_sub  = _ndcg_mrr_from_ranks(ranker_ranks_sub)
        cg_hits_sub  = hit_rates_from_ranks(cg_ranks_sub)
        rk_hits_sub  = hit_rates_from_ranks(ranker_ranks_sub)

        emit('')
        emit(f"══ Pure reranking quality (CG-retrieved subset, n={n_retrieved:,}) ═════")
        emit(f"{'Metric':<14} {'CG':>10} {'Ranker':>10} {'Delta':>12}")
        emit("─" * 50)
        emit(f"{'NDCG@10':<14} {cg_ndcg_sub:>10.4f} {rk_ndcg_sub:>10.4f} {rk_ndcg_sub - cg_ndcg_sub:>+12.4f}")
        emit(f"{'MRR':<14} {cg_mrr_sub:>10.4f} {rk_mrr_sub:>10.4f} {rk_mrr_sub - cg_mrr_sub:>+12.4f}")
        for k_name in cg_hits_sub:
            emit(f"{k_name:<14} {cg_hits_sub[k_name]:>10.4f} {rk_hits_sub[k_name]:>10.4f} "
                 f"{rk_hits_sub[k_name] - cg_hits_sub[k_name]:>+12.4f}")
        emit("─" * 50)
        emit("(E2E ceiling not applied — both CG and ranker scored on labels CG retrieved)")

    base = os.path.splitext(os.path.basename(checkpoint_path))[0]
    out_dir = 'ranker/eval_results'
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'{base}.txt')
    with open(out_path, 'w') as f:
        f.write(buf.getvalue())
    print(f"\nWrote eval results → {out_path}")
