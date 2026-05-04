"""
Stage 1 — MLP ranker training loop.

BCE loss on random (row, candidate_position) tuples sampled across the entire
training set. Eval metric is NDCG@10 on val rollback groups (NOT val loss).

Menon α logit adjustment (unified with CG):
  During training, add α · log1p(count_i) to each logit before BCE.
  At inference, use raw logits — popular items' representations have shrunk
  during training to compensate, so f(u, i) reflects user/content match rather
  than popularity. Same conceptual fix CG uses, just adapted from softmax to BCE.

  Why ranker α may be higher than CG α (Menon et al. 2021 + production practice):
    CG's job is RECALL — make sure tail items are in the candidate pool.
    Ranker's job is PRECISION — sort within the pool. Rankers are far more
    prone to memorizing popularity (BCE positives correlate strongly with
    popularity). A more aggressive α is often required (1.0 vs CG's 0.5).

ZERO src/ imports. get_device() is inlined locally rather than imported from
src/train.py to keep this file pure.
"""
import json
import os
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

from ranker.dataset import load_splits, sample_batch
from ranker.evaluate import (cg_baseline, compute_label_ranks,
                              evaluate_ndcg_mrr, hit_rates_from_ranks,
                              _ndcg_mrr_from_ranks)
from ranker.model import MLPRanker


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
        'batch_size':         1024,
        'training_steps':     50_000,
        'log_every':          5_000,
        'grad_clip':          1.0,
        'hidden_dims':        [256, 128, 64],
        'dropout':            0.0,
        'seed':               42,
        'checkpoint_dir':     'saved_models/ranker',
        # Menon α logit adjustment — added to BCE logits during training; inference uses raw logits.
        # Default 1.0 (more aggressive than CG's 0.5) per Menon et al. 2021 / production practice:
        # rankers are precision-critical and more prone to popularity memorization.
        'popularity_alpha':   0.0,
    }


def _config_path(checkpoint_path: str) -> str:
    return os.path.splitext(checkpoint_path)[0] + '_config.json'


def _save_config(config: dict, checkpoint_path: str) -> None:
    with open(_config_path(checkpoint_path), 'w') as f:
        json.dump(config, f, indent=2)


def train(checkpoint_dir: str | None = None) -> str:
    config = get_config()
    if checkpoint_dir:
        config['checkpoint_dir'] = checkpoint_dir

    print("Loading datasets ...")
    train_ds, val_ds = load_splits('data')

    device = get_device()
    print(f"\nDevice: {device}")
    train_ds.to(device)
    val_ds.to(device)

    cg_ndcg, cg_mrr = cg_baseline(val_ds)
    print(f"\nCG baseline (val): NDCG@10={cg_ndcg:.4f}  MRR={cg_mrr:.4f}")
    print(f"Ranker target:     beat NDCG@10={cg_ndcg:.4f}\n")

    model = MLPRanker(train_ds.user_dim, train_ds.item_dim,
                      hidden_dims=config['hidden_dims'],
                      dropout=config['dropout']).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"MLPRanker: input={train_ds.user_dim + train_ds.item_dim} → "
          f"hidden={config['hidden_dims']} → 1   ({n_params:,} params)")

    # ── Menon α popularity bias: α · log1p(count_i) for each corpus index.
    # Added to logits during training (BCE); raw logits used at inference.
    pop_alpha = float(config['popularity_alpha'])
    if pop_alpha > 0:
        if train_ds.movie_interaction_counts is None:
            raise RuntimeError("popularity_alpha > 0 but movie_interaction_counts_v2.npy is missing")
        counts_t       = torch.from_numpy(train_ds.movie_interaction_counts).to(device)
        popularity_bias = pop_alpha * torch.log1p(counts_t)             # (n_movies,)
        print(f"Menon α: alpha={pop_alpha}  bias range=[{popularity_bias.min():.3f}, "
              f"{popularity_bias.max():.3f}]  mean={popularity_bias.mean():.3f}")
    else:
        popularity_bias = None
        print(f"Menon α: DISABLED (alpha=0)")

    optimizer = torch.optim.Adam(model.parameters(),
                                  lr=config['lr'],
                                  weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['training_steps'], eta_min=config['lr'] * 0.01,
    )

    rng = np.random.default_rng(config['seed'])
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    run_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    alpha_tag = (str(pop_alpha).replace('.', '')
                 if pop_alpha != int(pop_alpha) else str(int(pop_alpha)))
    best_path = os.path.join(config['checkpoint_dir'],
                              f'ranker_mlp_alpha_{alpha_tag}_{run_ts}.pth')

    best_ndcg = -1.0
    loss_buf  = []
    grad_buf  = []
    pos_buf   = []

    print(f"\nStarting training loop ({config['training_steps']:,} steps, "
          f"batch={config['batch_size']}, lr={config['lr']}) ...\n")
    start = time.time()

    from tqdm import tqdm
    pbar = tqdm(range(config['training_steps']), desc="Training")
    for step in pbar:
        model.train()
        user_b, item_b, cand_b, label_b = sample_batch(train_ds, config['batch_size'], device, rng)
        logits = model(user_b, item_b)

        # Menon α: add α·log1p(count_i) to logits BEFORE BCE so popular items get a
        # free upward shift during training. Effect (per Menon et al. 2021 / DCN folklore):
        #   - positive popular items: σ(adjusted) ≈ 1, loss ≈ 0 → "lazy positive", model
        #     receives little gradient signal from them
        #   - negative popular items: σ(adjusted) ≈ 1 but target=0 → loss is large, model
        #     pushes weights down aggressively
        # Net: f(u, i) is forced to learn user/content match, since popularity is "absorbed"
        # by the offset. At inference, we don't add the bias — the learned f is debiased.
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
            ndcg, mrr = evaluate_ndcg_mrr(model, val_ds, device)
            avg_loss = float(np.mean(loss_buf[-config['log_every']:]))
            avg_gn   = float(np.mean(grad_buf[-config['log_every']:]))
            avg_pos  = float(np.mean(pos_buf[-config['log_every']:]))
            cur_lr   = scheduler.get_last_lr()[0]
            improved = ndcg > best_ndcg
            tag      = " ← new best" if improved else ""
            print(f"[{step:06d}]  loss={avg_loss:.4f}  grad={avg_gn:.3f}  pos_rate={avg_pos:.3f}  "
                  f"lr={cur_lr:.5f}  val NDCG@10={ndcg:.4f}  MRR={mrr:.4f}  "
                  f"(CG: {cg_ndcg:.4f}/{cg_mrr:.4f})  ({elapsed:.0f}s){tag}")
            pbar.set_postfix(loss=f"{avg_loss:.4f}", val_ndcg=f"{ndcg:.4f}")
            if improved:
                best_ndcg = ndcg
                torch.save(model.state_dict(), best_path)
                _save_config(config, best_path)

    # Final eval
    ndcg, mrr = evaluate_ndcg_mrr(model, val_ds, device)
    if ndcg > best_ndcg:
        best_ndcg = ndcg
        torch.save(model.state_dict(), best_path)
        _save_config(config, best_path)
        print(f"[final]  val NDCG@10={ndcg:.4f}  MRR={mrr:.4f}  ← new best")

    print(f"\nTraining complete.")
    print(f"  Best val NDCG@10: {best_ndcg:.4f}  (CG baseline: {cg_ndcg:.4f}, "
          f"delta: {best_ndcg - cg_ndcg:+.4f})")
    print(f"  Checkpoint:       {best_path}")
    return best_path


def evaluate_only(checkpoint_path: str):
    import io as _io

    print(f"Loading datasets ...")
    train_ds, val_ds = load_splits('data')
    device = get_device()
    print(f"Device: {device}")
    val_ds.to(device)

    config = get_config()
    model = MLPRanker(val_ds.user_dim, val_ds.item_dim,
                       hidden_dims=config['hidden_dims'],
                       dropout=config['dropout']).to(device)
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True, map_location=device))
    print(f"Loaded checkpoint: {checkpoint_path}")

    cg_ranks     = val_ds.cg_label_rank
    ranker_ranks = compute_label_ranks(model, val_ds, device)

    cg_ndcg, cg_mrr   = _ndcg_mrr_from_ranks(cg_ranks)
    rk_ndcg, rk_mrr   = _ndcg_mrr_from_ranks(ranker_ranks)
    cg_hits           = hit_rates_from_ranks(cg_ranks)
    rk_hits           = hit_rates_from_ranks(ranker_ranks)

    buf = _io.StringIO()

    def emit(line: str = ''):
        print(line)
        buf.write(line + '\n')

    emit(f"Checkpoint: {checkpoint_path}")
    emit(f"\n{'Metric':<12} {'CG':>10} {'Ranker':>10} {'Delta':>12}")
    emit("─" * 48)
    emit(f"{'NDCG@10':<12} {cg_ndcg:>10.4f} {rk_ndcg:>10.4f} {rk_ndcg - cg_ndcg:>+12.4f}")
    emit(f"{'MRR':<12} {cg_mrr:>10.4f} {rk_mrr:>10.4f} {rk_mrr - cg_mrr:>+12.4f}")
    for k_name in cg_hits:
        cg_v, rk_v = cg_hits[k_name], rk_hits[k_name]
        emit(f"{k_name:<12} {cg_v:>10.4f} {rk_v:>10.4f} {rk_v - cg_v:>+12.4f}")

    base = os.path.splitext(os.path.basename(checkpoint_path))[0]
    out_dir = 'ranker/eval_results'
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'{base}.txt')
    with open(out_path, 'w') as f:
        f.write(buf.getvalue())
    print(f"\nWrote eval results → {out_path}")
