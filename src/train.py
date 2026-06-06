"""
Full softmax training loop for the Two-Tower MovieRecommender.

Usage:
    python main.py train                              # Model A (genome content slot)
    CONTENT_SOURCE=none  python main.py train         # Model C (no content slot, floor baseline)
    CONTENT_SOURCE=llm   python main.py train         # Model B (LLM content slot)

CONTENT_SOURCE selects the swappable content slot at train time (parallel to the
CORPUS env var); it tags the checkpoint name so the ablation's A/B/C variants are
unambiguous on disk. Default 'genome' preserves the historical prod naming.
"""
import json
import os
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from src.corpus import CORPUS, corpus_suffix
from src.dataset import FeatureStore
from src.model import MovieRecommender


# ── Hyperparameters ───────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def get_config() -> dict:
    # Swappable content slot, selectable at train time via the CONTENT_SOURCE env var
    # (parallel to CORPUS): 'genome' (default, Model A) | 'llm' (Model B) | 'none' (Model C floor).
    content_source = os.environ.get('CONTENT_SOURCE', 'genome')
    if content_source == 'none':
        content_source = None
    elif content_source not in ('genome', 'llm'):
        raise ValueError(f"Unknown CONTENT_SOURCE={content_source!r}; "
                         f"expected 'genome', 'llm', or 'none'")
    return {
        # Sub-embedding sizes
        'item_movieId_embedding_size':      32,
        'item_year_embedding_size':         8,
        'timestamp_feature_embedding_size': 4,
        'item_tag_embedding_size':          16,
        'item_content_embedding_size':      32,
        'user_genre_embedding_size':        32,
        'item_genre_embedding_size':        8,
        'user_content_embedding_size':      32,
        # Swappable content slot: 'genome' (prod) | 'llm' | None (ablation Model C).
        'content_feature_source':           content_source,
        # Projection MLP
        'proj_hidden':  256,
        'output_dim':   128,
        # Training
        'lr':               0.001,
        'weight_decay':     0.0,
        'adam_eps':         1e-6,
        'minibatch_size':   512,
        'training_steps':   150_000,
        'log_every':        5_000,
        'checkpoint_every': 30_000,
        'checkpoint_dir':   'saved_models',
        'temperature':      0.1,
        'popularity_alpha': 0.0,
    }


# ── Model factory ─────────────────────────────────────────────────────────────

def build_model(config: dict, fs: FeatureStore) -> MovieRecommender:
    content_feature_source = config.get('content_feature_source', 'genome')

    # Content slot: 'genome' | 'llm' | None (Model C floor).
    # Genome: build from the FeatureStore's pre-loaded per-movie dicts (1128-dim).
    # LLM:    load the pre-built tensor from data/llm_features_<tag>_v1<sfx>.pt (132-dim).
    #         LLM_MODEL_TAG selects which extraction run to use (default 'claude-code-sonnet').
    # None:   no content slot — content towers and buffer are omitted entirely.
    content_context_buffer = None
    content_feature_len    = None   # set below; passed as genome_tags_len to MovieRecommender
    if content_feature_source == 'genome':
        content_matrix = np.array(
            [fs.movieId_to_genome_tag_context[mid] for mid in fs.top_movies],
            dtype=np.float32,
        )
        pad_row = np.zeros((1, content_matrix.shape[1]), dtype=np.float32)
        content_context_buffer = torch.from_numpy(np.vstack([content_matrix, pad_row]))
        content_feature_len = content_matrix.shape[1]   # 1128
    elif content_feature_source == 'llm':
        sfx = corpus_suffix()
        llm_tag = os.environ.get('LLM_MODEL_TAG', 'claude-code-sonnet')
        llm_path = os.path.join('data', f'llm_features_{llm_tag}_v1{sfx}.pt')
        if not os.path.exists(llm_path):
            raise FileNotFoundError(
                f"LLM feature tensor not found: {llm_path}\n"
                f"  Run: CORPUS={CORPUS} python -m llm_features.build_features {llm_tag}"
            )
        content_context_buffer = torch.load(llm_path, map_location='cpu', weights_only=True)
        content_feature_len = content_context_buffer.shape[1]   # 132
        print(f"  LLM features: {llm_path}  ({content_feature_len}-dim, "
              f"{content_context_buffer.shape[0]-1} movies)")
    else:
        content_feature_len = len(fs.genome_tag_ids)   # unused (no towers), but keep valid

    genre_matrix = np.array(
        [fs.movieId_to_genre_context[mid] for mid in fs.top_movies], dtype=np.float32)
    tag_matrix = np.array(
        [fs.movieId_to_tag_context[mid] for mid in fs.top_movies], dtype=np.float32)
    year_array = np.array(
        [fs.year_to_i[fs.movieId_to_year[mid]] for mid in fs.top_movies], dtype=np.int64)
    genre_context_buffer = torch.from_numpy(
        np.vstack([genre_matrix, np.zeros((1, genre_matrix.shape[1]), dtype=np.float32)]))
    tag_context_buffer = torch.from_numpy(
        np.vstack([tag_matrix, np.zeros((1, tag_matrix.shape[1]), dtype=np.float32)]))
    year_context_buffer = torch.from_numpy(
        np.concatenate([year_array, np.zeros((1,), dtype=np.int64)]))

    return MovieRecommender(
        genres_len=len(fs.genres_ordered),
        tags_len=len(fs.tags_ordered),
        genome_tags_len=content_feature_len,
        top_movies_len=len(fs.top_movies),
        all_years_len=len(fs.years_ordered),
        timestamp_num_bins=fs.timestamp_num_bins,
        user_context_size=fs.user_context_size,
        content_context_buffer=content_context_buffer,
        content_feature_source=content_feature_source,
        item_genre_embedding_size=config['item_genre_embedding_size'],
        item_tag_embedding_size=config['item_tag_embedding_size'],
        item_content_embedding_size=config['item_content_embedding_size'],
        item_movieId_embedding_size=config['item_movieId_embedding_size'],
        item_year_embedding_size=config['item_year_embedding_size'],
        user_genre_embedding_size=config['user_genre_embedding_size'],
        timestamp_feature_embedding_size=config['timestamp_feature_embedding_size'],
        user_content_embedding_size=config.get('user_content_embedding_size', 32),
        genre_context_buffer=genre_context_buffer,
        tag_context_buffer=tag_context_buffer,
        year_context_buffer=year_context_buffer,
        proj_hidden=config.get('proj_hidden', 256),
        output_dim=config.get('output_dim', 128),
    )


def print_model_summary(model: MovieRecommender) -> None:
    m = model
    def _out_dim(tower):
        for layer in reversed(tower):
            if isinstance(layer, torch.nn.Linear):
                return layer.out_features

    # Content slot (genome today, LLM later) is omitted entirely when source is None (Model C).
    has_content      = m.content_feature_source is not None
    id_dim           = m.item_embedding_lookup.embedding_dim
    content_dim      = _out_dim(m.item_content_tower) if has_content else 0
    cctx_dim         = _out_dim(m.user_content_tower) if has_content else 0
    genre_dim        = _out_dim(m.user_genre_tower)
    ts_dim           = m.timestamp_embedding_lookup.embedding_dim
    item_genre_dim   = _out_dim(m.item_genre_tower)
    item_tag_dim     = _out_dim(m.item_tag_tower)
    item_movieId_dim = _out_dim(m.item_embedding_tower)
    year_dim         = _out_dim(m.year_embedding_tower)
    item_total       = item_genre_dim + item_tag_dim + content_dim + item_movieId_dim + year_dim
    n_params         = sum(p.nelement() for p in model.parameters() if p.requires_grad)

    pool_total = 4 * id_dim
    user_total = pool_total + cctx_dim + genre_dim + ts_dim
    content_user_desc = f"content_ctx({cctx_dim}) + " if has_content else ""
    content_item_desc = f"content({content_dim}) + " if has_content else ""
    user_desc  = (f"4×sum_pool_id({id_dim}) + {content_user_desc}"
                  f"genre({genre_dim}) + ts({ts_dim})")

    proj_h  = m.user_projection[0].out_features
    out_dim = m.user_projection[2].out_features

    print(f"\n── Model dimensions ──")
    print(f"  Content slot: {m.content_feature_source}")
    print(f"  User side:  {user_desc}  =  {user_total}")
    print(f"  Item side:  genre({item_genre_dim}) + tag({item_tag_dim}) + {content_item_desc}"
          f"movieId({item_movieId_dim}) + year({year_dim})  =  {item_total}")
    print(f"  Projection: Linear({proj_h}) → ReLU → Linear({out_dim})  [both towers]")
    print(f"  Parameters: {n_params:,}")


# ── Config sidecar ────────────────────────────────────────────────────────────

def _config_path(checkpoint_path: str) -> str:
    return os.path.splitext(checkpoint_path)[0] + '_config.json'

def _save_config(config: dict, checkpoint_path: str) -> None:
    with open(_config_path(checkpoint_path), 'w') as f:
        json.dump(config, f, indent=2)

def load_config_for_checkpoint(checkpoint_path: str) -> dict:
    path = _config_path(checkpoint_path)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    cfg = get_config()
    cfg['popularity_alpha'] = 0.0
    return cfg


def _val_ranking_metrics(raw_scores: torch.Tensor, targets: torch.Tensor) -> tuple:
    """Reciprocal-rank sum and Hit@10 count of each target vs all items, on the *raw* scores.

    Ranking is done on raw dot products (no temperature, no popularity adjustment) so the
    selection metric matches offline inference, which ranks on raw scores — the Menon popularity
    correction is training-only. The strict-greater tie convention mirrors src/offline_eval.py.
    Returns (sum_of_reciprocal_ranks, num_hits_at_10) for the batch; caller accumulates."""
    rows         = torch.arange(raw_scores.shape[0], device=raw_scores.device)
    target_score = raw_scores[rows, targets].unsqueeze(1)
    ranks        = (raw_scores > target_score).sum(dim=1) + 1
    return (1.0 / ranks.float()).sum().item(), int((ranks <= 10).sum().item())


# ── Training loop ─────────────────────────────────────────────────────────────

def train_softmax(model: MovieRecommender, train_data: tuple, val_data: tuple,
                  config: dict, fs: FeatureStore) -> str:
    """
    Full softmax training loop. Returns the path of the best checkpoint.

    train_data / val_data: 7-tuple from dataset.build_softmax_dataset()
      (X_genre, X_history, X_hist_liked, X_hist_disliked, X_hist_ratings, timestamp, target_movieId)

    Each step:
      V_all  = model.full_item_embedding()       (n_movies, dim)  — all corpus items
      U      = model.user_embedding(batch...)    (B, dim)
      scores = (U @ V_all.T) / temperature + alpha*log1p(count_i)   (B, n_movies)
      loss   = cross_entropy(scores, target_idx)
    Menon et al. (2021) logit adjustment: add alpha*log(count_i) to all logits.
    """
    device = get_device()
    print(f"Using device: {device}")
    model.to(device)

    (X_genre_train, X_history_train, X_hist_liked_train, X_hist_disliked_train,
     X_hist_ratings_train, timestamp_train, target_movieId_train) = train_data

    (X_genre_val, X_history_val, X_hist_liked_val, X_hist_disliked_val,
     X_hist_ratings_val, timestamp_val, target_movieId_val) = val_data

    X_genre_train        = X_genre_train.to(device)
    timestamp_train      = timestamp_train.to(device)
    target_movieId_train = target_movieId_train.to(device)
    X_genre_val          = X_genre_val.to(device)
    timestamp_val        = timestamp_val.to(device)
    target_movieId_val   = target_movieId_val.to(device)

    print_model_summary(model)

    n_train = X_genre_train.shape[0]
    n_val   = X_genre_val.shape[0]

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'],
                                 weight_decay=config['weight_decay'],
                                 eps=config['adam_eps'])
    training_steps   = config['training_steps']
    scheduler        = torch.optim.lr_scheduler.CosineAnnealingLR(
                           optimizer, T_max=training_steps, eta_min=1e-4)
    minibatch_size   = config['minibatch_size']
    temperature      = config['temperature']
    log_every        = config['log_every']
    checkpoint_every = config['checkpoint_every']
    checkpoint_dir   = config['checkpoint_dir']

    os.makedirs(checkpoint_dir, exist_ok=True)
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    alpha         = config['popularity_alpha']
    alpha_tag     = str(alpha).replace('.', '') if alpha != int(alpha) else str(int(alpha))
    corpus_sfx    = corpus_suffix()  # '' for the full corpus, '_<corpus>' otherwise
    # Content-slot tag: genome keeps the historical (prod) name; llm/nocontent are tagged so the
    # ablation's A/B/C checkpoints are unambiguous on disk (config is also in the JSON sidecar).
    content_src   = config['content_feature_source']
    content_tag   = '' if content_src == 'genome' else f"_{content_src or 'nocontent'}"
    print(f"  Corpus: {CORPUS}  (checkpoint suffix: {corpus_sfx!r})  "
          f"content slot: {content_src}  (name tag: {content_tag!r})")
    best_val_mrr  = float('-inf')   # best_path is selected on val-MRR (the reported metric), not val CE
    best_path     = os.path.join(checkpoint_dir, f'best_softmax_v2{content_tag}_popularity_alpha_{alpha_tag}{corpus_sfx}_{run_timestamp}.pth')

    val_eval_size = min(8_192, n_val)
    rng_val = torch.Generator()
    rng_val.manual_seed(0)
    val_eval_idx = torch.randperm(n_val, generator=rng_val)[:val_eval_size].tolist()

    pad_idx = len(fs.top_movies)
    counts_cpu = torch.bincount(target_movieId_train.cpu(), minlength=pad_idx).float()
    popularity_bias = (alpha * torch.log1p(counts_cpu)).to(device)
    print(f"  Popularity bias: alpha={alpha}  "
          f"max_adj={popularity_bias.max():.3f}  min_adj={popularity_bias.min():.3f}")

    loss_train = []
    grad_norms = []

    print(f"\nStarting softmax training ({training_steps:,} steps, "
          f"batch={minibatch_size}, temp={temperature}) ...")
    print(f"  Train: {n_train:,}  |  Val: {n_val:,}  (eval subset: {val_eval_size:,})")
    start = time.time()

    from tqdm import tqdm
    pbar = tqdm(range(training_steps), desc="Training")
    for i in pbar:
        is_val = (i % log_every == 0)

        if is_val:
            model.eval()
            with torch.no_grad():
                V_all = model.full_item_embedding()

                if i == 0:
                    vidx = torch.randint(0, n_val, (minibatch_size,))
                    U = model.user_embedding(
                        X_genre_val[vidx],
                        X_history_val[vidx].to(device),
                        X_hist_liked_val[vidx].to(device),
                        X_hist_disliked_val[vidx].to(device),
                        X_hist_ratings_val[vidx].to(device),
                        timestamp_val[vidx],
                    )
                    raw    = U @ V_all.T
                    scaled = raw / temperature
                    print(f"  [step 0 logits] raw dot  mean={raw.mean():.4f}  std={raw.std():.4f}  "
                          f"min={raw.min():.4f}  max={raw.max():.4f}")
                    print(f"  [step 0 logits] /temp={temperature}  mean={scaled.mean():.4f}  "
                          f"std={scaled.std():.4f}")
                    print(f"  [step 0 logits] random-baseline loss = {np.log(V_all.shape[0]):.4f}")

                # Val loss is the training objective (CE on temp-scaled, popularity-adjusted
                # logits); val MRR/Hit@10 are computed on the raw scores (matching inference) over
                # the same subset. best_path is selected on val-MRR — see _val_ranking_metrics.
                val_losses = []
                val_rr_sum = 0.0
                val_hits10 = 0
                val_n      = 0
                for vs in range(0, val_eval_size, minibatch_size):
                    ve   = min(vs + minibatch_size, val_eval_size)
                    vidx = val_eval_idx[vs:ve]
                    U = model.user_embedding(
                        X_genre_val[vidx],
                        X_history_val[vidx].to(device),
                        X_hist_liked_val[vidx].to(device),
                        X_hist_disliked_val[vidx].to(device),
                        X_hist_ratings_val[vidx].to(device),
                        timestamp_val[vidx],
                    )
                    raw_scores = U @ V_all.T
                    scores     = raw_scores / temperature + popularity_bias
                    targets    = target_movieId_val[vidx]
                    val_losses.append(F.cross_entropy(scores, targets).item())
                    rr, h10     = _val_ranking_metrics(raw_scores, targets)
                    val_rr_sum += rr
                    val_hits10 += h10
                    val_n      += int(targets.shape[0])
                val_loss  = float(np.mean(val_losses))
                val_mrr   = val_rr_sum / val_n
                val_hit10 = val_hits10 / val_n

            avg_train     = np.mean(loss_train[i - log_every:i]) if i >= log_every else (loss_train[-1] if loss_train else 0.0)
            avg_grad_norm = np.mean(grad_norms[i - log_every:i]) if i >= log_every else (grad_norms[-1] if grad_norms else 0.0)
            elapsed    = time.time() - start
            start      = time.time()
            current_lr = scheduler.get_last_lr()[0] if i > 0 else config['lr']
            pbar.set_postfix(train=f"{avg_train:.4f}", val=f"{val_loss:.4f}", mrr=f"{val_mrr:.4f}")
            print(f"[{i:06d}]  train={avg_train:.4f}  val={val_loss:.4f}  "
                  f"val_mrr={val_mrr:.4f}  hit@10={val_hit10:.4f}  "
                  f"lr={current_lr:.6f}  grad_norm={avg_grad_norm:.3f}  ({elapsed:.0f}s)")

            if val_mrr > best_val_mrr:
                best_val_mrr = val_mrr
                torch.save(model.state_dict(), best_path)
                _save_config(config, best_path)
                print(f"  → new best val_mrr={best_val_mrr:.4f} → {best_path}")

            if i > 0 and i % checkpoint_every == 0:
                periodic = os.path.join(checkpoint_dir,
                                        f'softmax_v2{content_tag}_popularity_alpha_{alpha_tag}{corpus_sfx}_{run_timestamp}_step_{i:06d}.pth')
                torch.save(model.state_dict(), periodic)
                _save_config(config, periodic)
                print(f"  → periodic checkpoint → {periodic}")

        else:
            model.train()
            V_all = model.full_item_embedding()
            ix    = torch.randint(0, n_train, (minibatch_size,))
            U = model.user_embedding(
                X_genre_train[ix],
                X_history_train[ix].to(device),
                X_hist_liked_train[ix].to(device),
                X_hist_disliked_train[ix].to(device),
                X_hist_ratings_train[ix].to(device),
                timestamp_train[ix],
            )
            logits = (U @ V_all.T) / temperature
            scores = logits + popularity_bias
            loss   = F.cross_entropy(scores, target_movieId_train[ix])
            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).item()
            optimizer.step()
            scheduler.step()
            loss_train.append(loss.item())
            grad_norms.append(grad_norm)

    # Final val pass
    model.eval()
    with torch.no_grad():
        V_all = model.full_item_embedding()
        val_losses = []
        val_rr_sum = 0.0
        val_hits10 = 0
        val_n      = 0
        for vs in range(0, val_eval_size, minibatch_size):
            ve   = min(vs + minibatch_size, val_eval_size)
            vidx = val_eval_idx[vs:ve]
            U = model.user_embedding(
                X_genre_val[vidx],
                X_history_val[vidx].to(device),
                X_hist_liked_val[vidx].to(device),
                X_hist_disliked_val[vidx].to(device),
                X_hist_ratings_val[vidx].to(device),
                timestamp_val[vidx],
            )
            raw_scores = U @ V_all.T
            scores     = raw_scores / temperature + popularity_bias
            targets    = target_movieId_val[vidx]
            val_losses.append(F.cross_entropy(scores, targets).item())
            rr, h10     = _val_ranking_metrics(raw_scores, targets)
            val_rr_sum += rr
            val_hits10 += h10
            val_n      += int(targets.shape[0])
    final_val_loss = float(np.mean(val_losses))
    final_val_mrr  = val_rr_sum / val_n
    final_val_hit10 = val_hits10 / val_n
    print(f"[{training_steps:06d}]  val={final_val_loss:.4f}  "
          f"val_mrr={final_val_mrr:.4f}  hit@10={final_val_hit10:.4f}  (final)")
    if final_val_mrr > best_val_mrr:
        best_val_mrr = final_val_mrr
        torch.save(model.state_dict(), best_path)
        _save_config(config, best_path)
        print(f"  → new best val_mrr={best_val_mrr:.4f} → {best_path}")

    print(f"\nTraining complete. Best val MRR: {best_val_mrr:.4f}")
    print(f"Best checkpoint: {best_path}")
    return best_path
