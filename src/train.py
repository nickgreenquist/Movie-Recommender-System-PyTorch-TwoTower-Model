"""
Full softmax training loop for the Two-Tower MovieRecommender.

Usage:
    python main.py train                                # Model A (genome-tag tower)
    FEATURE_TOWERS=llm   python main.py train           # Model B (LLM-feature tower)
    FEATURE_TOWERS=none  python main.py train           # Model C (no semantic-feature towers)
    FEATURE_TOWERS=both  python main.py train           # Model D (genome + LLM towers) — new prod

FEATURE_TOWERS selects which semantic-feature sub-towers the model includes at train time
(parallel to the CORPUS env var): 'genome' | 'llm' | 'both' | 'none'. It tags the checkpoint
name so the variants are unambiguous on disk. Default 'genome' preserves the historical prod
naming. The legacy CONTENT_SOURCE env var is still read as a fallback.

BASE_TOWERS selects the always-on base towers: 'all' (default — every model so far) | 'idonly'
(the stripped CF-base ablation: genre/tag/year/timestamp towers off AND the user tower collapsed
to the single full-history ID sum-pool — no liked/disliked/rating-weighted pools. The ONLY thing
left besides the ID embeddings is the semantic-feature slot). 'idonly' tags the checkpoint name:

    BASE_TOWERS=idonly FEATURE_TOWERS=none   python main.py train    # C′ pure CF floor
    BASE_TOWERS=idonly FEATURE_TOWERS=genome python main.py train    # A′ ID + genome
    BASE_TOWERS=idonly FEATURE_TOWERS=llm    python main.py train    # B′ ID + LLM

USER_POOLS / USER_FEATURES / ITEM_FEATURES are fine-grained overrides (comma-separated subsets;
'none' = empty) that select individual towers/pools independently of FEATURE_TOWERS/BASE_TOWERS.
Unset → the legacy set those coarse knobs imply (so existing runs are byte-identical). Used for the
multi-pool user-tower ablation (docs/plans/multipool_user_tower_ablation_plan.md); all arms below
keep the item tower ID-only and the user tower pooling-only:

    BASE_TOWERS=idonly FEATURE_TOWERS=none USER_POOLS=full                                python main.py train  # only history (= C′)
    BASE_TOWERS=idonly FEATURE_TOWERS=none USER_POOLS=full,liked                          python main.py train  # + likes
    BASE_TOWERS=idonly FEATURE_TOWERS=none USER_POOLS=weighted                            python main.py train  # rating-weighted pool only
    BASE_TOWERS=idonly FEATURE_TOWERS=none USER_POOLS=full,liked,disliked,weighted        python main.py train  # 4-pool
    BASE_TOWERS=idonly FEATURE_TOWERS=none USER_POOLS=full,liked,disliked,weighted,last_liked python main.py train  # 4-pool + last-liked
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
from src.model import (MovieRecommender, POOL_ORDER, USER_FEATURE_ORDER, ITEM_FEATURE_ORDER,
                       default_user_pools, default_user_features, default_item_features, _canon)


# ── Hyperparameters ───────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def _env_feature_set(name: str, order: tuple, default) -> list:
    """Parse a comma-separated tower/pool selector env var into a canonical-order list.

    Unset → the supplied legacy default (list). 'none' (case-insensitive) → empty list. Otherwise a
    comma list validated against `order` (unknown tokens raise, mirroring FEATURE_TOWERS/BASE_TOWERS)."""
    raw = os.environ.get(name)
    if raw is None:
        return list(default)
    toks = [t.strip().lower() for t in raw.split(',') if t.strip()]
    if toks == ['none']:
        return []
    return list(_canon(toks, order, name))


def get_config() -> dict:
    # Which semantic-feature sub-towers to include, selectable at train time via the
    # FEATURE_TOWERS env var (parallel to CORPUS): 'genome' (default, Model A) | 'llm' (Model B)
    # | 'both' (Model D) | 'none' (Model C floor). Legacy CONTENT_SOURCE is read as a fallback.
    feature_towers = os.environ.get('FEATURE_TOWERS', os.environ.get('CONTENT_SOURCE', 'genome'))
    if feature_towers == 'none':
        feature_towers = None
    elif feature_towers not in ('genome', 'llm', 'both'):
        raise ValueError(f"Unknown FEATURE_TOWERS={feature_towers!r}; "
                         f"expected 'genome', 'llm', 'both', or 'none'")
    # Which always-on base towers to include, via the BASE_TOWERS env var: 'all' (default) |
    # 'idonly' (stripped CF-base ablation — genre/tag/year/timestamp towers off, user tower
    # collapsed to the single full-history sum pool).
    base_towers = os.environ.get('BASE_TOWERS', 'all')
    if base_towers not in ('all', 'idonly'):
        raise ValueError(f"Unknown BASE_TOWERS={base_towers!r}; expected 'all' or 'idonly'")
    # Fine-grained per-tower/per-pool overrides (comma-separated subsets; 'none' = empty). Each
    # defaults to the legacy set implied by feature_towers + base_towers, so an unset knob is a
    # byte-identical model. Validated + canonicalized here so a typo'd env fails fast before training.
    user_pools    = _env_feature_set('USER_POOLS',    POOL_ORDER,         default_user_pools(base_towers))
    user_features = _env_feature_set('USER_FEATURES', USER_FEATURE_ORDER, default_user_features(feature_towers, base_towers))
    item_features = _env_feature_set('ITEM_FEATURES', ITEM_FEATURE_ORDER, default_item_features(feature_towers, base_towers))
    if not user_pools:
        raise ValueError("USER_POOLS must select at least one pool")
    return {
        # Sub-embedding sizes
        'item_movieId_embedding_size':      32,
        'item_year_embedding_size':         8,
        'timestamp_feature_embedding_size': 4,
        'item_tag_embedding_size':          16,
        'item_genome_embedding_size':       32,
        'item_llm_embedding_size':          32,
        'user_genre_embedding_size':        32,
        'item_genre_embedding_size':        8,
        'user_genome_embedding_size':       32,
        'user_llm_embedding_size':          32,
        # Which semantic-feature towers: 'genome' | 'llm' | 'both' | None (Model C).
        'feature_towers':                   feature_towers,
        # Which base towers: 'all' | 'idonly' (stripped CF-base ablation).
        'base_towers':                      base_towers,
        # Fine-grained per-side selectors (canonical-order lists); override feature_towers/base_towers.
        'user_pools':                       user_pools,
        'user_features':                    user_features,
        'item_features':                    item_features,
        # Projection MLP
        'proj_hidden':  256,
        'output_dim':   128,
        # Training
        'lr':               0.001,
        'weight_decay':     0.0,
        'adam_eps':         1e-6,
        'minibatch_size':   512,
        'training_steps':   200_000,
        'log_every':        10_000,
        'checkpoint_every': 30_000,
        'checkpoint_dir':   'saved_models',
        'temperature':      0.1,
        'popularity_alpha': 0.0,
        # Recorded so every checkpoint sidecar states its seed (SEED env overrides; default 42).
        # main.py seeds the RNGs from this value, so the seed used == the seed saved.
        'seed':             int(os.environ.get('SEED', 42)),
    }


# ── Model factory ─────────────────────────────────────────────────────────────

def _genome_buffer(fs: FeatureStore):
    """Per-movie genome scores → (top_movies_len+1, 1128) buffer with a zero padding row."""
    m = np.array([fs.movieId_to_genome_tag_context[mid] for mid in fs.top_movies], dtype=np.float32)
    pad = np.zeros((1, m.shape[1]), dtype=np.float32)
    return torch.from_numpy(np.vstack([m, pad])), m.shape[1]


def _llm_buffer():
    """Load the pre-built LLM feature tensor (already padded). LLM_MODEL_TAG selects the run."""
    sfx     = corpus_suffix()
    llm_tag = os.environ.get('LLM_MODEL_TAG', 'claude-code-sonnet')
    llm_path = os.path.join('data', f'llm_features_{llm_tag}_v1{sfx}.pt')
    if not os.path.exists(llm_path):
        raise FileNotFoundError(
            f"LLM feature tensor not found: {llm_path}\n"
            f"  Run: CORPUS={CORPUS} python -m llm_features.build_features {llm_tag}"
        )
    buf = torch.load(llm_path, map_location='cpu', weights_only=True)
    return buf, buf.shape[1], llm_path


def build_model(config: dict, fs: FeatureStore) -> MovieRecommender:
    feature_towers = config.get('feature_towers', config.get('content_feature_source', 'genome'))
    base_towers = config.get('base_towers', 'all')

    # Effective per-side selectors. config carries them for new runs and resolved checkpoints; fall
    # back to the legacy feature_towers/base_towers default for old configs and direct callers.
    user_pools    = config.get('user_pools')    or list(default_user_pools(base_towers))
    user_features = config.get('user_features')
    item_features = config.get('item_features')
    uf = set(user_features if user_features is not None else default_user_features(feature_towers, base_towers))
    itf = set(item_features if item_features is not None else default_item_features(feature_towers, base_towers))

    # A context buffer is built if EITHER side needs it. genome/llm buffers are shared by the item
    # and user towers; genre/tag/year buffers feed only the item towers.
    need_genome = ('genome' in uf) or ('genome' in itf)
    need_llm    = ('llm' in uf)    or ('llm' in itf)
    need_genre  = 'genre' in itf
    need_tag    = 'tag'   in itf
    need_year   = 'year'  in itf

    # Semantic-feature buffers — genome from the FeatureStore (1128-dim), LLM from the pre-built
    # tensor (132-dim). Built only when a side uses them.
    genome_context_buffer = None
    genome_tags_len       = len(fs.genome_tag_ids)   # always valid; used as in_features if a genome tower is on
    llm_feature_buffer    = None
    llm_feature_len       = None
    if need_genome:
        genome_context_buffer, genome_tags_len = _genome_buffer(fs)
    if need_llm:
        llm_feature_buffer, llm_feature_len, llm_path = _llm_buffer()
        print(f"  LLM features: {llm_path}  ({llm_feature_len}-dim, "
              f"{llm_feature_buffer.shape[0]-1} movies)")

    # Item-side base buffers — each built independently (genre/tag/year may now be toggled apart).
    genre_context_buffer = None
    tag_context_buffer   = None
    year_context_buffer  = None
    if need_genre:
        genre_matrix = np.array(
            [fs.movieId_to_genre_context[mid] for mid in fs.top_movies], dtype=np.float32)
        genre_context_buffer = torch.from_numpy(
            np.vstack([genre_matrix, np.zeros((1, genre_matrix.shape[1]), dtype=np.float32)]))
    if need_tag:
        tag_matrix = np.array(
            [fs.movieId_to_tag_context[mid] for mid in fs.top_movies], dtype=np.float32)
        tag_context_buffer = torch.from_numpy(
            np.vstack([tag_matrix, np.zeros((1, tag_matrix.shape[1]), dtype=np.float32)]))
    if need_year:
        year_array = np.array(
            [fs.year_to_i[fs.movieId_to_year[mid]] for mid in fs.top_movies], dtype=np.int64)
        year_context_buffer = torch.from_numpy(
            np.concatenate([year_array, np.zeros((1,), dtype=np.int64)]))

    return MovieRecommender(
        genres_len=len(fs.genres_ordered),
        tags_len=len(fs.tags_ordered),
        genome_tags_len=genome_tags_len,
        top_movies_len=len(fs.top_movies),
        all_years_len=len(fs.years_ordered),
        timestamp_num_bins=fs.timestamp_num_bins,
        user_context_size=fs.user_context_size,
        feature_towers=feature_towers,
        base_towers=base_towers,
        user_pools=user_pools,
        user_features=user_features,
        item_features=item_features,
        genome_context_buffer=genome_context_buffer,
        llm_feature_buffer=llm_feature_buffer,
        llm_feature_len=llm_feature_len,
        item_genre_embedding_size=config['item_genre_embedding_size'],
        item_tag_embedding_size=config['item_tag_embedding_size'],
        item_genome_embedding_size=config.get('item_genome_embedding_size', 32),
        item_llm_embedding_size=config.get('item_llm_embedding_size', 32),
        item_movieId_embedding_size=config['item_movieId_embedding_size'],
        item_year_embedding_size=config['item_year_embedding_size'],
        user_genre_embedding_size=config['user_genre_embedding_size'],
        timestamp_feature_embedding_size=config['timestamp_feature_embedding_size'],
        user_genome_embedding_size=config.get('user_genome_embedding_size', 32),
        user_llm_embedding_size=config.get('user_llm_embedding_size', 32),
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

    # Gated towers are present only when switched on (now independent per side).
    id_dim            = m.item_embedding_lookup.embedding_dim
    genome_item_dim   = _out_dim(m.item_genome_tag_tower)     if m.has_item_genome else 0
    genome_user_dim   = _out_dim(m.user_genome_context_tower) if m.has_user_genome else 0
    llm_item_dim      = _out_dim(m.item_llm_feature_tower)    if m.has_item_llm    else 0
    llm_user_dim      = _out_dim(m.user_llm_feature_tower)    if m.has_user_llm    else 0
    genre_dim         = _out_dim(m.user_genre_tower)          if m.has_user_genre  else 0
    ts_dim            = m.timestamp_embedding_lookup.embedding_dim if m.has_timestamp else 0
    item_genre_dim    = _out_dim(m.item_genre_tower)          if m.has_item_genre  else 0
    item_tag_dim      = _out_dim(m.item_tag_tower)            if m.has_item_tag    else 0
    item_movieId_dim  = _out_dim(m.item_embedding_tower)
    year_dim          = _out_dim(m.year_embedding_tower)      if m.has_item_year   else 0
    item_total        = (item_genre_dim + item_tag_dim + genome_item_dim + llm_item_dim
                         + item_movieId_dim + year_dim)
    n_params          = sum(p.nelement() for p in model.parameters() if p.requires_grad)

    n_pools    = len(m.user_pools)
    pool_total = n_pools * id_dim
    user_total = pool_total + genome_user_dim + llm_user_dim + genre_dim + ts_dim
    genome_item_desc = f"genome({genome_item_dim}) + "     if m.has_item_genome else ""
    llm_item_desc    = f"llm({llm_item_dim}) + "           if m.has_item_llm    else ""
    genre_item_desc  = f"genre({item_genre_dim}) + "       if m.has_item_genre  else ""
    tag_item_desc    = f"tag({item_tag_dim}) + "           if m.has_item_tag    else ""
    year_item_desc   = f" + year({year_dim})"              if m.has_item_year   else ""
    user_parts = [f"pools[{'+'.join(m.user_pools)}]({pool_total})"]
    if m.has_user_genome: user_parts.append(f"genome_ctx({genome_user_dim})")
    if m.has_user_llm:    user_parts.append(f"llm_ctx({llm_user_dim})")
    if m.has_user_genre:  user_parts.append(f"genre({genre_dim})")
    if m.has_timestamp:   user_parts.append(f"ts({ts_dim})")
    user_desc  = " + ".join(user_parts)

    proj_h  = m.user_projection[0].out_features
    out_dim = m.user_projection[2].out_features

    # The printed breakdown is derived from the live model's modules, but cross-check it against the
    # actual projection input dims so this summary can NEVER misrepresent the model being trained —
    # a mismatch means a wiring bug, so abort here rather than train an hours-long run on a model
    # that differs from what's printed.
    real_user, real_item = m.user_projection[0].in_features, m.item_projection[0].in_features
    assert user_total == real_user, (
        f"user-tower summary {user_total} != actual concat dim {real_user} "
        f"(pools={list(m.user_pools)}, user_features={list(m.user_features)})")
    assert item_total == real_item, (
        f"item-tower summary {item_total} != actual concat dim {real_item} "
        f"(item_features={list(m.item_features)})")

    print(f"\n── Model dimensions ──")
    print(f"  Feature towers: {m.feature_towers}  |  Base towers: {m.base_towers}")
    print(f"  Pools: {list(m.user_pools)}  |  User feats: {list(m.user_features)}  |  "
          f"Item feats: {list(m.item_features)}")
    print(f"  User side:  {user_desc}  =  {user_total}  (proj in={real_user})")
    print(f"  Item side:  {genre_item_desc}{tag_item_desc}{genome_item_desc}{llm_item_desc}"
          f"movieId({item_movieId_dim}){year_item_desc}  =  {item_total}  (proj in={real_item})")
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
    # Checkpoint name composes the enabled towers (no 'v2'); each model is unambiguous on disk:
    # best_softmax[_idonly][_genome_tags][_llm_features]_popularity_alpha_<a>[_<corpus>]_<ts>.pth
    # (config is also in the JSON sidecar). None (Model C) → best_softmax_popularity_alpha_...;
    # the stripped CF-base arms (BASE_TOWERS=idonly) lead with '_idonly'.
    feat_towers   = config['feature_towers']
    base_towers   = config.get('base_towers', 'all')
    tag_parts     = []
    if base_towers == 'idonly':           tag_parts.append('idonly')
    if feat_towers in ('genome', 'both'): tag_parts.append('genome_tags')
    if feat_towers in ('llm', 'both'):    tag_parts.append('llm_features')
    # Fine-grained selectors append a descriptive token ONLY when they deviate from the set
    # feature_towers/base_towers would imply — so coarse runs keep their exact historical names,
    # while the multi-pool arms get distinct, self-describing stems (hyphen-joined; pool names may
    # carry an inner '_', e.g. last_liked — the stem is descriptive only, config is re-resolved
    # from weight keys at load, never parsed from the filename).
    up  = tuple(config.get('user_pools')    or default_user_pools(base_towers))
    uf  = tuple(config.get('user_features') if config.get('user_features') is not None
                else default_user_features(feat_towers, base_towers))
    itf = tuple(config.get('item_features') if config.get('item_features') is not None
                else default_item_features(feat_towers, base_towers))
    if up  != default_user_pools(base_towers):
        tag_parts.append('pools-' + '-'.join(up))
    if uf  != default_user_features(feat_towers, base_towers):
        tag_parts.append('uf-' + ('-'.join(uf) if uf else 'none'))
    if itf != default_item_features(feat_towers, base_towers):
        tag_parts.append('if-' + ('-'.join(itf) if itf else 'none'))
    feat_tag      = ('_' + '_'.join(tag_parts)) if tag_parts else ''
    print(f"  Corpus: {CORPUS}  (checkpoint suffix: {corpus_sfx!r})  "
          f"feature towers: {feat_towers}  base towers: {base_towers}  (name tag: {feat_tag!r})")
    best_val_mrr  = float('-inf')   # best_path is selected on val-MRR (the reported metric), not val CE
    best_path     = os.path.join(checkpoint_dir, f'best_softmax{feat_tag}_popularity_alpha_{alpha_tag}{corpus_sfx}_{run_timestamp}.pth')

    # Bigger fixed val subset drives both the reported val-MRR and best-checkpoint selection.
    # The old 8,192-example subset made selection high-variance (±0.003-0.004 run-to-run MRR
    # noise, enough to flip the genome-vs-LLM ordering); the MRR estimate's SE ~ 1/sqrt(N), so
    # 100,000 cuts that ~3.5x (to ±~0.001) and for the smaller phase1 corpus likely uses all of
    # n_val (no subsampling at all). val-eval is forward-only and runs every 10k steps, so even
    # at 100k it's ~1s per eval — a negligible fraction of the 200k-step training loop.
    val_eval_size = min(100_000, n_val)
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
                                        f'softmax{feat_tag}_popularity_alpha_{alpha_tag}{corpus_sfx}_{run_timestamp}_step_{i:06d}.pth')
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
