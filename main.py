"""
Movie Recommender System — CLI entry point.

Usage:
    python main.py preprocess          # Stage 1: raw CSVs → data/base_*.parquet
    python main.py features            # Stage 2: base parquets → data/features_*.parquet
    python main.py dataset             # Stage 3: features → data/dataset_softmax_*_v2.pt
    python main.py train               # Stage 4: full softmax training
    python main.py canary              # Canary user recommendations (most recent checkpoint)
    python main.py canary <path>       # Canary user recommendations (specific checkpoint)
    python main.py probe               # Embedding probes (most recent checkpoint)
    python main.py probe <path>        # Embedding probes (specific checkpoint)
    python main.py eval                # Offline eval: Recall@K, NDCG@K, Hit Rate@K, MRR
    python main.py eval <path>         # Same, specific checkpoint
    python main.py export              # Stage 5: export serving/ artifacts for Streamlit app
    python main.py export <path>       # Export using specific checkpoint
    python main.py posters             # Fetch movie poster URLs from TMDB → serving/posters.json
"""
import os
import sys

DATA_DIR = 'data'
RAW_DIR  = 'data/ml-32m'
# Artifact versions live with the modules that produce them — features_movies_*.parquet uses
# src/features.py FEATURES_VERSION (the library defaults below resolve to it); the softmax
# dataset .pt / counts use src/dataset.py DATASET_VERSION, threaded through explicitly.


def cmd_preprocess():
    from src.preprocess import run
    run(raw_dir=RAW_DIR, out_dir=DATA_DIR)


def cmd_features():
    from src.features import run
    run(data_dir=DATA_DIR)


def cmd_dataset():
    from src.dataset import (DATASET_VERSION, load_features, make_softmax_splits,
                             save_softmax_splits)
    print("Loading features ...")
    fs = load_features(DATA_DIR)
    print("\nBuilding softmax datasets ...")
    train_data, val_data = make_softmax_splits(fs, DATA_DIR)
    save_softmax_splits(train_data, val_data, DATA_DIR, version=DATASET_VERSION)


def cmd_train():
    import random

    import numpy as np
    import torch
    from src.dataset import DATASET_VERSION, load_features, load_softmax_splits
    from src.train import get_config, build_model, train_softmax
    print("Loading features ...")
    fs = load_features(DATA_DIR)
    print("\nLoading softmax datasets ...")
    train_data, val_data = load_softmax_splits(DATA_DIR, version=DATASET_VERSION)
    config = get_config()
    # Seed every RNG BEFORE build_model: weight init draws from the global torch RNG, so the
    # seed must land before init for a run to be reproducible — and so the only thing that
    # differs across the ablation arms is the content slot, not the random init. Seeding inside
    # train_softmax (after the model is built) is too late. SEED env overrides the default.
    seed = int(os.environ.get('SEED', 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    print(f"Seed: {seed}")
    model  = build_model(config, fs)
    train_softmax(model, train_data, val_data, config, fs)


def cmd_canary(checkpoint_path=None):
    from src.evaluate import run_canary
    run_canary(data_dir=DATA_DIR, checkpoint_path=checkpoint_path)


def cmd_probe(checkpoint_path=None):
    from src.evaluate import run_probes
    run_probes(data_dir=DATA_DIR, checkpoint_path=checkpoint_path)


def cmd_eval(checkpoint_path=None):
    from src.evaluate import _setup
    from src.offline_eval import run_offline_eval

    result = _setup(data_dir=DATA_DIR, checkpoint_path=checkpoint_path)
    model, fs = result[0], result[1]
    checkpoint_path = result[-1]
    # EVAL_N_USERS overrides the val-user sample size (default 5,000). Raising it
    # gives the long-tail tiers more examples — the tail is ~3% of targets, so the
    # whole-corpus numbers barely move but the tail comparison gets far more signal.
    n_users = int(os.environ['EVAL_N_USERS']) if 'EVAL_N_USERS' in os.environ else 5_000
    run_offline_eval(model, fs, checkpoint_path=checkpoint_path or '', data_dir=DATA_DIR,
                     n_users=n_users)


def cmd_export(checkpoint_path=None, variant=None):
    from src.export import run_export
    run_export(data_dir=DATA_DIR, checkpoint_path=checkpoint_path, variant=variant)


def cmd_posters():
    from src.fetch_posters import run_fetch_posters
    run_fetch_posters(data_dir=DATA_DIR)


COMMANDS = {
    'preprocess': cmd_preprocess,
    'features':   cmd_features,
    'dataset':    cmd_dataset,
    'train':      cmd_train,
    'canary':     cmd_canary,
    'probe':      cmd_probe,
    'eval':       cmd_eval,
    'export':     cmd_export,
    'posters':    cmd_posters,
}

if __name__ == '__main__':
    args = sys.argv[1:]

    if not args:
        print("Running all stages: preprocess → features → dataset → train → canary\n")
        cmd_preprocess()
        cmd_features()
        cmd_dataset()
        cmd_train()
        cmd_canary()
    elif args[0] in COMMANDS:
        cmd = args[0]
        if cmd == 'export':
            # export [<checkpoint>] [--variant <name>] — --variant writes a secondary model
            # (model_<name>.pth + movie_embeddings_<name>.pt) alongside prod, sharing feature_store.
            rest    = args[1:]
            variant = None
            if '--variant' in rest:
                i       = rest.index('--variant')
                variant = rest[i + 1] if i + 1 < len(rest) else None
                rest    = rest[:i] + rest[i + 2:]
            cmd_export(checkpoint_path=(rest[0] if rest else None), variant=variant)
        elif cmd in ('canary', 'probe', 'eval') and len(args) > 1:
            COMMANDS[cmd](checkpoint_path=args[1])
        else:
            COMMANDS[cmd]()
    else:
        print(__doc__)
        sys.exit(1)
