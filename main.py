"""
Movie Recommender System — CLI entry point.

Usage:
    python main.py preprocess          # Stage 1: raw CSVs → data/base_*.parquet
    python main.py features            # Stage 2: base parquets → data/features_*.parquet
    python main.py dataset rollback    # Stage 3: features → data/dataset_mse_rollback_*_v1.pt
    python main.py train rollback      # Stage 4: MSE training (rollback dataset)
    python main.py canary              # Canary user recommendations (most recent checkpoint)
    python main.py canary <path>       # Canary user recommendations (specific checkpoint)
    python main.py probe               # Embedding probes (most recent checkpoint)
    python main.py probe <path>        # Embedding probes (specific checkpoint)
    python main.py eval                # Offline eval: Recall@K, NDCG@K, Hit Rate@K, MRR (rollback)
    python main.py eval <path>         # Same, specific checkpoint
    python main.py export              # Stage 5: export serving/ artifacts for Streamlit app
    python main.py export <path>       # Export using specific checkpoint
    python main.py posters             # Fetch movie poster URLs from TMDB → serving/posters.json
"""
import sys

DATA_DIR = 'data'
RAW_DIR  = 'data/ml-32m'
VERSION  = 'v1'


def cmd_preprocess():
    from src.preprocess import run
    run(raw_dir=RAW_DIR, out_dir=DATA_DIR)


def cmd_features():
    from src.features import run
    run(data_dir=DATA_DIR, version=VERSION)


def cmd_dataset(mode='rollback', **kwargs):
    from src.dataset import load_features, make_mse_rollback_splits, save_mse_rollback_splits
    print("Loading features ...")
    fs = load_features(DATA_DIR, VERSION)
    print("\nBuilding MSE rollback datasets ...")
    train_data, val_data = make_mse_rollback_splits(fs, DATA_DIR)
    save_mse_rollback_splits(train_data, val_data, DATA_DIR, VERSION)


def cmd_train(**kwargs):
    from src.dataset import load_features, load_mse_rollback_splits
    from src.train import get_config, build_model, train
    print("Loading features ...")
    fs = load_features(DATA_DIR, VERSION)
    print("\nLoading MSE rollback datasets ...")
    train_data, val_data = load_mse_rollback_splits(DATA_DIR, VERSION)
    config = get_config()
    model  = build_model(config, fs)
    train(model, train_data, val_data, config, fs)


def cmd_canary(checkpoint_path=None):
    from src.evaluate import run_canary
    run_canary(data_dir=DATA_DIR, checkpoint_path=checkpoint_path, version=VERSION)


def cmd_probe(checkpoint_path=None):
    from src.evaluate import run_probes
    run_probes(data_dir=DATA_DIR, checkpoint_path=checkpoint_path, version=VERSION)


def cmd_eval(checkpoint_path=None):
    from src.evaluate import _setup
    from src.offline_eval import run_offline_eval

    model, fs, _, _, _, _, _ = _setup(data_dir=DATA_DIR, checkpoint_path=checkpoint_path,
                                       version=VERSION)
    run_offline_eval(model, fs, checkpoint_path=checkpoint_path or '', data_dir=DATA_DIR)


def cmd_export(checkpoint_path=None):
    from src.export import run_export
    run_export(data_dir=DATA_DIR, checkpoint_path=checkpoint_path, version=VERSION)


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
        if cmd in ('canary', 'probe', 'eval', 'export') and len(args) > 1:
            COMMANDS[cmd](checkpoint_path=args[1])
        else:
            COMMANDS[cmd]()
    else:
        print(__doc__)
        sys.exit(1)
