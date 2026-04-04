"""
Movie Recommender System — CLI entry point.

Usage:
    python main.py preprocess          # Stage 1: raw CSVs → data/base_*.parquet
    python main.py features            # Stage 2: base parquets → data/features_*.parquet
    python main.py dataset             # Stage 3: features → data/dataset_*_v1.pt
    python main.py train               # Stage 4: load datasets, train model, save checkpoint
    python main.py canary              # Canary user recommendations (most recent checkpoint)
    python main.py canary <path>       # Canary user recommendations (specific checkpoint)
    python main.py probe               # Embedding probes (most recent checkpoint)
    python main.py probe <path>        # Embedding probes (specific checkpoint)
    python main.py                     # Run all stages in order
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


def cmd_dataset():
    from src.dataset import load_features, make_splits, save_splits

    print("Loading features ...")
    fs = load_features(DATA_DIR, VERSION)

    print("\nBuilding datasets ...")
    train_data, val_data = make_splits(fs)
    save_splits(train_data, val_data, DATA_DIR, VERSION)


def cmd_train():
    from src.dataset import load_features, load_splits
    from src.train import get_config, build_model, train

    print("Loading features ...")
    fs = load_features(DATA_DIR, VERSION)

    print("\nLoading datasets ...")
    train_data, val_data = load_splits(DATA_DIR, VERSION)

    config = get_config()
    model  = build_model(config, fs)
    train(model, train_data, val_data, config, fs)


def cmd_canary(checkpoint_path=None):
    from src.evaluate import run_canary
    run_canary(data_dir=DATA_DIR, checkpoint_path=checkpoint_path, version=VERSION)


def cmd_probe(checkpoint_path=None):
    from src.evaluate import run_probes
    run_probes(data_dir=DATA_DIR, checkpoint_path=checkpoint_path, version=VERSION)



COMMANDS = {
    'preprocess': cmd_preprocess,
    'features':   cmd_features,
    'dataset':    cmd_dataset,
    'train':      cmd_train,
    'canary':     cmd_canary,
    'probe':      cmd_probe,
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
        if args[0] in ('canary', 'probe') and len(args) > 1:
            COMMANDS[args[0]](checkpoint_path=args[1])
        else:
            COMMANDS[args[0]]()
    else:
        print(__doc__)
        sys.exit(1)
