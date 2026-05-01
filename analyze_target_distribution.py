"""
Print top-100 movies by label frequency for both dataset orderings.

Usage: python analyze_target_distribution.py
"""
import collections
import pandas as pd

DATA_DIR     = 'data'
VERSION      = 'v1'
TOP_K        = 100

from src.dataset import load_features, get_val_users, build_mse_rollback_dataset, MAX_MSE_ROLLBACK_EXAMPLES_PER_USER

print("Loading feature store ...")
fs = load_features(DATA_DIR, VERSION)

print("Getting val users ...")
val_users, raw_df = get_val_users(fs, DATA_DIR)

movies_df = pd.read_parquet(f'{DATA_DIR}/base_movies.parquet')[['movieId', 'title']].set_index('movieId')
i_to_mid  = {v: k for k, v in fs.item_emb_movieId_to_i.items()}

def title(mid):
    try:
        return movies_df.loc[mid, 'title'][:53]
    except KeyError:
        return str(mid)

def run_and_print(sort_by_ts: bool):
    label = "timestamp sort" if sort_by_ts else "shuffle"
    col_count = f"Count - {'ts sorted' if sort_by_ts else 'shuffled'}"
    col_pct   = f"% - {'ts sorted' if sort_by_ts else 'shuffled'}"

    print(f"\nBuilding rollback examples — {label} ...")
    _, _, _, _, _, target_movieId_t = build_mse_rollback_dataset(
        val_users, fs, raw_df, MAX_MSE_ROLLBACK_EXAMPLES_PER_USER,
        sort_by_ts=sort_by_ts,
    )

    counter = collections.Counter(int(x) for x in target_movieId_t)
    total   = sum(counter.values())

    print(f"{total:,} total labels\n")
    print(f"{'Rk':<5} {'Movie':<55} {col_count:>18}  {col_pct:>14}")
    print("-" * 97)
    for rank, (emb_idx, count) in enumerate(counter.most_common(TOP_K), 1):
        mid = i_to_mid[emb_idx]
        print(f"{rank:<5} {title(mid):<55} {count:>18,}  {100*count/total:>13.3f}%")

run_and_print(sort_by_ts=True)
run_and_print(sort_by_ts=False)
