"""
Smoke test for the Two-Tower MovieRecommender forward pass.

Builds a tiny model directly (no dataset / no checkpoint) with synthetic context buffers,
runs one forward pass, and asserts:
  - user_embedding() and item_embedding() return (batch, output_dim)
  - both tower outputs are L2-normalized (unit norm per row)
  - forward() returns (batch,) cosine scores in [-1, 1], all finite

Cheap insurance that the architecture's shape/normalization contract holds. Kept fast and
deterministic: CPU only, no_grad, fixed seed. The _build_tiny_model / _synthetic_batch
helpers are parameterized so a later content-slot ablation can build variants from them.

Run with pytest (`pytest tests/`) or as a plain script (`python -m tests.test_model_shapes`).
"""
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.model import MovieRecommender  # noqa: E402


# Tiny vocab dims — small enough to be instant, large enough to exercise every tower.
N_GENRES      = 4
N_TAGS        = 5
N_GENOME_TAGS = 8
N_MOVIES      = 10   # corpus size; pad_idx == N_MOVIES
N_YEARS       = 6
N_TS_BINS     = 12
OUTPUT_DIM    = 128  # prod default — documents the output contract


def _build_tiny_model(seed: int = 0) -> MovieRecommender:
    """Construct a MovieRecommender with synthetic context buffers (last row = padding)."""
    torch.manual_seed(seed)
    n_pad = N_MOVIES + 1  # corpus rows + one padding row

    genome_buf = torch.rand(n_pad, N_GENOME_TAGS); genome_buf[-1] = 0.0
    genre_buf  = torch.rand(n_pad, N_GENRES);      genre_buf[-1]  = 0.0
    tag_buf    = torch.rand(n_pad, N_TAGS);        tag_buf[-1]    = 0.0
    year_buf   = torch.randint(0, N_YEARS, (n_pad,), dtype=torch.long); year_buf[-1] = 0

    return MovieRecommender(
        genres_len=N_GENRES,
        tags_len=N_TAGS,
        genome_tags_len=N_GENOME_TAGS,
        top_movies_len=N_MOVIES,
        all_years_len=N_YEARS,
        timestamp_num_bins=N_TS_BINS,
        user_context_size=2 * N_GENRES,
        genome_context_buffer=genome_buf,
        genre_context_buffer=genre_buf,
        tag_context_buffer=tag_buf,
        year_context_buffer=year_buf,
        output_dim=OUTPUT_DIM,
    )


def _synthetic_batch(batch_size: int = 2, hist_len: int = 3, seed: int = 1) -> tuple:
    """Return a 7-tuple of forward() inputs with the dataset's dtypes/ranges."""
    torch.manual_seed(seed)
    pad_idx = N_MOVIES
    user_genre_contexts = torch.rand(batch_size, 2 * N_GENRES)
    # History ids may include pad_idx; force at least one real id per row for a meaningful pool.
    watch_history = torch.randint(0, pad_idx + 1, (batch_size, hist_len), dtype=torch.long)
    watch_history[:, 0] = torch.randint(0, N_MOVIES, (batch_size,))
    hist_liked    = torch.randint(0, pad_idx + 1, (batch_size, hist_len), dtype=torch.long)
    hist_disliked = torch.randint(0, pad_idx + 1, (batch_size, hist_len), dtype=torch.long)
    history_ratings = torch.randn(batch_size, hist_len)
    timestamps    = torch.randint(0, N_TS_BINS, (batch_size,), dtype=torch.long)
    target_movieId = torch.randint(0, N_MOVIES, (batch_size,), dtype=torch.long)  # < pad_idx
    return (user_genre_contexts, watch_history, hist_liked, hist_disliked,
            history_ratings, timestamps, target_movieId)


def _assert_unit_norm(emb: torch.Tensor, batch_size: int, name: str) -> None:
    assert emb.shape == (batch_size, OUTPUT_DIM), f"{name} shape {emb.shape} != {(batch_size, OUTPUT_DIM)}"
    norms = emb.norm(p=2, dim=1)
    assert torch.allclose(norms, torch.ones(batch_size), atol=1e-5), f"{name} not unit-norm: {norms}"
    assert torch.isfinite(emb).all(), f"{name} has non-finite values"


def test_user_embedding_shape_and_norm():
    model = _build_tiny_model()
    model.eval()
    batch = _synthetic_batch()
    user_ctx, hist, liked, disliked, ratings, ts, _ = batch
    with torch.no_grad():
        user_emb = model.user_embedding(user_ctx, hist, liked, disliked, ratings, ts)
    _assert_unit_norm(user_emb, user_ctx.shape[0], "user_emb")


def test_item_embedding_shape_and_norm():
    model = _build_tiny_model()
    model.eval()
    target = _synthetic_batch()[6]
    with torch.no_grad():
        item_emb = model.item_embedding(target)
    _assert_unit_norm(item_emb, target.shape[0], "item_emb")


def test_forward_scores_shape_and_range():
    model = _build_tiny_model()
    model.eval()
    batch = _synthetic_batch()
    batch_size = batch[0].shape[0]
    with torch.no_grad():
        scores = model(*batch)
    assert scores.shape == (batch_size,), f"scores shape {scores.shape} != {(batch_size,)}"
    assert torch.isfinite(scores).all(), "scores contain non-finite values"
    # Cosine of two L2-normalized vectors ∈ [-1, 1].
    assert scores.abs().max().item() <= 1.0 + 1e-5, f"score out of range: {scores}"


if __name__ == '__main__':
    test_user_embedding_shape_and_norm()
    test_item_embedding_shape_and_norm()
    test_forward_scores_shape_and_range()
    print("All model shape/norm smoke tests passed.")
