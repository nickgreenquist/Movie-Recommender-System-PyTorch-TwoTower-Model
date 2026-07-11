"""
Smoke test for the Two-Tower MovieRecommender forward pass.

Builds tiny models directly (no dataset / no checkpoint) with synthetic context buffers,
runs forward passes, and asserts:
  - user_embedding() / item_embedding() / full_item_embedding() return (batch, output_dim)
  - both tower outputs are L2-normalized (unit norm per row)
  - forward() returns (batch,) cosine scores in [-1, 1], all finite
  - the feature_towers slot selector builds exactly Model A/B/C/D (genome / llm / None / both):
    the right semantic sub-towers + buffers exist, and each projection's input dim shrinks by
    exactly the missing slot's contribution — the structural guard behind the ablation results
  - base_towers='idonly' strips the genre/tag/year/timestamp towers and collapses the pools
    to the single full-history sum (the stripped-CF-base ablation arms)
  - the user_pools fine-grained knob builds exactly its LayerNorms, and the recency pool
    helpers (_last_liked / _last_watched / _second_to_last_watched) return the same ids for
    right-aligned (training) and left-aligned (rollback eval) history layouts

Cheap insurance that the architecture's shape/normalization contract holds. Kept fast and
deterministic: CPU only, no_grad, fixed seed. The tiny genome and LLM buffers use DIFFERENT
widths so a genome↔llm cross-wiring bug fails loudly instead of passing by coincidence.

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
N_GENRES        = 4
N_TAGS          = 5
N_GENOME_TAGS   = 8    # ≠ N_LLM_FEATURES on purpose: catches genome↔llm buffer cross-wiring
N_LLM_FEATURES  = 7
N_MOVIES        = 10   # corpus size; pad_idx == N_MOVIES
N_YEARS         = 6
N_TS_BINS       = 12
OUTPUT_DIM      = 128  # prod default — documents the output contract
SLOT_DIM        = 32   # every semantic sub-tower (genome/llm × item/user) is 32-dim out


def _build_tiny_model(seed: int = 0, feature_towers='both', base_towers: str = 'all',
                      user_pools=None) -> MovieRecommender:
    """Construct a MovieRecommender with synthetic context buffers (last row = padding).

    feature_towers: 'genome' | 'llm' | 'both' | None — Model A / B / D / C. Defaults to
    'both' so the default tiny model matches the PROD config (Model D, 4-pool user tower).
    base_towers / user_pools are passed through to exercise the ablation knobs.
    """
    torch.manual_seed(seed)
    n_pad = N_MOVIES + 1  # corpus rows + one padding row

    genome_buf = torch.rand(n_pad, N_GENOME_TAGS);  genome_buf[-1] = 0.0
    llm_buf    = torch.rand(n_pad, N_LLM_FEATURES); llm_buf[-1]    = 0.0
    genre_buf  = torch.rand(n_pad, N_GENRES);       genre_buf[-1]  = 0.0
    tag_buf    = torch.rand(n_pad, N_TAGS);         tag_buf[-1]    = 0.0
    year_buf   = torch.randint(0, N_YEARS, (n_pad,), dtype=torch.long); year_buf[-1] = 0

    return MovieRecommender(
        genres_len=N_GENRES,
        tags_len=N_TAGS,
        genome_tags_len=N_GENOME_TAGS,
        top_movies_len=N_MOVIES,
        all_years_len=N_YEARS,
        timestamp_num_bins=N_TS_BINS,
        user_context_size=2 * N_GENRES,
        feature_towers=feature_towers,
        base_towers=base_towers,
        user_pools=user_pools,
        genome_context_buffer=(genome_buf if feature_towers in ('genome', 'both') else None),
        llm_feature_buffer=(llm_buf if feature_towers in ('llm', 'both') else None),
        llm_feature_len=N_LLM_FEATURES,
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


def _assert_unit_norm(emb: torch.Tensor, n_rows: int, name: str) -> None:
    assert emb.shape == (n_rows, OUTPUT_DIM), f"{name} shape {emb.shape} != {(n_rows, OUTPUT_DIM)}"
    norms = emb.norm(p=2, dim=1)
    assert torch.allclose(norms, torch.ones(n_rows), atol=1e-5), f"{name} not unit-norm: {norms}"
    assert torch.isfinite(emb).all(), f"{name} has non-finite values"


def _assert_towers_run(model: MovieRecommender, name: str) -> None:
    """Both towers emit unit-norm (batch, OUTPUT_DIM) for a synthetic batch."""
    model.eval()
    user_ctx, hist, liked, disliked, ratings, ts, target = _synthetic_batch()
    with torch.no_grad():
        user_emb = model.user_embedding(user_ctx, hist, liked, disliked, ratings, ts)
        item_emb = model.item_embedding(target)
    _assert_unit_norm(user_emb, user_ctx.shape[0], f"{name} user_emb")
    _assert_unit_norm(item_emb, target.shape[0], f"{name} item_emb")


def _param_shapes(model: MovieRecommender) -> dict:
    return {name: tuple(p.shape) for name, p in model.named_parameters()}


# ── Shape / normalization contract (prod Model D config) ──────────────────────

def test_user_embedding_shape_and_norm():
    model = _build_tiny_model()
    model.eval()
    user_ctx, hist, liked, disliked, ratings, ts, _ = _synthetic_batch()
    with torch.no_grad():
        user_emb = model.user_embedding(user_ctx, hist, liked, disliked, ratings, ts)
    _assert_unit_norm(user_emb, user_ctx.shape[0], "user_emb")


def test_item_embedding_shape_and_norm():
    model = _build_tiny_model()
    model.eval()
    target = _synthetic_batch()[6]
    with torch.no_grad():
        item_emb = model.item_embedding(target)
        all_emb  = model.full_item_embedding()   # the full-softmax scoring path
    _assert_unit_norm(item_emb, target.shape[0], "item_emb")
    _assert_unit_norm(all_emb, N_MOVIES, "full_item_embedding")


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


# ── feature_towers slot selector: Models A / B / C / D ────────────────────────

def test_feature_tower_variants_differ_only_in_slot():
    """The ablation's validity rests on the variants differing ONLY in the semantic slot(s):
    each of A (genome) / B (llm) / C (None) / D (both) builds exactly its towers + buffers,
    the projection input dims shrink by exactly the missing slots' contributions, and every
    non-slot parameter is shape-identical across all four."""
    genome = _build_tiny_model(feature_towers='genome')   # Model A
    llm    = _build_tiny_model(feature_towers='llm')      # Model B
    none_  = _build_tiny_model(feature_towers=None)       # Model C (CF floor)
    both   = _build_tiny_model(feature_towers='both')     # Model D (prod)

    slot_attrs = {
        'genome': ('item_genome_tag_tower', 'user_genome_context_tower', 'genome_context_buffer'),
        'llm':    ('item_llm_feature_tower', 'user_llm_feature_tower', 'llm_feature_buffer'),
    }
    expected = {'genome': ('genome',), 'llm': ('llm',), 'none': (), 'both': ('genome', 'llm')}
    models   = {'genome': genome, 'llm': llm, 'none': none_, 'both': both}
    for name, m in models.items():
        for slot, attrs in slot_attrs.items():
            for attr in attrs:
                present = slot in expected[name]
                assert hasattr(m, attr) == present, \
                    f"Model {name}: {attr} {'missing' if present else 'unexpectedly present'}"

    # Each absent slot removes exactly SLOT_DIM from both projections' input dims.
    def proj_dims(m):
        return m.user_projection[0].in_features, m.item_projection[0].in_features
    u_none, i_none = proj_dims(none_)
    for name, n_slots in (('genome', 1), ('llm', 1), ('both', 2)):
        u, i = proj_dims(models[name])
        assert u - u_none == n_slots * SLOT_DIM, f"{name}: user proj dim off"
        assert i - i_none == n_slots * SLOT_DIM, f"{name}: item proj dim off"

    # Every non-slot, non-projection parameter is shape-identical across all four variants.
    base_shapes = {k: v for k, v in _param_shapes(both).items()
                   if 'genome' not in k and 'llm' not in k
                   and not k.startswith(('user_projection', 'item_projection'))}
    for name, m in models.items():
        shapes = _param_shapes(m)
        for k, v in base_shapes.items():
            assert shapes.get(k) == v, f"non-slot parameter {k} changed in Model {name}"

    for name, m in models.items():
        _assert_towers_run(m, name)


# ── base_towers='idonly': the stripped-CF-base ablation arms ──────────────────

def test_idonly_strips_base_towers_and_pools():
    model = _build_tiny_model(feature_towers='genome', base_towers='idonly')  # ablation A′
    assert model.user_pools == ('full',), f"idonly pools {model.user_pools} != ('full',)"
    for absent in ('item_genre_tower', 'item_tag_tower', 'year_embedding_tower',
                   'user_genre_tower', 'timestamp_embedding_tower',
                   'hist_liked_norm', 'hist_disliked_norm', 'hist_weighted_norm',
                   'genre_context_buffer', 'tag_context_buffer', 'year_context_buffer'):
        assert not hasattr(model, absent), f"idonly must omit {absent}"
    assert hasattr(model, 'item_genome_tag_tower') and hasattr(model, 'user_genome_context_tower')
    _assert_towers_run(model, 'idonly')


# ── user_pools fine-grained knob + recency-pool helpers ───────────────────────

def test_user_pools_knob_builds_exact_norms():
    pools = ('full', 'last_liked', 'last_watched', 'second_to_last_watched')
    model = _build_tiny_model(user_pools=pools)
    for p in pools:
        assert hasattr(model, f'hist_{p}_norm'), f"missing hist_{p}_norm"
    for absent in ('hist_liked_norm', 'hist_disliked_norm', 'hist_weighted_norm'):
        assert not hasattr(model, absent), f"unselected pool norm {absent} present"
    # 4 pools × 32 + genome(32) + llm(32) + genre(32) + timestamp(4) = 228.
    assert model.user_projection[0].in_features == len(pools) * 32 + 32 + 32 + 32 + 4
    _assert_towers_run(model, 'pools-knob')


def test_recency_pool_ids_are_alignment_independent():
    """_last_liked / _last_watched / _second_to_last_watched must return the same ids for the
    training layout (right-aligned pads-first) and the rollback-eval layout (left-aligned),
    and resolve to pad_idx when the pool is empty."""
    model = _build_tiny_model()
    pad = N_MOVIES
    # Row 0: watches 3 (liked) then 7 (disliked). Row 1: single disliked watch 5.
    right_hist = torch.tensor([[pad, 3, 7], [pad, pad, 5]])
    right_rats = torch.tensor([[0.0, 1.0, -1.0], [0.0, 0.0, -2.0]])
    left_hist  = torch.tensor([[3, 7, pad], [5, pad, pad]])
    left_rats  = torch.tensor([[1.0, -1.0, 0.0], [-2.0, 0.0, 0.0]])
    for hist, rats in ((right_hist, right_rats), (left_hist, left_rats)):
        assert model._last_watched_ids(hist).tolist() == [7, 5]
        assert model._second_to_last_watched_ids(hist).tolist() == [3, pad]  # row 1: only 1 watch
        assert model._last_liked_ids(hist, rats).tolist() == [3, pad]        # row 1: no likes


if __name__ == '__main__':
    test_user_embedding_shape_and_norm()
    test_item_embedding_shape_and_norm()
    test_forward_scores_shape_and_range()
    test_feature_tower_variants_differ_only_in_slot()
    test_idonly_strips_base_towers_and_pools()
    test_user_pools_knob_builds_exact_norms()
    test_recency_pool_ids_are_alignment_independent()
    print("All model shape/norm smoke tests passed.")
