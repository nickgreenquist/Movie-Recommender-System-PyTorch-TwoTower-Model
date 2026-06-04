"""
Checkpoint → model loading.

The single place that turns a saved checkpoint into a (config, state_dict) pair ready
for build_model(). Both the evaluate/canary/probe path (src/evaluate.py) and the export
path (src/export.py) go through here, so checkpoint-compatibility logic lives in exactly
one spot.

Two responsibilities:
  - resolve_config_from_state_dict: infer model dims from weight shapes. Content-slot dims
    (genome today, LLM later) are OPTIONAL — a no-content-slot checkpoint omits those tower
    weights, so the resolver guards those reads instead of KeyError-ing.
  - load_checkpoint: torch.load + drop legacy non-persistent buffers + resolve config.

Note: get_config() (the training hyperparameter defaults) lives in src/train.py; this module
only reshapes those defaults to match a specific checkpoint.
"""
import torch

from src.train import get_config


# Buffers that older checkpoints persisted but the model now rebuilds from the FeatureStore
# on load (registered persistent=False). Dropped from incoming state_dicts so legacy
# checkpoints still load cleanly under strict=True.
LEGACY_NONPERSISTENT_BUFFERS = ('genome_context_buffer',)


def resolve_config_from_state_dict(state_dict: dict) -> dict:
    """Infer model dims from a checkpoint's weight shapes. Returns a config dict."""
    sd  = state_dict
    cfg = get_config()

    cfg['item_movieId_embedding_size']      = sd['item_embedding_lookup.weight'].shape[1]
    cfg['user_genre_embedding_size']        = sd['user_genre_tower.0.weight'].shape[0]
    cfg['timestamp_feature_embedding_size'] = sd['timestamp_embedding_lookup.weight'].shape[1]
    cfg['item_genre_embedding_size']        = sd['item_genre_tower.0.weight'].shape[0]
    cfg['item_tag_embedding_size']          = sd['item_tag_tower.0.weight'].shape[0]
    cfg['item_year_embedding_size']         = sd['year_embedding_lookup.weight'].shape[1]
    cfg['proj_hidden']                      = sd['user_projection.0.weight'].shape[0]
    cfg['output_dim']                       = sd['user_projection.2.weight'].shape[0]

    # Content slot (genome today, LLM later) — optional; a no-content model omits these keys.
    if 'item_genome_tag_tower.0.weight' in sd:
        cfg['item_genome_tag_embedding_size'] = sd['item_genome_tag_tower.0.weight'].shape[0]
    if 'user_genome_context_tower.0.weight' in sd:
        cfg['user_genome_context_embedding_size'] = sd['user_genome_context_tower.0.weight'].shape[0]

    return cfg


def load_checkpoint(checkpoint_path: str) -> tuple:
    """Load a checkpoint (CPU), drop legacy non-persistent buffers, resolve config from shapes.
    Returns (config, state_dict). The returned state_dict is ready for model.load_state_dict()."""
    state_dict = torch.load(checkpoint_path, weights_only=True, map_location='cpu')
    for key in LEGACY_NONPERSISTENT_BUFFERS:
        state_dict.pop(key, None)
    config = resolve_config_from_state_dict(state_dict)
    return config, state_dict
