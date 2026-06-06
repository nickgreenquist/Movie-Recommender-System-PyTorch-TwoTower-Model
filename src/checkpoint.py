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
  - load_checkpoint: torch.load + remap legacy content-slot tower keys (genome→content) +
    drop legacy non-persistent buffers + resolve config.

Note: get_config() (the training hyperparameter defaults) lives in src/train.py; this module
only reshapes those defaults to match a specific checkpoint.
"""
import json
import os

import torch

from src.train import get_config


# Buffers that older checkpoints persisted but the model now rebuilds from the FeatureStore
# on load (registered persistent=False). Dropped from incoming state_dicts so legacy
# checkpoints still load cleanly under strict=True. Both the legacy 'genome_context_buffer'
# name and the renamed 'content_context_buffer' are dropped (Tier 1.1 rename).
LEGACY_NONPERSISTENT_BUFFERS = ('genome_context_buffer', 'content_context_buffer')

# Tier 1.1 swappable-slot rename: legacy checkpoints store the content towers under their old
# genome-specific names. Remap them to the new content-slot names before load_state_dict, so the
# prod checkpoint (and every other pre-rename checkpoint) still loads under strict=True.
LEGACY_KEY_REMAP = {
    'item_genome_tag_tower.0.weight':     'item_content_tower.0.weight',
    'item_genome_tag_tower.0.bias':       'item_content_tower.0.bias',
    'user_genome_context_tower.0.weight': 'user_content_tower.0.weight',
    'user_genome_context_tower.0.bias':   'user_content_tower.0.bias',
}


def resolve_config_from_state_dict(state_dict: dict) -> dict:
    """Infer model dims from a checkpoint's weight shapes. Returns a config dict.

    Assumes LEGACY_KEY_REMAP has already been applied (the content towers carry their new
    names). Callers other than load_checkpoint must remap first if they pass a legacy dict."""
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

    # Content slot — optional; the no-content model (Model C) omits the tower keys. Key presence
    # tells us the slot is filled; the *source* (genome 1128-dim vs LLM 132-dim) can't be told from
    # shapes alone (both are Linear(N→32)), so 'genome' here is only the legacy default — load_checkpoint
    # overrides it from the train-time config sidecar when one exists (the LLM checkpoints have one).
    if 'item_content_tower.0.weight' in sd:
        cfg['content_feature_source']      = 'genome'
        cfg['item_content_embedding_size'] = sd['item_content_tower.0.weight'].shape[0]
        cfg['user_content_embedding_size'] = sd['user_content_tower.0.weight'].shape[0]
    else:
        cfg['content_feature_source'] = None

    return cfg


def load_checkpoint(checkpoint_path: str) -> tuple:
    """Load a checkpoint (CPU), remap legacy keys, drop legacy non-persistent buffers, resolve
    config from shapes. Returns (config, state_dict) ready for model.load_state_dict()."""
    state_dict = torch.load(checkpoint_path, weights_only=True, map_location='cpu')
    # Order matters: remap tower keys first (so the resolver sees the new names), then drop the
    # non-persistent buffers (old + new names), then resolve config.
    state_dict = {LEGACY_KEY_REMAP.get(k, k): v for k, v in state_dict.items()}
    for key in LEGACY_NONPERSISTENT_BUFFERS:
        state_dict.pop(key, None)
    config = resolve_config_from_state_dict(state_dict)

    # Disambiguate the content slot's source (genome vs llm) — shapes can't, so trust the config
    # sidecar written next to the checkpoint at train time. Legacy checkpoints have no sidecar and
    # keep the shape-based 'genome' default. Only the source label is overridden; dims stay from shapes.
    if config.get('content_feature_source') is not None:
        sidecar_path = os.path.splitext(checkpoint_path)[0] + '_config.json'
        if os.path.exists(sidecar_path):
            with open(sidecar_path) as f:
                saved_source = json.load(f).get('content_feature_source')
            if saved_source is not None:
                config['content_feature_source'] = saved_source

    return config, state_dict
