"""
Checkpoint → model loading.

The single place that turns a saved checkpoint into a (config, state_dict) pair ready
for build_model(). Both the evaluate/canary/probe path (src/evaluate.py) and the export
path (src/export.py) go through here, so checkpoint-compatibility logic lives in exactly
one spot.

Two responsibilities:
  - resolve_config_from_state_dict: infer model dims AND the active tower/pool set from weight
    shapes + key presence. Every sub-tower is OPTIONAL and now independent per side — a checkpoint
    may have any subset of item/user genre, item tag/year, item/user genome, item/user llm,
    timestamp, and any subset of the {full,liked,disliked,weighted,last_liked,last_watched,second_to_last_watched} history pools (the
    hist_*_norm keys). The resolver guards each read and emits user_pools / user_features /
    item_features (plus back-compat base_towers / feature_towers) so a fine-grained or stripped
    checkpoint rebuilds from weights alone.
  - load_checkpoint: torch.load + remap legacy tower keys + drop legacy non-persistent buffers
    + resolve config.

Checkpoint-name history (all still load):
  - Pre-"content" era (incl. the prod-v3 checkpoint): genome towers under their explicit names
    item_genome_tag_tower / user_genome_context_tower — which are the CURRENT names, so these
    load natively (the persisted genome buffer is dropped below).
  - "content" era (the A/B ablation checkpoints): a single swappable slot named
    item_content_tower / user_content_tower, holding genome OR llm depending on the train-time
    source. These are remapped here to the explicit genome/llm names, disambiguated by the
    config sidecar (genome vs llm).

Note: get_config() (the training hyperparameter defaults) lives in src/train.py; this module
only reshapes those defaults to match a specific checkpoint.
"""
import json
import os

import torch

from src.train import get_config
from src.model import POOL_ORDER, USER_FEATURE_ORDER, ITEM_FEATURE_ORDER


# Non-persistent buffers the model rebuilds from the FeatureStore on load. Older checkpoints
# persisted them; drop from incoming state_dicts so they load cleanly under strict=True. Covers
# the legacy 'content' name and both current names.
LEGACY_NONPERSISTENT_BUFFERS = ('genome_context_buffer', 'content_context_buffer',
                                'llm_feature_buffer')


def _content_era_remap(source: str) -> dict:
    """Remap the 'content' era's single-slot tower keys to the explicit genome/llm names.

    The slot reused one name (item_content_tower / user_content_tower) for whichever source it
    held, so the genome-vs-llm target is taken from the config sidecar (default genome)."""
    base = 'llm_feature' if source == 'llm' else 'genome_tag'
    user = 'user_llm_feature' if source == 'llm' else 'user_genome_context'
    return {
        'item_content_tower.0.weight': f'item_{base}_tower.0.weight',
        'item_content_tower.0.bias':   f'item_{base}_tower.0.bias',
        'user_content_tower.0.weight': f'{user}_tower.0.weight',
        'user_content_tower.0.bias':   f'{user}_tower.0.bias',
    }


def _read_sidecar_source(checkpoint_path: str):
    """The feature-tower selection from the config sidecar, or None if absent. Reads the current
    'feature_towers' key, falling back to the legacy 'content_feature_source' key."""
    sidecar = os.path.splitext(checkpoint_path)[0] + '_config.json'
    if not os.path.exists(sidecar):
        return None
    with open(sidecar) as f:
        cfg = json.load(f)
    return cfg.get('feature_towers', cfg.get('content_feature_source'))


def resolve_config_from_state_dict(state_dict: dict) -> dict:
    """Infer model dims from a checkpoint's weight shapes. Returns a config dict.

    Assumes any legacy 'content' tower keys have already been remapped to the explicit
    genome/llm names (load_checkpoint does this first)."""
    sd  = state_dict
    cfg = get_config()

    cfg['item_movieId_embedding_size']      = sd['item_embedding_lookup.weight'].shape[1]
    cfg['proj_hidden']                      = sd['user_projection.0.weight'].shape[0]
    cfg['output_dim']                       = sd['user_projection.2.weight'].shape[0]
    if 'timestamp_embedding_lookup.weight' in sd:   # absent in the stripped 'idonly' arms
        cfg['timestamp_feature_embedding_size'] = sd['timestamp_embedding_lookup.weight'].shape[1]

    # Per-side towers — each optional and now INDEPENDENT (item vs user). Presence of a tower's
    # weight key says it's on; read its out_features for the dim. Stripped 'idonly' CF-base arms
    # drop the base towers; Model C drops the semantic slot; the new fine-grained arms can drop any
    # subset on either side — so guard every read and rebuild from whatever keys are present.
    has_item_genre  = 'item_genre_tower.0.weight'          in sd
    has_user_genre  = 'user_genre_tower.0.weight'          in sd
    has_item_tag    = 'item_tag_tower.0.weight'            in sd
    has_item_year   = 'year_embedding_lookup.weight'       in sd
    has_item_genome = 'item_genome_tag_tower.0.weight'     in sd
    has_user_genome = 'user_genome_context_tower.0.weight' in sd
    has_item_llm    = 'item_llm_feature_tower.0.weight'    in sd
    has_user_llm    = 'user_llm_feature_tower.0.weight'    in sd
    has_timestamp   = 'timestamp_embedding_lookup.weight'  in sd
    if has_item_genre:
        cfg['item_genre_embedding_size'] = sd['item_genre_tower.0.weight'].shape[0]
    if has_user_genre:
        cfg['user_genre_embedding_size'] = sd['user_genre_tower.0.weight'].shape[0]
    if has_item_tag:
        cfg['item_tag_embedding_size']   = sd['item_tag_tower.0.weight'].shape[0]
    if has_item_year:
        cfg['item_year_embedding_size']  = sd['year_embedding_lookup.weight'].shape[1]
    if has_item_genome:
        cfg['item_genome_embedding_size'] = sd['item_genome_tag_tower.0.weight'].shape[0]
    if has_user_genome:
        cfg['user_genome_embedding_size'] = sd['user_genome_context_tower.0.weight'].shape[0]
    if has_item_llm:
        cfg['item_llm_embedding_size'] = sd['item_llm_feature_tower.0.weight'].shape[0]
    if has_user_llm:
        cfg['user_llm_embedding_size'] = sd['user_llm_feature_tower.0.weight'].shape[0]

    # The three fine-grained selectors, resolved from key presence — the authoritative pool/tower
    # set (overwrites whatever get_config()'s env defaults seeded above). user_pools comes from the
    # per-pool hist_*_norm LayerNorm keys (each is a persisted param).
    cfg['user_pools']    = [p for p in POOL_ORDER if f'hist_{p}_norm.weight' in sd]
    present_user = {'genre': has_user_genre, 'genome': has_user_genome,
                    'llm': has_user_llm, 'timestamp': has_timestamp}
    present_item = {'genre': has_item_genre, 'tag': has_item_tag, 'genome': has_item_genome,
                    'llm': has_item_llm, 'year': has_item_year}
    cfg['user_features'] = [f for f in USER_FEATURE_ORDER if present_user[f]]
    cfg['item_features'] = [f for f in ITEM_FEATURE_ORDER if present_item[f]]

    # Coarse selectors kept for display/back-compat, derived to stay consistent with the per-side
    # sets (an old checkpoint resolves to exactly its historical base_towers/feature_towers).
    cfg['base_towers'] = ('all' if (has_item_genre or has_item_tag or has_item_year
                                    or has_user_genre or has_timestamp) else 'idonly')
    any_genome = has_item_genome or has_user_genome
    any_llm    = has_item_llm    or has_user_llm
    cfg['feature_towers'] = ('both'   if any_genome and any_llm else
                             'genome' if any_genome else
                             'llm'    if any_llm else None)

    return cfg


def load_checkpoint(checkpoint_path: str) -> tuple:
    """Load a checkpoint (CPU), remap legacy keys, drop legacy non-persistent buffers, resolve
    config from shapes. Returns (config, state_dict) ready for model.load_state_dict()."""
    state_dict = torch.load(checkpoint_path, weights_only=True, map_location='cpu')

    # Order matters: remap any 'content' era tower keys to explicit names (so the resolver sees the
    # current names), then drop the non-persistent buffers, then resolve config from shapes.
    if any(k.startswith(('item_content_tower', 'user_content_tower')) for k in state_dict):
        source = _read_sidecar_source(checkpoint_path)   # genome | llm (default genome)
        remap  = _content_era_remap(source)
        state_dict = {remap.get(k, k): v for k, v in state_dict.items()}
    for key in LEGACY_NONPERSISTENT_BUFFERS:
        state_dict.pop(key, None)

    config = resolve_config_from_state_dict(state_dict)
    return config, state_dict
