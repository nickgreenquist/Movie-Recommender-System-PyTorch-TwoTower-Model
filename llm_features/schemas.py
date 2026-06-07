"""
Stage 2 (schemas) — Pydantic group models for structured LLM extraction
                     (LLM-vs-genome ablation)

Builds the six Pydantic schemas the grouped extraction calls validate against —
one per anti-fatigue group (THEMES, TONE, SETTING, PROVENANCE, RECEPTION, VISUAL)
— directly from data/llm_schema_dimensions.json (produced by derive_schema.py).
The JSON is the single source of truth for which dimensions exist, so the models
are generated dynamically rather than hand-typed: 116 fields across 6 classes that
must stay in lock-step with the derived schema would rot the moment the schema is
re-derived. See docs/plans/llm_vs_genome_ablation_plan.md (Stage 2, schemas.py).

Each dimension becomes a `float` field constrained to [0.0, 1.0] with the genome
tags it derives from as its description — that gloss is what tells the extraction
model what the axis MEANS (e.g. `romance` ← "romance, romantic, love, love story").
`extra='forbid'` makes the schema strict (→ `additionalProperties: false` in the
generated JSON Schema), so the structured-output call cannot invent or drop fields.

NOTE on range enforcement: the Anthropic structured-output JSON Schema does NOT
support numeric `minimum`/`maximum`, so the [0,1] bound is NOT enforced on the wire.
`client.messages.parse()` validates it CLIENT-SIDE against these Pydantic `Field`
constraints and raises on violation — which is exactly the guarantee we want, just
located in the SDK rather than the API. Do not assume the wire schema clamps range.

Usage (imported by prompts.py + llm_extract.py; also runnable for a self-check):
    python llm_features/schemas.py
"""
import json
import os

from pydantic import ConfigDict, Field, create_model


# ── Paths ────────────────────────────────────────────────────────────────────

REPO_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCHEMA_JSON = os.path.join(REPO_ROOT, 'data', 'llm_schema_dimensions.json')


# ── Group → Pydantic model name ──────────────────────────────────────────────

# Stable class names per group key (used as the structured-output tool/schema name
# the model sees, so keep them descriptive and stable across re-derivations).
MODEL_NAMES = {
    'themes':     'ThemesFeatures',
    'tone':       'ToneFeatures',
    'setting':    'SettingFeatures',
    'provenance': 'ProvenanceFeatures',
    'reception':  'ReceptionFeatures',
    'visual':     'VisualFeatures',
}


# ── Build ────────────────────────────────────────────────────────────────────

def _gloss(dim: dict) -> str:
    """
    One-line semantic gloss for a dimension = its deduped genome source tags, anchor
    first (e.g. 'romance, romantic, love, love story'). This is the field description
    the extraction model reads to understand what the axis measures.
    """
    return ', '.join(t['tag'] for t in dim['genome_tags'])


def _build_model(group: dict):
    """
    Create one strict Pydantic model for a group: a float field in [0,1] per
    dimension, glossed with its genome tags, `extra='forbid'` so no field can be
    dropped or invented.
    """
    fields = {
        dim['name']: (float, Field(ge=0.0, le=1.0, description=_gloss(dim)))
        for dim in group['dimensions']
    }
    return create_model(
        MODEL_NAMES[group['key']],
        __config__=ConfigDict(extra='forbid'),
        **fields,
    )


def _load():
    """Load the derived schema and build the per-group models + flat feature order."""
    with open(SCHEMA_JSON) as f:
        schema = json.load(f)

    groups = []
    for g in schema['groups']:
        groups.append({
            'key':       g['key'],
            'title':     g['title'],
            'model':     _build_model(g),
            'dim_names': [d['name'] for d in g['dimensions']],
        })

    # Canonical flat feature order (group order, then within-group order) — the
    # order build_features.py will lay out the tensor columns in.
    feature_order = [name for g in groups for name in g['dim_names']]
    return schema, groups, feature_order


SCHEMA, GROUPS, FEATURE_ORDER = _load()


# ── Self-check ───────────────────────────────────────────────────────────────

def _selfcheck() -> None:
    """Print the built models so the schema can be eyeballed without an API key."""
    print(f"Built {len(GROUPS)} group models from "
          f"{os.path.relpath(SCHEMA_JSON, REPO_ROOT)}  "
          f"({len(FEATURE_ORDER)} total dimensions)\n")
    for g in GROUPS:
        props = g['model'].model_json_schema()['properties']
        print(f"── {g['model'].__name__:<20} ({g['title']}) — {len(g['dim_names'])} fields")
        for name in g['dim_names'][:3]:
            print(f"     {name:<22} float[0,1]  « {props[name].get('description', '')}")
        if len(g['dim_names']) > 3:
            print(f"     … and {len(g['dim_names']) - 3} more")
    assert len(FEATURE_ORDER) == len(set(FEATURE_ORDER)), "duplicate feature name!"
    print(f"\n✓ {len(FEATURE_ORDER)} unique fields, ranges [0,1], extra=forbid")


if __name__ == '__main__':
    _selfcheck()
