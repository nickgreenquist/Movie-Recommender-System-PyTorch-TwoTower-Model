"""
Stage 3 (merge) — Combine the six per-group extraction files into one per-movie dict
                  (LLM-vs-genome ablation)

For each movie that has all six group files under a given model tag, merges them into a
single flat {feature_name: score} dict (132 dims, canonical FEATURE_ORDER) written to
cache/llm_merged/<model_tag>/<movieId>.json. Model-tag-aware so the Claude-Code dry-run
('claude-code-sonnet'), the manual stopgap ('manual-opus-4-8'), and the real API runs
('claude-sonnet-4-6' / 'claude-haiku-4-5') each merge into their own namespace and never
overwrite one another.

Also runs a light cross-group consistency / calibration scan (contradictory tone pairs,
all-zero extractions) so a broken extraction is caught here, before it silently becomes a
training tensor. See docs/plans/llm_vs_genome_ablation_plan.md (Stage 3).

Usage (standalone — no API / GPU):
    python -m llm_features.merge_extractions                    # DEFAULT_TAG
    python -m llm_features.merge_extractions manual-opus-4-8    # a specific tag
"""
import json
import os
import sys

from llm_features.schemas import FEATURE_ORDER, GROUPS


# ── Paths ────────────────────────────────────────────────────────────────────

REPO_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GROUPS_DIR = os.path.join(REPO_ROOT, 'llm_features', 'cache', 'llm_groups')
MERGED_DIR = os.path.join(REPO_ROOT, 'llm_features', 'cache', 'llm_merged')

DEFAULT_TAG = 'claude-code-sonnet'


# ── Consistency scan knobs ───────────────────────────────────────────────────

# Tone pairs that should not BOTH be high in a sane extraction (mutually exclusive in
# practice). Flagged, not corrected — a few percent is LLM noise; many across the corpus
# is an extraction-setup bug worth investigating before training (plan Stage 3).
CONTRADICTIONS = [
    ('feel_good', 'bleak'),
    ('feel_good', 'disturbing'),
    ('comedic',   'disturbing'),
]
CONTRA_THRESH = 0.7


# ── Discovery / merge ─────────────────────────────────────────────────────────

def group_path(group_key: str, tag: str, movie_id: int) -> str:
    return os.path.join(GROUPS_DIR, group_key, tag, f'{movie_id}.json')


def discover_movies(tag: str) -> tuple:
    """
    Split this tag's cached movies into (complete, partial): complete = has all six
    group files (ready to merge); partial = has ≥1 but not all (skipped, surfaced so a
    half-extracted movie can't silently vanish).
    """
    per_group = []
    for g in GROUPS:
        d = os.path.join(GROUPS_DIR, g['key'], tag)
        ids = ({int(f[:-5]) for f in os.listdir(d) if f.endswith('.json')}
               if os.path.isdir(d) else set())
        per_group.append(ids)
    if not per_group:
        return [], []
    complete = set.intersection(*per_group)
    partial  = set.union(*per_group) - complete
    return sorted(complete), sorted(partial)


def merge_one(tag: str, movie_id: int) -> dict:
    """Merge a movie's six group files → flat dict in canonical FEATURE_ORDER."""
    merged = {}
    for g in GROUPS:
        blob = json.load(open(group_path(g['key'], tag, movie_id)))
        merged.update(blob['features'])
    # Reorder to canonical order; default any absent dim to 0.0 (defensive — the
    # structured schema should already guarantee every field is present).
    return {name: float(merged.get(name, 0.0)) for name in FEATURE_ORDER}


# ── Orchestrator ──────────────────────────────────────────────────────────────

def run(tag: str) -> None:
    complete, partial = discover_movies(tag)
    print(f"Merging tag '{tag}': {len(complete)} complete movies"
          + (f", {len(partial)} partial (missing ≥1 group, skipped)" if partial else ""))
    if partial:
        print(f"   partial movieIds: {partial[:20]}{' …' if len(partial) > 20 else ''}")

    out_dir = os.path.join(MERGED_DIR, tag)
    os.makedirs(out_dir, exist_ok=True)

    n_contra = n_allzero = 0
    contra_examples = []
    for mid in complete:
        feats = merge_one(tag, mid)
        with open(os.path.join(out_dir, f'{mid}.json'), 'w') as f:
            json.dump(feats, f, indent=2)

        if all(v == 0.0 for v in feats.values()):
            n_allzero += 1
        for a, b in CONTRADICTIONS:
            if feats.get(a, 0) >= CONTRA_THRESH and feats.get(b, 0) >= CONTRA_THRESH:
                n_contra += 1
                if len(contra_examples) < 10:
                    contra_examples.append((mid, a, b, feats[a], feats[b]))

    print(f"\n✓ Wrote {len(complete)} merged dicts → {os.path.relpath(out_dir, REPO_ROOT)}")
    print(f"   all-zero extractions (likely failures): {n_allzero}")
    print(f"   contradictory tone pairs (≥{CONTRA_THRESH} on both): {n_contra}")
    for mid, a, b, va, vb in contra_examples:
        print(f"      movie {mid}: {a}={va:.2f} & {b}={vb:.2f}")


if __name__ == '__main__':
    tag = next((a for a in sys.argv[1:] if not a.startswith('-')), DEFAULT_TAG)
    run(tag)
