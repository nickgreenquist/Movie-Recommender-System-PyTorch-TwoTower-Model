# Tier 1 Handoff (read with `repo_cleanup_plan.md`)

Short addendum for whoever starts **Tier 1** of the repo cleanup. The full spec is
still `repo_cleanup_plan.md` (sections **Tier 1.1** swappable content slot, **Tier 1.2**
corpus-namespaced artifacts, and the **Resolved decision** = Option 1 clean rename + remap
shim). This file only records what changed in **Tier 0** that the plan was written *before*,
plus the verification recipe. Delete both docs once Tier 1 lands.

## Status

- **Tier 0 is done, verified, committed, and pushed** — branch `cleanup` (6 commits:
  `Tier 0.1`–`0.5` + a PROD canary snapshot refresh). Eval (MRR 0.1153) and canary are
  byte-identical to the prod checkpoint; the refactor is a proven no-op.
- Start Tier 1 on top of `cleanup` (or branch from it).

## Tier-0 deltas that change how Tier 1 is done

The plan predates these. Account for them:

1. **`src/checkpoint.py` now exists and is the single checkpoint→model loader.** This is
   where the plan's `LEGACY_KEY_REMAP` belongs. It already has:
   - `load_checkpoint(path)` → `(config, state_dict)`; already drops
     `LEGACY_NONPERSISTENT_BUFFERS = ('genome_context_buffer',)` from incoming state_dicts.
   - `resolve_config_from_state_dict(sd)` — **already makes the content-slot dim reads
     optional** (guards `item_genome_tag_tower.0.weight` / `user_genome_context_tower.0.weight`
     with `in sd`). Tier 1 renames those guarded reads to the new key names and teaches it
     `content_feature_source=None`.
   - **Loader order for Tier 1:** in `load_checkpoint`, apply `LEGACY_KEY_REMAP` **first**
     (so old checkpoints' old key names become the new names), **then** drop buffers
     (rename `genome_context_buffer`→`content_context_buffer` in the drop set), **then**
     `resolve_config_from_state_dict`.
2. **`genome_context_buffer` is already `persistent=False`** (Tier 0.1). After the rename it
   becomes `content_context_buffer`, still non-persistent — update the drop set to the new name.
   The plan's "0.1 buffer-drop shim" is already implemented; Tier 1 just renames it.
3. **`export.py` no longer reads the tower weight keys** — it went through `load_checkpoint`
   in Tier 0. The plan's old inventory listing `export.py:52,61` is stale; export needs **no**
   rename touch (it gets the renamed config via the shared loader).
4. **`src/inference.py` (shared user path) does NOT reference the slot symbols by name** — it
   calls `model.user_embedding(...)`, which uses the towers internally. The model-internal
   rename doesn't touch inference.py.
5. **Version-constant pattern for Tier 1.2 already exists:** `src/features.FEATURES_VERSION`
   and `src/dataset.DATASET_VERSION`. Corpus namespacing (`_phase1` / `_full`) should follow
   that single-constant-per-artifact pattern, not directory-swapping.
6. **Smoke test exists** (`tests/test_model_shapes.py`) and passes a `genome_context_buffer`
   to the constructor. Per plan 0.4, **extend it** to build all content-slot variants
   (`genome` / `llm` / `None`) and assert they differ **only** in the slot — that test is the
   guard against a confounded ablation.

## Blast radius — files referencing the swappable-slot symbols (hat a, to rename)

Verified via `git grep` on current `cleanup` (line numbers drift — re-grep, don't trust these):

- `src/model.py` — the towers + buffer + concat dims (the core rename).
- `src/train.py` — `build_model` constructs the buffers/towers; `get_config` keys.
- `src/checkpoint.py` — the two guarded key reads + the buffer drop set (+ new `LEGACY_KEY_REMAP`).
- `src/evaluate.py` — `print_model_summary`/probe wiring that names the towers.
- `streamlit_app.py` — model reconstruction (Tier 2-gated for serving, but the symbol names appear).
- `tests/test_model_shapes.py` — constructor arg name.

Symbols to rename (per plan): `genome_context_buffer`→`content_context_buffer`,
`item_genome_tag_tower`→`item_content_tower`, `user_genome_context_tower`→`user_content_tower`,
config keys `item_genome_tag_embedding_size`→`item_content_embedding_size`,
`user_genome_context_embedding_size`→`user_content_embedding_size`; add `content_feature_source`.

## Do NOT rename (hat b — genome as a product feature)

These are a real feature, not the swappable slot. Leave genome-named:
`probe_genome_tag`, `probe_genome_context`, `build_user_genome_context`, the Streamlit
"Explore Genome" tab, the canary anchor system (`_get_anchor_titles`,
`USER_TYPE_TO_GENOME_TAGS`), and the FeatureStore genome vocab fields
(`genome_tag_ids`, `genome_tag_to_i`, `genome_tag_names`). Present in: `src/dataset.py`,
`src/evaluate.py`, `src/export.py`, `src/features.py`, `streamlit_app.py`. A blanket
genome→content rename that also renames these is **wrong**.

## No-op verification recipe (for the rename)

A pure rename doesn't change the math, so verify by **loading the prod checkpoint through the
remap shim and confirming the metrics match** — NOT a retrain. Prod checkpoint:
`saved_models/best_softmax_v2_popularity_alpha_05_20260505_182728.pth`.

```bash
# Eval — expect MRR 0.1153 etc.; runs write the non-prefixed <stem>.txt (won't clobber PROD_)
python main.py eval saved_models/best_softmax_v2_popularity_alpha_05_20260505_182728.pth
diff eval_results/PROD_<stem>.txt eval_results/<stem>.txt      # must be identical

# Canary — must be byte-identical to the refreshed PROD snapshot
python main.py canary saved_models/best_softmax_v2_popularity_alpha_05_20260505_182728.pth
diff canary_results/PROD_<stem>.txt canary_results/<stem>.txt  # must be identical
```

`PROD_`-prefixed files are the tracked references; runs write the bare `<stem>.txt`, so they
don't overwrite the references. Back up references to `/tmp` first if paranoid.

Anything that **alters the forward pass** (the `content_feature_source=None` Model C path, or
any accidental change to init/concat order during the rename) gets the **full
train→canary→eval gate** per CLAUDE.md — code-only, then wait for the user's numbers before
committing.

## Process

- Code-only for forward-pass-affecting changes; the user runs train/canary/eval and confirms
  before anything is committed (CLAUDE.md working-style contract).
- Do **not** build a second content slot now (YAGNI) — just keep the abstraction from being
  hostile to one.
- Tier 2 (export/Streamlit serving, comparison tab, re-export, deploy) stays **gated** behind
  the ablation's decision gate — do not start it during Tier 1.
