# Repo Cleanup Plan

Pre-work refactor to make the repo healthier in general and ready for the
LLM-vs-genome ablation (`llm_vs_genome_ablation_plan.md`) in particular.

This plan is **disposable** — execute it, verify, commit, then delete it.

## Why now

The codebase is in good shape but carries a few load-bearing assumptions that
will bite the moment we train more than one model variant. Most of the items
here are pure hygiene worth doing regardless of the LLM project; a few are
specifically what the ablation needs. They are tagged by **blast radius** so the
safe ones can land immediately and the prod-path ones stay gated.

## The framing decision (read this first)

**Classify every change by whether it can move an eval number.**

- Most items here cannot — they are deduplication, file layout, tests, and
  storage format. Safe to land and verify mechanically.
- A pure **rename** of the content slot is a no-op on the model math, so it is
  verified by *"load the prod checkpoint, run eval, confirm metrics match the
  known PROD numbers"* — **not** a retrain.
- Only a change that alters the forward pass needs the full
  train → canary → eval gate from CLAUDE.md.

Each item below states which verification applies.

## Guiding principles

- **Surgical.** Every changed line traces to an item in this plan. No drive-by
  "improvements" to adjacent code.
- **Prod checkpoint is sacred.** `best_softmax_v2_popularity_alpha_05_20260505_182728.pth`
  must still load and still produce its known eval numbers after every Tier 0–1
  change. If it can't, the change is wrong.
- **Verification belongs to the user for any forward-pass change.** Tier 1's
  content-slot work gets the code-only treatment until the user confirms metrics.

---

## Tier 0 — Pure hygiene (no eval-number change, land anytime)

These touch storage format, duplication, layout, and docs. None changes the
model's math. Verification: imports compile, shapes match, and the prod
checkpoint still evals to its known numbers.

### 0.1 — Make `genome_context_buffer` non-persistent (42 MB → ~1 MB checkpoints)

**Problem.** `model.py:65` registers `genome_context_buffer` with the default
`persistent=True`. It is a `(9376, 1128)` float32 matrix ≈ 42.3 MB, **identical
across every checkpoint**, saved into all of them. Verified on disk: the prod
checkpoint is 42 MB and `saved_models/` is 4.2 GB across 184 checkpoints —
almost entirely duplicated copies of this one matrix. The other three context
buffers (`genre`, `tag`, `year`) are already `persistent=False` and rebuilt from
`FeatureStore` on load; genome is the lone exception.

**Fix.** Flip genome to `persistent=False` (it is already rebuilt in
`train.py:build_model` and `streamlit_app.py:load_artifacts`, so nothing else
changes). New checkpoints drop to ~1 MB.

**Caveat — old checkpoints.** Existing 42 MB checkpoints have the buffer *in*
their state_dict; loading them after this change yields an "unexpected key"
unless loaded with `strict=False` or the key is dropped on load. Add a one-line
shim in the shared loader (see 0.2) that discards `genome_context_buffer` from
incoming state_dicts before `load_state_dict`. Verify: prod checkpoint loads and
evals to known numbers; a freshly saved checkpoint is ~1 MB.

**Also fix the docs lie.** CLAUDE.md states "Checkpoints are weights-only
(~1MB)" — currently false (42 MB). Correct it once this lands.

### 0.2 — Unify and harden `_resolve_config`

**Problem.** The state-dict-shape → config logic is copy-pasted verbatim in
`evaluate.py:531` and `export.py:48`. Both hardcode
`sd['item_genome_tag_tower.0.weight']`. **A no-content-slot model (ablation
Model C) has no such key → both crash with KeyError.**

**Fix.** Extract one `resolve_config_from_state_dict(sd)` (suggested home:
`src/model.py` or a small `src/config.py`) and have both callers use it. Make the
content-slot dims **optional**: if the genome/content tower keys are absent,
leave the slot disabled rather than KeyError. Fold the 0.1 buffer-drop shim into
the same shared loader so there is exactly one place that knows how to turn a
checkpoint into a model.

Verification: prod checkpoint resolves + loads identically; a synthetic
no-content state_dict resolves without error.

### 0.3 — Extract the shared inference path into `src/inference.py`

**Problem.** "Build a user embedding from a list of liked/disliked titles" is
copy-pasted across `evaluate.py:247` (`_build_user_embedding`) and
`streamlit_app.py:111`, both labeled "mirrors canary exactly." They are kept in
sync by hand. The ablation adds a comparison consumer later — a third copy.

**Fix.** Move the title-list → user-embedding logic into `src/inference.py` and
have canary, eval, and Streamlit import it. One source of truth for the user
path. Verification: canary output for the prod checkpoint is byte-identical
before/after.

### 0.4 — Add a minimal smoke test

**Problem.** No tests exist. The ablation's entire validity rests on A/B/C being
identical except the content slot ("identical training is the experiment").

**Fix.** Add `tests/test_model_shapes.py`: build the model, run one forward pass,
assert output shapes and L2-norm. Once Tier 1 lands, extend it to build all
content-slot variants and assert they differ **only** in the slot. Cheap
insurance against a confounded result. Verification: test passes.

### 0.5 — Version-constant consistency

**Problem.** Versioning is ad-hoc: `main.py:23` sets `VERSION='v1'` for features,
but `dataset.py` save/load default to `'v2'`, and `fetch_posters`/export default
to `'v1'`. It works today only because the right strings happen to line up.

**Fix.** Make the version source explicit and single (pass it through, or name
the constants per-artifact so the coupling is visible). No behavior change.
Verification: pipeline runs end-to-end on the existing data.

---

## Tier 1 — Experiment-enabling (touches training/eval path)

Needed for the ablation. A pure rename does not change the math, so verify by
loading the prod checkpoint (via the 0.2 remap shim) and confirming eval metrics
match the known PROD numbers — not a retrain. Any change that *does* alter the
forward pass gets the full train → canary → eval gate.

**Note — the prod checkpoint is NOT an experiment baseline.** The ablation
retrains Model A (genome) fresh at `alpha=0`; the deployed prod checkpoint
(`alpha=0.5`) is never used as a comparison point (see ablation plan Stage 5 /
Phase 2). The remap shim still earns its keep for two reasons: (1) it is how we
*verify the rename is a no-op* — load prod, eval, confirm metrics unchanged — and
(2) it keeps the live Streamlit demo loadable on the existing frozen artifacts.
Do not delete the shim thinking it is unused.

### 1.1 — Introduce a swappable content slot

**Problem.** The "content feature" is hardcoded as the proper noun `genome`,
end to end. It feeds two towers — `item_genome_tag_tower` (`model.py:89`) and
`user_genome_context_tower` (`model.py:113`) — both off the same
`genome_context_buffer`. There is no way to express "the content slot is filled
by X" or "there is no content slot."

**Key subtlety — genome wears two hats.** It is (a) the swappable content slot
under test, and (b) a first-class product feature with its own probes
(`probe_genome_tag`, `probe_genome_context`), the Streamlit "Explore Genome"
tab, and the canary anchor system (`_get_anchor_titles`). **The abstraction must
only generalize hat (a); hat (b) stays genome-specific.** A blanket
genome→content rename that also renames the probes/anchors is wrong.

**Fix — DECIDED: Option 1 (Clean Rename + Remap Shim).** Rename the swappable
slot from `genome` to `content` so LLM features never live in a thing named
"genome." `build_model` takes a `content_feature_source` (`'genome'` |
`'llm'` | `None`) and width, and omits the towers entirely when `None` (Model C),
adapting the concat dims accordingly. The genome *tooling* (hat b) is untouched.

**Renames in `model.py`:**

| Old | New |
|---|---|
| `genome_context_buffer` | `content_context_buffer` |
| `item_genome_tag_tower` | `item_content_tower` |
| `user_genome_context_tower` | `user_content_tower` |

**Config-key renames (not in the state_dict, so no compat risk — but rename for
consistency, or you get a confusing half-rename):**

| Old | New |
|---|---|
| `item_genome_tag_embedding_size` | `item_content_embedding_size` |
| `user_genome_context_embedding_size` | `user_content_embedding_size` |

Plus the new `content_feature_source` config key.

**State-dict remap shim** (folded into the 0.2 loader, alongside the 0.1
persistent-buffer drop). Verified against the actual prod state_dict — these four
weight keys exist and remap cleanly; `content_context_buffer` is dropped (now
non-persistent):

```python
LEGACY_KEY_REMAP = {
    'item_genome_tag_tower.0.weight':     'item_content_tower.0.weight',
    'item_genome_tag_tower.0.bias':       'item_content_tower.0.bias',
    'user_genome_context_tower.0.weight': 'user_content_tower.0.weight',
    'user_genome_context_tower.0.bias':   'user_content_tower.0.bias',
}
# On load: drop 'genome_context_buffer' (0.1), then apply LEGACY_KEY_REMAP.
```

**Keep genome-named (hat b — do NOT rename):** `probe_genome_tag`,
`probe_genome_context`, `build_user_genome_context`, the Streamlit
"Explore Genome" tab, the canary anchor system (`_get_anchor_titles`,
`USER_TYPE_TO_GENOME_TAGS`), and the genome vocab fields in `FeatureStore`
(`genome_tag_ids`, `genome_tag_to_i`, `genome_tag_names`). These are a real
product feature, not the swappable slot.

**Forethought, not features.** The plan currently lists "LLM on top of genome"
as a non-goal, but the user has floated "added" as a possible *outcome*. Design
the slot so it is **not hostile** to a future second content slot — but do
**not** build the second slot now (YAGNI).

### 1.2 — Namespace experiment artifacts by corpus

**Problem.** `.gitignore` shows the current plan for the reduced-vs-full corpus
split: `data_4k_backup/`, `saved_models_4k_backup/`, `serving_4k_backup/` — i.e.
**swapping whole directories back and forth.** No provenance; trivially easy to
train against the wrong corpus and silently invalidate the ablation. This is the
single most likely source of a "which corpus was that?" mistake.

**Fix.** Namespace artifacts by corpus in the **filename** (e.g. `_phase1` /
`_full` suffix on parquets, datasets, and checkpoints) so both corpora coexist
on disk with clear provenance, instead of directory-swapping. Verification:
Phase 1 and full artifacts can sit side by side and the loader picks the right
one by name.

---

## Tier 2 — Deploy path (DEFERRED behind the ablation's decision gate)

**Do not start any of this until the experiment has produced A/B/C metrics and
the user has decided LLM features are worth deploying.** In the most likely
outcome (LLM does not clearly win), none of this is ever done.

The live demo is unaffected by Tiers 0–1: `serving/` holds already-frozen
artifacts with their config embedded, and Streamlit loads those — not experiment
checkpoints. Nothing in the experiment can break the deployed app.

Gated work, only if a positive result justifies it:

- Flow the content-slot abstraction through `export.py` and
  `streamlit_app.py`'s model reconstruction.
- Re-export serving artifacts for the chosen model.
- Add the Streamlit side-by-side comparison tab (ablation plan Stage 7).
- Decide **swap vs. add**, retrain a real prod model on the full corpus, and
  deploy.

---

## Resolved decision — content-slot refactor scope (Tier 1.1)

**DECIDED: Option 1 — Clean Rename + Remap Shim.** Rename the swappable slot
genome→content with a state_dict key-remap so legacy checkpoints still load.
Chosen so LLM features never live inside variables/buffers/modules named after
"genome" — keeps the codebase readable as the experiment grows. Full mechanics
in Tier 1.1 above. (Rejected: "config-only, keep names" — leaves LLM features in
a `genome_tower`, the exact confusion we want to avoid.)

---

## Suggested order

1. Tier 0 (all) — independent, safe, immediately valuable. Commit per item.
2. Tier 1 — code-only (Option 1 rename + corpus namespacing), then the user
   verifies the prod checkpoint still evals to known numbers (rename is a no-op)
   or runs the full gate (any forward-pass change).
3. Begin the ablation (`llm_vs_genome_ablation_plan.md`, Stage 0).
4. Tier 2 — only after the ablation's decision gate, only on a positive result.
