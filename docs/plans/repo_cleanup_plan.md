# Repo Cleanup — Tier 2 (Deferred Deploy Path)

Pre-work refactor to make the repo ready for the LLM-vs-genome ablation
(`llm_vs_genome_ablation_plan.md`). **Tiers 0 and 1 are complete** and landed on branch
`cleanup`:

- **Tier 0** — pure hygiene: non-persistent genome buffer (42 MB → ~1 MB checkpoints), unified
  `src/checkpoint.py` loader, shared `src/inference.py` user path, the `tests/` smoke test,
  version-constant consistency.
- **Tier 1** — experiment-enabling: the swappable **content slot** (`genome → content` rename +
  `content_feature_source` = `'genome'` | `'llm'` | `None`, with a `LEGACY_KEY_REMAP` shim so
  pre-rename checkpoints still load) and **corpus-namespaced artifacts** (`src/corpus.py`,
  `_phase1` suffix; `full` stays unsuffixed).

Both were verified as a no-op on the prod model by loading
`best_softmax_v2_popularity_alpha_05_20260505_182728.pth` through the remap shim and reproducing
its known numbers (MRR 0.1153; canary byte-identical).

**Why this doc still exists.** Only **Tier 2** (below) remains, and it is **gated** — in the
most likely outcome (LLM does not clearly win) it is **never executed**. So its spec must outlive
Tier 1. Delete this doc only once the Tier 2 decision gate is resolved: either Tier 2 ships, or
it is formally abandoned after a negative ablation result.

## Guiding principles (still apply to Tier 2)

- **Surgical.** Every changed line traces to an item in this plan. No drive-by "improvements"
  to adjacent code.
- **Prod checkpoint is sacred.** `best_softmax_v2_popularity_alpha_05_20260505_182728.pth` must
  still load and still produce its known eval numbers after any change. If it can't, the change
  is wrong.
- **Verification belongs to the user for any forward-pass change.** Code-only until the user
  confirms metrics (canary → eval), per CLAUDE.md.

---

## Tier 2 — Deploy path (DEFERRED behind the ablation's decision gate)

**Do not start any of this until the experiment has produced A/B/C metrics and the user has
decided LLM features are worth deploying.** In the most likely outcome (LLM does not clearly
win), none of this is ever done.

The live demo is unaffected by Tiers 0–1: `serving/` holds already-frozen artifacts with their
config embedded, and Streamlit loads those — not experiment checkpoints. Nothing in the
experiment can break the deployed app.

Gated work, only if a positive result justifies it:

- Flow the content-slot abstraction through `export.py` and `streamlit_app.py`'s model
  reconstruction. (`export.py` already resolves the renamed config via the shared
  `load_checkpoint`; the remaining work is supporting a non-genome `content_feature_source` and
  re-flowing `streamlit_app.py`, which still reconstructs the model with the legacy genome key
  names.)
- Re-export serving artifacts for the chosen model.
- Add the Streamlit side-by-side comparison tab (ablation plan Stage 7).
- Decide **swap vs. add**, retrain a real prod model on the full corpus, and deploy.
