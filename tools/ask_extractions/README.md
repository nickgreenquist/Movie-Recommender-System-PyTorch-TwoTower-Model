# Ask-tab example-chip extractions (committed, canonical)

One `<id>.json` per query in `tools/ask_examples_spec.py` (`all_entries()` — 7 roots + 42 shown
children + 2 backburner = 51). Each file is the **extraction object** the LLM front-end produces
for that query (the "front bookend" output: liked/disliked items, genome tags, soft/hard genres,
require/exclude people, keyword concepts, year fences, rating caps, …).

## Why this dir exists
The extraction step is the **only** part of the example-chip pipeline that needs an LLM. Committing
the extractions makes `serving/ask_examples.json` reproducible **with no API key and no re-tracing** —
regen is just `recommend()` over `serving/`. Before this dir existed, the extractions were transient
and got lost, forcing a full re-trace every time. Don't let that happen again: if you add/reword a
spec entry, save its extraction here.

## Regenerate the artifact (no API key)
```bash
python tools/gen_ask_examples.py            # reads THIS dir by default → serving/ask_examples.json
python tools/gen_ask_examples.py --only r4c2,r5c1   # just some ids (merges into the artifact)
```
`recommend()` is deterministic, so the boards are stable across runs given the same `serving/` model.

## After a model promotion / re-export
The extractions stay valid (they don't depend on the model) — just re-run the command above to rebuild
the boards against the new `serving/` embeddings. No key, no subagents.

## Adding / changing an entry
1. Edit `tools/ask_examples_spec.py` (id/label/query).
2. Produce that entry's extraction with no key via the `/trace` harness — one disposable Haiku subagent
   per query (avoids context bleed): it runs `python tools/llm_frontend_trace.py --emit-prompt`, applies
   that system prompt to the query, and writes the JSON here as `<id>.json`.
3. `python tools/gen_ask_examples.py --only <id>` to regenerate just that board.

## Curator overrides (hand-tuned, keep them)
A few extractions were hand-corrected because fresh Haiku extraction is stochastic and picked a weaker
route than the traced v3 boards. Preserve these on any refresh:
- `r5c1` (time loops), `r6c3` (alien invasion), `r6c5` (wasteland): moved the theme into a **hard**
  `require_keyword_concepts` (`time loop` / `alien invasion` / `post-apocalyptic`) — resolver keys in
  `src/llm_frontend.py` `KEYWORD_CONCEPTS`.
- `r4c2` (Kurosawa): anchored on his actual films (the `require_people` person filter didn't resolve).
- `r1c3` (American animation): intentionally keeps `require_country=['American']` → a benign `fallback`
  flag (debug-panel only), because its fallback board — the Toy Story→Up canon — is the better *visible*
  board.
