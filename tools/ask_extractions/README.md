# Ask-tab example-chip extractions (committed, canonical)

One `<id>.json` per query in `tools/ask_examples_spec.py` (`all_entries()` — 9 roots + 53 shown
children + 5 backburner = 67). Each file is the **extraction object** the LLM front-end produces
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

## Adding / changing an entry — the apply loop (HARD RULE)
Only ever modify the **prompt** (`query` in `tools/ask_examples_spec.py`, plus `label` for relabels) —
**never hand-write the extraction JSON**. A user copy-pasting a pill's query into the input bar should
reproduce the board; that honesty was verified pill-by-pill before launch (2026-07-09).
1. Edit `tools/ask_examples_spec.py` (id/label/query).
2. Produce that entry's extraction with no key via the `/trace` harness — one disposable Haiku subagent
   per query (avoids context bleed): it runs `python tools/llm_frontend_trace.py --emit-prompt`, applies
   that system prompt to the query, and writes the JSON here as `<id>.json`.
3. `python tools/gen_ask_examples.py --only <id>` to regenerate just that board; eyeball the top-10.

Never run `gen_ask_examples.py --live` — it re-extracts every pill and destroys the frozen
extractions. Grade pill-vs-live drift read-only with `tools/ask_live_vs_frozen.py` (Haiku is
non-deterministic even at temperature 0 — use `--k 3` before trusting a per-pill delta). Both the
generator and the grader resolve keywords against `serving/` alone (`Serving(serving_only=True)`) —
see the TRAIN/SERVE SKEW warning in CLAUDE.md.

## Routing learnings (why the queries are phrased the way they are)
The extraction route the *phrasing* triggers decides the board — same theme, different route,
different result ("ancient Rome" → Rome-the-city / Italian cinema, but "roman times. ancient rome."
→ antiquity).
- **Routes that work / are showcaseable:** anchor ("like X") · person (prolific directors only —
  Kurosawa/Scorsese/Allen resolve; Miyazaki/Kon land ≤3 films) · genre · year fence · keyword topic
  (resolver hard-filter) · exclude-genre · max/min rating.
- **Routes that fail / fold:** place/city mostly, mood-only, language. Anchorless niche keywords are
  dangerous ("katana" → 6 films corpus-wide); compound keyword terms die in the resolver.
- **Traps:** "scientists" floods sci-fi (say "mathematicians, professors"); "real"/"biopic" phrasing
  trips the based-on-true-story filter; a bare "dramas" head-noun wants an explicit Drama genre gate;
  seed anchors also *remove* those exact films from the visible board (they count as watched history).
