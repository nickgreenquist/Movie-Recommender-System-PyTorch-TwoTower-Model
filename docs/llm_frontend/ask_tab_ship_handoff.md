# Ask-Tab Ship Handoff — prompt caching + request caps, then deploy (written 2026-07-05)

**Mission:** make the Ask tab's hosted Haiku call cheap and bill-safe (prompt caching + a small
global daily cap), then ship the tab on the deployed Streamlit demo. Nothing else.

**Usage profile this is sized for (Nick, 2026-07-05):** a demo link shared with a few important
people per day, or a live walkthrough in an interview (in person / Zoom). No public promotion, no
Reddit/HN traffic expected. The **Anthropic Console limit on the key's workspace is the real bill
protection**; code-side caps are polite in-app friction, not a security boundary.

---

## Locked decisions — do not relitigate

- **NO vocab trimming, NO prompt edits, NO schema edits.** The genome vocab stays inlined in the
  system prompt. Nick explicitly rejected the vocab lever (2026-07-05) — quality was hard-won and
  demo-scale cost doesn't justify re-validation. Do not touch `src/llm_frontend_prompt.py`.
- **Caching = 5-minute default TTL**, not 1h (2× write premium never pays off at this cadence).
- Keep `_LLM_SESSION_CAP = 20` as is; **add** a global daily cap alongside it.
- No retraining, no eval, no export — this is serving-side only and the model is unchanged.

## Cost audit facts (verified 2026-07-05, this repo)

- One API call per request: `src/llm_frontend_extraction.py:extract_query` → `claude-haiku-4-5`,
  forced tool use, `max_tokens=300`. It is the only component that talks to the Anthropic API.
- Payload today: 46,079 chars/request = 37,694 system prompt + 8,385 tool schema ≈ ~12–13k input
  tokens ≈ **$0.013/request, ~$4/mo at 10 req/day**. **No `cache_control` anywhere yet.**
- With caching, repeat calls within 5 min pay 0.1× input (~$0.002/request); the first call in a
  session pays 1.25× (~$0.016). An interview demo (10–20 queries in minutes) becomes ~$0.04 total.
- Haiku 4.5 minimum cacheable prefix is 4096 tokens — we're ~3× above it, so caching engages.

## Work item 1 — prompt caching in `extract_query`

File: `src/llm_frontend_extraction.py` (call site is ~line 45). Change `system=system_prompt` to a
one-block list carrying the cache marker:

```python
system=[{
    'type': 'text',
    'text': system_prompt,
    'cache_control': {'type': 'ephemeral'},   # 5-min TTL; tools render before system,
}],                                            # so this one breakpoint caches tools+system together
```

Notes:
- Request render order is `tools` → `system` → `messages`, so a single breakpoint on the system
  block caches the 8.4k-char tool schema too. One breakpoint is sufficient.
- The prompt is already byte-deterministic across calls (sorted vocab, constant `CURRENT_YEAR`,
  stable genre order from `feature_store.pt`) — no silent invalidators to fix.
- Add one line inside `extract_query` after the call: `LAST_USAGE = response.usage` (module-level
  global) so verification and any future debug UI can read cache stats without changing the return
  shape. Keep it to that — no other signature changes.
- If unsure of current SDK syntax, load the `claude-api` skill (Prompt Caching section) first.

## Work item 2 — global daily cap in the Streamlit app

File: `streamlit_app.py`. Existing pieces: `_LLM_SESSION_CAP = 20` (line ~59, per-browser-session,
resets on refresh — keep), key resolution `_anthropic_api_key()` (~line 689), the gate inside
`tab_ask` (~line 774) where `calls >= _LLM_SESSION_CAP` is checked.

Add a **global** counter shared across all sessions in the container:

```python
_LLM_DAILY_CAP = 60   # global across ALL visitors; Console spend limit is the hard backstop

@st.cache_resource
def _llm_daily_budget():
    return {'lock': threading.Lock(), 'date': None, 'count': 0}
```

In `tab_ask`, next to the session-cap check: under the lock, roll `date` to
`datetime.date.today().isoformat()` (reset `count` on change), refuse when `count >=
_LLM_DAILY_CAP` with a friendly message (e.g. "Today's demo budget is used up — the Recommend tab
works without the LLM, or come back tomorrow."), and increment on a successful extraction (mirror
where `ask_calls` increments). Sizing: 60/day ≈ one full interview demo plus several sharees;
worst case ~$0.80/day uncached, ~$0.25/day cached. Counter resets on app restart/redeploy —
acceptable; the Console limit is the backstop. Match house code style (docstring headers, comment
banners) and keep the diff surgical.

## Work item 3 — ship it

Pre-verified facts (2026-07-05, don't re-derive): `anthropic==0.105.2` already in
`requirements.txt`; committed `serving/feature_store.pt` (commit 90018b0, 2026-07-02) **already
contains the `facets` bake** (people + keyword filters work from it); `streamlit_app.py` reads only
`serving/` (never `data/`); the Ask tab is fully wired with a graceful no-key notice; working tree
clean at 1217f45 (pushed). **No re-export needed.**

1. **Smoke locally, no key:** `streamlit run streamlit_app.py` → Ask tab must show the
   setup notice and never call the API; other tabs unaffected.
2. **Verify caching with a real key** (Nick runs, or sets `ANTHROPIC_API_KEY` for you): two
   identical `extract_query` calls within 5 min from a scratch script; assert call 1 shows
   `usage.cache_creation_input_tokens > 0` and call 2 shows `cache_read_input_tokens` ≈ 10k+ with
   small `input_tokens`. Costs ~$0.02 total. Also confirm the extraction dict still parses and
   `recommend()` renders (one `/trace`-style local run needs no key and covers the harness side).
3. **Commit** (do NOT push in the same command; ask Nick before pushing). Suggested message:
   `feat(llm-frontend): ask-tab prompt caching + global daily cap (ship prep)`.
4. **Push after approval** → Streamlit Community Cloud auto-redeploys from main.
5. **Nick's dashboard actions** (remind him, he does these — not code):
   - Create a **dedicated API key** for the demo, ideally in its own Console workspace.
   - Set that workspace's **spend limit** (e.g. $5/month) — this is the actual bill guarantee —
     and optionally dial its rate limits down.
   - Paste the key into Streamlit Cloud → app → Settings → Secrets as `ANTHROPIC_API_KEY`.
6. **Live verify on the deployed app:** one query renders recs + the "how your request was
   interpreted" expander; repeat the query to confirm snappier response (cache read); confirm the
   daily/session cap messages by temporarily lowering the constants ONLY if trivial, otherwise
   skip — don't burn real quota proving a counter works.

## Acceptance checklist

- [ ] `extract_query` sends `cache_control`; second call within 5 min reads ~10k+ tokens from cache
- [ ] `LAST_USAGE` exposes usage; no other behavior/signature change in `extract_query`
- [ ] `_LLM_DAILY_CAP` enforced across sessions with a clear message; `_LLM_SESSION_CAP` unchanged
- [ ] No edits to `src/llm_frontend_prompt.py`, the schema, or anything model/training-side
- [ ] Committed (then pushed only after Nick approves); deployed app serves the Ask tab with the
      secret set; Nick confirmed the Console spend limit exists
- [ ] Per-request cost after caching ≈ $0.002 cached / ~$0.016 first-in-session (from real usage numbers)

## Out of scope (explicitly)

Vocab trimming / curated tag subset, schema-description slimming, any prompt wording change, query
router changes, new tabs, narrative-dimension work, retraining/eval/export, abuse-hardening beyond
the caps above.

## File map

- `src/llm_frontend_extraction.py` — the one API call (edit here for caching)
- `streamlit_app.py` — `_LLM_SESSION_CAP` ~line 59, `_anthropic_api_key()` ~line 689, `tab_ask`
  ~lines 745–800 (edit here for the daily cap)
- `src/llm_frontend_prompt.py` — DO NOT EDIT (prompt + schema, quality-tuned)
- `docs/llm_frontend/ask_tab_step4_handoff.md`, memory `project_ask_tab_cost` — background
- Full cost analysis + rejected alternatives: memory file `project_ask_tab_cost.md`
