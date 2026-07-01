"""
src/llm_frontend_extraction.py — hosted LLM extraction call for the conversational front-end (v1).

Turns a free-text utterance into the structured extraction object that src/llm_frontend.py
consumes, via a single forced-tool-use call to a small, fast hosted Claude model (Haiku). This is
the ONLY component that talks to the Anthropic API; everything downstream — title resolution,
Mode-2 anchor synthesis, retrieval, post-filtering — is the local trained two-tower model. The LLM
is the interface, never the recommender, and its output is consumed internally and never shown to
the user (see docs/llm_frontend/llm_frontend_plan.md → "Critical Design Constraint").

Structured output is enforced via tool use: build_schema() is passed as a tool input_schema and the
tool is forced with tool_choice, so the model must return a parseable object with the expected
fields — the single most important protection (the pipeline never breaks on malformed output, plan
§Protections). max_tokens is capped (~300) since the extraction JSON is tiny.

The prompt + schema were tuned against a Haiku subagent in the in-repo Claude Code test loop, so the
same model family here transfers with no re-tuning (docs/llm_frontend/llm_frontend_plan.md → "v1 Build Handoff").
"""

from src.llm_frontend_prompt import build_schema, build_system_prompt

# Same model FAMILY the extraction prompt was validated against in the test loop, so the tuned
# prompt transfers directly. Alias (no date suffix) per the Anthropic model-id guidance.
DEFAULT_MODEL = 'claude-haiku-4-5'
MAX_TOKENS    = 300          # extraction JSON is tiny; caps cost / runaway generations
_TOOL_NAME    = 'emit_query'


def extract_query(utterance, fs=None, *, api_key=None, model=DEFAULT_MODEL, max_tokens=MAX_TOKENS):
    """Call the hosted model once and return the extraction dict (the same shape
    src.llm_frontend.recommend consumes). Raises on API / auth / transport errors — the caller
    handles user-facing messaging.

    utterance:  the user's free-text request.
    fs:         the already-loaded feature_store dict, so the prompt + schema reuse the live vocab
                instead of re-reading serving/ (the Streamlit tab passes its cached art.fs).
    api_key:    explicit key; when None the SDK resolves ANTHROPIC_API_KEY from the environment.
    """
    import anthropic  # lazy: only the hosted path needs the SDK installed

    system_prompt = build_system_prompt(fs)
    schema = build_schema(fs=fs)

    client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system_prompt,
        tools=[{
            'name': _TOOL_NAME,
            'description': 'Record the structured query extracted from the user\'s movie request.',
            'input_schema': schema,
        }],
        tool_choice={'type': 'tool', 'name': _TOOL_NAME},  # force the tool → guaranteed parseable
        messages=[{'role': 'user', 'content': utterance}],
    )

    for block in response.content:
        if block.type == 'tool_use' and block.name == _TOOL_NAME:
            return dict(block.input)
    # Forced tool_choice guarantees a tool_use block; defensive for an unexpected stop (e.g. refusal).
    raise ValueError('LLM returned no extraction (no tool_use block)')
