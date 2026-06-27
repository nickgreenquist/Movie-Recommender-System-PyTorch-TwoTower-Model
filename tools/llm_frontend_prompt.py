"""
tools/llm_frontend_prompt.py — extraction prompt + structured-output schema for the
LLM conversational front-end (v1).

The LLM's ONLY job is to translate a free-text movie request into the structured query
that the two-tower model's user tower consumes (see tools/llm_frontend_probe.py and
docs/plans/plan.md). It never recommends; the trained model does retrieval. This module
holds the system prompt, the JSON schema (for Claude tool use / structured outputs), and
helpers that inject the live vocabularies from serving/feature_store.pt.

In the Claude Code test loop we feed build_system_prompt()/EXTRACTION_SCHEMA to a Haiku
subagent and run its JSON through the harness. The same prompt + schema are what the
hosted Haiku API call will use once v1 moves off the test loop — no re-tuning, because the
subagent is the same model family.
"""

import json
import os

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_FEATURE_STORE = os.path.join(_REPO_ROOT, 'serving', 'feature_store.pt')

# Current year is injected so the model can resolve relative dates ("last 10 years").
# Bump this (or wire it to the real clock at the API call site) as needed.
CURRENT_YEAR = 2026


# ── Vocab loading (the closed sets the model must select from) ───────────────
def load_vocab(fs=None):
    """Return (genres, genome_tags) from serving/feature_store.pt.

    genres: the 19 real MovieLens genres (drops the '(no genres listed)' sentinel).
    genome_tags: the full closed 1,128-tag vocabulary, the only valid genome_tags values.
    """
    if fs is None:
        fs = torch.load(_FEATURE_STORE, weights_only=False)
    genres = [g for g in fs['genres_ordered'] if g != '(no genres listed)']
    genome_tags = [fs['genome_tag_names'][t] for t in sorted(fs['genome_tag_names'])]
    return genres, genome_tags


# ── Structured-output schema (Claude tool use / OpenAI structured outputs) ───
def build_schema(genres=None, fs=None):
    """JSON schema for the extraction object. Genres are enum-constrained (only 19, cheap
    to enforce structurally); genome_tags are left as free strings — the 1,128-tag enum is
    too large to inline, and the harness already drops any tag not in the live vocab, so
    the prompt enforces the closed vocab while the schema enforces structure."""
    if genres is None:
        genres, _ = load_vocab(fs)
    genre_enum = {'type': 'string', 'enum': genres}
    return {
        'type': 'object',
        'additionalProperties': False,
        'properties': {
            'liked_items':     {'type': 'array', 'items': {'type': 'string'},
                                'description': 'Movie titles the user names positively. Use the most '
                                'specific real title; include the year when known.'},
            'disliked_items':  {'type': 'array', 'items': {'type': 'string'},
                                'description': 'Movie titles the user names negatively.'},
            'genome_tags':     {'type': 'array', 'items': {'type': 'string'},
                                'description': 'Mood/style/theme tags chosen EXACTLY from the provided '
                                'genome vocabulary. Drives Mode-2 synthesis.'},
            'liked_genres':    {'type': 'array', 'items': genre_enum,
                                'description': 'Soft genre taste preferences.'},
            'disliked_genres': {'type': 'array', 'items': genre_enum,
                                'description': 'Soft genre aversions.'},
            'hard_constraints': {
                'type': 'object',
                'additionalProperties': False,
                'properties': {
                    'year_min':       {'type': ['integer', 'null']},
                    'year_max':       {'type': ['integer', 'null']},
                    'require_genres': {'type': 'array', 'items': genre_enum,
                                       'description': 'HARD filter: every result must have all of these.'},
                    'exclude_genres': {'type': 'array', 'items': genre_enum,
                                       'description': 'HARD filter: drop any result with any of these.'},
                },
            },
        },
        'required': [],
    }


# ── System prompt ────────────────────────────────────────────────────────────
_SYSTEM_TEMPLATE = """\
You translate a person's free-text movie request into a STRUCTURED QUERY for a trained \
two-tower recommender. You are the interface, not the recommender: you never name \
recommendations or write prose to the user. You only emit the structured fields below; a \
separate trained model does the actual retrieval from those fields. Your output is consumed \
internally and discarded.

The current year is {current_year} (use it to resolve relative dates like "in the last 10 years").

Extract ONLY what the request actually states or clearly implies. Empty fields are expected \
and correct — never pad. Two modes, usually mixed:
  • Mode 1 (titles): the user names real movies. Put them in liked_items / disliked_items.
  • Mode 2 (mood/no titles): the user describes a vibe, theme, or constraints. Map the vibe \
to genome_tags.

FIELDS
- liked_items / disliked_items: real movie titles the user mentions positively / negatively.
  Use the MOST SPECIFIC real title and include the year when you know it — e.g. \
"The Lord of the Rings: The Fellowship of the Ring (2001)", NOT "Lord of the Rings"; \
"The Dark Knight (2008)", NOT "Batman". Specific titles resolve correctly; vague ones \
resolve to the wrong film.
- genome_tags: mood/style/theme tags that capture the vibe. You MUST copy them EXACTLY \
from the GENOME TAGS list below (exact spelling/hyphenation; e.g. "thought-provoking" and \
"feel-good" are valid, "thought provoking" and "feel good" are NOT). If nothing in the list \
fits, leave genome_tags empty rather than inventing a tag. Choose DISTINCTIVE descriptors of \
feel/atmosphere/style — NOT broad tags that merely restate a genre you already captured \
("comedy", "crime", "thriller", "epic", "funny", "romantic", "action" are too generic and \
pull off-target results; prefer specific ones like "quirky", "neo-noir", "atmospheric", \
"slow", "stylized"). HOW MANY depends on the mode:
    • No titles given (pure-mood request): use 3–5 specific tags — they ARE the query.
    • Titles given in liked_items: use AT MOST 2, and only for a vibe the titles don't already \
imply. The named titles are the real query; a pile of tags will drown them. Often 0 is right.
- liked_genres / disliked_genres: SOFT taste signals ("I like sci-fi", "not really into \
horror"). Choose only from the GENRES list. Soft = shapes the recommendation, does not \
hard-filter.
- hard_constraints: things the model can't express, applied as a post-filter.
    • year_min / year_max: explicit date limits ("after 2010", "90s movies" → 1990–1999, \
"recent" → roughly year_min {current_year_minus_10}). Use integers or null.
    • require_genres: HARD genre requirement ("only westerns" → ["Western"]). Every result \
must match all of these.
    • exclude_genres: HARD genre exclusion ("absolutely no horror" → ["Horror"]).
  Distinguish SOFT vs HARD by phrasing: a preference goes in liked_genres; an absolute \
("only", "must be", "nothing but") goes in require_genres; a hard "no X" goes in exclude_genres.

NOT SUPPORTED IN v1 — silently ignore these, do not invent fields for them: constraints about \
specific directors, actors, studios, or content/age ratings ("no Nolan films", "nothing with \
Tom Cruise", "rated PG"). Still capture the rest of such a request (e.g. "I'm sick of Nolan, \
show me other smart sci-fi" → liked_genres ["Sci-Fi"] + fitting genome_tags; just drop the \
Nolan exclusion).

GENRES (closed list — the only valid values for any genre field):
{genres}

GENOME TAGS (closed list — the ONLY valid values for genome_tags; copy exactly):
{genome_tags}

Return ONLY the structured object. No explanation, no recommendations, no extra text.\
"""


def build_system_prompt(fs=None):
    """The full extraction system prompt with live vocabularies injected."""
    genres, genome_tags = load_vocab(fs)
    return _SYSTEM_TEMPLATE.format(
        current_year=CURRENT_YEAR,
        current_year_minus_10=CURRENT_YEAR - 10,
        genres=json.dumps(genres),
        genome_tags=json.dumps(genome_tags),
    )


# A handful of worked examples used to anchor the Haiku test loop (all genome tags below are
# verified members of the live vocab). Kept out of the system prompt to stay lightweight; the
# loop can prepend any subset it wants.
FEWSHOT_EXAMPLES = [
    ('Something funny but smart, like Hot Fuzz or The Grand Budapest Hotel',
     {'liked_items': ['Hot Fuzz (2007)', 'The Grand Budapest Hotel (2014)'],
      'genome_tags': ['witty', 'quirky', 'clever'],
      'liked_genres': ['Comedy']}),
    ('Slow, atmospheric, melancholy — the kind of film where not much happens but it sticks with you',
     {'genome_tags': ['atmospheric', 'melancholy', 'slow', 'cerebral']}),
    ('I liked Inception and Interstellar but I\'m sick of Nolan, show me other smart sci-fi',
     {'liked_items': ['Inception (2010)', 'Interstellar (2014)'],
      'liked_genres': ['Sci-Fi'],
      'genome_tags': ['cerebral', 'thought-provoking', 'mindfuck']}),
    ('A family-friendly animated adventure from the last ten years, nothing scary',
     {'liked_genres': ['Animation', 'Adventure', 'Children'],
      'hard_constraints': {'year_min': CURRENT_YEAR - 10, 'exclude_genres': ['Horror']}}),
]


if __name__ == '__main__':
    # Quick visual check of the assembled prompt + schema.
    sp = build_system_prompt()
    print(f'[system prompt: {len(sp)} chars]')
    print(sp[:1200], '...\n')
    print('[schema]')
    print(json.dumps(build_schema(), indent=2)[:800], '...')
