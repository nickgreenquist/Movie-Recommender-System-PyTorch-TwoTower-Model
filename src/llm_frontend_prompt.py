"""
src/llm_frontend_prompt.py — extraction prompt + structured-output schema for the
LLM conversational front-end (v1).

The LLM's ONLY job is to translate a free-text movie request into the structured query
that the two-tower model's user tower consumes (see src/llm_frontend.py and
docs/llm_frontend/llm_frontend_plan.md). It never recommends; the trained model does retrieval. This module
holds the system prompt, the JSON schema (for Claude tool use / structured outputs), and
helpers that inject the live vocabularies from serving/feature_store.pt.

The prompt + schema were tuned against a Haiku subagent in the in-repo Claude Code test loop
(tools/llm_frontend_probe.py). The hosted path (src/llm_frontend_extraction.py, used by the
Streamlit "Ask" tab) feeds the SAME build_system_prompt()/build_schema() to the Claude Haiku
API — no re-tuning, because the subagent is the same model family.
"""

import json
import os

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_FEATURE_STORE = os.path.join(_REPO_ROOT, 'serving', 'feature_store.pt')

# NOTE (step 2, genome-first resolver): the closed KEYWORD_CONCEPTS list is no longer injected into
# the prompt or schema. require/exclude_keyword_concepts carry FREE topic terms, resolved
# server-side (src/llm_frontend.py:resolve_topic_term) against the genome vocab, the frozen curated
# concept core, and the raw TMDB keyword index — the LLM never needs the list.

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
            'mood':            {'type': 'array', 'items': {'type': 'string'},
                                'description': 'SOFT affect / emotional-goal phrases — how the user '
                                'wants to FEEL — as FREE natural language ("make me cry", "feel-good", '
                                '"something cozy", "uplifting", "darker", "more mature"). Unlike '
                                'genome_tags these need NOT come from the vocab; the harness maps them '
                                'to tone tags. Soft anchor, not a filter. A hard "absolutely no sad '
                                'stuff" is NOT mood — that is exclude_mood.'},
            'unsupported_notes': {'type': 'array', 'items': {'type': 'string'},
                                'description': 'LAST RESORT: short phrases for demands NO other '
                                'field can express — specific plot mechanics ("where the twist '
                                'recontextualizes everything", "where the dog dies at the end") or '
                                'craft techniques ("shot in one take"). Note the demand here instead '
                                'of dropping it silently, and still extract the rest of the request '
                                'normally. NEVER route anything a field above can express here.'},
            'out_of_domain':   {'type': 'boolean',
                                'description': 'True ONLY when the request is not asking for movie '
                                'recommendations at all (they want a video game, TV series, book, '
                                'music, or non-movie help). Leave every other field empty then. A '
                                'movie request that merely MENTIONS another medium stays in-domain.'},
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
                    'require_people':  {'type': 'array', 'items': {'type': 'string'},
                                        'description': 'HARD filter: every result must involve ALL of '
                                        'these people (matched as actor, director, or writer). Full '
                                        'names as free strings — "Tom Hanks", "Christopher Nolan".'},
                    'exclude_people':  {'type': 'array', 'items': {'type': 'string'},
                                        'description': 'HARD filter: drop any result involving any of '
                                        'these people. Full names as free strings.'},
                    'require_genome_tags': {'type': 'array', 'items': {'type': 'string'},
                                        'description': 'HARD floor: every result must genuinely carry '
                                        'ALL of these genome tags (copied EXACTLY from the genome '
                                        'vocab). Use it when the request is DEFINED by a SETTING — '
                                        '"set in Japan" -> ["japan"], "takes place in high school" -> '
                                        '["high school"]. DISTINCT from require_country (nationality). '
                                        'ALL listed tags must hold (AND semantics) — alternatives '
                                        '("like Tokyo or Paris") are exemplars, NOT demands: keep them '
                                        'in soft genome_tags. The tag MUST appear verbatim in the '
                                        'vocab — never invent one; when the mention is mild rather '
                                        'than defining, prefer soft genome_tags instead.'},
                    'exclude_genome_tags': {'type': 'array', 'items': {'type': 'string'},
                                        'description': 'HARD anti-filter: drop any result carrying any '
                                        'of these genome tags (exact vocab). For an emphatic content '
                                        'aversion — "no gore" -> ["gore"].'},
                    'exclude_mood':    {'type': 'array', 'items': {'type': 'string'},
                                        'description': 'HARD affect exclusion as FREE phrases — '
                                        '"nothing sad", "no dark depressing stuff". The harness maps '
                                        'these to tone tags and drops films that carry them.'},
                    'require_country': {'type': 'array', 'items': {'type': 'string'},
                                        'description': 'HARD PRODUCTION-nationality filter — "French", '
                                        '"South Korean", "Scandinavian", "foreign". Who MADE the film, '
                                        'NOT where it is set (a Paris-SET US film is not French — use '
                                        'require_genome_tags for setting).'},
                    'require_language': {'type': 'array', 'items': {'type': 'string'},
                                        'description': 'HARD original-language filter — "in French", '
                                        '"originally Japanese, not dubbed".'},
                    'require_attributes': {'type': 'array', 'items': {'type': 'string'},
                                        'description': 'HARD format/attribute filter — "black and '
                                        'white", "silent", "directed by women", "based on a book", '
                                        '"based on a true story", "anime", "stop motion", "independent".'},
                    'require_keyword_concepts': {'type': 'array', 'items': {'type': 'string'},
                                        'description': 'HARD content-topic filter: concrete SUBJECTS '
                                        'the film must be ABOUT, as short free lowercase noun terms — '
                                        '"movies about chess" -> ["chess"], "movies with dogs" -> '
                                        '["dog"], "submarine movies" -> ["submarine"]. The harness '
                                        'resolves each term against its content indexes; do NOT '
                                        'restrict yourself to any list. Multiple terms are '
                                        'ALTERNATIVES (a film matching ANY qualifies): "a boxing or '
                                        'MMA fighter" -> ["boxing", "mma"]. DISTINCT from '
                                        'require_genres (a topic is not a genre) and from '
                                        'require_genome_tags (a setting/place).'},
                    'exclude_keyword_concepts': {'type': 'array', 'items': {'type': 'string'},
                                        'description': 'HARD topic exclusion, same free-term shape — '
                                        '"no zombie movies" -> ["zombie"], "nothing with vampires" -> '
                                        '["vampire"].'},
                    'require_max_rating': {'type': ['string', 'null'],
                                        'description': 'HARD US content-rating ceiling — one of "G", '
                                        '"PG", "PG-13", "R", "NC-17". "nothing R-rated" -> "PG-13"; '
                                        '"kid-friendly" -> "PG". Drops anything stricter.'},
                    'require_min_rating': {'type': ['string', 'null'],
                                        'description': 'HARD US content-rating FLOOR (mirror of the '
                                        'ceiling) — one of "G", "PG", "PG-13", "R", "NC-17". A POSITIVE '
                                        'maturity ask WANTS mature content: "R-rated comedies" / "adult '
                                        'animation" / "raunchy" / "hard R" -> "R"; "adults only" / '
                                        '"NC-17" -> "NC-17". Drops anything TAMER.'},
                    'max_runtime':     {'type': ['integer', 'null'],
                                        'description': 'HARD max runtime in MINUTES ("under two hours" '
                                        '-> 120, "nothing over 90 minutes" -> 90).'},
                    'min_runtime':     {'type': ['integer', 'null'],
                                        'description': 'HARD min runtime in minutes ("no shorts", '
                                        '"at least feature length").'},
                    'min_vote_average': {'type': ['number', 'null'],
                                        'description': 'HARD quality floor, TMDB score 0-10. "actually '
                                        'good" / "critically acclaimed" / "no trash" -> ~7.0.'},
                    'require_franchise': {'type': 'array', 'items': {'type': 'string'},
                                        'description': 'HARD franchise/collection membership — "the '
                                        'Marvel movies", "James Bond films", "Star Wars". Free names.'},
                    'exclude_franchise': {'type': 'array', 'items': {'type': 'string'},
                                        'description': 'HARD franchise exclusion — "no MCU", "skip the '
                                        'sequels / the DCEU". Free names.'},
                    'require_composers': {'type': 'array', 'items': {'type': 'string'},
                                        'description': 'HARD filter: every result scored by ALL these '
                                        'composers — "scored by Hans Zimmer", "John Williams '
                                        'soundtrack". Full names.'},
                    'exclude_composers': {'type': 'array', 'items': {'type': 'string'},
                                        'description': 'HARD filter: drop any result scored by these '
                                        'composers. Full names.'},
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
feel/atmosphere/style — NOT broad tags that merely restate a genre you already captured. These \
restate-a-genre tags are FORBIDDEN in any mode (they pull off-target): "comedy", "crime", \
"thriller", "epic", "funny", "romantic", "action", "scary", "dramatic", "adventure". Prefer \
specific, evocative ones like "quirky", "neo-noir", "atmospheric", "surreal", "stylized", \
"dreamlike". COMFORT-VIBE choice: a cozy / comfort-watch / "lazy Sunday" ask wants "feel-good" / \
"heartwarming" (+ mood "cozy") — NEVER "meditative" / "dreamlike" / "slow-paced", which read \
art-house and swap a comfort list for bleak slow cinema. CONCRETE-TOPIC GUARD (critical): genome_tags \
are abstract FEEL / atmosphere / style ONLY. A word naming a concrete SUBJECT the film is ABOUT (dogs, chess, submarines, a serial \
killer, an amusement park, christmas) is never a vibe — it goes in \
hard_constraints.require_keyword_concepts as a free term, NOT in genome_tags, where it either \
dilutes the query or is silently dropped. \
HOW MANY depends on whether the user named any titles:
    • NO titles given: decide whether a hard CATEGORY head noun (a genre, a named person, or a \
concrete topic / setting / nationality / format / franchise) defines the request FIRST — that, NOT the \
mere presence of a year / runtime / rating fence, decides how to handle tags. A vibe WITH an orthogonal \
fence is STILL a vibe.
        – PURE VIBE or SOFT-GENRE TASTE (the request is defined by a FEELING / atmosphere or a soft genre \
lean, with NO hard CATEGORY head noun — no require_genres, require_people, require_genome_tags, \
require_keyword_concepts, require_country, require_language, require_attributes, require_franchise, \
require_composers. A year / runtime / rating / quality fence MAY be present — it is ORTHOGONAL and does \
NOT disqualify this branch — "I love a good romance", "something dreamy and melancholy", "big fan of war \
movies", "into crime and mystery", "scare me but nothing older than 2010", "make me cry, keep it under \
two hours"): the tags ARE the query, so ALWAYS emit 3–5 specific mood/style tags — even when you also set \
a SOFT liked_genres, AND even when you ALSO capture a year / runtime / rating fence (that fence is applied \
SEPARATELY and NEVER replaces the vibe; an implied genre stays SOFT in liked_genres, NEVER require_genres). \
A bare taste preference is a weak signal on its own, so add concrete vibe tags \
(romance → "intimate", "heartfelt", "bittersweet"; crime → "noir", "detective", "hardboiled"; war \
→ "anti-war", "visceral", "harrowing"). Never leave genome_tags empty for a pure-vibe/taste request.
        – HARD-CATEGORY request (the HEAD NOUN is a genre → require_genres, a named person → \
require_people, or a concrete topic / setting / nationality / format / franchise → its slot — "comedies \
from the 80s", "show me horror movies", "Tom Hanks movies", "recent thrillers", "submarine movies"): the \
CATEGORY is the query, not a vibe. Set the hard slot(s) plus any year_min / year_max, and emit mood tags \
ONLY for a GENUINE extra vibe stated beyond the category ("dark gritty westerns" → require ["Western"] + \
"gritty"). For a plain category(+era) request — "comedies from the 80s" → require ["Comedy"] + \
year_min 1980 / year_max 1989, NO invented tags — do NOT manufacture quirky/campy/atmospheric-type \
tags: an invented vibe tag pulls results off the required genre. When unsure here, prefer NO tag. \
CRUCIAL — a fence is NOT a category: a year / runtime / rating limit ADDED to a vibe ("scare me, nothing \
old"; "make me cry, under 90 min") does NOT move the request here — the head noun is still the VIBE, so it \
stays in the pure-vibe bullet above (emit the 3–5 vibe tags; keep any implied genre SOFT in liked_genres). \
Only a genre / person / topic HEAD NOUN belongs here. \
A named-person request ("Tom Hanks movies", "I like Tom Hanks movies" → require_people ["Tom Hanks"]) \
is likewise defined by that person, NOT a vibe — emit NO mood tags unless a genuine extra vibe is stated. \
The SAME holds for a request defined by a SETTING or genome-vocab SUBJECT ("set in Japan" → \
require_genome_tags ["japan"]; "courtroom dramas" → require_genome_tags ["courtroom"]), a \
NATIONALITY ("French films" → require_country), a concrete TOPIC ("submarine movies" → \
require_keyword_concepts ["submarine"]), or a FORMAT / RATING / RUNTIME / FRANCHISE facet: set that \
hard slot and do NOT manufacture vibe tags — the facet is the query.
    • Titles GIVEN in liked_items: the named titles are the real query and they already encode \
tone, intensity, and era — so use FEW tags (0–2), and only when they add something the titles \
don't. Decide by HOW the user frames the mood:
        – EQUATIVE ("just as dark", "same gritty vibe", "similarly intense") merely RESTATES what \
the named films already convey — emit NO tag for it. Generic restatement words ("dark", \
"gritty", "grim", "bleak", "intense", "slow", "ominous", "serious") drag results toward a bleak \
art-house cluster, so never emit them as an echo of the titles.
        – COMPARATIVE / PIVOT ("darker", "even more disturbing", "but more emotional", "but \
WHIMSICAL", "but a MUSICAL") pushes a NEW direction the titles don't already go — emit AT MOST \
1–2 PRECISE tags for it, preferring specific ones ("psychological", "surreal", "whimsical", \
"mindfuck", "neo-noir") over vague intensity words ("dark", "intense"). When in doubt, none.
- liked_genres / disliked_genres: SOFT taste signals ("I like sci-fi", "not really into \
horror"). Choose only from the GENRES list. Soft = shapes the recommendation, does not \
hard-filter.
- mood: SOFT affect / emotional-goal phrases — how the user wants to FEEL — as FREE natural \
language: "make me cry", "feel-good", "something cozy", "uplifting", "darker", "more mature", \
"something scary". Unlike genome_tags these need NOT be in the vocab (the model maps them to tone). \
Use mood for the emotional GOAL; use genome_tags for concrete style/atmosphere descriptors. A hard \
"absolutely NO sad stuff" is NOT a mood — it goes in exclude_mood.
- hard_constraints: things the model can't express, applied as a post-filter. EVERY key below \
(year_min/year_max, require_genres, require_people, require_keyword_concepts, require_genome_tags, \
require_attributes, require_country, rating/runtime/franchise, and their exclude_ siblings) MUST be \
NESTED INSIDE the hard_constraints object — NEVER emitted at the top level of the extraction. A key \
placed at the root is silently ignored and the whole filter is lost.
    • year_min / year_max: explicit date limits ("after 2010", "90s movies" → 1990–1999, \
"recent" → roughly year_min {current_year_minus_10}; "popular right now" / "what's big right now" → \
year_min {current_year_minus_3}, the tightest recency window). Use integers or null.
    • require_genres: HARD genre requirement. Use it when a genre is the PRIMARY CATEGORY being \
requested — the head noun of the ask — not just a taste aside: "only westerns" → ["Western"], \
and ALSO "comedies from the 80s" → ["Comedy"], "show me horror movies" → ["Horror"], \
"documentaries about nature" → ["Documentary"], "animated movies" → ["Animation"] (an animation ask \
is this GENRE — never the require_attributes value "animated"). Every result must contain all of these.
    • exclude_genres: HARD genre exclusion ("absolutely no horror" → ["Horror"]; "live-action", \
"no cartoons" → ["Animation"] — live-action has no facet of its own, it is the ABSENCE of animation).
    • require_people: HARD filter on PEOPLE — every result must involve ALL named people (matched \
as actor, director, or writer). Use it whenever the request names a person whose films are wanted, \
in ANY phrasing: "movies with Tom Hanks", "Tom Hanks movies", "I like Tom Hanks movies", "starring \
Denzel", "directed by Sofia Coppola", "by Tarantino", "a Scorsese film", "only Christopher Nolan". \
Emit each person's FULL NAME as a free string ("Tom Hanks", not "Hanks"; "Christopher Nolan", not \
"Nolan"). Naming a person whose films you want is ALWAYS a HARD membership constraint — never a soft \
taste, and never a genome_tag. (The model has no person concept, so this is the only way people work.)
    • exclude_people: HARD people exclusion → list the full names. "no Nolan films", "nothing with \
Tom Cruise", "I'm sick of Nolan / show me other..." → exclude_people ["Christopher Nolan"] etc.
    • require_genome_tags: HARD floor on genome tags copied EXACTLY from the GENOME TAGS list — for a \
request DEFINED by a SETTING: "set in Japan" / "takes place in Paris" / "based in New York" → \
["japan"] / ["paris"] / ["new york city"], "a movie set in high school" → ["high school"], "set on \
Mars" → ["mars"]. SETTING is NOT NATIONALITY: "set in \
Japan" → require_genome_tags ["japan"] (WHERE it takes place), but "Japanese films" / "French cinema" → \
require_country (WHO made it) — a Japan-set film can be a US production. DISCIPLINE: do NOT hard-require a \
genome tag for a passing mention or a mere atmosphere ("a bit of a Paris vibe" → SOFT genome_tags ["paris"], \
not require). Reserve require_genome_tags for an unmistakable "it MUST be set in ___". \
Multiple require_genome_tags are AND semantics — every result must carry ALL of them — so NEVER \
hard-require alternatives: "a city like Tokyo or Paris" names EXEMPLARS of a vibe (an "or", and a \
"like"), not a demand — keep such alternatives SOFT in genome_tags. When in \
doubt, prefer soft genome_tags — an over-eager hard floor empties the pool.
    • exclude_genome_tags / exclude_mood: HARD content/affect aversions. exclude_genome_tags lists EXACT \
genome tags to forbid ("no gore" → ["gore"]); exclude_mood takes FREE affect phrases ("nothing sad", "no \
dark depressing stuff"). Only for an EMPHATIC "absolutely no ___" — a mild preference is not an exclusion.
    • require_country: HARD PRODUCTION-nationality filter — "French films", "Korean cinema", "Scandinavian \
movies", "foreign" (non-English). Free strings. Region phrases work too: "Scandinavian" / "East Asian" / \
"European" ("a European film", "somewhere in Europe" → ["European"]). This is WHO PRODUCED the film, NOT \
where it is set (see require_genome_tags). "Japanese films" → require_country ["Japanese"]; "movies set \
in Japan" → require_genome_tags ["japan"].
    • require_language: HARD original-language filter — "in French", "originally Japanese, not dubbed", \
"Korean-language". Free language names.
    • require_attributes: HARD format/attribute filter — "black and white", "silent", "directed by women", \
"based on a book", "based on a true story", "anime", "stop motion", "independent / indie". Reserve \
"anime" for JAPANESE-animation asks specifically; a plain "animated / animation" ask is the Animation \
GENRE (require_genres) and "live-action" is exclude_genres ["Animation"], never an attribute.
    • require_keyword_concepts / exclude_keyword_concepts: HARD content-topic filter — concrete SUBJECTS the \
film must be ABOUT, as short free lowercase noun terms: "movies about chess" → ["chess"], "movies with \
dogs" → ["dog"], "submarine movies" → ["submarine"], "something with dinosaurs" → ["dinosaur"]; "no zombie \
movies", "nothing with vampires" → exclude_keyword_concepts ["zombie"] / ["vampire"]. The harness resolves \
each term against its content indexes, so do NOT restrict yourself to any list — but keep terms concrete \
and noun-like (the SUBJECT itself, not a sentence). Multiple require terms are ALTERNATIVES (ANY match \
qualifies): "a boxing or MMA fighter" → ["boxing", "mma"]. Precise TOPIC membership, DISTINCT \
from require_genres (a topic is not a genre) and require_genome_tags (a setting/place). \
A viewing OCCASION or season is a topic too: "a movie to watch on Christmas Eve", "something for the \
holidays" → ["christmas"]; "a Halloween-night movie" → ["halloween"]; "a summer / beach movie" → \
["summer"]; "a winter/snowbound story" → ["winter"]. A film SET IN or evoking a decade ("set in the 90s", \
"80s synthwave aesthetic") is the term "1980s"/"1990s" — but a RELEASE-era ask ("movies FROM the 80s", \
"90s comedies", "old 80s films") is year_min/year_max, NOT a topic term. "Nostalgic FOR the 90s" / \
"feels like the 90s" names the decade FEEL, so it is the SETTING term ["1990s"] too, not a release \
window; and "a western set in modern times" → ["neo western"] (a modern SETTING is never year_min). \
TERM FORM: the resolver matches exact tokens, so keep each term SHORT and CANONICAL — "space" not \
"space exploration", "election" not "political campaign", "whodunit" not "whodunit mystery", "dreams" \
not "stuck in a dream". When unsure the term is known, EMIT IT ANYWAY: an unknown term is reported and \
degrades honestly, so trying is always safe — NEVER park a concrete subject in unsupported_notes or \
demote it to a soft tag instead. A subject stays HARD even wrapped in a vibe / nostalgia / \
effect-on-viewer phrasing: "a cozy small-town vibe" → ["small town"] (+ mood "cozy"), "nostalgic for \
college life" → ["college"], "makes me want to travel the world" → ["travel"], "that movie where the \
guy is stuck in a dream" → ["dreams"]. A comedic topic ask pairs the genre gate with the topic: \
"funny stoner movies" → ["stoner"] + require_genres ["Comedy"].
    • require_max_rating: HARD US content-rating ceiling — one of "G", "PG", "PG-13", "R", "NC-17". \
"nothing R-rated" → "PG-13"; "kid-friendly" / "nothing too mature" → "PG". Drops anything stricter.
    • require_min_rating: HARD US content-rating FLOOR (mirror of the ceiling) — one of "G", "PG", "PG-13", \
"R", "NC-17". A POSITIVE maturity ask WANTS mature content, so it drops anything TAMER: "R-rated comedies" / \
"adult animation" / "raunchy" / "hard R" → "R"; "adults only" / "NC-17" → "NC-17". Do NOT confuse the two: \
"nothing R-rated" is a max ("PG-13"); "give me R-rated" / "for adults" is a min ("R").
    • max_runtime / min_runtime: HARD runtime bounds in MINUTES — "under two hours" → max_runtime 120, \
"nothing over 90 minutes" → 90, "no shorts" → a min_runtime.
    • min_vote_average: HARD quality floor, TMDB score 0–10 — "actually good", "critically acclaimed", \
"no trash" → ~7.0.
    • require_franchise / exclude_franchise: HARD franchise/collection membership or exclusion — "the \
Marvel movies" → require_franchise ["Marvel"]; "no MCU", "skip the sequels", "not the DCEU" → \
exclude_franchise. Free franchise/universe names.
    • require_composers / exclude_composers: HARD filter on the FILM-SCORE composer — "scored by Hans \
Zimmer", "John Williams soundtrack" → require_composers ["Hans Zimmer"]. Full names. (A composer is a \
person, but goes in this slot, not require_people, so an actor of the same name is never confused.)
  Distinguish SOFT vs HARD by the genre's ROLE, not just keywords: a genre that IS the thing \
being requested (the head noun — "show me comedies", "I want a western", "horror movies") is \
HARD → require_genres; a genre mentioned as a side preference ("I'm into sci-fi", "I usually \
like crime stuff") is SOFT → liked_genres. Absolutes ("only", "must be", "nothing but") are \
always HARD. A hard "no X" goes in exclude_genres. A genre you put in require_genres may also go \
in liked_genres so the model leans toward it, but a SOFT preference must NEVER go in require_genres.

SOCIAL-CONTEXT REQUESTS ("something to watch with my ___"): mine the TASTE SIGNAL embedded in the \
social frame and extract it like any other preference — "watch with my teenage son" → liked_genres \
["Action", "Adventure", "Sci-Fi"]; "my grandfather who served in the military" → \
require_keyword_concepts ["military"]; "a double date night" → mood ["romantic", "fun"] + liked_genres \
["Romance", "Comedy"]; "my cousin who loves true crime podcasts" → require_genres ["Crime"] + \
require_keyword_concepts ["true story"]. Only a PURE social frame with NO taste signal at all ("a movie \
to watch with my new roommate") stays empty — never invent a preference from the relationship alone.

NOT SUPPORTED — silently ignore these, do not invent fields for them: constraints about specific \
studios / production companies ("A24 films", "a Pixar movie" as a STUDIO) and streaming availability \
("on Netflix", "what's streaming"). Still capture the rest of such a request. (Actors, directors, \
writers, and composers ARE supported → require_people / require_composers; content ratings, runtime, \
quality, franchises, nationality, language, format, and concrete content TOPICS ARE supported → the \
hard_constraints slots above. Route each to its slot, never to genome_tags.)

OUT OF DOMAIN: if the request is not asking for MOVIE recommendations at all (they want a video \
game, a TV series, a book, music, or help unrelated to movies), set the top-level out_of_domain: \
true and leave every other field empty. A movie request that merely MENTIONS another medium \
("a movie like The Last of Us game") is still IN domain — extract it normally.

CAN'T-EXPRESS RESIDUE (top-level unsupported_notes, LAST resort): a demand NO field above can \
express — a specific plot mechanic or a craft technique — goes into unsupported_notes as a short \
phrase instead of being dropped silently, and you STILL extract every expressible part of the \
request normally: "movies where the dog dies at the end" → require_keyword_concepts ["dog"] + \
unsupported_notes ["the dog dies at the end"]; "shot entirely in one take" → unsupported_notes \
["shot in one take"]. CHECK the fields first — never park something a field CAN express: a \
twist/tone ask is genome_tags ("a big twist" → ["twist ending"]), a format/credit ask is \
require_attributes ("black and white", "directed by women"), a concrete subject is \
require_keyword_concepts — and when you are unsure the subject term is known, TRY it there anyway \
(an unknown term is reported and degrades honestly; parking it here guarantees the loss). Notes are \
for plot MECHANICS and craft techniques, not subjects. Most requests leave unsupported_notes EMPTY.

WORKED ROUTING EXAMPLES (each shows only the fields that matter — the routing patterns above, applied):
- "a movie with a really cozy, small-town vibe" → require_keyword_concepts ["small town"] + mood ["cozy"]
- "a very dark, gothic romance vibe" → require_keyword_concepts ["gothic"] + liked_genres ["Romance"] + \
mood ["dark"]
- "a sports movie where the underdog wins" → require_keyword_concepts ["underdog"] + mood ["pumped up"]
- "the main character talks directly to the camera" → require_keyword_concepts \
["breaking the fourth wall"]
- "takes place over the course of one single night" → require_keyword_concepts ["one night"]
- "makes me feel nostalgic for the 90s" → require_keyword_concepts ["1990s"] + mood ["nostalgic"] \
(feel-of-decade; "movies FROM the 90s" would instead be year_min 1990 / year_max 1999)
- "a western set in modern times" → require_keyword_concepts ["neo western"] + liked_genres ["Western"]
- "a political campaign behind the scenes" → require_keyword_concepts ["election"]
- "a movie about a musician who loses their hearing" → require_keyword_concepts ["musicians", "deaf"]
- "a romance about two older people who get a second chance" → require_genres ["Romance"] + \
require_keyword_concepts ["aging", "second chance"]
- "siblings fighting over an inheritance after a parent dies" → require_keyword_concepts \
["inheritance"] + liked_genres ["Drama"]
- "a sci-fi movie about a utopian society that is secretly terrible" → liked_genres ["Sci-Fi"] + \
require_keyword_concepts ["utopia"] + mood ["dark"]
- "feels like a vacation somewhere in Europe" → require_keyword_concepts ["vacation"] + \
require_country ["European"]
- "funny animated movies for adults" → require_genres ["Animation"] + require_min_rating "R" + \
mood ["funny"]
- "a movie that feels like a lazy Sunday morning" → genome_tags ["feel-good", "heartwarming"] + \
mood ["cozy"]
- "what was that movie where a guy's whole life is secretly being filmed for TV" → \
require_keyword_concepts ["reality tv"] (tip-of-tongue lookups: name the CANONICAL concept for the \
premise, never the literal surface words — "television show"/"tv" resolve to nothing)

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
        current_year_minus_3=CURRENT_YEAR - 3,
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
      'genome_tags': ['cerebral', 'thought-provoking', 'mindfuck'],
      'hard_constraints': {'exclude_people': ['Christopher Nolan']}}),
    ('Movies with Tom Hanks',
     {'hard_constraints': {'require_people': ['Tom Hanks']}}),
    ('A family-friendly animated adventure from the last ten years, nothing scary',
     {'liked_genres': ['Animation', 'Adventure', 'Children'],
      'hard_constraints': {'year_min': CURRENT_YEAR - 10, 'exclude_genres': ['Horror']}}),
    # LOCATION setting (hard genome floor) — "set in Japan" is WHERE, distinct from require_country.
    ('Going to Tokyo soon — recommend movies actually set in Japan, but nothing animated or anime',
     {'hard_constraints': {'require_genome_tags': ['japan'], 'exclude_genres': ['Animation']}}),
    # Composition: nationality (hard) + head-noun genre (hard) + runtime bound.
    ('Korean thrillers please, and nothing over two hours',
     {'liked_genres': ['Thriller'],
      'hard_constraints': {'require_genres': ['Thriller'], 'require_country': ['South Korean'],
                           'max_runtime': 120}}),
    # Affect goal (soft mood) + emphatic affect exclusion (hard).
    ('Something to make me cry, but nothing too dark or depressing',
     {'mood': ['make me cry'],
      'hard_constraints': {'exclude_mood': ['dark or depressing']}}),
    # Format/attribute facets.
    ('Classic black-and-white films directed by women',
     {'hard_constraints': {'require_attributes': ['black and white', 'directed by women']}}),
    # DISCIPLINE: a "vibe", not "set in Paris" → SOFT genome_tags, NOT require_genome_tags.
    ('I love a romantic movie with a bit of a Paris vibe',
     {'liked_genres': ['Romance'],
      'genome_tags': ['paris']}),
    # Concrete TOPIC → free topic term (hard), NOT a genre and NOT a genome setting; the facet is the query.
    ('Movies about chess',
     {'hard_constraints': {'require_keyword_concepts': ['chess']}}),
    # TOPIC + a genuine liked seed: the topic hard-filters, the seed shapes ranking.
    ('Loved Das Boot — any other good submarine movies?',
     {'liked_items': ['Boot, Das (Boat, The) (1981)'],
      'hard_constraints': {'require_keyword_concepts': ['submarine']}}),
    # TOPIC exclusion.
    ('A fun horror night but PLEASE no zombie movies',
     {'liked_genres': ['Horror'],
      'hard_constraints': {'exclude_keyword_concepts': ['zombie']}}),
    # Free topic term — the server-side resolver routes it genome-first (no list to consult).
    ('Something about surfing on the beach',
     {'hard_constraints': {'require_keyword_concepts': ['surfing']}}),
    # Out-of-domain: not a movie ask at all — flag it, extract nothing else.
    ('Can you recommend a good video game with dogs in it?',
     {'out_of_domain': True}),
    # Un-routable plot-mechanics residue → unsupported_notes; the expressible parts still extract.
    ('Movies like The Prestige where the ending recontextualizes the whole story',
     {'liked_items': ['The Prestige (2006)'],
      'genome_tags': ['twist ending', 'mindfuck'],
      'unsupported_notes': ['ending recontextualizes the whole story']}),
    # Another free topic; the resolver lands it on the genome machinery (graded floor + anchors).
    # (require_genome_tags would also work — the two slots converge for a single subject term —
    # but topics belong in require_keyword_concepts; require_genome_tags is the SETTINGS slot.)
    ('Courtroom dramas with a big twist',
     {'liked_genres': ['Drama'],
      'genome_tags': ['twist ending'],
      'hard_constraints': {'require_keyword_concepts': ['courtroom']}}),
]


if __name__ == '__main__':
    # Quick visual check of the assembled prompt + schema.
    sp = build_system_prompt()
    print(f'[system prompt: {len(sp)} chars]')
    print(sp[:1200], '...\n')
    print('[schema]')
    print(json.dumps(build_schema(), indent=2)[:800], '...')
