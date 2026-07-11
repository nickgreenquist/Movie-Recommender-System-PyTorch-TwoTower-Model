"""
Stage 2 (prompts) — Six grouped extraction prompts  (LLM-vs-genome ablation)

One static system prompt per anti-fatigue group. Splitting the 132 dimensions into
six focused calls is the documented mitigation for the "lost in the middle" failure
mode — a single 132-field call gives careful values to the first ~25 fields then
defaults the rest to 0.5. See docs/plans/llm_vs_genome_ablation_plan.md.

The static instructions (shared calibration rules + per-group guidance + the
enumerated feature list with genome glosses) live in the SYSTEM block so they can be
prompt-cached across every movie in a group; only the per-movie content varies, and
it rides in the USER turn (after the cache breakpoint). Enumerating the features here
— even though the structured-output schema also carries them — is deliberate: it
sharpens calibration AND lengthens the static prefix toward the cacheable minimum
(2048 tok Sonnet / 4096 Haiku).

The dimension LIST is generated from the derived schema (schemas.SCHEMA) so it never
drifts from what the Pydantic models validate. Only the per-group GUIDANCE prose is
hand-authored — it encodes how to score each family (e.g. reception is factual-only,
visual is medium-only), which is the judgment the schema alone can't convey.

Bump PROMPT_VERSION on any wording change so cached extractions from an older prompt
can be told apart from new ones.
"""
from llm_features.schemas import SCHEMA


PROMPT_VERSION = 'v2'   # v2: sub-genre granularity (crime split + sci-fi/fantasy/horror)


# ── Shared calibration preamble (identical across groups) ────────────────────

PREAMBLE = """\
You extract structured content features from a movie for a recommendation system.
The user message gives you what is known about one film (metadata + plot + any
reception text). For EACH feature listed below, assign a relevance score from 0.0
to 1.0:

  • 0.0  — the feature is definitely absent
  • 1.0  — the feature is central or extremely prominent
  • in between — partial, minor, or background presence

Calibration rules:
  • Use the FULL 0.0–1.0 range. Do NOT default to 0.5 when unsure — make your best
    calibrated estimate.
  • Most features are 0.0 for any given film. Only the few that genuinely apply
    should be high. A long list of mid-range scores is a mistake.
  • Score from the information provided. Where the text is silent on something
    factual (an award, a setting), treat it as absent rather than guessing."""


# ── Per-group guidance (hand-authored — the scoring judgment per family) ──────

GROUP_GUIDANCE = {
    'themes': """\
These are THEMES & PLOT axes. Score how central each theme is to the story — a
motif glimpsed once scores low; a theme that drives the plot scores high.""",

    'tone': """\
These are TONE & MOOD axes — emotional register and atmosphere. They are
calibration-sensitive (a film can be mildly or intensely "tense"). Avoid
contradictions: a film is rarely strongly "feel_good" AND strongly "bleak" at once.""",

    'setting': """\
These are factual SETTING, ERA & SUB-GENRE axes — where and when the film takes place,
plus its speculative or genre world (cyberpunk, high fantasy, zombie, vampire, robots,
…). Score a place/era high only if the film is actually set there or then, and a
sub-genre high only if the film genuinely is that kind of film. Most films touch only
a few of these; the rest are 0.0.""",

    'provenance': """\
These are factual PROVENANCE & STRUCTURE axes — the film's origin and narrative form
(adapted from a book, a sequel, a remake, a documentary, nonlinear, an ensemble,
etc.). Score from metadata and plot facts, not mood. These tend toward 0.0 or 1.0;
use a middle value only when the evidence is genuinely partial.""",

    'reception': """\
These are FACTUAL RECEPTION & PRESTIGE axes. Score them ONLY from objective evidence
in the provided text — stated awards or nominations, critical-reception passages,
release era, and box-office / rating signals. If the text gives no evidence for a
prestige signal, score it 0.0. Do NOT infer prestige from how good the plot sounds,
and do NOT recall awards from memory — only what the text supports.""",

    'visual': """\
These are FACTUAL VISUAL-MEDIUM axes — the production medium only (animated,
black-and-white, stop-motion, CGI-heavy, anime, etc.). Score from medium facts, not
aesthetic judgment. A standard live-action colour film scores 0.0 on all of these.""",
}


# ── Build the static system prompt per group ─────────────────────────────────

def _feature_lines(group: dict) -> str:
    """'  • name — genome glosses' for every dimension in the group."""
    return '\n'.join(
        f"  • {d['name']} — {', '.join(t['tag'] for t in d['genome_tags'])}"
        for d in group['dimensions']
    )


def _build_system_prompt(group: dict) -> str:
    return (
        f"{PREAMBLE}\n\n"
        f"{GROUP_GUIDANCE[group['key']]}\n\n"
        f"Features to score ({group['title']}):\n"
        f"{_feature_lines(group)}"
    )


# {group_key: static system-prompt text}. Built once at import; safe to cache.
SYSTEM_PROMPTS = {g['key']: _build_system_prompt(g) for g in SCHEMA['groups']}


# ── Per-movie user turn (the only part that varies → not cached) ─────────────

def user_message(movie_content: str) -> str:
    """Wrap the formatted per-movie content (from llm_extract.format_for_prompt)."""
    return (
        f"Movie information:\n\n{movie_content}\n\n"
        f"Score every listed feature for this film."
    )


if __name__ == '__main__':
    # Self-check: show one built system prompt and the per-group prefix sizes.
    for key, text in SYSTEM_PROMPTS.items():
        print(f"{key:<11} system prompt: {len(text):>5} chars")
    print("\n" + "═" * 70 + "\n")
    print(SYSTEM_PROMPTS['visual'])
