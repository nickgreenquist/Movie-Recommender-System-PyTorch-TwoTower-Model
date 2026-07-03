"""
src/llm_frontend.py — shared natural-language → two-tower retrieval core for the LLM
conversational front-end (v1).

PURPOSE
    The single, serving/-only implementation of the v1 pipeline that turns an LLM
    extraction object into recommendations FROM THE TRAINED MODEL:

        extraction JSON (from the LLM — see tools/llm_frontend_prompt.py)
          → resolve named titles to catalog titles (Mode 1, fuzzy)
          → resolve named people to TMDB person IDs (scraped-facet store)
          → synthesize a Mode-2 query: genome tags → representative anchor movies
          → build the user embedding via the real serving model (src/inference.py)
          → rank the whole corpus (raw dot product == cosine; both L2-normed)
          → apply post-filters (year + genre + people)
          → top-N recommendations (never raw LLM output).

    This module is imported by BOTH consumers so the logic lives in exactly one place:
      • tools/llm_frontend_probe.py — the in-repo test/QA harness (CLI + --smoke).
      • streamlit_app.py — the conversational tab (reuses the cached load_artifacts()
        model rather than reloading serving/).

    Everything here is serving/-only: it reads exactly what the deployed app loads
    (feature_store.pt, movie_embeddings.pt, model.pth) and never touches data/. The
    functions take a FrontendContext (built once from the loaded artifacts) so the
    Streamlit app can feed its @st.cache_resource Artifacts straight in.

LOCKED-IN v1 POLICY (validated in the Claude Code test loop — see docs/llm_frontend/llm_frontend_plan.md
"v1 Build Handoff" and memory project_llm_frontend_v1):
    Subordinated hybrid. Pure Mode 2 (no named titles) → 5 anchors/tag at weight 1.0
    (the anchors ARE the query). With named titles → ≤1 anchor/tag, cap 3 total, at
    weight 0.5, so the explicit likes (weight 2.0 each) always dominate the embedding.
    Like 2.0, dislike −2.0; most-recent timestamp bin; rank the full corpus, post-filter,
    then take top-N. Empty extraction → popularity fallback.
"""

import difflib
import math
import re
import unicodedata
from typing import NamedTuple

import numpy as np
import torch
import torch.nn.functional as F

from src.inference import build_user_embedding
from src.model import MovieRecommender

# ── Constants ────────────────────────────────────────────────────────────────
# History weights mirror streamlit_app.py's rec path: explicit likes 2.0, genome
# anchors 1.0, dislikes -2.0 (the same values the manual UI and canaries use).
LIKED_MOVIE_WEIGHT    =  2.0
ANCHOR_MOVIE_WEIGHT   =  1.0
DISLIKED_MOVIE_WEIGHT = -2.0
# Anchor budget. In pure Mode 2 (no named titles) the anchors ARE the query, so we use the
# full 5/tag at full weight (mirrors src/evaluate.py:ANCHORS_PER_TAG) — this path retrieves
# beautifully. But when the user named real titles (Mode 1), the titles ARE the query and
# anchors must only nudge: empirically, even 6 weight-1.0 anchors (sum 6.0) still outweigh
# 2 likes at 2.0 (sum 4.0) and drag the result off-target. So with likes present we cap
# anchors hard AND down-weight them, keeping total anchor signal well below the likes'
# (3 × 0.5 = 1.5 ≪ 4.0), so the named titles always dominate the user embedding.
ANCHORS_PER_TAG            = 5    # pure Mode 2 — anchors carry the whole query
ANCHORS_PER_TAG_WITH_LIKES = 1    # Mode 1 — at most one anchor per distinctive vibe tag
MAX_ANCHORS_WITH_LIKES     = 3    # ...and never enough, at half weight, to overwhelm the likes
ANCHOR_WEIGHT_WITH_LIKES   = 0.5  # down-weighted vs ANCHOR_MOVIE_WEIGHT so likes dominate
CANDIDATE_POOL        = 400    # retrieve large, post-filter, then surface TOP_N
TOP_N                 = 15

# Soft genome re-rank (Post-Retrieval Filtering "Source A"). The item embedding clusters by release
# era + co-watch, so even a CORRECT anchor set drifts to era-neighbours when ranked by cosine alone
# (a "western" vibe whose anchors are real westerns still surfaces 80s action first; "set in Paris"
# leaks French cinema at large). After scoring, ADD lambda * (mean genome relevance over the requested
# tags) to each movie's score so films that actually CARRY the tag rise. Additive (cosine is signed),
# soft (never a hard filter) → moods still ride the anchors and the pool never empties. 0.0 disables.
GENOME_RERANK_LAMBDA  = 0.5   # tuned via tools/llm_frontend_eval.py (+5 regression / +7 spec vs 0.0; plateaus after)
# Mood re-rank weight — a SEPARATE additive term for the Engine-2 mood tags (step B lever 1), kept
# apart from the subject-tag re-rank above so a tone modifier ("feel-good", "darker") nudges without
# out-ranking the subject the user actually asked for. Equal to the subject λ for now (a pure-mood
# request should feel as strong as a genome-tagged one); its own knob for future tuning.
MOOD_RERANK_LAMBDA    = 0.5

# Hard genome floor (require_genome_tags — step B lever 2). The soft re-rank above only NUDGES; an
# EMPHATIC setting/theme request ("movies *actually* set in Paris", "*only* WWII, not the Cold War")
# gates the candidate pool to films whose mean genome relevance over the required tags clears this
# floor, dropping the rest (Rome-set La Dolce Vita, paris 0.08, goes; Midnight in Paris, 0.95, stays).
# ONLY hard_constraints.require_genome_tags triggers it — soft moods (genome_tags) and Mode-1.5 title
# tags never do. A required tag that is out-of-vocab contributes nothing; if NONE resolve there is no
# floor at all (graceful, like the year gate: we never hard-drop on a signal we can't compute).
GENOME_HARD_FLOOR     = 0.35

# Dedicated re-rank strength for the HARD require_genome_tags subject — SEPARATE from and stronger than
# the soft GENOME_RERANK_LAMBDA (0.5). The 0.35 floor only GATES; within the gated pool the base ranking
# is anchor cosine (era/co-watch/taste). On a THIN axis (e.g. chess: only ~7 of the 34 films clearing the
# floor are genuinely about chess) the ~0.7-wide cosine swing dominates the weak 0.5*relevance nudge, so
# incidental carriers (an X-Men chess-metaphor scene, chess 0.49) and genome homonyms (Cadillac Records =
# Chess RECORDS, 0.74) out-rank real subject films (The Seventh Seal 0.92, Fresh 0.91) — found in the bug
# sweep. A stronger lambda makes SUBJECT RELEVANCE the primary in-floor ordering (a Δ0.5 relevance gap
# ~ 1.0 pts overcomes the cosine spread) while cosine still breaks ties; on a DENSE axis (racing: nearly
# all ~1.0) the boost is near-uniform so the anchor/taste order is preserved. Tuned on the ruler.
REQUIRE_GT_RERANK_LAMBDA = 2.0

# Hard ANTI-floor for negative affect / anti-vibe exclusion (exclude_mood + exclude_genome_tags — the
# mirror of the require floor above). An emphatic "absolutely nothing dark", "I genuinely cannot handle
# gore" is a gate, not a nudge: drop any film whose genome relevance on ANY excluded tag clears this
# ceiling. Set ABOVE the require floor (0.35) so we only cut films that STRONGLY carry the unwanted
# affect — a moderate/incidental score survives, protecting the pool from over-pruning (the eval fails a
# vacuous empty pool). Only exclude_mood (routed via MOOD_TAGS) and exclude_genome_tags (verbatim) feed
# it; an out-of-vocab excluded tag contributes nothing, and an all-OOV set means no gate at all (graceful).
# Tuned on the ruler: 0.4 drops a horror-comedy like Shaun of the Dead (dark 0.46) on "nothing dark" while
# leaving the light-comedy pool full; 0.5 lets it leak, ≤0.35 starts cutting moderate incidental carriers.
GENOME_EXCLUDE_CEILING = 0.4

# Mode-1.5 title-genome injection (step B lever 3). On a PURE-TITLE request — a liked title but no
# genome_tags — cosine retrieval drifts to release-era / co-watch neighbours (a 1994 Pulp Fiction
# anchor pulls Forrest Gump / Jurassic Park). We inject the liked title's OWN most-relevant genome
# tags as SOFT re-rank tags (never anchors — that would over-expand the query), so films sharing its
# vibe rise. Skipped entirely when the extraction already carries genome_tags (an explicit vibe list
# wins) and non-discriminative acclaim/meta tags are filtered out (see MODE15_STOP_*), because
# "masterpiece"/"imdb top 250" would re-summon the very prestige-era neighbours we are fighting.
MODE15_TAGS_PER_TITLE = 6      # top genome tags injected per liked title (after the acclaim stoplist)
MODE15_TAG_MIN_REL    = 0.6    # only inject a title's tag if the title carries it this strongly
MODE15_MAX_TAGS       = 8      # global cap across all liked titles (keep the re-rank query focused)

# ── Genre-pool degradation (step C thread 2) ─────────────────────────────────
# A genre-oriented ask can fail three ways the levers above don't touch: (I) an under-specified genre
# (only liked_genres, no anchors) barely surfaces the genre — popular off-genre films win on cosine;
# (II) a dual-genre "X and Y" ask over a THIN intersection (Sci-Fi∩Film-Noir = 2 films) can't fill a
# result under strict AND; (III) an unrequested co-genre (Drama) swamps the pool (war/romance/thriller
# films are mostly Drama). Three composable levers — a soft genre re-rank, a near-empty-only AND→OR
# relax, and rolling genre-diversity caps in selection.
#
# NOTE (intersection-first, per the 2026-07-02 review + user call): a dual-genre "X and Y" ask means the
# INTERSECTION — "a horror comedy" / "both in the same film" wants films that are BOTH (Shaun of the
# Dead), NOT a mix of separate horror and comedy films. So a healthy X∩Y pool stays strict AND; we do
# NOT OR-fan it (an earlier exemplar-based blend trigger did, diluting the head — removed).
#
# L1 — genre-affinity soft re-rank. After scoring, ADD lambda * (fraction of the WANTED genres —
# require_genres ∪ liked_genres — a film carries), lifting on-genre films. Additive/soft like the
# genome re-rank (a post-hoc OUTPUT boost, not the weak soft-genre EMBEDDING input the residual work
# found inert); never a hard filter. Fixes (I).
GENRE_RERANK_LAMBDA = 0.3
# L2 — near-empty AND→OR relax. A dual-genre AND-pool smaller than this can't rank a coherent result,
# so relax to OR (the plan's "near-empty AND-pool → OR-fan") — Sci-Fi∩Film-Noir = 2 fans out; a healthy
# intersection stays strict AND (Fantasy∩Horror 86, Romance∩Thriller 118, Horror∩Comedy 160, Action∩
# Comedy 390, and the green Musical∩Thriller 5 / Sci-Fi∩Western 7). 3 is a principled floor (an
# intersection of <3 films can't be ranked, not a value read off the eval), so it degrades rather than
# returning almost nothing. Fixes (II).
GENRE_AND_POOL_MIN = 3
# L3 — genre-diversity selection. Two composable mechanisms in _select_diverse:
#   • UNREQUESTED co-genre rolling cap ≤ COGENRE_CAP_FRAC (always, any genre ask) — keeps a co-genre
#     like Drama from swamping a War/Romance/Thriller pool (a war film is usually also Drama), fix (III).
#   • genre-SIGNATURE MMR, active only for a near-empty-pool OR-fan (blend_genres set) — penalize a
#     candidate by MMR_SIGNATURE_LAMBDA per already-picked film sharing its requested-genre signature,
#     interleaving the {G1}/{G2}/{G1,G2} signatures so a THIN-pool fan-out (Sci-Fi∩Film-Noir → Sci-Fi OR
#     Film-Noir) gives a fair share of each rather than the denser cluster swamping it. Intentionally a
#     strong (near-round-robin) diversity penalty at this λ — that IS the goal for a fanned-out pool; it
#     never fires on the strict-AND intersection cases (no blend set), which stay concentrated.
COGENRE_CAP_FRAC       = 0.4
MMR_SIGNATURE_LAMBDA   = 0.35

FUZZY_CUTOFF          = 0.74   # difflib ratio floor for title resolution (raised from 0.6 — a
                               # wrong-film fuzzy hit poisons the whole anchor; better to drop)
FUZZY_YEAR_TOL        = 4      # reject a fuzzy hit whose catalog year disagrees with the year the
                               # LLM emitted by more than this (Star Wars 1977 ≠ Star Maps 1997)

# US MPAA content-rating ceiling order (require_max_rating). Higher ordinal = more restrictive; a
# film passes require_max_rating="PG" iff its own rating ordinal ≤ PG's. A film with no scraped
# cert has ordinal None → passes (mirrors the year gate: absent metadata never excludes).
MPAA_ORDER = {'G': 1, 'PG': 2, 'PG-13': 3, 'R': 4, 'NC-17': 5}

# Buffers rebuilt in build_serving_model (genome/content/LLM context); pop them from any
# older serving/model.pth before a strict load so old and new checkpoints both load.
NON_PERSISTENT_BUFFERS = ('genome_context_buffer', 'content_context_buffer', 'llm_feature_buffer')


# ── Serving model (mirrors streamlit_app.py:_build_serving_model) ─────────────
def build_serving_model(fs, cfg):
    """Reconstruct the serving MovieRecommender from serving/ artifacts alone.

    Sources the genome buffer from movieId_to_genome_tag_context and the LLM buffer from
    the baked llm_feature_buffer, so no data/ dir is needed (data/ is gitignored / absent
    on Streamlit Cloud). feature_towers ('genome' | 'llm' | 'both' | None), read from the
    exported model_config, selects which semantic-feature towers are built. Kept byte-for-byte
    in step with streamlit_app.py's loader — both build the same model from the same artifacts.
    """
    feature_towers = cfg.get('feature_towers', cfg.get('content_feature_source', 'genome'))
    has_genome = feature_towers in ('genome', 'both')
    has_llm    = feature_towers in ('llm', 'both')

    top_movies = fs['top_movies']

    genome_context_buffer = None
    genome_tags_len       = len(fs['genome_tag_ids'])
    if has_genome:
        genome_matrix = np.array(
            [fs['movieId_to_genome_tag_context'][mid] for mid in top_movies], dtype=np.float32)
        pad_row = np.zeros((1, genome_matrix.shape[1]), dtype=np.float32)
        genome_context_buffer = torch.from_numpy(np.vstack([genome_matrix, pad_row]))
        genome_tags_len = genome_matrix.shape[1]

    llm_feature_buffer = None
    llm_feature_len    = None
    if has_llm:
        llm_feature_buffer = fs['llm_feature_buffer']
        llm_feature_len    = llm_feature_buffer.shape[1]

    return MovieRecommender(
        genres_len=len(fs['genres_ordered']),
        tags_len=len(fs['tags_ordered']),
        genome_tags_len=genome_tags_len,
        top_movies_len=len(top_movies),
        all_years_len=len(fs['years_ordered']),
        timestamp_num_bins=fs['timestamp_num_bins'],
        user_context_size=fs['user_context_size'],
        feature_towers=feature_towers,
        genome_context_buffer=genome_context_buffer,
        llm_feature_buffer=llm_feature_buffer,
        llm_feature_len=llm_feature_len,
        item_genre_embedding_size=cfg['item_genre_embedding_size'],
        item_tag_embedding_size=cfg['item_tag_embedding_size'],
        item_genome_embedding_size=cfg.get('item_genome_embedding_size', 32),
        item_llm_embedding_size=cfg.get('item_llm_embedding_size', 32),
        item_movieId_embedding_size=cfg['item_movieId_embedding_size'],
        item_year_embedding_size=cfg['item_year_embedding_size'],
        user_genre_embedding_size=cfg['user_genre_embedding_size'],
        timestamp_feature_embedding_size=cfg['timestamp_feature_embedding_size'],
        user_genome_embedding_size=cfg.get('user_genome_embedding_size', 32),
        user_llm_embedding_size=cfg.get('user_llm_embedding_size', 32),
        proj_hidden=cfg.get('proj_hidden', 256),
        output_dim=cfg.get('output_dim', 128),
    )


# ── Frontend context (the loaded artifacts + derived lookups the pipeline needs) ──
class FrontendContext(NamedTuple):
    """Everything recommend()/resolve_title()/anchors_for() read, bundled once so both the
    harness and the Streamlit tab can feed their already-loaded artifacts in. Named fields
    so the derived lookups can't drift out of sync with the model/embedding matrix."""
    model:              MovieRecommender
    fs:                 dict
    all_ids:            list          # row i of all_embs ↔ movieId all_ids[i]
    all_embs:           torch.Tensor  # [N, 128] L2-normed combined item embeddings
    ts_inference:       torch.Tensor  # most-recent timestamp bin (canary / streamlit approach)
    title_index:        dict          # normalized-title → [canonical catalog titles]
    pop_rank:           dict          # canonical title → popularity rank (tie-break / fallback)
    genome_name_to_idx: dict          # genome tag NAME → 0-based column in the genome vector
    facets:             dict          # scraped-facet store (people tables) or None if unbaked
    genome_sim_matrix:  torch.Tensor  # [N, 1128] L2-normed raw genome vectors (row i ↔ all_ids[i]); the
                                      # Similar-tab genome content space, for 'movies like X' ranking


def build_frontend_context(model, fs, all_ids, all_embs, ts_inference, facets=None) -> FrontendContext:
    """Bundle the loaded serving artifacts into a FrontendContext, deriving the title-resolution
    index, popularity rank, and genome-name→column lookups from fs. Call once after loading (the
    harness does it in Serving; the Streamlit tab builds it from its cached Artifacts).

    `facets` is the scraped-facet store (movieId_to_people / person_name_to_ids / …, baked into
    serving/feature_store.pt by export); pass it through so recommend() can resolve and filter on
    people. None (an old serving artifact without the bake) degrades gracefully: people facets in
    an extraction are simply reported unresolved and dropped, leaving the rest of the query intact."""
    gn, gi = fs['genome_tag_names'], fs['genome_tag_to_i']
    # Raw genome content space (the Similar tab's "Genome Tags → Raw features" cell), L2-normed so a
    # single matmul against a seed row yields cosine. This — not the combined item embedding, which
    # clusters by release era / co-watch — is the faithful "movies like X" substrate (Toy Story → the
    # Pixar films, not the 1995 co-watch cohort). Built once (~40MB); row i ↔ all_ids[i].
    gt_ctx = fs['movieId_to_genome_tag_context']
    genome_sim_matrix = F.normalize(
        torch.stack([torch.as_tensor(gt_ctx[m], dtype=torch.float32) for m in all_ids]), dim=1)
    return FrontendContext(
        model=model,
        fs=fs,
        all_ids=all_ids,
        all_embs=all_embs,
        ts_inference=ts_inference,
        title_index=_build_title_index(fs),
        pop_rank={t: i for i, t in enumerate(fs['popularity_ordered_titles'])},
        genome_name_to_idx={gn[tid]: gi[tid] for tid in gi},
        facets=facets,
        genome_sim_matrix=genome_sim_matrix,
    )


# ── Title resolution (fuzzy-match LLM-extracted titles → catalog titles) ──────
_YEAR_SUFFIX_RE = re.compile(r'\s*\(\d{4}\)\s*$')
_YEAR_RE        = re.compile(r'\((\d{4})\)')
_ARTICLE_RE     = re.compile(
    r"^(.*),\s+(the|a|an|le|la|les|los|las|il|der|die|das|l')$", re.IGNORECASE)


def _norm_title(s):
    """Normalize a title for matching: drop "(year)", un-invert MovieLens "Name, The"
    → "the name", fold accents (é→e), lowercase, strip punctuation. Handles user input
    ("The Matrix") matching the catalog's inverted, year-suffixed form ("Matrix, The (1999)"),
    and accented user/LLM titles ("Amélie") matching the catalog without fragile fuzzy scoring
    (without folding, "amélie" → "am lie" and mis-fuzzes to "My Life")."""
    s = _YEAR_SUFFIX_RE.sub('', s).strip()
    m = _ARTICLE_RE.match(s)
    if m:
        art = m.group(2)
        sep = '' if art.lower() == "l'" else ' '
        s = f"{art}{sep}{m.group(1)}"
    s = s.lower()
    s = unicodedata.normalize('NFKD', s)
    s = ''.join(c for c in s if not unicodedata.combining(c))  # drop accents
    s = re.sub(r'[^a-z0-9]+', ' ', s).strip()
    return s


def _leading_title(s):
    """The main title before any "(alternate / original title)" parenthetical. MovieLens
    titles often carry a foreign or aka subtitle — "Amélie (Fabuleux destin d'Amélie Poulain,
    Le) (2001)", "Seven (a.k.a. Se7en) (1995)" — that the English request never names. Indexing
    the leading part lets "Amélie"/"Seven" resolve by exact normalized hit instead of fuzzy."""
    cut = s.find(' (')
    return s[:cut].strip() if cut > 0 else s


def _extract_year(s):
    """The first 4-digit "(year)" in a title string, or None."""
    m = _YEAR_RE.search(s)
    return int(m.group(1)) if m else None


def _pick_candidate(cands, raw, ctx):
    """Choose among catalog titles that share a normalized key. Prefer the one whose year matches
    the year the LLM emitted (within FUZZY_YEAR_TOL), then by popularity. Fixes same-name,
    different-year collisions — e.g. "Oldboy (2003)" (Korean original) must not lose to the more
    popular "Oldboy (2013)" remake just because the leading-title key collides."""
    want = _extract_year(raw)
    def key(t):
        gy = _extract_year(t)
        year_pen = 0 if (want is None or gy is None or abs(want - gy) <= FUZZY_YEAR_TOL) else 1
        return (year_pen, ctx.pop_rank.get(t, 1 << 30))
    return min(cands, key=key)


def _build_title_index(fs):
    """normalized-title → list of canonical catalog titles (with "(year)" suffix). Each catalog
    title is indexed under both its full normalized form AND its leading-title form (before any
    alternate-title parenthetical), so a request that names only the main title still resolves
    exactly. Collisions (and a leading key shadowing a full title) are broken by popularity in
    resolve_title."""
    index = {}
    for _mid, title in fs['movieId_to_title'].items():
        keys = {_norm_title(title), _norm_title(_leading_title(title))}
        for k in keys:
            if k:
                index.setdefault(k, []).append(title)
    return index


def resolve_title(raw, ctx):
    """Resolve one LLM-extracted title to a canonical catalog title.

    Returns (canonical_title | None, note). Strategy: exact (case-insensitive,
    year-aware) → exact normalized → difflib fuzzy. On a normalized collision (or
    fuzzy tie) prefer the most popular candidate. Unresolved titles return None and
    are reported, then dropped (build_user_embedding would silently drop them too)."""
    fs = ctx.fs
    # 1. Exact catalog hit (the LLM sometimes returns the full "Title (year)" string).
    if raw in fs['title_to_movieId']:
        return raw, 'exact'

    norm = _norm_title(raw)
    if not norm:
        return None, 'empty'

    # 2. Exact normalized hit (handles missing year + article inversion). If the LLM emitted a
    #    year and even the best candidate's year is off by more than FUZZY_YEAR_TOL, the named film
    #    isn't in the catalog (only a same-name different-year film is, e.g. "Oldboy (2003)" when
    #    only the 2013 remake exists) — drop rather than anchor on the wrong movie.
    if norm in ctx.title_index:
        cands = ctx.title_index[norm]
        best = _pick_candidate(cands, raw, ctx)
        want_year, got_year = _extract_year(raw), _extract_year(best)
        if want_year and got_year and abs(want_year - got_year) > FUZZY_YEAR_TOL:
            return None, f'normalized rejected (year {got_year}≠{want_year})'
        note = 'normalized' if len(cands) == 1 else f'normalized, {len(cands)} matches → year/pop'
        return best, note

    # 3. Fuzzy match on normalized keys (last resort). A wrong-film fuzzy hit poisons the whole
    #    user embedding, so the bar is high: above FUZZY_CUTOFF, and — when the LLM emitted a
    #    year — within FUZZY_YEAR_TOL of the candidate's year. A year mismatch is the tell for a
    #    spurious match (Se7en→Sheena, Star Wars 1977→Star Maps 1997, Pan's Labyrinth 2006→
    #    Labyrinth 1986); we drop rather than anchor on the wrong movie.
    keys = list(ctx.title_index.keys())
    hit = difflib.get_close_matches(norm, keys, n=1, cutoff=FUZZY_CUTOFF)
    if hit:
        ratio = difflib.SequenceMatcher(None, norm, hit[0]).ratio()
        cands = ctx.title_index[hit[0]]
        best = _pick_candidate(cands, raw, ctx)
        want_year, got_year = _extract_year(raw), _extract_year(best)
        if want_year and got_year and abs(want_year - got_year) > FUZZY_YEAR_TOL:
            return None, f'fuzzy {ratio:.2f} rejected (year {got_year}≠{want_year})'
        return best, f'fuzzy {ratio:.2f}'

    return None, 'no match'


# ── Person resolution (scraped-facet store: name → canonical TMDB person ID) ──
# The two-tower model has no actor/director concept, so people facets ("Tom Hanks movies")
# come entirely from the TMDB credits scrape, distilled by llm_features/build_facet_store.py
# into the facet-store tables (person_name_to_ids / person_id_to_name / person_id_to_film_count)
# that the export step bakes into serving/. resolve_person mirrors resolve_title but is simpler:
# canonical TMDB person IDs disambiguate same-name people for free, so an exact normalized-name
# hit suffices — no fuzzy-year guard. See docs/llm_frontend/facet_store_plan.md.
_PERSON_PUNCT_RE = re.compile(r'[^a-z0-9]+')


def _norm_name(s):
    """Normalize a person name for matching: lowercase, fold accents (Penélope→penelope),
    collapse punctuation/whitespace runs to single spaces. Mirrors _norm_title's accent-fold +
    lowercase + punctuation strip, but WITHOUT the title-only logic (no "(year)" strip, no
    "Name, The" article inversion) — people are neither year-suffixed nor article-inverted.

    Used on BOTH sides so the keys always agree: llm_features/build_facet_store.py imports this
    to key person_name_to_ids at build time, and resolve_person re-normalizes the LLM-emitted
    name at inference time."""
    s = s.strip().lower()
    s = unicodedata.normalize('NFKD', s)
    s = ''.join(c for c in s if not unicodedata.combining(c))  # drop accents
    s = _PERSON_PUNCT_RE.sub(' ', s).strip()
    return s


def resolve_person(raw, facets, role=None):
    """Resolve one LLM-extracted person name to a canonical TMDB person ID.

    Returns (pid | None, note). Strategy: exact normalized-name hit. Unlike resolve_title there is
    no fuzzy/year machinery — TMDB person IDs make same-name people distinct, so resolution is an
    exact lookup. On a normalized collision (two real people share a name) prefer the one with more
    in-corpus films; the name indexes are pre-sorted that way at build time, so we take the head.
    A miss returns None and is reported, then dropped by the caller (like an unresolved title).

    `role` routes the two SEPARATE name namespaces (see build_facet_store.py):
      • None (default, the require_people path): BILLED first (actor/director/writer via
        person_name_to_ids), falling back to composer only if no billed person matches — so
        "movies with John Williams" resolves to the actor, not the prolific same-named composer,
        yet "scored by Hans Zimmer" (a composer-only name) still resolves.
      • 'composer' (the require_composers path): composer_name_to_ids only.
    `facets` is the baked facet-store dict; carried on the FrontendContext as ctx.facets."""
    norm = _norm_name(raw)
    if not norm:
        return None, 'empty'
    billed   = facets.get('person_name_to_ids') or {}
    composer = facets.get('composer_name_to_ids') or {}
    if role == 'composer':
        ids, src = composer.get(norm), 'composer'
    else:
        ids, src = billed.get(norm), 'billed'
        if not ids:                              # billed-preferred: fall back to a composer-only name
            ids, src = composer.get(norm), 'composer'
    if not ids:
        return None, 'no match'
    pid = ids[0]  # pre-sorted: most in-corpus films first (fame / coverage tie-break)
    note = 'exact' if len(ids) == 1 else f'exact, {len(ids)} same-name → most-films'
    if src == 'composer' and role != 'composer':
        note += ' (composer)'
    return pid, note


# ── Facet resolution: country / language / format (Engine-1a membership, step C) ──
# The two-tower model has no nationality/language/format concept, so these come entirely from the
# TMDB scrape (production_countries / original_language / a curated top-distribution keyword set),
# distilled by build_facet_store into serving/ tables (movieId_to_countries/_language/_attributes)
# and gated in _passes_constraints. resolve_facet is the country/language/format analogue of
# resolve_person: it maps a free LLM phrase ("French", "in Chinese", "black and white") to the
# canonical facet value(s) the tables are keyed on. Membership is EXACT (alias-mapped) — never a
# substring probe (per the plan's overturned #4: `world war i` ⊂ `world war ii`, `paris` ⊂ `parish`).
#
# Country: production nationality (ISO 3166-1 alpha-2). A region phrase ("Scandinavian") expands to
# a member-code list; require_country then matches a film whose production_countries intersect the
# set (ANY — "French films" = FR-produced; a co-production listing FR passes). NOT a setting proxy
# (overturned #5: a Paris-SET film is often a US/GB production — "set in Paris" is a location keyword,
# a separate axis routed to require_genome_tags, not here).
COUNTRY_ALIASES = {
    'us': 'US', 'usa': 'US', 'u s a': 'US', 'america': 'US', 'american': 'US',
    'united states': 'US', 'united states of america': 'US', 'hollywood': 'US',
    'uk': 'GB', 'u k': 'GB', 'britain': 'GB', 'british': 'GB', 'england': 'GB', 'english': 'GB',
    'great britain': 'GB', 'united kingdom': 'GB',
    'france': 'FR', 'french': 'FR',
    'germany': 'DE', 'german': 'DE', 'west germany': 'DE', 'east germany': 'DE',
    'italy': 'IT', 'italian': 'IT',
    'spain': 'ES', 'spanish': 'ES',
    'japan': 'JP', 'japanese': 'JP',
    'china': 'CN', 'chinese': 'CN', 'mainland china': 'CN',
    'hong kong': 'HK', 'hongkong': 'HK',
    'taiwan': 'TW', 'taiwanese': 'TW',
    'korea': 'KR', 'korean': 'KR', 'south korea': 'KR', 'south korean': 'KR',
    'india': 'IN', 'indian': 'IN', 'bollywood': 'IN',
    'canada': 'CA', 'canadian': 'CA',
    'australia': 'AU', 'australian': 'AU',
    'new zealand': 'NZ',
    'ireland': 'IE', 'irish': 'IE',
    'mexico': 'MX', 'mexican': 'MX',
    'brazil': 'BR', 'brazilian': 'BR',
    'argentina': 'AR', 'argentine': 'AR', 'argentinian': 'AR',
    'russia': 'RU', 'russian': 'RU', 'soviet': 'SU', 'soviet union': 'SU', 'ussr': 'SU',
    'sweden': 'SE', 'swedish': 'SE',
    'denmark': 'DK', 'danish': 'DK',
    'norway': 'NO', 'norwegian': 'NO',
    'finland': 'FI', 'finnish': 'FI',
    'iceland': 'IS', 'icelandic': 'IS',
    'netherlands': 'NL', 'dutch': 'NL', 'holland': 'NL',
    'belgium': 'BE', 'belgian': 'BE',
    'switzerland': 'CH', 'swiss': 'CH',
    'austria': 'AT', 'austrian': 'AT',
    'poland': 'PL', 'polish': 'PL',
    'czech': 'CZ', 'czech republic': 'CZ', 'czechoslovakia': 'CZ',
    'hungary': 'HU', 'hungarian': 'HU',
    'portugal': 'PT', 'portuguese': 'PT',
    'greece': 'GR', 'greek': 'GR',
    'iran': 'IR', 'iranian': 'IR', 'persian': 'IR',
    'thailand': 'TH', 'thai': 'TH',
    'israel': 'IL', 'israeli': 'IL',
    'turkey': 'TR', 'turkish': 'TR',
    'luxembourg': 'LU',
}
# Region / supra-national phrase → member ISO codes (require_country matches ANY member). Curated
# from the corpus's production_countries distribution (see facet_store_plan Engine-1a "~26-code
# region table"); a film need only list ONE member country to satisfy the region.
COUNTRY_REGIONS = {
    'scandinavian': ['SE', 'DK', 'NO', 'FI', 'IS'],
    'nordic':       ['SE', 'DK', 'NO', 'FI', 'IS'],
    'east asian':   ['JP', 'KR', 'CN', 'HK', 'TW'],
    'asian':        ['JP', 'KR', 'CN', 'HK', 'TW', 'IN', 'TH', 'IR', 'IL'],
    'european':     ['GB', 'FR', 'DE', 'IT', 'ES', 'SE', 'DK', 'NO', 'FI', 'IS', 'NL', 'BE',
                     'CH', 'AT', 'PL', 'CZ', 'HU', 'PT', 'GR', 'IE', 'LU', 'RU', 'SU'],
    'latin american': ['MX', 'BR', 'AR'],
    'south american': ['BR', 'AR'],
}
# Language: ORIGINAL production language (ISO 639-1). require_language / require_original_language both
# gate on original_language ("original French, nothing dubbed"). A phrase may map to several codes —
# "Chinese" spans Mandarin (TMDB 'zh') and Cantonese (TMDB's non-standard 'cn'); the filter matches ANY.
LANGUAGE_ALIASES = {
    'english': ['en'],
    'french': ['fr'],
    'german': ['de'],
    'italian': ['it'],
    'spanish': ['es'],
    'japanese': ['ja'],
    'chinese': ['zh', 'cn'], 'mandarin': ['zh'], 'cantonese': ['cn'],
    'korean': ['ko'],
    'russian': ['ru'],
    'swedish': ['sv'],
    'danish': ['da'],
    'norwegian': ['no'],
    'finnish': ['fi'],
    'dutch': ['nl'],
    'portuguese': ['pt'],
    'polish': ['pl'],
    'czech': ['cs'],
    'hungarian': ['hu'],
    'greek': ['el'],
    'persian': ['fa'], 'farsi': ['fa'],
    'thai': ['th'],
    'hindi': ['hi'],
    'arabic': ['ar'],
    'hebrew': ['he'],
    'turkish': ['tr'],
    'vietnamese': ['vi'],
}
# Format / attribute facets — a curated set of single, clean, high-precision TMDB keywords (the plan's
# Engine-1a "single clean top-25 tag"). Each canonical attr key maps to the TMDB keyword name(s)
# build_facet_store scans for; FORMAT_ALIASES routes the user's phrasing to that key. Import
# FORMAT_ATTR_KEYWORDS into the builder so the stored membership and the resolver share one vocabulary.
FORMAT_ATTR_KEYWORDS = {
    'black and white':    ['black and white'],
    'woman director':     ['woman director'],
    'based on book':      ['based on novel or book'],
    'based on true story': ['based on true story'],
    'silent film':        ['silent film'],
    'anime':              ['anime'],
    'stop motion':        ['stop motion'],
    'independent film':   ['independent film'],
}
FORMAT_ALIASES = {
    'black and white': 'black and white', 'b w': 'black and white', 'b and w': 'black and white',
    'black white': 'black and white', 'monochrome': 'black and white', 'greyscale': 'black and white',
    'grayscale': 'black and white',
    'directed by women': 'woman director', 'woman director': 'woman director',
    'women directors': 'woman director', 'female director': 'woman director',
    'female directors': 'woman director', 'directed by a woman': 'woman director',
    'based on a book': 'based on book', 'based on a novel': 'based on book', 'based on book': 'based on book',
    'based on novel': 'based on book', 'based on novel or book': 'based on book',
    'literary adaptation': 'based on book', 'book adaptation': 'based on book', 'novel adaptation': 'based on book',
    'based on a true story': 'based on true story', 'based on true story': 'based on true story',
    'true story': 'based on true story', 'based on real events': 'based on true story',
    'based on true events': 'based on true story', 'based on a real story': 'based on true story',
    'silent': 'silent film', 'silent film': 'silent film', 'silent movie': 'silent film',
    'anime': 'anime',
    'stop motion': 'stop motion', 'claymation': 'stop motion', 'clay animation': 'stop motion',
    'indie': 'independent film', 'independent': 'independent film', 'independent film': 'independent film',
    'indie film': 'independent film',
}
# Keyword CONTENT concepts — a curated set of concrete "movies about X" nouns (chess / submarine / boxing /
# dinosaur / …) where the two-tower + genome path is WEAK: genome is thin on niche topics, its top signal is
# award/meta-polluted (Rocky's top genome tags are 4 Oscars + `sports`; `boxing` is 13th), and cheap surface
# nouns collide with homonyms. TMDB's crowd-sourced keywords carry these cleanly, so a HARD boolean membership
# pre-filter (require_keyword_concepts) beats a genome floor for concrete-noun intent, and composes with the
# empty→popularity fallback. Each canonical concept maps to the EXACT (lowercase) TMDB keyword name(s)
# build_facet_store scans for; the allow-list is hand-curated to EXCLUDE homonyms (chess ≠ `duchess`; shark ≠
# `loan shark`; heist ≠ `atheist`; alien invasion ≠ `alienation`; outer space includes `space opera` but wine ≠
# `swine`, opera ≠ `space opera`). Import KEYWORD_CONCEPTS into the builder so stored membership + the resolver
# share one vocabulary (mirrors FORMAT_ATTR_KEYWORDS). Verdict-driven: replaces the abandoned narrative-dimension
# feature (redundant with genome's story-shape tags) — see project_narrative_dimension_plan / facet_store_plan.
KEYWORD_CONCEPTS = {
    'chess': ['chess', 'playing chess', 'chess tournament', 'chess match', 'chess champion'],
    'submarine': ['submarine', 'nuclear submarine', 'submarine commander', 'mini submarine', 'russian submarine', 'submarine warfare', 'submarine crew'],
    'boxing': ['boxing', 'boxer', 'bare knuckle boxing', 'boxing trainer', 'boxing match', 'boxing school', 'boxing champion', 'kick boxing', 'kickboxing', 'boxing ring'],
    'dinosaur': ['dinosaur', 'robot dinosaur'],
    'shark': ['shark', 'shark attack', 'great white shark', 'killer shark', 'shark cage'],
    'poker': ['poker', 'poker game'],
    'surfing': ['surfing', 'surfer', 'surfboard', 'surf', 'surfing contest'],
    'mountain climbing': ['climbing', 'mountain climbing', 'rock climbing', 'free climbing', 'solo climbing', 'climbing accident'],
    'sailing': ['sailing', 'sailor', 'sailboat', 'sailing ship', 'sailing trip', 'sail boat'],
    'wine': ['wine', 'wine cellar', 'winery', 'winegrowing'],
    'cooking': ['cooking', 'chef', 'celebrity chef', 'cookbook', 'gourmet cook', 'masterchef'],
    'jazz': ['jazz', 'jazz singer or musician', 'jazz club', 'jazz band', 'jazz age', 'jazz music', 'hot jazz'],
    'zombie': ['zombie', 'zombies', 'zombie apocalypse', 'zombie horror', 'zombie comedy'],
    'vampire': ['vampire', 'vampire hunter (slayer)', 'vampiress (female vampire)', 'child vampire', 'vampire human love', 'vampire bat'],
    'werewolf': ['werewolf', 'werewolf child'],
    'ghost': ['ghost', 'ghost ship', 'ghost story', 'ghost child', 'vengeful ghost', 'ghost hunting'],
    'witch': ['witch', 'witchcraft', 'school of witchcraft', 'witch hunt', 'witch burning', 'evil witch'],
    'samurai': ['samurai', 'samurai sword', 'code of the samurai', 'samurai western'],
    'pirate': ['pirate', 'pirate gang', 'pirate ship', 'space pirate'],
    'viking': ['vikings (norsemen)'],
    'gladiator': ['gladiator', 'gladiator fight'],
    'knight': ['knight', 'knights of the round table', 'knight templars', 'knights templar', 'medieval knight'],
    'cowboy': ['cowboy', 'rodeo cowboy', 'singing cowboy'],
    'robot': ['robot', 'giant robot', 'killer robot', 'humanoid robot', 'robot cop', 'robot as menace', 'robotics'],
    'cyborg': ['cyborg', 'female cyborg'],
    'android': ['android', 'android horror', 'synthetic android', 'human android relationship'],
    'virtual reality': ['virtual reality'],
    'time travel': ['time travel'],
    'time loop': ['time loop'],
    'alien invasion': ['alien', 'aliens', 'alien invasion', 'alien life-form', 'alien planet', 'alien abduction', 'alien contact'],
    'outer space': ['space travel', 'spacecraft', 'space station', 'spaceship', 'outer space', 'space opera', 'space marine', 'astronaut'],
    'nuclear weapons': ['nuclear war', 'nuclear weapons', 'nuclear missile', 'nuclear explosion', 'nuclear threat', 'nuclear radiation', 'nuclear bomb', 'atomic bomb'],
    'pandemic': ['pandemic', 'epidemic', 'outbreak', 'lethal virus'],
    'circus': ['circus', 'traveling circus', 'circus freak'],
    'casino': ['casino', 'casino owner', 'casino heist', 'casino vault'],
    'prison': ['prison', 'prisoner', 'prison escape', 'imprisonment', "women's prison", 'prison break', 'release from prison', 'prisoner of war'],
    'heist': ['heist', 'bank heist', 'jewelry heist', 'diamond heist', 'art heist', 'gold heist', 'heist gone wrong', 'casino heist'],
    'dragon': ['dragon', 'dragonslayer', 'dragon egg', 'dragon rider', 'talking dragon', 'komodo dragon'],
    'wizard': ['wizard', 'wizardry'],
    'kung fu': ['kung fu', 'kung fu master', 'shaolin kung fu'],
    'martial arts': ['martial arts', 'martial arts master', 'martial arts tournament', 'martial arts training', 'martial arts school', 'female martial artist'],
    'mixed martial arts': ['mixed martial arts (mma)'],
    'wrestling': ['wrestling', 'wrestler', 'pro wrestling', 'wrestling coach', 'pro wrestlers', 'pro wrestler', 'arm wrestling'],
    'baseball': ['baseball', 'baseball player', 'baseball stadium', 'major league baseball (mlb)', 'baseball bat', 'baseball pitcher', 'baseball team', 'baseball hall of fame'],
    'basketball': ['basketball', 'basketball player', 'national basketball association (nba)', 'basketball coach', 'basketball team'],
    'american football': ['american football', 'american football coach', 'american football team', 'high school american football', 'nfl (national football league)'],
    'soccer': ['football (soccer)', 'amateur football (soccer)', 'football (soccer) coach'],
    'golf': ['golf', 'golf tournament', 'golfers', 'golf course', 'golf instructor'],
    'skateboarding': ['skateboarding', 'skateboarder'],
    'motorcycle': ['motorcycle', 'motorcycle gang', 'motorcycle chase', 'motorcycle crash'],
    'mafia': ['mafia', 'bratva (russian mafia)', 'mafia boss', 'chinese mafia', 'sicilian mafia', 'mafia family', 'italian mafia', 'japanese mafia'],
    'yakuza': ['yakuza', 'female yakuza', 'yakuza eiga'],
    'cannibal': ['cannibal', 'cannibalism', 'self-cannibalism'],
    'spy': ['spy', 'british spy', 'russian spy', 'female spy', 'spy thriller', 'american spy', 'teen spy', 'spy ring'],
    'ballet': ['ballet', 'ballet dancer', 'ballet school', 'ballet performance', 'ballet dancing', 'ballet company', 'bolshoi ballet'],
    'opera': ['opera', 'opera singer', 'rock opera'],
    'magic': ['magic', 'black magic', 'magician', 'magic show', 'magical creature', 'magical object'],
}
# User phrasing (normalized via _norm_name) → canonical concept, for phrasings the LLM may emit that aren't the
# concept key itself (plurals + close synonyms). The resolver also accepts any concept key VERBATIM, so this only
# needs the deltas. Kept exact (no substring) — same discipline as the other facet resolvers.
KEYWORD_CONCEPT_ALIASES = {
    'submarines': 'submarine',
    'boxer': 'boxing', 'boxers': 'boxing', 'prizefighting': 'boxing', 'prize fighting': 'boxing',
    'dinosaurs': 'dinosaur',
    'sharks': 'shark',
    'surfer': 'surfing', 'surfers': 'surfing', 'surfboarding': 'surfing',
    'climbing': 'mountain climbing', 'rock climbing': 'mountain climbing', 'mountaineering': 'mountain climbing',
    'sailboat': 'sailing', 'sailors': 'sailing', 'yachting': 'sailing',
    'vineyard': 'wine', 'winemaking': 'wine', 'sommelier': 'wine',
    'chef': 'cooking', 'chefs': 'cooking', 'culinary': 'cooking', 'cuisine': 'cooking', 'cook': 'cooking', 'cooks': 'cooking',
    'zombies': 'zombie', 'undead': 'zombie',
    'vampires': 'vampire',
    'werewolves': 'werewolf', 'lycanthrope': 'werewolf',
    'ghosts': 'ghost', 'haunting': 'ghost', 'haunted house': 'ghost', 'poltergeist': 'ghost',
    'witches': 'witch', 'sorceress': 'witch',
    'samurais': 'samurai', 'ronin': 'samurai',
    'pirates': 'pirate',
    'vikings': 'viking', 'norse': 'viking', 'norsemen': 'viking',
    'gladiators': 'gladiator',
    'knights': 'knight',
    'cowboys': 'cowboy',
    'robots': 'robot',
    'cyborgs': 'cyborg',
    'androids': 'android',
    'vr': 'virtual reality',
    'time travelling': 'time travel', 'time traveling': 'time travel',
    'aliens': 'alien invasion', 'alien': 'alien invasion', 'extraterrestrial': 'alien invasion', 'extraterrestrials': 'alien invasion', 'ufo': 'alien invasion', 'ufos': 'alien invasion',
    'space': 'outer space', 'in space': 'outer space', 'spaceship': 'outer space', 'spaceships': 'outer space', 'astronaut': 'outer space', 'astronauts': 'outer space', 'space travel': 'outer space',
    'nuclear war': 'nuclear weapons', 'nuclear weapon': 'nuclear weapons', 'nuclear bomb': 'nuclear weapons', 'atomic bomb': 'nuclear weapons', 'nukes': 'nuclear weapons',
    'epidemic': 'pandemic', 'outbreak': 'pandemic', 'plague': 'pandemic', 'contagion': 'pandemic', 'virus outbreak': 'pandemic',
    'casinos': 'casino', 'gambling': 'casino',
    'prisons': 'prison', 'jail': 'prison', 'incarceration': 'prison', 'prison break': 'prison',
    'heists': 'heist',
    'dragons': 'dragon',
    'wizards': 'wizard', 'sorcerer': 'wizard', 'sorcery': 'wizard',
    'kungfu': 'kung fu',
    'martial art': 'martial arts', 'karate': 'martial arts',
    'mma': 'mixed martial arts', 'cage fighting': 'mixed martial arts',
    'wrestlers': 'wrestling', 'pro wrestling': 'wrestling',
    'nfl': 'american football',
    'motorcycles': 'motorcycle', 'biker': 'motorcycle', 'bikers': 'motorcycle', 'motorbike': 'motorcycle',
    'mob': 'mafia', 'mobster': 'mafia', 'mobsters': 'mafia', 'cosa nostra': 'mafia',
    'cannibals': 'cannibal', 'cannibalism': 'cannibal',
    'spies': 'spy', 'espionage': 'spy', 'secret agent': 'spy',
    'magician': 'magic', 'magicians': 'magic',
}
_KEYWORD_CONCEPT_KEYS = set(KEYWORD_CONCEPTS)
# Valid raw codes accepted verbatim (the LLM often emits the ISO code directly, e.g. require_original_language="zh").
_COUNTRY_CODES = set(COUNTRY_ALIASES.values()) | {c for cs in COUNTRY_REGIONS.values() for c in cs}
_LANGUAGE_CODES = {c for cs in LANGUAGE_ALIASES.values() for c in cs}


def resolve_facet(phrase, kind):
    """Resolve one free LLM facet phrase to its canonical stored value(s). Returns (values, note).

    kind='country'  → [ISO 3166-1 alpha-2 …]  ("French"→['FR']; "Scandinavian"→['SE','DK','NO','FI','IS'])
    kind='language' → [ISO 639-1 …]           ("Chinese"→['zh','cn']; "fr"→['fr'])
    kind='format'   → [canonical attr key …]  ("black & white"→['black and white'])
    kind='keyword'  → [canonical concept …]   ("submarines"→['submarine']; "chess"→['chess'])

    Strategy mirrors resolve_person/resolve_mood: normalize (lowercase, accent-fold, punctuation→space
    via _norm_name) → exact alias-map hit → for country/language also accept a raw ISO code verbatim
    (uppercase/lowercase respectively). A miss returns ([], 'no match') and the caller drops it +
    reports it, so the rest of the query still runs. No fuzzy/substring matching (overturned #4)."""
    norm = _norm_name(phrase)
    if not norm:
        return [], 'empty'
    if kind == 'country':
        if norm in COUNTRY_REGIONS:
            return list(COUNTRY_REGIONS[norm]), 'region'
        if norm in COUNTRY_ALIASES:
            return [COUNTRY_ALIASES[norm]], 'exact'
        code = norm.upper()
        if len(code) == 2 and code in _COUNTRY_CODES:
            return [code], 'iso'
        return [], 'no match'
    if kind == 'language':
        if norm in LANGUAGE_ALIASES:
            return list(LANGUAGE_ALIASES[norm]), 'exact'
        if len(norm) == 2 and norm in _LANGUAGE_CODES:
            return [norm], 'iso'
        return [], 'no match'
    if kind == 'format':
        if norm in FORMAT_ALIASES:
            return [FORMAT_ALIASES[norm]], 'exact'
        return [], 'no match'
    if kind == 'keyword':
        if norm in _KEYWORD_CONCEPT_KEYS:   # canonical concept key emitted verbatim
            return [norm], 'exact'
        if norm in KEYWORD_CONCEPT_ALIASES:
            return [KEYWORD_CONCEPT_ALIASES[norm]], 'alias'
        return [], 'no match'
    return [], f'unknown kind {kind!r}'


# ── Mode-2 synthesis: genome tag → anchor movies (mirror evaluate._get_anchor_titles) ──
def anchors_for(ctx, genome_tags, exclude, per_tag=ANCHORS_PER_TAG, max_total=None):
    """Top `per_tag` representative real movies per genome tag (descending genome relevance
    over the whole corpus), globally de-duplicated against `exclude` and each other, capped
    at `max_total` anchors overall when given. Returns (anchor_titles, unresolved_tags).
    Mirrors src/evaluate.py's anchor derivation so Mode-2 reuses the tested machinery rather
    than a synthetic query vector; `per_tag`/`max_total` add the Mode-1 subordination budget."""
    fs = ctx.fs
    name_to_idx = ctx.genome_name_to_idx
    anchor_titles, unresolved = [], []
    seen = set(exclude)
    for tag in genome_tags:
        if max_total is not None and len(anchor_titles) >= max_total:
            break
        if tag not in name_to_idx:
            unresolved.append(tag)
            continue
        ti = name_to_idx[tag]
        ordered = sorted(
            fs['top_movies'],
            key=lambda mid: float(fs['movieId_to_genome_tag_context'][mid][ti]),
            reverse=True,
        )
        n = 0
        for mid in ordered:
            if n >= per_tag or (max_total is not None and len(anchor_titles) >= max_total):
                break
            title = fs['movieId_to_title'][mid]
            if title not in seen:
                anchor_titles.append(title)
                seen.add(title)
                n += 1
    return anchor_titles, unresolved


def _genome_relevance(ctx, tags):
    """Mean genome relevance (0–1) over the resolvable `tags`, as a [N] float tensor aligned to
    ctx.all_ids (the corpus order of ctx.all_embs). Returns None if no tag resolves. Feeds the soft
    genome re-rank (GENOME_RERANK_LAMBDA), lifting films that actually carry the requested tag."""
    cols = [ctx.genome_name_to_idx[t] for t in tags if t in ctx.genome_name_to_idx]
    if not cols:
        return None
    gctx = ctx.fs['movieId_to_genome_tag_context']
    vals = [sum(float(gctx[mid][c]) for c in cols) / len(cols) for mid in ctx.all_ids]
    return torch.tensor(vals, dtype=torch.float32)


# ── Engine-2 vibe/affect: mood phrase → genome tags (step B lever 1) ─────────
# The 1,128 genome tags already carry a full affect/tone vocabulary (the item content tower + the
# genome-derived LLM-132 vector both encode mood), so "make me cry" / "cozy" / "darker" is a ROUTING
# problem, not a new-data one: map the raw mood phrase to a small set of genome tags and feed them
# into the SAME anchor + re-rank machinery a genome_tags request uses (soft, never a hard filter —
# per the plan the facet filter is the wrong tool for affect). Every tag below is verified in-vocab
# in serving/ (OOV synonyms like "uplifting"/"cozy"/"mature" are mapped onto the in-vocab tag).
MOOD_TAGS = {
    'cry':          ['heartbreaking', 'emotional', 'poignant', 'tragedy', 'sad but good'],
    'feel-good':    ['feel-good', 'heartwarming', 'happy ending'],
    'funny':        ['humor', 'funny', 'quirky'],
    'dark':         ['dark', 'gritty', 'bleak'],
    'scary':        ['creepy', 'scary', 'atmospheric', 'tense'],
    'suspenseful':  ['suspense', 'tense'],
    'mind-bending': ['mindfuck', 'cerebral', 'twist ending', 'nonlinear'],
    'epic':         ['epic'],
    'whimsical':    ['quirky', 'whimsical'],
    'romantic':     ['romance', 'love story'],
    'psychological':['psychological', 'dark'],
    'disturbing':   ['disturbing', 'bleak'],
    'atmospheric':  ['atmospheric'],
    'surreal':      ['surreal', 'nonlinear'],
    'nostalgic':    ['nostalgic', 'nostalgia', 'bittersweet', 'coming-of-age'],
}
# Trigger phrase (normalized, whole-word) → canonical mood key above. Multi-word triggers are matched
# as contiguous token runs (mirrors _franchise_match's whole-word discipline, so "warm" ⊄ "warmth").
MOOD_ALIASES = {
    'cry': 'cry', 'make me cry': 'cry', 'tearjerker': 'cry', 'tear jerker': 'cry', 'weepy': 'cry',
    'sad': 'cry', 'emotional': 'cry', 'heartbreaking': 'cry', 'moving': 'cry', 'poignant': 'cry',
    'feel good': 'feel-good', 'feel-good': 'feel-good', 'feelgood': 'feel-good', 'cozy': 'feel-good',
    'cosy': 'feel-good', 'warm': 'feel-good', 'comforting': 'feel-good', 'wholesome': 'feel-good',
    'uplifting': 'feel-good', 'cheer me up': 'feel-good', 'cheering up': 'feel-good',
    'heartwarming': 'feel-good', 'happy': 'feel-good',
    'funny': 'funny', 'fun': 'funny', 'light': 'funny', 'lighthearted': 'funny',
    'light hearted': 'funny', 'light-hearted': 'funny', 'comedic': 'funny', 'hilarious': 'funny',
    'humor': 'funny', 'humorous': 'funny', 'silly': 'funny', 'goofy': 'funny',
    'dark': 'dark', 'darker': 'dark', 'mature': 'dark', 'serious': 'dark', 'grim': 'dark',
    'heavy': 'dark', 'gritty': 'dark', 'bleak': 'dark',
    'scary': 'scary', 'creepy': 'scary', 'frightening': 'scary', 'spooky': 'scary',
    'eerie': 'scary', 'chilling': 'scary', 'unsettling': 'scary',
    'suspenseful': 'suspenseful', 'suspense': 'suspenseful', 'tense': 'suspenseful',
    'gripping': 'suspenseful', 'thrilling': 'suspenseful', 'thriller': 'suspenseful',
    'nail biting': 'suspenseful', 'riveting': 'suspenseful',
    'mind bending': 'mind-bending', 'mind-bending': 'mind-bending', 'mindbending': 'mind-bending',
    'cerebral': 'mind-bending', 'trippy': 'mind-bending', 'twisty': 'mind-bending',
    'thought provoking': 'mind-bending', 'thought-provoking': 'mind-bending',
    'epic': 'epic', 'grand': 'epic', 'sweeping': 'epic', 'sprawling': 'epic',
    'whimsical': 'whimsical', 'quirky': 'whimsical', 'offbeat': 'whimsical', 'charming': 'whimsical',
    'romantic': 'romantic', 'romance': 'romantic', 'love story': 'romantic', 'swoony': 'romantic',
    'psychological': 'psychological', 'psychologically': 'psychological',
    'disturbing': 'disturbing', 'depressing': 'disturbing', 'harrowing': 'disturbing',
    'atmospheric': 'atmospheric', 'moody': 'atmospheric', 'brooding': 'atmospheric',
    'surreal': 'surreal', 'dreamlike': 'surreal', 'dream like': 'surreal', 'abstract': 'surreal',
    'nostalgic': 'nostalgic', 'nostalgia': 'nostalgic', 'wistful': 'nostalgic',
    'bittersweet': 'nostalgic', 'sentimental': 'nostalgic',
    # Comparative (-er) forms of the base moods — "scarier"/"funnier" are single tokens so the whole-word
    # matcher misses them (unlike "more scary", which already hits the base 'scary'). "darker" is above.
    'scarier': 'scary', 'creepier': 'scary', 'funnier': 'funny', 'sadder': 'cry', 'grimmer': 'dark',
}


def resolve_mood(phrase):
    """Resolve one free mood phrase to a de-duplicated list of genome tags (Engine-2 routing).

    Normalizes the phrase (lowercase, accent-fold, punctuation→space via _norm_name), then collects
    every MOOD_ALIASES trigger that appears as a whole-word contiguous token run — so "something that
    makes me cry" hits `cry`, "darker and more mature" hits `dark` (twice, de-duped). Whole-word (not
    substring) matching keeps "warm" out of "warmth" and mirrors the franchise resolver. Returns the
    union of the matched moods' MOOD_TAGS (empty if nothing matches — the caller then just drops it,
    like an unresolved title). Soft signal only: the caller merges these into genome_tags."""
    norm_tokens = _norm_name(phrase).split()
    if not norm_tokens:
        return []
    canon, tags = [], []
    for trig, c in MOOD_ALIASES.items():
        if c not in canon and _token_subseq(norm_tokens, _norm_name(trig)):
            canon.append(c)
    for c in canon:
        for t in MOOD_TAGS[c]:
            if t not in tags:
                tags.append(t)
    return tags


def _resolve_mood_slots(extraction, hc):
    """Collect + resolve every mood phrase carried by an extraction: a top-level `mood`/`vibe` slot
    and a `hard_constraints.mood` (soft affect the extractor may nest beside a hard filter). Each may
    be a string or a list of strings. Returns the de-duplicated union of resolved genome tags.
    (`exclude_mood` is deliberately NOT handled here — negative-affect exclusion is a separate,
    hard-exclude sibling, now handled by _resolve_exclude_slots below.)"""
    phrases = []
    for src in (extraction.get('mood'), extraction.get('vibe'), (hc or {}).get('mood')):
        if isinstance(src, str):
            phrases.append(src)
        elif isinstance(src, (list, tuple)):
            phrases.extend(p for p in src if isinstance(p, str))
    tags = []
    for p in phrases:
        for t in resolve_mood(p):
            if t not in tags:
                tags.append(t)
    return tags


def _resolve_exclude_slots(hc):
    """Genome tags to HARD-exclude, collected from a hard_constraints block: `exclude_mood` phrases
    routed through resolve_mood (the same MOOD_TAGS table the positive mood path uses) plus
    `exclude_genome_tags` named verbatim. Each slot may be a string or a list of strings. Returns the
    de-duplicated union (empty if neither slot is present). The caller turns these into the
    GENOME_EXCLUDE_CEILING anti-floor — soft moods (genome_tags / mood) never reach here, only the
    explicit hard-exclude slots (mirrors how only require_genome_tags reaches the require floor)."""
    tags = []
    em = (hc or {}).get('exclude_mood')
    for phrase in ([em] if isinstance(em, str) else (em or [])):
        if isinstance(phrase, str):
            for t in resolve_mood(phrase):
                if t not in tags:
                    tags.append(t)
    egt = (hc or {}).get('exclude_genome_tags')
    for t in ([egt] if isinstance(egt, str) else (egt or [])):
        if isinstance(t, str) and t not in tags:
            tags.append(t)
    return tags


# ── Mode-1.5: liked title → its own discriminative genome tags (step B lever 3) ──
# Non-discriminative genome tags to skip when injecting a title's tags — pure acclaim / meta / craft
# labels that describe HOW GOOD a film is, not WHAT IT IS ABOUT, and so re-summon prestige-era
# neighbours (a 1994 "masterpiece"/"imdb top 250" query pulls Forrest Gump right back). Content tags
# that merely contain a craft word ("space opera", "twist ending", "based on a true story") are NOT
# here — they stay eligible. Prefix families (oscar…, saturn award…, best of…, imdb top…) are caught
# by MODE15_STOP_PREFIX so every award/list variant is covered without enumerating them.
MODE15_STOP_EXACT = {
    'masterpiece', 'classic', 'cult classic', 'cult film', 'cult', 'criterion', 'overrated',
    'underrated', 'great movie', 'great acting', 'good acting', 'exceptional acting', 'bad acting',
    'great dialogue', 'good dialogue', 'great ending', 'powerful ending', 'great music', 'good music',
    'great soundtrack', 'good soundtrack', 'awesome soundtrack', 'notable soundtrack',
    'great cinematography', 'amazing cinematography', 'cinematography', 'storytelling', 'writing',
    'entertaining', 'boring', 'predictable', 'beautiful', 'beautifully filmed', 'visually appealing',
    'visually stunning', 'stunning', "so bad it's good", "so bad it's funny", 'oscar winner',
    # production / provenance meta — describes a film's release status, not its content (an Oldboy
    # remake's top tags are "remake"/"original", which drag in unrelated films by provenance).
    'remake', 'original', 'sequel', 'sequels', 'prequel', 'franchise', 'trilogy', 'directorial debut',
}  # every entry is an actual genome-vocab tag (the check is against tag NAMES) — no inert strings.
MODE15_STOP_PREFIX = ('oscar', 'saturn award', 'best of', 'imdb top', 'potential oscar')


def _title_genome_tags(ctx, titles, per_title=MODE15_TAGS_PER_TITLE,
                       min_rel=MODE15_TAG_MIN_REL, max_tags=MODE15_MAX_TAGS):
    """The discriminative genome tags a set of liked titles most strongly carry, for Mode-1.5 re-rank
    injection. For each title, take its top `per_title` genome tags by relevance (descending), keep
    only those ≥ `min_rel` and NOT in the acclaim stoplist, and union across titles (de-duped, order
    preserved) up to `max_tags`. Craft/acclaim tags are dropped so the injected query describes the
    films' SUBJECT (hit men, dark humor, wuxia, dystopia) rather than their prestige — otherwise the
    era-neighbours we are trying to suppress come straight back. Returns [] if nothing qualifies."""
    fs = ctx.fs
    gctx = fs['movieId_to_genome_tag_context']
    idx_to_name = {i: n for n, i in ctx.genome_name_to_idx.items()}
    out = []
    for title in titles:
        mid = fs['title_to_movieId'].get(title)
        if mid is None:
            continue
        vec = gctx[mid]
        ranked = sorted(idx_to_name, key=lambda c: float(vec[c]), reverse=True)
        kept = 0
        for c in ranked:
            if kept >= per_title or len(out) >= max_tags:
                break
            if float(vec[c]) < min_rel:
                break  # tags are relevance-sorted, so nothing below qualifies either
            name = idx_to_name[c]
            if name in MODE15_STOP_EXACT or name.startswith(MODE15_STOP_PREFIX):
                continue
            if name not in out:
                out.append(name)
                kept += 1
    return out


# ── Facet helpers (people union + franchise matching + numeric coercion) ─────
def _people_union(roles):
    """The set of TMDB person IDs a facet-store role dict covers: actors ∪ directors ∪ writers ∪
    composers. Composers are folded in so "scored by Hans Zimmer" resolves through the same
    require_people membership path. None/empty → empty set. `.get` keeps it safe on a pre-composer store."""
    if not roles:
        return set()
    return (set(roles.get('actors', [])) | set(roles.get('directors', []))
            | set(roles.get('writers', [])) | set(roles.get('composers', [])))


def _as_number(v):
    """Coerce an LLM-supplied numeric constraint to float, or None if absent/non-numeric — so a
    stringified bound (max_runtime:"90") degrades gracefully instead of raising on comparison
    (mirrors the year gate's try/except philosophy)."""
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _token_subseq(name_tokens, phrase):
    """True if the whitespace tokens of normalized `phrase` appear as a CONTIGUOUS whole-word
    subsequence of `name_tokens`. Whole-word matching (not raw substring) is what keeps
    'saw'⊄'chainsaw' and 'the ring'⊄'the lord of the rings' — see _franchise_match."""
    pt = phrase.split()
    if not pt:
        return False
    n = len(pt)
    return any(name_tokens[i:i + n] == pt for i in range(len(name_tokens) - n + 1))


def _franchise_match(coll, spec, aliases):
    """True if a movie's TMDB collection `coll` ({'id','name'} or None) matches `spec`.

    spec == True → member of ANY collection (the "no sequels / nothing from a franchise" case).
    spec == str / [str…] → the collection NAME contains any requested phrase as a whole-word token
    run, first expanding a cinematic-universe alias ("mcu" → its member-collection token-phrases,
    since TMDB has no single MCU collection) via `aliases`. Matching is on an EXPLICIT curated name
    space via CONTIGUOUS-TOKEN membership (never raw substring — that over-matches: 'saw'⊂'chainsaw',
    'ted'⊂'enchanted', 'it'⊂65 collections)."""
    if coll is None:
        return False
    if spec is True:
        return True
    name_tokens = _norm_name(coll.get('name') or '').split()
    if not name_tokens:
        return False
    phrases = [spec] if isinstance(spec, str) else list(spec)
    subs = []
    for p in phrases:
        np = _norm_name(p)
        if np:
            subs.extend(_norm_name(s) for s in aliases.get(np, [np]))
    return any(_token_subseq(name_tokens, s) for s in subs)


# ── Post-filter (year + genre + people + F1 structured facets) ───────────────
def _passes_constraints(mid, fs, hc, facets=None):
    """True if movie `mid` satisfies the hard constraints. Every gate treats absent metadata as
    'no info' and passes it (mirrors the year gate) — a hard filter never drops a film merely for
    a missing scraped value.

    People constraints use `require_people_ids` / `exclude_people_ids` (TMDB person IDs resolved
    upstream in recommend(), composers included) checked against the facet store's people union;
    require = ALL present (mirrors require_genres), exclude = ANY. The F1 structured facets —
    runtime, US content-rating ceiling, vote_average floor, franchise/collection — read the baked
    `facets` tables (see llm_features/build_facet_store.py)."""
    if not hc:
        return True
    facets = facets or {}
    year_min, year_max = hc.get('year_min'), hc.get('year_max')
    if year_min is not None or year_max is not None:
        raw_year = fs['movieId_to_year'].get(mid)
        try:
            year = int(raw_year)
            if year_min is not None and year < year_min:
                return False
            if year_max is not None and year > year_max:
                return False
        except (TypeError, ValueError):
            pass  # no usable year → don't filter it out
    genres = set(fs['movieId_to_genres'].get(mid, []))
    require = hc.get('require_genres') or []
    exclude = hc.get('exclude_genres') or []
    require_or = hc.get('require_genres_or') or []  # step C: blend AND→OR relax (ANY of these)
    if require and not all(g in genres for g in require):
        return False
    if require_or and not any(g in genres for g in require_or):
        return False
    if exclude and any(g in genres for g in exclude):
        return False

    # Runtime window (max_runtime / min_runtime) — numeric, guarded like the year gate against a
    # non-numeric LLM-supplied bound (e.g. max_runtime:"90") that would otherwise raise on compare.
    rt_max, rt_min = _as_number(hc.get('max_runtime')), _as_number(hc.get('min_runtime'))
    if rt_max is not None or rt_min is not None:
        rt = (facets.get('movieId_to_runtime') or {}).get(mid)
        if rt is not None:
            if rt_max is not None and rt > rt_max:
                return False
            if rt_min is not None and rt < rt_min:
                return False

    # US content-rating ceiling (require_max_rating = 'PG'/'PG-13'/…): drop films rated stricter.
    max_rating = hc.get('require_max_rating')
    if max_rating:
        ceil = MPAA_ORDER.get(str(max_rating).strip().upper())
        cert = (facets.get('movieId_to_content_rating') or {}).get(mid)
        if ceil is not None and cert is not None:
            r = MPAA_ORDER.get(cert)
            if r is not None and r > ceil:
                return False

    # US content-rating FLOOR (require_min_rating = 'R'/'PG-13'/…): the mirror of the ceiling — a POSITIVE
    # maturity ask ("R-rated comedies", "adult animation") wants films AT LEAST this mature, so drop films
    # rated TAMER. Same absence contract as the ceiling/year gate (unknown cert → keep) so it never over-prunes.
    min_rating = hc.get('require_min_rating')
    if min_rating:
        floor = MPAA_ORDER.get(str(min_rating).strip().upper())
        cert = (facets.get('movieId_to_content_rating') or {}).get(mid)
        if floor is not None and cert is not None:
            r = MPAA_ORDER.get(cert)
            if r is not None and r < floor:
                return False

    # Quality floor (min_vote_average = "actually good"): drop films below the TMDB score.
    min_vote = _as_number(hc.get('min_vote_average'))
    if min_vote is not None:
        v = (facets.get('movieId_to_vote') or {}).get(mid)
        if v is not None and v.get('average') is not None and v['average'] < min_vote:
            return False

    # Franchise / collection membership (require_franchise keeps; exclude_franchise drops).
    req_fr, exc_fr = hc.get('require_franchise'), hc.get('exclude_franchise')
    if req_fr or exc_fr:
        coll    = (facets.get('movieId_to_collection') or {}).get(mid)
        aliases = facets.get('franchise_universe_aliases') or {}
        if req_fr and not _franchise_match(coll, req_fr, aliases):
            return False
        if exc_fr and _franchise_match(coll, exc_fr, aliases):
            return False

    # Origin/nationality country (require_country_codes = ISO 3166-1 resolved upstream). movieId_to_countries
    # is baked from TMDB origin_country, NOT production_countries — the latter lists every co-FINANCIER, so a
    # US film with a sliver of foreign financing wrongly satisfied "French/Japanese films" (see build_facet_store).
    # A film passes if its origin codes intersect the required set (ANY — a genuine multi-national co-production
    # where the required country is a real co-ORIGIN still counts, and a region like "Scandinavian" is any member
    # code; a mere co-financier does NOT). Two absence levels, distinguished so we keep the membership semantics
    # without breaking graceful degradation: (a) the WHOLE table missing (no bake / an old serving artifact,
    # ctx.facets None) → skip the gate entirely, like the year gate, so the rest of the query still runs (mirrors
    # the people path's 'no facet store' degrade — otherwise EVERY film would drop and the pool would empty);
    # (b) the table present but THIS film has no origin country (~81 films: 69 with no scraped TMDB record + 12
    # whose TMDB omits origin_country; ~99.1% whole-corpus coverage) → DROP, since "French films" is an explicit
    # membership demand and a metadata-gap film must not float in masquerading as a member.
    req_country = hc.get('require_country_codes') or []
    ctab = facets.get('movieId_to_countries')
    if req_country and ctab:
        countries = ctab.get(mid)
        if not countries or not (set(countries) & set(req_country)):
            return False

    # Original language (require_language_codes = ISO 639-1 resolved upstream). "original French,
    # nothing dubbed" → original_language ∈ the set. Same two-level absence as country: whole table
    # missing → skip (graceful); present but this film's language unknown → DROP.
    req_lang = hc.get('require_language_codes') or []
    ltab = facets.get('movieId_to_language')
    if req_lang and ltab:
        lang = ltab.get(mid)
        if lang is None or lang not in req_lang:
            return False

    # Format / attribute keywords (require_attribute_keys = canonical attr keys resolved upstream), e.g.
    # black-and-white / woman-director / based-on-a-book. ALL required attrs must be present (mirrors
    # require_genres). Whole table missing → skip (graceful). Present → a MISSING attribute keyword is
    # 'film lacks it' and DROPS the film: these are high-precision crowd tags applied when the trait
    # holds, and an emphatic "black and white ONLY" wants precision. (Recall ≤ coverage — an untagged
    # member is a false negative; documented in facet_store_plan "Keyword recall gaps".)
    req_attrs = hc.get('require_attribute_keys') or []
    atab = facets.get('movieId_to_attributes')
    if req_attrs and atab is not None:
        attrs = set(atab.get(mid) or [])
        if not all(a in attrs for a in req_attrs):
            return False

    # Keyword CONTENT concepts (require/exclude_keyword_concept_keys = canonical concepts resolved upstream),
    # e.g. chess / submarine / heist — a HARD boolean membership pre-filter over the curated TMDB-keyword store
    # (KEYWORD_CONCEPTS). require = ANY present (OR); exclude = NONE present (ANY hit drops). require is OR —
    # NOT AND like require_genres — because a genre pair (horror-comedy) routinely co-occurs in one film, but two
    # distinct CONCRETE topics almost never do: "a boxing or MMA fighter" wants ['boxing','mixed martial arts'] as
    # ALTERNATIVES, and an AND-intersection there is vacuously empty (no film is tagged BOTH), which read as a
    # zero-result loss in the 500-query run. A multi-concept list is virtually always alternatives, so ANY-of is
    # the faithful default; a single concept is unaffected (any-of-one == all-of-one). Same two-level absence as
    # the other facets: whole table missing (ctx.facets None / old artifact) → skip the gate (graceful, else every
    # film drops); present but this film carries no concepts → its set is empty, so it FAILS a require (an explicit
    # "about X" demand must not admit a film with no such tag) and PASSES an exclude. Recall ≤ keyword coverage —
    # an untagged member is a false negative, same precision/recall trade as the format facet.
    req_kw = hc.get('require_keyword_concept_keys') or []
    exc_kw = hc.get('exclude_keyword_concept_keys') or []
    ktab = facets.get('movieId_to_keyword_concepts')
    if (req_kw or exc_kw) and ktab is not None:
        concepts = set(ktab.get(mid) or [])
        if req_kw and not any(k in concepts for k in req_kw):
            return False
        if exc_kw and any(k in concepts for k in exc_kw):
            return False

    # People (actors/directors/writers/composers), resolved to IDs upstream.
    require_p = hc.get('require_people_ids') or []
    exclude_p = hc.get('exclude_people_ids') or []
    if require_p or exclude_p:
        people = _people_union((facets.get('movieId_to_people') or {}).get(mid))
        if require_p and not all(p in people for p in require_p):
            return False
        if exclude_p and any(p in people for p in exclude_p):
            return False
    return True


# ── Genre-diversity selection (step C thread 2, L3) ──────────────────────────
def _select_diverse(eligible, fs, wanted, blend_genres, top_n, scores):
    """Pick up to `top_n` from score-sorted `eligible` [(i, mid, title)…] with genre diversification.

    Greedy: at each slot pick the highest-value remaining candidate, where value = its score minus,
    for a BLEND ask, MMR_SIGNATURE_LAMBDA × (# already-picked films sharing its requested-genre
    signature). Skip a candidate that would push an UNREQUESTED co-genre over its rolling per-prefix
    cap (COGENRE_CAP_FRAC) — with a relax-to-best fallback if the cap blocks everything, so the list
    never comes up short (degrade, never empty). The signature penalty interleaves the {G1}/{G2}/{G1,G2}
    genre signatures of a blend so it doesn't collapse to N identical both-genre films; a non-blend
    (empty `blend_genres`) gets no penalty and stays score-ordered (concentrated). `scores` is indexed
    by corpus position i (== eligible item[0]). Returns the chosen [(i, mid, title)…] in output order."""
    m2g = fs['movieId_to_genres']
    bset = set(blend_genres)
    gsets = {item[1]: set(m2g.get(item[1], ())) for item in eligible}
    remaining = list(eligible)
    picked, counts, sig_count = [], {}, {}

    while remaining and len(picked) < top_n:
        cap = math.ceil((len(picked) + 1) * COGENRE_CAP_FRAC)
        best_i, best_val = None, None
        for idx, item in enumerate(remaining):
            gset = gsets[item[1]]
            if any(g not in wanted and counts.get(g, 0) + 1 > cap for g in gset):
                continue  # co-genre cap
            val = scores[item[0]]
            if bset:
                val -= MMR_SIGNATURE_LAMBDA * sig_count.get(frozenset(gset & bset), 0)
            if best_val is None or val > best_val:
                best_i, best_val = idx, val
        if best_i is None:                       # co-genre cap blocked all → relax to best remaining
            best_i = 0
        item = remaining.pop(best_i)
        gset = gsets[item[1]]
        picked.append(item)
        for g in gset:
            counts[g] = counts.get(g, 0) + 1
        if bset:
            sig = frozenset(gset & bset)
            sig_count[sig] = sig_count.get(sig, 0) + 1
    return picked


# ── Recommend ────────────────────────────────────────────────────────────────
def _content_similar_scores(ctx, title):
    """'Movies like X' ranking: cosine of every catalog movie to the seed title in the raw GENOME
    content space (the Similar tab's "Genome Tags → Raw features" cell). Returns a [N] tensor aligned
    to ctx.all_ids, or None if the seed isn't in the matrix (caller then falls back to the user tower).
    The seed scores 1.0 but is dropped downstream via seed_titles, exactly as probe_similar skips
    candidate == mid. We rank in genome space — NOT the combined item embedding (which clusters by
    release era / co-watch, so Toy Story drifts to its 1995 cohort) and NOT a 1-movie mock through the
    user tower (trained to predict the NEXT item, not content-similar ones). Genome space keeps 'like
    Toy Story' → the Pixar films, 'like Alien' → sci-fi horror."""
    mid = ctx.fs['title_to_movieId'].get(title)
    if mid is None:
        return None
    try:
        i = ctx.all_ids.index(mid)
    except ValueError:
        return None
    q = ctx.genome_sim_matrix[i:i + 1]                    # [1, 1128], already unit-norm
    return (ctx.genome_sim_matrix @ q.T).squeeze(-1)      # [N] cosine scores, aligned to ctx.all_ids


def recommend(ctx, extraction, top_n=TOP_N):
    """Run one extraction object through the full v1 pipeline. Returns a report dict
    (UI-agnostic — the harness CLI and the Streamlit tab each render it their own way)."""
    fs = ctx.fs
    liked_raw    = extraction.get('liked_items') or []
    disliked_raw = extraction.get('disliked_items') or []
    genome_tags  = extraction.get('genome_tags') or []
    liked_genres = extraction.get('liked_genres') or []
    disliked_genres = extraction.get('disliked_genres') or []
    hc = extraction.get('hard_constraints') or {}

    # Engine-2 mood routing (step B lever 1): resolve any free mood/vibe phrase → genome tags. Kept
    # SEPARATE from the subject genome_tags on purpose — "a feel-good movie about cooking" wants the
    # subject (food) to rank, with feel-good only nudging; folding both into one mean re-rank lets the
    # tone axis outweigh the subject. So mood gets its own re-rank term (5b) and only drives the Mode-2
    # anchors when it is the SOLE vibe (no explicit genome_tags — the pure "make me cry" case). Soft
    # signal throughout, never a hard filter (per the plan, the facet filter is the wrong tool for affect).
    mood_tags = _resolve_mood_slots(extraction, hc)

    # Negative affect / anti-vibe exclusion (exclude_mood + exclude_genome_tags): the anti-vibe siblings
    # of resolve_mood / require_genome_tags. "Absolutely nothing dark", "I cannot handle gore" resolve to
    # genome tags (mood phrases via the MOOD_TAGS table, exclude_genome_tags verbatim) that become a HARD
    # anti-floor below (5d) — dropping films that strongly carry them. Emphatic exclusion is a gate, not a nudge.
    exclude_tags = _resolve_exclude_slots(hc)

    # 1. Resolve mentioned titles (Mode 1).
    liked_resolved, disliked_resolved = [], []
    resolution_log = {'liked': [], 'disliked': []}
    for raw in liked_raw:
        canon, note = resolve_title(raw, ctx)
        resolution_log['liked'].append((raw, canon, note))
        if canon:
            liked_resolved.append(canon)
    for raw in disliked_raw:
        canon, note = resolve_title(raw, ctx)
        resolution_log['disliked'].append((raw, canon, note))
        if canon:
            disliked_resolved.append(canon)

    # Mode-1.5 title-genome injection (step B lever 3): a PURE-TITLE request (a liked title but no
    # explicit vibe) drifts to release-era / co-watch neighbours under cosine alone, so inject the
    # liked titles' own discriminative genome tags as SOFT re-rank tags (never anchors — that would
    # over-expand the query). Fires only on a true pure-title: skipped when the extraction already
    # carries genome_tags OR a resolved mood (an explicit vibe — "I loved X but I want to cry" — wins).
    mode15_tags = (_title_genome_tags(ctx, liked_resolved)
                   if (liked_resolved and not genome_tags and not mood_tags) else [])

    # Pure single-title "movies like X": exactly one named title and NO other taste signal (no vibe, no
    # genre, no dislikes) → rank by genome content-space cosine to the seed (Similar-tab semantics), NOT
    # by pushing a 1-movie 'history' through the user tower. Hard constraints still post-filter and the
    # seed is excluded via seed_titles. This path is PURE content cosine, so we drop the Mode-1.5 genome
    # injection (which existed only to steady the user-tower drift this path avoids by construction).
    # Two+ titles keep the user-tower centroid path (a genuine multi-seed taste, not a single-item lookup).
    pure_single_title = (len(liked_resolved) == 1 and not disliked_resolved
                         and not genome_tags and not mood_tags
                         and not liked_genres and not disliked_genres)
    if pure_single_title:
        mode15_tags = []

    # 1b. Resolve people facets (Phase 1: HARD require/exclude only). Names → canonical TMDB
    #     person IDs via the facet store; unresolved names (or no facet store at all) are reported
    #     and dropped, so the rest of the query still runs. The model has no person concept, so
    #     people are expressed purely as a post-filter on the corpus (see _passes_constraints).
    #     Composers ride the same filter path (the plan's "composer = a require_people extension"):
    #     the facet store folds composer credits into the people membership union, so a resolved
    #     composer ID filters exactly like a required actor/director. But they resolve through a
    #     SEPARATE name namespace (role='composer') so a require_people name never shadows to a
    #     same-named composer, and vice-versa (see resolve_person).
    people_log = {'require': [], 'exclude': []}
    require_pids, exclude_pids = [], []
    _PEOPLE_SLOTS = (
        ('require', 'require_people',    None),
        ('require', 'require_composers', 'composer'),
        ('exclude', 'exclude_people',    None),
        ('exclude', 'exclude_composers', 'composer'),
    )
    for bucket, hc_key, role in _PEOPLE_SLOTS:
        for raw in (hc.get(hc_key) or []):
            # No facet store (an old serving artifact without the bake) → report every name as
            # unresolved so the drop is visible, rather than silently skipping the loop.
            pid, note = resolve_person(raw, ctx.facets, role) if ctx.facets else (None, 'no facet store')
            people_log[bucket].append((raw, pid, note))
            if pid is not None:
                (require_pids if bucket == 'require' else exclude_pids).append(pid)
    hc = {**hc, 'require_people_ids': require_pids, 'exclude_people_ids': exclude_pids}

    # 1c. Resolve country/language/format facets (Engine-1a membership, step C). Free LLM phrases
    #     ("French", "in Chinese", "black and white") → canonical stored values via resolve_facet;
    #     require_language and require_original_language are synonyms (both gate original_language).
    #     Each slot may be a string or list. Unresolved phrases are reported and dropped, like
    #     unresolved people/titles. The resolved values feed _passes_constraints as require_country_codes /
    #     require_language_codes / require_attribute_keys. Like people, these are a post-filter on the
    #     corpus — the model has no nationality/language/format concept.
    facet_codes = {'country': [], 'language': [], 'attribute': []}
    facet_log = {'country': [], 'language': [], 'attribute': []}
    _FACET_SLOTS = (
        ('country',   'country',  ('require_country',)),
        ('language',  'language', ('require_language', 'require_original_language')),
        ('attribute', 'format',   ('require_attributes',)),
    )
    for bucket, kind, hc_keys in _FACET_SLOTS:
        for hc_key in hc_keys:
            raw = hc.get(hc_key)
            for phrase in ([raw] if isinstance(raw, str) else (raw or [])):
                if not isinstance(phrase, str):
                    continue
                vals, note = resolve_facet(phrase, kind)
                facet_log[bucket].append((phrase, vals, note))
                for v in vals:
                    if v not in facet_codes[bucket]:
                        facet_codes[bucket].append(v)
    hc = {**hc,
          'require_country_codes':  facet_codes['country'],
          'require_language_codes': facet_codes['language'],
          'require_attribute_keys': facet_codes['attribute']}

    # 1c-bis. Resolve keyword CONTENT concepts (require/exclude_keyword_concepts → canonical concept keys via
    #     resolve_facet(kind='keyword')). Concrete-noun "movies about chess" / "no zombie movies" intent → a
    #     HARD boolean membership gate on the curated TMDB-keyword store (KEYWORD_CONCEPTS), distinct from the
    #     genome floor (award/meta-polluted on niche topics) and from require_genres (chess is not a genre).
    #     Each slot is a string or list; unresolved phrases are reported + dropped like the other facets.
    kw_codes = {'require': [], 'exclude': []}
    kw_log   = {'require': [], 'exclude': []}
    for bucket, hc_key in (('require', 'require_keyword_concepts'), ('exclude', 'exclude_keyword_concepts')):
        raw = hc.get(hc_key)
        for phrase in ([raw] if isinstance(raw, str) else (raw or [])):
            if not isinstance(phrase, str):
                continue
            vals, note = resolve_facet(phrase, 'keyword')
            kw_log[bucket].append((phrase, vals, note))
            for v in vals:
                if v not in kw_codes[bucket]:
                    kw_codes[bucket].append(v)
    hc = {**hc,
          'require_keyword_concept_keys': kw_codes['require'],
          'exclude_keyword_concept_keys': kw_codes['exclude']}

    # 1d. Genre-pool degradation (step C thread 2 — see the L1/L2/L3 constants). Record the WANTED
    #     genre set (require_genres ∪ liked_genres) for the L1 re-rank + L3 diversity caps, and decide
    #     whether a multi-genre require_genres must relax AND→OR. INTERSECTION-FIRST: a dual-genre
    #     "X and Y" ask means films that are BOTH ("a horror comedy" = Shaun of the Dead, not a mix of
    #     separate horror and comedy films), so keep strict AND while the X∩Y pool can rank a result.
    #     Relax to OR only when that intersection is too thin to rank (< GENRE_AND_POOL_MIN films, e.g.
    #     Sci-Fi∩Film-Noir = 2) — degrade to the union rather than return almost nothing.
    require_genres_orig = hc.get('require_genres') or []
    wanted_genres = set(liked_genres) | set(require_genres_orig)
    blend = False
    if len(require_genres_orig) >= 2:
        and_pool = sum(1 for mg in fs['movieId_to_genres'].values()
                       if all(g in mg for g in require_genres_orig))
        blend = and_pool < GENRE_AND_POOL_MIN
    if blend:
        hc = {**hc, 'require_genres': [], 'require_genres_or': require_genres_orig}

    # 2. Mode-2 synthesis: genome tags → anchor movies (exclude already-named seeds).
    #    When the user named real titles (Mode 1) the anchors are subordinated so they refine
    #    rather than swamp the explicit likes; pure Mode 2 keeps the full anchor strength. The subject
    #    genome_tags drive the anchors; a mood falls back to driving them only when it is the sole vibe
    #    (no explicit genome_tags), so "make me cry" still synthesizes a query but "feel-good cooking"
    #    anchors on food, not on feel-good. A PURE require_genome_tags request (a hard subject/setting
    #    floor with NO soft companion — "movies about racing" → require_genome_tags ['racing']) ALSO
    #    seeds the anchors: otherwise it has no taste signal, collapses to the popularity fallback, and the
    #    0.35 floor (a binary gate) lets incidental blockbusters through — Star Wars racing=0.36 clears it
    #    and popularity floats it over real racing films (Rush=1.0). Anchoring on the required tag ranks by
    #    taste WITHIN the floor. Scoped to the no-soft-signal case, so Mode-1 (likes) queries are untouched.
    require_gt = hc.get('require_genome_tags') or []
    anchor_tags = genome_tags if genome_tags else (require_gt if require_gt else mood_tags)
    seed_exclude = set(liked_resolved) | set(disliked_resolved)
    has_likes = bool(liked_resolved)
    per_tag      = ANCHORS_PER_TAG_WITH_LIKES if has_likes else ANCHORS_PER_TAG
    max_total    = MAX_ANCHORS_WITH_LIKES if has_likes else None
    anchor_weight = ANCHOR_WEIGHT_WITH_LIKES if has_likes else ANCHOR_MOVIE_WEIGHT
    anchors, unresolved_tags = anchors_for(ctx, anchor_tags, seed_exclude, per_tag, max_total)

    # 3. Assemble model input: explicit likes (2.0) + anchors (1.0, or 0.5 when subordinated),
    #    dislikes (-2.0).
    liked_with_weights = (
        [(t, LIKED_MOVIE_WEIGHT) for t in liked_resolved] +
        [(t, anchor_weight)      for t in anchors]
    )

    # Validate genre names against the model's vocabulary (surface, don't crash).
    genre_vocab = set(fs['genres_ordered'])
    unknown_genres = sorted(
        (set(liked_genres) | set(disliked_genres)
         | set(hc.get('require_genres') or []) | set(hc.get('require_genres_or') or [])
         | set(hc.get('exclude_genres') or []))
        - genre_vocab)

    # Empty-signal fallback → popularity (a sensible, diverse default).
    fallback = not liked_with_weights and not disliked_resolved \
        and not liked_genres and not disliked_genres

    # 4-5. Build the user embedding + score the whole corpus (raw dot product == cosine; both L2-normed;
    #      no alpha/temp) — but ONLY when there's a taste signal to rank by. On the empty-signal fallback
    #      the popularity branch (step 6) ignores these scores, so building the embedding and scoring the
    #      corpus is pure wasted work; skip the forward pass entirely and leave raw_scores None.
    raw_scores = None
    ranked_by_similarity = False   # True → 'movies like X' ranked by item-item cosine, not the user tower
    if not fallback:
        if pure_single_title:
            # 'movies like X' — genome content-space cosine off the seed (see _content_similar_scores).
            raw_scores = _content_similar_scores(ctx, liked_resolved[0])
            ranked_by_similarity = raw_scores is not None
        if raw_scores is None:  # not a pure single title, OR seed missing from the matrix → user tower
            with torch.no_grad():
                user_emb = build_user_embedding(
                    ctx.model, fs, liked_with_weights, disliked_resolved, ctx.ts_inference,
                    liked_genres=liked_genres, disliked_genres=disliked_genres,
                    disliked_movie_value=DISLIKED_MOVIE_WEIGHT,
                )
            raw_scores = (ctx.all_embs @ user_emb.T).squeeze(-1)

    # 5b. Soft genome re-rank (Source A — see GENOME_RERANK_LAMBDA). Lift films that actually carry
    #     the requested genome tags so a correct anchor/title set isn't out-ranked by the item
    #     embedding's era/co-watch neighbours. THREE separate additive terms so orthogonal axes don't
    #     dilute each other (see MOOD_RERANK_LAMBDA):
    #       • soft subject: explicit genome_tags + Mode-1.5 title tags (GENOME_RERANK_LAMBDA).
    #       • hard subject: require_genome_tags, a SEPARATE stronger term (REQUIRE_GT_RERANK_LAMBDA) so
    #         aboutness — not co-watch cosine — orders films WITHIN the hard floor (thin-axis fix).
    #       • mood:         the Engine-2 affect tags routed from a mood/vibe phrase (MOOD_RERANK_LAMBDA).
    if not fallback:
        # soft subject term: explicit genome_tags + Mode-1.5 title tags (require_gt gets its own term below).
        subject_tags = list(genome_tags) + list(mode15_tags)
        if GENOME_RERANK_LAMBDA and subject_tags:
            subj = _genome_relevance(ctx, subject_tags)
            if subj is not None:
                raw_scores = raw_scores + GENOME_RERANK_LAMBDA * subj.to(raw_scores.device)
        # hard subject term: require_genome_tags re-rank, stronger than the soft term so subject relevance
        # (not co-watch cosine) is the primary ordering WITHIN the 0.35 floor — fixes the thin-axis
        # 'movies about chess' displacement; near-uniform on a dense axis (racing) so anchor order holds.
        if REQUIRE_GT_RERANK_LAMBDA and require_gt:
            reqv = _genome_relevance(ctx, require_gt)
            if reqv is not None:
                raw_scores = raw_scores + REQUIRE_GT_RERANK_LAMBDA * reqv.to(raw_scores.device)
        # mood term: Engine-2 affect, gated by its OWN knob so it survives GENOME_RERANK_LAMBDA=0.
        if MOOD_RERANK_LAMBDA and mood_tags:
            mood_boost = _genome_relevance(ctx, mood_tags)
            if mood_boost is not None:
                raw_scores = raw_scores + MOOD_RERANK_LAMBDA * mood_boost.to(raw_scores.device)
        # genre term (step C thread 2 L1): lift films carrying the WANTED genres so an under-specified
        # genre ask (only liked_genres, no anchors) doesn't drown in popular off-genre films. Fraction
        # of wanted genres present, so a fuller genre match rises more. Its own knob (GENRE_RERANK_LAMBDA).
        if GENRE_RERANK_LAMBDA and wanted_genres:
            wl = list(wanted_genres)
            gvec = torch.tensor(
                [sum(1 for g in wl if g in fs['movieId_to_genres'].get(mid, ())) / len(wl)
                 for mid in ctx.all_ids],
                dtype=torch.float32)
            raw_scores = raw_scores + GENRE_RERANK_LAMBDA * gvec.to(raw_scores.device)

    # 5c. require_genome_tags HARD floor (step B lever 2). Gate the pool to films that genuinely carry
    #     the required tags, then drop the rest in BOTH ranking loops. Semantics mirror the sibling
    #     hard gates in _passes_constraints (require_genres / require_people = ALL): a film must clear
    #     GENOME_HARD_FLOOR on EVERY required tag, NOT on their average — so a compound "set in Paris
    #     during WWII" require doesn't admit a WWII film with ~0 Paris relevance (a mean would). A
    #     required-tag set that resolves to nothing in vocab → no floor (gt_floor_ok stays None),
    #     mirroring the year gate's "don't filter on what we can't compute." Soft moods/Mode-1.5 tags
    #     never reach here — only hard_constraints.require_genome_tags.
    gt_floor_ok = None  # None → no floor; else the set of mids clearing the floor on EVERY required tag
    if require_gt:
        floor_cols = [ctx.genome_name_to_idx[t] for t in require_gt if t in ctx.genome_name_to_idx]
        if floor_cols:
            gt_ctx = fs['movieId_to_genome_tag_context']
            gt_floor_ok = {mid for mid in ctx.all_ids
                           if all(float(gt_ctx[mid][c]) >= GENOME_HARD_FLOOR for c in floor_cols)}

    # 5d. Anti-vibe HARD anti-floor (exclude_mood / exclude_genome_tags — see GENOME_EXCLUDE_CEILING).
    #     The mirror of the require floor: drop any film that STRONGLY carries an excluded tag (relevance
    #     ≥ ceiling on ANY excluded tag — an emphatic "nothing dark AND nothing gory" cuts a film that is
    #     heavy on EITHER). Out-of-vocab excluded tags contribute nothing; an all-OOV set → empty set → no gate.
    gt_exclude_bad = set()
    if exclude_tags:
        ex_cols = [ctx.genome_name_to_idx[t] for t in exclude_tags if t in ctx.genome_name_to_idx]
        if ex_cols:
            ex_ctx = fs['movieId_to_genome_tag_context']
            gt_exclude_bad = {mid for mid in ctx.all_ids
                              if any(float(ex_ctx[mid][c]) >= GENOME_EXCLUDE_CEILING for c in ex_cols)}

    # 6. Rank → drop only USER-NAMED seeds → apply post-filters → take top_n.
    #    Liked/disliked titles are films the user explicitly named, so we don't surface them back
    #    (classic "don't recommend what they already know"). Genome anchors are the opposite:
    #    SYNTHESIZED representatives of a mood the user never named — excluding them would hide
    #    exactly the on-the-nose films the request asks for (a "western vibe" should be able to
    #    surface the canonical westerns it anchored on; a future facet seed like Tom Hanks →
    #    Cast Away should be recommendable). So anchors stay eligible for the output.
    seed_titles = set(liked_resolved) | set(disliked_resolved)
    facets = ctx.facets  # baked facet store (people + F1 structured tables), or None if unbaked
    order = raw_scores.argsort(descending=True).tolist() if not fallback else None
    diversify = bool(wanted_genres)   # step C thread 2 L3: only genre-oriented asks get diversified

    def _run_select(hc_, gt_floor_ok_, diversify_):
        """Collect up to top_n recs under the given (possibly relaxed) hard constraints + genome floor.
        Shared by the popularity-fallback and model-ranked paths, and re-called by the relaxation ladder
        below. gt_exclude_bad / seed_titles / exclusions are captured fixed — never relaxed. Returns
        (recs, filtered_count)."""
        out, filt = [], 0
        if fallback:
            title_to_mid = fs['title_to_movieId']
            for title in fs['popularity_ordered_titles']:
                mid = title_to_mid.get(title)
                if mid is None or title in seed_titles:
                    continue
                if gt_floor_ok_ is not None and mid not in gt_floor_ok_:
                    filt += 1; continue
                if mid in gt_exclude_bad:
                    filt += 1; continue
                if not _passes_constraints(mid, fs, hc_, facets):
                    filt += 1; continue
                out.append((title, fs['movieId_to_genres'].get(mid, []),
                            fs['movieId_to_year'].get(mid), None))
                if len(out) >= top_n:
                    break
            return out, filt
        # model-ranked path: collect eligible in score order; scan a bounded pool when diversifying so
        # the backfill has room to inject under-represented genres (see _select_diverse).
        scan_cap = max(CANDIDATE_POOL, top_n * 4) if diversify_ else top_n
        eligible = []
        for i in order:
            mid = ctx.all_ids[i]
            title = fs['movieId_to_title'][mid]
            if title in seed_titles:
                continue
            if gt_floor_ok_ is not None and mid not in gt_floor_ok_:
                filt += 1; continue
            if mid in gt_exclude_bad:
                filt += 1; continue
            if not _passes_constraints(mid, fs, hc_, facets):
                filt += 1; continue
            eligible.append((i, mid, title))
            if len(eligible) >= scan_cap:
                break
        chosen = (_select_diverse(eligible, fs, wanted_genres,
                                  require_genres_orig if blend else [], top_n, raw_scores.tolist())
                  if diversify_ else eligible[:top_n])
        for i, mid, title in chosen:
            out.append((title, fs['movieId_to_genres'].get(mid, []),
                        fs['movieId_to_year'].get(mid), float(raw_scores[i])))
        return out, filt

    recs, filtered = _run_select(hc, gt_floor_ok, diversify)

    # Graceful relaxation: when the hard filters empty the pool, progressively drop the SOFTEST require
    # gates — attributes → genome-tag floor → genre → keyword topic — keeping user IDENTITY (people,
    # franchise, rating, year) and every exclude_ gate intact, until something surfaces. The order runs
    # modifier→core: format (indie/b&w) and the genome vibe/setting floor go first ("dark gritty western"
    # with no matches keeps WESTERN, drops the vibe), then the genre label, and the concrete keyword topic
    # is dropped LAST ("comedy heist" with no matches keeps HEIST, drops comedy — a topic defines the ask
    # more than its genre). Identity gates are never relaxed (relaxing "Zendaya" or "PG for my kid"
    # defeats the request — an empty pool there is the honest answer). Only fires on an EMPTY pool; a
    # thin-but-valid result (e.g. 5 Zendaya films) is left untouched. Each relaxed rung is recorded so
    # the UI/trace can label the output "closest matches (relaxed: …)".
    relaxed_constraints = []
    if not recs:
        cur_hc, cur_floor = dict(hc), gt_floor_ok
        for name in ('require_attributes', 'require_genome_tags', 'require_genres', 'require_keyword_concepts'):
            applied = False
            if name == 'require_genome_tags' and cur_floor is not None:
                cur_floor = None; applied = True
            elif name == 'require_attributes' and cur_hc.get('require_attribute_keys'):
                cur_hc = {k: v for k, v in cur_hc.items() if k != 'require_attribute_keys'}; applied = True
            elif name == 'require_genres' and (cur_hc.get('require_genres') or cur_hc.get('require_genres_or')):
                cur_hc = {k: v for k, v in cur_hc.items() if k not in ('require_genres', 'require_genres_or')}; applied = True
            elif name == 'require_keyword_concepts' and cur_hc.get('require_keyword_concept_keys'):
                cur_hc = {k: v for k, v in cur_hc.items() if k != 'require_keyword_concept_keys'}; applied = True
            if not applied:
                continue
            relaxed_constraints.append(name)
            recs, filtered = _run_select(cur_hc, cur_floor, False)
            if recs:
                break

    return {
        'extraction': extraction,
        'resolution': resolution_log,
        'people_resolution': people_log,
        'facet_resolution': facet_log,   # country/language/format phrase → resolved codes (step C)
        'anchors': anchors,
        'anchor_weight': anchor_weight,
        'mood_tags': mood_tags,          # genome tags routed from a free mood/vibe phrase (Engine-2)
        'mode15_tags': mode15_tags,      # discriminative genome tags injected from a pure-title seed
        'ranked_by_similarity': ranked_by_similarity,  # 'movies like X' → item-item cosine (Similar-tab)
        'unresolved_tags': unresolved_tags,
        'unknown_genres': unknown_genres,
        'seed_count': len(seed_titles),
        'fallback': fallback,
        'filtered': filtered,
        'relaxed_constraints': relaxed_constraints,  # soft gates dropped to rescue an empty pool (else [])
        'recs': recs,
    }
