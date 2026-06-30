"""
src/llm_frontend.py — shared natural-language → two-tower retrieval core for the LLM
conversational front-end (v1).

PURPOSE
    The single, serving/-only implementation of the v1 pipeline that turns an LLM
    extraction object into recommendations FROM THE TRAINED MODEL:

        extraction JSON (from the LLM — see tools/llm_frontend_prompt.py)
          → resolve named titles to catalog titles (Mode 1, fuzzy)
          → synthesize a Mode-2 query: genome tags → representative anchor movies
          → build the user embedding via the real serving model (src/inference.py)
          → rank the whole corpus (raw dot product == cosine; both L2-normed)
          → apply v1 post-filters (year + genre)
          → top-N recommendations (never raw LLM output).

    This module is imported by BOTH consumers so the logic lives in exactly one place:
      • tools/llm_frontend_probe.py — the in-repo test/QA harness (CLI + --smoke).
      • streamlit_app.py — the conversational tab (reuses the cached load_artifacts()
        model rather than reloading serving/).

    Everything here is serving/-only: it reads exactly what the deployed app loads
    (feature_store.pt, movie_embeddings.pt, model.pth) and never touches data/. The
    functions take a FrontendContext (built once from the loaded artifacts) so the
    Streamlit app can feed its @st.cache_resource Artifacts straight in.

LOCKED-IN v1 POLICY (validated in the Claude Code test loop — see docs/plans/plan.md
"v1 Build Handoff" and memory project_llm_frontend_v1):
    Subordinated hybrid. Pure Mode 2 (no named titles) → 5 anchors/tag at weight 1.0
    (the anchors ARE the query). With named titles → ≤1 anchor/tag, cap 3 total, at
    weight 0.5, so the explicit likes (weight 2.0 each) always dominate the embedding.
    Like 2.0, dislike −2.0; most-recent timestamp bin; rank the full corpus, post-filter,
    then take top-N. Empty extraction → popularity fallback.
"""

import difflib
import re
import unicodedata
from typing import NamedTuple

import numpy as np
import torch

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
FUZZY_CUTOFF          = 0.74   # difflib ratio floor for title resolution (raised from 0.6 — a
                               # wrong-film fuzzy hit poisons the whole anchor; better to drop)
FUZZY_YEAR_TOL        = 4      # reject a fuzzy hit whose catalog year disagrees with the year the
                               # LLM emitted by more than this (Star Wars 1977 ≠ Star Maps 1997)

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


def build_frontend_context(model, fs, all_ids, all_embs, ts_inference) -> FrontendContext:
    """Bundle the loaded serving artifacts into a FrontendContext, deriving the title-resolution
    index, popularity rank, and genome-name→column lookups from fs. Call once after loading (the
    harness does it in Serving; the Streamlit tab builds it from its cached Artifacts)."""
    gn, gi = fs['genome_tag_names'], fs['genome_tag_to_i']
    return FrontendContext(
        model=model,
        fs=fs,
        all_ids=all_ids,
        all_embs=all_embs,
        ts_inference=ts_inference,
        title_index=_build_title_index(fs),
        pop_rank={t: i for i, t in enumerate(fs['popularity_ordered_titles'])},
        genome_name_to_idx={gn[tid]: gi[tid] for tid in gi},
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
# hit suffices — no fuzzy-year guard. See docs/plans/facet_store_plan.md.
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


def resolve_person(raw, facets):
    """Resolve one LLM-extracted person name to a canonical TMDB person ID.

    Returns (pid | None, note). Strategy: exact normalized-name hit in person_name_to_ids.
    Unlike resolve_title there is no fuzzy/year machinery — TMDB person IDs make same-name
    people distinct, so resolution is an exact lookup. On a normalized collision (two real
    people share a name, e.g. two "Michael Williams") prefer the one with more in-corpus films;
    person_name_to_ids is pre-sorted that way at build time, so we take the head. A miss returns
    None and is reported, then dropped by the caller (exactly like an unresolved title).

    `facets` is the baked facet-store dict (person_name_to_ids / person_id_to_name /
    person_id_to_film_count); Phase 1 will carry it on the FrontendContext."""
    norm = _norm_name(raw)
    if not norm:
        return None, 'empty'
    ids = facets['person_name_to_ids'].get(norm)
    if not ids:
        return None, 'no match'
    pid = ids[0]  # pre-sorted: most in-corpus films first (fame / coverage tie-break)
    note = 'exact' if len(ids) == 1 else f'exact, {len(ids)} same-name → most-films'
    return pid, note


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


# ── v1 post-filter (year + genre only; director/rating deferred to v1.5) ─────
def _passes_constraints(mid, fs, hc):
    """True if movie `mid` satisfies the hard constraints. Unknown/non-numeric year
    passes the year gate (no info to filter on)."""
    if not hc:
        return True
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
    if require and not all(g in genres for g in require):
        return False
    if exclude and any(g in genres for g in exclude):
        return False
    return True


# ── Recommend ────────────────────────────────────────────────────────────────
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

    # 2. Mode-2 synthesis: genome tags → anchor movies (exclude already-named seeds).
    #    When the user named real titles (Mode 1) the anchors are subordinated so they refine
    #    rather than swamp the explicit likes; pure Mode 2 keeps the full anchor strength.
    seed_exclude = set(liked_resolved) | set(disliked_resolved)
    has_likes = bool(liked_resolved)
    per_tag      = ANCHORS_PER_TAG_WITH_LIKES if has_likes else ANCHORS_PER_TAG
    max_total    = MAX_ANCHORS_WITH_LIKES if has_likes else None
    anchor_weight = ANCHOR_WEIGHT_WITH_LIKES if has_likes else ANCHOR_MOVIE_WEIGHT
    anchors, unresolved_tags = anchors_for(ctx, genome_tags, seed_exclude, per_tag, max_total)

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
         | set(hc.get('require_genres') or []) | set(hc.get('exclude_genres') or []))
        - genre_vocab)

    # Empty-signal fallback → popularity (a sensible, diverse default).
    fallback = not liked_with_weights and not disliked_resolved \
        and not liked_genres and not disliked_genres

    # 4. Build the user embedding via the real serving model.
    with torch.no_grad():
        user_emb = build_user_embedding(
            ctx.model, fs, liked_with_weights, disliked_resolved, ctx.ts_inference,
            liked_genres=liked_genres, disliked_genres=disliked_genres,
            disliked_movie_value=DISLIKED_MOVIE_WEIGHT,
        )

    # 5. Score whole corpus (raw dot product == cosine; both L2-normed). No alpha/temp.
    raw_scores = (ctx.all_embs @ user_emb.T).squeeze(-1)

    # 6. Rank → drop only USER-NAMED seeds → apply post-filters → take top_n.
    #    Liked/disliked titles are films the user explicitly named, so we don't surface them back
    #    (classic "don't recommend what they already know"). Genome anchors are the opposite:
    #    SYNTHESIZED representatives of a mood the user never named — excluding them would hide
    #    exactly the on-the-nose films the request asks for (a "western vibe" should be able to
    #    surface the canonical westerns it anchored on; a future facet seed like Tom Hanks →
    #    Cast Away should be recommendable). So anchors stay eligible for the output.
    seed_titles = set(liked_resolved) | set(disliked_resolved)
    recs, kept, filtered = [], 0, 0
    if fallback:
        ranked_titles = fs['popularity_ordered_titles']
        title_to_mid = fs['title_to_movieId']
        for title in ranked_titles:
            mid = title_to_mid.get(title)
            if mid is None or title in seed_titles:
                continue
            if not _passes_constraints(mid, fs, hc):
                filtered += 1
                continue
            recs.append((title, fs['movieId_to_genres'].get(mid, []), fs['movieId_to_year'].get(mid), None))
            kept += 1
            if kept >= top_n:
                break
    else:
        order = raw_scores.argsort(descending=True).tolist()
        for i in order:
            if kept >= top_n:
                break
            mid = ctx.all_ids[i]
            title = fs['movieId_to_title'][mid]
            if title in seed_titles:
                continue
            if not _passes_constraints(mid, fs, hc):
                filtered += 1
                continue
            recs.append((title, fs['movieId_to_genres'].get(mid, []),
                         fs['movieId_to_year'].get(mid), float(raw_scores[i])))
            kept += 1

    return {
        'extraction': extraction,
        'resolution': resolution_log,
        'anchors': anchors,
        'anchor_weight': anchor_weight,
        'unresolved_tags': unresolved_tags,
        'unknown_genres': unknown_genres,
        'seed_count': len(seed_titles),
        'fallback': fallback,
        'filtered': filtered,
        'recs': recs,
    }
