"""
Stage 2 (schema) — Derive the LLM extraction dimensions from genome tags
                    (LLM-vs-genome ablation)

Turns data/top_genome_tags_by_discriminability.csv (1,128 genome tags ranked by
std_score = separating power) into the deduped, bucketed dimension list the LLM
extraction will score per movie, and writes it to data/llm_schema_dimensions.json
for schemas.py (Pydantic) + prompts.py to consume. See
docs/plans/llm_vs_genome_ablation_plan.md (Stage 2, "Derive the schema from genome
discriminability").

WHY derive instead of hand-invent: the experiment asks "can an LLM match genome on
the SAME content axes?". Inventing dimensions would measure different axes and make
"LLM ≈ genome" meaningless. So every dimension here traces back to a high-
discriminability genome tag (its rank/std are carried into the output for audit).

The derivation is judgment-heavy — synonym merges, drops, and bucketing are
decisions, not a formula — so the decisions live as EXPLICIT, inspectable data
below (GROUPS + DROPPED), and this script's job is the mechanical part: validate
that every referenced tag exists in the CSV, enrich each dimension with its genome
stats, and report three things that keep the judgment honest —
  • deep-sourced dims        (best source ranks below the top-~150 primary pool;
                              expected for SETTING & ERA and the factual families,
                              which the plan flags as under-represented in the head)
  • group-5 scrape backing   (which prestige dims actually have a scraped signal vs.
                              the ones deferred in Stage 1 — flagged, not hidden)
  • top-150 accounting       (every tag in the primary pool is either mapped to a
                              dimension or in DROPPED with a reason — nothing silently
                              falls through)

The six groups mirror the plan's anti-fatigue split (one focused LLM call each):
THEMES & PLOT, TONE & MOOD, SETTING, ERA & SUB-GENRE, PROVENANCE & STRUCTURE, FACTUAL
RECEPTION & PRESTIGE, VISUAL MEDIUM (factual-medium descriptors only — subjective
aesthetics like 'cinematography'/'visually stunning' are dropped, they would be
hallucinated from a synopsis).

Usage (standalone — not part of the main.py pipeline CLI; pure code, no GPU / API):
    python llm_features/derive_schema.py
"""
import json
import os

import pandas as pd


# ── Paths ────────────────────────────────────────────────────────────────────

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RANKING   = os.path.join(REPO_ROOT, 'data', 'top_genome_tags_by_discriminability.csv')
OUT_PATH  = os.path.join(REPO_ROOT, 'data', 'llm_schema_dimensions.json')


# ── Derivation knobs ─────────────────────────────────────────────────────────

# The plan's "top ~150 by std_score" primary pool. Dims whose best (lowest-rank)
# source tag sits beyond this are 'deep-sourced' — legitimate but reported, since
# SETTING & ERA and the factual PROVENANCE / RECEPTION / VISUAL families are highly
# discriminative yet rank below the sentiment-heavy head (the plan calls this out).
PRIMARY_POOL_RANK = 150

# Scrape-backing tags for group-5 dims (mirrors what scrape.py actually stores vs.
# what Stage 1 deferred). Surfaced per dim so the "no scrape backing" prestige dims
# are flagged, not silently trusted — see the plan's Reception/prestige validity note.
#   tmdb_budget    — details_raw.budget                      (production-budget scale)
#   tmdb_vote      — vote_average / vote_count               (quality PROXY only)
#   wikipedia_text — full_extract / reception (accolades prose, not a structured field)
#   deferred       — explicit Oscar / Criterion fields were NOT scraped in Stage 1


# ── The schema (explicit, inspectable judgment) ──────────────────────────────
#
# Each dimension is (canonical_name, [genome source tags it derives from / merges]).
# The first/lowest-rank source is the dimension's anchor; extra tags are the
# deduped synonyms folded into it (e.g. the four spellings of sci-fi, the three of
# "based on a book"). Names are snake_case → become Pydantic field names downstream.

# (1) THEMES & PLOT — content-derivable from plot text.
THEMES_DIMS = [
    ('murder',           ['murder']),
    ('serial_killer',    ['serial killer']),
    ('obsession',        ['obsession']),
    ('loneliness',       ['loneliness', 'solitude']),
    ('revenge',          ['revenge', 'vengeance']),
    ('relationships',    ['relationships']),
    ('romance',          ['romance', 'romantic', 'love story', 'love']),
    ('betrayal',         ['betrayal']),
    ('survival',         ['survival']),
    ('redemption',       ['redemption']),
    ('coming_of_age',    ['coming of age', 'adolescence', 'childhood',
                          'teen', 'teens', 'teen movie', 'teenager']),
    ('family',           ['family', 'family drama', 'children', 'kids',
                          'kids and family']),
    ('friendship',       ['friendship', 'unlikely friendships']),
    ('corruption',       ['corruption']),
    ('greed',            ['greed']),
    # Crime mega-bucket split into sub-genres (see SETTING note on the granularity
    # restoration): the old single 'crime' dim folded heist/gangster/mafia together,
    # making Model B coarser than genome's distinct tags. Each now stands alone.
    ('crime',            ['crime', 'crime gone awry']),
    ('heist',            ['heist']),
    ('gangster',         ['gangster', 'gangsters', 'mafia', 'organized crime']),
    ('hitman',           ['hitman', 'assassin']),
    ('conspiracy',       ['conspiracy', 'paranoia', 'surveillance']),
    ('espionage',        ['espionage', 'spy', 'spying']),
    ('politics',         ['politics', 'political']),
    ('social_commentary', ['social commentary', 'allegory']),
    ('existentialism',   ['existentialism', 'philosophy', 'philosophical']),
    ('insanity',         ['insanity', 'psychology', 'psychological']),
    ('supernatural',     ['supernatural', 'magic']),
    ('mystery',          ['mystery', 'investigation', 'detective', 'murder mystery']),
    ('secrets',          ['secrets']),
    ('destiny',          ['destiny']),
    ('sacrifice',        ['sacrifice']),
    ('mortality',        ['death', 'life & death', 'tragedy']),
    ('addiction',        ['drugs', 'drinking']),
]

# (2) TONE & MOOD — inferable from text, calibration-sensitive.
TONE_DIMS = [
    ('tense',            ['tense', 'suspense', 'suspenseful', 'intense']),
    ('dark',             ['dark']),
    ('bleak',            ['bleak', 'downbeat', 'grim', 'depressing', 'depression']),
    ('melancholic',      ['melancholic', 'melancholy', 'bittersweet', 'poignant']),
    ('emotional',        ['emotional', 'heartbreaking', 'touching']),
    ('feel_good',        ['feel-good', 'heartwarming', 'feel good movie']),
    ('comedic',          ['comedy', 'funny', 'hilarious', 'humorous', 'very funny',
                          'humor', 'goofy', 'silly']),
    ('dark_humor',       ['dark humor']),
    ('satirical',        ['satire', 'satirical', 'parody']),
    ('quirky',           ['quirky', 'whimsical', 'eccentricity']),
    ('absurd',           ['absurd']),
    ('weird',            ['weird']),
    ('surreal',          ['surreal', 'surrealism']),
    ('cerebral',         ['cerebral', 'complex', 'thought-provoking', 'intellectual']),
    ('enigmatic',        ['enigmatic']),
    ('intimate',         ['intimate']),
    ('atmospheric',      ['atmospheric']),
    ('creepy',           ['creepy', 'eerie', 'ominous']),
    ('scary',            ['scary']),
    ('violent',          ['violence', 'violent', 'brutal', 'brutality']),
    ('gory',             ['gory', 'bloody', 'blood', 'splatter']),
    ('disturbing',       ['disturbing', 'macabre']),
    ('fast_paced',       ['fast paced', 'action packed']),
    ('stylish',          ['stylish', 'stylized']),
    ('reflective',       ['reflective', 'meditative']),
    ('nostalgic',        ['nostalgic']),
]

# (3) SETTING, ERA & SUB-GENRE — factually extractable. Mostly deep-sourced
# (ranks > 150): the era/place and sub-genre axes are highly discriminative but rank
# below the sentiment head.
#
# GRANULARITY RESTORATION (why the sub-genre block exists): the top-150-by-std_score
# derivation systematically DROPS sharp sub-genre tags — a tag like 'zombies' (#810)
# or 'cyberpunk' (#658) is 0.0 for ~99% of films, so its corpus std is low even though
# it is exactly the axis that drives "recommend more like this". Model A (genome) keeps
# all 1,128 tags including these; the original 116-dim schema kept none of them, so
# Model B competed with ~10x fewer, coarser content axes — a confound unrelated to LLM
# extraction quality. These dims (every one a real genome tag → still "same axes" as A)
# restore that granularity. NB 'sword and sorcery' is intentionally absent: it is NOT a
# genome tag, so it cannot be grounded here (it belongs to a future LLM-invents-axes
# B' arm); its concept is folded into 'high_fantasy'.
SETTING_DIMS = [
    ('world_war_ii',     ['world war ii', 'wwii', 'nazis', 'holocaust']),
    ('war',              ['war', 'wartime', 'vietnam war']),
    ('cold_war',         ['cold war']),
    ('space',            ['space', 'space travel', 'space opera']),
    ('aliens',           ['alien', 'aliens']),
    ('future',           ['future', 'futuristic', 'technology']),
    ('dystopia',         ['dystopia']),
    ('post_apocalyptic', ['post-apocalyptic', 'apocalypse', 'end of the world']),
    ('time_travel',      ['time travel']),
    ('medieval',         ['medieval']),
    ('western_frontier', ['western']),
    ('historical',       ['historical', 'history', 'period piece', 'us history']),
    ('new_york',         ['new york', 'new york city']),
    ('los_angeles',      ['los angeles']),
    ('britain',          ['england', 'british', 'london']),
    ('france',           ['france', 'paris', 'french']),
    ('japan',            ['japan', 'japanese', 'samurai']),
    ('small_town',       ['small town', 'suburbia']),
    ('school',           ['high school', 'college']),
    ('eighties',         ['1980s']),
    # Sci-fi sub-genres (speculative worlds / devices).
    ('cyberpunk',              ['cyberpunk']),
    ('artificial_intelligence', ['artificial intelligence']),
    ('robots',                 ['robots', 'robot']),
    ('clones',                 ['clones']),
    # Fantasy sub-genres ('sword and sorcery' folds in here — not a genome tag).
    ('high_fantasy',           ['high fantasy', 'fantasy world']),
    ('dragons',                ['dragons']),
    ('wizards',                ['wizards']),
    # Horror sub-genres (creatures / slasher).
    ('slasher',                ['slasher']),
    ('zombies',                ['zombies', 'zombie']),
    ('vampires',               ['vampires', 'vampire']),
    ('ghosts',                 ['ghosts']),
    ('monster',                ['monster']),
]

# (4) PROVENANCE & STRUCTURE — factually extractable from metadata.
PROVENANCE_DIMS = [
    ('based_on_book',      ['based on a book', 'adapted from:book', 'based on book',
                            'books', 'literature', 'adaptation']),
    ('based_on_play',      ['based on a play']),
    ('based_on_comic',     ['based on a comic', 'comic book']),
    ('based_on_tv_show',   ['based on a tv show']),
    ('based_on_video_game', ['based on a video game', 'video game adaptation']),
    ('based_on_true_story', ['based on true story', 'based on a true story',
                            'true story']),
    ('biographical',       ['biographical', 'biopic']),
    ('sequel',             ['sequel', 'sequels', 'good sequel']),
    ('prequel',            ['prequel']),
    ('remake',             ['remake']),
    ('franchise',          ['franchise', 'trilogy']),
    ('nonlinear',          ['nonlinear', 'non-linear']),
    ('twist_ending',       ['twist ending', 'plot twist', 'twist', 'twists & turns']),
    ('multiple_storylines', ['multiple storylines', 'ensemble cast']),
    ('narration',          ['narrated']),
    ('character_study',    ['character study']),
    ('independent_film',   ['independent film', 'indie']),
    ('art_house',          ['art house']),
    ('foreign_language',   ['foreign']),
    ('documentary',        ['documentary']),
    ('mockumentary',       ['mockumentary']),
    ('superhero',          ['superheroes', 'superhero']),
    ('musical',            ['musical']),
    ('fairy_tale',         ['fairy tale', 'mythology']),
]

# (5) FACTUAL RECEPTION & PRESTIGE — mapped from scraped OBJECTIVE indicators, not
# inferred from plot. Each carries its scrape backing (see knobs above); the dims
# whose backing is 'deferred'/'tmdb_vote' are flagged in the summary as the part the
# scrape can't yet substantiate. Pure-sentiment prestige tags (masterpiece, great
# acting, …) are NOT here — they're in DROPPED as the disclosed genome-only residue.
RECEPTION_DIMS = [
    ('oscar_winner',          ['oscar winner', 'oscar (best picture)',
                               'oscar (best directing)', 'oscar (best actor)',
                               'oscar (best actress)'],              'wikipedia_text+deferred'),
    ('oscar_nominated',       ['oscar', 'oscar (best supporting actor)',
                               'oscar (best supporting actress)'],   'wikipedia_text+deferred'),
    ('oscar_foreign_language', ['oscar (best foreign language film)'], 'wikipedia_text+deferred'),
    ('oscar_technical',       ['oscar (best cinematography)',
                               'oscar (best effects - visual effects)',
                               'oscar (best editing)'],              'wikipedia_text+deferred'),
    ('criterion',             ['criterion'],                         'deferred'),
    ('palme_dor',             ['golden palm'],                       'wikipedia_text'),
    ('imdb_top_250',          ['imdb top 250'],                      'tmdb_vote'),
    ('afi_recognized',        ['afi 100', 'afi 100 (movie quotes)',
                               'afi 100 (laughs)'],                  'wikipedia_text'),
    ('classic',               ['classic'],                           'wikipedia_text'),
    ('cult_classic',          ['cult classic', 'cult', 'cult film'], 'wikipedia_text'),
    ('big_budget',            ['big budget'],                        'tmdb_budget'),
]

# (6) VISUAL MEDIUM — FACTUAL medium descriptors only. Deliberately small: the most
# discriminative "visual" genome tags (cinematography, visually stunning, beautifully
# filmed, …) are subjective and would be hallucinated from a synopsis, so they're
# dropped (group-6 rule). Only objectively-checkable production-medium facts remain.
VISUAL_DIMS = [
    ('animated',           ['animation', 'animated', 'cartoon']),
    ('computer_animation', ['computer animation', 'pixar']),
    ('stop_motion',        ['stop motion', 'claymation']),
    ('anime',              ['anime']),
    ('black_and_white',    ['black and white']),
    ('cgi_heavy',          ['cgi', 'bad cgi', 'special effects']),
    ('three_d',            ['3d']),
]

GROUPS = [
    ('themes',     'THEMES & PLOT',                32, THEMES_DIMS),
    ('tone',       'TONE & MOOD',                  25, TONE_DIMS),
    ('setting',    'SETTING, ERA & SUB-GENRE',     32, SETTING_DIMS),
    ('provenance', 'PROVENANCE & STRUCTURE',       25, PROVENANCE_DIMS),
    ('reception',  'FACTUAL RECEPTION & PRESTIGE', 20, RECEPTION_DIMS),
    ('visual',     'VISUAL MEDIUM',                12, VISUAL_DIMS),
]

# Top-150 tags deliberately NOT turned into dimensions, each with a reason. Keeping
# this explicit lets the script prove the primary pool is fully accounted for —
# every top-150 tag is either a dimension source above or dropped here.
DROPPED = [
    # Redundant with the item tower's existing genre one-hot (sub-genre CONTENT is
    # still captured by finer dims — horror→scary/creepy, sci-fi→space/aliens/future,
    # fantasy→fairy_tale, etc.).
    ('action',            'genre one-hot redundant'),
    ('drama',             'genre one-hot redundant'),
    ('dramatic',          'genre one-hot redundant (drama)'),
    ('horror',            'genre one-hot redundant (tone via scary/creepy/disturbing)'),
    ('thriller',          'genre one-hot redundant (tone via tense)'),
    ('adventure',         'genre one-hot redundant'),
    ('sci-fi',            'genre one-hot redundant (setting via space/aliens/future)'),
    ('scifi',             'genre one-hot redundant (spelling of sci-fi)'),
    ('science fiction',   'genre one-hot redundant (spelling of sci-fi)'),
    ('sci fi',            'genre one-hot redundant (spelling of sci-fi)'),
    ('fantasy',           'genre one-hot redundant (content via fairy_tale)'),
    ('romantic comedy',   'genre combo redundant (captured by romance + comedic)'),
    # Pure crowd-sentiment / quality judgments — not derivable from content and not
    # factually scrapeable. This is the genome-only residue disclosed in the plan's
    # Limitations: an LLM cannot ground these from a synopsis.
    ('great acting',      'crowd-sentiment (not content-derivable)'),
    ('good acting',       'crowd-sentiment (not content-derivable)'),
    ('great movie',       'crowd-sentiment (not content-derivable)'),
    ('masterpiece',       'crowd-sentiment (not content-derivable)'),
    ('good action',       'crowd-sentiment (not content-derivable)'),
    ('fun movie',         'crowd-sentiment (not content-derivable)'),
    ('silly fun',         'crowd-sentiment (not content-derivable)'),
    ('predictable',       'crowd-sentiment (not content-derivable)'),
    ('interesting',       'crowd-sentiment (not content-derivable)'),
    ('realistic',         'crowd-sentiment (not content-derivable)'),
    ('storytelling',      'crowd-sentiment / generic (not discriminative content)'),
    ('good soundtrack',   'crowd-sentiment (audio, not derivable from synopsis)'),
    ('girlie movie',      'crowd-sentiment / dated audience label'),
    ('chick flick',       'crowd-sentiment / dated audience label'),
    ('talky',             'crowd-sentiment / style judgment'),
    ('forceful',          'crowd-sentiment (vague)'),
    ('visceral',          'crowd-sentiment (vague)'),
    ('affectionate',      'crowd-sentiment (vague)'),
    ('artistic',          'subjective aesthetic'),
    ('imagination',       'subjective / vague'),
    # Subjective VISUAL — group-6 rule: cinematography quality can't be read off a
    # synopsis; these would be hallucinated.
    ('cinematography',    'subjective visual (hallucinated from synopsis)'),
    ('visually stunning', 'subjective visual (hallucinated from synopsis)'),
    ('visually appealing', 'subjective visual (hallucinated from synopsis)'),
    ('beautifully filmed', 'subjective visual (hallucinated from synopsis)'),
    ('beautiful scenery', 'subjective visual (hallucinated from synopsis)'),
    ('breathtaking',      'subjective visual (hallucinated from synopsis)'),
    ('visual',            'subjective visual / generic'),
    # Generic filler — too broad to discriminate.
    ('life',              'generic filler'),
    ('story',             'generic filler'),
    # Action set-pieces — low semantic value, genre-adjacent; tone covered by
    # fast_paced / violent.
    ('gunfight',          'action set-piece (genre-adjacent)'),
    ('chase',             'action set-piece (genre-adjacent)'),
    ('fight scenes',      'action set-piece (genre-adjacent)'),
    # No home in the 6-bucket schema.
    ('pg-13',             'MPAA rating — no audience group in the 6-bucket schema'),
    ('pornography',       'NSFW noise — not a useful recommendation content axis'),
]


# ── Lookup / validation ──────────────────────────────────────────────────────

def load_ranking() -> dict:
    """
    Load the discriminability CSV into {tag_name: {rank, std_score, mean_score,
    max_score}} for O(1) genome-stat lookup while enriching the schema below.
    """
    df = pd.read_csv(RANKING)
    return {
        row.tag_name: {
            'rank':       int(row.rank),
            'std_score':  float(row.std_score),
            'mean_score': float(row.mean_score),
            'max_score':  float(row.max_score),
        }
        for row in df.itertuples()
    }


def validate_tags(ranking: dict) -> None:
    """
    Fail loudly if any dimension source tag or dropped tag is not a real genome tag
    in the CSV — a typo here would silently drop a feature axis, so it's a hard stop
    BEFORE anything is written.
    """
    referenced = [t for _, _, _, dims in GROUPS for _, tags in
                  ((d[0], d[1]) for d in dims) for t in tags]
    referenced += [t for t, _ in DROPPED]

    unknown = sorted({t for t in referenced if t not in ranking})
    if unknown:
        print("✗ ERROR — these referenced tags are not in the ranking CSV "
              "(typos? fix before writing):")
        for t in unknown:
            print(f"    {t!r}")
        raise SystemExit(1)


# ── Enrichment ───────────────────────────────────────────────────────────────

def enrich_dim(name: str, tags: list, ranking: dict, backing: str = None) -> dict:
    """
    Attach each source tag's genome stats to a dimension, compute its anchor rank
    (lowest rank = strongest source), and mark whether it is deep-sourced (anchor
    below the top-PRIMARY_POOL_RANK head). Source tags are returned sorted by rank
    so the anchor reads first.
    """
    sources = sorted(
        ({'tag': t, 'rank': ranking[t]['rank'], 'std_score': ranking[t]['std_score']}
         for t in tags),
        key=lambda s: s['rank'],
    )
    best_rank = sources[0]['rank']
    dim = {
        'name':         name,
        'genome_tags':  sources,
        'anchor_rank':  best_rank,
        'deep_sourced': best_rank > PRIMARY_POOL_RANK,
    }
    if backing is not None:
        dim['scrape_backing'] = backing
    return dim


def build_schema(ranking: dict) -> dict:
    """Assemble the full output structure (groups → enriched dims, plus dropped)."""
    groups_out = []
    for key, title, target, dims in GROUPS:
        enriched = [enrich_dim(d[0], d[1], ranking, d[2] if len(d) > 2 else None)
                    for d in dims]
        groups_out.append({
            'key':        key,
            'title':      title,
            'target':     target,
            'n':          len(enriched),
            'dimensions': enriched,
        })

    dropped_out = sorted(
        ({'tag': t, 'rank': ranking[t]['rank'],
          'std_score': ranking[t]['std_score'], 'reason': r}
         for t, r in DROPPED),
        key=lambda d: d['rank'],
    )

    total = sum(g['n'] for g in groups_out)
    return {
        'schema_version':         'v1',
        'source_csv':             os.path.relpath(RANKING, REPO_ROOT),
        'primary_pool_rank':      PRIMARY_POOL_RANK,
        'n_dimensions':           total,
        'n_groups':               len(groups_out),
        'groups':                 groups_out,
        'dropped':                dropped_out,
    }


# ── Reporting ────────────────────────────────────────────────────────────────

def top150_accounting(schema: dict, ranking: dict) -> list:
    """
    Every genome tag in the top-PRIMARY_POOL_RANK primary pool should be either a
    dimension source or in DROPPED. Return the ranks/tags that are NEITHER, so a
    silently-skipped head tag can't hide. (Deep-sourced dims pull in tags from below
    the pool too — those are fine and not part of this check.)
    """
    accounted = {s['tag'] for g in schema['groups'] for d in g['dimensions']
                 for s in d['genome_tags']}
    accounted |= {d['tag'] for d in schema['dropped']}

    head = {tag: meta['rank'] for tag, meta in ranking.items()
            if meta['rank'] <= PRIMARY_POOL_RANK}
    return sorted(((rank, tag) for tag, rank in head.items() if tag not in accounted))


def print_summary(schema: dict, ranking: dict) -> None:
    """Full bucketed list + the three audit reports, for in-terminal review."""
    print("\n" + "═" * 78)
    print("DERIVED LLM EXTRACTION SCHEMA  (genome-traced, deduped, bucketed)")
    print("═" * 78)

    for g in schema['groups']:
        print(f"\n── ({g['key']}) {g['title']}  —  {g['n']} dims (target ~{g['target']})")
        for d in g['dimensions']:
            anchor = d['genome_tags'][0]
            syns = [s['tag'] for s in d['genome_tags'][1:]]
            flag = ' [deep]' if d['deep_sourced'] else ''
            back = f"  {{{d['scrape_backing']}}}" if 'scrape_backing' in d else ''
            merged = f"   ← {', '.join(syns)}" if syns else ''
            print(f"   {d['name']:<22} #{anchor['rank']:<4} "
                  f"std={anchor['std_score']:.3f}{flag}{back}{merged}")

    # ── Audit 1: deep-sourced dims (anchor below the primary pool) ───────────
    deep = [(g['title'], d['name'], d['anchor_rank'])
            for g in schema['groups'] for d in g['dimensions'] if d['deep_sourced']]
    print("\n" + "─" * 78)
    print(f"DEEP-SOURCED dims (anchor rank > {schema['primary_pool_rank']}): "
          f"{len(deep)}")
    for title, name, rank in sorted(deep, key=lambda x: x[2]):
        print(f"   #{rank:<4} {name:<22} ({title})")

    # ── Audit 2: group-5 scrape backing ─────────────────────────────────────
    reception = next(g for g in schema['groups'] if g['key'] == 'reception')
    print("\n" + "─" * 78)
    print("GROUP-5 (reception/prestige) SCRAPE BACKING:")
    weak = []
    for d in reception['dimensions']:
        backing = d['scrape_backing']
        note = ''
        if 'deferred' in backing:
            note = '  ⚠ explicit signal deferred in Stage 1 scrape'
            weak.append(d['name'])
        elif backing == 'tmdb_vote':
            note = '  ⚠ PROXY only (vote_average, not the IMDb list)'
            weak.append(d['name'])
        elif backing == 'tmdb_budget':
            note = '  ⚠ production-budget signal (mild popularity correlate; gross NOT fed)'
        print(f"   {d['name']:<22} {backing:<24}{note}")
    print(f"   → {len(weak)} dim(s) lack a direct scraped signal: {', '.join(weak)}")

    # ── Audit 3: top-150 accounting ─────────────────────────────────────────
    unaccounted = top150_accounting(schema, ranking)
    print("\n" + "─" * 78)
    print(f"TOP-{schema['primary_pool_rank']} ACCOUNTING "
          f"(every head tag mapped or explicitly dropped):")
    if not unaccounted:
        print("   ✓ all accounted — none silently skipped")
    else:
        print(f"   ⚠ {len(unaccounted)} unaccounted (neither mapped nor dropped):")
        for rank, tag in unaccounted:
            print(f"      #{rank:<4} {tag}")


# ── Orchestrator ─────────────────────────────────────────────────────────────

def run() -> None:
    ranking = load_ranking()
    print(f"Loaded {len(ranking)} ranked genome tags  ←  "
          f"{os.path.relpath(RANKING, REPO_ROOT)}")

    validate_tags(ranking)
    schema = build_schema(ranking)
    print_summary(schema, ranking)

    with open(OUT_PATH, 'w') as f:
        json.dump(schema, f, indent=2, ensure_ascii=False)

    counts = '  '.join(f"{g['key']}={g['n']}" for g in schema['groups'])
    print("\n" + "═" * 78)
    print(f"✓ Wrote {schema['n_dimensions']} dimensions across "
          f"{schema['n_groups']} groups  →  {os.path.relpath(OUT_PATH, REPO_ROOT)}")
    print(f"   per-group: {counts}")
    print(f"   dropped:   {len(schema['dropped'])} top-pool tags "
          f"(crowd-sentiment / genre-redundant / subjective-visual)")


if __name__ == '__main__':
    run()
