"""
tools/ask_examples_spec.py — hand-curated query tree for the Ask tab's example chips.

STRUCTURE
    6 roots × 6 children = 42 queries. Each root showcases one capability family of the
    extraction schema; its children are REFINEMENTS — the natural follow-up asks — each phrased
    as a full standalone query so it (a) pre-generates independently, (b) reads correctly when
    it back-fills the Ask text box, and (c) still works if a user edits it and reruns live.

    id      stable key ('r3', 'r3c2', …) — the artifact and the UI reference queries by id,
            so labels/queries can be reworded without invalidating a pre-generated report.
    label   short chip text (st.pills); the full query is what actually runs.
    query   the standalone natural-language request that was / will be pre-generated.

CAPABILITY MAP (why these six)
    r1  title anchors + year fence + genre exclude + franchise exclude   (Mode 1 / 1.5)
    r2  pure vibe → mood/genome, occasions, rating ceilings              (Mode 2)
    r3  people facets: require/exclude, pairs, person×era, person×franchise
    r4  hard category + era fences, country, keyword topics              (post-filters)
    r5  genome vibe stacks + emphatic excludes (no-gore dread)
    r6  attributes (anime / stop-motion / adult animation) + person + genome setting

CURATION RULES (enforced by the generation + review pass, not by this file)
    - Corpus is MovieLens 32M @200+ ratings, well-covered through ~2019 — no 2020+ recency asks.
    - No studio/streaming asks (unsupported by the extraction schema, silently ignored).
    - Every query must pre-generate with fallback=False, no dropped anchor titles, and a
      coherent top-10 board; weak entries get reworded or swapped here and regenerated
      (python tools/gen_ask_examples.py).
"""

ROOTS = [
    {
        'id': 'r1', 'label': 'Blade Runner vibes',
        'query': ('Slow, atmospheric sci-fi like Blade Runner and Arrival — '
                  'nothing before 2000, and no horror.'),
        'children': [
            {'id': 'r1c1', 'label': 'Slower & headier',
             'query': ('Slow, cerebral sci-fi driven by ideas and dialogue — like Arrival '
                       'and Ex Machina, but even quieter. No action blockbusters.')},
            {'id': 'r1c2', 'label': 'By Villeneuve',
             'query': 'More films by Denis Villeneuve.'},
            {'id': 'r1c3', 'label': 'Outside the franchise',
             'query': ('That Blade Runner feel, but nothing from the Blade Runner '
                       'or Alien franchises.')},
            {'id': 'r1c4', 'label': "The '70s–'80s roots",
             'query': ('Slow, philosophical sci-fi from the 70s and 80s, '
                       'like Stalker and 2001: A Space Odyssey.')},
            {'id': 'r1c5', 'label': 'Set in space',
             'query': ('Contemplative movies set in space like Interstellar and Gravity — '
                       'awe and isolation over action.')},
            {'id': 'r1c6', 'label': 'Bleaker: dystopia',
             'query': 'Bleak dystopian futures like Children of Men and Snowpiercer.'},
        ],
    },
    {
        'id': 'r2', 'label': 'Feel-good night',
        'query': 'Something feel-good and heartwarming for a cozy night in.',
        'children': [
            {'id': 'r2c1', 'label': 'Underdog sports',
             'query': 'A feel-good sports movie where the underdog wins.'},
            {'id': 'r2c2', 'label': 'Quirky indies',
             'query': 'Quirky, offbeat comedies like Little Miss Sunshine and Juno.'},
            {'id': 'r2c3', 'label': 'Happy tears',
             'query': 'A movie that will make me cry happy tears — uplifting, not depressing.'},
            {'id': 'r2c4', 'label': 'Family movie night',
             'query': ('A fun adventure for family movie night — nothing scary, '
                       'fine for a 10-year-old.')},
            {'id': 'r2c5', 'label': 'Modern rom-com',
             'query': 'A warm romantic comedy from the 2010s.'},
            {'id': 'r2c6', 'label': 'Food & warmth',
             'query': 'Movies about food and cooking that feel like a warm hug.'},
        ],
    },
    {
        'id': 'r3', 'label': 'Tom Hanks era',
        'query': 'Tom Hanks movies from the 90s.',
        'children': [
            {'id': 'r3c1', 'label': 'Dramas only',
             'query': 'Tom Hanks dramas — skip the comedies.'},
            {'id': 'r3c2', 'label': 'With Meg Ryan',
             'query': 'The Tom Hanks and Meg Ryan romantic comedies.'},
            {'id': 'r3c3', 'label': 'Spielberg at war',
             'query': "Steven Spielberg's war movies."},
            {'id': 'r3c4', 'label': 'Like Ryan, minus Hanks',
             'query': 'War epics like Saving Private Ryan, but without Tom Hanks.'},
            {'id': 'r3c5', 'label': 'Denzel thrillers',
             'query': '90s thrillers starring Denzel Washington.'},
            {'id': 'r3c6', 'label': 'Nolan, not Batman',
             'query': 'Christopher Nolan films, but skip the Batman ones.'},
        ],
    },
    {
        'id': 'r4', 'label': 'Noir & crime',
        'query': 'Classic film noir from the 1940s and 50s.',
        'children': [
            {'id': 'r4c1', 'label': 'Neo-noir',
             'query': 'Neo-noir like Blood Simple and L.A. Confidential.'},
            {'id': 'r4c2', 'label': 'Hitchcock',
             'query': 'The essential Alfred Hitchcock thrillers.'},
            {'id': 'r4c3', 'label': 'Modern noir vibe',
             'query': 'Recent movies with a noir feel, like Drive and Nightcrawler.'},
            {'id': 'r4c4', 'label': 'Hard-boiled detectives',
             'query': 'Hard-boiled private-detective movies.'},
            {'id': 'r4c5', 'label': 'Heists',
             'query': 'Stylish heist movies where a crew plans one big job.'},
            {'id': 'r4c6', 'label': 'Korean crime',
             'query': 'Korean crime thrillers like Oldboy and Memories of Murder.'},
        ],
    },
    {
        'id': 'r5', 'label': 'Twisty thrillers',
        'query': 'Dark psychological thrillers with a twist ending.',
        'children': [
            {'id': 'r5c1', 'label': 'Dread, no gore',
             'query': ('Psychological horror built on dread and atmosphere — '
                       'absolutely no gore.')},
            {'id': 'r5c2', 'label': 'Mindbenders',
             'query': 'Mind-bending puzzles like Memento and Shutter Island.'},
            {'id': 'r5c3', 'label': 'Serial-killer cases',
             'query': 'Serial-killer investigations like Zodiac and Se7en.'},
            {'id': 'r5c4', 'label': 'Cults & isolation',
             'query': 'Unsettling movies about cults and isolated communities.'},
            {'id': 'r5c5', 'label': 'All-time twists',
             'query': ('The all-time great twist endings, '
                       'like The Sixth Sense and The Usual Suspects.')},
            {'id': 'r5c6', 'label': 'Paranoia classics',
             'query': ('Cold-war paranoia thrillers like The Conversation '
                       'and Three Days of the Condor.')},
        ],
    },
    {
        'id': 'r6', 'label': 'Ghibli & whimsy',
        'query': 'Movies with the gentle magic of Studio Ghibli.',
        'children': [
            {'id': 'r6c1', 'label': 'More anime',
             'query': 'Anime films for someone who loved Spirited Away and Your Name.'},
            {'id': 'r6c2', 'label': 'Stop-motion',
             'query': 'Stop-motion films like Coraline and Fantastic Mr. Fox.'},
            {'id': 'r6c3', 'label': 'Animated tearjerkers',
             'query': ('Animated movies that make grown-ups cry, '
                       'like Up and Grave of the Fireflies.')},
            {'id': 'r6c4', 'label': 'Grown-ups only',
             'query': ('Animation for adults — like Akira and A Scanner Darkly, '
                       'definitely not for kids.')},
            {'id': 'r6c5', 'label': 'Whimsical live-action',
             'query': ('Whimsical live-action films like Amélie '
                       'and The Grand Budapest Hotel.')},
            {'id': 'r6c6', 'label': 'Miyazaki only',
             'query': 'Just the Hayao Miyazaki films.'},
        ],
    },
]


def all_entries():
    """Flatten to (entry, parent_id) pairs, roots first — the generation script's iteration order."""
    out = []
    for root in ROOTS:
        out.append((
            {k: root[k] for k in ('id', 'label', 'query')}, None))
        for child in root['children']:
            out.append((child, root['id']))
    return out
