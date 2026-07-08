"""
tools/ask_examples_spec.py — hand-curated query tree for the Ask tab's example chips.

STRUCTURE
    7 roots × 6 children = 42 SHOWN queries, + 2 backburner leaves (generated but not
    surfaced). Each root is a theme; its children are REFINEMENTS — the natural follow-up
    asks — each phrased as a full standalone query so it (a) pre-generates independently,
    (b) reads correctly when it back-fills the Ask text box, and (c) still works if a user
    edits it and reruns live. Every leaf below was traced through the real Haiku→two-tower
    pipeline before it earned a slot (worksheet: ask_leaf_candidates_v3.md).

    id      stable key ('r3', 'r3c2', …) — the artifact and the UI reference queries by id,
            so labels/queries can be reworded without invalidating a pre-generated report.
    label   short chip text (st.pills); the full query is what actually runs.
    query   the standalone natural-language request that was / will be pre-generated.

CAPABILITY MAP (why these seven — the leaves fan across extraction ROUTES, not all "like X")
    r1  Anime       attributes (anime / stop-motion / american animation) + soft genre
    r2  Christmas   occasion keyword + max_rating (kid-safe / PG) + tone exclude (R) + B&W era
    r3  Good cry    subject/mood + exclude-genre ("no comedies") + based-on-true keyword
    r4  Samurai     hard genre + person (Kurosawa) + keyword-topic resolver (ancient-rome, katana)
    r5  Time travel keyword topics + anchors + subject (academia) — one continuous pipeline
    r6  End of world sub-genre keyword topics (zombie / pandemic / kaiju) + anchors
    r7  Gritty NYC  place + year fence + person (Scorsese, Woody Allen) + keyword resolver (disco)

BACKBURNER (see BACKBURNER below)
    Traced-good leaves the curator wants pre-generated but NOT shown yet: Grief (r3c7),
    Isolation (r6c7), Adult cartoons (r1c7 — edgy non-Disney/Pixar American animation).
    all_entries() yields them so gen_ask_examples.py writes their boards into
    serving/ask_examples.json['examples'], but they are absent from every root's `children`, so
    the generated `tree` (which the Ask tab renders pills from) never surfaces them. To promote
    one later: move its dict into the parent root's `children` and regenerate — no id churn.

CURATION RULES (enforced by the generation + review pass, not by this file)
    - Corpus is MovieLens 32M @200+ ratings, well-covered through ~2019 — no 2020+ recency asks.
    - No studio/streaming asks (unsupported by the extraction schema, silently ignored).
    - Every query must pre-generate with fallback=False, no dropped anchor titles, and a
      coherent board; weak entries get reworded or swapped here and regenerated
      (python tools/gen_ask_examples.py).
    - INTEGRITY: every committed extraction equals what its PROMPT actually emits live (each
      verified through the Haiku subagent) — NO hand-invented facets, so a canned chip is exactly
      what a user typing that prompt would get. Boards that need a filter EARN it in the wording:
      "no musicals" → exclude_genres['Musical'] (Gritty-NYC root); "no fantasy or sci-fi" → exclude
      both (Academia); "directed by Akira Kurosawa" → require_people; naming example films → the
      liked_items anchors that drive Ancient Rome ("ancient Rome and gladiators" → ancient-rome +
      gladiator concepts), War&loss (anchored on The Pianist / Come and See / Grave of the Fireflies),
      Pandemic, etc. Word-choice traps found the hard way: "ancient Rome" alone drifts to the 'rome'
      CITY genome tag (Fellini/Italian cinema) — say "gladiators" too; "love and loss" in a war query
      mis-fires to the 'love'/'loss' keyword pool (rom-coms) — so War&loss anchors on titles instead;
      the katana leaf avoids naming a film so it rides the 'katana' resolver (naming one → 'samurai'/
      'sword fighting').
    - Extraction is STOCHASTIC, so regenerate from the committed tools/ask_extractions/ (bare
      `python tools/gen_ask_examples.py`), NEVER --live — a fresh roll can re-drift a board, which is
      exactly what bit the 07-08 rebuild.
"""

ROOTS = [
    {
        'id': 'r1', 'label': 'Anime',
        'query': ("Breathtaking anime films, from Studio Ghibli's warmth to the "
                  "neon chaos of Akira and Ghost in the Shell."),
        'children': [
            {'id': 'r1c1', 'label': 'Ghibli',
             'query': 'Studio Ghibli movies like Spirited Away and My Neighbor Totoro.'},
            {'id': 'r1c2', 'label': 'Adult anime',
             'query': 'Dark, violent adult anime like Akira, Ghost in the Shell, and Ninja Scroll.'},
            {'id': 'r1c3', 'label': 'American animation',
             'query': 'American animated films like Toy Story, The Lion King, and Shrek.'},
            {'id': 'r1c4', 'label': 'Stop-motion',
             'query': 'Stop-motion animated films like Coraline and Fantastic Mr. Fox.'},
            {'id': 'r1c5', 'label': 'Anime action',
             'query': 'High-energy anime action like Cowboy Bebop and Redline.'},
            {'id': 'r1c6', 'label': 'Anime romance',
             'query': 'Tender anime romances like Your Name and Whisper of the Heart.'},
        ],
    },
    {
        'id': 'r2', 'label': 'Christmas',
        'query': 'Cozy Christmas movies to watch by the tree.',
        'children': [
            {'id': 'r2c1', 'label': 'R-rated',
             'query': 'Naughty, R-rated Christmas movies like Bad Santa and Gremlins.'},
            {'id': 'r2c2', 'label': 'Kid-safe',
             'query': 'Christmas movies safe for little kids.'},
            {'id': 'r2c3', 'label': 'Santa',
             'query': 'Christmas movies where Santa Claus is a main character.'},
            {'id': 'r2c4', 'label': 'Animated',
             'query': 'Animated Christmas movies for the whole family.'},
            {'id': 'r2c5', 'label': 'B&W classics',
             'query': 'Black-and-white Christmas classics from the 1940s and 50s.'},
            {'id': 'r2c6', 'label': 'Kids Halloween',
             'query': ('Kid-friendly Halloween movies rated PG or under, '
                       'like Hocus Pocus and Casper.')},
        ],
    },
    {
        'id': 'r3', 'label': 'Good cry',
        'query': 'A devastating drama that will absolutely make me sob.',
        'children': [
            {'id': 'r3c1', 'label': 'War & loss',
             'query': ('Devastating war dramas like The Pianist, Come and See, '
                       'and Grave of the Fireflies, no documentaries.')},
            {'id': 'r3c2', 'label': 'Terminal illness',
             'query': 'Cancer and terminal-illness dramas that wreck you.'},
            {'id': 'r3c3', 'label': 'True story',
             'query': 'Based-on-a-true-story dramas that break your heart.'},
            {'id': 'r3c4', 'label': 'Dog movies',
             'query': 'A dog movie that will make me sob, no comedies, like Marley & Me and Hachi.'},
            {'id': 'r3c5', 'label': 'Father & son',
             'query': 'Father-and-son dramas that hit hard.'},
            {'id': 'r3c6', 'label': 'Inspiring',
             'query': 'Movies that inspire you to be a better person.'},
        ],
    },
    {
        'id': 'r4', 'label': 'Samurai',
        'query': 'Classic samurai films full of sword duels and honor.',
        'children': [
            {'id': 'r4c1', 'label': 'Kung fu',
             'query': 'Classic kung-fu movies like Enter the Dragon and Drunken Master.'},
            {'id': 'r4c2', 'label': 'Kurosawa',
             'query': 'Movies directed by Akira Kurosawa, like Rashomon, Ran, and Ikiru.'},
            {'id': 'r4c3', 'label': 'Medieval knights',
             'query': 'Knights and medieval battles like Braveheart and Kingdom of Heaven.'},
            {'id': 'r4c4', 'label': 'American samurai',
             'query': 'Lone-gunslinger Westerns — the American samurai.'},
            {'id': 'r4c5', 'label': 'Ancient Rome',
             'query': 'Epic movies about ancient Rome and gladiators.'},
            {'id': 'r4c6', 'label': 'Katana',
             'query': 'Movies where the hero wields a katana.'},
        ],
    },
    {
        'id': 'r5', 'label': 'Time travel',
        'query': 'Movies where time travel is the whole point, like Back to the Future and Primer.',
        'children': [
            {'id': 'r5c1', 'label': 'Time loops',
             'query': 'Groundhog Day-style time loops, like Edge of Tomorrow.'},
            {'id': 'r5c2', 'label': 'Comedies',
             'query': 'Time-travel comedies like Bill & Ted and Hot Tub Time Machine.'},
            {'id': 'r5c3', 'label': 'Space travel',
             'query': 'Space travel movies like Interstellar and Gravity.'},
            {'id': 'r5c4', 'label': 'World travel',
             'query': 'Movies about traveling the world and going on a journey.'},
            {'id': 'r5c5', 'label': 'Academia',
             'query': 'Movies about real mathematicians and professors, no fantasy or sci-fi.'},
            {'id': 'r5c6', 'label': 'Hard physics',
             'query': 'Brain-melting physics puzzles like Primer.'},
        ],
    },
    {
        'id': 'r6', 'label': 'End of the world',
        'query': ("End-of-the-world movies about humanity's final days, "
                  "like Children of Men and The Road."),
        'children': [
            {'id': 'r6c1', 'label': 'Zombie',
             'query': 'Zombie apocalypse movies like 28 Days Later and Dawn of the Dead.'},
            {'id': 'r6c2', 'label': 'Pandemic',
             'query': 'Deadly pandemic outbreak movies like Contagion and Outbreak.'},
            {'id': 'r6c3', 'label': 'Alien invasion',
             'query': 'Alien invasion movies like War of the Worlds and Independence Day.'},
            {'id': 'r6c4', 'label': 'Climate',
             'query': 'Climate-collapse disaster movies like The Day After Tomorrow.'},
            {'id': 'r6c5', 'label': 'Wasteland',
             'query': 'Bleak post-apocalyptic wasteland survival, like Mad Max and The Road.'},
            {'id': 'r6c6', 'label': 'Giant monsters',
             'query': 'Giant monster destruction movies like Godzilla and Cloverfield.'},
        ],
    },
    {
        'id': 'r7', 'label': 'Gritty NYC',
        'query': 'Gritty New York City movies from before 2000, no musicals.',
        'children': [
            {'id': 'r7c1', 'label': '70s paranoia',
             'query': 'Paranoid conspiracy thrillers about political cover-ups, from the 1970s and 80s.'},
            {'id': 'r7c2', 'label': 'Scorsese',
             'query': "Martin Scorsese's New York movies."},
            {'id': 'r7c3', 'label': 'Disco era',
             'query': 'New York in the disco era of the late 1970s.'},
            {'id': 'r7c4', 'label': 'Woody Allen',
             'query': "Woody Allen's New York comedies."},
            {'id': 'r7c5', 'label': '60s Paris',
             'query': 'Atmospheric older films set in 1960s Paris.'},
            {'id': 'r7c6', 'label': 'Urban decay',
             'query': 'Grimy, decaying-city crime dramas of the 70s and 80s.'},
        ],
    },
]

# Traced-good leaves the curator wants pre-generated but held back from the UI. They land in
# serving/ask_examples.json['examples'] (so a board is ready) but are absent from every root's
# `children`, so the generated `tree` never renders a pill for them. 'parent' records the root
# they belong to, for a clean promotion later (move the dict into that root's `children`).
BACKBURNER = [
    {'id': 'r3c7', 'label': 'Grief', 'parent': 'r3',
     'query': 'Aching movies about grief and moving on.'},
    {'id': 'r6c7', 'label': 'Isolation', 'parent': 'r6',
     'query': 'Movies about leaving society behind to live alone in the wilderness.'},
    {'id': 'r1c7', 'label': 'Adult cartoons', 'parent': 'r1',
     'query': 'Adult, R-rated American animation like South Park, Waking Life, and Batman: The Killing Joke.'},
]


def all_entries():
    """Flatten to (entry, parent_id) pairs — the generation script's iteration order: each root,
    then its shown children, then the backburner leaves (generated so their board is ready, but
    excluded from every root's `children` and therefore from the rendered `tree`)."""
    out = []
    for root in ROOTS:
        out.append((
            {k: root[k] for k in ('id', 'label', 'query')}, None))
        for child in root['children']:
            out.append((child, root['id']))
    for leaf in BACKBURNER:
        out.append((
            {k: leaf[k] for k in ('id', 'label', 'query')}, leaf['parent']))
    return out
