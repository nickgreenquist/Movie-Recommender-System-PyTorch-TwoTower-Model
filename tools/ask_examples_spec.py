"""
tools/ask_examples_spec.py — hand-curated query tree for the Ask tab's example chips.

STRUCTURE
    7 roots × 6 children = 42 SHOWN queries, + 4 backburner leaves (generated but not
    surfaced). Each root is a theme; its children are REFINEMENTS — the natural follow-up
    asks — each phrased as a full standalone query so it (a) pre-generates independently,
    (b) reads correctly when it back-fills the Ask text box, and (c) still works if a user
    edits it and reruns live. Every leaf below was traced through the real Haiku→two-tower
    pipeline before it earned a slot (apply loop + routing learnings: tools/ask_extractions/README.md).

    id      stable key ('r3', 'r3c2', …) — the artifact and the UI reference queries by id,
            so labels/queries can be reworded without invalidating a pre-generated report.
    label   short chip text (st.pills); the full query is what actually runs. Chip voice: a few
            evocative words lifted from the query where possible; exclusions ('no X') belong in
            the query, not the label.
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
    Isolation (r6c7), Adult cartoons (r1c7 — edgy non-Disney/Pixar American animation),
    Kids Halloween (r2c6 — kid-safe G/PG; broadening it kept pulling adult/off-theme films, so
    the clean-but-thin board is generated and held back rather than shown).
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
        'id': 'r1', 'label': 'Anime, Ghibli to Akira',
        'query': ("Breathtaking anime films, from Studio Ghibli's warmth to the "
                  "neon chaos of Akira and Ghost in the Shell."),
        'children': [
            {'id': 'r1c1', 'label': 'Studio Ghibli classics',
             'query': 'Studio Ghibli movies like Spirited Away and My Neighbor Totoro.'},
            {'id': 'r1c2', 'label': 'Dark, violent adult anime',
             'query': 'Dark, violent adult anime like Akira, Ghost in the Shell, and Ninja Scroll.'},
            {'id': 'r1c3', 'label': 'American animation, Pixar to Shrek',
             'query': 'American animated films like Toy Story, The Lion King, and Shrek.'},
            {'id': 'r1c4', 'label': 'Handmade stop-motion worlds',
             'query': 'Stop-motion animated films like Coraline and Fantastic Mr. Fox.'},
            {'id': 'r1c5', 'label': 'High-energy anime action',
             'query': 'High-energy anime action like Cowboy Bebop and Redline.'},
            {'id': 'r1c6', 'label': 'Anime love stories',
             'query': 'Romantic anime love stories like Your Name and Whisper of the Heart.'},
        ],
    },
    {
        'id': 'r2', 'label': 'Christmas by the tree',
        'query': 'Cozy Christmas movies to watch by the tree.',
        'children': [
            {'id': 'r2c1', 'label': 'R-rated, nothing wholesome',
             'query': ('Dark, cynical, irreverent R-rated Christmas comedies like Bad Santa, '
                       'Gremlins, and Krampus. Nothing sentimental or wholesome.')},
            {'id': 'r2c2', 'label': 'Safe for little kids',
             'query': 'Christmas movies safe for little kids.'},
            {'id': 'r2c3', 'label': 'Starring Santa himself',
             'query': 'Christmas movies where Santa Claus is a main character.'},
            {'id': 'r2c4', 'label': 'Animated family Christmas',
             'query': 'Animated Christmas movies for the whole family.'},
            {'id': 'r2c5', 'label': '1940s black-and-white classics',
             'query': 'Black-and-white Christmas classics from the 1940s and 50s.'},
            {'id': 'r2c7', 'label': 'Thanksgiving family gatherings',
             'query': 'Movies that take place on Thanksgiving, like Home for the Holidays, Pieces of April, and The Ice Storm.'},
        ],
    },
    {
        'id': 'r3', 'label': 'Make me sob',
        'query': 'A devastating drama that will absolutely make me sob.',
        'children': [
            {'id': 'r3c1', 'label': 'The human cost of war',
             'query': ('Harrowing anti-war films about the human cost of combat, like The Pianist, '
                       'Come and See, and Grave of the Fireflies. War genre only, no comedies, '
                       'cartoons, or documentaries.')},
            {'id': 'r3c2', 'label': 'Illness dramas that wreck you',
             'query': 'Cancer and terminal-illness dramas that wreck you.'},
            {'id': 'r3c3', 'label': 'Heartbreaking true stories',
             'query': "Heartbreaking dramas based on real, true events like 12 Years a Slave and Schindler's List."},
            {'id': 'r3c4', 'label': 'Dogs that break your heart',
             'query': "Emotional dramas about the bond between a person and their dog, like Hachi: A Dog's Story and My Dog Skip. No comedies."},
            {'id': 'r3c5', 'label': 'Father–son gut punches',
             'query': 'Father-and-son dramas that hit hard.'},
            {'id': 'r3c6', 'label': 'Be a better person',
             'query': 'Movies that inspire you to be a better person.'},
        ],
    },
    {
        'id': 'r4', 'label': 'Samurai duels & honor',
        'query': 'Classic samurai films full of sword duels and honor.',
        'children': [
            {'id': 'r4c1', 'label': 'Old-school kung fu',
             'query': 'Classic kung-fu movies like Enter the Dragon and Drunken Master.'},
            {'id': 'r4c2', 'label': 'Directed by Kurosawa',
             'query': 'Movies directed by Akira Kurosawa, like Rashomon, Ran, and Ikiru.'},
            {'id': 'r4c3', 'label': 'Knights & medieval battles',
             'query': 'Knights and medieval battles like Braveheart and Kingdom of Heaven.'},
            {'id': 'r4c4', 'label': 'Westerns, the American samurai',
             'query': 'Lone-gunslinger Westerns — the American samurai.'},
            {'id': 'r4c5', 'label': 'Gladiators & Roman epics',
             'query': 'Epic movies set in ancient Rome, no cartoons or animation'},
            {'id': 'r4c6', 'label': 'Real soldiers, true stories',
             'query': 'Modern combat movies about real soldiers based on true events, like Lone Survivor, American Sniper, and Black Hawk Down. No science fiction.'},
        ],
    },
    {
        'id': 'r5', 'label': 'Nothing but time travel',
        'query': 'Movies where time travel is the whole point, like Back to the Future and Primer.',
        'children': [
            {'id': 'r5c1', 'label': 'Stuck in a time loop',
             'query': 'Groundhog Day-style time loops, like Edge of Tomorrow.'},
            {'id': 'r5c2', 'label': 'Time travel, played for laughs',
             'query': 'Time-travel comedies like Bill & Ted and Hot Tub Time Machine.'},
            {'id': 'r5c3', 'label': 'Voyages into deep space',
             'query': 'Space travel movies like Interstellar and Gravity.'},
            {'id': 'r5c4', 'label': 'Wanderlust & far-off places',
             'query': ('Movies about traveling around the world to far-off places, like Around the '
                       'World in 80 Days, Romancing the Stone, and The Secret Life of Walter Mitty. '
                       'No science fiction, fantasy, documentaries, or comedies.')},
            {'id': 'r5c5', 'label': 'Beautiful minds & mathematicians',
             'query': 'Movies about mathematicians, no documentaries or sci-fi.'},
            {'id': 'r5c6', 'label': 'Brain-melting physics puzzles',
             'query': 'Brain-melting physics puzzles like Primer.'},
        ],
    },
    {
        'id': 'r6', 'label': "Humanity's final days",
        'query': ("End-of-the-world movies about humanity's final days, "
                  "like Children of Men and The Road."),
        'children': [
            {'id': 'r6c1', 'label': 'Full-on zombie apocalypse',
             'query': 'Zombie apocalypse movies like 28 Days Later and Dawn of the Dead.'},
            {'id': 'r6c2', 'label': 'Grounded pandemic dramas',
             'query': 'Grounded dramas about a deadly pandemic and epidemic spreading, like Contagion and Outbreak. No creature features, zombies, aliens, comedies, or animation.'},
            {'id': 'r6c3', 'label': 'Alien invasions of Earth',
             'query': 'Alien invasion of Earth movies, like War of the Worlds and Independence Day. No horror or comedy.'},
            {'id': 'r6c4', 'label': 'Epic disaster spectacle',
             'query': 'Epic disaster movies like Twister, Armageddon, and San Andreas'},
            {'id': 'r6c5', 'label': 'Bleak wasteland survival',
             'query': 'Bleak post-apocalyptic wasteland survival, like Mad Max and The Road.'},
            {'id': 'r6c6', 'label': 'Giant monsters level cities',
             'query': 'Giant monster destruction movies like Godzilla and Cloverfield.'},
        ],
    },
    {
        'id': 'r7', 'label': 'Gritty pre-2000 New York',
        'query': 'Gritty New York City movies from before 2000, no musicals.',
        'children': [
            {'id': 'r7c1', 'label': "Paranoid '70s conspiracy thrillers",
             'query': "Paranoid 1970s and 80s conspiracy thrillers like All the President's Men and Three Days of the Condor."},
            {'id': 'r7c2', 'label': "Scorsese's New York",
             'query': "Martin Scorsese's New York movies."},
            {'id': 'r7c3', 'label': 'Disco & nightclub nightlife',
             'query': 'Movies about the disco scene and nightclub nightlife, like Saturday Night Fever and The Last Days of Disco.'},
            {'id': 'r7c4', 'label': "Woody Allen's New York",
             'query': "Woody Allen's New York comedies."},
            {'id': 'r7c5', 'label': 'Smoky 1960s Paris',
             'query': 'Atmospheric older films set in 1960s Paris.'},
            {'id': 'r7c6', 'label': 'Grime, crime & decay',
             'query': 'Grimy, decaying-city crime dramas of the 70s and 80s.'},
        ],
    },
]

# Traced-good leaves the curator wants pre-generated but held back from the UI. They land in
# serving/ask_examples.json['examples'] (so a board is ready) but are absent from every root's
# `children`, so the generated `tree` never renders a pill for them. 'parent' records the root
# they belong to, for a clean promotion later (move the dict into that root's `children`).
BACKBURNER = [
    {'id': 'r3c7', 'label': 'Grief and moving on', 'parent': 'r3',
     'query': 'Aching movies about grief and moving on.'},
    {'id': 'r6c7', 'label': 'Alone in the wilderness', 'parent': 'r6',
     'query': 'Solitary survival movies about a person alone in the wilderness, like Into the Wild and Never Cry Wolf. No documentaries.'},
    {'id': 'r1c7', 'label': 'Animation for grownups', 'parent': 'r1',
     'query': 'Satirical adult animated comedies like South Park, Rejected, and Team America — animation for grownups, not kids.'},
    {'id': 'r2c6', 'label': 'Halloween without the horror', 'parent': 'r2',
     'query': ('Halloween movies for children rated G or PG only. No scariness, '
               'no horror. Like Hocus Pocus and Casper.')},
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
