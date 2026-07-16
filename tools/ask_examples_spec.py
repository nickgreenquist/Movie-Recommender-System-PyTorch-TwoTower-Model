"""
tools/ask_examples_spec.py — hand-curated query tree for the Ask tab's example chips.

STRUCTURE
    9 roots, 53 SHOWN children (r1–r7 six each, r8 five, r9 six), + 5 backburner leaves
    (generated but not surfaced). Each root is a theme; its children are REFINEMENTS — the natural follow-up
    asks — each phrased as a full standalone query so it (a) pre-generates independently,
    (b) reads correctly when it back-fills the Ask text box, and (c) still works if a user
    edits it and reruns live. Every leaf below was traced through the real Haiku→two-tower
    pipeline before it earned a slot (apply loop + routing learnings: tools/ask_extractions/README.md).

    id      stable key ('r3', 'r3c2', …) — the artifact and the UI reference queries by id,
            so labels/queries can be reworded without invalidating a pre-generated report.
    label   legacy field — the two-view Ask UI (landing cards + 'More:' riff chips) renders the
            QUERY itself (whitespace-collapsed, truncated at a word boundary if long), NOT this
            field. New entries (r8/r9) set label == query; older r1–r7 still carry their
            pre-redesign short-chip labels. streamlit_app.py never reads label.
    query   the standalone natural-language request that runs — AND the pill/card text the user
            actually sees.

CAPABILITY MAP (why these nine — the leaves fan across extraction ROUTES, not all "like X")
    r1  Anime       attributes (anime / stop-motion / american animation) + soft genre
    r2  Christmas   occasion keyword + max_rating (kid-safe / PG) + tone exclude (R) + B&W era
    r3  Good cry    subject/mood + exclude-genre ("no comedies") + based-on-true keyword
    r4  Samurai     hard genre + person (Kurosawa) + keyword-topic resolver (ancient-rome, katana)
    r5  Time travel keyword topics + anchors + subject (academia) — one continuous pipeline
    r6  End of world sub-genre keyword topics (zombie / pandemic / kaiju) + anchors
    r7  Gritty NYC  place + year fence + person (Scorsese, Woody Allen) + keyword resolver (disco)
    r8  Sharks      keyword-topic resolver across a creature-&-sea family (shark / dinosaur /
                    ocean / surfing / creature / pirate) + tone exclude ("no comedies")
    r9  Sports      per-sport keyword gates (boxing / baseball / american football / basketball)
                    + genome (underdog / sports / racing) + anchors ("more than the game") +
                    death-game concept w/ Horror exclude (competition to the death)

BACKBURNER (see BACKBURNER below)
    Traced-good leaves the curator wants pre-generated but NOT shown yet: Grief (r3c7),
    Isolation (r6c7), Adult cartoons (r1c7 — edgy non-Disney/Pixar American animation),
    Kids Halloween (r2c6 — kid-safe G/PG; broadening it kept pulling adult/off-theme films, so
    the clean-but-thin board is generated and held back rather than shown), Basketball (r9c7 —
    a spare sixth Sports leaf held in reserve behind boxing / baseball / football / racing).
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
        'query': "Breathtaking anime films.",
        'children': [
            {'id': 'r1c1', 'label': 'Studio Ghibli classics',
             'query': 'The lush, hand-drawn worlds of Studio Ghibli.'},
            {'id': 'r1c2', 'label': 'Dark, violent adult anime',
             'query': 'Dark and violent anime.'},
            {'id': 'r1c3', 'label': 'American animation, Pixar to Shrek',
             'query': 'Heartwarming American animated movies for the whole family.'},
            {'id': 'r1c4', 'label': 'Handmade stop-motion worlds',
             'query': 'Charming handmade stop-motion movies.'},
            {'id': 'r1c5', 'label': 'High-energy anime action',
             'query': 'High-energy anime action.'},
            {'id': 'r1c6', 'label': 'Anime love stories',
             'query': 'Achingly romantic anime love stories.'},
        ],
    },
    {
        'id': 'r2', 'label': 'Christmas by the tree',
        'query': 'Cozy Christmas movies to watch by the tree.',
        'children': [
            {'id': 'r2c1', 'label': 'R-rated, nothing wholesome',
             'query': 'Dark, cynical, irreverent R-rated Christmas comedies.'},
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
             'query': 'Harrowing anti-war films about the human cost of combat.'},
            {'id': 'r3c2', 'label': 'Illness dramas that wreck you',
             'query': 'Cancer and terminal-illness dramas that wreck you.'},
            {'id': 'r3c3', 'label': 'Heartbreaking true stories',
             'query': 'Heartbreaking dramas based on a true story.'},
            {'id': 'r3c4', 'label': 'Dogs that break your heart',
             'query': 'Emotional dramas about the bond between a person and their dog.'},
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
             'query': 'Classic kung-fu movies.'},
            {'id': 'r4c2', 'label': 'Directed by Kurosawa',
             'query': 'Movies directed by Akira Kurosawa, like Rashomon and Ran.'},
            {'id': 'r4c3', 'label': 'Knights & medieval battles',
             'query': 'Sweeping tales of knights and medieval battle.'},
            {'id': 'r4c4', 'label': 'Westerns, the American samurai',
             'query': 'Lone-gunslinger Westerns — the American samurai.'},
            {'id': 'r4c5', 'label': 'Gladiators & Roman epics',
             'query': 'Movies set in ancient Rome.'},
            {'id': 'r4c6', 'label': 'Real soldiers, true stories',
             'query': 'Combat movies about real soldiers based on true events.'},
        ],
    },
    {
        'id': 'r5', 'label': 'Nothing but time travel',
        'query': 'Movies where time travel is the whole point.',
        'children': [
            {'id': 'r5c1', 'label': 'Stuck in a time loop',
             'query': 'Groundhog Day-style time loops.'},
            {'id': 'r5c2', 'label': 'Time travel, played for laughs',
             'query': "Time-travel comedies that don't take themselves too seriously."},
            {'id': 'r5c3', 'label': 'Voyages into deep space',
             'query': 'Breathtaking, visually stunning movies about deep space travel. No horror.'},
            {'id': 'r5c4', 'label': 'Wanderlust & far-off places',
             'query': 'Movies about traveling around the world to far-off places.'},
            {'id': 'r5c5', 'label': 'Beautiful minds & mathematicians',
             'query': 'Movies about brilliant, driven mathematicians.'},
            {'id': 'r5c6', 'label': 'Brain-melting physics puzzles',
             'query': 'Brain-melting physics puzzles like Primer.'},
        ],
    },
    {
        'id': 'r6', 'label': "Humanity's final days",
        'query': "End-of-the-world movies about humanity's final days.",
        'children': [
            {'id': 'r6c1', 'label': 'Gory zombie horror',
             'query': 'Blood-soaked, gory zombie horror.'},
            {'id': 'r6c2', 'label': 'Grounded pandemic dramas',
             'query': 'Grounded dramas about a deadly pandemic spreading. No zombies or creature features.'},
            {'id': 'r6c3', 'label': 'Alien invasions of Earth',
             'query': 'When alien armadas invade Earth.'},
            {'id': 'r6c4', 'label': 'Epic disaster spectacle',
             'query': 'Epic, city-flattening disaster spectacle.'},
            {'id': 'r6c5', 'label': 'Bleak wasteland survival',
             'query': 'Bleak post-apocalyptic wasteland survival.'},
            {'id': 'r6c6', 'label': 'Giant monsters level cities',
             'query': 'Giant monsters leveling entire cities.'},
        ],
    },
    {
        'id': 'r7', 'label': 'Gritty pre-2000 New York',
        'query': 'Gritty New York City movies from before 2000.',
        'children': [
            {'id': 'r7c1', 'label': "Paranoid '70s conspiracy thrillers",
             'query': 'Paranoid 1970s and 80s conspiracy thrillers.'},
            {'id': 'r7c2', 'label': "Scorsese's New York",
             'query': "Martin Scorsese's New York movies."},
            {'id': 'r7c3', 'label': 'Disco & nightclub nightlife',
             'query': 'Movies about the disco scene and nightclub nightlife.'},
            {'id': 'r7c4', 'label': "Woody Allen's New York",
             'query': "Woody Allen's New York comedies."},
            {'id': 'r7c5', 'label': 'Smoky 1960s Paris',
             'query': 'Atmospheric older films set in 1960s Paris.'},
            {'id': 'r7c6', 'label': 'Grime, crime & decay',
             'query': 'Grimy, decaying-city crime dramas of the 70s and 80s.'},
        ],
    },
    {
        'id': 'r8', 'label': 'Bloodthirsty sharks that make you afraid to get back in the water.',
        'query': 'Bloodthirsty sharks that make you afraid to get back in the water.',
        'children': [
            {'id': 'r8c1', 'label': 'Dinosaurs brought back to life and running wild.',
             'query': 'Dinosaurs brought back to life and running wild.'},
            {'id': 'r8c2', 'label': 'Ocean survival movies about people struggling to stay alive out on the open water. No comedies or cartoons.',
             'query': 'Ocean survival movies about people struggling to stay alive out on the open water. No comedies or cartoons.'},
            {'id': 'r8c3', 'label': 'Surf movies about chasing the perfect wave.',
             'query': 'Surf movies about chasing the perfect wave.'},
            {'id': 'r8c4', 'label': 'Horror movies about a deadly creature on the loose.',
             'query': 'Horror movies about a deadly creature on the loose.'},
            {'id': 'r8c5', 'label': 'Swashbuckling pirate adventures on the high seas.',
             'query': 'Swashbuckling pirate adventures on the high seas.'},
        ],
    },
    {
        'id': 'r9', 'label': 'Underdog sports movies where the longshot goes the distance.',
        'query': 'Underdog sports movies where the longshot goes the distance.',
        'children': [
            {'id': 'r9c1', 'label': 'Boxing movies about leaving it all in the ring.',
             'query': 'Boxing movies about leaving it all in the ring.'},
            {'id': 'r9c2', 'label': 'Baseball movies about the boys of summer.',
             'query': 'Baseball movies about the boys of summer.'},
            {'id': 'r9c3', 'label': 'American football movies about leaving it all on the field, from Friday night lights to the NFL.',
             'query': 'American football movies about leaving it all on the field, from Friday night lights to the NFL.'},
            {'id': 'r9c4', 'label': 'Racing movies about fast cars and the drivers who live for the track.',
             'query': 'Racing movies about fast cars and the drivers who live for the track.'},
            {'id': 'r9c5', 'label': 'Sports movies that are really about life, not just the game, like Million Dollar Baby, Moneyball, and The Wrestler.',
             'query': 'Sports movies that are really about life, not just the game, like Million Dollar Baby, Moneyball, and The Wrestler.'},
            {'id': 'r9c6', 'label': 'Brutal death-game thrillers where contestants are forced to fight to the death. No gory torture horror.',
             'query': 'Brutal death-game thrillers where contestants are forced to fight to the death. No gory torture horror.'},
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
     'query': 'Survival movies about someone stranded alone in the wilderness. No documentaries.'},
    {'id': 'r1c7', 'label': 'Animation for grownups', 'parent': 'r1',
     'query': 'Satirical adult animated comedies — animation for grownups, not kids.'},
    {'id': 'r2c6', 'label': 'Halloween without the horror', 'parent': 'r2',
     'query': 'Halloween movies for kids, nothing scary or gory.'},
    {'id': 'r9c7', 'label': 'Basketball movies about making it on the court.', 'parent': 'r9',
     'query': 'Basketball movies about making it on the court.'},
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
