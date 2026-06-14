"""
Poster-wall generator for the popularity-correction blog post.

Renders a canary persona's top-10 recommendations from BOTH deployed twins
(α=0 naive vs α=0.5 Menon-corrected) side by side as TMDB poster grids, each
poster badged with the movie's raw ratings.csv count so the head→tail shift is
visible at a glance. Recommendation lists are the exact `main.py canary` output
for the two prod checkpoints; this script is the visualization layer over them.

    python analysis/poster_board.py        # → docs/poster_board.html
"""
import json
import os

import numpy as np
import torch

POSTER_BASE = 'https://image.tmdb.org/t/p/w342'   # posters.json already includes the full URL


# Canary recs verbatim from main.py canary on the two prod twins. off=True marks a
# pick that is off the persona's genre (the popularity drift we're illustrating).
PERSONAS = [
    {
        'name': 'The WW2 movie buff',
        'genome': 'wwii',
        'seeds': ['Saving Private Ryan (1998)', 'Enemy at the Gates (2001)',
                  'Stalingrad (1993)', 'Great Escape, The (1963)',
                  'Inglourious Basterds (2009)'],
        'off_label': 'not WW2',
        'alpha0': [
            ('Lord of the Rings: The Two Towers, The (2002)', True),
            ('Gladiator (2000)', True),
            ('Godfather: Part II, The (1974)', True),
            ('Bridge on the River Kwai, The (1957)', False),
            ('Lord of the Rings: The Return of the King, The (2003)', True),
            ('Good, the Bad and the Ugly, The (Buono, il brutto, il cattivo, Il) (1966)', True),
            ('Pianist, The (2002)', False),
            ('Braveheart (1995)', True),
            ('Bridge Too Far, A (1977)', False),
            ('Guns of Navarone, The (1961)', False),
        ],
        'alpha05': [
            ('Bridge Too Far, A (1977)', False),
            ('Longest Day, The (1962)', False),
            ("Devil's Brigade, The (1968)", False),
            ('Tora! Tora! Tora! (1970)', False),
            ('Cross of Iron (1977)', False),
            ('Battle of Britain (1969)', False),
            ('Zulu (1964)', True),
            ('Gettysburg (1993)', True),
            ('Bridge at Remagen, The (1969)', False),
            ('Eagle Has Landed, The (1976)', False),
        ],
    },
    {
        'name': 'The high-fantasy fan',
        'genome': 'high fantasy',
        'seeds': ['Lord of the Rings: The Fellowship of the Ring, The (2001)',
                  'Lord of the Rings: The Return of the King, The (2003)',
                  'Dark Crystal, The (1982)', 'Dragonslayer (1981)', 'Dune (1984)'],
        'off_label': 'not fantasy',
        'alpha0': [
            ('The Hobbit: The Battle of the Five Armies (2014)', False),
            ('Indiana Jones and the Temple of Doom (1984)', True),
            ('Star Trek III: The Search for Spock (1984)', True),
            ('Star Trek V: The Final Frontier (1989)', True),
            ('Star Trek VI: The Undiscovered Country (1991)', True),
            ('Star Trek: The Motion Picture (1979)', True),
            ('Thor: The Dark World (2013)', True),
            ('Star Wars: Episode III - Revenge of the Sith (2005)', True),
            ('Star Wars: Episode V - The Empire Strikes Back (1980)', True),
            ('Star Trek II: The Wrath of Khan (1982)', True),
        ],
        'alpha05': [
            ('Ladyhawke (1985)', False),
            ('Solomon Kane (2009)', False),
            ('Conan the Destroyer (1984)', False),
            ('47 Ronin (2013)', False),
            ('Krull (1983)', False),
            ('The Golden Voyage of Sinbad (1973)', False),
            ('NeverEnding Story, The (1984)', False),
            ('Mad Max Beyond Thunderdome (1985)', True),
            ('Percy Jackson: Sea of Monsters (2013)', False),
            ('John Carter (2012)', True),
        ],
    },
    {
        'name': 'The organized-crime fanatic',
        'genome': 'organized crime / mafia',
        'seeds': ['Donnie Brasco (1997)', 'Casino (1995)', 'American Gangster (2007)',
                  'Narc (2002)', 'Sicario (2015)', 'The Irishman (2019)'],
        'off_label': 'not crime',
        'alpha0': [
            ('Scarface (1983)', False),
            ('Godfather: Part II, The (1974)', False),
            ('Heat (1995)', False),
            ('Full Metal Jacket (1987)', True),
            ('Usual Suspects, The (1995)', False),
            ('Reservoir Dogs (1992)', False),
            ('Fight Club (1999)', True),
            ("Carlito's Way (1993)", False),
            ('Dark Knight, The (2008)', True),
            ("One Flew Over the Cuckoo's Nest (1975)", True),
        ],
        'alpha05': [
            ('Sicario: Day of the Soldado (2018)', False),
            ('Brawl in Cell Block 99 (2017)', False),
            ('Mean Streets (1973)', False),
            ('Deep Cover (1992)', False),
            ("Brooklyn's Finest (2010)", False),
            ('King of New York (1990)', False),
            ('Layer Cake (2004)', False),
            ('Shot Caller (2017)', False),
            ('Too Big to Fail (2011)', True),
            ('New Jack City (1991)', False),
        ],
    },
    {
        'name': 'The courtroom-drama devotee',
        'genome': 'courtroom',
        'seeds': ['Anatomy of a Murder (1959)', 'Judgment at Nuremberg (1961)',
                  '...And Justice for All (1979)', 'Verdict, The (1982)',
                  'Witness for the Prosecution (1957)', 'Inherit the Wind (1960)'],
        'off_label': 'not courtroom',
        'alpha0': [
            ("Schindler's List (1993)", True),
            ('Godfather, The (1972)', True),
            ('12 Angry Men (1957)', False),
            ('Shawshank Redemption, The (1994)', False),
            ('Fight Club (1999)', True),
            ('Raiders of the Lost Ark (Indiana Jones and the Raiders of the Lost Ark) (1981)', True),
            ('Usual Suspects, The (1995)', True),
            ('Godfather: Part II, The (1974)', True),
            ('Dark Knight, The (2008)', True),
            ('Casablanca (1942)', True),
        ],
        'alpha05': [
            ('Absence of Malice (1981)', False),
            ('Elmer Gantry (1960)', False),
            ('Lilies of the Field (1963)', False),
            ('Caine Mutiny, The (1954)', False),
            ('Norma Rae (1979)', False),
            ('In the Heat of the Night (1967)', False),
            ('Seven Days in May (1964)', False),
            ("Twelve O'Clock High (1949)", False),
            ('Days of Wine and Roses (1962)', False),
            ("Summer of '42 (1971)", False),
        ],
    },
    {
        'name': 'The 1950s creature-feature fan',
        'genome': '1950s sci-fi',
        'seeds': ['Them! (1954)', 'Thing from Another World, The (1951)',
                  'Creature from the Black Lagoon, The (1954)', 'Tarantula (1955)',
                  'It Came from Outer Space (1953)', 'Incredible Shrinking Man, The (1957)',
                  'Earth vs. the Flying Saucers (1956)'],
        'off_label': 'not sci-fi',
        'alpha0': [
            ('Star Trek II: The Wrath of Khan (1982)', False),
            ('Dr. Strangelove or: How I Learned to Stop Worrying and Love the Bomb (1964)', False),
            ('Butch Cassidy and the Sundance Kid (1969)', True),
            ('Coneheads (1993)', False),
            ("City Slickers II: The Legend of Curly's Gold (1994)", True),
            ('Clockwork Orange, A (1971)', False),
            ('Citizen Kane (1941)', True),
            ('Taxi Driver (1976)', True),
            ('2001: A Space Odyssey (1968)', False),
            ('Psycho (1960)', False),
        ],
        'alpha05': [
            ('Silent Running (1972)', False),
            ('X: The Man with the X-Ray Eyes (1963)', False),
            ('Quatermass and the Pit (1967)', False),
            ('Day of the Triffids, The (1962)', False),
            ('Invaders from Mars (1953)', False),
            ('Starman (1984)', False),
            ('Outland (1981)', False),
            ('Soylent Green (1973)', False),
            ('Hidden, The (1987)', False),
            ('Brother from Another Planet, The (1984)', False),
        ],
    },
    {
        'name': 'The Argento-Fulci giallo cultist',
        'genome': 'giallo',
        'seeds': ['Suspiria (1977)', 'Deep Red (Profondo rosso) (1975)', 'Tenebre (1982)',
                  'Phenomena (a.k.a. Creepers) (1985)', 'Inferno (1980)',
                  'City of the Living Dead (a.k.a. Gates of Hell, The) (Paura nella città dei morti viventi) (1980)',
                  "Bird with the Crystal Plumage, The (Uccello dalle piume di cristallo, L') (1970)"],
        'off_label': 'mainstream',
        # No off-genre chips here: every α=0 pick is itself a horror film, so the story is
        # pure popularity (famous-horror badges 10k–30k vs cult/giallo deep cuts 200–1.8k).
        'alpha0': [
            ('Exorcist, The (1973)', False),
            ('Evil Dead, The (1981)', False),
            ('Halloween (1978)', False),
            ('Hellraiser (1987)', False),
            ('Birds, The (1963)', False),
            ('Texas Chainsaw Massacre, The (1974)', False),
            ('Re-Animator (1985)', False),
            ('Jaws (1975)', False),
            ("Rosemary's Baby (1968)", False),
            ('Wicker Man, The (1973)', False),
        ],
        'alpha05': [
            ('Black Christmas (1974)', False),
            ('Society (1989)', False),
            ('Sleepaway Camp (1983)', False),
            ('Motel Hell (1980)', False),
            ('From Beyond (1986)', False),
            ("Beyond, The (E tu vivrai nel terrore - L'aldilà) (1981)", False),
            ('Night of the Demons (1988)', False),
            ('Black Sabbath (Tre volti della paura, I) (1963)', False),
            ('Zombie (a.k.a. Zombie 2: The Dead Are Among Us) (Zombi 2) (1979)', False),
            ('Return of the Living Dead, The (1985)', False),
        ],
    },
    {
        'name': 'The 1930s screwball-comedy fan',
        'genome': 'screwball',
        'seeds': ['His Girl Friday (1940)', 'Lady Eve, The (1941)', 'My Man Godfrey (1936)',
                  'Awful Truth, The (1937)', 'Palm Beach Story, The (1942)',
                  'Twentieth Century (1934)', 'Ball of Fire (1941)'],
        'off_label': 'not comedy',
        'alpha0': [
            ('Thin Man, The (1934)', False),
            ('Philadelphia Story, The (1940)', False),
            ('Top Hat (1935)', True),
            ('North by Northwest (1959)', True),
            ('Roman Holiday (1953)', True),
            ('Shop Around the Corner, The (1940)', False),
            ('Arsenic and Old Lace (1944)', False),
            ("Singin' in the Rain (1952)", True),
            ('Charade (1963)', True),
            ('Butch Cassidy and the Sundance Kid (1969)', True),
        ],
        'alpha05': [
            ("You Can't Take It with You (1938)", False),
            ('Another Thin Man (1939)', False),
            ('Swing Time (1936)', True),
            ('Shadow of the Thin Man (1941)', False),
            ('Mr. Blandings Builds His Dream House (1948)', False),
            ('Thin Man Goes Home, The (1945)', False),
            ("Bishop's Wife, The (1947)", False),
            ('After the Thin Man (1936)', False),
            ('Talk of the Town, The (1942)', False),
            ('Thin Man, The (1934)', False),
        ],
    },
]


def _fmt_count(n: int) -> str:
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}k"
    return str(n)


def _load_lookups():
    fs      = torch.load('serving/feature_store.pt', map_location='cpu', weights_only=False)
    posters = json.load(open('serving/posters.json'))
    counts  = np.load('data/corpus_raw_rating_counts.npy')
    top     = [int(m) for m in fs['top_movies']]
    mid_to_count = {top[i]: int(counts[i]) for i in range(len(top))}
    return fs['title_to_movieId'], posters, mid_to_count


def _card(title, off, off_label, t2m, posters, mid_to_count):
    mid   = t2m.get(title)
    url   = posters.get(str(mid)) if mid is not None else None
    cnt   = mid_to_count.get(int(mid)) if mid is not None else None
    short = title.rsplit(' (', 1)[0]
    year  = title.rsplit('(', 1)[-1].rstrip(')') if '(' in title else ''
    img   = (f'<img src="{url}" loading="lazy" alt="">' if url
             else f'<div class="noimg">{short}</div>')
    chip  = f'<span class="offchip">{off_label}</span>' if off else ''
    badge = f'<span class="badge">{_fmt_count(cnt)} ratings</span>' if cnt is not None else ''
    return (f'<figure class="card{" off" if off else ""}">'
            f'<div class="poster">{img}{chip}{badge}</div>'
            f'<figcaption>{short}<span class="yr">{year}</span></figcaption></figure>')


def _panel(kind, label, sub, recs, off_label, lk):
    t2m, posters, mid_to_count = lk
    med = int(np.median([mid_to_count.get(int(t2m.get(t, -1)), 0) for t, _ in recs]))
    cards = "\n".join(_card(t, off, off_label, *lk) for t, off in recs)
    return (f'<div class="panel {kind}">'
            f'<div class="phead"><span class="dot"></span>{label}'
            f'<span class="sub">{sub} · median {_fmt_count(med)} ratings</span></div>'
            f'<div class="grid">{cards}</div></div>')


def _seed_strip(seeds, lk):
    t2m, posters, _ = lk
    out = []
    for t in seeds:
        mid = t2m.get(t)
        url = posters.get(str(mid)) if mid is not None else None
        short = t.rsplit(' (', 1)[0]
        out.append(f'<img src="{url}" title="{short}" alt="">' if url
                   else f'<span class="seedtxt">{short}</span>')
    return f'<div class="seeds"><span class="seedlbl">Their taste →</span>{"".join(out)}</div>'


# Shared stylesheet for the combined page and the per-persona screenshot pages. Plain
# string (single braces) so it embeds into either f-string doc shell unescaped.
_CSS = """
  :root { --bg:#0d1117; --card:#161b22; --line:#21262d; --txt:#e6edf3; --mut:#8b949e;
           --bad:#f85149; --good:#3fb950; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--txt);
          font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif; }
  .wrap { max-width:1180px; margin:0 auto; padding:32px 28px 48px; }
  .wrap.solo { max-width:none; padding:16px; }
  .wrap.solo h2 { margin-top:2px; }
  .kicker { color:var(--good); font-weight:700; letter-spacing:.08em; text-transform:uppercase; font-size:13px; }
  h1 { font-size:30px; line-height:1.2; margin:6px 0 6px; }
  .lede { color:var(--mut); font-size:16px; margin:0 0 8px; max-width:820px; }
  h2 { font-size:20px; margin:30px 0 10px; }
  .gtag { font-size:13px; font-weight:600; color:var(--mut); margin-left:8px;
          padding:2px 8px; border:1px solid var(--line); border-radius:20px; vertical-align:middle; }
  .seeds { display:flex; align-items:center; gap:8px; flex-wrap:wrap; margin:0 0 14px;
            padding:10px 12px; background:var(--card); border:1px solid var(--line); border-radius:10px; }
  .seedlbl { color:var(--mut); font-size:13px; font-weight:600; margin-right:4px; white-space:nowrap; }
  .seeds img { width:42px; height:63px; object-fit:cover; border-radius:4px; }
  .seedtxt { font-size:12px; color:var(--mut); padding:0 6px; }
  .panels { display:grid; grid-template-columns:1fr 1fr; gap:18px; }
  .panel { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:14px 14px 16px; }
  .phead { font-weight:700; font-size:15px; display:flex; align-items:center; gap:8px; flex-wrap:wrap;
            padding-bottom:12px; border-bottom:1px solid var(--line); margin-bottom:12px; }
  .phead .sub { font-weight:500; color:var(--mut); font-size:12.5px; flex-basis:100%; margin-left:18px; }
  .dot { width:10px; height:10px; border-radius:50%; display:inline-block; }
  .bad .dot { background:var(--bad); } .good .dot { background:var(--good); }
  .bad .phead { color:#ffb4ae; } .good .phead { color:#a5e7b0; }
  .grid { display:grid; grid-template-columns:repeat(5,1fr); gap:10px; }
  .card { margin:0; }
  .poster { position:relative; aspect-ratio:2/3; border-radius:6px; overflow:hidden; background:#0a0d12; }
  .poster img { width:100%; height:100%; object-fit:cover; display:block; }
  .noimg { width:100%; height:100%; display:flex; align-items:center; justify-content:center;
            text-align:center; font-size:11px; color:var(--mut); padding:6px; }
  .badge { position:absolute; left:5px; bottom:5px; background:rgba(0,0,0,.78); color:#fff;
            font-size:10.5px; font-weight:600; padding:2px 6px; border-radius:20px; }
  .offchip { position:absolute; top:5px; right:5px; background:var(--bad); color:#fff;
              font-size:9.5px; font-weight:700; padding:2px 6px; border-radius:20px; text-transform:uppercase; letter-spacing:.03em; }
  .card.off .poster { outline:2px solid var(--bad); outline-offset:-2px; }
  figcaption { font-size:11px; line-height:1.25; margin-top:5px; color:var(--txt);
            display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; overflow:hidden; height:2.5em; }
  .yr { color:var(--mut); margin-left:4px; }
  .foot { color:var(--mut); font-size:13px; margin-top:26px; border-top:1px solid var(--line); padding-top:14px; }
"""


def _section(p, lk):
    """One persona block: header + genome chip + seed strip + the α=0 / α=0.5 panels."""
    return (f'<section class="persona">'
            f'<h2>{p["name"]}<span class="gtag">{p["genome"]}</span></h2>'
            f'{_seed_strip(p["seeds"], lk)}'
            f'<div class="panels">'
            f'{_panel("bad",  "Popularity correction OFF (α = 0)",  "what users used to get", p["alpha0"],  p["off_label"], lk)}'
            f'{_panel("good", "Popularity correction ON (α = 0.5)", "what ships today",       p["alpha05"], p["off_label"], lk)}'
            f'</div></section>')


def build_html(lk=None):
    """Full combined page: intro + every persona section + methodology footer."""
    lk = lk or _load_lookups()
    sections = "\n".join(_section(p, lk) for p in PERSONAS)
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>One line of code, half the popularity bias</title>
<style>{_CSS}</style></head>
<body><div class="wrap">
  <div class="kicker">Recsys · popularity debiasing</div>
  <h1>Same fan, same model — one line of training code apart</h1>
  <p class="lede">Two separately-trained versions of the same two-tower recommender. The only difference:
  one scalar in the loss function (Menon logit adjustment, α). Inference code is identical. Each poster is
  badged with how many times that movie was rated in MovieLens-32M — the bigger the number, the more popular.</p>
  {sections}
  <p class="foot">Red outline = off-genre pick. Recommendations are verbatim <code>canary</code> output from the two
  deployed checkpoints (α=0 vs α=0.5); badges are raw MovieLens rating counts. Across 60k held-out contexts the
  α=0.5 model cuts median recommendation popularity by 51% and lifts catalog coverage from 48% to 84%.</p>
</div></body></html>"""


def build_one(p, lk):
    """Standalone single-persona doc, full-bleed, for a tight per-board screenshot."""
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">'
            f'<title>{p["name"]}</title><style>{_CSS}</style></head>'
            f'<body><div class="wrap solo">{_section(p, lk)}</div></body></html>')


def _slug(name):
    return ''.join(c if c.isalnum() else '-' for c in name.lower()).strip('-')


if __name__ == '__main__':
    import sys
    lk = _load_lookups()
    os.makedirs('docs', exist_ok=True)
    with open('docs/poster_board.html', 'w') as f:
        f.write(build_html(lk))
    print("  → wrote docs/poster_board.html")

    # --split DIR : also emit one standalone HTML per persona (for screenshotting).
    if '--split' in sys.argv:
        outdir = sys.argv[sys.argv.index('--split') + 1]
        os.makedirs(outdir, exist_ok=True)
        for p in PERSONAS:
            fn = os.path.join(outdir, f'{_slug(p["name"])}.html')
            with open(fn, 'w') as f:
                f.write(build_one(p, lk))
            print(f"  → wrote {fn}")
