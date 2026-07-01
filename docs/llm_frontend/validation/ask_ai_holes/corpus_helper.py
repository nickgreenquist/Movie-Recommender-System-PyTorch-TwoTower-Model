"""Corpus grounding helper for the rec/oracle subagents. Shows exactly what data we have per movie.

Usage:
  python corpus.py inventory                 # print the full data schema we have vs likely-missing
  python corpus.py search "blade"            # find in-corpus titles containing substring
  python corpus.py title "Blade Runner"      # dump ALL facets we have for a movie
  python corpus.py kw "car chase"            # how many movies carry a keyword (substring)
  python corpus.py genome sad                # genome/user tags matching a term (affect probe)
"""
import sys, pickle, os
# torch is lazy-imported inside serving()/facet() only — most commands don't need it (fast).

SC = '/private/tmp/claude-501/-Users-nickgreenquist-Documents-Movie-Recommender-System-PyTorch-TwoTower-Model/978b835f-d6d2-45f2-8a29-cda39e52b3d1/scratchpad'
REPO = '/Users/nickgreenquist/Documents/Movie-Recommender-System-PyTorch-TwoTower-Model'
agg = pickle.load(open(os.path.join(SC,'agg.pkl'),'rb'))
mt=agg['movie_title']; my=agg['movie_year']; mk=agg['movie_keywords']; mc=agg['movie_countries']; mol=agg['movie_olang']
_title_to_mid = {}
for m,t in mt.items():
    if t: _title_to_mid.setdefault(t.lower(), m)

_fs=_facet=None
def serving():
    global _fs
    if _fs is None:
        import torch
        _fs=torch.load(os.path.join(REPO,'serving/feature_store.pt'),weights_only=False)
    return _fs
def facet():
    global _facet
    if _facet is None:
        import torch
        _facet=torch.load(os.path.join(REPO,'llm_features/cache/facet_store.pt'),weights_only=False)
    return _facet

def prompts(i):
    """Print the 10 prompts of archetype-group i from the harvested run (no torch)."""
    import json
    h=json.load(open(os.path.join(SC,'harvest.json')))
    groups=h.get('gen_prompt_groups',[])
    try: g=groups[int(i)]
    except Exception: print(f'bad group index {i} (have {len(groups)})'); return
    for k,p in enumerate(g):
        print(f"{k+1}. {p.get('prompt')}  [intent: {p.get('intent')}]")

def inventory():
    fs=serving()
    print("DATA WE HAVE per movie (all baked into serving/, the recommender's real inputs):")
    print("  - genres (20-way one-hot), release year")
    print("  - user-applied tags (306), GENOME tags (1128 ML-relevance 0-1) -- INCLUDES affect/tone:")
    print("      sad/emotional/heartbreaking, dark/gritty/bleak, feel-good/heartwarming, tense/creepy/atmospheric,")
    print("      epic/visually-stunning, mindfuck/cerebral/twist-ending, quirky/funny -- these feed the item tower")
    print("  - TMDB keywords (17,820 raw: themes, settings/places, format, eras, some props)")
    print("  - production_countries (ISO), spoken/original language")
    print("  - people: top-10 billed cast + directors + writers (canonical TMDB ids) [people facet store]")
    print("  - popularity rank, TMDB vote_average / vote_count")
    print("  - a 132-dim LLM semantic vector (genome-derived) + 128-dim learned item embedding (co-watch+content)")
    print("LIKELY MISSING / NOT WIRED (candidate holes -- verify per movie, don't assume):")
    print("  - MPAA/content rating; runtime (scraped but unused); streaming availability; box office; awards/Oscars;")
    print("    explicit pacing / cinematography / soundtrack-composer axes; franchise/universe membership;")
    print("    TV/video-game cross-media; 'set in' vs 'filmed in' distinction")
    print(f"CORPUS: {len(mt)} movies, MovieLens 32M (>=200 ratings), mostly well-known, through ~2023.")

def search(q):
    q=q.lower(); hits=[(mt[m],my.get(m)) for m in mt if mt[m] and q in mt[m].lower()]
    for t,y in sorted(hits)[:40]: print(f"  {t} ({y})")
    print(f"  ({len(hits)} matches)")

def title(q):
    m=_title_to_mid.get(q.lower())
    if m is None:
        cands=[mt[x] for x in mt if mt[x] and q.lower() in mt[x].lower()][:8]
        print(f"NOT an exact title. Did you mean: {cands}"); return
    print(f"TITLE: {mt[m]} ({my.get(m)})  movieId={m}")
    fs=serving()
    print(f"  genres      : {fs['movieId_to_genres'].get(m)}")
    print(f"  country      : {mc.get(m)}   language: {mol.get(m)}")
    ks=mk.get(m,[])
    print(f"  keywords({len(ks)}): {ks}")
    fac=facet(); ppl=fac['movieId_to_people'].get(m,{})
    idn=fac['person_id_to_name']
    for role in ('directors','actors','writers'):
        names=[idn.get(p,p) for p in ppl.get(role,[])][:8]
        print(f"  {role:10s}: {names}")

def kw(term):
    n=sum(1 for ks in mk.values() if any(term in k for k in ks))
    from collections import Counter
    heads=Counter()
    for ks in mk.values():
        for k in ks:
            if term in k: heads[k]+=1
    print(f"keyword '{term}': {n} movies. heads: {heads.most_common(8)}")

def genome(term):
    fs=serving(); gs=[g for g in fs['genome_tag_names'].values() if term in g.lower()]
    us=[u for u in fs['tags_ordered'] if term in u.lower()]
    print(f"GENOME tags matching '{term}' ({len(gs)}): {gs[:20]}")
    print(f"USER tags matching '{term}' ({len(us)}): {us[:15]}")

if __name__=='__main__':
    cmd=sys.argv[1] if len(sys.argv)>1 else 'inventory'
    arg=' '.join(sys.argv[2:])
    {'inventory':lambda:inventory(),'search':lambda:search(arg),'title':lambda:title(arg),
     'kw':lambda:kw(arg),'genome':lambda:genome(arg),'prompts':lambda:prompts(arg)}.get(cmd, inventory)()
