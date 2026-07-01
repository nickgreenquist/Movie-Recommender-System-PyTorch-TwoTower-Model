export const meta = {
  name: 'ask-ai-holes-20',
  description: 'First cut: oracle-rec 200 harvested Ask-AI prompts (20 hole-rich archetypes) on Sonnet with grounded coverage tags; no synthesis (done by main loop)',
  phases: [ { title: 'OracleRec', detail: '20 Sonnet agents x 10 harvested prompts', model: 'sonnet' } ],
}

const HELPER = '/private/tmp/claude-501/-Users-nickgreenquist-Documents-Movie-Recommender-System-PyTorch-TwoTower-Model/978b835f-d6d2-45f2-8a29-cda39e52b3d1/scratchpad/corpus.py'

const ARCHETYPES = [
  "casual surprise-me browser","emotional cry/catharsis","feel-good comfort watch","dark/bleak/disturbing",
  "date-night couple","family w/ young kids","friends party movie","solo late-night unwind",
  "auteur/arthouse cinephile","film-history classics","foreign cinema by country","horror subgenres",
  "sci-fi deep-diver","actor fan","director fan","movies-like-X similarity","80s/90s nostalgia",
  "set-in-city location","travel inspiration","specific subject/theme","runtime-constrained",
  "content-sensitivity","exclusion-driven","compositional multi-constraint","vibe/aesthetic","pacing",
  "ending preference","based-on source","awards/acclaim","period/era","hidden gems/obscure","recency/new",
  "rewatchable comfort","thought-provoking/mind-bending","adrenaline/action","romance flavors",
  "comedy flavors","documentary/true-crime","animation/anime","music-driven","heist/caper",
  "hyper-specific object/scene","character-type","representation/diversity","franchise/universe",
  "cross-media","seasonal/occasion","emotional-need/self-help","guilty-pleasure/trashy","niche intersection",
]
// 20 chosen indices: 5 well-served baselines + 15 gap-prone (to surface holes in a small first cut)
const SUBSET = [2,10,13,15,17, 20,21,22,23,24, 25,26,28,30,31, 43,44,45,46,47]

const AXES = ["genre","subgenre","theme_subject","setting_location","nationality_country","language",
  "era_period","decade_year","person_actor","person_director","person_writer","person_composer",
  "mood_affect","tone_maturity","pacing","visual_style","ending_type","based_on_source",
  "format_bw_silent","franchise_universe","similar_to_title","content_rating","runtime",
  "awards_acclaim","popularity_obscurity","streaming_availability","character_type",
  "representation_diversity","occasion_audience","cross_media","object_prop","plot_structure","other"]

const INVENTORY =
  "OUR RECOMMENDER'S DATA (what it can actually use — do NOT invent capabilities):\n" +
  "  HAVE: genres(20), release year/decade, user-tags(306), GENOME tags(1128 ML-relevance incl. affect/tone:\n" +
  "    sad/emotional/heartbreaking, dark/gritty/bleak, feel-good/heartwarming, tense/creepy/atmospheric, epic,\n" +
  "    mindfuck/cerebral/twist-ending, quirky/funny), TMDB keywords(17820: themes/places/format/era/some props),\n" +
  "    production_countries(ISO nationality), language, people(top-billed cast+director+writer ids), popularity,\n" +
  "    vote_average, a 132-d semantic vector + 128-d item embedding (co-watch+content).\n" +
  "  MISSING / NOT WIRED (be honest — mark these MISSING/PARTIAL): MPAA/content-rating, runtime(scraped but unused),\n" +
  "    streaming availability, box-office, awards/Oscars, explicit pacing/cinematography/soundtrack-composer axes,\n" +
  "    franchise/cinematic-universe membership, TV/video-game cross-media, 'set in' vs 'filmed in'.\n" +
  "  CORPUS: 9366 movies (MovieLens >=200 ratings), mostly well-known, through ~2023."

const REC_SCHEMA = { type:'object', required:['records'], properties:{ records:{ type:'array',
  items:{ type:'object', required:['prompt','recs','signals','holes'], properties:{
    prompt:{type:'string'},
    recs:{type:'array', items:{type:'string'}, description:'4-6 IN-CORPUS titles that ideally satisfy the prompt'},
    signals:{type:'array', items:{ type:'object', required:['axis','coverage','layer'], properties:{
      axis:{type:'string', enum:AXES},
      coverage:{type:'string', enum:['HAVE','PARTIAL','MISSING']},
      layer:{type:'string', enum:['intake','postfilter','both']},
      note:{type:'string', description:'max 12 words'} } } },
    holes:{type:'array', items:{type:'string'}, description:'0-2 blunt statements of what our pipeline cannot do here'} } } } } }

phase('OracleRec')
const pairs = await parallel(SUBSET.map(function(i){ return function(){ return agent(
  "You are a MASTER FILM CURATOR acting as an ORACLE. Produce IDEAL movie recs for each user prompt AND expose which signals you used, tagged for whether OUR recommender can actually serve them — so we can find gaps.\n\n" +
  INVENTORY + "\n\n" +
  "First run this to get YOUR 10 prompts:\n  python3 " + HELPER + " prompts " + i + "\n" +
  "You MAY also verify titles/coverage (cheap, optional):\n" +
  "  python3 " + HELPER + " search \"<text>\"   (confirm a title is in-corpus)\n" +
  "  python3 " + HELPER + " genome \"<term>\"   (does an affect/tone tag exist?)\n" +
  "  python3 " + HELPER + " kw \"<term>\"       (keyword coverage)\n" +
  "Don't over-call the helper; the inventory above is authoritative for coverage.\n\n" +
  "For EACH of the 10 prompts return a record:\n" +
  "  - recs: 4-6 in-corpus titles (well-known, through ~2023) that ideally satisfy it.\n" +
  "  - signals: every axis you used to pick them; tag axis (fixed list), coverage (HAVE/PARTIAL/MISSING vs OUR data above), layer (intake=must be EXTRACTED from the prompt / postfilter=must be VERIFIED on candidates / both).\n" +
  "  - holes: 0-2 blunt statements of what our pipeline literally CANNOT do for this prompt. Empty if fully served.\n\n" +
  "Be skeptical — the GOAL is to find gaps, not to look complete. Return all 10.",
  { label:"rec:"+i+" "+(ARCHETYPES[i]||""), phase:'OracleRec', model:'sonnet', schema:REC_SCHEMA }
).then(function(r){ return { archetype: ARCHETYPES[i]||("g"+i), records: (r&&r.records)||[] } }) } }))

const allRecords = []
for (const p of pairs.filter(Boolean)) for (const rec of (p.records||[])) allRecords.push(Object.assign({}, rec, { archetype: p.archetype }))
log("collected " + allRecords.length + " oracle-rec records from " + pairs.filter(Boolean).length + " archetypes")

const axisTally = {}; const holeCounts = {}
for (const r of allRecords) {
  for (const s of (r.signals||[])) {
    const a = axisTally[s.axis] || (axisTally[s.axis] = {HAVE:0,PARTIAL:0,MISSING:0,total:0,intake:0,postfilter:0})
    a[s.coverage] = (a[s.coverage]||0)+1; a.total++
    if (s.layer==='intake'||s.layer==='both') a.intake++
    if (s.layer==='postfilter'||s.layer==='both') a.postfilter++
  }
  for (const h of (r.holes||[])) { const k=h.trim().toLowerCase(); holeCounts[k]=(holeCounts[k]||0)+1 }
}
const axisRows = Object.entries(axisTally).sort(function(x,y){return y[1].total-x[1].total})
  .map(function(e){ const ax=e[0],t=e[1]; return ax+": n="+t.total+" HAVE="+t.HAVE+" PARTIAL="+t.PARTIAL+" MISSING="+t.MISSING+" (intake "+t.intake+"/postfilter "+t.postfilter+")" })
const gapAxes = Object.entries(axisTally).map(function(e){ const ax=e[0],t=e[1]; return {ax:ax,gap:t.PARTIAL+t.MISSING,total:t.total} })
  .filter(function(x){return x.gap>0}).sort(function(a,b){return b.gap-a.gap}).map(function(x){return x.ax+" ("+x.gap+"/"+x.total+")"})

return { n_records: allRecords.length, axis_table: axisRows, gap_axes: gapAxes,
  top_holes: Object.entries(holeCounts).sort(function(a,b){return b[1]-a[1]}).slice(0,40),
  all_records: allRecords }
