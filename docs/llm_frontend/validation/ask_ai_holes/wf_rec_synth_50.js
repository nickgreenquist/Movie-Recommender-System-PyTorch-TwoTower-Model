export const meta = {
  name: 'ask-ai-holes-rec',
  description: 'Oracle-rec the 500 harvested Ask-AI prompts on Sonnet (grounded, coverage-tagged) then synthesize the intake+post-filter plan holes',
  phases: [
    { title: 'OracleRec', detail: '50 Sonnet agents x 10 harvested prompts, grounded coverage tags', model: 'sonnet' },
    { title: 'Synthesize', detail: '6 Sonnet lenses over aggregated coverage/holes', model: 'sonnet' },
  ],
}

const HELPER = '/private/tmp/claude-501/-Users-nickgreenquist-Documents-Movie-Recommender-System-PyTorch-TwoTower-Model/978b835f-d6d2-45f2-8a29-cda39e52b3d1/scratchpad/corpus.py'
const N_GROUPS = 50

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
const idx = []
for (let i=0;i<N_GROUPS;i++) idx.push(i)
const pairs = await parallel(idx.map(function(i){ return function(){ return agent(
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
  "  - holes: 0-2 blunt statements of what our pipeline literally CANNOT do for this prompt (a signal we lack, an intent the extractor isn't taught, a filter we can't compute). Empty if fully served.\n\n" +
  "Be skeptical — the GOAL is to find gaps, not to look complete. Return all 10.",
  { label:"rec:"+(i+1), phase:'OracleRec', model:'sonnet', schema:REC_SCHEMA }
).then(function(r){ return { archetype: ARCHETYPES[i]||("g"+i), records: (r&&r.records)||[] } }) } }))

const allRecords = []
for (const p of pairs.filter(Boolean)) for (const rec of (p.records||[])) allRecords.push(Object.assign({}, rec, { archetype: p.archetype }))
log("collected " + allRecords.length + " oracle-rec records")

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
const AGG = "AXIS COVERAGE (over " + allRecords.length + " oracle records):\n" + axisRows.join("\n") + "\n\nMOST-GAPPED AXES (PARTIAL+MISSING / total):\n" + gapAxes.slice(0,22).join("\n")
const stride = Math.max(1, Math.floor(allRecords.length/45))
const sampleRecs = allRecords.filter(function(_,i){return i%stride===0}).slice(0,45).map(function(r){
  const gaps=(r.signals||[]).filter(function(s){return s.coverage!=='HAVE'}).map(function(s){return s.axis+':'+s.coverage}).join(", ")||"none"
  const hole=(r.holes&&r.holes.length)?" | HOLE: "+r.holes.join("; "):""
  return "["+r.archetype+"] \""+r.prompt+"\" -> "+(r.recs||[]).slice(0,4).join(", ")+" | gaps: "+gaps+hole }).join("\n")

phase('Synthesize')
const LENSES = [
  { key:'missing_data', q:"MISSING-DATA axes: signals prompts need that we store NO data for. Rank by frequency; name the concrete source that fills each; cheap (already scraped: runtime/vote_average) vs expensive (new: streaming/awards)." },
  { key:'partial_lossy', q:"PARTIAL/lossy axes we approximate imperfectly (mood via genome, keyword recall gaps, no set-in/filmed-in, popularity proxy). Where does a HARD filter misfire or a soft anchor drift? Which go soft-only vs need a data upgrade?" },
  { key:'intake_gaps', q:"INTAKE-MAPPING gaps: intents the extraction LLM isn't taught to pull (today: liked_titles, liked_people, require/exclude genres+people, require_country/location/keywords, mood). What NEW slots does the real distribution demand? Cover negation/exclusion, multi-constraint composition, occasion, 'no preference/surprise me'." },
  { key:'postfilter_gaps', q:"POST-FILTER gaps: candidate checks the filter LLM needs but CANNOT run with our data (content-rating exclusion, runtime cap, 'no sad ending', franchise membership). Which are filter-side vs intake-side; which are un-fixable without new data?" },
  { key:'unhandled_classes', q:"WHOLE INTENT CLASSES the plan omits: occasion/audience (date night, kids, party), cross-media transfer, streaming, recency, 'surprise me/no preference', self-help/emotional-need. For each: in-scope? minimal handling (route to popularity? follow-up question? new signal)?" },
  { key:'composition_thin', q:"COMPOSITION & THIN-POOL risks: prompts stacking 3+ constraints or needing near-empty intersections in a 9,375-corpus. How should intake mark soft-vs-hard, and how should retrieval degrade (relax which constraint first) instead of empty/popularity?" },
]
const SYN_SCHEMA = { type:'object', required:['lens','findings'], properties:{
  lens:{type:'string'},
  findings:{type:'array', items:{ type:'object', required:['hole','frequency','fix','layer','priority'], properties:{
    hole:{type:'string'}, frequency:{type:'string'}, fix:{type:'string'},
    layer:{type:'string', enum:['intake','postfilter','data-build','multiple']},
    priority:{type:'string', enum:['high','medium','low']} } } } } }
const findings = await parallel(LENSES.map(function(L){ return function(){ return agent(
  "Analyze a measurement run to find HOLES in a movie-recommender plan using 'LLM as intake-mapping layer + LLM as post-retrieval filter layer' over a fixed corpus.\n\n" +
  INVENTORY + "\n\n" +
  "PLAN today (3 engines): (1) membership hard-filters + candidate anchors from keywords/countries/people; (2) vibe/affect soft-anchors routed to existing genome tags; (3) LLM-over-plot extraction DEMOTED (measured non-additive).\n\n" +
  "EVIDENCE from 500 real Ask-AI prompts, oracle-recommended + coverage-tagged:\n\n" + AGG + "\n\nSAMPLE PROMPTS -> recs | gaps:\n" + sampleRecs + "\n\n" +
  "YOUR LENS: " + L.q + "\n\nReturn specific, evidence-backed findings. For each: the hole, ~frequency (cite axis counts), the concrete fix, the layer (intake/postfilter/data-build/multiple), priority. Concrete and prioritized; no fluff.",
  { label:"synth:"+L.key, phase:'Synthesize', model:'sonnet', schema:SYN_SCHEMA }
).then(function(r){ return Object.assign({ lens:L.key }, r||{}) }) } }))

return { n_records: allRecords.length, axis_table: axisRows, gap_axes: gapAxes,
  top_holes: Object.entries(holeCounts).sort(function(a,b){return b[1]-a[1]}).slice(0,40),
  synthesis: findings.filter(Boolean), all_records: allRecords }
