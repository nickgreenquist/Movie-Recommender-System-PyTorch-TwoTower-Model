export const meta = {
  name: 'llm-frontend-judge-v5',
  description: 'Strict sonnet QA judge: score extraction + recommendations per case, cluster failure modes',
  phases: [{ title: 'Judge', detail: 'one sonnet agent per case, reads its case JSON' }],
}

const DIR = '/Users/nickgreenquist/Documents/Movie-Recommender-System-PyTorch-TwoTower-Model/docs/llm_frontend_validation/v5/cases_v5'
// Case ids written by Stage 2 (0..159 minus 17, whose extraction errored). Embedded directly
// so the run does not depend on args plumbing.
const IDS = [45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 143, 144, 145]

const V = {
  type: 'object', additionalProperties: false,
  properties: {
    intent_capture:       { type: 'integer', description: '1-5: did the extraction capture the stated intent (right titles/mood/genres/constraints), nothing invented or missed' },
    tag_quality:          { type: 'integer', description: '1-5: genome_tags specific & in-vibe, in closed vocab, NOT generic genre-echoes (comedy/crime/epic/action). 5 if none were needed' },
    resolution_quality:   { type: 'integer', description: '1-5: named titles resolved to the RIGHT films. 5 if no titles named' },
    recs_relevance:       { type: 'integer', description: '1-5: do the top recs match the request taste/mood/era/constraints' },
    constraints_respected:{ type: 'integer', description: '1-5: year/genre hard filters correctly applied. 5 if none' },
    severity:             { type: 'string', enum: ['none', 'minor', 'major', 'critical'] },
    failure_mode:         { type: 'string', description: 'SHORT clusterable label, "none" if clean' },
    suggested_fix:        { type: 'string', description: 'CONCRETE: prompt edit / harness edit / "none"' },
    fix_layer:            { type: 'string', enum: ['prompt', 'harness', 'schema', 'none', 'wontfix'] },
    rationale:            { type: 'string', description: 'one sentence' },
  },
  required: ['intent_capture', 'tag_quality', 'resolution_quality', 'recs_relevance', 'constraints_respected', 'severity', 'failure_mode', 'suggested_fix', 'fix_layer', 'rationale'],
}

function jp(i) {
  return `You are a STRICT QA judge for a movie recommender's natural-language front-end. Use the Read tool to read this test-case JSON: ${DIR}/case_${i}.json

It contains: the user "utterance", the LLM-"extraction" (the structured query the LLM parsed from it), and the trained two-tower model's top-15 "recs" (each with title/genres/year/cosine). It also has "resolution" (how named titles were fuzzy-matched to catalog films — [raw, matched_title_or_null, note]), "anchors" (Mode-2 anchor movies synthesized from genome_tags), "unresolved_tags" (genome tags the LLM emitted that were NOT in the closed vocab — these are BUGS), "unknown_genres", "fallback" (true = empty extraction routed to a popularity default, which is CORRECT for vague requests), and "filtered" (count dropped by year/genre post-filter).

ARCHITECTURE you are judging: the LLM ONLY parses intent into the structured query. The trained model does all retrieval. So judge the EXTRACTION on faithfulness to the utterance, and judge the RECS on whether they serve the request.

Score each 1-5 (5 = perfect; be skeptical, reserve 5):
- intent_capture: did the extraction capture what the user actually said — correct titles in liked/disliked_items, mood in genome_tags, soft genres, and hard constraints — without inventing or dropping intent? Unsupported facets (director/actor/studio/content-rating/runtime/"black & white") MUST be silently dropped while the rest is still captured; dropping them correctly is GOOD, inventing a field for them is BAD.
- tag_quality: are genome_tags distinctive vibe/style/atmosphere words drawn from the closed vocab, NOT lazy genre-echoes (comedy/crime/thriller/epic/funny/romantic/action) that restate a genre and pull results off-target? Any tag in unresolved_tags = out-of-vocab = a real defect. 5 if no tags were needed (e.g. pure title request) and none were emitted.
- resolution_quality: did each named title resolve to the RIGHT film (check "resolution" notes — a fuzzy match to a wrong-year/wrong franchise film, e.g. "Lord of the Rings" → the 1978 animated film, is a FAILURE)? 5 if no titles were named.
- recs_relevance: do the top recs actually match the request's taste/mood/era/genre? Watch specifically for: dark/gritty requests collapsing into a war/documentary cluster; mixed (titles + mood) requests where the mood anchors swamp the named titles and drag results off-target; generic-popular drift on specific requests.
- constraints_respected: were year_min/year_max and require/exclude_genres correctly applied (the recs' years/genres should obey them)? 5 if there were no hard constraints.

Then: severity (none/minor/major/critical — critical = unusable or broken output, major = clearly wrong results, minor = imperfect but usable); failure_mode (a SHORT reusable cluster label like "genre-echo-tags", "anchor-dilution", "wrong-title-resolution", "dark-gritty-war-drift", "unsupported-not-dropped", "missed-constraint", "out-of-vocab-tag", or "none"); suggested_fix (CONCRETE — name the prompt rule to add/change, or "none"); fix_layer (prompt = fixable by editing the extraction system prompt; harness = needs retrieval/weight/post-filter code; schema = needs a new field; none; wontfix = inherent v1 limitation); and a one-sentence rationale.

Reward correct empties: a vague request ("something good") that yields an all-empty extraction and fallback=true is CORRECT and should score well. Emit your scores via the StructuredOutput tool.`
}

phase('Judge')
const verdicts = await parallel(IDS.map(i => () =>
  agent(jp(i), { model: 'sonnet', schema: V, label: `j:${i}`, phase: 'Judge' })
    .then(v => ({ id: i, ...v }))
    .catch(e => ({ id: i, error: String(e) }))))

// Reduce in-script so the returned object stays compact.
const ok = verdicts.filter(v => !v.error)
const avg = k => +(ok.reduce((s, v) => s + (v[k] || 0), 0) / ok.length).toFixed(2)
const sevRank = { critical: 0, major: 1, minor: 2, none: 3 }
const fails = ok.filter(v => v.severity === 'major' || v.severity === 'critical')
  .sort((a, b) => (sevRank[a.severity] - sevRank[b.severity]))
const byMode = {}
ok.forEach(v => { if (v.failure_mode && v.failure_mode !== 'none') byMode[v.failure_mode] = (byMode[v.failure_mode] || 0) + 1 })
const byLayer = {}
ok.forEach(v => { if (v.fix_layer && v.fix_layer !== 'none') byLayer[v.fix_layer] = (byLayer[v.fix_layer] || 0) + 1 })
const sevCounts = { none: 0, minor: 0, major: 0, critical: 0 }
ok.forEach(v => { sevCounts[v.severity] = (sevCounts[v.severity] || 0) + 1 })

return {
  n: ok.length,
  errors: verdicts.length - ok.length,
  avg: {
    intent: avg('intent_capture'), tags: avg('tag_quality'),
    resolution: avg('resolution_quality'), recs: avg('recs_relevance'),
    constraints: avg('constraints_respected'),
  },
  severity_counts: sevCounts,
  failure_modes: byMode,
  fix_layers: byLayer,
  failures: fails.map(v => ({ id: v.id, cat: undefined, severity: v.severity, failure_mode: v.failure_mode, fix_layer: v.fix_layer, fix: v.suggested_fix, rationale: v.rationale })),
  all_scores: ok.map(v => ({ id: v.id, ic: v.intent_capture, tq: v.tag_quality, rq: v.resolution_quality, rr: v.recs_relevance, cr: v.constraints_respected, sev: v.severity, mode: v.failure_mode, layer: v.fix_layer })),
}
