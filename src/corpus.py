"""
Corpus selector — namespaces every pipeline artifact by which corpus it belongs to, so the
reduced ('phase1') and full corpora coexist on disk with clear provenance instead of swapping
whole directories back and forth (the old `*_4k_backup/` plan). This is the single most likely
source of a "which corpus was that?" mistake, so the corpus is baked into the filename AND
printed by every stage at runtime.

Single source of truth, set per-invocation via the CORPUS env var (default 'full'):

    CORPUS=phase1 python main.py dataset

The 'full' corpus keeps the historical *unsuffixed* filenames (e.g. features_movies_v1.parquet),
so existing artifacts and the prod checkpoint load unchanged; only non-full corpora get a
'_<corpus>' suffix. This namespacing is orthogonal to the per-artifact version constants
(FEATURES_VERSION, DATASET_VERSION) — both appear in a filename, version first:
features_movies_v1_phase1.parquet.

NOTE: this module only provides the naming plumbing. It does NOT implement the phase1 *filter*
(the higher rating-count threshold that produces the reduced movie list) — that belongs to the
ablation (see llm_vs_genome_ablation_plan.md, Stage 0).
"""
import os

KNOWN_CORPORA = ('full', 'phase1')

CORPUS = os.environ.get('CORPUS', 'full')
if CORPUS not in KNOWN_CORPORA:
    raise ValueError(f"Unknown CORPUS={CORPUS!r}; expected one of {KNOWN_CORPORA}")


def corpus_suffix(corpus: str = CORPUS) -> str:
    """Filename suffix for a corpus: '' for 'full' (historical names), '_<corpus>' otherwise."""
    return '' if corpus == 'full' else f'_{corpus}'
