# Place-Name Metonymy Classification

This EPFL CS-433 machine learning project studies binary classification of marked
historical place-name mentions as **LITERAL** (0) or **METONYMIC** (1).
For example, “The delegation arrived in [TGT] Paris [/TGT].” is literal;
“[TGT] Paris [/TGT] rejected the agreement.” is metonymic.
The later research comparison is **standard mmBERT classification vs mmBERT +
supervised contrastive learning**.

## Current stage

**Stage 1 — Dataset inspection + majority-class baseline**

The existing `src/metonymy/`, `scripts/`, `tests/`, `data/`, `outputs/`, and
`notebooks/` scaffold is retained. No transformer or training framework is used.

## Setup

Python 3.10 or newer is required. Reuse your existing environment or create one:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Dependencies are the existing pandas, scikit-learn, and pytest packages. No new
runtime dependencies were added. Run commands from the repository root.
If dependencies are already installed, the commands below work without an
editable install by using `PYTHONPATH=src`.

## Data and current availability

The project uses **raw HIPE-2022 v2.1, dataset `hipe2020`**, with **French and
German** as the main languages. The local source is
`data/raw/HIPE-2022-data/`, pinned to revision
`2439cb5cfbcd7b65aff6a25b7715398f8f85e034`. Raw data remain unchanged and gitignored.
The complete dataset audit is deferred to Batch 2.

Batch 1 adds `metonymy.hipe.load_hipe_mentions(path, language="fr", split="train")`
as a separate raw parsing path. It extracts complete `NE-COARSE-LIT` location
spans, labels all-O aligned METO spans LITERAL and fully aligned contextual
entities METONYMIC (including loc → loc), and rejects malformed BIO or misaligned
spans. Returned records preserve document metadata, source rows/line numbers,
zero-based end-exclusive document token boundaries, and stable IDs. They do not
yet contain model context `text` and are not inputs to the prepared-data baseline.
`parse_documents` also exposes all document rows and comments for later inspection.
Callers must verify the source version and explicitly supply language and split.

Mention reconstruction respects `NoSpaceAfter`. Partial flags are preserved
verbatim without exclusion or character slicing; `mention` retains the complete
source-token surface. Nested annotations are retained but never add targets.
Upstream documentation warns of erroneous Partial offsets. Its statement that
no training set exists conflicts with the supplied TRAIN filenames; resolve
split provenance during Batch 2 before experiments.

The prepared-data loader and CLI defaults remain unchanged (including `en`);
they are not yet adapted to the French/German raw-data workflow. They never
create splits or mix languages within a run.

### Prepared mention-level input contract

The loader accepts JSON arrays (`.json`), JSON Lines (`.jsonl`), CSV, or TSV.
Each record represents one target mention and contains:

```json
{
  "example_id": "train-en-001",
  "text": "The delegation arrived in [TGT] Paris [/TGT].",
  "mention": "Paris",
  "label": 0,
  "language": "en",
  "document_id": "newspaper-001"
}
```

Required fields are nonempty string `example_id`, `text`, `mention`, `language`,
and an explicit label. Labels may be integer 0/1, string "0"/"1", or exact names
"LITERAL"/"METONYMIC". A `label_name` field may supply the label when `label` is
absent; when both are present they must agree. Normalized output always contains
integer `label`, `label_name`, and nullable `document_id`. Additional metadata,
text, mention spelling, and target markers are preserved verbatim. Missing
`document_id` becomes null; no target markers, offsets, or labels are inferred.
Keep target markers or offsets supplied by supervisors so later models can
identify the intended occurrence when a name repeats in context.

This is a supported input contract, **not a claim about the supervisors’ actual
schema**. Adapt field mapping after inspecting their files if necessary. Invalid
records fail baseline loading with a row-level error. Inspection reports them
without changing source files; statistics explicitly cover valid selected-language
rows, with total input, invalid, and other-language counts also shown.

Raw HIPE files use the separate `metonymy.hipe` parser described above; the
prepared-data loader continues to reject token TSV input.

## Commands

Inventory the currently available data (works with empty directories):

```bash
PYTHONPATH=src python scripts/inspect_data.py --data-dir data
```

Legacy prepared-data commands (illustrative paths, not the raw HIPE workflow):

```bash
PYTHONPATH=src python scripts/inspect_data.py --train data/processed/train.jsonl --dev data/processed/dev.jsonl --language en
PYTHONPATH=src python scripts/run_majority_baseline.py --train data/processed/train.jsonl --dev data/processed/dev.jsonl --language en
python -m pytest -q
```

Add `--test data/processed/test.jsonl` to inspection only when an official TEST
split exists. Final TEST evaluation requires explicit opt-in:

```bash
PYTHONPATH=src python scripts/run_majority_baseline.py --train data/processed/train.jsonl --test data/processed/test.jsonl --split test --language en
```

Inspection prints a JSON report with counts and percentages for both classes,
unique mentions, character-length min/max/mean/median for text and mentions,
and up to five examples per class sampled with `--seed 433`. It warns about
invalid/empty fields, unexpected labels, duplicate IDs, exact duplicate content,
ID overlap across supplied splits, and a class share of at least 90%.
Duplicates are reported without removal. Duplicate content is defined using
text, mention, label, language, and document ID, ignoring example ID and extra
metadata. Split overlap checks use valid selected-language example IDs; these
checks cannot prove the absence of document-level or semantic leakage.

Baseline results are written to `outputs/baselines/majority_en_dev.json`
(configurable with `--output-dir`). They include input paths, sizes, majority
label, tie rule, and all metrics. **Macro-F1** is the main metric; accuracy,
per-class precision/recall/F1/support, and the confusion matrix are also reported.
Both classes always participate in Macro-F1, even when absent. Undefined scores
use `zero_division=0`. Confusion matrix rows are true classes and columns are
predicted classes, ordered LITERAL, METONYMIC.

## Experimental rules

- TRAIN alone determines the majority class. Ties deterministically select
  LITERAL (0); every evaluation example receives that class.
- DEV is used for development and is the default evaluation split.
- TEST is reserved for final evaluation, never model selection.
- Empty TRAIN/evaluation data, malformed records, or TRAIN/evaluation ID overlap
  stop the baseline. No missing split is fabricated, no labels are manually
  revised, and no raw data are modified.

Before experiments, complete the Batch 2 audit and confirm split provenance and
Partial-offset treatment. Batch 1 uses the explicitly supplied target/label rules
and does not run baseline experiments on raw HIPE data.

## Next steps

1. Standard mmBERT + cross-entropy baseline.
2. Supervised contrastive objective.
3. Controlled CE vs CE+SupCon comparison.
4. Representation analysis.
5. Error analysis.
6. Optional auxiliary datasets / multilingual extensions.
