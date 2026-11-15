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

At implementation time, `data/raw/` and `data/processed/` contain only `.gitkeep`.
**No TRAIN, DEV, or TEST data are available**: class distributions and research
baseline scores cannot yet be reported. No data were downloaded or synthesized
for experiments. Synthetic records are used only in tests.

Put supervisor-provided files under these directories; datasets and generated
outputs are gitignored. Original files must remain unchanged. Use the official
project split assignments; the tools never create or infer splits. CLI paths
explicitly identify split roles, and the user must verify their provenance.
English (`--language en`) is the default for the first baseline. This is a planned
language choice, not an observation about the missing data. Other languages
require an explicit flag and are never mixed within a run. Language codes match
exactly; the inspector reports available languages and excluded row counts.

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

Raw HIPE extraction is deliberately not implemented because no raw HIPE files
were supplied. A token TSV is not accepted as prepared mention-level input. If
raw data arrive, first inspect column values, IOB spans, document/sentence
boundaries, and spacing metadata, then confirm the place-name and metonymy
annotation semantics with supervisors before implementing a separate parser.

## Commands

Inventory the currently available data (works with empty directories):

```bash
PYTHONPATH=src python scripts/inspect_data.py --data-dir data
```

Once prepared official splits exist (replace these illustrative paths with their
actual filenames):

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

Before conducting an experiment, ask supervisors for the official split files,
the designated first language, and either the prepared schema/target encoding
or documented HIPE label semantics and context boundaries. No annotation
interpretation has been assumed in the absence of those files.

## Next steps

1. Standard mmBERT + cross-entropy baseline.
2. Supervised contrastive objective.
3. Controlled CE vs CE+SupCon comparison.
4. Representation analysis.
5. Error analysis.
6. Optional auxiliary datasets / multilingual extensions.
