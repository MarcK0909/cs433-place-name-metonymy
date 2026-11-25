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
The Batch 2 count audit is described below.

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

Before experiments, review the Batch 2 audit findings and confirm split provenance
and Partial-offset treatment. Batch 1 uses the explicitly supplied target/label rules
and does not run baseline experiments on raw HIPE data.

## Batch 2 — pinned French/German count audit

```bash
python scripts/audit_hipe.py
python -m pytest -q
HIPE_REAL_DATA_TESTS=1 python -m pytest -q tests/integration/test_hipe_counts.py
```

The audit reads only the unmasked `train`, `dev`, and `test` TSVs under
`data/raw/HIPE-2022-data/data/v2.1/hipe2020/{fr,de}/`. It verifies both the checkout
revision and each file's bytes against that revision. `--data-root` and `--output`
are configurable; the detailed JSON defaults to ignored
`outputs/audits/hipe_counts.json`. No splits or source annotations are changed.
TEST is inspected only for counts, not for model selection or experiments.

| Language | Split | LITERAL | METONYMIC | Total |
|---|---|---:|---:|---:|
| fr | train | 2644 | 445 | 3089 |
| fr | dev | 670 | 104 | 774 |
| fr | test | 743 | 111 | 854 |
| de | train | 1415 | 325 | 1740 |
| de | dev | 501 | 87 | 588 |
| de | test | 477 | 118 | 595 |
| combined | train | 4059 | 770 | 4829 |
| combined | dev | 1171 | 191 | 1362 |
| combined | test | 1220 | 229 | 1449 |

All workbench counts match **audit candidates**, including two flagged TRAIN
mentions. Strict extraction remains unchanged and rejects their documents:

- French `GDL-1948-04-15-a-i0035`: token 1461 `France` is `B-loc` with METO `O`;
  token 1462 `moderne` is the inconsistent `I-org`. Source lines 135786–135787.
  The flagged candidate `[1461:1462]` is LITERAL by the supplied METO rule.
- German `luxwort-1938-05-20-a-i0023`: token 26 `Frankreichs` is `B-loc` with
  METO `B-pers`; tokens `[27:31]` (`größten Außenminister Deleass .`) are
  inconsistent `I-pers` continuations. Source lines 82459–82463. The flagged
  candidate `[26:27]` is METONYMIC by the supplied aligned METO rule.

Smallest audit-only handling: explicitly pass a `review_issues` list to
`extract_mentions`, retain the source `B-loc` span ending before the type change,
and log the malformed I run without promoting it to a new entity or changing
any tags. Adjacent candidates retain `annotation_issues`. Default parsing still
raises. Invalid tags, METO overlap with a malformed run, and span misalignment
remain errors even in audit mode. Partial examples remain included, with offsets
unchanged; nested annotations never add targets. Neither known case is Partial.

The JSON separates strict counts from candidate counts and records source rows,
line numbers, hashes, and review issues. Strict successful-document-only counts
are fr TRAIN 2639/440 and de TRAIN 1412/320 (LITERAL/METONYMIC); these are incomplete,
not alternative dataset totals. All DEV/TEST files pass strict extraction.
Matching counts does not resolve the malformed annotations: manually review both
flagged candidates before adopting an experimental inclusion policy. The audit
writes no model-ready dataset and does not alter baseline behavior.

## Contrastive batch 1 — isolated loss, no training yet

`src/metonymy/contrastive.py` implements `supervised_contrastive_loss` on a
batch of pooled mention vectors and integer binary labels. It is independent
of mmBERT and the data parser. The four steps in the code are:

1. Identify other mentions with the same label (positive pairs).
2. Normalize vectors to unit length and compute cosine similarities divided
   by an explicit temperature.
3. Compute log probabilities over all other mentions, excluding self-pairs.
4. Average negative log probabilities over positives, then over anchors with
   at least one positive.

An anchor without a positive is omitted from this average; it can still be a
negative comparison for other anchors. With no positive pairs anywhere, the
loss is differentiable zero. No mentions are removed from the classification
task. A single-class batch is allowed, although it supplies no negative pairs.
No projection head, extra views, sampler changes, or model downloads are added.

Later integration will use `CE + lambda * SupCon`, keeping the classifier's
original pooled vectors and normalizing only for SupCon. Temperature and lambda
are not selected by this batch. Compute the loss outside mixed-precision
autocast; half-precision feature inputs are promoted to float32 internally.

PyTorch is optional for the existing data pipeline. The `contrastive` extra pins
`torch==2.9.1` to Group 1's version; the GPU build will be installed explicitly
when setting up the server. Synthetic loss tests run with installed PyTorch
and skip when it is absent:

```bash
python -m pytest -q tests/test_contrastive.py
python -m pytest -q
```

These tests check a hand-computed loss, label separation, normalization,
singleton/empty-positive behavior, numerical gradients, and invalid inputs.
They require neither a pretrained model nor the real dataset.

## Next steps

1. Standard mmBERT + cross-entropy baseline.
2. Supervised contrastive objective.
3. Controlled CE vs CE+SupCon comparison.
4. Representation analysis.
5. Error analysis.
6. Optional auxiliary datasets / multilingual extensions.
