# Place-Name Metonymy Classification

EPFL CS-433 project: classify supplied historical place-name mentions as
**LITERAL (0)** or **METONYMIC (1)**. The research comparison is **mmBERT with
cross-entropy (CE) versus the same mmBERT with CE + supervised contrastive loss**.
For example, “arrived in Paris” is literal; “Paris rejected the agreement” is
metonymic.

The data audit, Group 1 input compatibility, contrastive loss, trainer integration
and CPU tests are implemented. The server environment and data preparation have
passed their checks. **This implementation has not yet passed a real mmBERT GPU
smoke test or produced verified experiment results.**

The shared workbench now also implements SupCon on its
[`supcon` branch](https://github.com/impresso/impresso-metonymy-workbench/tree/supcon).
Consolidation is tracked in [issue #4](https://github.com/MarcK0909/cs433-place-name-metonymy/issues/4):
retain this branch as a reference while transferring useful tests, data checks,
and diagnostics into one shared trainer. The local loss PR is on hold.
The commands below describe this repository's implementation; confirm the shared
run configuration before launching duplicate experiments.

## Setup

Run commands from the repository root. The data tools need Python 3.10+ and the
base dependencies; the shared encoder environment uses Python 3.12.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
# Optional CPU/Mac encoder environment:
python -m pip install -e '.[encoder]'
```

The base dependencies are pandas, scikit-learn and pytest. The `contrastive`
extra adds PyTorch; `encoder` pins Group 1's training dependencies. Data inspection
does not require the encoder extras.

On the EPFL server, create the separate `~/venvs/metonymy-group2` environment:

```bash
METONYMY_PYTHON=python3.12 bash server/setup_env.sh \
  && source ~/venvs/metonymy-group2/bin/activate
```

The server's default `python3` may be 3.13: select `python3.12` explicitly.
The script installs PyTorch 2.9.1 with CUDA 12.6 and runs CPU tests offline. It
starts no GPU job and downloads no model weights. `METONYMY_VENV` can override
the environment directory.

## Data preparation and audit

Use **HIPE-2022 v2.1, dataset `hipe2020`, French and German**, pinned to commit
`2439cb5cfbcd7b65aff6a25b7715398f8f85e034`. Keep raw files unchanged. Datasets,
prepared files, checkpoints and generated outputs are gitignored.

If the checkout is not already available:

```bash
mkdir -p data/raw
git clone https://github.com/hipe-eval/HIPE-2022-data.git data/raw/HIPE-2022-data
git -C data/raw/HIPE-2022-data checkout 2439cb5cfbcd7b65aff6a25b7715398f8f85e034
```

```bash
python scripts/audit_hipe.py
python scripts/prepare_encoder_data.py
```

Both commands accept `--data-root`. The audit checks six unmasked FR/DE splits
and writes `outputs/audits/hipe_counts.json`. Preparation reads **TRAIN and DEV
only**, writing four JSONL files and `preparation_report.json` under
`data/processed/group1/`. Use `--output-dir` to change that destination; replacing
existing prepared files requires `--overwrite`.

Preparation verifies the source revision and raw bytes, source mention/label
agreement with our parser, unique IDs, counts, and complete prepared-file MD5s
against Group 1's recorded CE run. All four prepared checksums match, covering
**4,829 TRAIN and 1,362 DEV examples**. A new malformed annotation stops preparation.
The expected checksums and upstream reference are in
`src/metonymy/group1/data_reference.json`.

| Language | Split | LITERAL | METONYMIC | Total |
|---|---|---:|---:|---:|
| French | TRAIN | 2644 | 445 | 3089 |
| French | DEV | 670 | 104 | 774 |
| French | TEST | 743 | 111 | 854 |
| German | TRAIN | 1415 | 325 | 1740 |
| German | DEV | 501 | 87 | 588 |
| German | TEST | 477 | 118 | 595 |

### Annotation rules and reviewed cases

Targets are complete `B-loc`/`I-loc` spans in `NE-COARSE-LIT`. Aligned METO tags
all equal to `O` mean LITERAL; one exactly aligned contextual entity means
METONYMIC, including **loc → loc**. Nested annotations never add targets.
Partial flags are retained without exclusion or character slicing. Mention
reconstruction respects `NoSpaceAfter` and keeps OCR/source-token spelling.

`metonymy.hipe` is the strict parser. It rejects malformed BIO, invalid tags and
misalignment. An explicit `review_issues` list enables audit candidates without
promoting orphan I runs to new entities. The shared-data bridge keeps the two
agreed TRAIN targets, matching Group 1:

- `GDL-1948-04-15-a-i0035`: `France` at `[1461:1462]`, LITERAL; followed by
  inconsistent `moderne / I-org`. Source lines 135786–135787.
- `luxwort-1938-05-20-a-i0023`: `Frankreichs` at `[26:27]`, METONYMIC; followed
  by inconsistent `I-pers` tokens at `[27:31]`. Source lines 82459–82463.

Offsets above are zero-based, document-relative, end-exclusive. These cases
remain flagged in the audit and prepared notes; neither is Partial. Strict
successful-document-only TRAIN counts are 2639/440 (FR) and 1412/320 (DE),
which omit the entire affected documents and are not dataset totals. All
DEV/TEST documents pass strict extraction. Group 1's in-memory BIO repair and
our audited extraction give the same supplied location targets; raw files are
never edited. Upstream documentation also warns of incorrect Partial offsets.

The encoder consumes Group 1's schema: `id`, `tokens`, `space_after`, `start`,
`end`, `label`, plus provenance. Prepared files keep up to 128 source tokens on
each side; the shared trainer uses **64 per side**, within document boundaries,
and at most 512 subwords. Checksums cover metadata, row order and serialization
as well as inputs and labels.

## Shared encoder and contrastive objective

The encoder code is adapted from Group 1 commit
`d5234ba3110b55101111685e0df9da521ba2c0d7`. Attribution, the retained AGPL-3.0
license and local changes are in `src/metonymy/group1/NOTICE.md` and `LICENSE`.

| Setting | Shared baseline |
|---|---|
| Model | `jhu-clsp/mmBERT-base`, full fine-tuning |
| Representation | Mean of target-mention subwords; dropout 0.1 and linear head |
| Training data | French + German TRAIN |
| Optimizer | AdamW, LR 5e-5, weight decay 0; cosine decay, 10% warm-up |
| Batch / duration | 32; up to 20 epochs, patience 6; gradient clipping 1.0 |
| Selection | Pooled French/German DEV Macro-F1; prediction threshold 0.5 |
| Seeds | 42, 123, 456, 789, 2025 |

The optional objective is `CE + lambda * SupCon`. SupCon normalizes a copy of
the pooled vectors, uses same-label positives and opposite-label negatives
within the existing batch, and excludes self-pairs. It averages positive log
probabilities per anchor, then averages over anchors with a positive. An anchor
without a positive has no own SupCon term but may remain a negative for others;
its CE is retained. With no positive pairs, SupCon returns differentiable zero.
A single-class batch supplies no negatives. There is no projection head, extra
view or new sampler.

`--supcon-weight 0` uses the original CE forward without calling SupCon. Positive
weights require `--temperature`. The encoder runs once per batch; the classifier
uses the original vectors. Loss computation is outside autocast, with float32
features for SupCon. Logs record both losses and positive-partner coverage.
Frozen-encoder probing is not part of this implementation.

## GPU smoke test and training

Use a GPU allocated by the lab, and check `nvidia-smi` immediately before launch.
**Never use GPU 3.** Do not assume GPU 1 is available: it has hosted services.
Replace `G` with an allocated index (0, 1 or 2), then run the smoke tests sequentially:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=G python scripts/smoke_encoder.py --device cuda --supcon-weight 0 --output outputs/smoke/ce.json
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=G python scripts/smoke_encoder.py --device cuda --supcon-weight 0.1 --temperature 0.1 --output outputs/smoke/supcon.json
```

These use the same seeded 32-mention TRAIN sample and perform one optimizer
update. `0.1` is a diagnostic value, not a selected hyperparameter. The first
run downloads mmBERT/tokenizer if uncached; `--model` accepts a local directory.
The smoke test checks finite/nonzero gradients and a classifier update. It
writes IDs, losses, hashes, model revision when available, versions, timing and
memory; no checkpoint or DEV/TEST evaluation. Existing reports are not overwritten.
Nonfinite fp16 gradients stop the test; `--no-fp16` is available for diagnosis.

After the smoke tests pass and pilot settings are agreed, training commands are:

```bash
# CE control
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=G python scripts/train_encoder.py --train hipe_fr_train hipe_de_train --dev hipe_fr_dev hipe_de_dev --data-dir data/processed/group1 --device cuda --supcon-weight 0 --save-model
# Replace WEIGHT and TEMPERATURE with the selected pilot settings.
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=G python scripts/train_encoder.py --train hipe_fr_train hipe_de_train --dev hipe_fr_dev hipe_de_dev --data-dir data/processed/group1 --device cuda --supcon-weight WEIGHT --temperature TEMPERATURE --save-model
```

Outputs go to `outputs/runs/`: configuration, per-epoch metrics and predictions
keyed by mention ID, plus weights with `--save-model`. Confirm Group 1's base-model
revision before comparing full runs. Use a committed Git checkout for recorded
experiments; an rsynced directory without `.git` cannot record the code commit.

Use TRAIN for learning and pairs, DEV for all decisions, and TEST only for the
frozen final evaluation. No splits are invented or merged. TEST filenames are
rejected outside `--final-test`; the preparation command does not export TEST.
Emphasize Macro-F1 and per-class results, report French/German separately, and
compare repeated seeds before interpreting gains. If sampling or views change,
include a CE control with the same change.

## Tests

```bash
python -m pytest -q
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest -q tests/test_encoder.py
HIPE_REAL_DATA_TESTS=1 CUDA_VISIBLE_DEVICES="" python -m pytest -q tests/integration
```

Ordinary tests use synthetic fixtures, tiny random models and local tokenizers;
no pretrained-model downloads. Optional loss/encoder tests skip if their
packages are absent. Real-data tests are separate and opt-in. The retained
Group 1 CE-loop fixture verifies exact CPU loss/selected-weight parity when
lambda is zero; numerical checks validate the contrastive gradients.

## Earlier inspection and majority-baseline tools

These remain available for separately prepared mention-level JSON/JSONL/CSV/TSV:

```bash
PYTHONPATH=src python scripts/inspect_data.py --data-dir data
PYTHONPATH=src python scripts/inspect_data.py --train TRAIN.jsonl --dev DEV.jsonl --language fr
PYTHONPATH=src python scripts/run_majority_baseline.py --train TRAIN.jsonl --dev DEV.jsonl --language fr
```

Their schema uses nonempty `example_id`, `text`, `mention`, `language`, binary
`label` (0/1 or LITERAL/METONYMIC), and optional `document_id`/`label_name`.
Extra metadata and text are preserved. This differs from the encoder schema;
raw HIPE requires the HIPE-specific parser. Inspection reports invalid rows,
duplicates, ID overlap, class balance, lengths and seeded samples without
editing data. The majority label uses TRAIN only; ties select LITERAL. DEV is
the default evaluation, with fixed two-class metrics (`zero_division=0`) and
confusion-matrix rows=true, columns=predicted. Results go to `outputs/baselines/`.

## Next steps

1. Complete the allocated-GPU smoke tests and resolve base-model revision parity.
2. Run a small CE+SupCon pilot; choose weight/temperature on DEV.
3. Compare CE and CE+SupCon under matched settings across repeated seeds.
4. Perform representation and error analysis, then frozen TEST evaluation.
5. Consider auxiliary datasets or multilingual extensions only as separate work.
