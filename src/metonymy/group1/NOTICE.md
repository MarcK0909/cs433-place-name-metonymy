# Group 1 attribution and local changes

Source: https://github.com/impresso/impresso-metonymy-workbench
Branch inspected: group1-ce-pipeline
Pinned commit: d5234ba3110b55101111685e0df9da521ba2c0d7
License: GNU Affero General Public License v3, retained in LICENSE here.
Original authors: the Impresso workbench / CS-433 Group 1 contributors.

Copied from group1/scripts: common.py, passages.py, torch_utils.py,
train_encoder.py. probe.py contains only fit_logistic from baselines.py.

Group 2 modifications:
- Package-relative imports; default paths relative to the working directory;
  generated results under outputs/runs instead of results/runs.
- Optional CE + lambda * SupCon, one encoder forward per batch, normalized
  pooled vectors for SupCon only. Zero lambda uses the original CE forward.
- Explicit lambda/temperature validation, distinct run names, component losses
  and positive-partner coverage in epoch logs; upstream commit in run metadata.
- Reject TEST filenames in train/dev/eval and overlapping train/evaluation names.

Pooling, passage reconstruction, tokenization, shuffled batches, optimizer,
learning-rate schedule, early stopping and prediction output follow upstream.
The upstream scheduler behavior on skipped fp16 steps is retained for parity.
No LLM code, results, newspaper text or model weights were imported.
