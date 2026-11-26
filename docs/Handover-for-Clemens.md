# Group 2 handover for Clemens

Updated 26 November 2026. Audience: Clemens, taking over the CE vs CE+SupCon work.

## 1. What we are trying to establish

Does adding supervised contrastive loss improve mmBERT's classification of historical place-name mentions as LITERAL (0) or METONYMIC (1)? Compare the same encoder, data, seed and training settings with CE alone versus CE + lambda * SupCon. Emphasize Macro-F1 and also report both classes' precision, recall and F1.

Train jointly on French and German HIPE-2020 TRAIN; select checkpoints on French and German DEV. English DEV is an auxiliary diagnostic, not part of the main FR/DE result. TEST stays reserved for final evaluation. Do not change splits or labels.

## 2. Which repository to use

**Develop the shared training implementation here:**
https://github.com/impresso/impresso-metonymy-workbench/tree/supcon

Verified branch tip: `4a9e9898b49de3eb9e51763ee197752197864bfe`. It adds optional SupCon to Group 1's trainer at `d5234ba3110b55101111685e0df9da521ba2c0d7`.

All Python and server commands below run from the shared repository's **group1/** directory. The folder name is historical: Group 2's SupCon changes currently live there too.

Marc's repository is a source of additional tests, diagnostics and data-audit tools, not a second trainer to maintain:
https://github.com/MarcK0909/cs433-place-name-metonymy/tree/feature/group1-data-bridge

Its preserved cleanup tip is `ba4d127`; consolidation is tracked in:
https://github.com/MarcK0909/cs433-place-name-metonymy/issues/4

**Use a separate branch and PR targeting supcon. Do not push directly to supcon.** Ask Sam to review; do not merge main or retire Marc's reference branch before the useful additions have been transferred.

## 3. What is already done, and what is not verified

- Group 1 has CE baseline configurations and DEV predictions in `group1/results/runs/`.
- Sam implemented SupCon in `group1/scripts/train_encoder.py`, with tests and README instructions.
- The loss normalizes pooled mention vectors before classifier dropout. Other examples with the same label in the batch are positives; self-pairs are excluded. Anchors without positives are skipped in the SupCon mean but still contribute to CE. There is no projection head or extra augmented view.
- The local review compared the two implementations on 180 float32 CPU loss/gradient cases: they matched within tolerance. Sam's two SupCon tests passed. These checks do not validate a real GPU run.
- Marc's reference implementation passed 117 tests, including real-data preparation checks. Its prepared FR/DE TRAIN and DEV checksums matched Group 1's reference configurations.
- Sam reported starting training. At the checked shared branch tip, the SupCon commit contains code/tests/docs, not new SupCon results.
- We have not received her actual run config, logs, results or exact running commit. Do not describe the experiment as completed or matched to CE yet.
- Sam said she did not use `--save-model`. Predictions/metrics remain useful; representation analysis will need a saved checkpoint from a future run.

## 4. First task: collect and verify Sam's run artifacts

Ask Sam for:

1. Exact launch command, code commit, and any uncommitted changes used by the job.
2. Run output directory and `config.json`.
3. Training log, completion/exit status, `metrics.json`, and DEV prediction files when available.
4. The corresponding Group 1 CE run directory/config for each seed.
5. Model/tokenizer revision information if available (do not assume a model name proves identical weights).

She supplied this command for seed 42 (historical evidence, **not a request to launch it again**):

```bash
bash server/run.sh supcon_seed42_retry1 --gpu 1 \
  python scripts/train_encoder.py --device cuda \
  --train hipe_fr_train hipe_de_train \
  --dev hipe_fr_dev hipe_de_dev --eval hipe_en_dev \
  --seed 42 --supcon-weight 0.1 --supcon-temperature 0.1
```

The wrapper job name is not the trainer's output-directory name. By default the trainer writes under `group1/results/runs/<generated-run-name>/`; get the actual path from Sam. Wrapper logs live in `group1/logs/`, with a `.exit` file on completion. `retry1` alone does not tell us why a previous attempt was retried.

Compare the saved CE and SupCon configurations, not just the commands. Check data checksums, seed, model/tokenizer, pooling, context, max sequence length, batch size, optimizer, LR, schedule/warmup, dropout, class weights, precision, epoch limit and patience. Document any difference before calling the runs controlled.

Expected shared defaults: mean mention pooling; 64 HIPE tokens per side; maximum 512 subwords; batch 32; AdamW LR 5e-5; cosine schedule, 10% warmup; weight decay 0; dropout 0.1; unweighted CE; up to 20 epochs, patience 6; best pooled FR/DE DEV Macro-F1. Verify these against the actual saved configuration.

Prepared target counts:

| Language | Split | LITERAL | METONYMIC | Total |
| --- | --- | ---: | ---: | ---: |
| French | TRAIN | 2644 | 445 | 3089 |
| German | TRAIN | 1415 | 325 | 1740 |
| French | DEV | 670 | 104 | 774 |
| German | DEV | 501 | 87 | 588 |

Use the shared prepared data and checksums. Do not replace the shared extraction with a strict parser just because both are HIPE parsers: the audit records malformed annotations and handling differences. Preserve the audit and ask before changing extraction semantics.

## 5. Set up your contribution branch

For a new checkout:

```bash
git clone --branch supcon https://github.com/impresso/impresso-metonymy-workbench.git
cd impresso-metonymy-workbench
git switch -c clemens/supcon-validation
git rev-parse HEAD
cd group1
```

For an existing checkout, first check `git status`, preserve local changes, fetch origin, and create a new branch from `origin/supcon`. Do not switch branches in a directory being used by an active training job; use a separate checkout.

### Environment and CPU tests

The shared environment uses Python 3.12 and pinned dependencies in `group1/requirements.txt`, including PyTorch 2.9.1. Use an existing compatible environment if available.

Important server differences:

- Shared `server/run.sh` activates `~/venvs/metonymy` inside the job. Activating a different environment in your shell does not override it.
- Marc's separate setup used `~/venvs/metonymy-group2`; these paths are not interchangeable.
- Shared `server/setup_env.sh` expects `python3` itself to be 3.12. The server's default has previously been 3.13. Check `python3 --version` and `python3.12 --version` first. If the default is wrong, propose an explicit-interpreter fix via PR rather than repeatedly running the script.
- The shared setup runs `scripts/check_setup.py`, which can load/download real models. It is not the same as Marc's offline CPU setup.

With a compatible environment active, run:

```bash
CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python -m unittest discover -s tests -v
```

These are synthetic tests; no real data or real pretrained model download should be needed. Record the command, environment and outcome in the PR. Do not substitute Marc's pytest count for the shared repository's test result.

## 6. First code PR: transfer useful additions, not a second trainer

Keep Sam's shared loss and trainer. Bring over small, reviewable pieces from Marc's repository:

- Additional loss/gradient edge-case tests and a CE-equivalence regression test. Avoid duplicate assertions that already exist.
- Separate epoch logging of CE, raw SupCon, weighted SupCon and eligible/skipped anchor counts. Retain total loss. State how epoch averages are calculated.
- Prepared-data checksum checks and useful smoke diagnostics adapted to shared paths/arguments.

Reference files in Marc's repo: `src/metonymy/contrastive.py`, `src/metonymy/group1/train_encoder.py`, `src/metonymy/group1/smoke.py`, `tests/test_encoder.py`, and `tests/integration/test_encoder_data_parity.py`.

Do not copy the entire trainer or overwrite Group 1's common helpers. Marc removed unused probe/helpers only from his local copy; other shared tools may need them. Shared temperature flag is `--supcon-temperature`; Marc's trainer uses `--temperature`.

Keep data ordering, RNG use, forward passes, optimizer and checkpoint selection unchanged in a diagnostics-only PR. Confirm CE-only behavior remains unchanged. Tests should cover singleton/no-positive batches, finite gradients, and known analytic loss values.

After testing, push your own branch and create a PR with base `supcon`:

```bash
git push -u origin clemens/supcon-validation
```

If you lack push permission, use a fork and a PR targeting the shared `supcon` branch. Explain what changed, why, and exactly what you tested; ask Sam for review. Do not include datasets, newspaper text, weights or raw server logs in the PR.

## 7. GPU work: only after coordination

Do not start another copy of Sam's experiment. First inventory completed/running seeds and confirm what still needs running. A low utilization snapshot is not a reservation; obtain an allocated GPU. The shared launcher explicitly refuses GPU 3 because it is reserved for services. Do not stop other users' processes.

For an agreed future run, use the shared command shape above with an allocated GPU and a unique job/output name, and add `--save-model` if a checkpoint is needed. Do not use `--overwrite` on existing evidence. New diagnostics do not require interrupting an otherwise valid active job.

For the matched CE run, omit the SupCon flags while keeping all other scientific settings identical. Reuse existing CE results only after the configuration checks pass. Confirm the seed plan with Sam before launching multiple jobs; Group 1's reference seeds are 42, 123, 456, 789 and 2025.

Weight 0.1 and temperature 0.1 are the supplied run settings, not proven optimal values. Any tuning uses DEV, never TEST. Keep English DEV results separate.

## 8. Compare results and finish the handover

After both runs complete, compare predictions by mention ID using the shared tool. Replace the two path placeholders with actual run directories:

```bash
python scripts/summarize_runs.py --compare PATH_TO_CE_RUN PATH_TO_SUPCON_RUN --set hipe_fr_dev
python scripts/summarize_runs.py --compare PATH_TO_CE_RUN PATH_TO_SUPCON_RUN --set hipe_de_dev
```

Report per-language Macro-F1, class metrics, and paired differences. For multiple seeds, pair the same seeds and report variation, not only the best run. DEV-selected results are development evidence; reserve the final TEST comparison until the protocol is fixed.

Clemens's first deliverable is complete when:

- [ ] Sam's run artifacts and matching CE configurations are located.
- [ ] Any configuration mismatch or unknown model revision is documented.
- [ ] Useful tests/logging are proposed in one small PR targeting supcon.
- [ ] CPU checks pass and their limits are stated; GPU validation is recorded separately.
- [ ] The team agrees which runs remain necessary, avoiding duplicate GPU jobs.
- [ ] Results/checkpoint locations and the next owner are documented without publishing raw data.
