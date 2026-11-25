# Revision experiments (TCPS major revision)

Everything needed to re-run the paper's tables on a rented multi-GPU box
and turn the outputs into LaTeX rows, without copying numbers from the W&B UI.

```
experiments/
  make_jobs.py        builds JSON job lists (train / ood / realtime phases)
  runner.py           GPU job queue: N independent runs per GPU, resumable, logs/ + _status.json
  aggregate.py        results/*.json -> results/tables/*.csv|.tex (asserts Dice >= IoU)
  runpod_setup.sh     pod bootstrap: deps, data, W&B, pregenerate episodes, launch runner in tmux
  pack_data.sh        tar PhysiQ (not in git) for upload
  requirements-min.txt
```

## What changed in the training code (TCPS revision)

* `main_meta_v2.py` writes `results/runs/<run_name>.json` (best + last metrics, args,
  host/GPU, git commit) for every run and every LOOCV fold.
* `--tag` is appended to the run name (W&B run, artifact, checkpoint, JSON), so
  ablation variants no longer overwrite each other. `--wandb_entity` added.
* Checkpoints are kept on clean exit. Previously an `atexit` hook deleted
  `saved_model/*.pth` on **every** exit; with `--nowandb` all weights were lost.
* `--eval_interval N` / `--n_test_tasks M`: meta-test every N epochs on M batches
  (the last epoch always runs). AdaBN testing was ~75% of wall-clock.
* `type=bool` argparse flags replaced by `str2bool` (`--caex_use_adabn false` now
  actually disables it; before, any value meant True).
* `eval_phytmo_ood.py --json_output` records per-seed metrics and the adaptation
  setting (`zeroshot`, `adabn`, `inner`, `both`).
* W&B group id no longer depends on `wandb.util.generate_id` (removed upstream).
* `DenseLabelTaskSampler` redraws the (label, variation) pair instead of raising
  when a small LOOCV test fold has too few windows for it (bounded at 50 draws
  per task). Before, a random draw could abort a multi-hour LOOCV run.
* `eval_phytmo_ood.py` and `eval_realtime.py` restore the architecture flags a
  checkpoint was trained with from `results/runs/<run_name>.json`
  (`argtypes.restore_train_args`) and load weights strictly. Both scripts had
  their own argparse defaults that differed from training (Segmenter: 6 layers /
  256 dims vs 2 / 128), and the real-time script loaded with `strict=False`,
  so a mismatched model could silently be benchmarked with random layers.
* `eval_phytmo_ood.py` exits non-zero when no seed could be evaluated.
* `eval_realtime.py` stripped the checkpoint prefix with `k.replace('net.', '')`,
  which also rewrote `conv_net.0.weight` to `conv_0.weight`; together with
  `strict=False` this meant the **CNN was benchmarked with random weights**.
  Latency/size/MACs are unaffected by weights, but any accuracy numbers from the
  streaming/voting evaluation of the CNN in the submission are suspect. Fixed
  (prefix-only strip, strict loading). Output directories are now created.

## Workflow

```bash
# 0. locally: pack PhysiQ, upload it to the pod (or host it and set PHYSIQ_URL)
bash experiments/pack_data.sh ../EXACT/data/physiq

# 1. on the pod (template env: WANDB_API_KEY, PHYSIQ_URL, PHASE=train, PER_GPU=3)
bash experiments/runpod_setup.sh            # clones, installs, pregenerates, launches tmux 'exact'

# 2. watch
tmux attach -t exact
python -c "import json,collections;print(collections.Counter(v['state'] for v in json.load(open('logs/_status.json')).values()))"

# 3. next phases (weights come from saved_model/ written by phase 1)
PHASE=ood      bash experiments/runpod_setup.sh
PHASE=realtime bash experiments/runpod_setup.sh

# 4. tables
python experiments/aggregate.py             # -> results/tables/
```

`results/`, `saved_model/` and `logs/` live on the pod's persistent volume.
Copy them back with `runpodctl send` or `rsync`; the JSONs are small.

## Smoke test (any machine with one GPU)

```bash
python experiments/make_jobs.py --smoke
python experiments/runner.py experiments/jobs/train_smoke.json --per_gpu 3
python experiments/aggregate.py
```

## Sizing

Per run on a 4090 with the defaults in `make_jobs.py` (200 epochs, 100 train
batches/epoch, test every 5 epochs on 25 batches): roughly 2.5 h for `caex`,
1.5 h for baselines, LOOCV x5 folds. Runs are launch-bound, not GPU-bound, so
3 per GPU is safe (about 1.5 GB VRAM each); 4 x 4090 with `PER_GPU=3` runs 12
at once. The full `train` phase is ~250 runs, so budget about two days of wall
clock on four 4090s, or one day on eight.

## Before a full launch, verify in `make_jobs.py`

* `DATASET_FLAGS` / `COMMON` against Section 5.1 of the paper (window sizes, shots).
* Whether Table 2 (LOOCV) used the same shots/steps as the standard runs.
* `N_EPOCHS`, `SEEDS`.

## Speed optimizations (all verified equivalent against the pre-patch code)

Profiled on the RTX 4090, one epoch = 100 train batches of 8 tasks.

| Change | Where | Effect |
|---|---|---|
| AdaBN: one forward pass with momentum `1-(1-m)^steps` instead of `steps` identical passes (exact: BN in train mode uses batch stats, so every pass sees the same inputs; no dropout in any model) | `methods/adabn.py` | 292 ms -> 4 ms per task; `EXACT_ADABN_EXACT_LOOP=1` restores the loop |
| Vectorised group z-score (was a per-sample Python loop) | `methods/EXP.py` | max abs diff 2e-7 |
| Confusion matrix via one `bincount` on the GPU; sklearn only for the printed per-batch summary | `MetricsAccumulator.py` | identical metrics; ~4.6 s/epoch saved |
| Sampler reads labels from a per-window histogram instead of fetching (and augmenting) every candidate window | `datasets/DenseLabelTaskSampler.py` | identical draws |
| `OMP_NUM_THREADS=2` per job, 4 jobs per GPU by default | `experiments/runner.py` | 8 concurrent runs give ~3.3x throughput on one 4090 |

Epoch wall-clock for CAEX went from ~41 s to ~23 s (plus a one-time ~19 s dataset
build). Note: the RNG stream changed (no augmentation draws while scanning
candidates, no `randperm` in AdaBN), so runs are not bit-identical to the old
code; they are statistically equivalent.

## What the original W&B runs say about the submitted tables (checked 2026-10-31)

Project `EXACT` holds the paper's runs. Recomputing from their logged metrics:

* LOCV (Table 2) for every dataset = seed 42 x 5 subject folds; the +- is over folds.
  Standard training (Table 4) = seeds 42-46.
* PhysiQ LOCV CNN / U-Net / EXACT rows reproduce exactly as fold-averaged
  last-epoch values. Segmenter and Transformer rows (0.132/0.181, 0.641/0.236)
  reproduce under no aggregation: those runs collapsed (last-epoch Dice 0.000 and
  0.006, ROC-AUC 0.54). Only their ROC-AUC matches the paper. These two rows have
  no provenance and must be replaced by the regenerated runs.
* PhysiQ standard rows mix aggregations: CNN = last epoch, Transformer and EXACT =
  best epoch, U-Net matches neither within 0.03. SPAR / MM-Fit rows = last epoch.
* `results/wandb_original_runs.json` has the per-run summary pulled from W&B.

### Metric definition changed between the submitted baseline runs and EXACT+ (found 2026-10-31)

Every run in W&B project `EXACT` created before 2025-02 (all baselines, ICCPS
`ex`, both LOCV and standard, all three datasets) logs `test/Covering Metric`
and `test/Balanced Accuracy`. Those keys only exist in the `loss_fn.py`
accumulator of the original EXACT repo at commit 2541ba63 (2024-11-06), which
computes **binary positive-class** Dice/IoU/precision/recall (exercise = 1).
Every later accumulator (EXACT repo from 2024-11-11, this repo from 2024-11-13,
`phytmo-ood-eval`, `tcps-revision`) computes the **macro average over the two
classes**. The `caex` rows (Dec 2025) and all revision runs use the macro
definition. Synthetic check (balanced labels, 8 x 500 samples):

| predictor | binary (paper baselines) Dice / IoU | macro (caex, revision) Dice / IoU |
|---|---|---|
| constant "exercise" | 0.673 / 0.507 | 0.337 / 0.254 |
| constant "rest" | 0.000 / 0.000 | 0.330 / 0.246 |
| 90 % accurate | 0.899 / 0.816 | 0.897 / 0.814 |

Consequences for the submitted tables:

* PhysiQ standard Segmenter 0.660 / 0.492 / ROC 0.591 is a collapsed model that
  predicts "exercise" everywhere (W&B: recall 1.0, precision 0.5, loss 0.694 =
  ln 2). The revision reruns collapse the same way (loss 0.69, ROC 0.57-0.64)
  and score 0.34 / 0.26 under the macro definition. This is a property of the
  Segmenter under MAML in this setup, not a logging error.
* PhysiQ LOCV Segmenter / Transformer folds collapsed to "rest" (binary Dice 0).
  The table's 0.132 / 0.181 and 0.641 / 0.236 match nothing.
* For non-collapsed models the two definitions agree within ~0.01, so the CNN /
  U-Net / Transformer / EXACT rows are comparable, and `caex` reproduces
  (0.968 / 0.939 submitted vs 0.964 / 0.931 at epoch 170 of the rerun).
* The revision reports the macro definition everywhere and should say so in the
  metrics paragraph (answers the reviewer's Dice < IoU / aggregation question).
