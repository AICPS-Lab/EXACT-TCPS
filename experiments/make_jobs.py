#!/usr/bin/env python3
"""
Generate job files for the TCPS revision experiments.

    python experiments/make_jobs.py                 # writes all phases to experiments/jobs/
    python experiments/make_jobs.py --phase ood     # one phase
    python experiments/make_jobs.py --smoke         # 1 seed, 1 epoch, few tasks: pipeline test

Phases (run them in this order; later phases load weights from earlier ones):
    train      Table 2 (LOOCV) + Table 4 (standard) baselines, EXACT+ (caex) and
               the component ablation + sensitivity variants  -> results/runs/*.json
    ood        PhysiQ -> PhyTMo 2x2 (zero-shot / AdaBN / inner-loop / both)
               for every model and every caex ablation variant  -> results/ood/*.json
    realtime   latency / memory / streaming benchmark per model  -> results/realtime_eval/*.json

Edit the CONFIG block to match the paper's setup section before launching.
"""
import argparse
import json
import os
import re

# ----------------------------------------------------------------- CONFIG
SEEDS = [42, 43, 44, 45, 46]
# The submitted paper's LOCV tables use ONE seed (42) x 5 subject folds for every
# dataset (verified from the W&B runs); the +- there is over folds. Keep that.
LOOCV_SEEDS = [42]
N_EPOCHS = 200
BASELINES = ["segmenter", "cnn", "transformer", "unet"]
OURS = "caex"               # EXACT+ in the paper; "ex" is the ICCPS EXACT
DATASETS = ["physiq", "spar", "mmfit"]

# Per-dataset flags. VERIFY against 5.1_setup.tex before a full run.
DATASET_FLAGS = {
    "physiq": "--window_size 500 --window_step 25",
    "spar":   "--window_size 300 --window_step 25",
    "mmfit":  "--window_size 300 --window_step 25",
}
# Profiling on a 4090 (1 epoch, caex): train 0.35 s/batch, test 1.2 s/batch because
# AdaBN runs 50 forward passes per task. Testing every epoch with 100 batches is
# ~75% of wall-clock, so meta-test every 5 epochs on 25 batches (last epoch always runs).
# Pre-generated episodes store full tensors (~19 GB per dataset/seed for 200
# epochs, loaded into RAM); only worth it on a big-RAM pod.
USE_PREGENERATED = False
EVAL = "--eval_interval 5 --n_test_tasks 25"
# Separate W&B project so run/artifact names do not collide with the submitted
# paper's artifacts in project "EXACT" (caex-physiq-42:v1 etc.).
WANDB = "--wandb_project EXACT-TCPS-rev"
COMMON = f"--add_side_noise --n_shot 5 --n_query 5 {EVAL} {WANDB}"

# Component ablation of the journal additions (all on top of EX, PhysiQ).
# tag -> extra flags. Flags not listed keep the CAEX defaults.
ABLATION = {
    # leave-one-out
    "full":          "",
    "no_gate":       "--caex_gate none",
    "no_groupgate":  "--caex_group_gate false",
    "no_norm":       "--caex_imu_normalize false",
    "no_adabn":      "--caex_use_adabn false",
    "no_rotaug":     "--caex_rotation_chance 0",
    # cumulative from EX
    "norm_only":     "--caex_gate none --caex_group_gate false --caex_use_adabn false",
    "norm_adabn":    "--caex_gate none --caex_group_gate false",
    "norm_adabn_se": "--caex_group_gate false",
}

# Sensitivity (Reviewer 2): tag -> flags, caex on physiq
SENSITIVITY = {
    "shot1":  "--n_shot 1 --n_query 1",
    "shot3":  "--n_shot 3 --n_query 3",
    "shot10": "--n_shot 10 --n_query 10",
    "inner3": "--n_inner_iter 3",
    "inner5": "--n_inner_iter 5",
    "win250": "--window_size 250",
    "win750": "--window_size 750",
}

OOD_SETTINGS = {
    "zeroshot": "--n_inner_iter 0 --no_adabn",
    "adabn":    "--n_inner_iter 0",
    "inner":    "--n_inner_iter 1 --no_adabn",
    "both":     "--n_inner_iter 1",
}
OOD_MODELS = BASELINES + ["ex", OURS]
# ------------------------------------------------------------------------


def run_name(model, dataset, seed, tag=""):
    n = f"{model}-{dataset}-{seed}"
    return n + (f"-{tag}" if tag else "")


def _parse_flags(s):
    """'--a 1 --b --c x' -> {'--a': '1', '--b': None, '--c': 'x'} (order kept)."""
    out, key = {}, None
    for tok in s.split():
        if tok.startswith("--"):
            key = tok
            out[key] = None
        elif key is not None:
            out[key] = tok
    return out


def _fmt_flags(d):
    return " ".join(k if v is None else f"{k} {v}" for k, v in d.items())


def train_cmd(model, dataset, seed, tag="", extra="", loocv=False, epochs=N_EPOCHS, smoke=False):
    flags = _parse_flags(f"{DATASET_FLAGS[dataset]} {COMMON} --n_epochs {epochs}")
    flags.update(_parse_flags(extra))      # variant flags override the defaults
    # Pre-generated episodes are keyed on dataset/seed/shot/query/batch/tasks
    # but NOT on window size, so a window-size variant must sample on the fly.
    if USE_PREGENERATED and "--window_size" not in _parse_flags(extra) and not loocv:
        flags["--use_pregenerated"] = None
    if loocv:
        flags["--loocv"] = None
    if tag:
        flags["--tag"] = tag
    if smoke:
        flags["--n_tasks"] = "4"
        flags["--nowandb"] = None
    return (f"python main_meta_v2.py --model {model} --dataset {dataset} --seed {seed} "
            + _fmt_flags(flags))


def phase_train(smoke):
    """Jobs ordered by priority so partial progress already fills whole table rows:
    1. standard training, all models x datasets (Table 4) - all seeds per model first
    2. ICCPS EX + CAEX component ablation on PhysiQ (new ablation table)
    3. LOOCV, all models x datasets (Table 2) - 5 folds each, the expensive ones
    4. sensitivity sweeps (Reviewer 2)
    """
    seeds = SEEDS[:1] if smoke else SEEDS
    epochs = 1 if smoke else N_EPOCHS
    datasets = DATASETS[:1] if smoke else DATASETS
    jobs = []

    def add(model, dataset, seed, tag="", extra="", loocv=False):
        name = run_name(model, dataset, seed, tag) + ("-loocv" if loocv else "")
        # main_loocv writes one JSON per fold with unknown fold names, so for
        # LOOCV jobs completion is judged by exit code only (done_if=None).
        done = None if loocv else f"results/runs/{run_name(model, dataset, seed, tag)}.json"
        jobs.append({"name": name,
                     "cmd": train_cmd(model, dataset, seed, tag, extra, loocv, epochs, smoke),
                     "done_if": done})

    # 1. standard
    for dataset in datasets:
        for model in [OURS] + BASELINES:
            for seed in seeds:
                add(model, dataset, seed)
    # 1b. three-class protocol on PhysiQ (rest / queried / other), Table "rest intervals"
    for model in [OURS] + BASELINES:
        for seed in seeds:
            add(model, "physiq", seed, tag="c3", extra="--out_channels 3 --fine_split")
    # 2. ablation (physiq)
    for seed in seeds:
        add("ex", "physiq", seed)
    for tag, extra in (list(ABLATION.items())[:2] if smoke else ABLATION.items()):
        if tag == "full":
            continue  # identical to the plain caex run above
        for seed in seeds:
            add(OURS, "physiq", seed, tag, extra)
    # 3. LOOCV (one seed x all folds, as in the submitted paper)
    for dataset in datasets:
        for model in [OURS] + BASELINES:
            for seed in (LOOCV_SEEDS if not smoke else seeds[:1]):
                add(model, dataset, seed, loocv=True)
    # 4. sensitivity
    for tag, extra in ([] if smoke else SENSITIVITY.items()):
        for seed in seeds:
            add(OURS, "physiq", seed, tag, extra)
    return jobs


def phase_ood(smoke):
    seeds = SEEDS[:1] if smoke else SEEDS
    seed_str = " ".join(map(str, seeds))
    jobs = []

    def add(model, tag, model_flags, setting, sflags):
        local = f"saved_model/{model}-physiq-{{seed}}" + (f"-{tag}" if tag else "") + ".pth"
        out = f"results/ood/{model}" + (f"-{tag}" if tag else "") + f"_{setting}.json"
        cmd = (f"python eval_phytmo_ood.py --model {model} --seeds {seed_str} "
               f"--local_path '{local}' {sflags} {model_flags} --json_output {out}")
        if smoke:
            cmd += " --n_tasks 4"
        jobs.append({"name": os.path.basename(out)[:-5], "cmd": cmd, "done_if": out})

    for model in OOD_MODELS:
        for setting, sflags in OOD_SETTINGS.items():
            # Only caex has AdaBN; for the others "adabn" == "zeroshot" and
            # "both" == "inner", so skip the duplicate experiments.
            if model != OURS and setting in ("adabn", "both"):
                continue
            add(model, "", "", setting, sflags)
    for tag, extra in ABLATION.items():
        if tag == "full":
            continue
        # evaluate each ablation variant in the 'both' setting (and zero-shot)
        for setting in ("both", "zeroshot"):
            flags = extra.replace("--caex_use_adabn false", "")  # eval uses --no_adabn
            # training-only augmentation flags are unknown to the eval parser
            flags = re.sub(r"--caex_rotation_\w+ \S+", "", flags)
            sflags = OOD_SETTINGS[setting]
            if "no_adabn" in tag and "--no_adabn" not in sflags:
                sflags += " --no_adabn"
            add(OURS, tag, flags, setting, sflags)
    return jobs


def phase_realtime(smoke):
    seeds = SEEDS[:1] if smoke else SEEDS
    jobs = []
    for model in OOD_MODELS:
        for seed in seeds:
            out = f"results/realtime_eval/{model}_seed{seed}.json"
            cmd = (f"python eval_realtime.py --model {model} "
                   f"--model_path saved_model/{model}-physiq-{seed}.pth --mode full "
                   f"--device cuda --out_channels 2 --stream_duration {5 if smoke else 20} "
                   # PhysiQ deployment protocol (10 s window at 50 Hz, 0.5 s stride), same as the on-device
                   # benchmark (experiments/edge). The script's own defaults are 300 / 50 / 100 Hz, which is what the
                   # first regenerated Table 8 was run with; rerun this phase to replace it.
                   f"--window_size 500 --window_step 25 --sensor_hz 50 "
                   f"--latency_iterations {20 if smoke else 300} --seed {seed} "
                   f"--json_output {out} --output results/realtime_eval/{model}_seed{seed}.md")
            jobs.append({"name": f"realtime-{model}-{seed}", "cmd": cmd, "done_if": out})
    return jobs


PHASES = {"train": phase_train, "ood": phase_ood, "realtime": phase_realtime}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=list(PHASES), default=None)
    ap.add_argument("--smoke", action="store_true", help="tiny config to test the pipeline")
    ap.add_argument("--out_dir", default=os.path.join(os.path.dirname(__file__), "jobs"))
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    for name, fn in PHASES.items():
        if a.phase and name != a.phase:
            continue
        jobs = fn(a.smoke)
        path = os.path.join(a.out_dir, f"{name}{'_smoke' if a.smoke else ''}.json")
        with open(path, "w") as f:
            json.dump(jobs, f, indent=1)
        print(f"{path}: {len(jobs)} jobs")


if __name__ == "__main__":
    main()
