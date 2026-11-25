#!/usr/bin/env python3
"""
Aggregate per-run JSON results into paper tables (mean ± std over seeds).

    python experiments/aggregate.py                 # all tables -> results/tables/
    python experiments/aggregate.py --which train   # only training runs

Inputs
    results/runs/*.json           written by main_meta_v2.py (one per run / per LOOCV fold)
    results/ood/*.json            written by eval_phytmo_ood.py --json_output
    results/realtime_eval/*.json  written by eval_realtime.py --json_output

Outputs (CSV + LaTeX rows) in results/tables/.
Every run is checked for Dice >= IoU; a violation aborts with the file name,
because that is exactly the inconsistency Reviewer 3 found in the submission.
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

METRICS = {"Dice": "test/Dice Score", "IoU": "test/IoU", "ROC-AUC": "test/ROC-AUC"}


def fmt(m, s, nd=3):
    return f"{m:.{nd}f} $\\pm$ {s:.{nd}f}"


def _check_dice_iou(d, i, where):
    if np.isfinite(d) and np.isfinite(i) and d < i - 1e-9:
        sys.exit(f"[aggregate] Dice ({d:.4f}) < IoU ({i:.4f}) in {where}: metric bug, refusing to tabulate")


# ------------------------------------------------------------------ train
def load_runs(root):
    rows = []
    for f in sorted(glob.glob(os.path.join(root, "results/runs/*.json"))):
        j = json.load(open(f))
        res = j.get("best") or {}
        if not res:
            continue
        m = re.match(r"^(.*)-subject-(.+)$", j["run_name"])
        fold = m.group(2) if m else None
        row = {"model": j["model"], "dataset": j["dataset"], "seed": j["seed"],
               "tag": j.get("tag", ""), "fold": fold, "loocv": fold is not None,
               "file": os.path.basename(f), "gpu": j.get("gpu"), "elapsed_min": (j.get("elapsed_sec") or 0) / 60}
        for k, key in METRICS.items():
            row[k] = res.get(key, np.nan)
        _check_dice_iou(row["Dice"], row["IoU"], f)
        rows.append(row)
    return pd.DataFrame(rows)


def table_train(df, loocv, out_dir):
    d = df[df.loocv == loocv].copy()
    if d.empty:
        return
    # standard: mean/std over seeds. LOOCV: the paper uses one seed and reports
    # mean/std over the held-out subject folds, so folds are the samples there.
    g = d.groupby(["dataset", "model", "tag"])
    agg = g[list(METRICS)].agg(["mean", "std", "count"])
    name = "table_loocv" if loocv else "table_standard"
    agg.to_csv(os.path.join(out_dir, name + ".csv"))
    lines = []
    for (ds, model, tag), r in agg.iterrows():
        label = model + (f"-{tag}" if tag else "")
        cells = " & ".join(fmt(r[(k, "mean")], r[(k, "std")]) for k in METRICS)
        lines.append(f"{ds} & {label} & {cells} \\\\  % n={int(r[('Dice', 'count')])}")
    open(os.path.join(out_dir, name + ".tex"), "w").write("\n".join(lines) + "\n")
    print(f"\n== {name} ==\n" + "\n".join(lines))


# -------------------------------------------------------------------- ood
def table_ood(root, out_dir):
    rows = []
    for f in sorted(glob.glob(os.path.join(root, "results/ood/*.json"))):
        j = json.load(open(f))
        for s in j.get("per_seed", []):
            _check_dice_iou(s.get("test/Dice Score", np.nan), s.get("test/IoU", np.nan), f)
        variant, requested = os.path.basename(f)[:-5].rsplit("_", 1)
        # label by the requested setting (file name); the JSON's "setting" says
        # what actually ran (e.g. a model without BatchNorm cannot use AdaBN)
        rows.append({"variant": variant, "setting": requested, "actual": j.get("setting"),
                     "n": j["n_evaluated"],
                     "Dice": fmt(j["dice_mean"], j["dice_std"]), "IoU": fmt(j["iou_mean"], j["iou_std"]),
                     "ROC-AUC": fmt(j["rocauc_mean"], j["rocauc_std"]),
                     "dice_mean": j["dice_mean"], "iou_mean": j["iou_mean"], "rocauc_mean": j["rocauc_mean"]})
    if not rows:
        return
    df = pd.DataFrame(rows).sort_values(["variant", "setting"])
    df.to_csv(os.path.join(out_dir, "table_ood.csv"), index=False)
    lines = [f"{r.variant} & {r.setting} & {r.Dice} & {r.IoU} & {r['ROC-AUC']} \\\\  % n={r.n} ran={r.actual}"
             for _, r in df.iterrows()]
    open(os.path.join(out_dir, "table_ood.tex"), "w").write("\n".join(lines) + "\n")
    print("\n== table_ood (variant, setting, Dice, IoU, ROC-AUC) ==\n" + "\n".join(lines))


# --------------------------------------------------------------- realtime
def _flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "/"))
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[key] = v
    return out


def table_realtime(root, out_dir):
    rows = []
    for f in sorted(glob.glob(os.path.join(root, "results/realtime_eval/*.json"))):
        j = json.load(open(f))
        m = re.match(r"(.+)_seed(\d+)\.json", os.path.basename(f))
        row = {"model": m.group(1) if m else os.path.basename(f), "seed": int(m.group(2)) if m else -1}
        row.update(_flatten(j))
        rows.append(row)
    if not rows:
        return
    df = pd.DataFrame(rows)
    num = [c for c in df.columns if c not in ("model", "seed")]
    agg = df.groupby("model")[num].agg(["mean", "std"])
    agg.to_csv(os.path.join(out_dir, "table_realtime.csv"))
    # the columns Table "Real-Time Deployment Benchmarks" uses (+ streaming health)
    paper_cols = {
        "memory/model_size_mb": "Size (MB)",
        "compute/params": "Params",
        "compute/macs": "MACs",
        "latency/batch_1/total_ms/mean": "Latency ms (B=1, end-to-end)",
        "latency/batch_1/inference_ms/mean": "Latency ms (B=1, forward only)",
        "streaming/drop_rate_pct": "Stream drop %",
    }
    want = [c for c in paper_cols if c in num]
    lines = []
    for model, r in agg.iterrows():
        cells = []
        for c in want:
            m, s = r[(c, "mean")], r[(c, "std")]
            cells.append(f"{m/1e3:.1f}K" if c == "compute/params" else
                         f"{m/1e6:.2f}M" if c == "compute/macs" else
                         fmt(m, 0 if np.isnan(s) else s, 2))
        lines.append(f"{model} & " + " & ".join(cells) + " \\\\")
    open(os.path.join(out_dir, "table_realtime.tex"), "w").write(
        "% " + " & ".join(paper_cols[c] for c in want) + "\n" + "\n".join(lines) + "\n")
    print("\n== table_realtime (" + ", ".join(paper_cols[c] for c in want) + ") ==\n" + "\n".join(lines))
    print("   full per-metric mean/std for all 300+ keys: results/tables/table_realtime.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("--which", choices=["train", "ood", "realtime", "all"], default="all")
    a = ap.parse_args()
    out_dir = os.path.join(a.root, "results/tables")
    os.makedirs(out_dir, exist_ok=True)
    if a.which in ("train", "all"):
        df = load_runs(a.root)
        if df.empty:
            print("no results/runs/*.json yet")
        else:
            df.to_csv(os.path.join(out_dir, "all_runs.csv"), index=False)
            table_train(df, loocv=False, out_dir=out_dir)
            table_train(df, loocv=True, out_dir=out_dir)
    if a.which in ("ood", "all"):
        table_ood(a.root, out_dir)
    if a.which in ("realtime", "all"):
        table_realtime(a.root, out_dir)
    print(f"\nwritten to {out_dir}")


if __name__ == "__main__":
    main()
