"""Shared argparse helpers.

`type=bool` in argparse is a trap: bool("False") is True, so a flag like
`--caex_use_adabn False` silently stays enabled. Use `type=str2bool` instead.
"""
import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("yes", "true", "t", "y", "1", "on"):
        return True
    if s in ("no", "false", "f", "n", "0", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got {v!r}")


# Keys that describe the evaluation setting, never the trained architecture.
_NON_ARCH_KEYS = {
    "device", "seed", "seeds", "nowandb", "wandb_project", "wandb_entity", "wandb_group",
    "wandb_run_name", "artifact_version", "local_path", "json_output", "log_wandb",
    "model_path", "output", "mode", "tag", "results_dir", "n_epochs", "lr", "meta_lr",
    "n_inner_iter", "no_adabn", "adabn_steps", "n_tasks", "n_test_tasks", "eval_interval",
    "batch_size", "n_shot", "n_query", "dataset", "train_dataset", "data_root", "loocv",
    "use_pregenerated", "pregenerated_dir", "num_workers", "pin_memory", "auto_tune",
    "multi_gpu", "mixed_precision", "log_interval", "shuffle", "dataset_seed",
    "stream_duration", "latency_iterations", "n_runs", "batch_sizes", "sensor_hz",
    "k_shot", "n_way", "meta_opt", "fomaml", "threshold_ratio", "add_side_noise",
    "noise_type", "fine_split", "window_step", "rotation_chance",
}


def restore_train_args(args, model_path, verbose=True):
    """Overlay the architecture flags a checkpoint was trained with.

    main_meta_v2.py writes results/runs/<run_name>.json next to every
    saved_model/<run_name>.pth. Evaluation scripts carry their own argparse
    defaults, which drift from the training defaults (e.g. Segmenter depth),
    so without this a checkpoint can fail to load or, worse, load partially.
    Returns the list of keys that were changed.
    """
    import json
    import os
    if not model_path:
        return []
    run_name = os.path.splitext(os.path.basename(model_path))[0]
    results_dir = getattr(args, "results_dir", None) or "results/runs"
    candidates = [os.path.join(results_dir, run_name + ".json"),
                  os.path.join(os.path.dirname(model_path) or ".", "..", "results", "runs", run_name + ".json")]
    path = next((c for c in candidates if os.path.exists(c)), None)
    if path is None:
        if verbose:
            print(f"  [restore_train_args] no results JSON for {run_name}; using eval defaults")
        return []
    saved = json.load(open(path)).get("args", {})
    changed = []
    for k, v in saved.items():
        if k in _NON_ARCH_KEYS or not hasattr(args, k):
            continue
        cur = getattr(args, k)
        if isinstance(v, str) and not isinstance(cur, str):
            continue  # stringified non-serializable object
        if cur != v:
            setattr(args, k, v)
            changed.append(f"{k}: {cur} -> {v}")
    if verbose and changed:
        print(f"  [restore_train_args] from {path}: " + ", ".join(changed))
    return changed
