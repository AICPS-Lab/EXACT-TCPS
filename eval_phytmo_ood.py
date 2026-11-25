"""
Out-of-Distribution (OOD) Evaluation Script for PhyTMo Dataset

This script loads models trained on PhysiQ from W&B artifacts and evaluates them
on the PhyTMo dataset for OOD generalization analysis.

Note: The DenseLabelTaskSampler randomly selects a class of interest per task
and binarizes labels (0=not target, 1=target exercise). This matches the
training protocol used for PhysiQ models.

Usage:
    # Load from W&B artifacts (unet-physiq-[42,43,44,45,46]:v0)
    python eval_phytmo_ood.py --model unet --seeds 42 43 44 45 46

    # Specify W&B entity and project
    python eval_phytmo_ood.py --model unet --wandb_entity your_entity --wandb_project EXACT
    
    # python eval_phytmo_ood.py --model caex --seeds 42 --artifact_version v1
    # 0.869 ± 0.009 | 0.773 ± 0.014 | 0.946 ± 0.008
    
    # python eval_phytmo_ood.py --model ex --seeds 43 44 45 46
    # 0.846 ± 0.0502 | 0.7456 ± 0.0669 | 0.904 ± 0.0320
    
    # python eval_phytmo_ood.py --model unet --seeds 43 44 45 46
    # 0.801 ± 0.0020 | 0.8273 ± 0.0061 | 0.886 ± 0.0098
    
    # python eval_phytmo_ood.py --model segmenter --seeds 43 44 45 46
    # 0.406 ± 0.0484 | 0.3365 ± 0.0819 | 0.574 ± 0.0431
    
    # python eval_phytmo_ood.py --model cnn --seeds 43 44 45 46
    # 0.873 ± 0.0049 | 0.7837 ± 0.0071 | 0.869 ± 0.0138
    
    # python eval_phytmo_ood.py --model transformer --seeds 43 44 45 46
    # 0.546 ± 0.0121 | 0.4112 ± 0.0140 | 0.571 ± 0.0097
    
    # Custom artifact version
    python eval_phytmo_ood.py --model unet --artifact_version latest

    # Load from local files
    python eval_phytmo_ood.py --model unet --local_path "saved_model/unet-physiq-{seed}.pth"

Example trained with:
    python main_meta_v2.py --add_side_noise --dataset physiq --model unet --seed 46 --n_epochs 200

    This creates artifact: unet-physiq-46:v0
"""

from argtypes import str2bool, restore_train_args
import argparse
import os
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import wandb
import higher

from datasets.PHYTMO import PHYTMO
from datasets.DenseLabelTaskSampler import DenseLabelTaskSampler
from MetricsAccumulator import MetricsAccumulator
from methods.adabn import adabn_adapt
from until_argparser import get_model
from utilities import seed


def is_exp_model(model):
    """Check if the model supports AdaBN (EXP, CAEX, or any model with use_adabn=True)."""
    # Check wrapped model (model.net)
    if hasattr(model, 'net'):
        net = model.net
        if hasattr(net, 'use_adabn'):
            return net.use_adabn
        # Also check for CAEX which stores it differently
        if hasattr(net, 'adabn_steps') and net.adabn_steps > 0:
            return True
    # Check direct model
    if hasattr(model, 'use_adabn'):
        return model.use_adabn
    if hasattr(model, 'adabn_steps') and model.adabn_steps > 0:
        return True
    return False


def get_adabn_steps(model):
    """Get the number of AdaBN steps from the model."""
    if hasattr(model, 'net'):
        if hasattr(model.net, 'adabn_steps'):
            return model.net.adabn_steps
    if hasattr(model, 'adabn_steps'):
        return model.adabn_steps
    return 50  # Default


def get_phytmo_dataset(args):
    """
    Create PhyTMo train and test datasets.

    Note: For OOD evaluation, we typically use the test split of PhyTMo
    since the model was never trained on any PhyTMo data.
    """
    train_dataset = PHYTMO(
        root=args.data_root,
        split="train",
        window_size=args.window_size,
        window_step=args.window_step,
        bg_fg=None,
        args=args,
        transforms=None,  # No augmentation for evaluation
        test_subject=None,
    )
    test_dataset = PHYTMO(
        root=args.data_root,
        split="test",
        window_size=args.window_size,
        window_step=args.window_step,
        bg_fg=None,
        args=args,
        test_subject=None,
    )
    return train_dataset, test_dataset


def load_model_from_wandb(args, run_name, seed_val):
    """
    Load a model from W&B artifacts.

    Args:
        args: Argument namespace with model configuration
        run_name: Base W&B run name (e.g., 'unet-physiq')
        seed_val: Seed value to construct full run name

    Returns:
        Loaded model with weights from W&B
    """
    # Construct the full artifact name
    full_run_name = f"{run_name}-{seed_val}"
    artifact_version = args.artifact_version  # e.g., 'v0', 'latest'

    # Initialize W&B API
    api = wandb.Api()

    # Try to download artifact from W&B
    model_path = None
    artifact_paths_tried = []

    # Build artifact path - handle with and without entity
    if args.wandb_entity:
        artifact_path = f"{args.wandb_entity}/{args.wandb_project}/{full_run_name}:{artifact_version}"
    else:
        artifact_path = f"{args.wandb_project}/{full_run_name}:{artifact_version}"

    artifact_paths_tried.append(artifact_path)

    try:
        print(f"Trying to download artifact: {artifact_path}")
        artifact = api.artifact(artifact_path, type="model")
        artifact_dir = artifact.download()

        # Find the .pth file in the artifact directory
        pth_files = [f for f in os.listdir(artifact_dir) if f.endswith('.pth')]
        if pth_files:
            model_path = os.path.join(artifact_dir, pth_files[0])
            print(f"Downloaded model from W&B: {model_path}")
        else:
            # Try with expected name
            model_path = os.path.join(artifact_dir, f"{full_run_name}.pth")
            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"No .pth file found in artifact: {artifact_dir}"
                )
            print(f"Downloaded model from W&B: {model_path}")

    except Exception as e:
        print(f"W&B artifact download failed: {e}")
        print("Falling back to local paths...")

        # Fallback to local saved_model directory
        local_paths = [
            f"saved_model/wandb_artifacts/{args.model}_physiq/{full_run_name}.pth",
            f"saved_model/{full_run_name}.pth",
            f"saved_model/wandb_artifacts/{args.model}_{args.train_dataset}/{full_run_name}.pth",
        ]

        for local_path in local_paths:
            if os.path.exists(local_path):
                model_path = local_path
                print(f"Using local model: {model_path}")
                break

    if model_path is None or not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found. Tried artifact: {artifact_paths_tried}, "
            f"Local paths checked."
        )

    # Create model and load weights
    model = get_model(args)
    model.load_state_dict(
        torch.load(model_path, map_location=args.device, weights_only=True)
    )
    model.to(args.device)

    return model


def load_model_from_local(args, model_path):
    """
    Load a model from a local file path.

    Args:
        args: Argument namespace with model configuration
        model_path: Path to the model weights file

    Returns:
        Loaded model with weights
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    restore_train_args(args, model_path)
    model = get_model(args)
    model.load_state_dict(torch.load(model_path, map_location=args.device), strict=True)
    model.to(args.device)

    return model


def evaluate_on_phytmo(model, test_loader, args):
    """
    Evaluate a model on the PhyTMo dataset using meta-learning protocol.

    The DenseLabelTaskSampler handles label binarization:
    - Randomly selects an exercise class of interest
    - Converts labels to binary (0=not target, 1=target exercise)

    This matches how PhysiQ models were trained.

    Args:
        model: The model to evaluate
        test_loader: DataLoader for PhyTMo test set
        args: Argument namespace

    Returns:
        Dictionary with evaluation metrics
    """
    model.to(args.device)
    model.train(
    )  # Keep in train mode for batch norm behavior during adaptation

    # Check if AdaBN should be used:
    # 1. Model has built-in AdaBN (EXP, CAEX) - auto-enabled unless --no_adabn
    # 2. User explicitly requests it with --use_adabn (works on any model with BN)
    model_has_adabn = is_exp_model(model)
    user_requested_adabn = getattr(args, 'use_adabn', False)
    user_disabled_adabn = getattr(args, 'no_adabn', False)

    use_adabn = (
        model_has_adabn or user_requested_adabn
    ) and not user_disabled_adabn

    # Get AdaBN steps: from model if available, otherwise from args
    if model_has_adabn:
        adabn_steps = get_adabn_steps(model)
    else:
        adabn_steps = getattr(args, 'adabn_steps', 50)

    if use_adabn:
        print(f"  Using AdaBN with {adabn_steps} steps")
    elif model_has_adabn and user_disabled_adabn:
        print(f"  AdaBN disabled (--no_adabn flag)")

    compute_metrics = MetricsAccumulator(
        dir_name="test", n_classes=args.out_channels
    )
    qry_losses = []

    db = iter(test_loader)

    for batch_idx in range(args.n_tasks):
        try:
            x_spt, y_spt, x_qry, y_qry, _ = next(db)
        except StopIteration:
            db = iter(test_loader)
            x_spt, y_spt, x_qry, y_qry, _ = next(db)

        # Move data to device
        # Labels are already binarized by DenseLabelTaskSampler.episodic_collate_fn
        x_spt, y_spt = x_spt.to(args.device), y_spt.to(args.device)
        x_qry, y_qry = x_qry.to(args.device), y_qry.to(args.device)

        task_num = x_spt.size(0)

        # Initialize inner optimizer for adaptation
        inner_opt = torch.optim.SGD(model.parameters(), lr=args.meta_lr)

        for i in range(task_num):
            # AdaBN: Adapt BatchNorm stats to support set before inner-loop
            if use_adabn:
                adabn_adapt(
                    model, x_spt[i], steps=adabn_steps, reset_stats=True
                )
                # After AdaBN, set to eval mode so BN uses the adapted running stats
                # (in train mode, BN uses batch stats which ignores the adaptation)
                model.eval()

            with higher.innerloop_ctx(
                model, inner_opt, track_higher_grads=False
            ) as (fnet, diffopt):
                # For inner-loop, we want BN to use batch stats for adaptation
                fnet.train()

                # Inner-loop adaptation on support set
                for _ in range(args.n_inner_iter):
                    spt_logits = fnet(x_spt[i])
                    spt_loss = F.cross_entropy(spt_logits, y_spt[i].long())
                    diffopt.step(spt_loss)

                # For query evaluation, use eval mode to use adapted running stats
                if use_adabn:
                    fnet.eval()

                # Evaluate on query set
                qry_logits = fnet(x_qry[i]).detach()
                qry_loss = F.cross_entropy(
                    qry_logits, y_qry[i].long(), reduction="none"
                )
                qry_losses.append(qry_loss.detach())
                compute_metrics.update(y_qry[i].long(), qry_logits)

            # Restore train mode for next task's AdaBN
            if use_adabn:
                model.train()

    qry_losses = torch.cat(qry_losses).mean().item()
    res_dict = compute_metrics.compute()
    res_dict["test/loss"] = qry_losses

    res_dict["adabn_used"] = bool(use_adabn)
    res_dict["adabn_steps"] = int(adabn_steps) if use_adabn else 0
    res_dict["n_inner_iter"] = int(args.n_inner_iter)
    res_dict["setting"] = (
        ("adabn" if use_adabn else "noadabn") + "+" +
        (f"inner{args.n_inner_iter}" if args.n_inner_iter > 0 else "zeroshot")
    )
    return res_dict


def main():
    parser = argparse.ArgumentParser(
        description="OOD Evaluation on PhyTMo Dataset"
    )

    # Model parameters
    parser.add_argument(
        "--model",
        type=str,
        default="unet",
        help="Model architecture (unet, transformer, ex, segmenter, cnn)"
    )
    parser.add_argument(
        "--in_channels", type=int, default=6, help="Number of input channels"
    )
    parser.add_argument(
        "--out_channels",
        type=int,
        default=2,  # Must match trained model (PhysiQ uses 2: bg + exercise)
        help="Number of output channels (classes) - must match the trained model"
    )

    # Segmenter-specific arguments
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--embed_dims", type=int, default=256)
    parser.add_argument("--mlp_ratio", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.5)

    # Transformer-specific arguments
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--nhid", type=int, default=256)
    parser.add_argument("--nlayers", type=int, default=6)
    parser.add_argument("--ninp", type=int, default=64)
    parser.add_argument("--init_std", type=float, default=0.02)
    parser.add_argument("--activation", type=str, default="relu")

    # EX-specific arguments
    parser.add_argument("--embed_dim", type=int, default=32)

    # CAEX-specific arguments
    parser.add_argument(
        "--caex_gate", type=str, default="se", choices=["se", "none"]
    )
    parser.add_argument("--caex_gate_reduction", type=int, default=2)
    parser.add_argument("--caex_gate_softmax", type=str2bool, default=True)
    parser.add_argument("--caex_group_gate", type=str2bool, default=True)
    parser.add_argument("--caex_imu_normalize", type=str2bool, default=True)
    parser.add_argument(
        "--caex_imu_norm_kind", type=str, default="group_zscore"
    )
    parser.add_argument("--caex_imu_norm_skip_noise", type=str2bool, default=True)
    parser.add_argument("--caex_use_adabn", type=str2bool, default=True)
    parser.add_argument("--caex_adabn_steps", type=int, default=50)

    # CNN-specific arguments
    parser.add_argument(
        "--conv_sizes", type=int, nargs="+", default=[32, 64, 128]
    )

    # Seeds to evaluate
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44, 45, 46],
        help="Seeds to evaluate (default: 42 43 44 45 46)"
    )

    # W&B parameters
    parser.add_argument(
        "--wandb_project", type=str, default="EXACT", help="W&B project name"
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="W&B entity (username or team)"
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="Base W&B run name (e.g., 'unet-physiq')"
    )
    parser.add_argument(
        "--artifact_version",
        type=str,
        default="v0",
        help="W&B artifact version (e.g., 'v0', 'latest')"
    )
    parser.add_argument(
        "--train_dataset",
        type=str,
        default="physiq",
        help="Dataset the model was trained on (for path construction)"
    )

    # Local model path (alternative to W&B)
    parser.add_argument(
        "--local_path",
        type=str,
        default=None,
        help="Local path pattern for model files (use {seed} as placeholder)"
    )

    # Dataset parameters
    parser.add_argument(
        "--data_root",
        type=str,
        default="data",
        help="Root directory for datasets"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="phytmo",
        help="Dataset name (for compatibility)"
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=500,
        help="Window size for sliding window"
    )
    parser.add_argument(
        "--window_step",
        type=int,
        default=25,
        help="Window step for sliding window"
    )
    parser.add_argument(
        "--dataset_seed",
        type=int,
        default=42,
        help="Seed for dataset splitting"
    )

    # Meta-learning parameters
    parser.add_argument(
        "--n_shot",
        type=int,
        default=5,
        help="Number of support examples per class"
    )
    parser.add_argument(
        "--n_query",
        type=int,
        default=5,
        help="Number of query examples per class"
    )
    parser.add_argument(
        "--n_tasks",
        type=int,
        default=100,
        help="Number of tasks for evaluation"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size (tasks per batch)"
    )
    parser.add_argument(
        "--threshold_ratio",
        type=float,
        default=0.25,
        help="Threshold ratio for sampler"
    )
    parser.add_argument(
        "--meta_lr", type=float, default=1e-2, help="Inner-loop learning rate"
    )
    parser.add_argument(
        "--n_inner_iter",
        type=int,
        default=1,
        help="Number of inner-loop iterations"
    )

    # Other parameters
    parser.add_argument(
        "--add_side_noise",
        action="store_true",
        default=True,
        help="Add background class noise"
    )
    parser.add_argument(
        "--loocv",
        action="store_true",
        default=False,
        help="Leave-one-out cross-validation (not used for OOD eval)"
    )
    parser.add_argument(
        "--fine_split",
        action="store_true",
        default=False,
        help="Use fine split for 3-class segmentation"
    )
    parser.add_argument(
        "--shuffle",
        type=str,
        default="random_variation",
        help="Shuffle mode for data (random, random_variation, sorted)"
    )
    parser.add_argument(
        "--noise_type", type=str, default="all", help="Type of noise to add"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of dataloader workers"
    )
    parser.add_argument(
        "--pin_memory", action="store_true", help="Use pinned memory"
    )
    parser.add_argument(
        "--no_adabn",
        action="store_true",
        help=
        "Disable AdaBN even for models that support it (for ablation studies)"
    )
    parser.add_argument(
        "--json_output", type=str, default=None,
        help="Write a JSON summary (per-seed metrics + adaptation setting) here",
    )
    parser.add_argument(
        "--use_adabn",
        action="store_true",
        help=
        "Enable AdaBN for any model (not just EXP/CAEX). Works on any model with BatchNorm."
    )
    parser.add_argument(
        "--adabn_steps",
        type=int,
        default=50,
        help="Number of AdaBN adaptation steps (default: 50)"
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Device (cuda/cpu/mps)"
    )
    parser.add_argument(
        "--log_wandb", action="store_true", help="Log results to W&B"
    )

    args = parser.parse_args()

    # Set device
    if args.device is None:
        if torch.cuda.is_available():
            args.device = torch.device("cuda")
        elif hasattr(torch.backends,
                     "mps") and torch.backends.mps.is_available():
            args.device = torch.device("mps")
        else:
            args.device = torch.device("cpu")
    else:
        args.device = torch.device(args.device)

    print(f"Using device: {args.device}")

    # Set default W&B run name based on model
    if args.wandb_run_name is None:
        args.wandb_run_name = f"{args.model}-physiq"

    # Initialize PhyTMo dataset
    print("\nLoading PhyTMo dataset...")
    train_dataset, test_dataset = get_phytmo_dataset(args)

    # Create test sampler and loader
    test_sampler = DenseLabelTaskSampler(
        test_dataset,
        n_shot=args.n_shot,
        batch_size=args.batch_size,
        n_query=args.n_query,
        n_tasks=args.n_tasks,
        threshold_ratio=args.threshold_ratio,
        add_side_noise=args.add_side_noise,
        args=args,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_sampler=test_sampler,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        collate_fn=test_sampler.episodic_collate_fn,
    )

    print(f"Test dataset size: {len(test_dataset)}")
    print(f"Number of classes: {args.out_channels}")

    # Collect results
    results = {
        "iou": [],
        "dice": [],
        "rocauc": [],
        "loss": [],
    }

    # Evaluate each seed
    for seed_val in args.seeds:
        print(f"\n{'='*50}")
        print(f"Evaluating seed: {seed_val}")
        print(f"{'='*50}")

        # Set seed for reproducibility
        seed(seed_val)
        args.seed = seed_val

        try:
            # Load model
            if args.local_path:
                model_path = args.local_path.format(seed=seed_val)
                model = load_model_from_local(args, model_path)
            else:
                model = load_model_from_wandb(
                    args, args.wandb_run_name, seed_val
                )

            # Evaluate
            res = evaluate_on_phytmo(model, test_loader, args)

            # Print results
            print(f"  IoU: {res['test/IoU']:.4f}")
            print(f"  Dice: {res['test/Dice Score']:.4f}")
            print(f"  ROC-AUC: {res['test/ROC-AUC']:.4f}")
            print(f"  Loss: {res['test/loss']:.4f}")

            # Collect results
            results["iou"].append(res["test/IoU"])
            results["dice"].append(res["test/Dice Score"])
            results["rocauc"].append(res["test/ROC-AUC"])
            results["loss"].append(res["test/loss"])
            results.setdefault("per_seed", []).append(
                {"seed": seed_val, **{k: v for k, v in res.items()
                                      if isinstance(v, (bool, int, float, str))}}
            )

        except FileNotFoundError as e:
            print(f"  Skipping seed {seed_val}: {e}")
            continue
        except Exception as e:
            print(f"  Error evaluating seed {seed_val}: {e}")
            continue

    # Print summary
    if len(results["iou"]) > 0:
        print(f"\n{'='*50}")
        print("SUMMARY - OOD Evaluation on PhyTMo")
        print(f"{'='*50}")
        print(f"Model: {args.model}")
        print(f"Trained on: PhysiQ")
        print(f"Evaluated on: PhyTMo (OOD)")
        print(f"Seeds evaluated: {len(results['iou'])}/{len(args.seeds)}")
        print()
        print(
            f"Dice Score: {np.mean(results['dice']):.4f} ± {np.std(results['dice']):.4f}"
        )
        print(
            f"IoU:        {np.mean(results['iou']):.4f} ± {np.std(results['iou']):.4f}"
        )
        print(
            f"ROC-AUC:    {np.mean(results['rocauc']):.4f} ± {np.std(results['rocauc']):.4f}"
        )
        print(
            f"Loss:       {np.mean(results['loss']):.4f} ± {np.std(results['loss']):.4f}"
        )

        # One-line summary (matching test.py format)
        print()
        print("One-line summary (Dice | IoU | ROC-AUC):")
        print(
            f"{args.model} (PhyTMo OOD): "
            f"{np.mean(results['dice']):.3f} ± {np.std(results['dice']):.4f} | "
            f"{np.mean(results['iou']):.4f} ± {np.std(results['iou']):.4f} | "
            f"{np.mean(results['rocauc']):.3f} ± {np.std(results['rocauc']):.4f}"
        )

        if getattr(args, "json_output", None):
            os.makedirs(os.path.dirname(args.json_output) or ".", exist_ok=True)
            summary = {
                "model": args.model,
                "train_dataset": getattr(args, "train_dataset", "physiq"),
                "test_dataset": "phytmo",
                "setting": results["per_seed"][0].get("setting") if results.get("per_seed") else None,
                "n_inner_iter": int(args.n_inner_iter),
                "no_adabn": bool(getattr(args, "no_adabn", False)),
                "seeds": list(args.seeds),
                "n_evaluated": len(results["iou"]),
                "dice_mean": float(np.mean(results["dice"])), "dice_std": float(np.std(results["dice"])),
                "iou_mean": float(np.mean(results["iou"])), "iou_std": float(np.std(results["iou"])),
                "rocauc_mean": float(np.mean(results["rocauc"])), "rocauc_std": float(np.std(results["rocauc"])),
                "loss_mean": float(np.mean(results["loss"])), "loss_std": float(np.std(results["loss"])),
                "per_seed": results.get("per_seed", []),
                "args": {k: (v if isinstance(v, (bool, int, float, str, list)) or v is None else str(v))
                         for k, v in vars(args).items()},
            }
            with open(args.json_output, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"JSON results saved to: {args.json_output}")

        # Log to W&B if requested
        if args.log_wandb:
            run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=f"{args.model}-phytmo-ood-eval",
                config=vars(args),
            )
            run.log(
                {
                    "ood/dice_mean": np.mean(results["dice"]),
                    "ood/dice_std": np.std(results["dice"]),
                    "ood/iou_mean": np.mean(results["iou"]),
                    "ood/iou_std": np.std(results["iou"]),
                    "ood/rocauc_mean": np.mean(results["rocauc"]),
                    "ood/rocauc_std": np.std(results["rocauc"]),
                    "ood/loss_mean": np.mean(results["loss"]),
                    "ood/loss_std": np.std(results["loss"]),
                }
            )
            run.finish()
    else:
        print("\nNo models were successfully evaluated.")
        sys.exit(1)

    return results


if __name__ == "__main__":
    main()
