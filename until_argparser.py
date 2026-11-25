import argparse

import torch.nn as nn

from datasets.MMFIT import MMFIT
from datasets.PhysiQ import PhysiQ
from datasets.SPAR import SPAR
from datasets.Transforms import IMUAugmentation
from methods.cnn import CNN
from methods.CAEX import CAEX
from methods.EX import EX
from methods.EXP import EXP
from methods.segmenter import Segmenter
from methods.transformer import TransformerModel
from methods.unet import UNet
from utilities import seed


def get_model_args(args, preliminary_args):
    model = preliminary_args.model.lower()
    if model == "unet":
        args = UNet.add_args(args)
    elif model == "transformer":
        args = TransformerModel.add_args(args)
    elif model == "caex":
        args = CAEX.add_args(args)
    elif model == "ex":
        args = EX.add_args(args)
    elif model == "exp":
        args = EXP.add_args(args)
    elif model == "segmenter":
        args = Segmenter.add_args(args)
    elif model == "cnn":
        args = CNN.add_args(args)
    return args


def get_dataset_args(args, preliminary_args):
    # without parse args to get dataset:
    dataset = preliminary_args.dataset.lower()
    if dataset == "physiq":
        args = PhysiQ.add_args(args)
    return args


def get_model(args):
    args.model = args.model.lower()
    if args.model == "unet":
        model = UNet
    elif args.model == "transformer":
        model = TransformerModel
    elif args.model == "caex":
        model = CAEX
    elif args.model == "ex":
        model = EX
    elif args.model == "exp":
        model = EXP
    elif args.model == "segmenter":
        model = Segmenter
    elif args.model == "cnn":
        model = CNN
    else:
        raise ValueError("Model not supported")

    class model_wrapper(nn.Module):

        def __init__(self):
            super(model_wrapper, self).__init__()
            self.net = model(args)

        def forward(self, x):
            x = x.float()
            x = self.net(x)
            x = x.permute(0, 2, 1)
            return x.squeeze(1)

    # Initialize model
    model = model_wrapper().float()
    return model


def _get_transforms(args):
    """Get the appropriate transforms based on model type."""
    model = args.model.lower() if hasattr(args, 'model') else ''

    # For EXP model, use its specialized augmentations
    if model == 'exp':
        return EXP.get_augmentation_transforms(args)

    # For CAEX model, use its augmentations (softer than EXP by default)
    if model == 'caex':
        return CAEX.get_augmentation_transforms(args)

    # Default: use IMUAugmentation with rotation_chance
    return IMUAugmentation(rotation_chance=args.rotation_chance)


def get_dataset(args, test_subject=None):
    dataset = args.dataset.lower()
    train_transforms = _get_transforms(args)

    if dataset == "physiq":
        train_dataset = PhysiQ(
            root=args.data_root,
            split="train",
            window_size=args.window_size,
            window_step=args.window_step,
            bg_fg=None,
            args=args,
            transforms=train_transforms,
            test_subject=test_subject,
        )
        test_dataset = PhysiQ(
            root=args.data_root,
            split="test",
            window_size=args.window_size,
            window_step=args.window_step,
            bg_fg=None,
            args=args,
            test_subject=test_subject,
        )
    elif dataset == "spar":
        train_dataset = SPAR(
            root=args.data_root,
            split="train",
            window_size=args.window_size,
            window_step=args.window_step,
            bg_fg=None,
            args=args,
            transforms=train_transforms,
            test_subject=test_subject,
        )
        test_dataset = SPAR(
            root=args.data_root,
            split="test",
            window_size=args.window_size,
            window_step=args.window_step,
            bg_fg=None,
            args=args,
            test_subject=test_subject,
        )
    elif dataset == "mmfit":
        train_dataset = MMFIT(
            root=args.data_root,
            split="train",
            window_size=args.window_size,
            window_step=args.window_step,
            bg_fg=None,
            args=args,
            transforms=train_transforms,
            test_subject=test_subject,
        )
        test_dataset = MMFIT(
            root=args.data_root,
            split="test",
            window_size=args.window_size,
            window_step=args.window_step,
            bg_fg=None,
            args=args,
            test_subject=test_subject,
        )
    else:
        raise ValueError("Dataset not supported")
    seed(args.seed)
    return train_dataset, test_dataset


def get_all_subjects(args):
    dataset = args.dataset.lower()
    # base 1 indexing
    if dataset == "physiq":
        all_subjects = 31
        return [
            [1, 2, 3, 9, 10, 25, 7],
            [3, 5, 15, 17, 24, 12],
            [6, 7, 14, 19, 31, 2, 7],
            [1, 4, 10, 26, 30, 5, 3],
            [6, 16, 19, 21, 23, 14, 26],
        ]
    elif dataset == "spar":
        all_subjects = 21
        return [[1, 3, 5], [2, 4, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]]
    elif dataset == "mmfit":
        all_subjects = 20
        return [[1, 3, 5], [2, 4, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]]
    else:
        raise ValueError("Dataset not supported")
    return all_subjects


def get_args():
    parser = argparse.ArgumentParser(
        description="Meta-learning for dense labeling tasks"
    )

    # Dataset and DataLoader parameters
    parser.add_argument(
        "--data_root",
        type=str,
        default="data",
        help="Root directory for the dataset",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="physiq",
        help="Dataset to use for training",
    )
    """Add dataset-specific arguments to the parser."""
    parser.add_argument(
        "--shuffle",
        type=str,
        default="random_variation",
        help="Shuffle the data on each subject",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for shuffling the data",
    )
    parser.add_argument(
        "--dataset_seed",
        type=int,
        default=42,
        help="Seed for shuffling the data",
    )
    parser.add_argument(
        "--add_side_noise",
        help="Add random noise to the data on each side (default: True)",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no_side_noise",
        help="Disable adding random noise to the data",
        action="store_true",
    )
    parser.add_argument(
        "--noise_type",
        type=str,
        default="all",
        help="Type of noise to add to the data",
    )

    parser.add_argument(
        "--in_channels",
        type=int,
        default=6,
        help="Input channels for the model",
    )
    parser.add_argument(
        "--out_channels",
        type=int,
        default=2,
        help="Output channels for the model",
    )

    parser.add_argument(
        "--window_size",
        type=int,
        default=500,
        help="Window size for the PhysiQ dataset",
    )
    parser.add_argument(
        "--window_step",
        type=int,
        default=25,
        help="Window step for the PhysiQ dataset",
    )
    parser.add_argument(
        "--n_shot",
        type=int,
        default=5,
        help="Number of support examples per class",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for training"
    )
    parser.add_argument(
        "--n_query",
        type=int,
        default=5,
        help="Number of query examples per class",
    )
    parser.add_argument(
        "--n_tasks",
        type=int,
        default=100,
        help="Number of tasks for training"
    )
    parser.add_argument(
        "--threshold_ratio",
        type=float,
        default=0.25,
        help="Threshold ratio for DenseLabelTaskSampler",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of workers for DataLoader",
    )
    parser.add_argument(
        "--pin_memory",
        action="store_true",
        help="Use pinned memory for DataLoader",
    )
    parser.add_argument(
        "--rotation_chance",
        type=float,
        default=0,
        help="Chance of rotating the data",
    )
    parser.add_argument(
        "--loocv",
        action="store_true",
        help="Leave-one-out cross-validation",
    )
    # Training parameters
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for the meta optimizer",
    )
    parser.add_argument(
        "--meta_lr",
        type=float,
        default=1e-2,
        help="Learning rate for the inner optimizer",
    )
    parser.add_argument(
        "--n_inner_iter",
        type=int,
        default=1,
        help="Number of inner-loop iterations",
    )
    parser.add_argument(
        "--n_epochs", type=int, default=30, help="Number of training epochs"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for training (cuda or cpu)",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=4,
        help="Interval for logging training metrics",
    )

    # WandB parameters
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="EXACT",
        help="WandB project name"
    )
    parser.add_argument(
        "--nowandb",
        action="store_true",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="W&B entity (team/user); None uses the logged-in default",
    )
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=1,
        help="Run the test/meta-test pass every N epochs (always on the last epoch). "
        "Testing dominates wall-clock for AdaBN models; 5-10 is a good value",
    )
    parser.add_argument(
        "--n_test_tasks",
        type=int,
        default=None,
        help="Task batches per test pass (default: same as --n_tasks)",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help="Suffix appended to the run name (e.g. ablation variant) so "
        "W&B runs, artifacts, checkpoints and result files do not collide",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/runs",
        help="Directory where a JSON summary of each run is written",
    )
    parser.add_argument(
        "--fomaml",
        action="store_true",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="unet",
        help="Model to use for training",
    )

    parser.add_argument(
        "--fine_split",
        action="store_true",
    )

    # Performance tuning arguments
    parser.add_argument(
        "--auto_tune",
        action="store_true",
        help=
        "Auto-tune DataLoader settings (num_workers, pin_memory) for best performance",
    )
    parser.add_argument(
        "--mixed_precision",
        action="store_true",
        help=
        "Enable mixed precision training (FP16) for faster training on compatible GPUs",
    )
    parser.add_argument(
        "--multi_gpu",
        action="store_true",
        help=
        "Auto-detect available GPUs and launch distributed training with torchrun. "
        "Automatically sets --nproc_per_node based on torch.cuda.device_count().",
    )
    parser.add_argument(
        "--use_pregenerated",
        action="store_true",
        help=
        "Use pre-generated episodes from disk instead of generating on-the-fly. "
        "Run pregenerate_episodes.py first to create the episodes.",
    )
    parser.add_argument(
        "--pregenerated_dir",
        type=str,
        default=None,
        help=
        "Directory containing pre-generated episodes (auto-detected if None)",
    )

    preliminary_parser = argparse.ArgumentParser(add_help=False)
    preliminary_parser.add_argument(
        "--dataset", type=str, default="physiq", help="Specify the dataset"
    )
    preliminary_parser.add_argument(
        "--model", type=str, default="unet", help="Specify the model"
    )
    # Parse known arguments to extract the dataset
    preliminary_args, _ = preliminary_parser.parse_known_args()

    # Add PhysiQ-specific arguments
    # parser = get_dataset_args(parser, preliminary_args)
    parser = get_model_args(parser, preliminary_args)
    # model = get_model(parser.parse_args())

    args = parser.parse_args()

    # Handle --no_side_noise flag to override --add_side_noise
    if args.no_side_noise:
        args.add_side_noise = False

    return args
