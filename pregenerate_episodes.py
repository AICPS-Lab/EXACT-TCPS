"""
Episode Pre-generation Script for Meta-Learning

Pre-generates all episodes (tasks) for training to avoid DataLoader overhead.
Episodes are saved to disk and can be loaded into RAM at training time.

Usage:
    # Generate episodes for all seeds
    python pregenerate_episodes.py --dataset physiq --n_epochs 200 --seeds 42 43 44 45 46

    # Generate for single seed
    python pregenerate_episodes.py --dataset physiq --n_epochs 200 --seeds 42

    # Use pre-generated episodes in training
    python main_meta_v2.py --model exp --dataset physiq --use_pregenerated
"""

import argparse
import os
import random
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Suppress matplotlib 3D projection warning
warnings.filterwarnings("ignore", message="Unable to import Axes3D")

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.DenseLabelTaskSampler import DenseLabelTaskSampler
from until_argparser import get_dataset, get_args
from utilities import seed


class PreGeneratedEpisodeDataset(torch.utils.data.Dataset):
    """
    Dataset that loads pre-generated episodes from disk or memory.
    
    Each episode contains:
        - support_images: (batch_size, channels, height, width)
        - support_labels: (batch_size, height, width) 
        - query_images: (batch_size, channels, height, width)
        - query_labels: (batch_size, height, width)
        - task_info: metadata about the task
    """

    def __init__(
        self,
        episodes_dir: str,
        epoch: Optional[int] = None,
        load_to_memory: bool = True,
        split: str = 'train',
    ):
        """
        Args:
            episodes_dir: Directory containing pre-generated episodes
            epoch: Specific epoch to load (None = load all)
            load_to_memory: If True, load all episodes into RAM
            split: 'train' or 'test'
        """
        self.episodes_dir = Path(episodes_dir)
        self.split = split
        self.load_to_memory = load_to_memory
        self.epoch = epoch

        # Find all episode files
        self.episode_files = self._find_episode_files()

        # Optionally load all into memory
        self.cached_episodes = None
        if load_to_memory:
            self._load_all_to_memory()

    def _find_episode_files(self) -> List[Path]:
        """Find all episode files for the given split."""
        pattern = f"{self.split}_epoch_*_task_*.pt"
        if self.epoch is not None:
            pattern = f"{self.split}_epoch_{self.epoch:04d}_task_*.pt"

        files = sorted(self.episodes_dir.glob(pattern))
        if not files:
            raise FileNotFoundError(
                f"No episode files found in {self.episodes_dir} with pattern {pattern}"
            )
        return files

    def _load_all_to_memory(self):
        """Load all episodes into memory."""
        print(
            f"[PreGenerated] Loading {len(self.episode_files)} episodes into memory..."
        )
        self.cached_episodes = []
        for f in tqdm(self.episode_files, desc="Loading episodes"):
            self.cached_episodes.append(torch.load(f, weights_only=False))
        print(f"[PreGenerated] Loaded {len(self.cached_episodes)} episodes")

    def __len__(self) -> int:
        return len(self.episode_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        """Get an episode by index."""
        if self.cached_episodes is not None:
            episode = self.cached_episodes[idx]
        else:
            episode = torch.load(self.episode_files[idx], weights_only=False)

        return (
            episode['support_images'],
            episode['support_labels'],
            episode['query_images'],
            episode['query_labels'],
            episode['task_info'],
        )

    def get_epoch_episodes(self, epoch: int) -> List[Tuple[torch.Tensor, ...]]:
        """Get all episodes for a specific epoch."""
        epoch_pattern = f"{self.split}_epoch_{epoch:04d}_task_"
        episodes = []

        for i, f in enumerate(self.episode_files):
            if epoch_pattern in f.name:
                episodes.append(self[i])

        return episodes


class PreGeneratedEpisodeLoader:
    """
    Fast episode loader that iterates through pre-generated episodes.
    
    Replaces DataLoader for training when using pre-generated episodes.
    """

    def __init__(
        self,
        episodes_dir: str,
        n_epochs: int,
        n_tasks: int,
        split: str = 'train',
        load_to_memory: bool = True,
        shuffle_per_epoch: bool = True,
        device: str = 'cuda',
    ):
        """
        Args:
            episodes_dir: Directory containing pre-generated episodes
            n_epochs: Number of epochs
            n_tasks: Number of tasks per epoch
            split: 'train' or 'test'
            load_to_memory: Load all episodes to RAM
            shuffle_per_epoch: Shuffle episode order within each epoch
            device: Target device for tensors
        """
        self.episodes_dir = Path(episodes_dir)
        self.n_epochs = n_epochs
        self.n_tasks = n_tasks
        self.split = split
        self.shuffle_per_epoch = shuffle_per_epoch
        self.device = device

        # Load all episodes into memory
        self.episodes = self._load_episodes(load_to_memory)

        # Index mapping: [epoch][task] -> episode_idx
        self.epoch_indices = self._build_epoch_indices()

    def _load_episodes(self, load_to_memory: bool) -> Dict:
        """Load episodes organized by epoch."""
        episodes = {}

        for epoch in range(self.n_epochs):
            epoch_file = self.episodes_dir / f"{self.split}_epoch_{epoch:04d}.pt"
            if epoch_file.exists():
                if load_to_memory:
                    episodes[epoch] = torch.load(
                        epoch_file, weights_only=False
                    )
                else:
                    episodes[epoch] = epoch_file
            else:
                # Try individual task files
                task_episodes = []
                for task in range(self.n_tasks):
                    task_file = self.episodes_dir / f"{self.split}_epoch_{epoch:04d}_task_{task:04d}.pt"
                    if task_file.exists():
                        if load_to_memory:
                            task_episodes.append(
                                torch.load(task_file, weights_only=False)
                            )
                        else:
                            task_episodes.append(task_file)
                if task_episodes:
                    episodes[epoch] = task_episodes

        return episodes

    def _build_epoch_indices(self) -> Dict[int, List[int]]:
        """Build index mapping for each epoch."""
        indices = {}
        for epoch in range(self.n_epochs):
            indices[epoch] = list(range(self.n_tasks))
        return indices

    def get_epoch_iterator(self, epoch: int):
        """Get an iterator for a specific epoch."""
        if epoch not in self.episodes:
            raise ValueError(
                f"Epoch {epoch} not found in pre-generated episodes"
            )

        indices = self.epoch_indices[epoch].copy()
        if self.shuffle_per_epoch:
            random.shuffle(indices)

        epoch_data = self.episodes[epoch]

        for task_idx in indices:
            if isinstance(epoch_data, list):
                episode = epoch_data[task_idx]
            else:
                episode = epoch_data[task_idx]

            # Handle file paths (lazy loading)
            if isinstance(episode, Path):
                episode = torch.load(episode, weights_only=False)

            yield (
                episode['support_images'],
                episode['support_labels'],
                episode['query_images'],
                episode['query_labels'],
                episode['task_info'],
            )


def generate_episodes(
    args,
    n_epochs: int,
    output_dir: str,
    split: str = 'train',
    seed_offset: int = 0,
):
    """
    Generate and save episodes for multiple epochs.
    
    Args:
        args: Training arguments
        n_epochs: Number of epochs to generate
        output_dir: Directory to save episodes
        split: 'train' or 'test'
        seed_offset: Offset to add to base seed
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Get dataset
    train_dataset, test_dataset = get_dataset(args)
    dataset = train_dataset if split == 'train' else test_dataset

    # Create sampler
    sampler = DenseLabelTaskSampler(
        dataset,
        n_shot=args.n_shot,
        batch_size=args.batch_size,
        n_query=args.n_query,
        n_tasks=args.n_tasks,
        threshold_ratio=args.threshold_ratio,
        add_side_noise=args.add_side_noise,
        args=args,
    )

    # Create DataLoader
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=0,  # Single-threaded for reproducibility
        pin_memory=False,
        collate_fn=sampler.episodic_collate_fn,
    )

    print(
        f"[PreGenerate] Generating {n_epochs} epochs x {args.n_tasks} tasks for {split}..."
    )
    print(f"[PreGenerate] Output directory: {output_path}")

    for epoch in tqdm(range(n_epochs), desc=f"Generating {split} epochs"):
        # Set seed for reproducibility
        seed(args.seed + seed_offset + epoch)

        # Generate epoch file with all tasks
        epoch_episodes = []

        loader_iter = iter(loader)
        for task_idx in tqdm(range(args.n_tasks), desc=f"Epoch {epoch} tasks", leave=False):
            try:
                x_spt, y_spt, x_qry, y_qry, task_info = next(loader_iter)
            except StopIteration:
                # Recreate iterator if exhausted
                loader_iter = iter(loader)
                x_spt, y_spt, x_qry, y_qry, task_info = next(loader_iter)

            episode = {
                'support_images': x_spt,
                'support_labels': y_spt,
                'query_images': x_qry,
                'query_labels': y_qry,
                'task_info': task_info,
                'epoch': epoch,
                'task_idx': task_idx,
            }
            epoch_episodes.append(episode)

        # Save entire epoch as single file (faster loading)
        epoch_file = output_path / f"{split}_epoch_{epoch:04d}.pt"
        torch.save(epoch_episodes, epoch_file)

    # Save metadata
    metadata = {
        'n_epochs': n_epochs,
        'n_tasks': args.n_tasks,
        'n_shot': args.n_shot,
        'n_query': args.n_query,
        'batch_size': args.batch_size,
        'seed': args.seed,
        'seed_offset': seed_offset,
        'split': split,
        'dataset': args.dataset,
        'add_side_noise': args.add_side_noise,
    }
    torch.save(metadata, output_path / f"{split}_metadata.pt")

    print(f"[PreGenerate] Saved {n_epochs} epochs to {output_path}")
    return output_path


def get_pregenerated_dir(args) -> Path:
    """Get the directory for pre-generated episodes based on args."""
    base_dir = Path(args.data_root) / "pregenerated_episodes"

    # Create unique identifier based on key parameters
    identifier = (
        f"{args.dataset}_seed{args.seed}_"
        f"shot{args.n_shot}_query{args.n_query}_"
        f"batch{args.batch_size}_tasks{args.n_tasks}"
    )

    return base_dir / identifier


def load_pregenerated_episodes(
    args,
    split: str = 'train',
    load_to_memory: bool = True,
) -> PreGeneratedEpisodeLoader:
    """
    Load pre-generated episodes for training/testing.
    
    Args:
        args: Training arguments
        split: 'train' or 'test'
        load_to_memory: Load all to RAM (recommended)
    
    Returns:
        PreGeneratedEpisodeLoader instance
    """
    episodes_dir = get_pregenerated_dir(args)

    # Load metadata to get n_epochs
    metadata_file = episodes_dir / f"{split}_metadata.pt"
    if not metadata_file.exists():
        raise FileNotFoundError(
            f"Pre-generated episodes not found at {episodes_dir}. "
            f"Run: python pregenerate_episodes.py --dataset {args.dataset} --seeds {args.seed}"
        )

    metadata = torch.load(metadata_file, weights_only=False)

    return PreGeneratedEpisodeLoader(
        episodes_dir=str(episodes_dir),
        n_epochs=metadata['n_epochs'],
        n_tasks=metadata['n_tasks'],
        split=split,
        load_to_memory=load_to_memory,
        device=args.device,
    )


def main():
    """Main function for episode pre-generation."""
    parser = argparse.ArgumentParser(
        description="Pre-generate episodes for meta-learning"
    )

    # Core arguments
    parser.add_argument(
        '--dataset', type=str, default='physiq', help='Dataset name'
    )
    parser.add_argument(
        '--data_root',
        type=str,
        default='data',
        help='Data root directory (should match main_meta_v2.py)'
    )
    parser.add_argument(
        '--n_epochs',
        type=int,
        default=200,
        help='Number of epochs to generate'
    )
    parser.add_argument(
        '--seeds',
        type=int,
        nargs='+',
        default=[42],
        help='Seeds to generate for'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory (auto-generated if None)'
    )

    # Task parameters (should match training)
    parser.add_argument(
        '--n_shot', type=int, default=5, help='Number of support examples'
    )
    parser.add_argument(
        '--n_query', type=int, default=5, help='Number of query examples'
    )
    parser.add_argument(
        '--batch_size', type=int, default=8, help='Batch size'
    )
    parser.add_argument(
        '--n_tasks', type=int, default=100, help='Number of tasks per epoch'
    )
    parser.add_argument(
        '--threshold_ratio',
        type=float,
        default=0.25,
        help='Threshold ratio for valid labels (matches until_argparser.py)'
    )
    parser.add_argument(
        '--add_side_noise',
        action='store_true',
        default=True,
        help='Add side noise'
    )
    parser.add_argument(
        '--no_side_noise', action='store_true', help='Disable side noise'
    )

    # Additional args needed by get_dataset
    parser.add_argument(
        '--window_size', type=int, default=500, help='Window size'
    )
    parser.add_argument(
        '--window_step', type=int, default=25, help='Window step'
    )
    parser.add_argument(
        '--fine_split', action='store_true', help='Use fine split'
    )
    parser.add_argument(
        '--out_channels',
        type=int,
        default=2,
        help='Number of output channels'
    )
    parser.add_argument(
        '--rotation_chance',
        type=float,
        default=0,
        help='Rotation augmentation chance'
    )
    parser.add_argument(
        '--model', type=str, default='exp', help='Model type (for transforms)'
    )
    parser.add_argument(
        '--loocv',
        action='store_true',
        default=False,
        help='Use Leave-One-Out Cross-Validation mode'
    )
    parser.add_argument(
        '--dataset_seed',
        type=int,
        default=42,
        help='Seed for dataset shuffling'
    )
    parser.add_argument(
        '--shuffle',
        type=str,
        default='random_variation',
        help='Shuffle mode: random, random_variation, sorted'
    )
    parser.add_argument(
        '--noise_type',
        type=str,
        default='all',
        help='Type of noise to add: static, idle, sudden, nonexercise, all'
    )

    args = parser.parse_args()

    # Handle --no_side_noise
    if args.no_side_noise:
        args.add_side_noise = False

    print("=" * 60)
    print("Episode Pre-Generation")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Epochs: {args.n_epochs}")
    print(f"Tasks per epoch: {args.n_tasks}")
    print(f"Seeds: {args.seeds}")
    print(f"Shot/Query: {args.n_shot}/{args.n_query}")
    print(f"Batch size: {args.batch_size}")
    print("=" * 60)

    for seed_val in args.seeds:
        print(f"\n[Seed {seed_val}]")
        args.seed = seed_val

        # Determine output directory
        output_dir = args.output_dir or get_pregenerated_dir(args)

        # Generate train episodes
        generate_episodes(
            args,
            n_epochs=args.n_epochs,
            output_dir=str(output_dir),
            split='train',
            seed_offset=0,
        )

        # Generate test episodes (fewer epochs typically needed)
        generate_episodes(
            args,
            n_epochs=args.n_epochs,
            output_dir=str(output_dir),
            split='test',
            seed_offset=10000,  # Different seed offset for test
        )

        print(f"[Seed {seed_val}] Complete! Episodes saved to: {output_dir}")

    print("\n" + "=" * 60)
    print("Pre-generation complete!")
    print(
        f"Use with: python main_meta_v2.py --use_pregenerated --dataset {args.dataset}"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
