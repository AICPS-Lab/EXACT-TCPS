"""
Performance utilities for automatic DataLoader tuning and multi-GPU support.

Features:
1. Auto-benchmark DataLoader settings (num_workers, pin_memory) and pick optimal
2. Multi-GPU support via DistributedDataParallel (DDP)
3. Mixed precision training support

Usage:
    # In main_meta_v2.py or eval scripts:
    from utils_performance import (
        auto_tune_dataloader,
        setup_distributed,
        cleanup_distributed,
        get_device,
    )
    
    # Auto-tune (runs once, caches result)
    best_settings = auto_tune_dataloader(args)
    args.num_workers = best_settings['num_workers']
    args.pin_memory = best_settings['pin_memory']
    
    # Multi-GPU setup
    rank, world_size = setup_distributed()
"""

from __future__ import annotations

import os
import json
import time
import statistics
import hashlib
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, List, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

# =============================================================================
# DataLoader Auto-Tuning
# =============================================================================


@dataclass
class BenchmarkResult:
    """Result from a single DataLoader benchmark run."""
    num_workers: int
    pin_memory: bool
    persistent_workers: bool
    prefetch_factor: Optional[int]
    samples_per_sec: float
    median_batch_ms: float


class SyntheticBenchmarkDataset(Dataset):
    """Synthetic dataset for benchmarking DataLoader throughput."""

    def __init__(
        self,
        length: int = 5000,
        shape: Tuple[int,
                     int] = (500, 6),  # (timesteps, channels) - matches IMU
        cpu_work: int = 3,  # Simulate transform cost
        seed: int = 42,
    ):
        self.length = length
        self.shape = shape
        self.cpu_work = cpu_work
        self.seed = seed

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        g = torch.Generator()
        g.manual_seed(self.seed + idx)

        # IMU-like data: (T, C)
        x = torch.randn(*self.shape, generator=g, dtype=torch.float32)
        y = torch.zeros(self.shape[0], dtype=torch.long)

        # Simulate CPU work (transforms/augmentations)
        for _ in range(self.cpu_work):
            x = x * 0.999 + 0.001

        return x, y


def _benchmark_single_config(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: Optional[int],
    device: torch.device,
    warmup_steps: int,
    bench_steps: int,
) -> BenchmarkResult:
    """Run benchmark for a single DataLoader configuration."""

    kwargs = {}
    if num_workers > 0:
        kwargs['persistent_workers'] = persistent_workers
        if prefetch_factor is not None:
            kwargs['prefetch_factor'] = prefetch_factor

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        **kwargs,
    )

    def iterate_loader(steps: int) -> Tuple[List[float], int]:
        batch_times = []
        seen = 0
        it = iter(loader)

        if device.type == 'cuda':
            torch.cuda.synchronize()

        for _ in range(steps):
            t0 = time.perf_counter()
            try:
                x, y = next(it)
            except StopIteration:
                it = iter(loader)
                x, y = next(it)

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            if device.type == 'cuda':
                torch.cuda.synchronize()

            t1 = time.perf_counter()
            batch_times.append((t1 - t0) * 1000.0)
            seen += x.shape[0]

        return batch_times, seen

    # Warmup
    if warmup_steps > 0:
        iterate_loader(warmup_steps)

    # Benchmark
    t0 = time.perf_counter()
    batch_times, seen = iterate_loader(bench_steps)
    t1 = time.perf_counter()

    elapsed = max(1e-9, t1 - t0)
    samples_per_sec = seen / elapsed

    return BenchmarkResult(
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        samples_per_sec=samples_per_sec,
        median_batch_ms=statistics.median(batch_times)
        if batch_times else float('nan'),
    )


def get_cache_path() -> Path:
    """Get path for caching benchmark results."""
    cache_dir = Path.home() / '.cache' / 'exact_iccps'
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / 'dataloader_benchmark.json'


def get_system_hash() -> str:
    """Generate a hash based on system configuration."""
    info = {
        'cpu_count':
        os.cpu_count(),
        'cuda_available':
        torch.cuda.is_available(),
        'cuda_device_count':
        torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    if torch.cuda.is_available():
        info['cuda_device_name'] = torch.cuda.get_device_name(0)

    info_str = json.dumps(info, sort_keys=True)
    return hashlib.md5(info_str.encode()).hexdigest()[:12]


def load_cached_settings() -> Optional[Dict[str, Any]]:
    """Load cached benchmark settings if available and valid."""
    cache_path = get_cache_path()
    if not cache_path.exists():
        return None

    try:
        with open(cache_path, 'r') as f:
            data = json.load(f)

        # Check if cache is for same system
        if data.get('system_hash') != get_system_hash():
            return None

        return data.get('settings')
    except Exception:
        return None


def save_cached_settings(settings: Dict[str, Any]):
    """Save benchmark settings to cache."""
    cache_path = get_cache_path()
    data = {
        'system_hash': get_system_hash(),
        'settings': settings,
        'timestamp': time.time(),
    }
    with open(cache_path, 'w') as f:
        json.dump(data, f, indent=2)


def auto_tune_dataloader(
    args,
    num_workers_candidates: Optional[List[int]] = None,
    batch_size: Optional[int] = None,
    warmup_steps: int = 10,
    bench_steps: int = 50,
    verbose: bool = True,
    force_retune: bool = False,
) -> Dict[str, Any]:
    """
    Auto-tune DataLoader settings by benchmarking different configurations.
    
    Results are cached per system configuration.
    
    Args:
        args: Argument namespace with device info
        num_workers_candidates: List of num_workers values to try
        batch_size: Batch size for benchmarking (uses args.batch_size if None)
        warmup_steps: Number of warmup iterations
        bench_steps: Number of benchmark iterations
        verbose: Print progress and results
        force_retune: Ignore cache and re-run benchmark
    
    Returns:
        Dict with optimal settings: {'num_workers': int, 'pin_memory': bool, ...}
    """
    # Check cache first
    if not force_retune:
        cached = load_cached_settings()
        if cached is not None:
            if verbose:
                print(
                    f"[AutoTune] Using cached settings: num_workers={cached['num_workers']}, "
                    f"pin_memory={cached['pin_memory']}"
                )
            return cached

    if verbose:
        print("[AutoTune] Benchmarking DataLoader configurations...")

    # Determine device
    if hasattr(args, 'device'):
        if isinstance(args.device, torch.device):
            device = args.device
        else:
            device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Determine batch size
    if batch_size is None:
        batch_size = getattr(args, 'batch_size', 8)

    # Determine candidates
    if num_workers_candidates is None:
        cpu_count = os.cpu_count() or 4
        # Test 0 (main process), and powers of 2 up to CPU count
        candidates = [0, 2, 4]
        if cpu_count >= 8:
            candidates.append(8)
        if cpu_count >= 16:
            candidates.append(16)
        num_workers_candidates = candidates

    # Create synthetic dataset matching typical IMU data shape
    window_size = getattr(args, 'window_size', 500)
    in_channels = getattr(args, 'in_channels', 6)

    dataset = SyntheticBenchmarkDataset(
        length=5000,
        shape=(window_size, in_channels),
        cpu_work=3,
    )

    results: List[BenchmarkResult] = []

    for nw in num_workers_candidates:
        for pm in [False, True]:
            # Skip pin_memory=True on CPU (no benefit)
            if pm and device.type == 'cpu':
                continue

            try:
                result = _benchmark_single_config(
                    dataset=dataset,
                    batch_size=batch_size,
                    num_workers=nw,
                    pin_memory=pm,
                    persistent_workers=(nw > 0),
                    prefetch_factor=2 if nw > 0 else None,
                    device=device,
                    warmup_steps=warmup_steps,
                    bench_steps=bench_steps,
                )
                results.append(result)

                if verbose:
                    print(
                        f"  num_workers={nw:>2}, pin_memory={int(pm)}: "
                        f"{result.samples_per_sec:>8.1f} samples/s, "
                        f"{result.median_batch_ms:>6.2f} ms/batch"
                    )
            except Exception as e:
                if verbose:
                    print(
                        f"  num_workers={nw:>2}, pin_memory={int(pm)}: FAILED ({e})"
                    )

    if not results:
        # Fallback to safe defaults
        settings = {
            'num_workers': 0,
            'pin_memory': False,
            'persistent_workers': False,
            'prefetch_factor': None,
        }
    else:
        # Pick best by samples/sec
        best = max(results, key=lambda r: r.samples_per_sec)
        settings = {
            'num_workers': best.num_workers,
            'pin_memory': best.pin_memory,
            'persistent_workers': best.persistent_workers,
            'prefetch_factor': best.prefetch_factor,
        }

        if verbose:
            print(
                f"\n[AutoTune] Best: num_workers={best.num_workers}, "
                f"pin_memory={best.pin_memory} "
                f"({best.samples_per_sec:.1f} samples/s)"
            )

    # Cache results
    save_cached_settings(settings)

    return settings


def apply_dataloader_settings(args, settings: Dict[str, Any]):
    """Apply auto-tuned settings to args."""
    args.num_workers = settings['num_workers']
    args.pin_memory = settings['pin_memory']
    return args


# =============================================================================
# Multi-GPU / Distributed Training Support
# =============================================================================


def is_distributed() -> bool:
    """Check if running in distributed mode."""
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    """Get current process rank (0 if not distributed)."""
    if is_distributed():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    """Get total number of processes (1 if not distributed)."""
    if is_distributed():
        return dist.get_world_size()
    return 1


def is_main_process() -> bool:
    """Check if this is the main process (rank 0)."""
    return get_rank() == 0


def setup_distributed(
    backend: str = 'nccl',
    init_method: str = 'env://',
) -> Tuple[int, int]:
    """
    Initialize distributed training.
    
    Call this at the start of your script when using multi-GPU.
    
    Args:
        backend: 'nccl' for GPU, 'gloo' for CPU
        init_method: How to discover other processes
    
    Returns:
        (rank, world_size) tuple
    
    Usage with torchrun:
        torchrun --nproc_per_node=4 main_meta_v2.py --model exp ...
    """
    if not dist.is_available():
        print("[Distributed] torch.distributed not available")
        return 0, 1

    # Check if launched with torchrun / torch.distributed.launch
    if 'RANK' not in os.environ:
        # Not launched in distributed mode
        return 0, 1

    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))

    # Set device for this process
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    # Initialize process group
    dist.init_process_group(
        backend=backend,
        init_method=init_method,
        world_size=world_size,
        rank=rank,
    )

    if rank == 0:
        print(f"[Distributed] Initialized with {world_size} processes")

    return rank, world_size


def cleanup_distributed():
    """Clean up distributed training."""
    if is_distributed():
        dist.destroy_process_group()


def get_device(args=None, local_rank: Optional[int] = None) -> torch.device:
    """
    Get the appropriate device, handling distributed training.
    
    Args:
        args: Optional argument namespace with device setting
        local_rank: Optional local rank for multi-GPU
    
    Returns:
        torch.device for this process
    """
    # Check for distributed training local rank
    if local_rank is None:
        local_rank = int(os.environ.get('LOCAL_RANK', 0))

    if torch.cuda.is_available():
        if is_distributed():
            return torch.device(f'cuda:{local_rank}')
        elif args is not None and hasattr(args, 'device'):
            if isinstance(args.device, torch.device):
                return args.device
            return torch.device(args.device)
        else:
            return torch.device('cuda')
    else:
        return torch.device('cpu')


def wrap_model_distributed(
    model: nn.Module,
    device: torch.device,
    find_unused_parameters: bool = False,
) -> nn.Module:
    """
    Wrap model with DistributedDataParallel if in distributed mode.
    
    Args:
        model: PyTorch model
        device: Device to place model on
        find_unused_parameters: Set True if some params aren't used every forward
    
    Returns:
        Wrapped model (or original if not distributed)
    """
    model = model.to(device)

    if is_distributed() and torch.cuda.is_available():
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=find_unused_parameters,
        )
        if is_main_process():
            print(f"[Distributed] Model wrapped with DDP")

    return model


def get_distributed_sampler(
    dataset: Dataset,
    shuffle: bool = True,
    seed: int = 42,
) -> Optional[DistributedSampler]:
    """
    Get a DistributedSampler if in distributed mode.
    
    Args:
        dataset: Dataset to sample from
        shuffle: Whether to shuffle
        seed: Random seed for shuffling
    
    Returns:
        DistributedSampler or None if not distributed
    """
    if is_distributed():
        return DistributedSampler(
            dataset,
            num_replicas=get_world_size(),
            rank=get_rank(),
            shuffle=shuffle,
            seed=seed,
        )
    return None


# =============================================================================
# Mixed Precision Training Support
# =============================================================================


class MixedPrecisionContext:
    """
    Context manager for mixed precision training.
    
    Usage:
        mp = MixedPrecisionContext(enabled=True, device=device)
        
        for batch in loader:
            with mp.autocast():
                loss = model(batch)
            
            mp.scale_loss(loss).backward()
            mp.step(optimizer)
            mp.update()
    """

    def __init__(
        self,
        enabled: bool = True,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float16,
    ):
        self.enabled = enabled and torch.cuda.is_available()
        self.device_type = 'cuda' if (
            device is None or device.type == 'cuda'
        ) else 'cpu'
        self.dtype = dtype

        if self.enabled:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None

    def autocast(self):
        """Get autocast context manager."""
        return torch.autocast(
            device_type=self.device_type,
            dtype=self.dtype,
            enabled=self.enabled,
        )

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss for mixed precision."""
        if self.scaler is not None:
            return self.scaler.scale(loss)
        return loss

    def step(self, optimizer: torch.optim.Optimizer):
        """Optimizer step with gradient unscaling."""
        if self.scaler is not None:
            self.scaler.step(optimizer)
        else:
            optimizer.step()

    def update(self):
        """Update scaler for next iteration."""
        if self.scaler is not None:
            self.scaler.update()


# =============================================================================
# Utility Functions
# =============================================================================


def get_available_gpus() -> int:
    """Get the number of available GPUs."""
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    return 0


def should_use_distributed(args=None) -> bool:
    """
    Determine if distributed training should be used.
    
    Returns True if:
    - Multiple GPUs are available
    - --multi_gpu flag is set (if args provided)
    - Not already in a distributed context
    """
    # Already in distributed mode
    if 'RANK' in os.environ:
        return False

    # Check for multi_gpu flag
    if args is not None and getattr(args, 'multi_gpu', False):
        return get_available_gpus() > 1

    return False


def launch_distributed(
    script_path: Optional[str] = None,
    args_list: Optional[List[str]] = None,
    nproc_per_node: Optional[int] = None,
    master_port: int = 29500,
) -> int:
    """
    Launch the current script with torchrun for multi-GPU training.
    
    Args:
        script_path: Path to the script to launch (defaults to current script)
        args_list: Command line arguments to pass (defaults to sys.argv[1:])
        nproc_per_node: Number of processes (defaults to GPU count)
        master_port: Port for distributed communication
    
    Returns:
        Exit code from torchrun
    """
    import sys
    import subprocess

    if script_path is None:
        script_path = sys.argv[0]

    if args_list is None:
        # Filter out --multi_gpu to avoid infinite recursion
        args_list = [arg for arg in sys.argv[1:] if arg != '--multi_gpu']

    if nproc_per_node is None:
        nproc_per_node = get_available_gpus()

    if nproc_per_node <= 1:
        print("[Distributed] Only 1 GPU available, running in single-GPU mode")
        return -1  # Signal to continue without distributed

    # Build torchrun command
    cmd = [
        sys.executable,
        '-m',
        'torch.distributed.run',
        f'--nproc_per_node={nproc_per_node}',
        f'--master_port={master_port}',
        '--standalone',
        script_path,
    ] + args_list

    print(f"[Distributed] Launching with {nproc_per_node} GPUs...")
    print(f"[Distributed] Command: {' '.join(cmd)}")

    # Execute torchrun
    result = subprocess.run(cmd)
    return result.returncode


def auto_distribute(args) -> bool:
    """
    Automatically launch distributed training if multiple GPUs are available.
    
    Call this at the very start of your main script. If it returns True,
    the script should exit (torchrun will re-launch it).
    
    Args:
        args: Parsed arguments with optional --multi_gpu flag
    
    Returns:
        True if script was re-launched with torchrun (caller should exit)
        False if should continue normally (single GPU or already distributed)
    
    Usage:
        if __name__ == "__main__":
            args = get_args()
            if auto_distribute(args):
                sys.exit(0)  # torchrun handles everything
            # ... rest of training code
    """
    # Skip if already in distributed context
    if 'RANK' in os.environ:
        return False

    # Check if multi_gpu requested
    if not getattr(args, 'multi_gpu', False):
        return False

    # Check GPU count
    gpu_count = get_available_gpus()
    if gpu_count <= 1:
        if gpu_count == 1:
            print(
                "[Distributed] Only 1 GPU available, continuing in single-GPU mode"
            )
        else:
            print("[Distributed] No GPUs available, continuing on CPU")
        return False

    # Launch with torchrun
    exit_code = launch_distributed(nproc_per_node=gpu_count)

    if exit_code == -1:
        # Signal to continue without distributed
        return False

    # torchrun completed (or failed)
    import sys
    sys.exit(exit_code)


def print_gpu_memory_summary(device: Optional[torch.device] = None):
    """Print GPU memory usage summary."""
    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    if device is None:
        device = torch.device('cuda')

    allocated = torch.cuda.memory_allocated(device) / 1024**3
    reserved = torch.cuda.memory_reserved(device) / 1024**3
    max_allocated = torch.cuda.max_memory_allocated(device) / 1024**3

    print(
        f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, "
        f"{max_allocated:.2f}GB peak"
    )


def get_gpu_info() -> Dict[str, Any]:
    """Get information about available GPUs."""
    info = {
        'cuda_available': torch.cuda.is_available(),
        'device_count': 0,
        'devices': [],
    }

    if torch.cuda.is_available():
        info['device_count'] = torch.cuda.device_count()
        for i in range(info['device_count']):
            props = torch.cuda.get_device_properties(i)
            info['devices'].append(
                {
                    'index': i,
                    'name': props.name,
                    'total_memory_gb': props.total_memory / 1024**3,
                    'compute_capability': f"{props.major}.{props.minor}",
                }
            )

    return info


def print_system_info():
    """Print system information relevant to training performance."""
    import platform

    print("\n" + "=" * 60)
    print("System Information")
    print("=" * 60)
    print(f"Python: {platform.python_version()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"cuDNN Version: {torch.backends.cudnn.version()}")
        print(f"GPU Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(
                f"  GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f}GB)"
            )

    print(f"CPU Count: {os.cpu_count()}")
    print("=" * 60 + "\n")
