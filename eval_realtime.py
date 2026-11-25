#!/usr/bin/env python3
"""
Real-Time Evaluation Script for ALL Models (Baseline & EXACT)

This script evaluates real-time deployment capabilities for all supported models:
- CAEX (EXACT with Channel-Aware gating)
- EX (EXACT base)
- UNet
- Transformer
- Segmenter
- CNN

Metrics evaluated:
1. Inference latency (end-to-end, preprocessing, model forward pass)
2. Throughput (samples/second, windows/second)
3. Memory usage (CPU, GPU peak and average)
4. Computational cost (FLOPs, MACs, Parameters)
5. Real-time streaming simulation with configurable sensor rates
6. MAML Few-Shot Adaptation latency and memory cost
7. AdaBN Adaptation overhead (for models that support it)

Usage:
    # Evaluate a specific model
    python eval_realtime.py --model caex --model_path artifacts/caex-physiq-42:v1/caex-physiq-42.pth
    
    # Evaluate all baselines
    python eval_realtime.py --model unet --model_path artifacts/unet-physiq-43:v0/unet-physiq-43.pth
    
    # Run full benchmark with multiple iterations
    python eval_realtime.py --model transformer --mode full --num_iterations 1000
    
    # Test real-time adaptation scenario
    python eval_realtime.py --model ex --mode adaptation --n_inner_iter 5 --k_shot 5

Author: EXACT-ICCPS2025
"""

import argparse
import gc
import os
import sys
import threading
import time
import json
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import warnings
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# For MAML adaptation
try:
    import higher
    HIGHER_AVAILABLE = True
except ImportError:
    HIGHER_AVAILABLE = False
    print(
        "Warning: 'higher' library not found. MAML adaptation benchmarks disabled."
    )

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# =============================================================================
# Metrics Data Classes
# =============================================================================


@dataclass
class LatencyMetrics:
    """Container for latency measurements in milliseconds."""
    preprocess_ms: float = 0.0
    inference_ms: float = 0.0
    postprocess_ms: float = 0.0
    total_ms: float = 0.0

    @property
    def breakdown(self) -> Dict[str, float]:
        return {
            "preprocess_ms": self.preprocess_ms,
            "inference_ms": self.inference_ms,
            "postprocess_ms": self.postprocess_ms,
            "total_ms": self.total_ms,
        }


@dataclass
class MemoryMetrics:
    """Container for memory usage measurements in MB."""
    cpu_current_mb: float = 0.0
    cpu_peak_mb: float = 0.0
    gpu_current_mb: float = 0.0
    gpu_peak_mb: float = 0.0
    model_size_mb: float = 0.0

    @property
    def summary(self) -> Dict[str, float]:
        return {
            "cpu_current_mb": self.cpu_current_mb,
            "cpu_peak_mb": self.cpu_peak_mb,
            "gpu_current_mb": self.gpu_current_mb,
            "gpu_peak_mb": self.gpu_peak_mb,
            "model_size_mb": self.model_size_mb,
        }


@dataclass
class ThroughputMetrics:
    """Container for throughput measurements."""
    samples_per_second: float = 0.0
    windows_per_second: float = 0.0
    batches_per_second: float = 0.0
    real_time_factor: float = 0.0  # >1 means faster than real-time


@dataclass
class ComputeMetrics:
    """Container for computational cost metrics."""
    flops: int = 0
    macs: int = 0
    params: int = 0
    params_trainable: int = 0

    @property
    def summary(self) -> Dict[str, Any]:
        return {
            "flops": self.flops,
            "flops_readable": self._format_number(self.flops),
            "macs": self.macs,
            "macs_readable": self._format_number(self.macs),
            "params": self.params,
            "params_readable": self._format_number(self.params),
            "params_trainable": self.params_trainable,
        }

    @staticmethod
    def _format_number(num: int) -> str:
        """Format large numbers with K/M/G suffixes."""
        if num >= 1e9:
            return f"{num/1e9:.2f}G"
        elif num >= 1e6:
            return f"{num/1e6:.2f}M"
        elif num >= 1e3:
            return f"{num/1e3:.2f}K"
        return str(num)


@dataclass
class AdaptationMetrics:
    """Container for MAML/AdaBN adaptation metrics."""
    # AdaBN metrics
    adabn_latency_ms: float = 0.0
    adabn_memory_mb: float = 0.0
    adabn_steps: int = 0

    # MAML inner-loop metrics
    maml_adaptation_latency_ms: float = 0.0
    maml_per_step_latency_ms: float = 0.0
    maml_memory_peak_mb: float = 0.0
    maml_gradient_memory_mb: float = 0.0
    n_inner_iter: int = 0

    # Combined adaptation
    total_adaptation_latency_ms: float = 0.0

    # Few-shot settings
    k_shot: int = 0
    n_way: int = 0

    @property
    def summary(self) -> Dict[str, Any]:
        return {
            "adabn_latency_ms": self.adabn_latency_ms,
            "adabn_memory_mb": self.adabn_memory_mb,
            "adabn_steps": self.adabn_steps,
            "maml_adaptation_latency_ms": self.maml_adaptation_latency_ms,
            "maml_per_step_latency_ms": self.maml_per_step_latency_ms,
            "maml_memory_peak_mb": self.maml_memory_peak_mb,
            "maml_gradient_memory_mb": self.maml_gradient_memory_mb,
            "n_inner_iter": self.n_inner_iter,
            "total_adaptation_latency_ms": self.total_adaptation_latency_ms,
            "k_shot": self.k_shot,
            "n_way": self.n_way,
        }


# =============================================================================
# IMU Sensor Emulator
# =============================================================================


class IMUSensorEmulator:
    """Emulates real-time IMU sensor data streaming."""

    def __init__(
        self,
        sampling_rate_hz: float = 100.0,
        channels: int = 6,
        noise_std: float = 0.1,
        pattern: str = "sine",
        replay_data: Optional[np.ndarray] = None,
    ):
        self.sampling_rate_hz = sampling_rate_hz
        self.sample_interval = 1.0 / sampling_rate_hz
        self.channels = channels
        self.noise_std = noise_std
        self.pattern = pattern
        self.replay_data = replay_data
        self.replay_idx = 0

        self._running = False
        self._buffer = deque(maxlen=10000)
        self._sample_count = 0
        self._start_time = None
        self._thread = None

    def _generate_sample(self) -> np.ndarray:
        """Generate a single IMU sample based on pattern."""
        t = self._sample_count * self.sample_interval

        if self.pattern == "replay" and self.replay_data is not None:
            sample = self.replay_data[self.replay_idx % len(self.replay_data)]
            self.replay_idx += 1
        elif self.pattern == "sine":
            sample = np.array(
                [
                    np.sin(2 * np.pi * (0.5 + i * 0.3) * t)
                    for i in range(self.channels)
                ]
            )
        else:
            sample = np.random.randn(self.channels)

        sample += np.random.randn(self.channels) * self.noise_std
        self._sample_count += 1

        return sample.astype(np.float32)

    def _streaming_loop(self):
        """Background thread for continuous data generation."""
        self._start_time = time.perf_counter()

        while self._running:
            expected_samples = int(
                (time.perf_counter() - self._start_time) *
                self.sampling_rate_hz
            )

            while self._sample_count < expected_samples and self._running:
                sample = self._generate_sample()
                self._buffer.append((time.perf_counter(), sample))

            time.sleep(self.sample_interval / 2)

    def start_streaming(self):
        """Start background data streaming."""
        if self._running:
            return

        self._running = True
        self._sample_count = 0
        self._buffer.clear()
        self._thread = threading.Thread(
            target=self._streaming_loop, daemon=True
        )
        self._thread.start()

    def stop_streaming(self):
        """Stop background data streaming."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None

    def get_samples(self, n: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Get n samples from the buffer."""
        if len(self._buffer) < n:
            return None

        samples = []
        timestamps = []
        for _ in range(n):
            ts, sample = self._buffer.popleft()
            timestamps.append(ts)
            samples.append(sample)

        return np.array(timestamps), np.array(samples)

    def generate_batch(
        self, window_size: int, batch_size: int = 1
    ) -> np.ndarray:
        """Generate a batch of windows for offline evaluation."""
        batch = np.zeros(
            (batch_size, window_size, self.channels), dtype=np.float32
        )

        for b in range(batch_size):
            for t in range(window_size):
                batch[b, t] = self._generate_sample()

        return batch


# =============================================================================
# Model Loading Utilities
# =============================================================================


def get_mock_args(
    model_name: str,
    in_channels: int = 6,
    out_channels: int = 17,
    embed_dim: int = 32,
    num_heads: int = 4,
) -> argparse.Namespace:
    """Create mock args for model instantiation."""
    args = argparse.Namespace()
    args.in_channels = in_channels
    args.out_channels = out_channels
    args.embed_dim = embed_dim
    args.num_heads = num_heads
    args.model = model_name

    # Common defaults
    args.dropout = 0.5
    args.conv_sizes = [32, 64, 128]

    # Transformer defaults
    args.ninp = 64
    args.num_layers = 6
    args.init_std = 0.02
    args.activation = 'relu'

    # Segmenter defaults (embed_dims=128 is the training default)
    args.embed_dims = 128
    args.mlp_ratio = 4
    args.num_classes = out_channels

    # CAEX-specific defaults
    args.caex_gate = "se"
    args.caex_gate_reduction = 2
    args.caex_gate_softmax = True
    args.caex_group_gate = True
    args.caex_imu_normalize = True
    args.caex_imu_norm_kind = "group_zscore"
    args.caex_imu_norm_skip_noise = True
    args.caex_use_adabn = False
    args.caex_adabn_steps = 50

    return args


def load_model(
    model_name: str,
    model_path: Optional[str] = None,
    device: str = "cuda",
    in_channels: int = 6,
    out_channels: int = 17,
) -> nn.Module:
    """
    Load any supported model for evaluation.
    
    Supported models: caex, ex, unet, transformer, segmenter, cnn
    """
    model_name = model_name.lower()

    # Import model classes
    if model_name == "caex":
        from methods.CAEX import CAEX
        ModelClass = CAEX
    elif model_name == "ex":
        from methods.EX import EX
        ModelClass = EX
    elif model_name == "unet":
        from methods.unet import UNet
        ModelClass = UNet
    elif model_name == "transformer":
        from methods.transformer import TransformerModel
        ModelClass = TransformerModel
    elif model_name == "segmenter":
        from methods.segmenter import Segmenter
        ModelClass = Segmenter
    elif model_name == "cnn":
        from methods.cnn import CNN
        ModelClass = CNN
    else:
        raise ValueError(
            f"Unsupported model: {model_name}. "
            f"Supported: caex, ex, unet, transformer, segmenter, cnn"
        )

    args = get_mock_args(
        model_name, in_channels=in_channels, out_channels=out_channels
    )
    from argtypes import restore_train_args
    restore_train_args(args, model_path)
    model = ModelClass(args)

    if model_path and os.path.exists(model_path):
        print(f"Loading model from: {model_path}")
        checkpoint = torch.load(
            model_path, map_location=device, weights_only=False
        )

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        # Handle wrapped model (keys prefixed with 'net.')
        if any(k.startswith('net.') for k in state_dict.keys()):
            print(
                "  Detected wrapped model checkpoint, stripping 'net.' prefix..."
            )
            # strip the prefix only; str.replace would also mangle
            # keys such as "conv_net.0.weight" -> "conv_0.weight"
            state_dict = {
                (k[len('net.'):] if k.startswith('net.') else k): v
                for k, v in state_dict.items()
            }

        # Handle module prefix from DataParallel
        if any(k.startswith('module.') for k in state_dict.keys()):
            print(
                "  Detected DataParallel checkpoint, stripping 'module.' prefix..."
            )
            state_dict = {
                (k[len('module.'):] if k.startswith('module.') else k): v
                for k, v in state_dict.items()
            }

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"Checkpoint does not match the {model_name} architecture "
                f"(missing={list(missing)[:5]}, unexpected={list(unexpected)[:5]}). "
                "Refusing to benchmark a partially loaded model."
            )
        print(f"  Model loaded successfully")
    else:
        print(
            f"Using randomly initialized {model_name} model (no checkpoint provided)"
        )

    model = model.to(device)
    model.eval()

    return model


# =============================================================================
# Measurement Utilities
# =============================================================================


def get_model_size_mb(model: nn.Module) -> float:
    """Calculate model size in MB."""
    param_size = sum(
        param.nelement() * param.element_size() for param in model.parameters()
    )
    buffer_size = sum(
        buffer.nelement() * buffer.element_size() for buffer in model.buffers()
    )
    return (param_size + buffer_size) / (1024**2)


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def get_memory_usage(device: str = "cuda") -> Dict[str, float]:
    """Get current memory usage in MB."""
    import psutil

    process = psutil.Process(os.getpid())
    cpu_mb = process.memory_info().rss / (1024**2)

    gpu_mb = 0.0
    gpu_peak_mb = 0.0

    if device == "cuda" and torch.cuda.is_available():
        gpu_mb = torch.cuda.memory_allocated() / (1024**2)
        gpu_peak_mb = torch.cuda.max_memory_allocated() / (1024**2)

    return {
        "cpu_mb": cpu_mb,
        "gpu_mb": gpu_mb,
        "gpu_peak_mb": gpu_peak_mb,
    }


def estimate_flops_macs(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    device: str = "cuda",
) -> Tuple[int, int]:
    """Estimate FLOPs and MACs for the model."""
    flops, macs = 0, 0

    # Try using thop library
    try:
        from thop import profile
        dummy_input = torch.randn(input_shape).to(device)
        macs, params = profile(model, inputs=(dummy_input, ), verbose=False)
        flops = macs * 2
        return int(flops), int(macs)
    except ImportError:
        pass

    # Try using fvcore
    try:
        from fvcore.nn import FlopCountAnalysis
        dummy_input = torch.randn(input_shape).to(device)
        flop_counter = FlopCountAnalysis(model, dummy_input)
        flops = flop_counter.total()
        macs = flops // 2
        return int(flops), int(macs)
    except ImportError:
        pass

    # Manual estimation
    total_params, _ = count_parameters(model)
    seq_len = input_shape[1] if len(input_shape) > 1 else 1
    flops = 2 * total_params * seq_len
    macs = flops // 2

    return int(flops), int(macs)


# =============================================================================
# Latency Benchmark
# =============================================================================


class LatencyBenchmark:
    """Comprehensive latency benchmarking for inference pipeline."""

    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        warmup_iterations: int = 50,
        benchmark_iterations: int = 500,
    ):
        self.model = model
        self.device = device
        self.warmup_iterations = warmup_iterations
        self.benchmark_iterations = benchmark_iterations

    def _preprocess(self, x: np.ndarray) -> torch.Tensor:
        """Preprocess numpy array to tensor."""
        tensor = torch.from_numpy(x).float()
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)
        return tensor.to(self.device)

    def _postprocess(self, output: torch.Tensor) -> np.ndarray:
        """Postprocess tensor to numpy array with class predictions."""
        probs = torch.softmax(output, dim=-1)
        preds = torch.argmax(probs, dim=-1)
        return preds.cpu().numpy()

    @torch.no_grad()
    def run_single(self, x: np.ndarray) -> Tuple[np.ndarray, LatencyMetrics]:
        """Run single inference with timing."""
        if self.device == "cuda":
            torch.cuda.synchronize()

        t_start = time.perf_counter()

        # Preprocess
        t_pre = time.perf_counter()
        tensor = self._preprocess(x)
        if self.device == "cuda":
            torch.cuda.synchronize()
        preprocess_time = (time.perf_counter() - t_pre) * 1000

        # Inference
        t_inf = time.perf_counter()
        output = self.model(tensor)
        if self.device == "cuda":
            torch.cuda.synchronize()
        inference_time = (time.perf_counter() - t_inf) * 1000

        # Postprocess
        t_post = time.perf_counter()
        predictions = self._postprocess(output)
        postprocess_time = (time.perf_counter() - t_post) * 1000

        total_time = (time.perf_counter() - t_start) * 1000

        metrics = LatencyMetrics(
            preprocess_ms=preprocess_time,
            inference_ms=inference_time,
            postprocess_ms=postprocess_time,
            total_ms=total_time,
        )

        return predictions, metrics

    def run_benchmark(
        self,
        input_shape: Tuple[int, ...] = (1, 300, 6),
        batch_sizes: List[int] = [1, 4, 8, 16, 32],
    ) -> Dict[str, Any]:
        """Run comprehensive latency benchmark."""
        results = {}
        seq_len, channels = input_shape[1], input_shape[2]

        for batch_size in batch_sizes:
            print(f"\n  Benchmarking batch_size={batch_size}...")

            # Generate test data
            emulator = IMUSensorEmulator(channels=channels)
            test_data = emulator.generate_batch(seq_len, batch_size)

            # Warmup
            for _ in range(self.warmup_iterations):
                _, _ = self.run_single(test_data)

            # Reset GPU memory stats
            if self.device == "cuda":
                torch.cuda.reset_peak_memory_stats()

            # Benchmark
            preprocess_times = []
            inference_times = []
            postprocess_times = []
            total_times = []

            for _ in range(self.benchmark_iterations):
                _, metrics = self.run_single(test_data)
                preprocess_times.append(metrics.preprocess_ms)
                inference_times.append(metrics.inference_ms)
                postprocess_times.append(metrics.postprocess_ms)
                total_times.append(metrics.total_ms)

            results[f"batch_{batch_size}"] = {
                "preprocess_ms":
                self._calc_stats(preprocess_times),
                "inference_ms":
                self._calc_stats(inference_times),
                "postprocess_ms":
                self._calc_stats(postprocess_times),
                "total_ms":
                self._calc_stats(total_times),
                "throughput_windows_per_sec":
                1000 / np.mean(total_times),
                "throughput_samples_per_sec":
                batch_size * seq_len * 1000 / np.mean(total_times),
            }

        return results

    @staticmethod
    def _calc_stats(times: List[float]) -> Dict[str, float]:
        """Calculate timing statistics."""
        arr = np.array(times)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
        }


# =============================================================================
# Real-Time Streaming Evaluator
# =============================================================================


class RealTimeStreamEvaluator:
    """Evaluates model performance with simulated real-time data streaming."""

    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        sensor_hz: float = 100.0,
        window_size: int = 300,
        window_step: int = 50,
    ):
        self.model = model
        self.device = device
        self.sensor_hz = sensor_hz
        self.window_size = window_size
        self.window_step = window_step

        self.window_duration_sec = window_size / sensor_hz
        self.step_duration_sec = window_step / sensor_hz

    def _preprocess(self, x: np.ndarray) -> torch.Tensor:
        """Preprocess numpy array to tensor."""
        tensor = torch.from_numpy(x).float()
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)
        return tensor.to(self.device)

    @torch.no_grad()
    def run_streaming_evaluation(
        self,
        duration_sec: float = 60.0,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Run real-time streaming evaluation."""
        emulator = IMUSensorEmulator(
            sampling_rate_hz=self.sensor_hz,
            channels=6,
            pattern="sine",
        )

        buffer = deque(maxlen=self.window_size)
        emulator.start_streaming()

        start_time = time.perf_counter()
        last_process_time = start_time
        last_print_time = start_time

        processed_windows = 0
        dropped_windows = 0
        latencies = []

        if verbose:
            print(
                f"\nRunning real-time streaming evaluation for {duration_sec}s..."
            )
            print(f"  Sensor rate: {self.sensor_hz} Hz")
            print(
                f"  Window: {self.window_size} samples ({self.window_duration_sec:.2f}s)"
            )
            print(
                f"  Step: {self.window_step} samples ({self.step_duration_sec:.2f}s)"
            )
            print(
                f"  Required processing time: <{self.step_duration_sec * 1000:.1f}ms per window"
            )

        try:
            while time.perf_counter() - start_time < duration_sec:
                result = emulator.get_samples(self.window_step)

                if result is not None:
                    timestamps, samples = result
                    buffer.extend(samples)

                    if len(buffer) >= self.window_size:
                        window_data = np.array(list(buffer)
                                               )[-self.window_size:]

                        process_interval = time.perf_counter(
                        ) - last_process_time
                        if process_interval > self.step_duration_sec * 1.5:
                            dropped_windows += 1

                        t_start = time.perf_counter()
                        tensor = self._preprocess(window_data)

                        if self.device == "cuda":
                            torch.cuda.synchronize()

                        output = self.model(tensor)

                        if self.device == "cuda":
                            torch.cuda.synchronize()

                        latency_ms = (time.perf_counter() - t_start) * 1000
                        latencies.append(latency_ms)
                        processed_windows += 1

                        last_process_time = time.perf_counter()

                if verbose and time.perf_counter() - last_print_time > 10.0:
                    elapsed = time.perf_counter() - start_time
                    print(
                        f"  [{elapsed:.0f}s] Processed: {processed_windows}, "
                        f"Dropped: {dropped_windows}, "
                        f"Avg latency: {np.mean(latencies):.2f}ms"
                    )
                    last_print_time = time.perf_counter()

                time.sleep(0.001)

        finally:
            emulator.stop_streaming()

        total_time = time.perf_counter() - start_time

        results = {
            "duration_sec":
            total_time,
            "sensor_hz":
            self.sensor_hz,
            "window_size":
            self.window_size,
            "window_step":
            self.window_step,
            "processed_windows":
            processed_windows,
            "dropped_windows":
            dropped_windows,
            "drop_rate_pct":
            dropped_windows / max(1, processed_windows + dropped_windows) *
            100,
            "windows_per_second":
            processed_windows / total_time,
            "real_time_capable":
            dropped_windows == 0,
        }

        if latencies:
            latency_arr = np.array(latencies)
            results["latency_ms"] = {
                "mean": float(np.mean(latency_arr)),
                "std": float(np.std(latency_arr)),
                "min": float(np.min(latency_arr)),
                "max": float(np.max(latency_arr)),
                "p50": float(np.percentile(latency_arr, 50)),
                "p95": float(np.percentile(latency_arr, 95)),
                "p99": float(np.percentile(latency_arr, 99)),
            }
            results["meets_realtime_budget"] = np.percentile(
                latency_arr, 99
            ) < self.step_duration_sec * 1000

        return results


# =============================================================================
# MAML Adaptation Benchmark
# =============================================================================


class MAMLAdaptationBenchmark:
    """Benchmark for MAML few-shot adaptation in real-time deployment scenarios."""

    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        meta_lr: float = 0.01,
    ):
        self.model = model
        self.device = device
        self.meta_lr = meta_lr
        self.has_adabn = hasattr(model,
                                 'adabn_adapt') or hasattr(model, 'use_adabn')

    def _generate_support_set(
        self,
        k_shot: int,
        n_way: int,
        window_size: int,
        channels: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate synthetic support set for adaptation benchmark."""
        n_samples = k_shot * n_way

        emulator = IMUSensorEmulator(channels=channels, pattern="sine")
        data = emulator.generate_batch(window_size, n_samples)

        labels = np.repeat(np.arange(n_way), k_shot)

        x_spt = torch.from_numpy(data).float().to(self.device)
        y_spt = torch.from_numpy(labels).long().to(self.device)

        return x_spt, y_spt

    def benchmark_maml_adaptation(
        self,
        support_data: torch.Tensor,
        support_labels: torch.Tensor,
        n_inner_iter: int = 5,
        first_order: bool = True,
        warmup_iterations: int = 5,
        benchmark_iterations: int = 20,
    ) -> Dict[str, Any]:
        """Benchmark MAML inner-loop adaptation."""
        if not HIGHER_AVAILABLE:
            return {
                "error": "higher library not available for MAML benchmarks"
            }

        latencies = []
        per_step_latencies = []
        memory_peaks = []

        inner_opt = torch.optim.SGD(self.model.parameters(), lr=self.meta_lr)

        # Warmup
        for _ in range(warmup_iterations):
            self.model.train()
            with higher.innerloop_ctx(
                self.model, inner_opt, track_higher_grads=not first_order
            ) as (fnet, diffopt):
                for step in range(n_inner_iter):
                    spt_logits = fnet(support_data)
                    # Handle both (B, T, C) and (B, C, T) output formats
                    if spt_logits.dim() == 3:
                        # Dense prediction: take mean across time dimension
                        spt_logits_flat = spt_logits.mean(dim=1)  # (B, C)
                    else:
                        spt_logits_flat = spt_logits
                    spt_loss = F.cross_entropy(spt_logits_flat, support_labels)
                    if first_order:
                        diffopt.step(
                            spt_loss,
                            grad_callback=lambda grads: [
                                g.detach() if g is not None else None
                                for g in grads
                            ]
                        )
                    else:
                        diffopt.step(spt_loss)

        # Benchmark
        for _ in range(benchmark_iterations):
            if self.device == "cuda":
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()

            self.model.train()
            t_start = time.perf_counter()
            step_times = []

            with higher.innerloop_ctx(
                self.model, inner_opt, track_higher_grads=not first_order
            ) as (fnet, diffopt):
                for step in range(n_inner_iter):
                    step_start = time.perf_counter()

                    spt_logits = fnet(support_data)
                    # Handle both (B, T, C) and (B, C, T) output formats
                    if spt_logits.dim() == 3:
                        spt_logits_flat = spt_logits.mean(dim=1)  # (B, C)
                    else:
                        spt_logits_flat = spt_logits
                    spt_loss = F.cross_entropy(spt_logits_flat, support_labels)

                    if first_order:
                        diffopt.step(
                            spt_loss,
                            grad_callback=lambda grads: [
                                g.detach() if g is not None else None
                                for g in grads
                            ]
                        )
                    else:
                        diffopt.step(spt_loss)

                    if self.device == "cuda":
                        torch.cuda.synchronize()

                    step_times.append(
                        (time.perf_counter() - step_start) * 1000
                    )

            if self.device == "cuda":
                torch.cuda.synchronize()

            total_latency = (time.perf_counter() - t_start) * 1000
            latencies.append(total_latency)
            per_step_latencies.extend(step_times)

            if self.device == "cuda":
                memory_peaks.append(
                    torch.cuda.max_memory_allocated() / (1024**2)
                )

        results = {
            "n_inner_iter": n_inner_iter,
            "first_order": first_order,
            "total_latency_ms": {
                "mean": float(np.mean(latencies)),
                "std": float(np.std(latencies)),
                "min": float(np.min(latencies)),
                "max": float(np.max(latencies)),
            },
            "per_step_latency_ms": {
                "mean": float(np.mean(per_step_latencies)),
                "std": float(np.std(per_step_latencies)),
            },
        }

        if memory_peaks:
            results["memory_peak_mb"] = {
                "mean": float(np.mean(memory_peaks)),
                "max": float(np.max(memory_peaks)),
            }

        return results

    def benchmark_full_adaptation_pipeline(
        self,
        k_shot: int = 5,
        n_way: int = 5,
        window_size: int = 300,
        channels: int = 6,
        adabn_steps: int = 50,
        n_inner_iter: int = 5,
        benchmark_iterations: int = 20,
    ) -> Dict[str, Any]:
        """Benchmark the full adaptation pipeline."""
        # Generate support set
        x_spt, y_spt = self._generate_support_set(
            k_shot, n_way, window_size, channels
        )

        results = {
            "settings": {
                "k_shot": k_shot,
                "n_way": n_way,
                "n_support_samples": k_shot * n_way,
                "window_size": window_size,
                "channels": channels,
                "device": self.device,
            }
        }

        # MAML First-Order
        print("\n  Benchmarking MAML First-Order adaptation...")
        results["maml_first_order"] = self.benchmark_maml_adaptation(
            x_spt,
            y_spt,
            n_inner_iter=n_inner_iter,
            first_order=True,
            benchmark_iterations=benchmark_iterations
        )

        # MAML Second-Order (if higher available)
        if HIGHER_AVAILABLE:
            print("  Benchmarking MAML Second-Order adaptation...")
            results["maml_second_order"] = self.benchmark_maml_adaptation(
                x_spt,
                y_spt,
                n_inner_iter=n_inner_iter,
                first_order=False,
                benchmark_iterations=benchmark_iterations
            )

        return results


# =============================================================================
# Comprehensive Evaluator
# =============================================================================


class RealTimeEvaluator:
    """Comprehensive real-time evaluation suite for any model."""

    def __init__(
        self,
        model_name: str,
        model_path: Optional[str] = None,
        device: str = "auto",
        in_channels: int = 6,
        out_channels: int = 17,
    ):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.model_name = model_name
        self.model = load_model(
            model_name=model_name,
            model_path=model_path,
            device=device,
            in_channels=in_channels,
            out_channels=out_channels,
        )

        self.in_channels = in_channels
        self.out_channels = out_channels

    def evaluate_memory(self) -> MemoryMetrics:
        """Evaluate memory usage."""
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        memory_info = get_memory_usage(self.device)

        dummy_input = torch.randn(1, 300, self.in_channels).to(self.device)
        with torch.no_grad():
            _ = self.model(dummy_input)

        if self.device == "cuda":
            torch.cuda.synchronize()

        peak_memory = get_memory_usage(self.device)

        return MemoryMetrics(
            cpu_current_mb=memory_info["cpu_mb"],
            cpu_peak_mb=peak_memory["cpu_mb"],
            gpu_current_mb=memory_info.get("gpu_mb", 0),
            gpu_peak_mb=peak_memory.get("gpu_peak_mb", 0),
            model_size_mb=get_model_size_mb(self.model),
        )

    def evaluate_compute(self, window_size: int = 300) -> ComputeMetrics:
        """Evaluate computational cost."""
        input_shape = (1, window_size, self.in_channels)
        total_params, trainable_params = count_parameters(self.model)
        flops, macs = estimate_flops_macs(self.model, input_shape, self.device)

        return ComputeMetrics(
            flops=flops,
            macs=macs,
            params=total_params,
            params_trainable=trainable_params,
        )

    def evaluate_latency(
        self,
        window_size: int = 300,
        batch_sizes: List[int] = [1, 4, 8, 16],
        warmup: int = 50,
        iterations: int = 500,
    ) -> Dict[str, Any]:
        """Evaluate inference latency."""
        benchmark = LatencyBenchmark(
            model=self.model,
            device=self.device,
            warmup_iterations=warmup,
            benchmark_iterations=iterations,
        )

        return benchmark.run_benchmark(
            input_shape=(1, window_size, self.in_channels),
            batch_sizes=batch_sizes,
        )

    def evaluate_streaming(
        self,
        sensor_hz: float = 100.0,
        window_size: int = 300,
        window_step: int = 50,
        duration_sec: float = 60.0,
    ) -> Dict[str, Any]:
        """Evaluate real-time streaming capability."""
        evaluator = RealTimeStreamEvaluator(
            model=self.model,
            device=self.device,
            sensor_hz=sensor_hz,
            window_size=window_size,
            window_step=window_step,
        )

        return evaluator.run_streaming_evaluation(
            duration_sec=duration_sec,
            verbose=True,
        )

    def evaluate_adaptation(
        self,
        k_shot: int = 5,
        n_way: int = None,  # Will use out_channels if not specified
        window_size: int = 300,
        n_inner_iter: int = 5,
        adabn_steps: int = 50,
        benchmark_iterations: int = 20,
    ) -> Dict[str, Any]:
        """Evaluate MAML few-shot adaptation performance."""
        # Use out_channels as n_way if not specified (ensures labels match model)
        if n_way is None:
            n_way = self.out_channels

        benchmark = MAMLAdaptationBenchmark(
            model=self.model,
            device=self.device,
        )

        return benchmark.benchmark_full_adaptation_pipeline(
            k_shot=k_shot,
            n_way=n_way,
            window_size=window_size,
            channels=self.in_channels,
            adabn_steps=adabn_steps,
            n_inner_iter=n_inner_iter,
            benchmark_iterations=benchmark_iterations,
        )

    def run_full_evaluation(
        self,
        sensor_hz: float = 100.0,
        window_size: int = 300,
        window_step: int = 50,
        stream_duration: float = 30.0,
        latency_iterations: int = 500,
        batch_sizes: List[int] = [1, 4, 8, 16],
        k_shot: int = 5,
        n_way: int = None,  # Will use out_channels if not specified
        n_inner_iter: int = 5,
        adabn_steps: int = 50,
    ) -> Dict[str, Any]:
        """Run comprehensive real-time evaluation."""
        # Use out_channels as n_way if not specified
        if n_way is None:
            n_way = self.out_channels

        results = {
            "model_name": self.model_name,
            "config": {
                "device": self.device,
                "in_channels": self.in_channels,
                "out_channels": self.out_channels,
                "sensor_hz": sensor_hz,
                "window_size": window_size,
                "window_step": window_step,
            }
        }

        print("\n" + "=" * 60)
        print(f"MODEL: {self.model_name.upper()}")
        print("=" * 60)

        # Memory evaluation
        print("\n1. Evaluating Memory Usage...")
        memory = self.evaluate_memory()
        results["memory"] = memory.summary
        print(f"   Model size: {memory.model_size_mb:.2f} MB")
        print(f"   GPU peak: {memory.gpu_peak_mb:.2f} MB")

        # Compute metrics
        print("\n2. Evaluating Computational Cost...")
        compute = self.evaluate_compute(window_size)
        results["compute"] = compute.summary
        print(f"   Parameters: {compute.summary['params_readable']}")
        print(f"   MACs: {compute.summary['macs_readable']}")

        # Latency benchmark
        print("\n3. Evaluating Inference Latency...")
        results["latency"] = self.evaluate_latency(
            window_size=window_size,
            batch_sizes=batch_sizes,
            iterations=latency_iterations,
        )

        # Print latency summary
        for bs_key, bs_data in results["latency"].items():
            bs = bs_key.split("_")[1]
            print(
                f"   Batch {bs}: {bs_data['total_ms']['mean']:.2f}ms ± {bs_data['total_ms']['std']:.2f}ms"
            )

        # Streaming evaluation
        print("\n4. Evaluating Real-Time Streaming...")
        results["streaming"] = self.evaluate_streaming(
            sensor_hz=sensor_hz,
            window_size=window_size,
            window_step=window_step,
            duration_sec=stream_duration,
        )

        # Adaptation benchmark (MAML)
        if HIGHER_AVAILABLE:
            print("\n5. Evaluating MAML Adaptation...")
            results["adaptation"] = self.evaluate_adaptation(
                k_shot=k_shot,
                n_way=n_way,
                window_size=window_size,
                n_inner_iter=n_inner_iter,
                adabn_steps=adabn_steps,
            )

        # Deployment assessment
        step_budget_ms = (window_step / sensor_hz) * 1000
        batch1_latency = results["latency"].get("batch_1", {}).get(
            "total_ms", {}
        ).get("p99", float('inf'))
        meets_realtime = batch1_latency < step_budget_ms

        results["deployment_assessment"] = {
            "step_budget_ms": step_budget_ms,
            "batch1_p99_latency_ms": batch1_latency,
            "meets_realtime_requirements": meets_realtime,
            "recommended_batch_size": 1 if meets_realtime else "N/A",
            "recommended_device": self.device,
        }

        return results


# =============================================================================
# Report Generation
# =============================================================================


def generate_markdown_report(
    results: Dict[str, Any], output_path: Optional[str] = None
) -> str:
    """Generate a Markdown report from evaluation results."""
    lines = [
        f"# Real-Time Evaluation Report: {results.get('model_name', 'Unknown').upper()}",
        "",
        f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Configuration",
        "",
        "| Parameter | Value |",
        "|-----------|-------|",
    ]

    config = results.get("config", {})
    for k, v in config.items():
        lines.append(f"| {k} | {v} |")

    lines.append("")

    # Memory section
    if "memory" in results:
        mem = results["memory"]
        lines.extend(
            [
                "## Memory Usage",
                "",
                "| Metric | Value (MB) |",
                "|--------|------------|",
                f"| Model Size | {mem['model_size_mb']:.2f} |",
                f"| GPU Current | {mem['gpu_current_mb']:.2f} |",
                f"| GPU Peak | {mem['gpu_peak_mb']:.2f} |",
                f"| CPU Current | {mem['cpu_current_mb']:.2f} |",
                "",
            ]
        )

    # Compute section
    if "compute" in results:
        comp = results["compute"]
        lines.extend(
            [
                "## Computational Cost",
                "",
                "| Metric | Value |",
                "|--------|-------|",
                f"| Parameters | {comp['params_readable']} |",
                f"| FLOPs | {comp['flops_readable']} |",
                f"| MACs | {comp['macs_readable']} |",
                "",
            ]
        )

    # Latency section
    if "latency" in results:
        lines.extend(
            [
                "## Inference Latency",
                "",
                "| Batch Size | Mean (ms) | Std (ms) | P95 (ms) | P99 (ms) | Throughput (win/s) |",
                "|------------|-----------|----------|----------|----------|-------------------|",
            ]
        )
        for bs_key, bs_data in results["latency"].items():
            bs = bs_key.split("_")[1]
            total = bs_data["total_ms"]
            throughput = bs_data["throughput_windows_per_sec"]
            lines.append(
                f"| {bs} | {total['mean']:.2f} | {total['std']:.2f} | {total['p95']:.2f} | {total['p99']:.2f} | {throughput:.1f} |"
            )
        lines.append("")

    # Streaming section
    if "streaming" in results:
        stream = results["streaming"]
        lines.extend(
            [
                "## Real-Time Streaming",
                "",
                "| Metric | Value |",
                "|--------|-------|",
                f"| Duration | {stream['duration_sec']:.1f} s |",
                f"| Processed Windows | {stream['processed_windows']} |",
                f"| Dropped Windows | {stream['dropped_windows']} |",
                f"| Drop Rate | {stream['drop_rate_pct']:.2f}% |",
                f"| Real-Time Capable | {'✅ Yes' if stream['real_time_capable'] else '❌ No'} |",
                "",
            ]
        )

        if "latency_ms" in stream:
            lat = stream["latency_ms"]
            lines.extend(
                [
                    "### Streaming Latency",
                    "",
                    f"- Mean: {lat['mean']:.2f} ms",
                    f"- P95: {lat['p95']:.2f} ms",
                    f"- P99: {lat['p99']:.2f} ms",
                    "",
                ]
            )

    # Adaptation section
    if "adaptation" in results:
        adapt = results["adaptation"]
        lines.extend([
            "## MAML Adaptation",
            "",
        ])

        if "maml_first_order" in adapt:
            fo = adapt["maml_first_order"]
            lines.extend(
                [
                    f"### First-Order MAML ({fo.get('n_inner_iter', 5)} steps)",
                    "",
                    f"- Total Latency: {fo['total_latency_ms']['mean']:.2f} ± {fo['total_latency_ms']['std']:.2f} ms",
                    f"- Per-Step Latency: {fo['per_step_latency_ms']['mean']:.2f} ms",
                    "",
                ]
            )

        if "maml_second_order" in adapt:
            so = adapt["maml_second_order"]
            lines.extend(
                [
                    f"### Second-Order MAML ({so.get('n_inner_iter', 5)} steps)",
                    "",
                    f"- Total Latency: {so['total_latency_ms']['mean']:.2f} ± {so['total_latency_ms']['std']:.2f} ms",
                    f"- Per-Step Latency: {so['per_step_latency_ms']['mean']:.2f} ms",
                    "",
                ]
            )

    # Deployment assessment
    if "deployment_assessment" in results:
        assess = results["deployment_assessment"]
        lines.extend(
            [
                "## Deployment Assessment",
                "",
                f"- **Step Time Budget:** {assess.get('step_budget_ms', 'N/A'):.1f} ms",
                f"- **Batch-1 P99 Latency:** {assess.get('batch1_p99_latency_ms', 'N/A'):.2f} ms",
                f"- **Meets Real-Time Requirements:** {'✅ Yes' if assess.get('meets_realtime_requirements', False) else '❌ No'}",
                "",
            ]
        )

    report = "\n".join(lines)

    if output_path:
        with open(output_path, "w") as f:
            f.write(report)
        print(f"\nReport saved to: {output_path}")

    return report


# =============================================================================
# CLI Interface
# =============================================================================


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Real-Time Evaluation Script for All Models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model settings
    parser.add_argument(
        "--model",
        type=str,
        default="caex",
        choices=["caex", "ex", "unet", "transformer", "segmenter", "cnn"],
        help="Model to evaluate",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to saved model checkpoint",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to run evaluation on",
    )

    # Sensor/data settings
    parser.add_argument(
        "--sensor_hz",
        type=float,
        default=100.0,
        help="Sensor sampling rate in Hz"
    )
    parser.add_argument(
        "--window_size", type=int, default=300, help="Window size in samples"
    )
    parser.add_argument(
        "--window_step", type=int, default=50, help="Window step in samples"
    )
    parser.add_argument(
        "--in_channels", type=int, default=6, help="Number of input channels"
    )
    parser.add_argument(
        "--out_channels",
        type=int,
        default=17,
        help="Number of output classes"
    )

    # Evaluation settings
    parser.add_argument(
        "--mode",
        type=str,
        default="full",
        choices=[
            "full", "latency", "stream", "memory", "compute", "adaptation"
        ],
        help="Evaluation mode",
    )
    parser.add_argument(
        "--stream_duration",
        type=float,
        default=30.0,
        help="Streaming evaluation duration (seconds)"
    )
    parser.add_argument(
        "--latency_iterations",
        type=int,
        default=500,
        help="Iterations for latency benchmark"
    )
    parser.add_argument(
        "--batch_sizes",
        type=str,
        default="1,4,8,16",
        help="Comma-separated batch sizes"
    )

    # MAML Adaptation settings
    parser.add_argument(
        "--k_shot", type=int, default=5, help="K-shot for few-shot adaptation"
    )
    parser.add_argument(
        "--n_way", type=int, default=5, help="N-way for few-shot adaptation"
    )
    parser.add_argument(
        "--n_inner_iter",
        type=int,
        default=5,
        help="MAML inner-loop iterations"
    )
    parser.add_argument(
        "--adabn_steps", type=int, default=50, help="AdaBN adaptation steps"
    )
    parser.add_argument(
        "--meta_lr",
        type=float,
        default=0.01,
        help="MAML inner-loop learning rate"
    )

    # Output settings
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for Markdown report"
    )
    parser.add_argument(
        "--json_output",
        type=str,
        default=None,
        help="Output path for JSON results"
    )

    # Multi-run settings for mean/std
    parser.add_argument(
        "--n_runs",
        type=int,
        default=1,
        help="Number of evaluation runs for mean/std"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 60)
    print("Real-Time Deployment Evaluation")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Model path: {args.model_path or 'None (random init)'}")
    print(f"Sensor rate: {args.sensor_hz} Hz")
    print(f"Window: {args.window_size} samples, step: {args.window_step}")

    # Parse batch sizes
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]

    # Initialize evaluator
    evaluator = RealTimeEvaluator(
        model_name=args.model,
        model_path=args.model_path,
        device=args.device,
        in_channels=args.in_channels,
        out_channels=args.out_channels,
    )

    # Run evaluation based on mode
    if args.mode == "full":
        # Use out_channels as n_way if specified n_way would be invalid
        n_way_to_use = args.n_way if args.n_way <= args.out_channels else None
        results = evaluator.run_full_evaluation(
            sensor_hz=args.sensor_hz,
            window_size=args.window_size,
            window_step=args.window_step,
            stream_duration=args.stream_duration,
            latency_iterations=args.latency_iterations,
            batch_sizes=batch_sizes,
            k_shot=args.k_shot,
            n_way=n_way_to_use,
            n_inner_iter=args.n_inner_iter,
            adabn_steps=args.adabn_steps,
        )
    elif args.mode == "latency":
        results = {
            "model_name":
            args.model,
            "config": {
                "device": evaluator.device
            },
            "latency":
            evaluator.evaluate_latency(
                window_size=args.window_size,
                batch_sizes=batch_sizes,
                iterations=args.latency_iterations,
            ),
        }
    elif args.mode == "stream":
        results = {
            "model_name":
            args.model,
            "config": {
                "device": evaluator.device
            },
            "streaming":
            evaluator.evaluate_streaming(
                sensor_hz=args.sensor_hz,
                window_size=args.window_size,
                window_step=args.window_step,
                duration_sec=args.stream_duration,
            ),
        }
    elif args.mode == "memory":
        memory = evaluator.evaluate_memory()
        results = {
            "model_name": args.model,
            "config": {
                "device": evaluator.device
            },
            "memory": memory.summary,
        }
        print(f"\nMemory Results:")
        for k, v in memory.summary.items():
            print(f"  {k}: {v:.2f} MB")
    elif args.mode == "compute":
        compute = evaluator.evaluate_compute(args.window_size)
        results = {
            "model_name": args.model,
            "config": {
                "device": evaluator.device
            },
            "compute": compute.summary,
        }
        print(f"\nCompute Results:")
        for k, v in compute.summary.items():
            print(f"  {k}: {v}")
    elif args.mode == "adaptation":
        print("\n" + "=" * 60)
        print("MAML FEW-SHOT ADAPTATION BENCHMARK")
        print("=" * 60)
        results = {
            "model_name":
            args.model,
            "config": {
                "device": evaluator.device
            },
            "adaptation":
            evaluator.evaluate_adaptation(
                k_shot=args.k_shot,
                n_way=args.n_way,
                window_size=args.window_size,
                n_inner_iter=args.n_inner_iter,
                adabn_steps=args.adabn_steps,
            ),
        }

    # Custom JSON encoder for numpy types
    class NumpyEncoder(json.JSONEncoder):

        def default(self, obj):
            if isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.bool_, )):
                return bool(obj)
            return super().default(obj)

    # Save JSON results
    if args.json_output:
        os.makedirs(os.path.dirname(args.json_output) or ".", exist_ok=True)
        with open(args.json_output, "w") as f:
            json.dump(results, f, indent=2, cls=NumpyEncoder)
        print(f"\nJSON results saved to: {args.json_output}")

    # Generate report
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        generate_markdown_report(results, args.output)

    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)

    return results


if __name__ == "__main__":
    main()
