import random
from typing import Optional, List, Tuple

import numpy as np
import torch
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R

# =============================================================================
# IMU Normalization Utilities
# =============================================================================


def group_zscore_normalize(
    imu_data: np.ndarray,
    skip_noise_indices: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Apply group z-score normalization to IMU data.
    
    Normalizes accelerometer (channels 0-2) and gyroscope (channels 3-5) separately.
    
    Args:
        imu_data: IMU data of shape (T, 6) or (B, T, 6)
        skip_noise_indices: Optional boolean mask of shape (T,) indicating noise samples to skip
                           during normalization stat computation
    
    Returns:
        Normalized IMU data with same shape
    """
    original_shape = imu_data.shape
    if imu_data.ndim == 2:
        imu_data = imu_data[np.newaxis, ...]  # (1, T, 6)

    normalized = np.zeros_like(imu_data, dtype=np.float32)

    for b in range(imu_data.shape[0]):
        sample = imu_data[b]  # (T, 6)

        if skip_noise_indices is not None and skip_noise_indices.any():
            # Use only non-noise samples for computing stats
            valid_mask = ~skip_noise_indices
            if valid_mask.sum() > 1:
                stats_sample = sample[valid_mask]
            else:
                stats_sample = sample
        else:
            stats_sample = sample

        # Accelerometer normalization (channels 0-2)
        acc_mean = stats_sample[:, :3].mean()
        acc_std = stats_sample[:, :3].std() + 1e-8
        normalized[b, :, :3] = (sample[:, :3] - acc_mean) / acc_std

        # Gyroscope normalization (channels 3-5)
        gyro_mean = stats_sample[:, 3:6].mean()
        gyro_std = stats_sample[:, 3:6].std() + 1e-8
        normalized[b, :, 3:6] = (sample[:, 3:6] - gyro_mean) / gyro_std

    if len(original_shape) == 2:
        normalized = normalized[0]

    return normalized


def channel_zscore_normalize(
    imu_data: np.ndarray,
    skip_noise_indices: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Apply per-channel z-score normalization to IMU data.
    
    Args:
        imu_data: IMU data of shape (T, 6) or (B, T, 6)
        skip_noise_indices: Optional boolean mask of shape (T,) indicating noise samples to skip
    
    Returns:
        Normalized IMU data with same shape
    """
    original_shape = imu_data.shape
    if imu_data.ndim == 2:
        imu_data = imu_data[np.newaxis, ...]

    normalized = np.zeros_like(imu_data, dtype=np.float32)

    for b in range(imu_data.shape[0]):
        sample = imu_data[b]

        if skip_noise_indices is not None and skip_noise_indices.any():
            valid_mask = ~skip_noise_indices
            if valid_mask.sum() > 1:
                stats_sample = sample[valid_mask]
            else:
                stats_sample = sample
        else:
            stats_sample = sample

        for c in range(sample.shape[1]):
            mean = stats_sample[:, c].mean()
            std = stats_sample[:, c].std() + 1e-8
            normalized[b, :, c] = (sample[:, c] - mean) / std

    if len(original_shape) == 2:
        normalized = normalized[0]

    return normalized


class IMUNormalizer:
    """
    On-the-fly IMU normalization transform.
    
    Supports multiple normalization modes:
    - 'group_zscore': Normalize acc and gyro groups separately
    - 'channel_zscore': Normalize each channel independently
    - 'minmax': Min-max normalization per channel
    """

    def __init__(
        self,
        norm_kind: str = "group_zscore",
        skip_noise: bool = False,
    ):
        """
        Args:
            norm_kind: Type of normalization ('group_zscore', 'channel_zscore', 'minmax')
            skip_noise: If True, skip noise samples when computing normalization stats
        """
        self.norm_kind = norm_kind
        self.skip_noise = skip_noise
        self._noise_mask = None

    def set_noise_mask(self, mask: np.ndarray):
        """Set the noise mask for skipping noise during normalization."""
        self._noise_mask = mask

    def __call__(self, imu_data, noise_mask: Optional[np.ndarray] = None):
        """
        Apply normalization to IMU data.
        
        Args:
            imu_data: torch.Tensor or np.ndarray of shape (T, 6)
            noise_mask: Optional boolean mask indicating noise samples
        
        Returns:
            Normalized IMU data as torch.Tensor
        """
        if isinstance(imu_data, torch.Tensor):
            imu_np = imu_data.detach().cpu().numpy().astype(np.float32)
        elif isinstance(imu_data, np.ndarray):
            imu_np = imu_data.astype(np.float32, copy=True)
        else:
            raise TypeError("imu_data must be torch.Tensor or np.ndarray")

        skip_indices = None
        if self.skip_noise and noise_mask is not None:
            skip_indices = noise_mask
        elif self.skip_noise and self._noise_mask is not None:
            skip_indices = self._noise_mask

        if self.norm_kind == "group_zscore":
            normalized = group_zscore_normalize(imu_np, skip_indices)
        elif self.norm_kind == "channel_zscore":
            normalized = channel_zscore_normalize(imu_np, skip_indices)
        elif self.norm_kind == "minmax":
            # Min-max normalization per channel
            for c in range(imu_np.shape[-1]):
                cmin, cmax = imu_np[..., c].min(), imu_np[..., c].max()
                if cmax - cmin > 1e-8:
                    imu_np[..., c] = (imu_np[..., c] - cmin) / (cmax - cmin)
            normalized = imu_np
        else:
            raise ValueError(f"Unknown norm_kind: {self.norm_kind}")

        return torch.tensor(normalized, dtype=torch.float32)


# =============================================================================
# Rotation Augmentation
# =============================================================================


class RotationAugmentation:
    """
    Enhanced rotation augmentation for IMU data with multiple rotation modes.
    
    Supports:
    - 'euler': Random Euler angles (original behavior, limited range)
    - 'axis_angle': Random axis-angle rotation (full SO(3) coverage)
    - 'quaternion': Random quaternion sampling (uniform on SO(3))
    """

    def __init__(
        self,
        rotation_chance: float = 0.5,
        rotation_mode: str = "euler",
        rotation_max_deg: float = 45.0,
    ):
        """
        Args:
            rotation_chance: Probability of applying rotation
            rotation_mode: 'euler', 'axis_angle', or 'quaternion'
            rotation_max_deg: Maximum rotation angle in degrees (for euler/axis_angle)
        """
        self.rotation_chance = rotation_chance
        self.rotation_mode = rotation_mode
        self.rotation_max_deg = rotation_max_deg

    def _random_euler_rotation(self) -> R:
        """Generate rotation from random Euler angles."""
        angles = np.random.uniform(
            -self.rotation_max_deg, self.rotation_max_deg, size=3
        )
        return R.from_euler("xyz", angles, degrees=True)

    def _random_axis_angle_rotation(self) -> R:
        """Generate rotation from random axis-angle representation."""
        # Random unit axis
        axis = np.random.randn(3)
        axis = axis / (np.linalg.norm(axis) + 1e-8)
        # Random angle up to max_deg
        angle = np.random.uniform(0, self.rotation_max_deg) * np.pi / 180.0
        rotvec = axis * angle
        return R.from_rotvec(rotvec)

    def _random_quaternion_rotation(self) -> R:
        """Generate uniformly random rotation using quaternion sampling."""
        # Uniform quaternion sampling on SO(3)
        u1, u2, u3 = np.random.uniform(0, 1, 3)
        q = np.array(
            [
                np.sqrt(1 - u1) * np.sin(2 * np.pi * u2),
                np.sqrt(1 - u1) * np.cos(2 * np.pi * u2),
                np.sqrt(u1) * np.sin(2 * np.pi * u3),
                np.sqrt(u1) * np.cos(2 * np.pi * u3),
            ]
        )
        return R.from_quat(q)

    def get_rotation(self) -> R:
        """Get a random rotation based on the configured mode."""
        if self.rotation_mode == "euler":
            return self._random_euler_rotation()
        elif self.rotation_mode == "axis_angle":
            return self._random_axis_angle_rotation()
        elif self.rotation_mode == "quaternion":
            return self._random_quaternion_rotation()
        else:
            raise ValueError(f"Unknown rotation_mode: {self.rotation_mode}")

    def apply_rotation(self, imu_data_np: np.ndarray) -> np.ndarray:
        """Apply rotation to IMU data (acc and gyro)."""
        rotation = self.get_rotation()
        rotated_data = np.zeros_like(imu_data_np, dtype=np.float32)

        # Apply same rotation to all timesteps
        rotated_data[:, :3] = rotation.apply(imu_data_np[:, :3])
        rotated_data[:, 3:6] = rotation.apply(imu_data_np[:, 3:6])

        return rotated_data

    def __call__(self, imu_data):
        """
        Apply rotation augmentation.
        
        Args:
            imu_data: torch.Tensor or np.ndarray of shape (T, 6)
        
        Returns:
            Augmented data as torch.Tensor
        """
        if random.random() >= self.rotation_chance:
            if isinstance(imu_data, torch.Tensor):
                return imu_data.to(torch.float32)
            return torch.tensor(imu_data, dtype=torch.float32)

        if isinstance(imu_data, torch.Tensor):
            imu_np = imu_data.detach().cpu().numpy().astype(np.float32)
        elif isinstance(imu_data, np.ndarray):
            imu_np = imu_data.astype(np.float32, copy=True)
        else:
            raise TypeError("imu_data must be torch.Tensor or np.ndarray")

        rotated = self.apply_rotation(imu_np)
        return torch.tensor(rotated, dtype=torch.float32)


# =============================================================================
# Leg Domain Randomization (for cross-body-part transfer)
# =============================================================================


class LegDomainRandomization:
    """
    Placement-informed IMU domain randomization.
    
    Emulates systematic differences from different IMU mounting locations
    (e.g., leg-mounted vs arm-mounted) without using target data.
    
    Applies:
    - Axis remapping (sensor coordinate convention changes)
    - Sign flips (180° rotations around axes)
    - Bias and bias drift
    - Scale and scale drift
    - Band-limited noise
    
    Expects windows shaped (T, 6): acc(x,y,z), gyro(x,y,z).
    """

    def __init__(
        self,
        prob: float = 0.5,
        axis_remap_prob: float = 0.3,
        sign_flip_prob: float = 0.3,
        bias_std: float = 0.05,
        bias_drift_std: float = 0.02,
        scale_std: float = 0.05,
        scale_drift_std: float = 0.02,
        noise_std: float = 0.02,
        noise_band_low: float = 0.0,
        noise_band_high: float = 0.5,
    ):
        self.prob = float(prob)
        self.axis_remap_prob = float(axis_remap_prob)
        self.sign_flip_prob = float(sign_flip_prob)
        self.bias_std = float(bias_std)
        self.bias_drift_std = float(bias_drift_std)
        self.scale_std = float(scale_std)
        self.scale_drift_std = float(scale_drift_std)
        self.noise_std = float(noise_std)
        self.noise_band_low = float(noise_band_low)
        self.noise_band_high = float(noise_band_high)

        if not (0.0 <= self.noise_band_low <= self.noise_band_high <= 0.5):
            raise ValueError(
                "noise_band_low/high must satisfy 0.0 <= low <= high <= 0.5"
            )

    def _slow_sine(self, T: int, C: int, amp_std: float) -> np.ndarray:
        if amp_std <= 0:
            return np.zeros((T, C), dtype=np.float32)
        t = np.linspace(0.0, 1.0, T, dtype=np.float32)
        out = np.zeros((T, C), dtype=np.float32)
        for c in range(C):
            amp = np.random.normal(0.0, amp_std)
            freq = np.random.uniform(0.25, 1.5)  # cycles per window
            phase = np.random.uniform(0.0, 2.0 * np.pi)
            out[:, c] = amp * np.sin(2.0 * np.pi * freq * t + phase)
        return out

    def _band_limited_noise(self, T: int, C: int, std: float) -> np.ndarray:
        if std <= 0:
            return np.zeros((T, C), dtype=np.float32)
        noise = np.random.normal(0.0, std, size=(T, C)).astype(np.float32)
        if self.noise_band_low <= 0.0 and self.noise_band_high >= 0.5:
            return noise

        fft = np.fft.rfft(noise, axis=0)
        freqs = np.fft.rfftfreq(T, d=1.0)
        if freqs.size <= 1:
            return noise
        freqs = freqs / freqs.max()  # normalize to [0, 0.5]
        mask = (freqs >= self.noise_band_low) & (freqs <= self.noise_band_high)
        fft[~mask, :] = 0
        return np.fft.irfft(fft, n=T, axis=0).astype(np.float32)

    def __call__(self, imu_data):
        if random.random() >= self.prob:
            if isinstance(imu_data, torch.Tensor):
                return imu_data.to(torch.float32)
            if isinstance(imu_data, np.ndarray):
                return torch.tensor(imu_data, dtype=torch.float32)
            raise TypeError("imu_data must be a torch.Tensor or np.ndarray")

        if isinstance(imu_data, torch.Tensor):
            x = imu_data.detach().cpu().numpy().astype(np.float32)
        elif isinstance(imu_data, np.ndarray):
            x = imu_data.astype(np.float32, copy=True)
        else:
            raise TypeError("imu_data must be a torch.Tensor or np.ndarray")

        if x.ndim != 2 or x.shape[-1] != 6:
            raise ValueError(
                f"Expected IMU window shape (T, 6), got {tuple(x.shape)}"
            )

        T = int(x.shape[0])

        # Axis remap (same for acc and gyro to mimic sensor axis conventions)
        if random.random() < self.axis_remap_prob:
            perm = np.random.permutation(3)
            x[:, 0:3] = x[:, perm]
            x[:, 3:6] = x[:, 3 + perm]

        # Sign flips (equivalent to 180deg rotations around axes)
        if random.random() < self.sign_flip_prob:
            signs = np.random.choice([-1.0, 1.0],
                                     size=(3, )).astype(np.float32)
            x[:, 0:3] = x[:, 0:3] * signs
            x[:, 3:6] = x[:, 3:6] * signs

        # Constant bias + slow bias drift
        if self.bias_std > 0:
            x = x + np.random.normal(0.0, self.bias_std,
                                     size=(1, 6)).astype(np.float32)
        if self.bias_drift_std > 0 and T > 1:
            x = x + self._slow_sine(T, 6, self.bias_drift_std)

        # Per-channel scaling + slow scale drift
        if self.scale_std > 0:
            scales = np.random.normal(1.0, self.scale_std,
                                      size=(1, 6)).astype(np.float32)
            x = x * scales
        if self.scale_drift_std > 0 and T > 1:
            drift = self._slow_sine(T, 6, self.scale_drift_std)
            x = x * (1.0 + drift)

        # Band-limited noise (different hardware / mounting vibration)
        if self.noise_std > 0 and T > 2:
            x = x + self._band_limited_noise(T, 6, self.noise_std)

        return torch.tensor(x, dtype=torch.float32)


# =============================================================================
# Composed Augmentation Pipeline
# =============================================================================


class ComposedAugmentation:
    """
    Compose multiple augmentations together.
    
    Supports two application modes:
    - 'single': Apply one randomly selected augmentation
    - 'independent': Apply each augmentation independently with its own probability
    """

    def __init__(
        self,
        augmentations: List,
        mode: str = "independent",
    ):
        """
        Args:
            augmentations: List of augmentation callables
            mode: 'single' or 'independent'
        """
        self.augmentations = augmentations
        self.mode = mode

    def __call__(self, imu_data):
        if isinstance(imu_data, np.ndarray):
            result = torch.tensor(imu_data, dtype=torch.float32)
        elif isinstance(imu_data, torch.Tensor):
            result = imu_data.to(torch.float32)
        else:
            raise TypeError("imu_data must be torch.Tensor or np.ndarray")

        if self.mode == "single":
            # Apply one randomly selected augmentation
            if self.augmentations:
                aug = random.choice(self.augmentations)
                result = aug(result)
        elif self.mode == "independent":
            # Apply each augmentation independently
            for aug in self.augmentations:
                result = aug(result)
        else:
            raise ValueError(f"Unknown augmentation mode: {self.mode}")

        return result


# =============================================================================
# Original IMUAugmentation (preserved for backward compatibility)
# =============================================================================


class IMUAugmentation:

    def __init__(
        self,
        rotation_chance=0.5,
        jitter_chance=0.5,
        scaling_chance=0.5,
        spline_chance=0.5,
    ):
        self.rotation_chance = rotation_chance
        self.jitter_chance = jitter_chance
        self.scaling_chance = scaling_chance
        self.spline_chance = spline_chance

    def random_quaternion(self):
        """Generate a random unit quaternion."""
        rand_nums = np.random.uniform(0, 45, size=3)
        q = R.from_euler("xyz", rand_nums, degrees=True).as_quat()
        return q

    def rotate_vector(self, v, q):
        """Rotate a vector using a quaternion."""
        rotation = R.from_quat(q)
        return rotation.apply(v)

    def apply_rotation(self, imu_data_np):
        """Apply rotation to IMU data."""
        qS_prime_S = self.random_quaternion()
        rotated_data = np.zeros(imu_data_np.shape)
        for i in range(imu_data_np.shape[0]):
            acc_data = imu_data_np[i, :3]
            gyro_data = imu_data_np[i, 3:6]
            rotated_data[i, :3] = self.rotate_vector(acc_data, qS_prime_S)
            rotated_data[i, 3:6] = self.rotate_vector(gyro_data, qS_prime_S)
        return rotated_data

    def apply_jitter(self, imu_data_np, sigma=0.01):
        """Add random jitter to IMU data."""
        jitter = np.random.normal(0, sigma, imu_data_np.shape)
        return imu_data_np + jitter

    def apply_scaling(self, imu_data_np, scale_range=(0.9, 1.1)):
        """Scale IMU data by a random factor."""
        scale_factor = np.random.uniform(*scale_range)
        return imu_data_np * scale_factor

    def apply_spline(self, imu_data_np):
        """Apply cubic spline interpolation to smooth the IMU data."""
        time_steps = np.arange(imu_data_np.shape[0])
        augmented_data = np.zeros_like(imu_data_np)
        for i in range(imu_data_np.shape[1]):
            cs = CubicSpline(time_steps, imu_data_np[:, i])
            augmented_data[:, i] = cs(time_steps)
        return augmented_data

    def __call__(self, imu_data):
        """Apply a random augmentation to 6-axis IMU data."""
        if isinstance(imu_data, torch.Tensor):
            imu_data_np = imu_data.numpy()
        elif isinstance(imu_data, np.ndarray):
            imu_data_np = imu_data
        else:
            raise TypeError("imu_data must be a torch.Tensor or np.ndarray")

        # random_number = random.random()
        if random.random() < self.rotation_chance:
            imu_data_np = self.apply_rotation(imu_data_np)
        elif random.random() < self.jitter_chance:
            imu_data_np = self.apply_jitter(imu_data_np)
        elif random.random() < self.scaling_chance:
            imu_data_np = self.apply_scaling(imu_data_np)
        elif random.random() < self.spline_chance:
            imu_data_np = self.apply_spline(imu_data_np)

        return torch.tensor(imu_data_np)
