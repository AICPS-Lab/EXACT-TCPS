"""
EX+ (EXP) - Refined EX model for cross-dataset stability

EX+ extends the EX architecture with built-in:
- On-the-fly IMU normalization (group_zscore)
- Stronger rotation augmentation (axis_angle, up to 180°)
- AdaBN (Adaptive Batch Normalization) at test-time
- LegDomainRandomization for domain shift robustness

Usage:
    python main_meta_v2.py --model exp --dataset physiq --add_side_noise
    
    # All EX+ specific features are automatically enabled
    # You can still override defaults:
    python main_meta_v2.py --model exp --exp_adabn_steps 100 --exp_rotation_max_deg 90
"""

from argtypes import str2bool
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional

from methods.EX import (
    ConvBlock1D,
    ResidualConvBlock1D,
    MultiHeadSelfAttention1D,
)
from methods.adabn import adabn_adapt, reset_bn_stats

# =============================================================================
# EX+ Default Configuration
# =============================================================================

EXP_DEFAULTS = {
    # IMU Normalization
    "imu_normalize": True,
    "imu_norm_kind":
    "group_zscore",  # 'group_zscore', 'channel_zscore', 'minmax'
    "imu_norm_skip_noise": True,

    # Rotation Augmentation
    "rotation_chance": 0.8,
    "rotation_mode": "axis_angle",  # 'euler', 'axis_angle', 'quaternion'
    "rotation_max_deg": 180.0,

    # Augmentation Mode
    "aug_mode": "independent",  # 'single', 'independent'

    # AdaBN
    "use_adabn": True,
    "adabn_steps": 50,

    # LegDomainRandomization
    "use_leg_domain_rand": False,  # Optional, disabled by default
    "leg_domain_rand_prob": 0.5,
}

# =============================================================================
# IMU Normalization (inline for self-contained model)
# =============================================================================


def group_zscore_normalize_batch(
    x: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
    skip_pure_noise: bool = True,
) -> torch.Tensor:
    """
    Apply group z-score normalization to batched IMU data.
    Normalizes accelerometer (channels 0-2) and gyroscope (channels 3-5) separately,
    each with one mean/std over (T x 3). Channels >= 6 are passed through.

    Vectorised over the batch (was a per-sample Python loop); identical values.

    Args:
        x: IMU data of shape (B, T, C>=6)
        labels: Optional labels of shape (B, T) - if provided and skip_pure_noise=True,
                windows where all labels are 0 (pure noise) will not be normalized
        skip_pure_noise: If True, skip normalization for pure-noise windows
    Returns:
        Normalized IMU data with same shape
    """
    B, T, C = x.shape
    if labels is not None and skip_pure_noise:
        has_signal = (labels != 0).any(dim=1)  # (B,)
    else:
        has_signal = torch.ones(B, dtype=torch.bool, device=x.device)

    acc, gyro = x[:, :, :3], x[:, :, 3:6]
    acc_flat, gyro_flat = acc.reshape(B, -1), gyro.reshape(B, -1)
    acc_mean = acc_flat.mean(dim=1).view(B, 1, 1)
    acc_std = acc_flat.std(dim=1).view(B, 1, 1) + 1e-8
    gyro_mean = gyro_flat.mean(dim=1).view(B, 1, 1)
    gyro_std = gyro_flat.std(dim=1).view(B, 1, 1) + 1e-8
    parts = [(acc - acc_mean) / acc_std, (gyro - gyro_mean) / gyro_std]
    if C > 6:
        parts.append(x[:, :, 6:])
    out = torch.cat(parts, dim=2)
    return torch.where(has_signal.view(B, 1, 1), out, x)


def channel_zscore_normalize_batch(
    x: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
    skip_pure_noise: bool = True,
) -> torch.Tensor:
    """
    Apply per-channel z-score normalization to batched IMU data.
    
    Args:
        x: IMU data of shape (B, T, 6)
        labels: Optional labels of shape (B, T) - if provided and skip_pure_noise=True,
                windows where all labels are 0 (pure noise) will not be normalized
        skip_pure_noise: If True, skip normalization for pure-noise windows
    
    Returns:
        Normalized IMU data with same shape
    """
    # x shape: (B, T, 6)
    B = x.shape[0]
    normalized = x.clone()

    # Determine which samples to normalize
    if labels is not None and skip_pure_noise:
        has_signal = (labels != 0).any(dim=1)  # (B,)
    else:
        has_signal = torch.ones(B, dtype=torch.bool, device=x.device)

    for b in range(B):
        if not has_signal[b]:
            continue
        mean = x[b].mean(dim=0, keepdim=True)  # (1, 6)
        std = x[b].std(dim=0, keepdim=True) + 1e-8  # (1, 6)
        normalized[b] = (x[b] - mean) / std

    return normalized


# =============================================================================
# EX+ Model
# =============================================================================


class EXP(nn.Module):
    """
    EX+ (EXP) - Refined EX model for improved cross-dataset generalization.
    
    Combines the EX architecture with:
    - On-the-fly IMU normalization
    - AdaBN support for test-time adaptation
    - Built-in augmentation configuration
    
    The model automatically applies normalization during forward pass.
    AdaBN should be called explicitly before inner-loop adaptation.
    """

    @staticmethod
    def add_args(parser):
        """Add EX+ specific arguments to the parser."""
        # Base EX args
        parser.add_argument("--embed_dim", type=int, default=32)
        parser.add_argument("--num_heads", type=int, default=4)

        # EX+ specific args (with exp_ prefix to avoid conflicts)
        parser.add_argument(
            "--exp_imu_normalize",
            type=str2bool,
            default=EXP_DEFAULTS["imu_normalize"],
            help="Enable on-the-fly IMU normalization"
        )
        parser.add_argument(
            "--exp_imu_norm_kind",
            type=str,
            default=EXP_DEFAULTS["imu_norm_kind"],
            choices=["group_zscore", "channel_zscore", "minmax"],
            help="Type of IMU normalization"
        )
        parser.add_argument(
            "--exp_rotation_chance",
            type=float,
            default=EXP_DEFAULTS["rotation_chance"],
            help="Probability of applying rotation augmentation"
        )
        parser.add_argument(
            "--exp_rotation_mode",
            type=str,
            default=EXP_DEFAULTS["rotation_mode"],
            choices=["euler", "axis_angle", "quaternion"],
            help="Rotation augmentation mode"
        )
        parser.add_argument(
            "--exp_rotation_max_deg",
            type=float,
            default=EXP_DEFAULTS["rotation_max_deg"],
            help="Maximum rotation angle in degrees"
        )
        parser.add_argument(
            "--exp_aug_mode",
            type=str,
            default=EXP_DEFAULTS["aug_mode"],
            choices=["single", "independent"],
            help="Augmentation application mode"
        )
        parser.add_argument(
            "--exp_use_adabn",
            type=str2bool,
            default=EXP_DEFAULTS["use_adabn"],
            help="Enable AdaBN at test-time"
        )
        parser.add_argument(
            "--exp_adabn_steps",
            type=int,
            default=EXP_DEFAULTS["adabn_steps"],
            help="Number of forward passes for AdaBN adaptation"
        )
        parser.add_argument(
            "--exp_use_leg_domain_rand",
            type=str2bool,
            default=EXP_DEFAULTS["use_leg_domain_rand"],
            help="Enable LegDomainRandomization augmentation"
        )
        parser.add_argument(
            "--exp_imu_norm_skip_noise",
            type=str2bool,
            default=EXP_DEFAULTS["imu_norm_skip_noise"],
            help="Skip normalization on pure-noise windows (all labels=0)"
        )

        return parser

    @staticmethod
    def get_augmentation_transforms(args):
        """
        Get the augmentation transforms configured for EX+.
        
        This is called by the dataset/sampler to apply augmentations.
        
        Returns:
            Callable transform or None
        """
        from datasets.Transforms import (
            RotationAugmentation,
            LegDomainRandomization,
            ComposedAugmentation,
        )

        augmentations = []

        # Rotation augmentation
        rotation_chance = getattr(
            args, 'exp_rotation_chance', EXP_DEFAULTS['rotation_chance']
        )
        if rotation_chance > 0:
            augmentations.append(
                RotationAugmentation(
                    rotation_chance=rotation_chance,
                    rotation_mode=getattr(
                        args, 'exp_rotation_mode',
                        EXP_DEFAULTS['rotation_mode']
                    ),
                    rotation_max_deg=getattr(
                        args, 'exp_rotation_max_deg',
                        EXP_DEFAULTS['rotation_max_deg']
                    ),
                )
            )

        # LegDomainRandomization (optional)
        if getattr(
            args, 'exp_use_leg_domain_rand',
            EXP_DEFAULTS['use_leg_domain_rand']
        ):
            augmentations.append(
                LegDomainRandomization(
                    prob=getattr(
                        args, 'leg_domain_rand_prob',
                        EXP_DEFAULTS['leg_domain_rand_prob']
                    ),
                )
            )

        if not augmentations:
            return None

        aug_mode = getattr(args, 'exp_aug_mode', EXP_DEFAULTS['aug_mode'])
        return ComposedAugmentation(augmentations, mode=aug_mode)

    def __init__(self, args):
        super(EXP, self).__init__()

        # Store configuration
        self.imu_normalize = getattr(
            args, 'exp_imu_normalize', EXP_DEFAULTS['imu_normalize']
        )
        self.imu_norm_kind = getattr(
            args, 'exp_imu_norm_kind', EXP_DEFAULTS['imu_norm_kind']
        )
        self.imu_norm_skip_noise = getattr(
            args, 'exp_imu_norm_skip_noise',
            EXP_DEFAULTS['imu_norm_skip_noise']
        )
        self.use_adabn = getattr(
            args, 'exp_use_adabn', EXP_DEFAULTS['use_adabn']
        )
        self.adabn_steps = getattr(
            args, 'exp_adabn_steps', EXP_DEFAULTS['adabn_steps']
        )

        # Architecture params
        embed_dim = args.embed_dim
        num_heads = args.num_heads
        in_channels = args.in_channels
        out_channels = args.out_channels

        self.embed_dim = embed_dim

        # Encoder with Residual Connections
        self.encoder1 = ResidualConvBlock1D(in_channels, embed_dim)
        self.encoder2 = ResidualConvBlock1D(embed_dim, embed_dim * 2)
        self.encoder3 = ResidualConvBlock1D(embed_dim * 2, embed_dim * 4)
        self.encoder4 = ResidualConvBlock1D(embed_dim * 4, embed_dim)

        self.pool = nn.MaxPool1d(2, 2)

        # Self-Attention in Bottleneck
        self.mhsa = MultiHeadSelfAttention1D(embed_dim, num_heads)

        # Decoder with Residual Connections
        self.upconv4 = nn.ConvTranspose1d(
            embed_dim, embed_dim, kernel_size=2, stride=2
        )
        self.decoder4 = ResidualConvBlock1D(
            embed_dim + embed_dim * 4, embed_dim * 4
        )

        self.upconv3 = nn.ConvTranspose1d(
            embed_dim * 4, embed_dim * 4, kernel_size=2, stride=2
        )
        self.decoder3 = ResidualConvBlock1D(
            embed_dim * 4 + embed_dim * 2, embed_dim * 2
        )

        self.upconv2 = nn.ConvTranspose1d(
            embed_dim * 2, embed_dim * 2, kernel_size=2, stride=2
        )
        self.decoder2 = ResidualConvBlock1D(
            embed_dim * 2 + embed_dim, embed_dim
        )

        self.decoder1 = nn.Conv1d(embed_dim, out_channels, kernel_size=1)

    def normalize_input(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply on-the-fly IMU normalization.
        
        Args:
            x: Input tensor of shape (B, T, C)
            labels: Optional labels of shape (B, T) - used to skip normalization
                    on pure-noise windows (all labels == 0)
        
        Returns:
            Normalized tensor
        """
        if not self.imu_normalize:
            return x

        skip_noise = getattr(self, 'imu_norm_skip_noise', True)

        if self.imu_norm_kind == "group_zscore":
            return group_zscore_normalize_batch(
                x, labels, skip_pure_noise=skip_noise
            )
        elif self.imu_norm_kind == "channel_zscore":
            return channel_zscore_normalize_batch(
                x, labels, skip_pure_noise=skip_noise
            )
        elif self.imu_norm_kind == "minmax":
            # Min-max per sample (with optional skip for pure noise)
            B = x.shape[0]
            normalized = x.clone()
            if labels is not None and skip_noise:
                has_signal = (labels != 0).any(dim=1)
            else:
                has_signal = torch.ones(B, dtype=torch.bool, device=x.device)

            for b in range(B):
                if not has_signal[b]:
                    continue
                x_min = x[b].min(dim=0, keepdim=True)[0]
                x_max = x[b].max(dim=0, keepdim=True)[0]
                normalized[b] = (x[b] - x_min) / (x_max - x_min + 1e-8)
            return normalized
        else:
            return x

    def adabn_adapt(
        self, support_data: torch.Tensor, steps: Optional[int] = None
    ):
        """
        Perform AdaBN adaptation using support data.
        
        Call this before inner-loop gradient adaptation.
        
        Args:
            support_data: Support set tensor of shape (N, T, C)
            steps: Number of forward passes (defaults to self.adabn_steps)
        """
        if not self.use_adabn:
            return

        if steps is None:
            steps = self.adabn_steps

        # Normalize support data first
        support_data = self.normalize_input(support_data)

        # Run AdaBN adaptation
        adabn_adapt(self, support_data, steps=steps, reset_stats=True)

    def forward(self, x):
        """
        Forward pass with automatic IMU normalization.
        
        Args:
            x: Input tensor of shape (B, T, C) where C=6 for 6-axis IMU
        
        Returns:
            Output tensor of shape (B, T, out_channels)
        """
        # Apply on-the-fly normalization
        x = self.normalize_input(x)

        x = x.permute(0, 2, 1)  # [B, in_channels, S]

        # Encoder
        e1 = self.encoder1(x)  # [B, embed_dim, S]
        e2 = self.encoder2(self.pool(e1))  # [B, embed_dim * 2, S/2]
        e3 = self.encoder3(self.pool(e2))  # [B, embed_dim * 4, S/4]
        e4 = self.encoder4(self.pool(e3))  # [B, embed_dim, S/8]

        # Self-Attention in Bottleneck
        e4 = self.mhsa(e4)  # [B, embed_dim, S/8]

        # Decoder with Skip Connections
        d4 = self.upconv4(e4)  # Upsample
        d4 = self._crop_and_concat(d4, e3)  # Match size and concat with e3
        d4 = self.decoder4(d4)  # [B, embed_dim * 4, S/4]

        d3 = self.upconv3(d4)  # Upsample
        d3 = self._crop_and_concat(d3, e2)  # Match size and concat with e2
        d3 = self.decoder3(d3)  # [B, embed_dim * 2, S/2]

        d2 = self.upconv2(d3)  # Upsample
        d2 = self._crop_and_concat(d2, e1)  # Match size and concat with e1
        d2 = self.decoder2(d2)  # [B, embed_dim, S]

        # Final output layer
        d1 = self.decoder1(d2)  # [B, out_channels, S]
        d1 = d1.permute(0, 2, 1)  # [B, S, out_channels]
        return d1

    def _crop_and_concat(self, upsampled, bypass):
        """Crop or pad the upsampled tensor to match the size of the bypass tensor."""
        diff = upsampled.size(-1) - bypass.size(-1)
        if diff == 0:
            return torch.cat([upsampled, bypass], dim=1)
        elif diff > 0:
            upsampled = upsampled[..., :bypass.size(-1)]
        else:
            upsampled = F.pad(upsampled, (0, -diff))
        return torch.cat([upsampled, bypass], dim=1)


# =============================================================================
# Utility Functions for EX+ Integration
# =============================================================================


def is_exp_model(args) -> bool:
    """Check if the model is EX+ (EXP)."""
    return getattr(args, 'model', '').lower() == 'exp'


def get_exp_config(args) -> dict:
    """Get the EX+ configuration as a dictionary."""
    return {
        "imu_normalize":
        getattr(args, 'exp_imu_normalize', EXP_DEFAULTS['imu_normalize']),
        "imu_norm_kind":
        getattr(args, 'exp_imu_norm_kind', EXP_DEFAULTS['imu_norm_kind']),
        "rotation_chance":
        getattr(args, 'exp_rotation_chance', EXP_DEFAULTS['rotation_chance']),
        "rotation_mode":
        getattr(args, 'exp_rotation_mode', EXP_DEFAULTS['rotation_mode']),
        "rotation_max_deg":
        getattr(
            args, 'exp_rotation_max_deg', EXP_DEFAULTS['rotation_max_deg']
        ),
        "aug_mode":
        getattr(args, 'exp_aug_mode', EXP_DEFAULTS['aug_mode']),
        "use_adabn":
        getattr(args, 'exp_use_adabn', EXP_DEFAULTS['use_adabn']),
        "adabn_steps":
        getattr(args, 'exp_adabn_steps', EXP_DEFAULTS['adabn_steps']),
        "use_leg_domain_rand":
        getattr(
            args, 'exp_use_leg_domain_rand',
            EXP_DEFAULTS['use_leg_domain_rand']
        ),
    }


def print_exp_config(args):
    """Print the EX+ configuration."""
    config = get_exp_config(args)
    print("\n" + "=" * 50)
    print("EX+ (EXP) Configuration")
    print("=" * 50)
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("=" * 50 + "\n")
