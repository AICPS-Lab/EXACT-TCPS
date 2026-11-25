"""
CAEX: Channel-Aware EX

Extends EX with:
1. Channel-aware gating (SE-style) to learn which IMU channels matter
2. Acc vs Gyro group gating to weight sensor modalities
3. EXP-style features: on-the-fly normalization, AdaBN, augmentation transforms

The augmentation defaults are *softer* than EXP to preserve high accuracy on
original (arm) exercises while still improving generalization to new domains (leg).

Usage:
    python main_meta_v2.py --model caex --dataset physiq
    
    # Tune augmentation strength (lower = preserve arm accuracy)
    python main_meta_v2.py --model caex --caex_rotation_chance 0.3 --caex_rotation_max_deg 45
"""

from argtypes import str2bool
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from methods.EX import EX
from methods.EXP import (
    group_zscore_normalize_batch,
    channel_zscore_normalize_batch,
    EXP_DEFAULTS,
)
from methods.adabn import adabn_adapt

# =============================================================================
# CAEX Default Configuration
# =============================================================================

CAEX_DEFAULTS = {
    # Channel gating
    "gate": "se",  # 'se' or 'none'
    "gate_reduction": 2,
    "gate_softmax": True,
    "group_gate": True,  # acc vs gyro gating

    # IMU Normalization (from EXP)
    "imu_normalize": True,
    "imu_norm_kind": "group_zscore",
    "imu_norm_skip_noise": True,

    # Rotation Augmentation - SOFTER defaults than EXP for arm accuracy
    "rotation_chance": 0.5,  # EXP uses 0.8
    "rotation_mode": "axis_angle",
    "rotation_max_deg": 90.0,  # EXP uses 180

    # Augmentation mode
    "aug_mode": "independent",

    # AdaBN
    "use_adabn": True,
    "adabn_steps": 50,
}

# =============================================================================
# Channel Gating Modules
# =============================================================================


class ChannelSEGate(nn.Module):
    """Squeeze-excitation style gate over channels for IMU sequences.

    Learns per-window channel importance weights.
    Input: (B, T, C)
    Output: gated (B, T, C)
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 2,
        use_softmax: bool = True,
    ):
        super().__init__()
        hidden = max(4, channels // max(1, reduction))
        self.use_softmax = use_softmax
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)
        self.last_weights = None  # for debugging/visualization

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        pooled = x.abs().mean(dim=1)  # (B, C) - channel energy
        h = F.relu(self.fc1(pooled))
        logits = self.fc2(h)  # (B, C)

        if self.use_softmax:
            # Softmax normalized, scaled so mean weight ≈ 1
            w = torch.softmax(logits, dim=-1) * x.size(-1)
        else:
            # Sigmoid in (0, 2) range
            w = torch.sigmoid(logits) * 2.0

        self.last_weights = w.detach()
        return x * w.unsqueeze(1)


class AccGyroGroupGate(nn.Module):
    """Gate between accelerometer (channels 0-2) and gyroscope (channels 3-5).

    Learns to weight acc vs gyro based on their signal characteristics.
    """

    def __init__(self, use_softmax: bool = True):
        super().__init__()
        self.use_softmax = use_softmax
        self.fc1 = nn.Linear(2, 8)
        self.fc2 = nn.Linear(8, 2)
        self.last_weights = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 6)
        acc = x[:, :, :3]
        gyro = x[:, :, 3:6]

        # Energy proxy per sensor group
        acc_e = acc.abs().mean(dim=(1, 2))  # (B,)
        gyro_e = gyro.abs().mean(dim=(1, 2))  # (B,)
        pooled = torch.stack([acc_e, gyro_e], dim=-1)  # (B, 2)

        h = F.relu(self.fc1(pooled))
        logits = self.fc2(h)  # (B, 2)

        if self.use_softmax:
            w = torch.softmax(logits, dim=-1) * 2.0  # mean weight ≈ 1
        else:
            w = torch.sigmoid(logits) * 2.0

        self.last_weights = w.detach()

        acc_w = w[:, 0].view(-1, 1, 1)
        gyro_w = w[:, 1].view(-1, 1, 1)

        out = x.clone()
        out[:, :, :3] = acc * acc_w
        out[:, :, 3:6] = gyro * gyro_w
        return out


# =============================================================================
# CAEX Model
# =============================================================================


class CAEX(EX):
    """CAEX: Channel-Aware EX.

    Inherits EX architecture and adds:
    - Channel-wise SE gating
    - Acc/Gyro group gating
    - On-the-fly IMU normalization (from EXP)
    - AdaBN support for test-time adaptation (from EXP)
    - Configurable augmentation transforms (softer than EXP by default)
    """

    @staticmethod
    def add_args(parser):
        # Base EX args
        parser.add_argument("--embed_dim", type=int, default=32)
        parser.add_argument("--num_heads", type=int, default=4)

        # CAEX channel gating
        parser.add_argument(
            "--caex_gate",
            type=str,
            default=CAEX_DEFAULTS["gate"],
            choices=["se", "none"],
            help=
            "Channel gating type: 'se' for squeeze-excitation, 'none' to disable",
        )
        parser.add_argument(
            "--caex_gate_reduction",
            type=int,
            default=CAEX_DEFAULTS["gate_reduction"],
            help="Reduction ratio for channel gate MLP",
        )
        parser.add_argument(
            "--caex_gate_softmax",
            type=str2bool,
            default=CAEX_DEFAULTS["gate_softmax"],
            help="Use softmax-normalized gates (recommended)",
        )
        parser.add_argument(
            "--caex_group_gate",
            type=str2bool,
            default=CAEX_DEFAULTS["group_gate"],
            help="Enable acc vs gyro group gating (requires C=6)",
        )

        # IMU Normalization (from EXP)
        parser.add_argument(
            "--caex_imu_normalize",
            type=str2bool,
            default=CAEX_DEFAULTS["imu_normalize"],
            help="Enable on-the-fly IMU normalization",
        )
        parser.add_argument(
            "--caex_imu_norm_kind",
            type=str,
            default=CAEX_DEFAULTS["imu_norm_kind"],
            choices=["group_zscore", "channel_zscore", "minmax"],
            help="Type of IMU normalization",
        )
        parser.add_argument(
            "--caex_imu_norm_skip_noise",
            type=str2bool,
            default=CAEX_DEFAULTS["imu_norm_skip_noise"],
            help="Skip normalization on pure-noise windows",
        )

        # Augmentation (softer defaults than EXP)
        parser.add_argument(
            "--caex_rotation_chance",
            type=float,
            default=CAEX_DEFAULTS["rotation_chance"],
            help=
            "Probability of rotation augmentation (lower = preserve arm accuracy)",
        )
        parser.add_argument(
            "--caex_rotation_mode",
            type=str,
            default=CAEX_DEFAULTS["rotation_mode"],
            choices=["euler", "axis_angle", "quaternion"],
            help="Rotation augmentation mode",
        )
        parser.add_argument(
            "--caex_rotation_max_deg",
            type=float,
            default=CAEX_DEFAULTS["rotation_max_deg"],
            help=
            "Max rotation angle in degrees (lower = preserve arm accuracy)",
        )
        parser.add_argument(
            "--caex_aug_mode",
            type=str,
            default=CAEX_DEFAULTS["aug_mode"],
            choices=["single", "independent"],
            help="Augmentation application mode",
        )

        # AdaBN
        parser.add_argument(
            "--caex_use_adabn",
            type=str2bool,
            default=CAEX_DEFAULTS["use_adabn"],
            help="Enable AdaBN at test-time",
        )
        parser.add_argument(
            "--caex_adabn_steps",
            type=int,
            default=CAEX_DEFAULTS["adabn_steps"],
            help="Number of forward passes for AdaBN adaptation",
        )

        return parser

    @staticmethod
    def get_augmentation_transforms(args):
        """Get augmentation transforms for CAEX (softer than EXP by default)."""
        from datasets.Transforms import (
            RotationAugmentation,
            ComposedAugmentation,
        )

        augmentations = []

        rotation_chance = getattr(
            args, 'caex_rotation_chance', CAEX_DEFAULTS['rotation_chance']
        )
        if rotation_chance > 0:
            augmentations.append(
                RotationAugmentation(
                    rotation_chance=rotation_chance,
                    rotation_mode=getattr(
                        args, 'caex_rotation_mode',
                        CAEX_DEFAULTS['rotation_mode']
                    ),
                    rotation_max_deg=getattr(
                        args, 'caex_rotation_max_deg',
                        CAEX_DEFAULTS['rotation_max_deg']
                    ),
                )
            )

        if not augmentations:
            return None

        aug_mode = getattr(args, 'caex_aug_mode', CAEX_DEFAULTS['aug_mode'])
        return ComposedAugmentation(augmentations, mode=aug_mode)

    def __init__(self, args):
        # Initialize parent EX
        super().__init__(args)

        in_channels = args.in_channels

        # Store CAEX-specific config
        self.imu_normalize = getattr(
            args, 'caex_imu_normalize', CAEX_DEFAULTS['imu_normalize']
        )
        self.imu_norm_kind = getattr(
            args, 'caex_imu_norm_kind', CAEX_DEFAULTS['imu_norm_kind']
        )
        self.imu_norm_skip_noise = getattr(
            args, 'caex_imu_norm_skip_noise',
            CAEX_DEFAULTS['imu_norm_skip_noise']
        )
        self.use_adabn = getattr(
            args, 'caex_use_adabn', CAEX_DEFAULTS['use_adabn']
        )
        self.adabn_steps = getattr(
            args, 'caex_adabn_steps', CAEX_DEFAULTS['adabn_steps']
        )

        # Channel gating modules
        gate_kind = getattr(args, "caex_gate", CAEX_DEFAULTS["gate"])
        gate_reduction = getattr(
            args, "caex_gate_reduction", CAEX_DEFAULTS["gate_reduction"]
        )
        gate_softmax = getattr(
            args, "caex_gate_softmax", CAEX_DEFAULTS["gate_softmax"]
        )
        group_gate = getattr(
            args, "caex_group_gate", CAEX_DEFAULTS["group_gate"]
        )

        self.channel_gate = (
            ChannelSEGate(
                channels=in_channels,
                reduction=gate_reduction,
                use_softmax=gate_softmax,
            ) if gate_kind == "se" else None
        )

        self.group_gate = (
            AccGyroGroupGate(use_softmax=gate_softmax) if
            (group_gate and in_channels == 6) else None
        )

    def normalize_input(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply on-the-fly IMU normalization (from EXP)."""
        if not self.imu_normalize:
            return x

        skip_noise = self.imu_norm_skip_noise

        if self.imu_norm_kind == "group_zscore":
            return group_zscore_normalize_batch(
                x, labels, skip_pure_noise=skip_noise
            )
        elif self.imu_norm_kind == "channel_zscore":
            return channel_zscore_normalize_batch(
                x, labels, skip_pure_noise=skip_noise
            )
        elif self.imu_norm_kind == "minmax":
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
        self,
        support_data: torch.Tensor,
        steps: Optional[int] = None,
    ):
        """Perform AdaBN adaptation using support data (from EXP)."""
        if not self.use_adabn:
            return

        if steps is None:
            steps = self.adabn_steps

        # Normalize support data first
        support_data = self.normalize_input(support_data)

        # Run AdaBN adaptation
        adabn_adapt(self, support_data, steps=steps, reset_stats=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with normalization and channel gating.

        Args:
            x: Input tensor (B, T, C) where C=6 for 6-axis IMU

        Returns:
            Output tensor (B, T, out_channels)
        """
        # 1. On-the-fly normalization (from EXP)
        x = self.normalize_input(x)

        # 2. Channel gating (CAEX-specific)
        if self.group_gate is not None:
            x = self.group_gate(x)
        if self.channel_gate is not None:
            x = self.channel_gate(x)

        # 3. EX encoder-decoder (parent class)
        x = x.permute(0, 2, 1)  # [B, C, T]

        # Encoder
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        e4 = self.encoder4(self.pool(e3))

        # Self-Attention in Bottleneck
        e4 = self.mhsa(e4)

        # Decoder with Skip Connections
        d4 = self.upconv4(e4)
        d4 = self._crop_and_concat(d4, e3)
        d4 = self.decoder4(d4)

        d3 = self.upconv3(d4)
        d3 = self._crop_and_concat(d3, e2)
        d3 = self.decoder3(d3)

        d2 = self.upconv2(d3)
        d2 = self._crop_and_concat(d2, e1)
        d2 = self.decoder2(d2)

        d1 = self.decoder1(d2)
        return d1.permute(0, 2, 1)  # [B, T, out_channels]

    def get_gate_weights(self):
        """Return the last computed gate weights for debugging/visualization."""
        weights = {}
        if self.channel_gate is not None and self.channel_gate.last_weights is not None:
            weights['channel'] = self.channel_gate.last_weights
        if self.group_gate is not None and self.group_gate.last_weights is not None:
            weights['group'] = self.group_gate.last_weights
        return weights


# =============================================================================
# Utility Functions
# =============================================================================


def is_caex_model(args) -> bool:
    """Check if the model is CAEX."""
    return getattr(args, 'model', '').lower() == 'caex'


def get_caex_config(args) -> dict:
    """Get the CAEX configuration as a dictionary."""
    return {
        # Gating
        "gate":
        getattr(args, 'caex_gate', CAEX_DEFAULTS['gate']),
        "gate_reduction":
        getattr(args, 'caex_gate_reduction', CAEX_DEFAULTS['gate_reduction']),
        "gate_softmax":
        getattr(args, 'caex_gate_softmax', CAEX_DEFAULTS['gate_softmax']),
        "group_gate":
        getattr(args, 'caex_group_gate', CAEX_DEFAULTS['group_gate']),
        # Normalization
        "imu_normalize":
        getattr(args, 'caex_imu_normalize', CAEX_DEFAULTS['imu_normalize']),
        "imu_norm_kind":
        getattr(args, 'caex_imu_norm_kind', CAEX_DEFAULTS['imu_norm_kind']),
        # Augmentation
        "rotation_chance":
        getattr(
            args, 'caex_rotation_chance', CAEX_DEFAULTS['rotation_chance']
        ),
        "rotation_mode":
        getattr(args, 'caex_rotation_mode', CAEX_DEFAULTS['rotation_mode']),
        "rotation_max_deg":
        getattr(
            args, 'caex_rotation_max_deg', CAEX_DEFAULTS['rotation_max_deg']
        ),
        # AdaBN
        "use_adabn":
        getattr(args, 'caex_use_adabn', CAEX_DEFAULTS['use_adabn']),
        "adabn_steps":
        getattr(args, 'caex_adabn_steps', CAEX_DEFAULTS['adabn_steps']),
    }


def print_caex_config(args):
    """Print the CAEX configuration."""
    config = get_caex_config(args)
    print("\n" + "=" * 50)
    print("CAEX Configuration")
    print("=" * 50)
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("=" * 50 + "\n")
