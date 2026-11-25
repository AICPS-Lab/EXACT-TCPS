"""
Adaptive Batch Normalization (AdaBN) for Domain Adaptation

AdaBN adapts batch normalization statistics to the target domain by 
running forward passes on target data to update running mean/variance.

Reference: Li et al., "Revisiting Batch Normalization For Practical Domain Adaptation" (2016)

Usage:
    # Before inner-loop adaptation for a task
    adabn_adapt(model, support_data, steps=50)
    
    # Then proceed with gradient-based inner-loop steps
"""

import torch
import torch.nn as nn
from typing import Optional, List


import os
ADABN_FAST = os.environ.get("EXACT_ADABN_EXACT_LOOP", "0") != "1"


def get_bn_layers(model: nn.Module) -> List[nn.BatchNorm1d]:
    """
    Get all BatchNorm1d layers from a model.
    
    Args:
        model: PyTorch model
    
    Returns:
        List of BatchNorm1d modules
    """
    bn_layers = []
    for module in model.modules():
        if isinstance(
            module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
        ):
            bn_layers.append(module)
    return bn_layers


def reset_bn_stats(model: nn.Module):
    """
    Reset running statistics of all BatchNorm layers.
    
    Args:
        model: PyTorch model
    """
    for module in model.modules():
        if isinstance(
            module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
        ):
            module.running_mean.zero_()
            module.running_var.fill_(1)
            module.num_batches_tracked.zero_()


def set_bn_momentum(model: nn.Module, momentum: float):
    """
    Set momentum for all BatchNorm layers.
    
    Args:
        model: PyTorch model
        momentum: New momentum value
    """
    for module in model.modules():
        if isinstance(
            module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
        ):
            module.momentum = momentum


def adabn_adapt(
    model: nn.Module,
    support_data: torch.Tensor,
    steps: int = 50,
    reset_stats: bool = True,
    momentum: Optional[float] = None,
) -> None:
    """
    Perform AdaBN adaptation by running forward passes on support data.
    
    This updates the running mean/variance of BatchNorm layers to adapt
    to the target domain statistics without gradient updates.
    
    Args:
        model: PyTorch model with BatchNorm layers
        support_data: Support set data tensor of shape (N, T, C) or (N, C, T)
        steps: Number of forward passes for adaptation
        reset_stats: Whether to reset BN stats before adaptation
        momentum: Optional custom momentum for BN layers during adaptation
    """
    # Store original training mode and momentum
    was_training = model.training
    original_momentums = {}

    bn_layers = get_bn_layers(model)
    if not bn_layers:
        return  # No BN layers to adapt

    # Store original momentums
    for i, bn in enumerate(bn_layers):
        original_momentums[i] = bn.momentum

    # Reset running stats if requested
    if reset_stats:
        reset_bn_stats(model)

    # Set custom momentum if provided
    if momentum is not None:
        set_bn_momentum(model, momentum)

    # Set model to train mode (needed for BN to update running stats)
    model.train()

    # Run forward passes without computing gradients
    with torch.no_grad():
        if ADABN_FAST and steps > 1:
            # Exact shortcut. In train mode BatchNorm normalises every layer with
            # the *batch* statistics, so each pass over the same support windows
            # (in any order) sees identical inputs and produces identical batch
            # stats. `steps` momentum updates toward the same target collapse to
            # ONE update with momentum 1-(1-m)^steps; momentum=None (cumulative
            # average after reset) already equals the batch stats after one pass.
            # Set EXACT_ADABN_EXACT_LOOP=1 to run the original loop instead.
            for bn in bn_layers:
                m = bn.momentum
                bn.momentum = 1.0 if m is None else 1.0 - (1.0 - m) ** steps
            _ = model(support_data)
        else:
            for step in range(steps):
                # If we have enough data, use different subsets each step
                if support_data.size(0) > 1:
                    indices = torch.randperm(support_data.size(0))
                    batch = support_data[indices]
                else:
                    batch = support_data
                _ = model(batch)

    # Restore original momentums
    for i, bn in enumerate(bn_layers):
        bn.momentum = original_momentums[i]

    # Restore original training mode
    if not was_training:
        model.eval()


def adabn_adapt_episodic(
    model: nn.Module,
    support_images: torch.Tensor,
    steps: int = 50,
    batch_size: Optional[int] = None,
) -> None:
    """
    AdaBN adaptation for episodic meta-learning.
    
    Adapts BN statistics using the support set before inner-loop gradient steps.
    
    Args:
        model: PyTorch model (or functional model from higher)
        support_images: Support set tensor of shape (batch_size, T, C)
        steps: Number of forward passes
        batch_size: Optional batch size for mini-batches during adaptation
    """
    # Flatten batch dimensions if needed
    if support_images.dim() > 3:
        original_shape = support_images.shape
        support_images = support_images.view(-1, *original_shape[-2:])

    if batch_size is None:
        batch_size = support_images.size(0)

    model.train()
    reset_bn_stats(model)

    with torch.no_grad():
        for step in range(steps):
            # Sample mini-batch with replacement
            if support_images.size(0) > batch_size:
                indices = torch.randint(
                    0, support_images.size(0), (batch_size, )
                )
                batch = support_images[indices]
            else:
                batch = support_images

            _ = model(batch)


class AdaBNWrapper:
    """
    Wrapper class for convenient AdaBN usage in training/evaluation loops.
    
    Example:
        adabn = AdaBNWrapper(model, steps=50)
        
        # During evaluation
        for task in tasks:
            x_spt, y_spt, x_qry, y_qry = task
            adabn.adapt(x_spt)
            # Now proceed with inner-loop adaptation
    """

    def __init__(
        self,
        model: nn.Module,
        steps: int = 50,
        reset_stats: bool = True,
        momentum: Optional[float] = None,
    ):
        """
        Args:
            model: PyTorch model
            steps: Number of forward passes for adaptation
            reset_stats: Whether to reset BN stats before each adaptation
            momentum: Optional custom momentum
        """
        self.model = model
        self.steps = steps
        self.reset_stats = reset_stats
        self.momentum = momentum

    def adapt(self, support_data: torch.Tensor):
        """
        Adapt BN statistics to support data.
        
        Args:
            support_data: Support set tensor
        """
        adabn_adapt(
            self.model,
            support_data,
            steps=self.steps,
            reset_stats=self.reset_stats,
            momentum=self.momentum,
        )

    def __call__(self, support_data: torch.Tensor):
        """Alias for adapt()."""
        self.adapt(support_data)


def functional_adabn_adapt(
    fnet,  # higher.patch.MonkeyPatched model
    support_data: torch.Tensor,
    steps: int = 50,
) -> None:
    """
    AdaBN adaptation for functional models created by higher.
    
    Note: This works with the patched model from higher.innerloop_ctx.
    The BN running stats are updated in the underlying model.
    
    Args:
        fnet: Functional model from higher.innerloop_ctx
        support_data: Support set tensor
        steps: Number of forward passes
    """
    # Access the underlying module
    if hasattr(fnet, '_fast_params'):
        # higher patched model - get base module
        base_model = fnet.module if hasattr(fnet, 'module') else fnet
    else:
        base_model = fnet

    # Set to train mode for BN stat updates
    fnet.train()

    with torch.no_grad():
        for step in range(steps):
            if support_data.size(0) > 1:
                indices = torch.randperm(support_data.size(0))
                batch = support_data[indices]
            else:
                batch = support_data

            _ = fnet(batch)
