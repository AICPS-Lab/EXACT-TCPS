"""
t-SNE Visualization of Domain Adaptation

Creates a side-by-side visualization showing:
- (Left) Baseline model (EX): disjoint clusters for Source (PhysiQ) and Target (PhyTMo)
- (Right) EXACT+ (CAEX): aligned domain distributions

Usage:
    python visualize_tsne_domain.py --baseline_model ex --exact_model caex --seed 42
    
    # With specific artifact versions
    python visualize_tsne_domain.py --baseline_model ex --exact_model caex \
        --baseline_seed 43 --exact_seed 42 --exact_version v1
"""

from argtypes import str2bool
import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde
import seaborn as sns

import wandb

from datasets.PhysiQ import PhysiQ
from datasets.PHYTMO import PHYTMO
from datasets.DenseLabelTaskSampler import DenseLabelTaskSampler
from until_argparser import get_model
from utilities import seed as set_seed


def get_feature_extractor_hook(model):
    """
    Create a hook to extract features from the bottleneck/embedding layer.
    
    For different architectures:
    - EX/CAEX: After MHSA bottleneck (before decoder)
    - UNet: conv3 (deepest encoder layer)
    - CNN: conv_net output (before final_conv)
    - Segmenter: Last transformer layer output
    - Transformer: transformer_encoder output (before decoder)
    """
    features = {}

    def get_features(name):

        def hook(module, input, output):
            # Store the output features
            if isinstance(output, tuple):
                features[name] = output[0].detach()
            else:
                features[name] = output.detach()

        return hook

    # Register hooks based on model architecture
    net = model.net if hasattr(model, 'net') else model

    # Try to find the bottleneck layer
    hook_registered = False

    # For EX/CAEX models - hook into MHSA output
    if hasattr(net, 'mhsa'):
        net.mhsa.register_forward_hook(get_features('bottleneck'))
        hook_registered = True
        print("  Hooked into MHSA bottleneck")

    # For UNet - hook into conv3 (bottleneck)
    elif hasattr(net, 'conv3'):
        net.conv3.register_forward_hook(get_features('bottleneck'))
        hook_registered = True
        print("  Hooked into UNet conv3 (bottleneck)")

    # For CNN - hook into conv_net (before final_conv)
    elif hasattr(net, 'conv_net'):
        net.conv_net.register_forward_hook(get_features('bottleneck'))
        hook_registered = True
        print("  Hooked into CNN conv_net")

    # For Segmenter - hook into the last transformer layer
    elif hasattr(net, 'layers') and hasattr(net, 'cls_emb'):
        net.layers[-1].register_forward_hook(get_features('bottleneck'))
        hook_registered = True
        print("  Hooked into Segmenter layers[-1]")

    # For Transformer - hook into transformer_encoder
    elif hasattr(net, 'transformer_encoder'):
        net.transformer_encoder.register_forward_hook(
            get_features('bottleneck')
        )
        hook_registered = True
        print("  Hooked into Transformer encoder")

    # Fallback: try to hook into any 'bottleneck' or 'mhsa' attribute
    if not hook_registered:
        for name, module in net.named_modules():
            if 'bottleneck' in name.lower() or 'mhsa' in name.lower():
                module.register_forward_hook(get_features('bottleneck'))
                hook_registered = True
                print(f"  Hooked into {name}")
                break

    if not hook_registered:
        print("  Warning: Could not find bottleneck layer, using model output")

    return features


def sample_data_once(dataloader, args, n_samples=500):
    """
    Sample data once from a dataloader to ensure consistent samples across models.
    
    Returns:
        List of input tensors
    """
    all_inputs = []

    for batch_idx, batch in enumerate(dataloader):
        if len(all_inputs) * args.batch_size >= n_samples:
            break

        # Handle different batch formats
        if len(batch) == 5:  # Meta-learning format
            x_spt, y_spt, x_qry, y_qry, _ = batch
            # Use query set for visualization
            x = x_qry.reshape(-1, *x_qry.shape[2:])
        elif len(batch) == 2:  # Standard format
            x, y = batch
        else:
            continue

        all_inputs.append(x)

    # Concatenate and limit to n_samples
    all_inputs = torch.cat(all_inputs, dim=0)[:n_samples]
    return all_inputs


def extract_features_from_inputs(model, inputs, args):
    """
    Extract features from a model using pre-sampled inputs.
    
    Args:
        model: The model to extract features from
        inputs: Pre-sampled input tensor
        args: Arguments namespace
        
    Returns:
        features: numpy array of shape (n_samples, feature_dim)
    """
    model.eval()
    features_dict = get_feature_extractor_hook(model)

    all_features = []

    with torch.no_grad():
        # Process in batches to avoid memory issues
        batch_size = args.batch_size
        for i in range(0, len(inputs), batch_size):
            x = inputs[i:i + batch_size].to(args.device)

            # Forward pass to trigger hooks
            _ = model(x)

            # Get features
            if 'bottleneck' in features_dict:
                feat = features_dict['bottleneck']
                # Global average pooling if needed
                if len(feat.shape) == 3:  # (B, T, C)
                    feat = feat.mean(dim=1)
                elif len(feat.shape) == 4:  # (B, C, H, W)
                    feat = feat.mean(dim=(2, 3))
            else:
                # Use model output as fallback
                output = model(x)
                feat = output.mean(dim=-1
                                   ) if len(output.shape) == 3 else output

            all_features.append(feat.cpu().numpy())

    features = np.concatenate(all_features, axis=0)
    return features


def extract_features(model, dataloader, args, n_samples=500):
    """
    Extract features from a model for visualization.
    
    Args:
        model: The model to extract features from
        dataloader: DataLoader for the dataset
        args: Arguments namespace
        n_samples: Maximum number of samples to extract
        
    Returns:
        features: numpy array of shape (n_samples, feature_dim)
        labels: numpy array of class labels
    """
    model.eval()
    features_dict = get_feature_extractor_hook(model)

    all_features = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if len(all_features) * args.batch_size >= n_samples:
                break

            # Handle different batch formats
            if len(batch) == 5:  # Meta-learning format
                x_spt, y_spt, x_qry, y_qry, _ = batch
                # Use query set for visualization
                x = x_qry.reshape(-1, *x_qry.shape[2:])
                y = y_qry.reshape(-1, *y_qry.shape[2:])
            elif len(batch) == 2:  # Standard format
                x, y = batch
            else:
                continue

            x = x.to(args.device)

            # Forward pass to trigger hooks
            _ = model(x)

            # Get features
            if 'bottleneck' in features_dict:
                feat = features_dict['bottleneck']
                # Global average pooling if needed
                if len(feat.shape) == 3:  # (B, T, C)
                    feat = feat.mean(dim=1)
                elif len(feat.shape) == 4:  # (B, C, H, W)
                    feat = feat.mean(dim=(2, 3))
            else:
                # Use model output as fallback
                output = model(x)
                feat = output.mean(dim=-1
                                   ) if len(output.shape) == 3 else output

            all_features.append(feat.cpu().numpy())

            # Get labels (use mode for segmentation labels)
            if len(y.shape) > 1:
                y_labels = y.mode(dim=-1
                                  )[0] if y.shape[-1] > 1 else y.squeeze()
            else:
                y_labels = y
            all_labels.append(y_labels.cpu().numpy())

    features = np.concatenate(all_features, axis=0)[:n_samples]
    labels = np.concatenate(all_labels, axis=0).flatten()[:n_samples]

    return features, labels


def load_model_from_wandb(args, model_type, seed_val, version="v0"):
    """Load a model from W&B artifacts."""
    run_name = f"{model_type}-physiq-{seed_val}"

    if args.wandb_entity:
        artifact_path = f"{args.wandb_entity}/{args.wandb_project}/{run_name}:{version}"
    else:
        artifact_path = f"{args.wandb_project}/{run_name}:{version}"

    try:
        api = wandb.Api()
        print(f"Downloading artifact: {artifact_path}")
        artifact = api.artifact(artifact_path, type="model")
        artifact_dir = artifact.download()

        pth_files = [f for f in os.listdir(artifact_dir) if f.endswith('.pth')]
        if pth_files:
            model_path = os.path.join(artifact_dir, pth_files[0])
        else:
            model_path = os.path.join(artifact_dir, f"{run_name}.pth")

        print(f"Loaded: {model_path}")
        return model_path

    except Exception as e:
        print(f"W&B download failed: {e}")
        # Try local fallback
        local_paths = [
            f"artifacts/{run_name}:{version}/{run_name}.pth",
            f"saved_model/{run_name}.pth",
        ]
        for path in local_paths:
            if os.path.exists(path):
                print(f"Using local: {path}")
                return path
        raise FileNotFoundError(f"Model not found: {run_name}")


def create_single_tsne_plot(
    features_source,
    features_target,
    tsne_result,
    n_source,
    model_name,
    save_path,
    is_exact=False,
    feature_dim=None
):
    """
    Create a single t-SNE plot for one model.
    
    Args:
        features_source: Source domain features
        features_target: Target domain features
        tsne_result: Pre-computed t-SNE embedding
        n_source: Number of source samples
        feature_dim: Feature dimension for filename
        model_name: Name for the file
        save_path: Directory to save
        is_exact: If True, add legend inside plot
    """
    # Colors
    source_color = '#2196F3'  # Blue
    target_color = '#FF5722'  # Orange/Red

    # Create figure - single plot
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    # Get source and target data
    source_data = tsne_result[:n_source]
    target_data = tsne_result[n_source:]

    # Draw KDE contours
    sns.kdeplot(
        x=source_data[:, 0],
        y=source_data[:, 1],
        ax=ax,
        thresh=0.1,
        levels=2,
        color=source_color,
        alpha=0.3,
        fill=True
    )

    sns.kdeplot(
        x=target_data[:, 0],
        y=target_data[:, 1],
        ax=ax,
        thresh=0.1,
        levels=2,
        color=target_color,
        alpha=0.3,
        fill=True
    )

    # Scatter points on top
    ax.scatter(
        source_data[:, 0],
        source_data[:, 1],
        c=source_color,
        alpha=0.7,
        s=15,
        edgecolors='white',
        linewidths=0.3,
        label='Source (PhysiQ)',
        zorder=10
    )
    ax.scatter(
        target_data[:, 0],
        target_data[:, 1],
        c=target_color,
        alpha=0.7,
        s=15,
        edgecolors='white',
        linewidths=0.3,
        label='Target (PhyTMo)',
        zorder=10
    )

    # No title - will be added in LaTeX
    ax.set_xlabel('t-SNE Dimension 1', fontsize=16, fontweight='bold')
    ax.set_ylabel('t-SNE Dimension 2', fontsize=16, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])

    # Only add legend for EXACT+
    if is_exact:
        legend_elements = [
            Line2D(
                [0], [0],
                marker='o',
                color='w',
                markerfacecolor=source_color,
                markersize=12,
                label='Source (PhysiQ)'
            ),
            Line2D(
                [0], [0],
                marker='o',
                color='w',
                markerfacecolor=target_color,
                markersize=12,
                label='Target (PhyTMo)'
            )
        ]
        ax.legend(
            handles=legend_elements,
            loc='upper right',
            fontsize=12,
            framealpha=0.9
        )

    plt.tight_layout()

    # Save figure
    if feature_dim is not None:
        output_file = os.path.join(
            save_path, f"tsne_{model_name}_{feature_dim}.png"
        )
    else:
        output_file = os.path.join(save_path, f"tsne_{model_name}.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")

    plt.close()


def create_tsne_visualization(
    baseline_features_source,
    baseline_features_target,
    exact_features_source,
    exact_features_target,
    baseline_model_name="ex",
    exact_model_name="caex",
    save_path="saved_figure"
):
    """
    Create separate t-SNE visualizations for each model.
    
    Baseline: No legend
    EXACT+: Legend inside plot
    """
    # Combine features for each model
    baseline_all = np.concatenate(
        [baseline_features_source, baseline_features_target]
    )
    exact_all = np.concatenate([exact_features_source, exact_features_target])

    n_source = len(baseline_features_source)

    print(f"Running t-SNE on {len(baseline_all)} samples...")

    # Run t-SNE for baseline
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    baseline_tsne = tsne.fit_transform(baseline_all)

    # Run t-SNE for EXACT+
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    exact_tsne = tsne.fit_transform(exact_all)

    # Create output directory
    os.makedirs(save_path, exist_ok=True)

    # Create baseline plot (no legend)
    print(f"Creating {baseline_model_name} plot...")
    create_single_tsne_plot(
        baseline_features_source,
        baseline_features_target,
        baseline_tsne,
        n_source,
        model_name=baseline_model_name,
        save_path=save_path,
        is_exact=False
    )

    # Create EXACT+ plot (with legend)
    print(f"Creating {exact_model_name} plot...")
    create_single_tsne_plot(
        exact_features_source,
        exact_features_target,
        exact_tsne,
        n_source,
        model_name=exact_model_name,
        save_path=save_path,
        is_exact=True
    )

    print(f"\nAll plots saved to: {save_path}/")


def get_dataloader(dataset_class, args, split="test"):
    """Create a dataloader for feature extraction."""
    dataset = dataset_class(
        root=args.data_root,
        split=split,
        window_size=args.window_size,
        window_step=args.window_step,
        bg_fg=None,
        args=args,
        transforms=None,
    )

    sampler = DenseLabelTaskSampler(
        dataset=dataset,
        n_shot=args.n_shot,
        n_query=args.n_query,
        threshold_ratio=args.threshold_ratio,
        batch_size=args.batch_size,
        n_tasks=args.n_tasks,
        add_side_noise=args.add_side_noise,
        args=args,
    )

    return DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=sampler.episodic_collate_fn,
        num_workers=0,
    )


def main():
    parser = argparse.ArgumentParser(description="t-SNE Domain Visualization")

    # Model selection
    parser.add_argument(
        "--baseline_model", type=str, default="ex", help="Baseline model type"
    )
    parser.add_argument(
        "--exact_model",
        type=str,
        default="caex",
        help="EXACT+ model type (caex)"
    )
    parser.add_argument(
        "--baseline_seed",
        type=int,
        default=43,
        help="Seed for baseline model"
    )
    parser.add_argument(
        "--exact_seed", type=int, default=43, help="Seed for EXACT+ model"
    )
    parser.add_argument("--baseline_version", type=str, default="v0")
    parser.add_argument("--exact_version", type=str, default="v1")

    # W&B parameters
    parser.add_argument("--wandb_project", type=str, default="EXACT")
    parser.add_argument("--wandb_entity", type=str, default=None)

    # Model architecture params (must match trained models)
    parser.add_argument("--in_channels", type=int, default=6)
    parser.add_argument("--out_channels", type=int, default=2)
    parser.add_argument("--embed_dim", type=int, default=32)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--mlp_ratio", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.5)

    # CNN-specific
    parser.add_argument(
        "--conv_sizes", type=int, nargs="+", default=[32, 64, 128]
    )

    # Segmenter-specific
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--embed_dims", type=int, default=128)

    # Transformer-specific
    parser.add_argument("--ninp", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=3)
    parser.add_argument("--nhid", type=int, default=256)
    parser.add_argument("--nlayers", type=int, default=6)
    parser.add_argument("--init_std", type=float, default=0.02)
    parser.add_argument("--activation", type=str, default="relu")

    # CAEX-specific
    parser.add_argument("--caex_gate", type=str, default="se")
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

    # Dataset parameters
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--window_size", type=int, default=500)
    parser.add_argument("--window_step", type=int, default=25)
    parser.add_argument("--dataset_seed", type=int, default=42)

    # Meta-learning parameters
    parser.add_argument("--n_shot", type=int, default=5)
    parser.add_argument("--n_query", type=int, default=5)
    parser.add_argument("--threshold_ratio", type=float, default=0.25)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--n_tasks", type=int, default=100)

    # Visualization parameters
    parser.add_argument(
        "--n_samples",
        type=int,
        default=500,
        help="Number of samples per domain for t-SNE"
    )
    parser.add_argument("--output_path", type=str, default="saved_figure")

    # Other
    parser.add_argument("--add_side_noise", action="store_true", default=True)
    parser.add_argument("--loocv", action="store_true", default=False)
    parser.add_argument("--fine_split", action="store_true", default=False)
    parser.add_argument("--shuffle", type=str, default="random_variation")
    parser.add_argument("--noise_type", type=str, default="all")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("t-SNE Domain Adaptation Visualization")
    print("=" * 60)

    # Set seed for reproducibility
    set_seed(42)

    # Create dataloaders
    print("\nLoading datasets...")
    args.dataset = "physiq"
    physiq_loader = get_dataloader(PhysiQ, args, split="test")

    args.dataset = "phytmo"
    phytmo_loader = get_dataloader(PHYTMO, args, split="test")

    # Sample data ONCE to ensure all models use the same samples
    print(
        f"\nSampling {args.n_samples} samples from each domain (ONCE for all models)..."
    )
    physiq_inputs = sample_data_once(
        physiq_loader, args, n_samples=args.n_samples
    )
    phytmo_inputs = sample_data_once(
        phytmo_loader, args, n_samples=args.n_samples
    )
    print(f"  PhysiQ samples: {physiq_inputs.shape}")
    print(f"  PhyTMo samples: {phytmo_inputs.shape}")

    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)

    # Define all models to process
    # Format: (model_name, seed, version, is_exact)
    all_models = [
        # Baselines
        ("cnn", 43, "v0", False),
        ("unet", 43, "v0", False),
        ("segmenter", 43, "v0", False),
        ("transformer", 43, "v0", False),
        ("ex", 43, "v0", False),
        # EXACT+ (last one gets the legend)
        ("caex", 42, "v1", True),
    ]

    # Store all features and t-SNE results
    all_results = {}

    for model_name, seed, version, is_exact in all_models:
        print(f"\n{'='*60}")
        print(f"Processing: {model_name}-{seed}")
        print(f"{'='*60}")

        # Load model
        args.model = model_name

        # Reset to defaults first, then set model-specific parameters
        args.embed_dim = 32
        args.num_heads = 4
        args.num_layers = 6
        args.embed_dims = 256
        args.ninp = 64
        args.mlp_ratio = 4
        args.dropout = 0.5

        # Set model-specific parameters to match trained checkpoints
        if model_name == "segmenter":
            args.num_layers = 2
            args.embed_dims = 128
            args.num_heads = 4  # 128 / 4 = 32
        elif model_name == "transformer":
            args.num_layers = 6
            args.embed_dims = 256
            args.num_heads = 4  # ninp=64, 64 / 4 = 16
            args.ninp = 64
        elif model_name in ["ex", "caex"]:
            args.embed_dim = 32
            args.num_heads = 4
            args.mlp_ratio = 4
            args.dropout = 0.5

        try:
            model_path = load_model_from_wandb(args, model_name, seed, version)
            model = get_model(args)
            model.load_state_dict(
                torch.load(
                    model_path, map_location=args.device, weights_only=True
                )
            )
            model.to(args.device)
        except Exception as e:
            print(f"  ERROR loading {model_name}: {e}")
            continue

        # Extract features using pre-sampled data (same samples for all models)
        print(
            f"  Extracting features ({len(physiq_inputs)} samples per domain)..."
        )

        print(f"    {model_name} on PhysiQ (source)...")
        feat_source = extract_features_from_inputs(model, physiq_inputs, args)

        print(f"    {model_name} on PhyTMo (target)...")
        feat_target = extract_features_from_inputs(model, phytmo_inputs, args)

        print(
            f"    Feature shapes: source={feat_source.shape}, target={feat_target.shape}"
        )

        # Run t-SNE
        print(f"    Running t-SNE...")
        all_features = np.concatenate([feat_source, feat_target])
        tsne = TSNE(
            n_components=2, random_state=42, perplexity=30, max_iter=1000
        )
        tsne_result = tsne.fit_transform(all_features)

        # Create plot
        print(f"    Creating plot...")
        feature_dim = feat_source.shape[1]
        create_single_tsne_plot(
            feat_source,
            feat_target,
            tsne_result,
            len(feat_source),
            model_name=model_name,
            save_path=args.output_path,
            is_exact=is_exact,
            feature_dim=feature_dim
        )

        # Store results
        all_results[model_name] = {
            'feat_source': feat_source,
            'feat_target': feat_target,
            'tsne': tsne_result,
            'is_exact': is_exact
        }

    print(f"\n{'='*60}")
    print(f"All plots saved to: {args.output_path}/")
    print(
        f"Generated: {', '.join([f'tsne_{m[0]}.png' for m in all_models if m[0] in all_results])}"
    )
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
