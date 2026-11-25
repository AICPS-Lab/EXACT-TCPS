import argparse
import os
import random
import time
import json
import socket
import subprocess
import typing

import higher
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
import torch.distributed as dist

import wandb
from datasets.DenseLabelTaskSampler import DenseLabelTaskSampler
from MetricsAccumulator import MetricsAccumulator, MetricsAccumulator_v2
from methods import EX, UNet
from methods.adabn import adabn_adapt
from until_argparser import get_all_subjects, get_args, get_dataset, get_model
from utilities import model_exception_handler, printc, seed, mark_model_complete
from utils_metrics import fsl_visualize_softmax, visualize_softmax
from pregenerate_episodes import (
    PreGeneratedEpisodeLoader,
    get_pregenerated_dir,
    load_pregenerated_episodes,
)
from utils_performance import (
    auto_tune_dataloader,
    apply_dataloader_settings,
    setup_distributed,
    cleanup_distributed,
    get_device,
    wrap_model_distributed,
    is_main_process,
    print_system_info,
    MixedPrecisionContext,
    auto_distribute,
    get_available_gpus,
    is_distributed,
    get_world_size,
    get_rank,
)


def reduce_tensor(tensor, world_size):
    """Average tensor across all GPUs."""
    if not is_distributed():
        return tensor
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt


def reduce_dict(input_dict, world_size):
    """Reduce a dictionary of tensors across all GPUs.
    
    Only reduces numeric values (int, float, tensor). Non-numeric values are skipped.
    """
    if not is_distributed():
        return input_dict

    with torch.no_grad():
        names = []
        values = []
        non_numeric = {}  # Store non-numeric values to preserve them

        for k, v in sorted(input_dict.items()):
            if isinstance(v, torch.Tensor):
                names.append(k)
                values.append(v.clone().cuda() if not v.is_cuda else v.clone())
            elif isinstance(v, (int, float)):
                names.append(k)
                values.append(
                    torch.tensor(v, device='cuda', dtype=torch.float64)
                )
            else:
                # Preserve non-numeric values without reduction
                non_numeric[k] = v

        if not values:
            return input_dict

        # Stack and reduce
        values = torch.stack(values, dim=0)
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        values /= world_size

        reduced_dict = {
            k: v.item()
            for k, v in zip(names, values)
        }
        reduced_dict.update(non_numeric)  # Add back non-numeric values
    return reduced_dict


def is_exp_model(model):
    """Check if the model is EXP (EX+) and supports AdaBN."""
    # Check if it's the wrapper or the actual EXP model
    if hasattr(model, 'net'):
        return hasattr(model.net, 'use_adabn') and model.net.use_adabn
    return hasattr(model, 'use_adabn') and model.use_adabn


def get_adabn_steps(model):
    """Get the number of AdaBN steps from the model."""
    if hasattr(model, 'net') and hasattr(model.net, 'adabn_steps'):
        return model.net.adabn_steps
    if hasattr(model, 'adabn_steps'):
        return model.adabn_steps
    return 50  # Default


def train(db, net, epoch, args, wandb_run=None):
    """
    Training function with proper multi-GPU support.
    
    In distributed mode:
    - Work is SPLIT across GPUs (each GPU does n_tasks/world_size)
    - Gradients are averaged across GPUs after each meta-update
    - This gives true linear speedup
    """
    # Get the base model (unwrap DDP if needed)
    # Note: For meta-learning with `higher`, we need the unwrapped model
    base_net = net.module if hasattr(net, 'module') else net

    # Ensure model is on the correct device and in training mode
    base_net.to(args.device)
    base_net.train()
    compute_metrics = MetricsAccumulator(
        dir_name="train", n_classes=args.out_channels
    )

    world_size = get_world_size()
    rank = get_rank()

    # Split tasks across GPUs for true speedup
    # Each GPU processes n_tasks / world_size tasks
    tasks_per_gpu = max(1, args.n_tasks // world_size)

    for batch_idx in range(tasks_per_gpu):
        start_time = time.time()

        # Sample a batch of support and query images and labels
        x_spt, y_spt, x_qry, y_qry, _ = next(db)

        # Move data to the specified device
        x_spt, y_spt = x_spt.to(args.device), y_spt.to(args.device)
        x_qry, y_qry = x_qry.to(args.device), y_qry.to(args.device)

        task_num, setsz, h, w = x_spt.size()
        querysz = x_qry.size(1)

        # Initialize the inner optimizer for adaptation (use base model for higher)
        inner_opt = torch.optim.SGD(base_net.parameters(), lr=args.meta_lr)

        qry_losses = []
        args.meta_opt.zero_grad()

        for i in range(task_num):
            # utilize the sec-derivative gradient information for the inner loop
            # Note: We use base_net (unwrapped) with higher library
            with higher.innerloop_ctx(
                base_net,
                inner_opt,
                copy_initial_weights=False,
                track_higher_grads=True
            ) as (fnet, diffopt):
                # Inner-loop adaptation
                for _ in range(args.n_inner_iter):
                    spt_logits = fnet(x_spt[i])
                    spt_loss = F.cross_entropy(spt_logits, y_spt[i].long())
                    if args.fomaml:
                        diffopt.step(
                            spt_loss,
                            grad_callback=lambda grads: [
                                g.detach() if g is not None else None
                                for g in grads
                            ]
                        )
                    else:
                        diffopt.step(spt_loss)

                # Compute the loss and accuracy on the query set
                qry_logits = fnet(x_qry[i])
                qry_loss = F.cross_entropy(qry_logits, y_qry[i].long())
                qry_losses.append(qry_loss.detach())
                string_score, score = compute_metrics.update(
                    y_qry[i].long(), qry_logits,
                    batch_metrics=(wandb_run is None and batch_idx % args.log_interval == 0),
                )
                # Backpropagation on query loss
                qry_loss.backward()

        # Average gradients across GPUs before optimizer step
        # This is necessary because we use `higher` library which doesn't support DDP directly
        if is_distributed():
            # Barrier to ensure all processes finished their backward pass
            dist.barrier()
            # Synchronize gradients across all processes
            for param in base_net.parameters():
                if param.grad is not None:
                    # Use all_reduce to sum gradients, then divide by world_size for average
                    dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                    param.grad.div_(
                        world_size
                    )  # In-place division is more efficient

        args.meta_opt.step()

        # Compute average loss across GPUs
        qry_losses = sum(qry_losses) / task_num
        if is_distributed():
            if not isinstance(qry_losses, torch.Tensor):
                qry_losses = torch.tensor(qry_losses, device=args.device)
            qry_losses = reduce_tensor(qry_losses, world_size)

        # Calculate epoch progress (account for split across GPUs)
        global_batch_idx = batch_idx * world_size + rank
        i = epoch + float(global_batch_idx) / args.n_tasks
        iter_time = time.time() - start_time

        # Only print on main process to avoid duplicate output
        if wandb_run is None and batch_idx % args.log_interval == 0 and is_main_process(
        ):
            print(
                f"[Epoch {i:.2f}] Train Loss: {qry_losses:.2f} | Time: {iter_time:.2f}",
                end=" | ",
            )
            print(string_score)

        res_dict = compute_metrics.compute()
        res_dict.update(
            {
                "train/epoch":
                i,
                "train/loss":
                qry_losses.item()
                if isinstance(qry_losses, torch.Tensor) else qry_losses,
                "train/time":
                time.time(),
            }
        )
        if wandb_run is not None:
            wandb_run.log(res_dict)
        compute_metrics.reset()
    # wandb.finish()
    return


def test(db, net, epoch, args, wandb_r=None):
    """
    Test function with proper multi-GPU support.
    
    In distributed mode:
    - Work is SPLIT across GPUs (each GPU does n_tasks/world_size)
    - Metrics are aggregated from all GPUs
    - This gives true linear speedup
    """
    # Get the base model (unwrap DDP if needed)
    base_net = net.module if hasattr(net, 'module') else net

    # Ensure model is on the correct device
    # Note: We use train() mode to allow BatchNorm adaptation during inner loop
    base_net.to(args.device)
    base_net.train()

    world_size = get_world_size()
    rank = get_rank()

    # Split tasks across GPUs for true speedup
    n_test_tasks = getattr(args, "n_test_tasks", None) or args.n_tasks
    tasks_per_gpu = max(1, n_test_tasks // world_size)

    # Check if model is EXP and supports AdaBN
    use_adabn = is_exp_model(base_net)
    adabn_steps = get_adabn_steps(base_net) if use_adabn else 0

    compute_metrics = MetricsAccumulator(
        dir_name="test", n_classes=args.out_channels
    )

    qry_losses = []

    for batch_idx in range(tasks_per_gpu):
        # Sample a batch of support and query images and labels
        try:
            x_spt, y_spt, x_qry, y_qry, _ = next(db)
        except StopIteration:
            break

        # Move data to the specified device
        x_spt, y_spt = x_spt.to(args.device), y_spt.to(args.device)
        x_qry, y_qry = x_qry.to(args.device), y_qry.to(args.device)

        task_num, setsz, h, w = x_spt.size()
        querysz = x_qry.size(1)

        # Initialize the inner optimizer for adaptation (use base model)
        inner_opt = torch.optim.SGD(base_net.parameters(), lr=args.meta_lr)

        for i in range(task_num):
            # AdaBN: Adapt BatchNorm stats to support set before inner-loop
            if use_adabn:
                adabn_adapt(
                    base_net, x_spt[i], steps=adabn_steps, reset_stats=True
                )

            with higher.innerloop_ctx(
                base_net, inner_opt, track_higher_grads=False
            ) as (fnet, diffopt):
                # Inner-loop adaptation
                for _ in range(args.n_inner_iter):
                    spt_logits = fnet(x_spt[i])
                    spt_loss = F.cross_entropy(spt_logits, y_spt[i].long())
                    diffopt.step(spt_loss)

                # Compute the query loss and accuracy
                qry_logits = fnet(x_qry[i]).detach()
                qry_loss = F.cross_entropy(
                    qry_logits, y_qry[i].long(), reduction="none"
                )
                qry_losses.append(qry_loss.detach())
                string_score, score = compute_metrics.update(
                    y_qry[i].long(), qry_logits
                )

    qry_losses = torch.cat(qry_losses).mean()

    # Ensure tensor is on CUDA before reduce
    if is_distributed():
        if not qry_losses.is_cuda:
            qry_losses = qry_losses.cuda()
        # Sync before reduce to ensure all processes are ready
        dist.barrier()
        qry_losses = reduce_tensor(qry_losses, world_size)

    qry_losses = qry_losses.item()

    res_dict = compute_metrics.compute()

    # Aggregate metrics across GPUs (only numerical values)
    if is_distributed():
        metrics_to_reduce = {
            k: v
            for k, v in res_dict.items()
            if isinstance(v, (int, float))
        }
        if metrics_to_reduce:  # Only reduce if there are metrics
            reduced_metrics = reduce_dict(metrics_to_reduce, world_size)
            res_dict.update(reduced_metrics)
        # Sync after reduce
        dist.barrier()

    res_dict.update(
        {
            "test/epoch": epoch + 1,
            "test/loss": qry_losses,
            "test/time": time.time(),
        }
    )

    # Print test results when not using wandb
    if wandb_r is None and is_main_process():
        print(
            f"[Test Epoch {epoch + 1}] Loss: {qry_losses:.4f} | "
            f"Prec: {res_dict.get('test/Precision', 0):.2f}, "
            f"Rec: {res_dict.get('test/Recall', 0):.2f}, "
            f"F1: {res_dict.get('test/F1-Score', 0):.2f}, "
            f"IoU: {res_dict.get('test/IoU', 0):.2f}"
        )

    if wandb_r is not None:
        # Log results
        wandb_r.log(res_dict)
        log_visualization(epoch, wandb_r, net, fnet, inner_opt, args)
    return qry_losses, res_dict


class PreGeneratedIterator:
    """
    Iterator wrapper for pre-generated episodes that matches DataLoader interface.
    
    Yields (support_images, support_labels, query_images, query_labels, task_info)
    just like the episodic_collate_fn does.
    
    In multi-GPU mode, each rank gets a different slice of the shuffled episodes
    to ensure work is split across GPUs.
    """

    def __init__(
        self,
        episode_loader: PreGeneratedEpisodeLoader,
        epoch: int,
        device: str = 'cuda',
        rank: int = 0,
        world_size: int = 1,
    ):
        self.episode_loader = episode_loader
        self.epoch = epoch
        self.device = device
        self.rank = rank
        self.world_size = world_size
        self._iterator = None

    def __iter__(self):
        # Get all episodes for this epoch
        all_episodes = list(self.episode_loader.get_epoch_iterator(self.epoch))
        
        # In multi-GPU mode, each rank gets a different slice
        # This ensures work is split (not duplicated) across GPUs
        if self.world_size > 1:
            # Shuffle with a consistent seed so all ranks see same order
            # then each rank takes its slice
            random.seed(self.epoch)  # Consistent shuffle across ranks
            random.shuffle(all_episodes)
            
            # Split episodes across ranks
            episodes_per_rank = len(all_episodes) // self.world_size
            start_idx = self.rank * episodes_per_rank
            end_idx = start_idx + episodes_per_rank
            all_episodes = all_episodes[start_idx:end_idx]
        
        self._episodes = iter(all_episodes)
        return self

    def __next__(self):
        episode = next(self._episodes)
        # Episode is already (support_images, support_labels, query_images, query_labels, task_info)
        return episode


def main(args):
    """
    Main training function with proper multi-GPU DDP support.
    
    In distributed mode:
    - Each GPU uses a different random seed for sampling different episodes
    - Model gradients are synchronized across GPUs
    - Only rank 0 handles wandb logging and model saving
    
    Supports pre-generated episodes with --use_pregenerated flag.
    """
    # Get distributed info
    rank = get_rank()
    world_size = get_world_size()

    # Set different random seed per GPU so each sees different episodes
    # This effectively multiplies the number of episodes per step by world_size
    local_seed = args.seed + rank
    seed(local_seed)

    if is_main_process():
        print(
            f"[Distributed] World size: {world_size}, using seeds {args.seed} to {args.seed + world_size - 1}"
        )

    # Check if using pre-generated episodes
    use_pregenerated = getattr(args, 'use_pregenerated', False)
    train_episode_loader = None
    test_episode_loader = None
    train_loader = None
    test_loader = None
    train_dataset = None
    test_dataset = None

    if use_pregenerated:
        # Load pre-generated episodes from disk into RAM
        if is_main_process():
            print(
                "[PreGenerated] Loading pre-generated episodes into memory..."
            )

        try:
            train_episode_loader = load_pregenerated_episodes(
                args, split='train', load_to_memory=True
            )
            test_episode_loader = load_pregenerated_episodes(
                args, split='test', load_to_memory=True
            )
            if is_main_process():
                print(
                    f"[PreGenerated] Loaded {train_episode_loader.n_epochs} epochs of train episodes"
                )
                print(
                    f"[PreGenerated] Loaded {test_episode_loader.n_epochs} epochs of test episodes"
                )
        except FileNotFoundError as e:
            if is_main_process():
                print(f"[PreGenerated] Warning: {e}")
                print(
                    "[PreGenerated] Falling back to on-the-fly generation..."
                )
            use_pregenerated = False

    if not use_pregenerated:
        # Original on-the-fly episode generation
        if is_main_process():
            print(
                "[DataLoader] Loading datasets for on-the-fly episode generation..."
            )
        train_dataset, test_dataset = get_dataset(args)
        if is_main_process():
            print("[DataLoader] Dataset loading complete.")

        # Sync after dataset loading
        if is_distributed():
            dist.barrier()

        # Initialize samplers with rank-specific seed for different episodes per GPU
        train_sampler = DenseLabelTaskSampler(
            train_dataset,
            n_shot=args.n_shot,
            batch_size=args.batch_size,
            n_query=args.n_query,
            n_tasks=args.n_tasks,
            threshold_ratio=args.threshold_ratio,
            add_side_noise=args.add_side_noise,
            args=args
        )
        test_sampler = DenseLabelTaskSampler(
            test_dataset,
            n_shot=args.n_shot,
            batch_size=args.batch_size,
            n_query=args.n_query,
            n_tasks=(getattr(args, "n_test_tasks", None) or args.n_tasks),
            threshold_ratio=args.threshold_ratio,
            add_side_noise=args.add_side_noise,
            args=args
        )

        # Initialize DataLoader
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            collate_fn=train_sampler.episodic_collate_fn,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_sampler=test_sampler,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            collate_fn=test_sampler.episodic_collate_fn,
        )

        # save 10 samples of data for visualization (only on main process)
        if is_main_process():
            capture_test_dataset_samples(args, test_dataset, test_loader)

    # Sync all ranks after data loading
    if is_distributed():
        dist.barrier()

    # Define the model architecture
    model = get_model(args)
    model.to(args.device)  # Move model to specified device

    # Initialize meta optimizer (before DDP wrap, optimizer needs base params)
    args.meta_opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Only initialize wandb on the main process (rank 0) to avoid duplicate logging
    run_name = build_run_name(args)
    if not args.nowandb and is_main_process():
        run = wandb.init(
            project=args.wandb_project,
            entity=getattr(args, 'wandb_entity', None),
            config=vars(args),
            name=run_name,
            group=getattr(args, 'wandb_group', None),
        )
    else:
        run = None
    model_path = "saved_model/" + run_name + ".pth"

    # Synchronize all processes before training starts
    # This ensures wandb init completes on rank 0 before other GPUs proceed
    if is_distributed():
        dist.barrier()

    model_exception_handler(model_path)

    # Training loop
    loss = np.inf
    best_res, last_res, t0 = None, None, time.time()
    for epoch in range(args.n_epochs):
        # Get iterator for this epoch
        if use_pregenerated:
            # Use pre-generated episodes (already in RAM)
            train_iter = PreGeneratedIterator(
                train_episode_loader, epoch % train_episode_loader.n_epochs,
                args.device, rank=rank, world_size=world_size
            )
            test_iter = PreGeneratedIterator(
                test_episode_loader, epoch % test_episode_loader.n_epochs,
                args.device, rank=rank, world_size=world_size
            )
        else:
            # Use on-the-fly generation
            train_iter = iter(train_loader)
            test_iter = iter(test_loader)

        train(train_iter, model, epoch, args, run)
        if (epoch + 1) % max(1, getattr(args, "eval_interval", 1)) and epoch + 1 != args.n_epochs:
            continue
        qry_loss, qry_acc = test(test_iter, model, epoch, args, run)
        last_res = qry_acc
        if qry_loss < loss:
            loss = qry_loss
            best_res = qry_acc
            # Only save on main process to avoid file conflicts
            if is_main_process():
                torch.save(model.state_dict(), model_path)
    if is_main_process():
        mark_model_complete(model_path)
        save_run_results(args, run_name, best_res, last_res, model_path,
                         {"elapsed_sec": time.time() - t0})
    if not args.nowandb and is_main_process():
        log_model_artifact(run, model_path)
        run.finish()
    return


def main_loocv(args):
    """
    Main LOOCV training function with proper multi-GPU DDP support.
    
    In distributed mode:
    - Each GPU uses a different random seed for sampling different episodes
    - Model gradients are synchronized across GPUs
    - Only rank 0 handles wandb logging and model saving
    """
    # Get distributed info
    rank = get_rank()
    world_size = get_world_size()

    # Set up for each subject as a separate LOOCV iteration
    all_subjects = get_all_subjects(
        args
    )  # Define a function to get all subject IDs

    for test_subject in all_subjects:  # Iterate over first 10 subjects
        # Set different random seed per GPU so each sees different episodes
        local_seed = args.seed + rank
        seed(local_seed)

        if is_main_process():
            print(
                f"[LOOCV] Testing subject: {test_subject}, World size: {world_size}"
            )

        # Initialize datasets with the current subject as test
        train_dataset, test_dataset = get_dataset(args, test_subject)

        # Initialize samplers with rank-specific randomness
        train_sampler = DenseLabelTaskSampler(
            train_dataset,
            n_shot=args.n_shot,
            batch_size=args.batch_size,
            n_query=args.n_query,
            n_tasks=args.n_tasks,
            threshold_ratio=args.threshold_ratio,
            add_side_noise=args.add_side_noise,
            args=args
        )
        test_sampler = DenseLabelTaskSampler(
            test_dataset,
            n_shot=args.n_shot,
            batch_size=args.batch_size,
            n_query=args.n_query,
            n_tasks=(getattr(args, "n_test_tasks", None) or args.n_tasks),
            threshold_ratio=args.threshold_ratio,
            add_side_noise=args.add_side_noise,
            args=args
        )

        # Initialize DataLoader
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            collate_fn=train_sampler.episodic_collate_fn,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_sampler=test_sampler,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            collate_fn=test_sampler.episodic_collate_fn,
        )

        # Only capture samples on main process
        if is_main_process():
            capture_test_dataset_samples(args, test_dataset, test_loader)

        # Sync all ranks after sample capture to avoid desync
        if is_distributed():
            dist.barrier()

        # Define the model architecture
        model = get_model(args)
        model.to(args.device)

        if is_main_process():
            print(
                "trainable parameters: ",
                sum(p.numel() for p in model.parameters() if p.requires_grad),
            )

        # Initialize meta optimizer
        args.meta_opt = torch.optim.Adam(model.parameters(), lr=args.lr)

        # Run a unique wandb session for each test subject (only on main process)
        run_name = build_run_name(
            args, f"-subject-{test_dataset.test_subject_filename()}"
        )
        if not args.nowandb and is_main_process():
            run = wandb.init(
                project=args.wandb_project,
                entity=getattr(args, 'wandb_entity', None),
                config=vars(args),
                name=run_name,
                group=getattr(args, 'wandb_group', None),
            )
        else:
            run = None
        model_path = f"saved_model/{run_name}.pth"

        # Synchronize all processes before training starts
        if is_distributed():
            dist.barrier()

        model_exception_handler(model_path)

        # Training loop
        loss = np.inf
        best_res, last_res, t0 = None, None, time.time()
        for epoch in range(args.n_epochs):
            train(iter(train_loader), model, epoch, args, run)
            if (epoch + 1) % max(1, getattr(args, "eval_interval", 1)) and epoch + 1 != args.n_epochs:
                continue
            qry_loss, qry_acc = test(
                iter(test_loader), model, epoch, args, run
            )
            last_res = qry_acc
            if qry_loss < loss:
                loss = qry_loss
                best_res = qry_acc
                # Only save on main process to avoid file conflicts
                if is_main_process():
                    torch.save(model.state_dict(), model_path)

        # Save the model to wandb for each run (only on main process)
        if is_main_process():
            mark_model_complete(model_path)
            save_run_results(
                args, run_name, best_res, last_res, model_path,
                {"elapsed_sec": time.time() - t0,
                 "test_subject": list(test_subject) if isinstance(test_subject, (list, tuple)) else test_subject},
            )
        if not args.nowandb and is_main_process():
            log_model_artifact(run, model_path)
            run.finish()

        # Synchronize before moving to next subject
        if is_distributed():
            dist.barrier()

    return


def build_run_name(args, suffix=""):
    """model-dataset-seed[-tag][suffix]; the tag keeps ablation variants apart."""
    name = f"{args.model}-{args.dataset}-{args.seed}"
    tag = getattr(args, "tag", "")
    if tag:
        name += f"-{tag}"
    return name + suffix


def _jsonable(d):
    out = {}
    for k, v in d.items():
        if isinstance(v, (bool, int, float, str)) or v is None:
            out[k] = v
        elif isinstance(v, (list, tuple)) and all(isinstance(x, (bool, int, float, str)) for x in v):
            out[k] = list(v)
        else:
            out[k] = str(v)
    return out


def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def save_run_results(args, run_name, best, last, model_path, extra=None):
    """Write results/<run_name>.json so numbers never have to be copied from the W&B UI."""
    results_dir = getattr(args, "results_dir", "results/runs") or "results/runs"
    os.makedirs(results_dir, exist_ok=True)
    payload = {
        "run_name": run_name,
        "model": args.model,
        "dataset": args.dataset,
        "seed": args.seed,
        "tag": getattr(args, "tag", ""),
        "model_path": model_path,
        "best": _jsonable(best or {}),
        "last": _jsonable(last or {}),
        "args": _jsonable(vars(args)),
        "host": socket.gethostname(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "git_commit": _git_commit(),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if extra:
        payload.update(extra)
    out = os.path.join(results_dir, run_name + ".json")
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[Results] wrote {out}")
    return out


def log_model_artifact(run, model_path):
    artifact = wandb.Artifact(f"{run.name}", type="model")
    artifact.add_file(model_path)
    run.log_artifact(artifact)


def capture_test_dataset_samples(args, test_dataset, test_loader):
    # if it exists already:
    if os.path.exists(
        os.path.join(
            args.data_root, args.dataset,
            f"{args.dataset}_{str(args.loocv)}_{args.seed}.pt"
        )
    ):
        # printc("Saved 10 samples data already exists")
        return
    sample_data_in_test = []
    for i in range(10):
        x_spt, y_spt, x_qry, y_qry, _ = next(iter(test_loader))
        sample_data_in_test.append((x_spt, y_spt, x_qry, y_qry))
    # torch save
    torch.save(
        sample_data_in_test,
        os.path.join(
            args.data_root, args.dataset,
            f"{args.dataset}_{str(args.loocv)}_{args.seed}.pt"
        )
    )
    # printc("Saved 10 samples of data for visualization")


def log_visualization(epoch, wandb_r, net, fnet, inner_opt, args):
    # load 10 samples of data for visualization
    sample_data_in_test = torch.load(
        os.path.join(
            args.data_root, args.dataset,
            f"{args.dataset}_{str(args.loocv)}_{args.seed}.pt"
        )
    )
    for image_idx, (x_spt, y_spt, x_qry,
                    y_qry) in enumerate(sample_data_in_test):
        # visualize the softmax output of the model
        x_spt, y_spt = x_spt.to(args.device), y_spt.to(args.device)
        x_qry, y_qry = x_qry.to(args.device), y_qry.to(args.device)
        with higher.innerloop_ctx(net, inner_opt, track_higher_grads=False
                                  ) as (fnet, diffopt):
            # Inner-loop adaptation
            for _ in range(args.n_inner_iter):
                spt_logits = fnet(x_spt[:, 0])
                spt_loss = F.cross_entropy(spt_logits, y_spt[:, 0].long())
                if args.fomaml:
                    diffopt.step(
                        spt_loss,
                        grad_callback=lambda grads:
                        [g.detach() if g is not None else None for g in grads]
                    )
                else:
                    diffopt.step(spt_loss)
        images = x_qry[:, 0]
        qry_logits = fnet(images).detach()
        labels = y_qry[:, 0]
        softmaxed = F.softmax(qry_logits, dim=1)
        fsl_visualize_softmax(
            softmaxed[0].cpu().detach().numpy(),
            labels[0].cpu().detach().numpy(),
            images[0].cpu().detach().numpy(),
            x_spt[0, 0].cpu().detach().numpy(),  # Support set data
            y_spt[0, 0].cpu().detach().numpy(),  # Support set labels
            dir=f"saved_figure/{wandb_r.name}_epoch_{epoch}_img_{image_idx}.png"
        )
        # log a visualization:
        # https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-log/Log_(Almost)_Anything_with_W%26B_Media.ipynb#scrollTo=hPqfetlsyXbg
        wandb_r.log(
            {
                f"img_{image_idx}":
                wandb.Image(
                    f"saved_figure/{wandb_r.name}_epoch_{epoch}_img_{image_idx}.png"
                )
            }
        )
        os.remove(
            f"saved_figure/{wandb_r.name}_epoch_{epoch}_img_{image_idx}.png"
        )

    # images = x_qry[0]
    # qry_logits = fnet(images).detach()
    # labels = y_qry[0]
    # softmaxed = F.softmax(qry_logits, dim=1)
    # visualize_softmax(softmaxed[i].cpu().detach().numpy(), labels[i].cpu().detach().numpy(), images[i].cpu().detach().numpy(), dir=f"saved_figure/{wandb_r.name}_epoch_{epoch}.png")
    #     # log a visualization:
    #     # https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-log/Log_(Almost)_Anything_with_W%26B_Media.ipynb#scrollTo=hPqfetlsyXbg
    # wandb_r.log({"img": wandb.Image(f"saved_figure/{wandb_r.name}_epoch_{epoch}.png")})
    # os.remove(f"saved_figure/{wandb_r.name}_epoch_{epoch}.png")


def setup_training(args):
    """
    Setup training environment with auto-tuning and multi-GPU support.
    
    Call this at the start of main() before creating datasets/models.
    """
    # Setup distributed training (if launched with torchrun)
    rank, world_size = setup_distributed()

    # Print system info (main process only)
    if is_main_process() and not args.nowandb:
        print_system_info()

    # Auto-tune DataLoader settings (main process, then broadcast)
    if getattr(args, 'auto_tune', False):
        if is_main_process():
            settings = auto_tune_dataloader(args, verbose=True)
            apply_dataloader_settings(args, settings)

        # In distributed mode, broadcast settings from rank 0
        if world_size > 1:
            import torch.distributed as dist
            settings_tensor = torch.tensor(
                [args.num_workers, int(args.pin_memory)], dtype=torch.int64
            ).cuda()
            dist.broadcast(settings_tensor, src=0)
            if rank != 0:
                args.num_workers = int(settings_tensor[0].item())
                args.pin_memory = bool(settings_tensor[1].item())

    # Set device
    args.device = get_device(args)

    return rank, world_size


if __name__ == "__main__":
    args = get_args()  # Get arguments from the argparse

    # Auto-launch with torchrun if --multi_gpu is set
    # If this returns True, the script exits (torchrun re-launches it)
    if auto_distribute(args):
        import sys
        sys.exit(0)

    # Setup training environment first to determine rank
    rank, world_size = setup_training(args)

    # Only generate wandb group ID on main process, then broadcast to others
    if is_main_process():
        _gen = getattr(getattr(wandb, "util", None), "generate_id", None)
        args.wandb_group = "experiment-" + (_gen() if _gen else __import__("uuid").uuid4().hex[:8])
    else:
        args.wandb_group = None

    # Broadcast wandb_group from rank 0 to all other processes
    if world_size > 1:
        import torch.distributed as dist
        if is_main_process():
            group_bytes = args.wandb_group.encode('utf-8')
            group_tensor = torch.tensor(list(group_bytes),
                                        dtype=torch.uint8).cuda()
            length_tensor = torch.tensor(
                [len(group_bytes)], dtype=torch.int64
            ).cuda()
        else:
            length_tensor = torch.tensor([0], dtype=torch.int64).cuda()
        dist.broadcast(length_tensor, src=0)

        if not is_main_process():
            group_tensor = torch.zeros(
                length_tensor.item(), dtype=torch.uint8
            ).cuda()
        dist.broadcast(group_tensor, src=0)

        if not is_main_process():
            args.wandb_group = bytes(group_tensor.cpu().tolist()
                                     ).decode('utf-8')

    try:
        if args.loocv:
            main_loocv(args)
        else:
            main(args)
    finally:
        cleanup_distributed()
