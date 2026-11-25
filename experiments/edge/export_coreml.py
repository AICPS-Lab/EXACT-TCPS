#!/usr/bin/env python3
"""
Export a trained segmentation model to Core ML for on-device latency measurement
(Apple Watch / iPhone), and check the exported model against PyTorch.

    python experiments/edge/export_coreml.py --model caex --checkpoint saved_model/caex-physiq-42.pth \
        --window 500 --out experiments/edge/models/EXACT_physiq.mlpackage

Input : "window"  float32 [1, T, 6]   (T samples x acc xyz, gyro xyz; same layout as training)
Output: "logits"  float32 [1, T, n_classes]
Conversion runs on Linux or macOS; the numerical check against PyTorch needs macOS
(Core ML prediction is macOS-only), so on Linux reference outputs are saved next to the
package and the check is done by `--verify-only` on the Mac.
"""
import argparse, json, os, platform, sys
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="caex")
    ap.add_argument("--checkpoint", default="saved_model/caex-physiq-42.pth")
    ap.add_argument("--window", type=int, default=500)
    ap.add_argument("--channels", type=int, default=6)
    ap.add_argument("--out", default="experiments/edge/models/EXACT_physiq.mlpackage")
    ap.add_argument("--precision", choices=["float32", "float16"], default="float32")
    ap.add_argument("--verify-only", action="store_true", help="macOS: only run the PyTorch-vs-CoreML check")
    a = ap.parse_args()
    os.chdir(ROOT)
    import coremltools as ct
    from eval_realtime import load_model

    ref_path = a.out + ".reference.npz"
    if not a.verify_only:
        model = load_model(a.model, a.checkpoint, device="cpu", in_channels=a.channels, out_channels=2)
        model.eval()
        torch.manual_seed(0)
        x = torch.randn(1, a.window, a.channels)
        with torch.no_grad():
            ref = model(x)
        # check_trace=False: the attention models trip torch's trace self-check on benign nondeterminism;
        # correctness is established by the Core ML-vs-PyTorch comparison below instead.
        traced = torch.jit.trace(model, x, check_trace=False)
        mlmodel = ct.convert(
            traced,
            inputs=[ct.TensorType(name="window", shape=(1, a.window, a.channels), dtype=np.float32)],
            outputs=[ct.TensorType(name="logits")],
            minimum_deployment_target=ct.target.iOS17,
            compute_precision=ct.precision.FLOAT32 if a.precision == "float32" else ct.precision.FLOAT16,
            convert_to="mlprogram",
        )
        mlmodel.short_description = f"EXACT ({a.model}) from {os.path.basename(a.checkpoint)}, window {a.window}"
        mlmodel.save(a.out)
        np.savez(ref_path, x=x.numpy(), logits=ref.numpy())
        n_params = sum(p.numel() for p in model.parameters())
        json.dump({"model": a.model, "checkpoint": a.checkpoint, "window": a.window, "channels": a.channels,
                   "params": n_params, "out_shape": list(ref.shape), "precision": a.precision},
                  open(a.out + ".json", "w"), indent=1)
        print(f"saved {a.out}  (params {n_params:,}, output {tuple(ref.shape)})")

    if platform.system() == "Darwin":
        mlmodel = ct.models.MLModel(a.out)
        r = np.load(ref_path)
        pred = mlmodel.predict({"window": r["x"]})["logits"]
        diff = np.abs(pred - r["logits"]).max()
        agree = (pred.argmax(-1) == r["logits"].argmax(-1)).mean()
        print(f"Core ML vs PyTorch: max |diff| = {diff:.2e}, argmax agreement = {agree:.4f}")
        if diff > 1e-2 or agree < 0.999:
            sys.exit("MISMATCH: do not benchmark this package")
    else:
        print("Linux: conversion done; run with --verify-only on the Mac to check numerics.")


if __name__ == "__main__":
    main()
