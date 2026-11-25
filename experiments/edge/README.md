# On-device latency (Apple Watch / iPhone)

Reviewer 3 asked for latency on a real edge device. This folder exports a trained EXACT
checkpoint to Core ML and times it on an iPhone and an Apple Watch.

## 1. Export (Mac or Linux)

```bash
pip install coremltools scikit-learn psutil thop fvcore matplotlib     # in the project env
# EXACT (journal model, 10 s PhysiQ window) and the four baselines, plus the 5 s / 15 s EXACT variants:
python experiments/edge/export_coreml.py --model caex --checkpoint saved_model/caex-physiq-42.pth --window 500 --out experiments/edge/models/EXACT_physiq.mlpackage
for m in unet cnn transformer segmenter; do
  python experiments/edge/export_coreml.py --model $m --checkpoint saved_model/$m-physiq-42.pth --window 500 --out experiments/edge/models/${m}_physiq.mlpackage
done
python experiments/edge/export_coreml.py --model caex --checkpoint saved_model/caex-physiq-42-win250.pth --window 250 --out experiments/edge/models/EXACT_physiq_win250.mlpackage
python experiments/edge/export_coreml.py --model caex --checkpoint saved_model/caex-physiq-42-win750.pth --window 750 --out experiments/edge/models/EXACT_physiq_win750.mlpackage
# on the Mac, check numerics (Core ML prediction is macOS-only):
python experiments/edge/export_coreml.py --verify-only --out experiments/edge/models/EXACT_physiq.mlpackage
```

Input `window` is float32 `[1, T, 6]` (acc xyz, gyro xyz at 50 Hz; T = 500 is the 10 s PhysiQ setting);
output `logits` is `[1, T, 2]`. All seven packages agree with PyTorch to max |diff| < 2e-5 and 100% argmax
agreement (the trace self-check is disabled for the attention models, which trip it on benign
nondeterminism; the numeric check is the real test). The exported weights are the trained checkpoint; in
deployment the 5-shot MAML adaptation runs once per session on the phone (or a server) and the adapted
model is exported the same way, so the on-device cost measured here is the inference cost per window.

## 2. Benchmark app (Xcode 26+, free Apple ID is enough for your own devices)

The Xcode project is generated from `EXACTBench/project.yml` with [xcodegen](https://github.com/yonaskolb/XcodeGen)
(`brew install xcodegen`), so only the spec and the Swift sources are versioned:

```bash
cd experiments/edge/EXACTBench && xcodegen generate && open EXACTBench.xcodeproj
```

Two targets share `EXACTBench/Shared/{BenchmarkRunner,ContentView}.swift` and the `.mlpackage` from
`../models` (referenced, not copied): `EXACTBench` (iOS app) and `EXACTBench Watch` (watchOS app,
embedded in the iOS app, runs independently). Signing is automatic with the team in `project.yml`;
change `DEVELOPMENT_TEAM` there (or in Signing & Capabilities) if Xcode complains. Xcode needs the
iOS and watchOS platform components installed (Settings > Components, or
`xcodebuild -downloadPlatform iOS` / `watchOS`); a fresh Xcode install has none.

The app has two buttons. **Run benchmark** times every bundled model (`BenchmarkRunner.defaultModels`:
EXACT 10 s, EXACT 5 s, EXACT 15 s, U-Net, CNN, Transformer, Segmenter) in both compute modes
(`all` = Neural Engine allowed, `cpuOnly`), 50 warm-up + 300 timed windows each, and reports mean/p50/p95/p99
plus load+compile time, thermal state and battery state. **Sustained 60 s** streams EXACT at the deployment
cadence (one window every 0.5 s) and reports per-minute latency, missed 500 ms deadlines and battery
before/after; the sustained run is the realistic number, since between windows the CPU clocks down and a
cold prediction costs ~10x the back-to-back figure. Every report is appended, time-stamped, to the app's
`Documents/benchmark.txt`. Launch arguments: `--autorun`, `--sustain <s>`, `--iterations <n>`,
`--models a,b`, `--exit` (see `ContentView.swift`).

1. Select scheme `EXACTBench` and the iPhone as run destination, Run, tap the buttons (or launch headless, below).
2. Select scheme `EXACTBench Watch` and the watch, Run, tap again (watch on the wrist, off the charger).

Headless alternative (no tapping; output lands in the terminal): build with signing for the device,
install and launch with the `--autorun --exit` launch arguments, which run the benchmark on appear and quit
after printing it.

```bash
cd experiments/edge/EXACTBench
xcodebuild -project EXACTBench.xcodeproj -scheme EXACTBench -destination 'id=<iphone udid>' \
    -configuration Release -derivedDataPath /tmp/dd -allowProvisioningUpdates build
xcrun devicectl list devices                                  # CoreDevice identifiers
xcrun devicectl device install app --device <iphone id> /tmp/dd/Build/Products/Release-iphoneos/EXACTBench.app
xcrun devicectl device process launch --console --terminate-existing --device <iphone id> \
    com.hcwdavid.EXACTBench --autorun --exit | tee results/edge/<device>.txt
# watch: install "Release-watchos/EXACTBench Watch.app" the same way (bundle id com.hcwdavid.EXACTBench.watchkitapp).
# `--console` hangs on watchOS and a watch app launched with the display dark is suspended before it runs,
# so on the watch open the app by hand, tap "Run benchmark" and keep the display awake until the two blocks
# appear. The report is also written to the app's Documents/benchmark.txt; pull it with
xcrun devicectl device copy from --device <watch id> --domain-type appDataContainer \
    --domain-identifier com.hcwdavid.EXACTBench.watchkitapp --source Documents/benchmark.txt \
    --destination results/edge/Watch7,5.txt
```
Both devices need Xcode whose SDK is at least their OS major (iOS/watchOS 27 -> Xcode 27); the watch connects
over Wi-Fi only and drops the tunnel when it sleeps, so retry `install`/`copy` if CoreDevice reports a
connection error.

Record for the paper: device model string (printed; `iPhone18,2` = iPhone 17 Pro Max, `iPhone18,1` =
iPhone 17 Pro, `Watch7,5` = Ultra 2), OS versions, thermal state, and that the watch was worn, not
charging. Report `all` and `cpuOnly`; the CPU-only figure is the conservative proxy for older watches.

## 3. What goes into the paper

`experiments/paper_tables.py` parses `results/edge/*.txt` (blocks of the multi-model app only, which must
all say `power: unplugged`) into `results/tables/paper/table_edge.tex` (`\TableEdgeRows`: one row per
model + the two window variants, columns iPhone NE/CPU, Watch NE/CPU, mean +- std over runs), and prints
the sustained blocks for the text. Copy the .tex into the manuscript's `tables/`. Latency chain per device
= window fill (10 s) + stride (0.5 s) + inference (sustained figure); the manuscript table is
`tab:edge_latency` in 6.1 Real-Time Evaluation.
