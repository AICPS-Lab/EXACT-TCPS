#!/usr/bin/env bash
# Pack the PhysiQ dataset (not in git) for upload to a pod.
#   bash experiments/pack_data.sh [path/to/data/physiq] [out.tar.gz]
# Then either
#   - upload physiq_data.tar.gz to the pod's /workspace (runpodctl send / scp / web UI), or
#   - host it somewhere private and set PHYSIQ_URL in the pod template.
set -euo pipefail
SRC=${1:-$(dirname "$0")/../data/physiq}
OUT=${2:-physiq_data.tar.gz}
SRC=$(readlink -f "$SRC")
[ -d "$SRC" ] || { echo "no such dir: $SRC"; exit 1; }
# keep the cached .pt / .npy files: they skip ~20 s of noise synthesis per run
tar -czf "$OUT" -C "$(dirname "$SRC")" --exclude='__pycache__' --exclude='pregenerated_episodes' physiq
ls -lh "$OUT"
