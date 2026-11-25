#!/usr/bin/env bash
# One-shot bootstrap for a RunPod (or any Ubuntu + CUDA) pod.
#
# Tested assumptions: RunPod "PyTorch 2.x" template (torch + CUDA preinstalled),
# a persistent volume mounted at /workspace, and these env vars set in the pod
# template (RunPod -> Edit Template -> Environment Variables):
#
#   WANDB_API_KEY   required for W&B logging (omit and pass --nowandb otherwise)
#   WANDB_ENTITY    optional
#   PHYSIQ_URL      optional: URL of physiq_data.tar.gz made by pack_data.sh
#                   (PhysiQ is not in git; SPAR / MM-Fit / PhyTMo are)
#   REPO_URL        default https://github.com/AICPS-Lab/EXACT-ICCPS2025.git
#   REPO_BRANCH     default tcps-revision
#   PHASE           default train  (train | ood | realtime)
#   PER_GPU         default 3      (concurrent runs per GPU)
#
# Usage on the pod:
#   bash <(curl -fsSL https://raw.githubusercontent.com/AICPS-Lab/EXACT-ICCPS2025/tcps-revision/experiments/runpod_setup.sh)
# or, after cloning:   bash experiments/runpod_setup.sh
set -euo pipefail

WORK=${WORK:-/workspace}
REPO_URL=${REPO_URL:-https://github.com/AICPS-Lab/EXACT-ICCPS2025.git}
REPO_BRANCH=${REPO_BRANCH:-tcps-revision}
PHASE=${PHASE:-train}
PER_GPU=${PER_GPU:-3}
REPO_DIR=$WORK/EXACT-ICCPS2025

echo "== system"
nvidia-smi --query-gpu=index,name,memory.total --format=csv || true
nproc; free -g | head -2

echo "== apt"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq tmux git curl > /dev/null

echo "== repo"
if [ ! -d "$REPO_DIR/.git" ]; then
  git clone --branch "$REPO_BRANCH" "$REPO_URL" "$REPO_DIR"
else
  git -C "$REPO_DIR" fetch -q && git -C "$REPO_DIR" checkout -q "$REPO_BRANCH" && git -C "$REPO_DIR" pull -q
fi
cd "$REPO_DIR"

echo "== python deps (torch comes from the image)"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
pip install -q -r experiments/requirements-min.txt

echo "== PhysiQ data"
if [ ! -d data/physiq ]; then
  if [ -n "${PHYSIQ_URL:-}" ]; then
    mkdir -p data && curl -fL "$PHYSIQ_URL" -o /tmp/physiq_data.tar.gz
    tar -xzf /tmp/physiq_data.tar.gz -C data && rm /tmp/physiq_data.tar.gz
  elif [ -f "$WORK/physiq_data.tar.gz" ]; then
    mkdir -p data && tar -xzf "$WORK/physiq_data.tar.gz" -C data
  else
    echo "!! data/physiq missing: set PHYSIQ_URL or put physiq_data.tar.gz in $WORK (see experiments/pack_data.sh)"
  fi
fi
ls data/

echo "== W&B"
if [ -n "${WANDB_API_KEY:-}" ]; then
  wandb login --relogin "$WANDB_API_KEY" > /dev/null && echo "wandb ok"
else
  echo "WANDB_API_KEY not set: runs must use --nowandb (edit make_jobs.py COMMON)"
fi

echo "== job files"
python experiments/make_jobs.py
mkdir -p results/runs results/ood results/realtime_eval logs saved_model

if [ "$PHASE" = "train" ]; then
  echo "== pre-generating episodes (standard-training runs reuse them; LOOCV samples on the fly)"
  for ds in physiq spar mmfit; do
    [ -d "data/$ds" ] || continue
    python pregenerate_episodes.py --dataset "$ds" --n_epochs 200 --seeds 42 43 44 45 46 \
      2>&1 | tail -2 || echo "pregenerate failed for $ds (training will fall back to on-the-fly)"
  done
fi

echo "== launching phase '$PHASE' in tmux session 'exact' ($PER_GPU jobs per GPU)"
tmux kill-session -t exact 2>/dev/null || true
tmux new-session -d -s exact \
  "python experiments/runner.py experiments/jobs/${PHASE}.json --per_gpu $PER_GPU 2>&1 | tee logs/runner_${PHASE}.log"
echo "attach with:  tmux attach -t exact     status:  cat logs/_status.json | python -m json.tool | grep state | sort | uniq -c"
