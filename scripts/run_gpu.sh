#!/usr/bin/env bash
# Neural suite: 4 backbones x training regimes x 3 seeds, background-only probes and the duplicate-leakage
# check (s10). Resumable: every run whose prediction file exists is skipped. About 2.5 h of training on a T4 GPU.
# The repository ships the paper's predictions, so remove results/preds/*.npz first to retrain from scratch.
# Needs the prepared arrays (scripts/run_all.sh stages s01-s05, or python -m src.prepare prepare).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}

if [ ! -f data/work/trashnet_frames.npy ] || [ ! -f data/work/variants6/test_base_ORIGINAL.npy ]; then
  echo "Data arrays missing: run python -m src.prepare prepare (or scripts/run_all.sh) first." >&2
  exit 1
fi

run() {
  local arch=$1 regime=$2 seed=$3
  local out="results/preds/${arch}_${regime}_s${seed}.npz"
  if [ -f "$out" ]; then
    echo "=== [skip] $arch $regime s$seed"
    return
  fi
  echo "=== $(date '+%F %T') start $arch $regime s$seed"
  $PY -m src.s08_cnn "$arch" "$regime" "$seed"
  echo "=== $(date '+%F %T') done  $arch $regime s$seed"
}

# Partly frozen ResNet-18 (primary model): six-class regimes, 4-class U2-Net regimes, all probes
for regime in orig jitter paste6 shuffle6 rand6 paste rand; do
  for seed in 0 1 2; do run cnn "$regime" "$seed"; done
done
for probe in probe_bg6 probe_corners probe_bg probe_fixed; do
  for seed in 0 1 2; do run cnn "$probe" "$seed"; done
done

# Fine-tuned ResNet-18, MobileNetV3-Large, Swin-T: six-class regimes and both six-class probes
for arch in resnet18_ft mobilenet_v3 swin_t; do
  for regime in orig jitter paste6 shuffle6 rand6; do
    for seed in 0 1 2; do run "$arch" "$regime" "$seed"; done
  done
  for probe in probe_bg6 probe_corners; do
    for seed in 0 1 2; do run "$arch" "$probe" "$seed"; done
  done
done

# Duplicate leakage on the official TrashNet split (RQ4)
[ -f results/rq4.json ] || $PY -m src.s10_leakage
echo "=== ALL GPU STAGES COMPLETE $(date '+%F %T')"
