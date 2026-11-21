#!/usr/bin/env bash
# Google Colab (or any CUDA machine): create .venv with the pinned versions in requirements.txt
# (torch 2.15.0 from PyPI is the CUDA 13.2 build). If this machine has a GPU whose driver cannot run that
# build, it falls back to Colab's preinstalled torch/torchvision, which always match Colab's driver.
set -euo pipefail
cd "$(dirname "$0")/.."
GPU=0; nvidia-smi >/dev/null 2>&1 && GPU=1
ok() {
  .venv/bin/python -c "
import torch, torchvision, cv2, sklearn, pandas, scipy, numpy
assert $GPU == 0 or torch.cuda.is_available(), 'torch cannot use the GPU'
torch.zeros(1, device='cuda' if $GPU else 'cpu') + 1"
}
if [ -x .venv/bin/python ] && ok 2>/dev/null; then
  echo "environment already installed"
else
  rm -rf .venv
  pip -q install uv
  REQ=$(mktemp)
  grep -v -E '^torch' requirements.txt > "$REQ"
  uv venv -q --python 3.12 .venv
  uv pip install -q --python .venv/bin/python -r "$REQ" torch==2.15.0 torchvision==0.29.2
  if ! ok; then
    echo "Pinned torch 2.15.0 cannot use this GPU (driver too old for CUDA 13.2): using Colab's preinstalled torch."
    rm -rf .venv
    uv venv -q --system-site-packages --python "$(command -v python3)" .venv
    if ! uv pip install -q --python .venv/bin/python -r "$REQ"; then
      echo "Pinned versions not available for this Python: installing the nearest available versions."
      uv pip install -q --python .venv/bin/python $(sed 's/==.*//' "$REQ")
    fi
    ok || { echo "FAILED: no working torch for this GPU."; exit 1; }
  fi
fi
.venv/bin/python -c "import sys, torch, torchvision, numpy; print('python', sys.version.split()[0], '| torch', torch.__version__, '| torchvision', torchvision.__version__, '| numpy', numpy.__version__, '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
