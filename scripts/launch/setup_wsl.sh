#!/usr/bin/env bash
# WSL/Linux setup script for this repository.
#
# Why this exists:
# - The repo already has a Windows venv at .venv/ (Scripts/), which is not
#   compatible with bash tooling in WSL.
# - This script creates a Linux-native venv outside /mnt/c for faster I/O.
#
# Usage:
#   chmod +x setup_wsl.sh
#   ./setup_wsl.sh
#
# Optional env vars:
#   WSL_VENV_PATH=/home/<you>/.venvs/gnn_scalability ./setup_wsl.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WSL_VENV_PATH="${WSL_VENV_PATH:-$HOME/.venvs/gnn_scalability}"

echo "=== WSL Setup ==="
echo "Project: $PROJECT_ROOT"
echo "Venv:    $WSL_VENV_PATH"

mkdir -p "$(dirname "$WSL_VENV_PATH")"

if [[ ! -d "$WSL_VENV_PATH" ]]; then
  echo "[1/5] Creating Linux virtualenv..."
  python3 -m venv --without-pip "$WSL_VENV_PATH"
fi

source "$WSL_VENV_PATH/bin/activate"
export PYTHONUTF8=1

echo "[2/5] Bootstrapping pip..."
if ! python -m pip --version >/dev/null 2>&1; then
  curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
  python /tmp/get-pip.py
fi
python -m pip install --upgrade pip setuptools wheel

echo "[3/5] Installing Torch + PyG + Python deps..."
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

TORCH_VER="$(python -c "import torch; print(torch.__version__.split('+')[0])")"
CUDA_TAG="$(python -c "import torch; d=torch.version.cuda; print('cu'+''.join(d.split('.')) if d else 'cpu')")"
PYG_URL="https://data.pyg.org/whl/torch-${TORCH_VER}+${CUDA_TAG}.html"
echo "      PyG wheels: $PYG_URL"

python -m pip install torch_geometric
python -m pip install torch_scatter torch_sparse torch_cluster -f "$PYG_URL"
python -m pip install \
  ogb pandas scikit-learn torchmetrics pytorch-lightning \
  matplotlib seaborn tabulate pytest

echo "[4/5] Building Linux C++ binary..."
cd "$PROJECT_ROOT"
mkdir -p bin
g++ -std=c++17 -O3 -Wall -o bin/graph_prep HUB/main.cpp HUB/param.cpp
echo "      Built: bin/graph_prep"

echo "[5/5] Verifying imports + quick test..."
python - <<'PY'
import torch, torch_geometric, torch_sparse, torch_scatter
print("torch:", torch.__version__)
print("torch_geometric:", torch_geometric.__version__)
print("torch_sparse:", torch_sparse.__version__)
print("torch_scatter:", torch_scatter.__version__)
PY
pytest -q tests/test_mprw_kernel.py

cat <<EOF

Setup complete.
Use this environment in WSL with:
  source "$WSL_VENV_PATH/bin/activate"
  export PYTHONUTF8=1

Binary:
  $PROJECT_ROOT/bin/graph_prep
EOF
