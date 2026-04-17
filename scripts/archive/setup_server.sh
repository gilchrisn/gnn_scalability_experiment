#!/usr/bin/env bash
# One-time environment setup for the school server (Linux).
# Run once after git pull, then use run_server.sh for experiments.
#
# Usage:
#   chmod +x setup_server.sh
#   ./setup_server.sh

set -e

echo "=== Server Setup ==="

# --- Python virtual environment ---
if [ ! -d ".venv" ]; then
    echo "[1/4] Creating virtual environment..."
    # --without-pip avoids the ensurepip dependency (missing on some Ubuntu installs)
    python3 -m venv --without-pip .venv
    source .venv/bin/activate
    echo "      Bootstrapping pip..."
    PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    PIP_URL="https://bootstrap.pypa.io/pip/${PY_VER}/get-pip.py"
    # Fall back to latest if version-specific URL doesn't exist (3.9+ only needs generic)
    curl -sf "$PIP_URL" | python3 || curl -sS https://bootstrap.pypa.io/get-pip.py | python3
else
    echo "[1/4] Virtual environment already exists."
fi

source .venv/bin/activate
echo "      Python: $(python --version)"

# --- Detect CUDA and install torch ---
echo "[2/4] Installing Python dependencies..."

CUDA_VERSION=$(nvcc --version 2>/dev/null | grep "release" | sed 's/.*release \([0-9]*\.[0-9]*\).*/\1/' | tr -d '.')
if [ -z "$CUDA_VERSION" ]; then
    # Try nvidia-smi
    CUDA_VERSION=$(nvidia-smi 2>/dev/null | grep "CUDA Version" | sed 's/.*CUDA Version: \([0-9]*\.[0-9]*\).*/\1/' | tr -d '.')
fi

if [ -z "$CUDA_VERSION" ]; then
    echo "      No GPU detected — installing CPU-only torch."
    TORCH_INDEX="https://download.pytorch.org/whl/cpu"
else
    # Map cuda version to torch index: 118->cu118, 121->cu121, 124->cu124
    echo "      Detected CUDA $CUDA_VERSION"
    if [ "$CUDA_VERSION" -ge 124 ]; then
        TORCH_INDEX="https://download.pytorch.org/whl/cu124"
    elif [ "$CUDA_VERSION" -ge 121 ]; then
        TORCH_INDEX="https://download.pytorch.org/whl/cu121"
    else
        TORCH_INDEX="https://download.pytorch.org/whl/cu118"
    fi
fi

echo "      Using torch index: $TORCH_INDEX"
pip install --quiet torch torchvision --index-url "$TORCH_INDEX"

# Core ML packages — get exact torch version for PyG wheels
TORCH_VER=$(python -c "import torch; v=torch.__version__; print(v.split('+')[0])")
CUDA_TAG=$(python -c "import torch; d=torch.version.cuda; print('cu'+''.join(d.split('.')) if d else 'cpu')")
PYG_URL="https://data.pyg.org/whl/torch-${TORCH_VER}+${CUDA_TAG}.html"
echo "      PyG wheel URL: $PYG_URL"

pip install --quiet torch_geometric
pip install --quiet torch_scatter torch_sparse torch_cluster -f "$PYG_URL" 2>/dev/null || \
    pip install --quiet torch_scatter torch_sparse  # fallback: build from source

# Other dependencies
pip install --quiet \
    ogb \
    numpy pandas tqdm scikit-learn \
    torchmetrics pytorch-lightning \
    matplotlib seaborn \
    tabulate

echo "      Packages installed."

# --- Compile C++ binary ---
echo "[3/4] Compiling C++ binary..."
mkdir -p bin

# Check for g++
if ! command -v g++ &>/dev/null; then
    echo "      ERROR: g++ not found. Install with: sudo apt-get install g++"
    exit 1
fi

# The root Makefile has a Windows-specific 'mkdir' call — compile directly
g++ -std=c++17 -O3 -Wall -o bin/graph_prep HUB/main.cpp HUB/param.cpp
echo "      Built: bin/graph_prep"

# --- Check Java for AnyBURL ---
echo "[4/4] Checking Java (required for AnyBURL rule mining)..."
if command -v java &>/dev/null; then
    echo "      Java: $(java -version 2>&1 | head -1)"
else
    echo "      WARNING: Java not found. AnyBURL mining will fail."
    echo "      Install with: sudo apt-get install default-jdk"
fi

echo ""
echo "Setup complete. Run experiments with:"
echo "  ./run_server.sh"
echo ""
