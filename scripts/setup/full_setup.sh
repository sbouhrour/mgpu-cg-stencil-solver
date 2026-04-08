#!/bin/bash
# Complete setup: CUDA project + optional AmgX
# Auto-detects GPU architecture
#
# Usage:
#   ./full_setup.sh           # Install main project only (default)
#   ./full_setup.sh --amgx    # Install main + AmgX (for comparison benchmarks)

set -e

INSTALL_AMGX=false

for arg in "$@"; do
    case $arg in
        --amgx) INSTALL_AMGX=true ;;
        --help|-h)
            echo "Usage: $0 [--amgx]"
            echo "  --amgx    Install AmgX for reference comparison"
            exit 0
            ;;
        *) echo "Unknown option: $arg"; echo "Use: --amgx or --help"; exit 1 ;;
    esac
done

echo "=========================================="
echo "Setup Configuration"
echo "=========================================="
echo "Main project: YES"
echo "AmgX:         $($INSTALL_AMGX && echo YES || echo NO)"
echo "=========================================="
echo ""

# Detect GPU
echo "Detecting GPU..."
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
GPU_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
echo "GPU: $GPU_NAME (Compute Capability $GPU_ARCH)"
echo ""

# Verify we're in project root
if [ ! -f "Makefile" ] || [ ! -d "src" ]; then
    echo "Error: Must be run from project root directory"
    echo "Usage: cd mgpu-cg-stencil-solver && ./scripts/setup/full_setup.sh"
    exit 1
fi

# 1. Install system dependencies
echo "Step 1: Installing system dependencies..."
if command -v apt-get &> /dev/null; then
    apt-get update -qq
    apt-get install -y build-essential git cmake openmpi-bin libopenmpi-dev
fi

# 2. Build main project
echo ""
echo "Step 2: Building main project..."
make clean
make -j$(nproc)

# Build multi-GPU solver if MPI available
if command -v mpic++ &> /dev/null; then
    make cg_solver_mgpu_stencil
fi

# 3. Install AmgX (optional) - also builds our AmgX benchmarks
if [ "$INSTALL_AMGX" = true ]; then
    echo ""
    echo "Step 3: Installing AmgX..."
    ./scripts/setup/install_amgx.sh
fi

# Verification
echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo "GPU: $GPU_NAME"
echo ""
echo "Built executables:"
ls -lh bin/spmv_bench bin/cg_solver_mgpu_stencil 2>/dev/null || true

if [ "$INSTALL_AMGX" = true ]; then
    ls -lh external/benchmarks/amgx/amgx_cg_solver 2>/dev/null || true
fi

echo ""
echo "=========================================="
echo "Quick start:"
echo "  ./scripts/run_all.sh --quick"
echo "=========================================="
