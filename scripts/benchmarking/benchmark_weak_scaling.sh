#!/bin/bash
# Weak scaling benchmark for multi-GPU CG solver
# Constant work per GPU: each GPU handles ~25M unknowns
# Tests: 1, 2, 4, 8 GPUs with proportionally larger problems
# Outputs: JSON and CSV files per config

set -e

# ============================================================
# CONFIGURATION - EDIT THIS
# ============================================================
RUNS=10

# Weak scaling: constant work per GPU (~25M unknowns per GPU)
# Format: "gpu_count:grid_size"
WEAK_SCALING_CONFIGS=(
    "1:5000"      # 25M unknowns (baseline)
    "2:7071"      # ~50M unknowns (2× baseline)
    "4:10000"     # 100M unknowns (4× baseline)
    "8:14142"     # ~200M unknowns (8× baseline)
)

# ============================================================
# Auto-detect configuration
# ============================================================
GPU_NAME=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader -i 0 | head -1 | tr -d ' ')
# GPUs this process may use: the entries of CUDA_VISIBLE_DEVICES when it is set, else every GPU
# nvidia-smi lists (nvidia-smi ignores CUDA_VISIBLE_DEVICES)
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
else
    NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
fi
DATE=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="results_weak_scaling_${GPU_NAME}_${DATE}"
mkdir -p "$RESULTS_DIR"

echo "============================================================"
echo "Configuration:"
echo "  GPU: $GPU_NAME"
echo "  Commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "  Runs per config: $RUNS"
echo "  Scaling type: WEAK (constant work per GPU)"
echo "  Results dir: $RESULTS_DIR"
echo "============================================================"

# Summary file
SUMMARY_FILE="$RESULTS_DIR/summary.txt"

# ============================================================
# Helper functions
# ============================================================
print_header() {
    echo ""
    echo "============================================================"
    echo "$1"
    echo "============================================================"
}

print_section() {
    echo ""
    echo "------------------------------------------------------------"
    echo "$1"
    echo "------------------------------------------------------------"
}

generate_matrix() {
    local size=$1
    local matrix_file="matrix/stencil_${size}x${size}_stub.mtx"

    # Header-only file: the binaries generate the 5-point matrix in memory (no multi-GB text file)
    if [ ! -f "$matrix_file" ]; then
        echo "% STENCIL_GRID_SIZE ${size}" > "$matrix_file"
        echo "✓ Matrix stub written: $matrix_file"
    else
        echo "✓ Matrix exists: $matrix_file"
    fi
}

# ============================================================
# Verify/build tools
# ============================================================
print_header "Setup"

# Build binaries
echo "[1/2] Building binaries (incremental)..."
if make cg_solver_mgpu_stencil > /dev/null 2>&1; then
    echo "✓ cg_solver_mgpu_stencil built"
else
    echo "✗ Build failed, showing errors:"
    make cg_solver_mgpu_stencil
    exit 1
fi

# Create matrix directory
mkdir -p matrix
echo "[2/2] Matrix directory ready"

# ============================================================
# Initialize summary file
# ============================================================
cat > "$SUMMARY_FILE" <<EOF
Multi-GPU CG Solver - Weak Scaling Benchmark
=============================================
GPU: $GPU_NAME
Commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)
Runs per config: $RUNS
Date: $(date)

Weak Scaling Configurations (constant ~25M unknowns per GPU):
EOF

for config in "${WEAK_SCALING_CONFIGS[@]}"; do
    NP=$(echo "$config" | cut -d':' -f1)
    SIZE=$(echo "$config" | cut -d':' -f2)
    UNKNOWNS=$((SIZE * SIZE))
    UNKNOWNS_PER_GPU=$((UNKNOWNS / NP))
    echo "  ${NP} GPUs: ${SIZE}×${SIZE} (${UNKNOWNS} unknowns, ${UNKNOWNS_PER_GPU} per GPU)" >> "$SUMMARY_FILE"
done

echo "" >> "$SUMMARY_FILE"

# ============================================================
# Main benchmark loop
# ============================================================
print_header "Weak Scaling Benchmark"
echo "Start time: $(date)"

# Parse and run each configuration
for config in "${WEAK_SCALING_CONFIGS[@]}"; do
    # Split config into GPU count and size
    NP=$(echo "$config" | cut -d':' -f1)
    if [ "$NP" -gt "$NUM_GPUS" ]; then
        echo "Skipping $NP GPUs (only $NUM_GPUS available)" | tee -a "$SUMMARY_FILE"
        continue
    fi
    SIZE=$(echo "$config" | cut -d':' -f2)

    print_header "GPU COUNT: $NP | Matrix Size: ${SIZE}×${SIZE}"

    MATRIX_FILE="matrix/stencil_${SIZE}x${SIZE}_stub.mtx"

    # Generate matrix if needed
    generate_matrix "$SIZE"

    # Calculate matrix stats
    UNKNOWNS=$((SIZE * SIZE))
    UNKNOWNS_PER_GPU=$((UNKNOWNS / NP))

    # Generate output filenames
    BASE_NAME="${GPU_NAME}_${SIZE}x${SIZE}_main_np${NP}"
    JSON_FILE="$RESULTS_DIR/${BASE_NAME}.json"
    CSV_FILE="$RESULTS_DIR/${BASE_NAME}.csv"

    # Write header to summary
    echo "" >> "$SUMMARY_FILE"
    echo "========================================" >> "$SUMMARY_FILE"
    echo "GPUs: $NP | Matrix: ${SIZE}×${SIZE} (${UNKNOWNS} unknowns, ${UNKNOWNS_PER_GPU} per GPU)" >> "$SUMMARY_FILE"
    echo "Files: ${BASE_NAME}.{json,csv}" >> "$SUMMARY_FILE"
    echo "========================================" >> "$SUMMARY_FILE"

    # Run benchmark
    echo "Running: mpirun -np $NP ./bin/cg_solver_mgpu_stencil $MATRIX_FILE --json=$JSON_FILE --csv=$CSV_FILE"

    if (set -o pipefail; mpirun --allow-run-as-root -np "$NP" ./bin/cg_solver_mgpu_stencil "$MATRIX_FILE" \
        --json="$JSON_FILE" --csv="$CSV_FILE" 2>&1 | tee -a "$SUMMARY_FILE"); then
        echo "✓ Test completed successfully"
        echo "  JSON: $JSON_FILE"
        echo "  CSV:  $CSV_FILE"
    else
        echo "✗ Test failed (exit code: $?)"
        echo "FAILED: GPUs=$NP, Size=${SIZE}x${SIZE}" >> "$SUMMARY_FILE"
    fi

    # Small delay between tests
    sleep 2
done

# ============================================================
# Summary
# ============================================================
print_header "Benchmark Complete"

TOTAL_TESTS=${#WEAK_SCALING_CONFIGS[@]}

echo ""
echo "============================================================"
echo "Summary:"
echo "  GPU: $GPU_NAME"
echo "  Commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "  Scaling type: WEAK (constant work per GPU)"
echo "  Total tests: $TOTAL_TESTS"
echo ""
echo "Results directory: $RESULTS_DIR"
echo "  - Summary: $SUMMARY_FILE"
echo "  - JSON files: $TOTAL_TESTS files"
echo "  - CSV files:  $TOTAL_TESTS files"
echo "============================================================"
echo "End time: $(date)"

# ============================================================
# Generate analysis script
# ============================================================
cat > "$RESULTS_DIR/analyze_weak_scaling.py" <<'PYTHON_EOF'
#!/usr/bin/env python3
"""
Analyze weak scaling results (constant work per GPU)
"""
import json
import glob
import matplotlib.pyplot as plt
import numpy as np

# Read all JSON files
results = []
for json_file in sorted(glob.glob("*.json")):
    with open(json_file) as f:
        # Read and fix inf values
        content = f.read().replace(': inf', ': null')
        data = json.loads(content)

        # Use grid_size (not rows)
        grid_size = data['matrix']['grid_size']
        results.append({
            'gpus': data['num_gpus'],
            'size': grid_size,
            'unknowns': data['matrix']['rows'],
            'unknowns_per_gpu': data['matrix']['rows'] // data['num_gpus'],
            'iterations': data['convergence']['iterations'],
            'time_ms': data['timing']['median_ms'],
            'time_per_iter': data['timing']['median_ms'] / data['convergence']['iterations']
        })

# Sort by GPU count
results = sorted(results, key=lambda x: x['gpus'])

# Create weak scaling plots (2-panel)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

colors = '#2E86AB'
gpus_list = [r['gpus'] for r in results]
times = [r['time_ms'] for r in results]
sizes = [r['size'] for r in results]

# Plot 1: Time vs GPU count (should be ~constant for perfect weak scaling)
baseline_time = results[0]['time_ms']
ax1.plot(gpus_list, times, marker='o', linestyle='-',
         color=colors, linewidth=2.5, markersize=10,
         markeredgewidth=1.5, markeredgecolor='white', label='Measured')
ax1.axhline(y=baseline_time, color='k', linestyle='--', linewidth=2,
            alpha=0.6, label=f'Ideal ({baseline_time:.1f} ms)')

ax1.set_xlabel('Number of GPUs', fontsize=13, fontweight='bold')
ax1.set_ylabel('Total Time (ms)', fontsize=13, fontweight='bold')
ax1.set_title('Weak Scaling: Time vs GPU Count', fontsize=14, fontweight='bold', pad=15)
ax1.set_xticks(gpus_list)
ax1.legend(fontsize=11, frameon=True, shadow=True)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_ylim([0, max(times) * 1.2])

# Plot 2: Weak scaling efficiency
efficiencies = [100 * baseline_time / t for t in times]
ax2.bar(range(len(gpus_list)), efficiencies, color=colors, alpha=0.8,
        edgecolor='black', linewidth=1.5)
ax2.axhline(y=100, color='k', linestyle='--', linewidth=2, alpha=0.6, label='Ideal')

ax2.set_xticks(range(len(gpus_list)))
ax2.set_xticklabels([f'{g} GPU{"s" if g>1 else ""}' for g in gpus_list])
ax2.set_ylabel('Weak Scaling Efficiency (%)', fontsize=13, fontweight='bold')
ax2.set_title('Weak Scaling Efficiency', fontsize=14, fontweight='bold', pad=15)
ax2.legend(fontsize=11, frameon=True, shadow=True)
ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
ax2.set_ylim([0, 110])

# Add value labels on bars
for i, (eff, time) in enumerate(zip(efficiencies, times)):
    ax2.text(i, eff + 2, f'{eff:.1f}%\n({time:.1f}ms)',
             ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('weak_scaling_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✅ Generated: weak_scaling_analysis.png")

# Print summary table
print("\n" + "="*100)
print("Weak Scaling Results (Constant ~25M unknowns per GPU)")
print("="*100)
print(f"{'GPUs':<6} {'Size':<10} {'Total Unknowns':<15} {'Per GPU':<12} {'Iter':<6} {'Time (ms)':<12} {'Efficiency':<12}")
print("-"*100)

for r in results:
    efficiency = 100 * baseline_time / r['time_ms']
    print(f"{r['gpus']:<6} {r['size']:<10} {r['unknowns']:<15,} {r['unknowns_per_gpu']:<12,} "
          f"{r['iterations']:<6} {r['time_ms']:<12.2f} {efficiency:<12.1f}%")

print("-"*100)
print(f"\nBaseline (1 GPU): {baseline_time:.2f} ms")
print(f"Ideal weak scaling: constant time ({baseline_time:.2f} ms) for all GPU counts")
print("="*100)
PYTHON_EOF

chmod +x "$RESULTS_DIR/analyze_weak_scaling.py"
echo ""
echo "Analysis script created: $RESULTS_DIR/analyze_weak_scaling.py"
echo "Run: cd $RESULTS_DIR && python3 analyze_weak_scaling.py"
