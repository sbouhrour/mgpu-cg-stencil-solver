#!/bin/bash
# Benchmark script for multi-GPU CG solver with varying problem sizes
# Tests different matrix sizes for each GPU count (1, 2, 4, 8)
# Outputs: JSON and CSV files per config

set -e

# ============================================================
# CONFIGURATION - EDIT THIS
# ============================================================
RUNS=10

# Problem sizes to test per GPU count
# Format: "gpu_count:matrix_size1,matrix_size2,..."
# Same sizes for all GPU counts to enable strong scaling analysis
PROBLEM_CONFIGS=(
    "1:10000,15000,20000"
    "2:10000,15000,20000"
    "4:10000,15000,20000"
    "8:10000,15000,20000"
)

# Alternative: Weak scaling (constant work per GPU)
# Uncomment to use this instead:
# PROBLEM_CONFIGS_WEAK=(
#     "1:5000"      # 25M unknowns (baseline)
#     "2:7071"      # ~50M unknowns (2× baseline)
#     "4:10000"     # 100M unknowns (4× baseline)
#     "8:14142"     # ~200M unknowns (8× baseline)
# )

# ============================================================
# Auto-detect configuration
# ============================================================
GPU_NAME=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader -i 0 | head -1 | tr -d ' ')
DATE=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="results_problem_size_scaling_${GPU_NAME}_${DATE}"
mkdir -p "$RESULTS_DIR"

echo "============================================================"
echo "Configuration:"
echo "  GPU: $GPU_NAME"
echo "  Commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "  Runs per config: $RUNS"
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

echo "✓ Build successful"

# Create matrix directory
mkdir -p matrix
echo "[2/2] Matrix directory ready"

# ============================================================
# Initialize summary file
# ============================================================
cat > "$SUMMARY_FILE" <<EOF
Multi-GPU CG Solver - Problem Size Scaling Benchmark
=====================================================
GPU: $GPU_NAME
Commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)
Runs per config: $RUNS
Date: $(date)

Problem Size Configurations:
EOF

for config in "${PROBLEM_CONFIGS[@]}"; do
    echo "  $config" >> "$SUMMARY_FILE"
done

echo "" >> "$SUMMARY_FILE"

# ============================================================
# Main benchmark loop
# ============================================================
print_header "Problem Size Scaling Benchmark"
echo "Start time: $(date)"

# Parse and run each configuration
for config in "${PROBLEM_CONFIGS[@]}"; do
    # Split config into GPU count and sizes
    NP=$(echo "$config" | cut -d':' -f1)
    SIZES=$(echo "$config" | cut -d':' -f2)

    print_header "GPU COUNT: $NP"

    # Split sizes by comma
    IFS=',' read -ra SIZE_ARRAY <<< "$SIZES"

    for SIZE in "${SIZE_ARRAY[@]}"; do
        print_section "Matrix Size: ${SIZE}×${SIZE}"

        MATRIX_FILE="matrix/stencil_${SIZE}x${SIZE}_stub.mtx"

        # Generate matrix if needed
        generate_matrix "$SIZE"

        # Calculate matrix stats for filename
        UNKNOWNS=$((SIZE * SIZE))
        SIZE_MB=$((UNKNOWNS * 8 / 1024 / 1024))  # Rough estimate

        # Generate output filenames
        BASE_NAME="${GPU_NAME}_${SIZE}x${SIZE}_main_np${NP}"
        JSON_FILE="$RESULTS_DIR/${BASE_NAME}.json"
        CSV_FILE="$RESULTS_DIR/${BASE_NAME}.csv"

        # Write header to summary
        echo "" >> "$SUMMARY_FILE"
        echo "========================================" >> "$SUMMARY_FILE"
        echo "GPUs: $NP | Matrix: ${SIZE}×${SIZE} (${UNKNOWNS} unknowns)" >> "$SUMMARY_FILE"
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

    echo "✓ GPU count $NP complete"
done

# ============================================================
# Summary
# ============================================================
print_header "Benchmark Complete"

# Count total tests
TOTAL_TESTS=0
for config in "${PROBLEM_CONFIGS[@]}"; do
    SIZES=$(echo "$config" | cut -d':' -f2)
    IFS=',' read -ra SIZE_ARRAY <<< "$SIZES"
    TOTAL_TESTS=$((TOTAL_TESTS + ${#SIZE_ARRAY[@]}))
done

echo ""
echo "============================================================"
echo "Summary:"
echo "  GPU: $GPU_NAME"
echo "  Commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "  GPU counts tested: 1, 2, 4, 8"
echo "  Total tests: $TOTAL_TESTS"
echo ""
echo "Results directory: $RESULTS_DIR"
echo "  - Summary: $SUMMARY_FILE"
echo "  - JSON files: $TOTAL_TESTS files"
echo "  - CSV files:  $TOTAL_TESTS files"
echo "============================================================"
echo "End time: $(date)"

# ============================================================
# Generate scaling analysis script
# ============================================================
cat > "$RESULTS_DIR/analyze_scaling.py" <<'PYTHON_EOF'
#!/usr/bin/env python3
"""
Analyze strong scaling results (same problem sizes on different GPU counts)
"""
import json
import glob
import matplotlib.pyplot as plt
import numpy as np

# Read all JSON files
results = []
for json_file in sorted(glob.glob("*.json")):
    with open(json_file) as f:
        data = json.load(f)
        results.append({
            'gpus': data['num_gpus'],
            'size': data['matrix']['rows'],
            'unknowns': data['matrix']['rows'] ** 2,
            'iterations': data['convergence']['iterations'],
            'time_ms': data['timing']['median_ms'],
            'time_per_iter': data['timing']['median_ms'] / data['convergence']['iterations']
        })

# Group by problem size (strong scaling)
by_size = {}
for r in results:
    if r['size'] not in by_size:
        by_size[r['size']] = []
    by_size[r['size']].append(r)

# Create strong scaling plots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

colors = {10000: '#2E86AB', 15000: '#A23B72', 20000: '#F18F01'}
markers = {10000: 'o', 15000: 's', 20000: '^'}

# Plot 1: Total time vs GPU count (strong scaling)
for size in sorted(by_size.keys()):
    data = sorted(by_size[size], key=lambda x: x['gpus'])
    gpus = [d['gpus'] for d in data]
    times = [d['time_ms'] for d in data]
    ax1.plot(gpus, times, marker=markers.get(size, 'o'), linestyle='-',
             label=f'{size}×{size}', color=colors.get(size, '#666666'),
             linewidth=2, markersize=10)

ax1.set_xlabel('Number of GPUs', fontsize=12, fontweight='bold')
ax1.set_ylabel('Total Time (ms)', fontsize=12, fontweight='bold')
ax1.set_title('Strong Scaling: Total Time', fontsize=13, fontweight='bold')
ax1.set_xticks([1, 2, 4, 8])
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# Plot 2: Speedup vs GPU count
for size in sorted(by_size.keys()):
    data = sorted(by_size[size], key=lambda x: x['gpus'])
    gpus = [d['gpus'] for d in data]
    baseline = data[0]['time_ms']  # 1 GPU time
    speedups = [baseline / d['time_ms'] for d in data]
    ax2.plot(gpus, speedups, marker=markers.get(size, 'o'), linestyle='-',
             label=f'{size}×{size}', color=colors.get(size, '#666666'),
             linewidth=2, markersize=10)

# Ideal scaling line
ax2.plot([1, 2, 4, 8], [1, 2, 4, 8], 'k--', linewidth=1.5, label='Ideal', alpha=0.5)
ax2.set_xlabel('Number of GPUs', fontsize=12, fontweight='bold')
ax2.set_ylabel('Speedup', fontsize=12, fontweight='bold')
ax2.set_title('Strong Scaling: Speedup', fontsize=13, fontweight='bold')
ax2.set_xticks([1, 2, 4, 8])
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

# Plot 3: Parallel efficiency
for size in sorted(by_size.keys()):
    data = sorted(by_size[size], key=lambda x: x['gpus'])
    gpus = [d['gpus'] for d in data]
    baseline = data[0]['time_ms']
    efficiencies = [100 * baseline / (d['time_ms'] * d['gpus']) for d in data]
    ax3.plot(gpus, efficiencies, marker=markers.get(size, 'o'), linestyle='-',
             label=f'{size}×{size}', color=colors.get(size, '#666666'),
             linewidth=2, markersize=10)

ax3.axhline(y=100, color='k', linestyle='--', linewidth=1.5, alpha=0.5, label='Ideal')
ax3.set_xlabel('Number of GPUs', fontsize=12, fontweight='bold')
ax3.set_ylabel('Parallel Efficiency (%)', fontsize=12, fontweight='bold')
ax3.set_title('Strong Scaling: Efficiency', fontsize=13, fontweight='bold')
ax3.set_xticks([1, 2, 4, 8])
ax3.set_ylim([0, 110])
ax3.legend(fontsize=11)
ax3.grid(True, alpha=0.3)

# Plot 4: Time per iteration
for size in sorted(by_size.keys()):
    data = sorted(by_size[size], key=lambda x: x['gpus'])
    gpus = [d['gpus'] for d in data]
    time_per_iter = [d['time_per_iter'] for d in data]
    ax4.plot(gpus, time_per_iter, marker=markers.get(size, 'o'), linestyle='-',
             label=f'{size}×{size}', color=colors.get(size, '#666666'),
             linewidth=2, markersize=10)

ax4.set_xlabel('Number of GPUs', fontsize=12, fontweight='bold')
ax4.set_ylabel('Time per Iteration (ms)', fontsize=12, fontweight='bold')
ax4.set_title('Iteration Cost vs GPU Count', fontsize=13, fontweight='bold')
ax4.set_xticks([1, 2, 4, 8])
ax4.legend(fontsize=11)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('strong_scaling_analysis.png', dpi=300, bbox_inches='tight')
print("✅ Generated: strong_scaling_analysis.png")

# Print table
print("\nStrong Scaling Results")
print("="*90)
print(f"{'Size':<10} {'GPUs':<6} {'Unknowns':<12} {'Iter':<6} {'Time (ms)':<12} {'Speedup':<10} {'Efficiency':<12}")
print("-"*90)
for size in sorted(by_size.keys()):
    data = sorted(by_size[size], key=lambda x: x['gpus'])
    baseline = data[0]['time_ms']
    for r in data:
        speedup = baseline / r['time_ms']
        efficiency = 100 * speedup / r['gpus']
        print(f"{r['size']:<10} {r['gpus']:<6} {r['unknowns']:<12} {r['iterations']:<6} "
              f"{r['time_ms']:<12.2f} {speedup:<10.2f} {efficiency:<12.1f}%")
    print("-"*90)
print("="*90)
PYTHON_EOF

chmod +x "$RESULTS_DIR/analyze_scaling.py"
echo ""
echo "Analysis script created: $RESULTS_DIR/analyze_scaling.py"
echo "Run: cd $RESULTS_DIR && python3 analyze_scaling.py"
