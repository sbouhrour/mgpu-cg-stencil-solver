#!/bin/bash
# Benchmark script for single-GPU CSR vs STENCIL5 comparison
# Tests different matrix sizes to compare format performance
# Outputs: JSON and CSV files per config

set -e

# ============================================================
# CONFIGURATION - EDIT THIS
# ============================================================
RUNS=10

# Matrix sizes to test (grid_size for 5-point stencil)
MATRIX_SIZES=(
    "10000"     # 100M unknowns, 500M nnz
    "15000"     # 225M unknowns, 1.1B nnz
    "20000"     # 400M unknowns, 2.0B nnz
)

# Formats to benchmark
FORMATS=("cusparse-csr" "stencil5-csr")  # spmv_bench --mode values

# ============================================================
# Auto-detect configuration
# ============================================================
GPU_NAME=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader -i 0 | head -1 | tr -d ' ')
DATE=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="results_single_gpu_formats_${GPU_NAME}_${DATE}"
mkdir -p "$RESULTS_DIR"

echo "============================================================"
echo "Configuration:"
echo "  GPU: $GPU_NAME"
echo "  Commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "  Runs per config: $RUNS"
echo "  Matrix sizes: ${MATRIX_SIZES[*]}"
echo "  Formats: ${FORMATS[*]}"
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
if make spmv_bench > /dev/null 2>&1; then
    echo "✓ spmv_bench built"
else
    echo "✗ Build failed, showing errors:"
    make spmv_bench
    exit 1
fi

# Create matrix directory
mkdir -p matrix
echo "[2/2] Matrix directory ready"

# ============================================================
# Initialize summary file
# ============================================================
cat > "$SUMMARY_FILE" <<EOF
Single-GPU Format Comparison - CSR vs STENCIL5-OPT
==================================================
GPU: $GPU_NAME
Commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)
Runs per config: $RUNS
Date: $(date)

Matrix Sizes Tested:
EOF

for size in "${MATRIX_SIZES[@]}"; do
    echo "  ${size}×${size}" >> "$SUMMARY_FILE"
done

echo "" >> "$SUMMARY_FILE"

# ============================================================
# Main benchmark loop
# ============================================================
print_header "Single-GPU Format Benchmark"
echo "Start time: $(date)"

# Loop over matrix sizes
for SIZE in "${MATRIX_SIZES[@]}"; do
    print_header "Matrix Size: ${SIZE}×${SIZE}"

    MATRIX_FILE="matrix/stencil_${SIZE}x${SIZE}_stub.mtx"

    # Generate matrix if needed
    generate_matrix "$SIZE"

    # Calculate matrix stats
    UNKNOWNS=$((SIZE * SIZE))
    NNZ=$((5 * UNKNOWNS - 2 * SIZE))  # 5-point stencil

    # Write header to summary
    echo "" >> "$SUMMARY_FILE"
    echo "========================================" >> "$SUMMARY_FILE"
    echo "Matrix: ${SIZE}×${SIZE} (${UNKNOWNS} unknowns, ${NNZ} nnz)" >> "$SUMMARY_FILE"
    echo "========================================" >> "$SUMMARY_FILE"

    # Benchmark each format
    for FORMAT in "${FORMATS[@]}"; do
        print_section "Format: $FORMAT"

        # Generate output filenames
        # spmv_bench appends _<mode> to the name given with --json
        JSON_ARG="$RESULTS_DIR/${GPU_NAME}_${SIZE}x${SIZE}.json"
        JSON_FILE="$RESULTS_DIR/${GPU_NAME}_${SIZE}x${SIZE}_${FORMAT}.json"

        # Run benchmark
        echo "Running: ./bin/spmv_bench $MATRIX_FILE --mode=$FORMAT --json=$JSON_ARG"

        if (set -o pipefail; ./bin/spmv_bench "$MATRIX_FILE" --mode="$FORMAT" --json="$JSON_ARG" 2>&1 | tee -a "$SUMMARY_FILE"); then
            echo "✓ Test completed successfully"
            echo "  JSON: $JSON_FILE"

            # Extract key metrics
            if [ -f "$JSON_FILE" ]; then
                TIME=$(jq -r '.timing.median_ms // .benchmark.performance.execution_time_ms' "$JSON_FILE" 2>/dev/null || echo "N/A")
                GFLOPS=$(jq -r '.performance.gflops // .benchmark.performance.gflops' "$JSON_FILE" 2>/dev/null || echo "N/A")
                BW=$(jq -r '.performance.bandwidth_gb_s // .benchmark.performance.bandwidth_gb_s' "$JSON_FILE" 2>/dev/null || echo "N/A")
                echo "  Time: ${TIME} ms, GFLOPS: ${GFLOPS}, Bandwidth: ${BW} GB/s"
            fi
        else
            echo "✗ Test failed (exit code: $?)"
            echo "FAILED: Size=${SIZE}x${SIZE}, Format=$FORMAT" >> "$SUMMARY_FILE"
        fi

        # Small delay between tests
        sleep 1
    done

    echo "✓ Matrix size $SIZE complete"
done

# ============================================================
# Summary
# ============================================================
print_header "Benchmark Complete"

TOTAL_TESTS=$((${#MATRIX_SIZES[@]} * ${#FORMATS[@]}))

echo ""
echo "============================================================"
echo "Summary:"
echo "  GPU: $GPU_NAME"
echo "  Commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "  Matrix sizes tested: ${#MATRIX_SIZES[@]}"
echo "  Formats tested: ${#FORMATS[@]}"
echo "  Total tests: $TOTAL_TESTS"
echo ""
echo "Results directory: $RESULTS_DIR"
echo "  - Summary: $SUMMARY_FILE"
echo "  - JSON files: $TOTAL_TESTS files"
echo "============================================================"
echo "End time: $(date)"

# ============================================================
# Generate analysis script
# ============================================================
cat > "$RESULTS_DIR/analyze_formats.py" <<'PYTHON_EOF'
#!/usr/bin/env python3
"""
Analyze single-GPU format comparison results (CSR vs STENCIL5)
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

        # Handle both old and new JSON formats
        if 'benchmark' in data:
            # Old format
            size = int(data['benchmark']['matrix']['rows'] ** 0.5)
            time_ms = data['benchmark']['performance']['execution_time_ms']
            gflops = data['benchmark']['performance']['gflops']
            bw = data['benchmark']['performance']['bandwidth_gb_s']
            fmt = data['benchmark']['operator']
        else:
            # New format (CG solver)
            size = data['matrix']['grid_size']
            time_ms = data['timing']['median_ms']
            gflops = data.get('performance', {}).get('gflops', 0)
            bw = data.get('performance', {}).get('bandwidth_gb_s', 0)
            fmt = json_file.split('_')[-1].replace('.json', '')

        # Normalize format names
        if 'csr' in fmt.lower():
            fmt = 'CSR'
        elif 'stencil5' in fmt.lower():
            fmt = 'STENCIL5'

        results.append({
            'format': fmt,
            'size': size,
            'unknowns': size ** 2,
            'time_ms': time_ms,
            'gflops': gflops,
            'bandwidth_gb_s': bw
        })

# Group by format
by_format = {}
for r in results:
    if r['format'] not in by_format:
        by_format[r['format']] = []
    by_format[r['format']].append(r)

# Create comparison plots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

colors = {'CSR': '#2E86AB', 'STENCIL5-OPT': '#F18F01'}
markers = {'CSR': 'o', 'STENCIL5-OPT': 's'}

# Plot 1: Execution time vs matrix size
for fmt in sorted(by_format.keys()):
    data = sorted(by_format[fmt], key=lambda x: x['size'])
    sizes = [d['size'] for d in data]
    times = [d['time_ms'] for d in data]
    ax1.plot(sizes, times, marker=markers.get(fmt, 'o'), linestyle='-',
             label=fmt, color=colors.get(fmt, '#666666'),
             linewidth=2.5, markersize=10, markeredgewidth=1.5, markeredgecolor='white')

ax1.set_xlabel('Matrix Size (N×N)', fontsize=13, fontweight='bold')
ax1.set_ylabel('Execution Time (ms)', fontsize=13, fontweight='bold')
ax1.set_title('SpMV Time vs Problem Size', fontsize=14, fontweight='bold', pad=15)
ax1.legend(fontsize=12, frameon=True, shadow=True)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_ylim(bottom=0)

# Plot 2: GFLOPS vs matrix size
for fmt in sorted(by_format.keys()):
    data = sorted(by_format[fmt], key=lambda x: x['size'])
    sizes = [d['size'] for d in data]
    gflops = [d['gflops'] for d in data]
    ax2.plot(sizes, gflops, marker=markers.get(fmt, 'o'), linestyle='-',
             label=fmt, color=colors.get(fmt, '#666666'),
             linewidth=2.5, markersize=10, markeredgewidth=1.5, markeredgecolor='white')

ax2.set_xlabel('Matrix Size (N×N)', fontsize=13, fontweight='bold')
ax2.set_ylabel('GFLOPS', fontsize=13, fontweight='bold')
ax2.set_title('Computational Performance', fontsize=14, fontweight='bold', pad=15)
ax2.legend(fontsize=12, frameon=True, shadow=True)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.set_ylim(bottom=0)

# Plot 3: Bandwidth vs matrix size
for fmt in sorted(by_format.keys()):
    data = sorted(by_format[fmt], key=lambda x: x['size'])
    sizes = [d['size'] for d in data]
    bw = [d['bandwidth_gb_s'] for d in data]
    ax3.plot(sizes, bw, marker=markers.get(fmt, 'o'), linestyle='-',
             label=fmt, color=colors.get(fmt, '#666666'),
             linewidth=2.5, markersize=10, markeredgewidth=1.5, markeredgecolor='white')

ax3.set_xlabel('Matrix Size (N×N)', fontsize=13, fontweight='bold')
ax3.set_ylabel('Memory Bandwidth (GB/s)', fontsize=13, fontweight='bold')
ax3.set_title('Memory Performance', fontsize=14, fontweight='bold', pad=15)
ax3.legend(fontsize=12, frameon=True, shadow=True)
ax3.grid(True, alpha=0.3, linestyle='--')
ax3.set_ylim(bottom=0)

# Plot 4: Speedup (STENCIL5-OPT vs CSR)
csr_data = sorted(by_format.get('CSR', []), key=lambda x: x['size'])
stencil_data = sorted(by_format.get('STENCIL5-OPT', []), key=lambda x: x['size'])

if len(csr_data) == len(stencil_data):
    sizes = [d['size'] for d in csr_data]
    speedups = [csr_data[i]['time_ms'] / stencil_data[i]['time_ms']
                for i in range(len(csr_data))]

    ax4.plot(sizes, speedups, marker='o', linestyle='-',
             color='#C73E1D', linewidth=2.5, markersize=10,
             markeredgewidth=1.5, markeredgecolor='white')
    ax4.axhline(y=1.0, color='k', linestyle='--', linewidth=2, alpha=0.6, label='Baseline (CSR)')

    ax4.set_xlabel('Matrix Size (N×N)', fontsize=13, fontweight='bold')
    ax4.set_ylabel('Speedup vs CSR', fontsize=13, fontweight='bold')
    ax4.set_title('STENCIL5-OPT Speedup', fontsize=14, fontweight='bold', pad=15)
    ax4.legend(fontsize=12, frameon=True, shadow=True)
    ax4.grid(True, alpha=0.3, linestyle='--')
    ax4.set_ylim([0, max(speedups) * 1.2])

plt.tight_layout()
plt.savefig('format_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✅ Generated: format_comparison.png")

# Print summary table
print("\n" + "="*90)
print("Single-GPU Format Comparison")
print("="*90)
print(f"{'Size':<10} {'Format':<15} {'Time (ms)':<12} {'GFLOPS':<12} {'Bandwidth (GB/s)':<18}")
print("-"*90)

for size in sorted(set(r['size'] for r in results)):
    size_results = [r for r in results if r['size'] == size]
    for r in sorted(size_results, key=lambda x: x['format']):
        print(f"{r['size']:<10} {r['format']:<15} {r['time_ms']:<12.2f} {r['gflops']:<12.1f} {r['bandwidth_gb_s']:<18.1f}")

    # Calculate speedup
    csr = [r for r in size_results if r['format'] == 'CSR']
    stencil = [r for r in size_results if r['format'] == 'STENCIL5-OPT']
    if csr and stencil:
        speedup = csr[0]['time_ms'] / stencil[0]['time_ms']
        print(f"{'':10} {'Speedup:':<15} {speedup:<12.2f}×")
    print("-"*90)

print("="*90)
PYTHON_EOF

chmod +x "$RESULTS_DIR/analyze_formats.py"
echo ""
echo "Analysis script created: $RESULTS_DIR/analyze_formats.py"
echo "Run: cd $RESULTS_DIR && python3 analyze_formats.py"
