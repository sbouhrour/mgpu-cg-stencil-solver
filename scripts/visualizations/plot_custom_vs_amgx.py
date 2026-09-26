#!/usr/bin/env python3
"""
Generate comparison plots: Custom CG vs NVIDIA AmgX
Shows single-GPU performance advantage and equivalent multi-GPU scaling
"""

import matplotlib.pyplot as plt
import numpy as np

# Data: [10k, 15k, 20k] problem sizes
sizes = ['10k×10k\n(100M)', '15k×15k\n(225M)', '20k×20k\n(400M)']
size_labels = ['10k×10k', '15k×15k', '20k×20k']

# Times (ms)
custom_1gpu = [133.9, 300.1, 531.4]
custom_8gpu = [19.3, 40.4, 71.0]
amgx_1gpu = [188.7, 420.0, 746.7]
amgx_8gpu = [27.0, 57.0, 102.3]

# Speedups
custom_speedup = [6.94, 7.43, 7.48]
amgx_speedup = [6.99, 7.36, 7.30]

# Efficiency (%)
custom_efficiency = [86.8, 92.9, 93.5]
amgx_efficiency = [87.4, 92.0, 91.3]

# Colors
color_custom = '#2E86AB'  # Blue
color_amgx = '#A23B72'    # Purple/magenta
color_grid = '#E8E8E8'

# Font sizes
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

# ============================================================================
# Figure 1: Performance Comparison (1 GPU vs 8 GPUs)
# ============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

x = np.arange(len(sizes))
width = 0.35

# Subplot 1: Single-GPU Performance
bars1 = ax1.bar(x - width/2, custom_1gpu, width, label='Custom CG',
                color=color_custom, edgecolor='black', linewidth=0.8)
bars2 = ax1.bar(x + width/2, amgx_1gpu, width, label='NVIDIA AmgX',
                color=color_amgx, edgecolor='black', linewidth=0.8)

ax1.set_xlabel('Problem Size', fontweight='bold')
ax1.set_ylabel('Time (ms)', fontweight='bold')
ax1.set_title('Single-GPU Performance\nCustom 1.40-1.41× Faster', fontweight='bold', pad=15)
ax1.set_xticks(x)
ax1.set_xticklabels(sizes)
ax1.legend(loc='upper left')
ax1.grid(axis='y', alpha=0.3, linestyle='--', color=color_grid)
ax1.set_axisbelow(True)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=9)

# Subplot 2: Multi-GPU Performance (8 GPUs)
bars3 = ax2.bar(x - width/2, custom_8gpu, width, label='Custom CG',
                color=color_custom, edgecolor='black', linewidth=0.8)
bars4 = ax2.bar(x + width/2, amgx_8gpu, width, label='NVIDIA AmgX',
                color=color_amgx, edgecolor='black', linewidth=0.8)

ax2.set_xlabel('Problem Size', fontweight='bold')
ax2.set_ylabel('Time (ms)', fontweight='bold')
ax2.set_title('8-GPU Performance\nCustom Maintains Advantage', fontweight='bold', pad=15)
ax2.set_xticks(x)
ax2.set_xticklabels(sizes)
ax2.legend(loc='upper left')
ax2.grid(axis='y', alpha=0.3, linestyle='--', color=color_grid)
ax2.set_axisbelow(True)

# Add value labels on bars
for bars in [bars3, bars4]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('docs/figures/custom_vs_amgx_performance.png', dpi=300, bbox_inches='tight')
print('✓ Generated: docs/figures/custom_vs_amgx_performance.png')
plt.close()

# ============================================================================
# Figure 2: Scaling Comparison
# ============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Subplot 1: Speedup Comparison
bars1 = ax1.bar(x - width/2, custom_speedup, width, label='Custom CG',
                color=color_custom, edgecolor='black', linewidth=0.8)
bars2 = ax1.bar(x + width/2, amgx_speedup, width, label='NVIDIA AmgX',
                color=color_amgx, edgecolor='black', linewidth=0.8)

ax1.axhline(y=8, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Perfect Scaling (8×)')
ax1.set_xlabel('Problem Size', fontweight='bold')
ax1.set_ylabel('Speedup (8 GPUs vs 1 GPU)', fontweight='bold')
ax1.set_title('Multi-GPU Scaling', fontweight='bold', pad=15)
ax1.set_xticks(x)
ax1.set_xticklabels(sizes)
ax1.legend(loc='lower right')
ax1.grid(axis='y', alpha=0.3, linestyle='--', color=color_grid)
ax1.set_axisbelow(True)
ax1.set_ylim(0, 9)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}×',
                ha='center', va='bottom', fontsize=9)

# Subplot 2: Efficiency Comparison
bars3 = ax2.bar(x - width/2, custom_efficiency, width, label='Custom CG',
                color=color_custom, edgecolor='black', linewidth=0.8)
bars4 = ax2.bar(x + width/2, amgx_efficiency, width, label='NVIDIA AmgX',
                color=color_amgx, edgecolor='black', linewidth=0.8)

ax2.axhline(y=100, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Perfect Efficiency')
ax2.set_xlabel('Problem Size', fontweight='bold')
ax2.set_ylabel('Parallel Efficiency (%)', fontweight='bold')
ax2.set_title('Scaling Efficiency\n87-94% on Both', fontweight='bold', pad=15)
ax2.set_xticks(x)
ax2.set_xticklabels(sizes)
ax2.legend(loc='lower right')
ax2.grid(axis='y', alpha=0.3, linestyle='--', color=color_grid)
ax2.set_axisbelow(True)
ax2.set_ylim(0, 110)

# Add value labels
for bars in [bars3, bars4]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('docs/figures/custom_vs_amgx_scaling.png', dpi=300, bbox_inches='tight')
print('✓ Generated: docs/figures/custom_vs_amgx_scaling.png')
plt.close()

# ============================================================================
# Figure 3: Combined Overview (Single figure for README)
# ============================================================================
fig = plt.figure(figsize=(16, 6))
gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

# Top row: Performance bars
ax1 = fig.add_subplot(gs[0, :2])
ax2 = fig.add_subplot(gs[1, :2])

# Subplot 1: Absolute Time Comparison
x_pos = np.arange(len(size_labels))
width_overview = 0.2

bars1 = ax1.bar(x_pos - 1.5*width_overview, custom_1gpu, width_overview,
                label='Custom 1 GPU', color=color_custom, alpha=0.9, edgecolor='black', linewidth=0.8)
bars2 = ax1.bar(x_pos - 0.5*width_overview, custom_8gpu, width_overview,
                label='Custom 8 GPUs', color=color_custom, alpha=0.6, edgecolor='black', linewidth=0.8)
bars3 = ax1.bar(x_pos + 0.5*width_overview, amgx_1gpu, width_overview,
                label='AmgX 1 GPU', color=color_amgx, alpha=0.9, edgecolor='black', linewidth=0.8)
bars4 = ax1.bar(x_pos + 1.5*width_overview, amgx_8gpu, width_overview,
                label='AmgX 8 GPUs', color=color_amgx, alpha=0.6, edgecolor='black', linewidth=0.8)

ax1.set_ylabel('Time (ms)', fontweight='bold')
ax1.set_title('Absolute Performance: Custom CG vs NVIDIA AmgX', fontweight='bold', fontsize=13, pad=10)
ax1.set_xticks(x_pos)
ax1.set_xticklabels(size_labels)
ax1.legend(ncol=4, loc='upper left', framealpha=0.9)
ax1.grid(axis='y', alpha=0.3, linestyle='--', color=color_grid)
ax1.set_axisbelow(True)

# Subplot 2: Speedup & Efficiency
x_pos2 = np.arange(len(size_labels))
width2 = 0.35

bars5 = ax2.bar(x_pos2 - width2/2, custom_speedup, width2,
                label='Custom Speedup', color=color_custom, edgecolor='black', linewidth=0.8)
bars6 = ax2.bar(x_pos2 + width2/2, amgx_speedup, width2,
                label='AmgX Speedup', color=color_amgx, edgecolor='black', linewidth=0.8)

ax2.axhline(y=8, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Perfect Scaling (8×)')
ax2.set_xlabel('Problem Size', fontweight='bold')
ax2.set_ylabel('Speedup (8 GPUs)', fontweight='bold')
ax2.set_title('Multi-GPU Scaling: 6.9-7.5× on 8 GPUs', fontweight='bold', fontsize=13, pad=10)
ax2.set_xticks(x_pos2)
ax2.set_xticklabels(size_labels)
ax2.legend(loc='lower right', framealpha=0.9)
ax2.grid(axis='y', alpha=0.3, linestyle='--', color=color_grid)
ax2.set_axisbelow(True)
ax2.set_ylim(0, 9)

# Add speedup value labels
for bars in [bars5, bars6]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.2f}×',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

# Right column: Key Metrics Text Box
ax3 = fig.add_subplot(gs[:, 2])
ax3.axis('off')

summary_text = """
KEY FINDINGS

Single GPU:
• Custom CG 1.40-1.41× faster
• 134-531 ms (Custom)
• 189-747 ms (AmgX)

8 GPUs:
• Custom 6.9-7.5×, AmgX 7.0-7.4×
• 87-94% efficiency for both

Where the gain comes from:
• Stencil SpMV: CSR with
  computed column indices
• Faster vector operations

Same algorithm on both sides:
• Unpreconditioned CG
• Same matrix, same tolerance
"""

ax3.text(0.1, 0.95, summary_text, transform=ax3.transAxes,
         fontsize=11, verticalalignment='top', family='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3, pad=1))

plt.savefig('docs/figures/custom_vs_amgx_overview.png', dpi=300, bbox_inches='tight')
print('✓ Generated: docs/figures/custom_vs_amgx_overview.png')
plt.close()

print('\n✓ All comparison plots generated successfully!')
print('  - custom_vs_amgx_performance.png (side-by-side bars)')
print('  - custom_vs_amgx_scaling.png (speedup + efficiency)')
print('  - custom_vs_amgx_overview.png (comprehensive single figure)')
