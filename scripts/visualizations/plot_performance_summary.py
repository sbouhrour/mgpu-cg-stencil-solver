#!/usr/bin/env python3
"""
Generate performance summary plot: All custom implementations vs references
Shows speedup gains in a single digestible visualization
"""

import matplotlib.pyplot as plt
import numpy as np

# Data: Speedup = Custom / Reference (>1.0 means custom is faster)
categories = ['SpMV\nSingle-GPU', 'CG\nSingle-GPU', 'CG\n8 GPUs']

# Speedup for each problem size
speedups_10k = [
    2.08,  # SpMV: STENCIL5 vs cuSPARSE CSR (6.77/3.25)
    1.409, # CG 1 GPU: Custom vs AmgX (188.7/133.9)
    1.399  # CG 8 GPUs: Custom vs AmgX (27.0/19.3)
]

speedups_15k = [
    2.06,  # SpMV: STENCIL5 vs cuSPARSE CSR (15.00/7.29)
    1.400, # CG 1 GPU: Custom vs AmgX (420.0/300.1)
    1.411  # CG 8 GPUs: Custom vs AmgX (57.0/40.4)
]

speedups_20k = [
    2.08,  # SpMV: STENCIL5 vs cuSPARSE CSR (26.77/12.86)
    1.405, # CG 1 GPU: Custom vs AmgX (746.7/531.4)
    1.441  # CG 8 GPUs: Custom vs AmgX (102.3/71.0)
]

# Average speedups
speedups_avg = [
    np.mean([speedups_10k[i], speedups_15k[i], speedups_20k[i]])
    for i in range(3)
]

# Colors
color_spmv = '#2E86AB'   # Blue
color_cg_1 = '#A23B72'   # Purple
color_cg_8 = '#F18F01'   # Orange
colors = [color_spmv, color_cg_1, color_cg_8]

# Font sizes
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10

# ============================================================================
# Main Performance Summary Figure
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 7))

x = np.arange(len(categories))
width = 0.2

# Plot bars for each problem size
bars1 = ax.bar(x - 1.5*width, speedups_10k, width, label='10k×10k',
               color=colors, alpha=0.7, edgecolor='black', linewidth=1)
bars2 = ax.bar(x - 0.5*width, speedups_15k, width, label='15k×15k',
               color=colors, alpha=0.85, edgecolor='black', linewidth=1)
bars3 = ax.bar(x + 0.5*width, speedups_20k, width, label='20k×20k',
               color=colors, alpha=1.0, edgecolor='black', linewidth=1)

# Reference line at 1.0× (equal performance)
ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, alpha=0.8,
           label='Reference Performance (1.0×)', zorder=0)

# Styling
ax.set_xlabel('Implementation Category', fontweight='bold', fontsize=13)
ax.set_ylabel('Speedup vs Industry Reference', fontweight='bold', fontsize=13)
ax.set_title('Custom Implementations: Performance Gains Summary\n' +
             'STENCIL5 vs cuSPARSE CSR | Custom CG vs NVIDIA AmgX',
             fontweight='bold', fontsize=15, pad=20)
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=12, fontweight='bold')
ax.legend(loc='upper center', framealpha=0.95, fontsize=11, ncol=4,
          bbox_to_anchor=(0.5, -0.08))
ax.grid(axis='y', alpha=0.3, linestyle='--', color='#E8E8E8')
ax.set_axisbelow(True)
ax.set_ylim(0, 2.5)

# Add value labels on bars
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        label_text = f'{height:.2f}×' if height > 1.5 else f'{height:.2f}×'
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                label_text,
                ha='center', va='bottom', fontsize=9, fontweight='bold')

# Add shaded region for "custom faster" zone
ax.axhspan(1.0, 2.5, alpha=0.1, color='green', zorder=0)
ax.text(0.02, 1.1, 'Custom Faster →', transform=ax.transData,
        fontsize=10, style='italic', color='green', alpha=0.7)

# Key findings as subtitle annotation (no overlap with legend)
ax.text(0.98, 0.97, 'All custom implementations outperform industry references',
        transform=ax.transAxes, fontsize=10, fontstyle='italic',
        verticalalignment='top', horizontalalignment='right',
        color='#555555')

plt.tight_layout()
plt.savefig('docs/figures/performance_summary.png', dpi=300, bbox_inches='tight')
print('✓ Generated: docs/figures/performance_summary.png')
plt.close()

# ============================================================================
# Alternative: Compact Version for README Top
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 5))

# Use average speedups for cleaner view
bars = ax.bar(x, speedups_avg, color=colors, alpha=0.9,
              edgecolor='black', linewidth=1.5, width=0.6)

# Reference line
ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, alpha=0.8,
           label='Reference (1.0×)', zorder=0)

# Styling
ax.set_xlabel('Implementation Category', fontweight='bold', fontsize=12)
ax.set_ylabel('Average Speedup vs Reference', fontweight='bold', fontsize=12)
ax.set_title('Custom Implementations Outperform Industry Standards\n' +
             'Average Performance Gains Across All Problem Sizes',
             fontweight='bold', fontsize=14, pad=15)
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=11, fontweight='bold')
ax.legend(loc='upper right', framealpha=0.95, fontsize=10)
ax.grid(axis='y', alpha=0.3, linestyle='--', color='#E8E8E8')
ax.set_axisbelow(True)
ax.set_ylim(0, 2.5)

# Add value labels
for i, (bar, speedup) in enumerate(zip(bars, speedups_avg)):
    height = bar.get_height()
    percentage = int((height - 1) * 100)
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
            f'{height:.2f}×\n(+{percentage}%)',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

# Add reference labels
ref_labels = ['vs cuSPARSE\nCSR', 'vs AmgX\nCG', 'vs AmgX\nCG']
for i, label in enumerate(ref_labels):
    ax.text(i, 0.15, label, ha='center', va='bottom',
            fontsize=9, style='italic', color='gray')

# Shaded region
ax.axhspan(1.0, 2.5, alpha=0.1, color='green', zorder=0)

plt.tight_layout()
plt.savefig('docs/figures/performance_summary_compact.png', dpi=300, bbox_inches='tight')
print('✓ Generated: docs/figures/performance_summary_compact.png')
plt.close()

# ============================================================================
# Alternative: Horizontal bars for better readability
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

categories_detailed = [
    'SpMV Single-GPU\n(STENCIL5 vs cuSPARSE CSR)',
    'CG Single-GPU\n(Custom vs AmgX)',
    'CG Multi-GPU (8×)\n(Custom vs AmgX)'
]

y = np.arange(len(categories_detailed))

# Create horizontal bars
bars = ax.barh(y, speedups_20k, color=colors, alpha=0.9,
               edgecolor='black', linewidth=1.5, height=0.6)

# Reference line
ax.axvline(x=1.0, color='red', linestyle='--', linewidth=2, alpha=0.8,
           label='Reference (1.0×)', zorder=0)

# Styling
ax.set_xlabel('Speedup vs Industry Reference', fontweight='bold', fontsize=12)
ax.set_title('Custom Implementations: Performance Summary\n' +
             'Showcase Configuration: 20k×20k (400M unknowns)',
             fontweight='bold', fontsize=14, pad=15)
ax.set_yticks(y)
ax.set_yticklabels(categories_detailed, fontsize=10)
ax.legend(loc='lower right', framealpha=0.95, fontsize=10)
ax.grid(axis='x', alpha=0.3, linestyle='--', color='#E8E8E8')
ax.set_axisbelow(True)
ax.set_xlim(0, 2.5)

# Add value labels
for i, (bar, speedup) in enumerate(zip(bars, speedups_20k)):
    width = bar.get_width()
    percentage = int((width - 1) * 100)
    ax.text(width + 0.05, bar.get_y() + bar.get_height()/2.,
            f'{width:.2f}× (+{percentage}%)',
            ha='left', va='center', fontsize=11, fontweight='bold')

# Shaded region
ax.axvspan(1.0, 2.5, alpha=0.1, color='green', zorder=0)
ax.text(1.05, 2.5, '← Custom Faster', fontsize=10, style='italic',
        color='green', alpha=0.7, va='center')

plt.tight_layout()
plt.savefig('docs/figures/performance_summary_horizontal.png', dpi=300, bbox_inches='tight')
print('✓ Generated: docs/figures/performance_summary_horizontal.png')
plt.close()

print('\n✓ All performance summary plots generated successfully!')
print('  - performance_summary.png (detailed with all sizes)')
print('  - performance_summary_compact.png (clean average view)')
print('  - performance_summary_horizontal.png (horizontal bars)')
