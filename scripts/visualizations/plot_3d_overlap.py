#!/usr/bin/env python3
"""
plot_3d_overlap.py - Visualize 3D stencil overlap benchmark results.

Usage:
  python scripts/visualizations/plot_3d_overlap.py [OPTIONS]
    --input-dir=DIR   JSON directory (default: results/3d/json/)
    --output-dir=DIR  PNG output directory (default: docs/figures/)
    --show            Display figures interactively

Produces 4 figures:
  1. 3d_overlap_gain_heatmap.png   - Heatmaps: overlap gain per (grid, GPU count)
  2. 3d_overlap_gain_by_gpu.png    - Line plot: overlap gain vs GPU count
  3. 3d_scaling_comparison.png     - Scaling: abs time + parallel efficiency
  4. 3d_stencil_comparison.png     - Grouped bar chart at max GPU count
"""

import json
import re
import sys
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results(input_dir):
    """
    Returns:
        data: dict mapping (stencil, grid, n_gpu, mode) -> median_ms
        overlap_meta: dict mapping (stencil, grid, n_gpu) -> dict with comm fields
    """
    data = {}
    overlap_meta = {}

    pattern = re.compile(r'3d_(\d+)pt_(\d+)_(\d+)gpu_(sync|overlap)\.json')
    for f in Path(input_dir).glob('3d_*.json'):
        m = pattern.match(f.name)
        if not m:
            continue
        stencil, grid, n_gpu, mode = int(m[1]), int(m[2]), int(m[3]), m[4]
        try:
            d = json.loads(f.read_text())
            data[(stencil, grid, n_gpu, mode)] = d['timing']['median_ms']
            if mode == 'overlap' and 'overlap' in d:
                ov = d['overlap']
                if ov.get('enabled'):
                    overlap_meta[(stencil, grid, n_gpu)] = {
                        'comm_total_ms': ov.get('comm_total_ms', 0),
                        'comm_hidden_ms': ov.get('comm_hidden_ms', 0),
                        'overlap_efficiency_pct': ov.get('overlap_efficiency_pct', 0),
                    }
        except (KeyError, json.JSONDecodeError):
            continue

    if not data:
        print(f"No matching JSON files found in {input_dir}")
        sys.exit(1)

    return data, overlap_meta

# ---------------------------------------------------------------------------
# Figure 1: Overlap gain heatmap (7pt / 27pt side by side)
# ---------------------------------------------------------------------------

def plot_gain_heatmap(data, output_dir, show=False):
    import matplotlib.pyplot as plt
    import numpy as np

    stencils = sorted({k[0] for k in data})
    if len(stencils) < 1:
        print("Warning: not enough data for heatmap — skipping figure 1")
        return

    all_grids = sorted({k[1] for k in data})
    multi_gpus = sorted({k[2] for k in data if k[2] > 1})
    if not multi_gpus:
        print("Warning: no multi-GPU data for heatmap — skipping figure 1")
        return

    # Compute gains
    all_gains = []
    for s in stencils:
        for g in all_grids:
            for n in multi_gpus:
                sync_t = data.get((s, g, n, 'sync'))
                ovl_t = data.get((s, g, n, 'overlap'))
                if sync_t and ovl_t and ovl_t > 0:
                    all_gains.append(sync_t / ovl_t)
    if not all_gains:
        print("Warning: no gain data for heatmap — skipping figure 1")
        return

    vmax = max(all_gains)
    vmin = 1.0

    n_stencils = len(stencils)
    fig, axes = plt.subplots(1, n_stencils, figsize=(6 * n_stencils, max(3, len(all_grids) * 0.8 + 1.5)))
    if n_stencils == 1:
        axes = [axes]

    fig.suptitle("Compute-Communication Overlap Gain\n"
                 "(sync_time / overlap_time — higher is better)", fontsize=13)

    for ax, s in zip(axes, stencils):
        grid_sizes = sorted({k[1] for k in data if k[0] == s})
        gpu_counts = sorted({k[2] for k in data if k[0] == s and k[2] > 1})
        if not grid_sizes or not gpu_counts:
            ax.set_visible(False)
            continue

        gain_matrix = np.full((len(grid_sizes), len(gpu_counts)), float('nan'))
        for gi, g in enumerate(grid_sizes):
            for ni, n in enumerate(gpu_counts):
                sync_t = data.get((s, g, n, 'sync'))
                ovl_t = data.get((s, g, n, 'overlap'))
                if sync_t and ovl_t and ovl_t > 0:
                    gain_matrix[gi, ni] = sync_t / ovl_t

        # Plot
        masked = np.ma.masked_invalid(gain_matrix)
        cmap = plt.cm.Blues.copy()
        cmap.set_bad(color='lightgrey')
        im = ax.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')

        ax.set_xticks(range(len(gpu_counts)))
        ax.set_xticklabels([str(n) for n in gpu_counts])
        ax.set_yticks(range(len(grid_sizes)))
        ax.set_yticklabels([f'{g}³' for g in grid_sizes])
        ax.set_xlabel("GPU Count")
        ax.set_ylabel("Grid Size")
        ax.set_title(f'{s}-point stencil')

        for gi in range(len(grid_sizes)):
            for ni in range(len(gpu_counts)):
                val = gain_matrix[gi, ni]
                text = f'{val:.2f}×' if not np.isnan(val) else 'n/a'
                color = 'white' if (not np.isnan(val) and val > (vmax + vmin) / 2) else 'black'
                ax.text(ni, gi, text, ha='center', va='center', fontsize=9, color=color)

        plt.colorbar(im, ax=ax, label='Gain (×)')

    plt.tight_layout()
    out = Path(output_dir) / '3d_overlap_gain_heatmap.png'
    plt.savefig(out, dpi=150)
    print(f"Saved: {out}")
    if show:
        plt.show()
    plt.close()

# ---------------------------------------------------------------------------
# Figure 2: Overlap gain by GPU count (line plot)
# ---------------------------------------------------------------------------

def plot_gain_by_gpu(data, output_dir, show=False):
    import matplotlib.pyplot as plt

    stencils = sorted({k[0] for k in data})
    grids = sorted({k[1] for k in data})

    # Color families: 7pt = blues, 27pt = oranges
    colors_7pt  = {128: 'lightblue', 256: 'steelblue', 512: 'darkblue'}
    colors_27pt = {128: 'moccasin', 256: 'darkorange', 512: 'saddlebrown'}
    colors_other = ['purple', 'green', 'brown', 'pink']

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhline(y=1.0, color='grey', linestyle='--', linewidth=0.8, label='No gain')

    has_data = False
    ci = 0
    for s in stencils:
        for g in grids:
            gpu_counts = sorted({k[2] for k in data if k[0] == s and k[1] == g and k[2] > 1})
            xs, ys = [], []
            for n in gpu_counts:
                sync_t = data.get((s, g, n, 'sync'))
                ovl_t = data.get((s, g, n, 'overlap'))
                if sync_t and ovl_t and ovl_t > 0:
                    xs.append(n)
                    ys.append(sync_t / ovl_t)
            if len(xs) < 1:
                continue

            if s == 7:
                c = colors_7pt.get(g, colors_other[ci % len(colors_other)])
            elif s == 27:
                c = colors_27pt.get(g, colors_other[ci % len(colors_other)])
            else:
                c = colors_other[ci % len(colors_other)]
                ci += 1

            ax.plot(xs, ys, marker='o', color=c, linewidth=2, label=f'{s}pt {g}³')
            has_data = True

    if not has_data:
        print("Warning: no multi-GPU overlap gain data — skipping figure 2")
        plt.close()
        return

    ax.set_xlabel("GPU Count")
    ax.set_ylabel("Overlap Gain (sync / overlap)")
    ax.set_title("Overlap Speedup vs GPU Count")

    all_gpus = sorted({k[2] for k in data if k[2] > 1})
    if all_gpus:
        ax.set_xticks(all_gpus)
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = Path(output_dir) / '3d_overlap_gain_by_gpu.png'
    plt.savefig(out, dpi=150)
    print(f"Saved: {out}")
    if show:
        plt.show()
    plt.close()

# ---------------------------------------------------------------------------
# Figure 3: Scaling comparison (abs time + parallel efficiency)
# ---------------------------------------------------------------------------

def plot_scaling(data, output_dir, show=False):
    import matplotlib.pyplot as plt
    import numpy as np

    stencils = sorted({k[0] for k in data})
    grids = sorted({k[1] for k in data})

    colors = ['steelblue', 'darkorange', 'green', 'red', 'purple', 'brown']

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("3D Stencil Multi-GPU Scaling", fontsize=13)

    has_abs = False
    has_eff = False
    ci = 0

    for s in stencils:
        for g in grids:
            all_gpus = sorted({k[2] for k in data if k[0] == s and k[1] == g})
            if not all_gpus:
                continue

            c = colors[ci % len(colors)]
            ci += 1

            # Absolute times
            ovl_times = [(n, data[(s, g, n, 'overlap')]) for n in all_gpus
                         if (s, g, n, 'overlap') in data]
            sync_times = [(n, data[(s, g, n, 'sync')]) for n in all_gpus
                          if (s, g, n, 'sync') in data]

            if ovl_times:
                xs, ys = zip(*ovl_times)
                ax_left.plot(xs, ys, '-o', color=c, linewidth=2, label=f'{s}pt {g}³ ovl')
                has_abs = True
            if sync_times:
                xs, ys = zip(*sync_times)
                ax_left.plot(xs, ys, '--s', color=c, linewidth=1.5, alpha=0.6,
                             label=f'{s}pt {g}³ sync')
                has_abs = True

            # Parallel efficiency: T_1_sync / (N * T_N_ovl)
            t1_sync = data.get((s, g, 1, 'sync'))
            if t1_sync:
                eff_points = [(n, t) for n, t in ovl_times if n > 1]
                if eff_points:
                    xs_eff = [n for n, _ in eff_points]
                    ys_eff = [t1_sync / (n * t) * 100.0 for n, t in eff_points]
                    ax_right.plot(xs_eff, ys_eff, '-o', color=c, linewidth=2,
                                  label=f'{s}pt {g}³')
                    has_eff = True

    if not has_abs:
        print("Warning: not enough scaling data — skipping figure 3")
        plt.close()
        return

    ax_left.set_yscale('log')
    ax_left.set_xlabel("GPU Count")
    ax_left.set_ylabel("Time (ms, log scale)")
    ax_left.set_title("Absolute Time")
    ax_left.legend(fontsize=7, ncol=2)
    ax_left.grid(True, alpha=0.3, which='both')

    if has_eff:
        ax_right.axhline(y=100.0, color='grey', linestyle='--', linewidth=0.8, label='Ideal')
        ax_right.set_xlabel("GPU Count")
        ax_right.set_ylabel("Parallel Efficiency (%)")
        ax_right.set_title("Parallel Efficiency (overlap, vs 1-GPU sync)")
        ax_right.legend(fontsize=7)
        ax_right.grid(True, alpha=0.3)
    else:
        ax_right.set_visible(False)

    plt.tight_layout()
    out = Path(output_dir) / '3d_scaling_comparison.png'
    plt.savefig(out, dpi=150)
    print(f"Saved: {out}")
    if show:
        plt.show()
    plt.close()

# ---------------------------------------------------------------------------
# Figure 4: Stencil comparison grouped bar chart
# ---------------------------------------------------------------------------

def plot_stencil_comparison(data, output_dir, show=False):
    import matplotlib.pyplot as plt
    import numpy as np

    # Use max available GPU count (prefer 8, then 4, 2)
    all_gpus = sorted({k[2] for k in data})
    target_n = None
    for prefer in [8, 4, 2]:
        if prefer in all_gpus:
            target_n = prefer
            break
    if target_n is None:
        print("Warning: no multi-GPU data for stencil comparison — skipping figure 4")
        return

    grids = sorted({k[1] for k in data})
    stencils = sorted({k[0] for k in data})

    # We need at least 2 data points
    total_points = sum(
        1 for g in grids for s in stencils
        for mode in ('sync', 'overlap')
        if (s, g, target_n, mode) in data
    )
    if total_points < 2:
        print(f"Warning: fewer than 2 data points for stencil comparison at {target_n} GPUs — skipping figure 4")
        return

    # Bar setup: 4 bars per grid (sync-7pt, ovl-7pt, sync-27pt, ovl-27pt)
    bar_colors = {
        (7, 'sync'):    'lightblue',
        (7, 'overlap'): 'steelblue',
        (27, 'sync'):   'lightsalmon',
        (27, 'overlap'): 'tomato',
    }
    bar_labels = {
        (7, 'sync'):    'sync 7pt',
        (7, 'overlap'): 'overlap 7pt',
        (27, 'sync'):   'sync 27pt',
        (27, 'overlap'): 'overlap 27pt',
    }

    n_grids = len(grids)
    n_bars = len(stencils) * 2  # sync + overlap per stencil
    bar_width = 0.18
    group_gap = 0.8
    x_positions = np.arange(n_grids) * group_gap

    fig, ax = plt.subplots(figsize=(max(8, n_grids * 2.5), 6))

    bar_offset = -(n_bars - 1) / 2 * bar_width
    legend_handles = {}

    for si, s in enumerate(stencils):
        for mi, mode in enumerate(['sync', 'overlap']):
            key_combo = (s, mode)
            offset = bar_offset + (si * 2 + mi) * bar_width
            color = bar_colors.get(key_combo, 'grey')
            label = bar_labels.get(key_combo, f'{s}pt {mode}')

            for gi, g in enumerate(grids):
                val = data.get((s, g, target_n, mode))
                if val is None:
                    continue
                bar = ax.bar(x_positions[gi] + offset, val, bar_width,
                             color=color, alpha=0.9)
                if key_combo not in legend_handles:
                    legend_handles[key_combo] = bar

                # Annotate overlap bars with gain
                if mode == 'overlap':
                    sync_t = data.get((s, g, target_n, 'sync'))
                    if sync_t and val > 0:
                        gain = sync_t / val
                        ax.text(x_positions[gi] + offset, val + val * 0.02,
                                f'{gain:.2f}×', ha='center', va='bottom',
                                fontsize=7, color='black', fontweight='bold')

    ax.set_xticks(x_positions)
    ax.set_xticklabels([f'{g}³' for g in grids])
    ax.set_xlabel("Grid Size")
    ax.set_ylabel("Time (ms)")
    ax.set_title(f"3D Stencil: Sync vs Overlap at {target_n} GPUs")
    ax.legend([legend_handles[k] for k in sorted(legend_handles)],
              [bar_labels[k] for k in sorted(legend_handles)],
              fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    out = Path(output_dir) / '3d_stencil_comparison.png'
    plt.savefig(out, dpi=150)
    print(f"Saved: {out}")
    if show:
        plt.show()
    plt.close()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Plot 3D overlap benchmark results')
    parser.add_argument('--input-dir',  default='results/3d/json/',
                        help='Directory containing JSON result files')
    parser.add_argument('--output-dir', default='docs/figures/',
                        help='Directory to write PNG files')
    parser.add_argument('--show', action='store_true',
                        help='Display figures interactively')
    args = parser.parse_args()

    # Support both --input-dir=... and --input-dir ...
    input_dir  = args.input_dir
    output_dir = args.output_dir
    show       = args.show

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"Loading results from: {input_dir}")
    data, overlap_meta = load_results(input_dir)
    print(f"Loaded {len(data)} result entries")

    try:
        import matplotlib
        matplotlib.use('Agg' if not show else 'TkAgg')
    except ImportError:
        print("ERROR: matplotlib not found. Install with: pip install matplotlib numpy")
        sys.exit(1)

    plot_gain_heatmap(data, output_dir, show=show)
    plot_gain_by_gpu(data, output_dir, show=show)
    plot_scaling(data, output_dir, show=show)
    plot_stencil_comparison(data, output_dir, show=show)

    print(f"\nAll figures written to: {output_dir}")


if __name__ == '__main__':
    main()
