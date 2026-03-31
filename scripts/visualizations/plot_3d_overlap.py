#!/usr/bin/env python3
"""
plot_3d_overlap.py - Visualize 3D stencil overlap benchmark results.

Usage:
  python scripts/visualizations/plot_3d_overlap.py [OPTIONS]
    --input-dir=DIR   JSON directory (default: results/json/3d_overlap/)
    --output-dir=DIR  PNG output directory (default: docs/figures/)
    --show            Display figures interactively

Produces 5 figures:
  0. 3d_scaling_overlap_a100.png    - Main: sync vs overlap speedup (7pt | 27pt)
  1. 3d_overlap_gain_heatmap.png    - Heatmaps: overlap gain per (grid, GPU count)
  2. 3d_overlap_gain_by_gpu.png     - Line plot: overlap gain vs GPU count
  3. 3d_scaling_comparison.png      - Scaling: abs time + parallel efficiency
  4. 3d_stencil_comparison.png      - Grouped bar chart at max GPU count
"""

import json
import re
import sys
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Fallback hardcoded data (used when JSON files are not found)
# ---------------------------------------------------------------------------

FALLBACK = {
    (7, 128, 1, 'sync'): 73.2,   (7, 128, 1, 'overlap'): 74.0,
    (7, 128, 2, 'sync'): 52.8,   (7, 128, 2, 'overlap'): 43.9,
    (7, 128, 4, 'sync'): 51.4,   (7, 128, 4, 'overlap'): 46.7,
    (7, 128, 8, 'sync'): 47.8,   (7, 128, 8, 'overlap'): 49.7,
    (7, 256, 1, 'sync'): 970.3,  (7, 256, 1, 'overlap'): 972.4,
    (7, 256, 2, 'sync'): 583.3,  (7, 256, 2, 'overlap'): 515.7,
    (7, 256, 4, 'sync'): 409.0,  (7, 256, 4, 'overlap'): 318.0,
    (7, 256, 8, 'sync'): 304.7,  (7, 256, 8, 'overlap'): 265.8,
    (7, 512, 1, 'sync'): 15127,  (7, 512, 1, 'overlap'): 15129,
    (7, 512, 2, 'sync'): 8211,   (7, 512, 2, 'overlap'): 7682,
    (7, 512, 4, 'sync'): 5088,   (7, 512, 4, 'overlap'): 3944,
    (7, 512, 8, 'sync'): 3323,   (7, 512, 8, 'overlap'): 2453,
    (27, 128, 1, 'sync'): 89.2,  (27, 128, 1, 'overlap'): 89.6,
    (27, 128, 2, 'sync'): 57.3,  (27, 128, 2, 'overlap'): 51.1,
    (27, 128, 4, 'sync'): 47.3,  (27, 128, 4, 'overlap'): 36.6,
    (27, 128, 8, 'sync'): 40.5,  (27, 128, 8, 'overlap'): 33.6,
    (27, 256, 1, 'sync'): 1315.4, (27, 256, 1, 'overlap'): 1315.4,
    (27, 256, 2, 'sync'): 718.9,  (27, 256, 2, 'overlap'): 680.3,
    (27, 256, 4, 'sync'): 447.5,  (27, 256, 4, 'overlap'): 367.5,
    (27, 256, 8, 'sync'): 294.0,  (27, 256, 8, 'overlap'): 203.5,
    (27, 512, 1, 'sync'): 22016, (27, 512, 1, 'overlap'): 21997,
    (27, 512, 2, 'sync'): 11438, (27, 512, 2, 'overlap'): 11142,
    (27, 512, 4, 'sync'): 6461,  (27, 512, 4, 'overlap'): 5815,
    (27, 512, 8, 'sync'): 3809,  (27, 512, 8, 'overlap'): 3110,
}

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------

COLOR_OVERLAP = '#2E86AB'
COLOR_SYNC    = '#A23B72'

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
    json_dir = Path(input_dir)
    if json_dir.exists():
        for f in json_dir.glob('3d_*.json'):
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
                            'comm_total_ms':         ov.get('comm_total_ms', 0),
                            'comm_hidden_ms':        ov.get('comm_hidden_ms', 0),
                            'overlap_efficiency_pct': ov.get('overlap_efficiency_pct', 0),
                        }
            except (KeyError, json.JSONDecodeError):
                continue

    if not data:
        print(f"Warning: no matching JSON files found in {input_dir} — using hardcoded fallback data")
        data = dict(FALLBACK)

    return data, overlap_meta

# ---------------------------------------------------------------------------
# Figure 0: Main scaling plot (sync vs overlap speedup, 7pt | 27pt)
# ---------------------------------------------------------------------------

def plot_scaling_main(data, output_dir, show=False):
    import matplotlib.pyplot as plt
    import numpy as np

    gpu_counts = [1, 2, 4, 8]
    stencils   = [7, 27]
    grids_main = [512, 256]
    eff_labels = {7: '77%', 27: '88%'}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    subplot_titles = {7: '7-Point Stencil', 27: '27-Point Stencil'}

    for ax, s in zip(axes, stencils):
        t1_sync = data.get((s, 512, 1, 'sync'))
        if t1_sync is None:
            ax.set_visible(False)
            continue

        # Ideal linear scaling
        ideal_xs = [1, 2, 4, 8]
        ideal_ys = [float(n) for n in ideal_xs]
        ax.plot(ideal_xs, ideal_ys, '--', color='grey', linewidth=1.5,
                label='Ideal', zorder=1)

        # 512³ sync
        xs_s512, ys_s512 = [], []
        for n in gpu_counts:
            t = data.get((s, 512, n, 'sync'))
            if t:
                xs_s512.append(n)
                ys_s512.append(t1_sync / t)
        if xs_s512:
            ax.plot(xs_s512, ys_s512, '-o', color=COLOR_SYNC, linewidth=2,
                    markersize=7, markeredgecolor='black', markeredgewidth=0.8,
                    label='Sync (512³)', zorder=3)

        # 512³ overlap
        xs_o512, ys_o512 = [], []
        for n in gpu_counts:
            t = data.get((s, 512, n, 'overlap'))
            if t:
                xs_o512.append(n)
                ys_o512.append(t1_sync / t)
        if xs_o512:
            ax.plot(xs_o512, ys_o512, '-o', color=COLOR_OVERLAP, linewidth=2,
                    markersize=7, markeredgecolor='black', markeredgewidth=0.8,
                    label='Overlap (512³)', zorder=3)
            # Annotate 8-GPU point
            if 8 in xs_o512:
                idx = xs_o512.index(8)
                ax.annotate(eff_labels[s],
                            xy=(8, ys_o512[idx]),
                            xytext=(8 - 0.6, ys_o512[idx] + 0.25),
                            fontsize=9, color=COLOR_OVERLAP, fontweight='bold')

        # 256³ sync (secondary, alpha)
        t1_sync_256 = data.get((s, 256, 1, 'sync'))
        xs_s256, ys_s256 = [], []
        if t1_sync_256:
            for n in gpu_counts:
                t = data.get((s, 256, n, 'sync'))
                if t:
                    xs_s256.append(n)
                    ys_s256.append(t1_sync_256 / t)
        if xs_s256:
            ax.plot(xs_s256, ys_s256, '-s', color=COLOR_SYNC, linewidth=1.2,
                    alpha=0.4, markersize=6, markeredgecolor='black', markeredgewidth=0.8,
                    label='Sync (256³)', zorder=2)

        # 256³ overlap (secondary, alpha)
        xs_o256, ys_o256 = [], []
        if t1_sync_256:
            for n in gpu_counts:
                t = data.get((s, 256, n, 'overlap'))
                if t:
                    xs_o256.append(n)
                    ys_o256.append(t1_sync_256 / t)
        if xs_o256:
            ax.plot(xs_o256, ys_o256, '-s', color=COLOR_OVERLAP, linewidth=1.2,
                    alpha=0.4, markersize=6, markeredgecolor='black', markeredgewidth=0.8,
                    label='Overlap (256³)', zorder=2)

        ax.set_title(subplot_titles[s], fontsize=12)
        ax.set_xlabel('Number of GPUs', fontsize=11)
        ax.set_ylabel('Speedup vs 1 GPU', fontsize=11)
        ax.set_xticks(gpu_counts)
        ax.grid(axis='y', alpha=0.3, linestyle='--', color='#E8E8E8')
        ax.legend(fontsize=9)

    plt.tight_layout()
    out = Path(output_dir) / '3d_scaling_overlap_a100.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f"Saved: {out}")
    if show:
        plt.show()
    plt.close()

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

    all_gains = []
    for s in stencils:
        for g in all_grids:
            for n in multi_gpus:
                sync_t = data.get((s, g, n, 'sync'))
                ovl_t  = data.get((s, g, n, 'overlap'))
                if sync_t and ovl_t and ovl_t > 0:
                    all_gains.append(sync_t / ovl_t)
    if not all_gains:
        print("Warning: no gain data for heatmap — skipping figure 1")
        return

    vmax = max(all_gains)
    vmin = 1.0

    n_stencils = len(stencils)
    fig, axes = plt.subplots(1, n_stencils,
                             figsize=(6 * n_stencils, max(3, len(all_grids) * 0.8 + 1.5)))
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
                ovl_t  = data.get((s, g, n, 'overlap'))
                if sync_t and ovl_t and ovl_t > 0:
                    gain_matrix[gi, ni] = sync_t / ovl_t

        masked = np.ma.masked_invalid(gain_matrix)
        cmap = plt.cm.Blues.copy()
        cmap.set_bad(color='lightgrey')
        im = ax.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')

        ax.set_xticks(range(len(gpu_counts)))
        ax.set_xticklabels([str(n) for n in gpu_counts])
        ax.set_yticks(range(len(grid_sizes)))
        ax.set_yticklabels([f'{g}\u00b3' for g in grid_sizes])
        ax.set_xlabel("GPU Count")
        ax.set_ylabel("Grid Size")
        ax.set_title(f'{s}-point stencil')

        for gi in range(len(grid_sizes)):
            for ni in range(len(gpu_counts)):
                val = gain_matrix[gi, ni]
                text = f'{val:.2f}\u00d7' if not np.isnan(val) else 'n/a'
                color = 'white' if (not np.isnan(val) and val > (vmax + vmin) / 2) else 'black'
                ax.text(ni, gi, text, ha='center', va='center', fontsize=9, color=color)

        plt.colorbar(im, ax=ax, label='Gain (\u00d7)')

    plt.tight_layout()
    out = Path(output_dir) / '3d_overlap_gain_heatmap.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
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
    grids    = sorted({k[1] for k in data})

    # Primary grid (512) uses full color; smaller grids use alpha=0.4
    primary_grid = max(grids) if grids else 512

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhline(y=1.0, color='grey', linestyle='--', linewidth=0.8, label='No gain')

    has_data = False
    stencil_colors = {7: COLOR_SYNC, 27: COLOR_OVERLAP}
    markers = {128: 's', 256: '^', 512: 'o'}

    for s in stencils:
        base_color = stencil_colors.get(s, '#888888')
        for g in grids:
            gpu_counts = sorted({k[2] for k in data if k[0] == s and k[1] == g and k[2] > 1})
            xs, ys = [], []
            for n in gpu_counts:
                sync_t = data.get((s, g, n, 'sync'))
                ovl_t  = data.get((s, g, n, 'overlap'))
                if sync_t and ovl_t and ovl_t > 0:
                    xs.append(n)
                    ys.append(sync_t / ovl_t)
            if len(xs) < 1:
                continue

            alpha = 1.0 if g == primary_grid else 0.4
            lw    = 2.0 if g == primary_grid else 1.2
            mk    = markers.get(g, 'o')
            ax.plot(xs, ys, marker=mk, color=base_color, linewidth=lw, alpha=alpha,
                    markeredgecolor='black', markeredgewidth=0.8,
                    label=f'{s}pt {g}\u00b3')
            has_data = True

    if not has_data:
        print("Warning: no multi-GPU overlap gain data — skipping figure 2")
        plt.close()
        return

    ax.set_xlabel("GPU Count", fontsize=11)
    ax.set_ylabel("Overlap Gain (sync / overlap)", fontsize=11)
    ax.set_title("Overlap Speedup vs GPU Count")

    all_gpus = sorted({k[2] for k in data if k[2] > 1})
    if all_gpus:
        ax.set_xticks(all_gpus)
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(axis='y', alpha=0.3, linestyle='--', color='#E8E8E8')

    plt.tight_layout()
    out = Path(output_dir) / '3d_overlap_gain_by_gpu.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
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
    grids    = sorted({k[1] for k in data})

    # Use two colors per stencil: sync=COLOR_SYNC, overlap=COLOR_OVERLAP
    stencil_colors = {7: COLOR_SYNC, 27: COLOR_OVERLAP}
    grid_alphas    = {128: 0.4, 256: 0.7, 512: 1.0}

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("3D Stencil Multi-GPU Scaling", fontsize=13)

    has_abs = False
    has_eff = False

    for s in stencils:
        base_color = stencil_colors.get(s, '#888888')
        for g in grids:
            all_gpus = sorted({k[2] for k in data if k[0] == s and k[1] == g})
            if not all_gpus:
                continue

            alpha = grid_alphas.get(g, 0.8)

            ovl_times  = [(n, data[(s, g, n, 'overlap')]) for n in all_gpus
                          if (s, g, n, 'overlap') in data]
            sync_times = [(n, data[(s, g, n, 'sync')]) for n in all_gpus
                          if (s, g, n, 'sync') in data]

            if ovl_times:
                xs, ys = zip(*ovl_times)
                ax_left.plot(xs, ys, '-o', color=COLOR_OVERLAP, linewidth=2,
                             alpha=alpha, markeredgecolor='black', markeredgewidth=0.8,
                             label=f'{s}pt {g}\u00b3 ovl')
                has_abs = True
            if sync_times:
                xs, ys = zip(*sync_times)
                ax_left.plot(xs, ys, '--s', color=COLOR_SYNC, linewidth=1.5,
                             alpha=alpha * 0.7, markeredgecolor='black', markeredgewidth=0.8,
                             label=f'{s}pt {g}\u00b3 sync')
                has_abs = True

            t1_sync = data.get((s, g, 1, 'sync'))
            if t1_sync:
                eff_points = [(n, t) for n, t in ovl_times if n > 1]
                if eff_points:
                    xs_eff = [n for n, _ in eff_points]
                    ys_eff = [t1_sync / (n * t) * 100.0 for n, t in eff_points]
                    ax_right.plot(xs_eff, ys_eff, '-o', color=base_color, linewidth=2,
                                  alpha=alpha, markeredgecolor='black', markeredgewidth=0.8,
                                  label=f'{s}pt {g}\u00b3')
                    has_eff = True

    if not has_abs:
        print("Warning: not enough scaling data — skipping figure 3")
        plt.close()
        return

    ax_left.set_yscale('log')
    ax_left.set_xlabel("GPU Count", fontsize=11)
    ax_left.set_ylabel("Time (ms, log scale)", fontsize=11)
    ax_left.set_title("Absolute Time")
    ax_left.legend(fontsize=7, ncol=2)
    ax_left.grid(axis='y', alpha=0.3, linestyle='--', color='#E8E8E8')

    if has_eff:
        ax_right.axhline(y=100.0, color='grey', linestyle='--', linewidth=0.8, label='Ideal')
        ax_right.set_xlabel("GPU Count", fontsize=11)
        ax_right.set_ylabel("Parallel Efficiency (%)", fontsize=11)
        ax_right.set_title("Parallel Efficiency (overlap, vs 1-GPU sync)")
        ax_right.legend(fontsize=7)
        ax_right.grid(axis='y', alpha=0.3, linestyle='--', color='#E8E8E8')
    else:
        ax_right.set_visible(False)

    plt.tight_layout()
    out = Path(output_dir) / '3d_scaling_comparison.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
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

    all_gpus = sorted({k[2] for k in data})
    target_n = None
    for prefer in [8, 4, 2]:
        if prefer in all_gpus:
            target_n = prefer
            break
    if target_n is None:
        print("Warning: no multi-GPU data for stencil comparison — skipping figure 4")
        return

    grids    = sorted({k[1] for k in data})
    stencils = sorted({k[0] for k in data})

    total_points = sum(
        1 for g in grids for s in stencils
        for mode in ('sync', 'overlap')
        if (s, g, target_n, mode) in data
    )
    if total_points < 2:
        print(f"Warning: fewer than 2 data points at {target_n} GPUs — skipping figure 4")
        return

    bar_colors = {
        (7,  'sync'):    COLOR_SYNC + '80',   # 50% alpha via hex
        (7,  'overlap'): COLOR_OVERLAP,
        (27, 'sync'):    COLOR_SYNC,
        (27, 'overlap'): COLOR_OVERLAP + '80',
    }
    bar_colors = {
        (7,  'sync'):    '#C97BA8',
        (7,  'overlap'): '#7ABFD6',
        (27, 'sync'):    COLOR_SYNC,
        (27, 'overlap'): COLOR_OVERLAP,
    }
    bar_labels = {
        (7,  'sync'):    'sync 7pt',
        (7,  'overlap'): 'overlap 7pt',
        (27, 'sync'):    'sync 27pt',
        (27, 'overlap'): 'overlap 27pt',
    }

    n_grids  = len(grids)
    n_bars   = len(stencils) * 2
    bar_width  = 0.18
    group_gap  = 0.8
    x_positions = np.arange(n_grids) * group_gap

    fig, ax = plt.subplots(figsize=(max(8, n_grids * 2.5), 6))

    bar_offset = -(n_bars - 1) / 2 * bar_width
    legend_handles = {}

    for si, s in enumerate(stencils):
        for mi, mode in enumerate(['sync', 'overlap']):
            key_combo = (s, mode)
            offset = bar_offset + (si * 2 + mi) * bar_width
            color  = bar_colors.get(key_combo, 'grey')
            label  = bar_labels.get(key_combo, f'{s}pt {mode}')

            for gi, g in enumerate(grids):
                val = data.get((s, g, target_n, mode))
                if val is None:
                    continue
                bar = ax.bar(x_positions[gi] + offset, val, bar_width,
                             color=color, alpha=0.9,
                             edgecolor='black', linewidth=0.5)
                if key_combo not in legend_handles:
                    legend_handles[key_combo] = bar

                if mode == 'overlap':
                    sync_t = data.get((s, g, target_n, 'sync'))
                    if sync_t and val > 0:
                        gain = sync_t / val
                        ax.text(x_positions[gi] + offset, val + val * 0.02,
                                f'{gain:.2f}\u00d7', ha='center', va='bottom',
                                fontsize=7, color='black', fontweight='bold')

    ax.set_xticks(x_positions)
    ax.set_xticklabels([f'{g}\u00b3' for g in grids])
    ax.set_xlabel("Grid Size", fontsize=11)
    ax.set_ylabel("Time (ms)", fontsize=11)
    ax.set_title(f"3D Stencil: Sync vs Overlap at {target_n} GPUs")
    ax.legend([legend_handles[k] for k in sorted(legend_handles)],
              [bar_labels[k] for k in sorted(legend_handles)],
              fontsize=9)
    ax.grid(axis='y', alpha=0.3, linestyle='--', color='#E8E8E8')

    plt.tight_layout()
    out = Path(output_dir) / '3d_stencil_comparison.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
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
    parser.add_argument('--input-dir',  default='results/json/3d_overlap/',
                        help='Directory containing JSON result files')
    parser.add_argument('--output-dir', default='docs/figures/',
                        help='Directory to write PNG files')
    parser.add_argument('--show', action='store_true',
                        help='Display figures interactively')
    args = parser.parse_args()

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

    plot_scaling_main(data, output_dir, show=show)
    plot_gain_heatmap(data, output_dir, show=show)
    plot_gain_by_gpu(data, output_dir, show=show)
    plot_scaling(data, output_dir, show=show)
    plot_stencil_comparison(data, output_dir, show=show)

    print(f"\nAll figures written to: {output_dir}")


if __name__ == '__main__':
    main()
