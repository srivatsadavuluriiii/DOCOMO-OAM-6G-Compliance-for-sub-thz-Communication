#!/usr/bin/env python3
"""
Visualize OAM modes: crosstalk matrix, mode coupling, and mode comparison.

This script generates:
- Crosstalk matrix heatmap across OAM modes
- Coupling strength between modes (row-wise off-diagonal power)
- Comparison of per-mode SINR and estimated throughput at a fixed distance

Outputs are saved under an output directory (default: plots/oam_modes/).
"""

import os
import sys
import argparse
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt


# Ensure project root in sys.path (robust for standalone execution)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from utils.path_utils import ensure_project_root_in_path
ensure_project_root_in_path()

from utils.config_utils import load_config
from simulator.channel_simulator import ChannelSimulator


def build_simulator(config_path: str) -> Tuple[ChannelSimulator, dict]:
    cfg = load_config(config_path)
    sim = ChannelSimulator(cfg)
    return sim, cfg


def compute_crosstalk_matrix(sim: ChannelSimulator, distance_m: float) -> np.ndarray:
    # Generate turbulence screen and compute crosstalk matrix using simulator internals
    turbulence_screen = sim._generate_turbulence_screen(distance_m)
    x_talk = sim._calculate_crosstalk(distance_m, turbulence_screen)
    return np.abs(x_talk)


def _set_publication_style() -> None:
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except Exception:
        pass
    plt.rcParams.update({
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 10,
        'figure.titlesize': 14,
    })

# Centralized palettes (aligned with gallery)
HEATMAP_CMAP = 'plasma'
THROUGHPUT_CMAP = 'turbo'


def plot_crosstalk_heatmap(crosstalk: np.ndarray, mode_indices: List[int], save_path: str) -> None:
    _set_publication_style()
    plt.figure(figsize=(8.5, 7.0))
    im = plt.imshow(crosstalk, cmap=HEATMAP_CMAP, origin='lower', aspect='auto')
    plt.colorbar(im, label='Crosstalk Magnitude', fraction=0.046, pad=0.04)
    plt.xticks(ticks=np.arange(len(mode_indices)), labels=mode_indices)
    plt.yticks(ticks=np.arange(len(mode_indices)), labels=mode_indices)
    plt.xlabel('Receiving mode index')
    plt.ylabel('Transmitting mode index')
    plt.title('OAM Crosstalk Matrix (magnitude)')
    # annotate
    for i in range(len(mode_indices)):
        for j in range(len(mode_indices)):
            val = crosstalk[i, j]
            color = 'white' if val > 0.6 else 'black'
            plt.text(j, i, f'{val:.2f}', ha='center', va='center', color=color, fontsize=8)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_mode_coupling_bars(crosstalk: np.ndarray, mode_indices: List[int], save_path: str) -> None:
    _set_publication_style()
    # Coupling strength per mode as sum of off-diagonal entries on each row
    off_diag = crosstalk.copy()
    np.fill_diagonal(off_diag, 0.0)
    coupling = off_diag.sum(axis=1)

    plt.figure(figsize=(10, 4.5))
    # Color bars with turbo colormap based on normalized coupling strength
    coupling_arr = np.asarray(coupling, dtype=float)
    max_c = float(np.max(coupling_arr)) if np.any(coupling_arr > 0) else 1.0
    norm_vals = (coupling_arr / max_c) if max_c > 0 else coupling_arr
    cmap = plt.get_cmap(THROUGHPUT_CMAP)
    colors = [cmap(v) for v in norm_vals]
    bars = plt.bar(mode_indices, coupling_arr, color=colors, alpha=0.95)
    plt.xlabel('Mode index')
    plt.ylabel('Coupling Strength (sum of off-diagonals)')
    plt.title('OAM Mode Coupling Strengths')
    plt.grid(True, axis='y', alpha=0.3)
    # annotate bars
    for b in bars:
        height = b.get_height()
        plt.text(b.get_x() + b.get_width()/2.0, height, f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def compare_modes_sinr_throughput(
    sim: ChannelSimulator,
    mode_indices: List[int],
    distance_m: float,
    save_path: str,
) -> None:
    # Fixed position along x-axis, z=2m (as used in tests)
    user_position = np.array([distance_m, 0.0, 2.0])
    sinrs = []
    throughputs = []
    bandwidth = getattr(sim, 'bandwidth', 400e6)

    for mode in mode_indices:
        _, sinr_db = sim.run_step(user_position, mode)
        sinrs.append(sinr_db)
        # Shannon capacity: C = B * log2(1 + SNR)
        snr_lin = max(10 ** (sinr_db / 10.0), 0.0)
        cap_bps = bandwidth * np.log2(1.0 + snr_lin)
        throughputs.append(cap_bps)

    _set_publication_style()
    # Plot side-by-side bars for SINR and throughput
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

    bars0 = axes[0].bar(mode_indices, sinrs, color='darkorange', alpha=0.9)
    axes[0].set_xlabel('Mode index')
    axes[0].set_ylabel('SINR (dB)')
    axes[0].set_title(f'Mode SINR at distance {distance_m:.0f} m')
    axes[0].grid(True, axis='y', alpha=0.3)
    for b in bars0:
        h = b.get_height()
        axes[0].text(b.get_x() + b.get_width()/2.0, h, f'{h:.1f}', ha='center', va='bottom', fontsize=8)

    tp_gbps = np.array(throughputs) / 1e9
    # Color throughput bars by turbo colormap based on normalized throughput
    max_tp = float(np.max(tp_gbps)) if np.any(tp_gbps > 0) else 1.0
    tp_norm = (tp_gbps / max_tp) if max_tp > 0 else tp_gbps
    cmap_tp = plt.get_cmap(THROUGHPUT_CMAP)
    tp_colors = [cmap_tp(v) for v in tp_norm]
    bars1 = axes[1].bar(mode_indices, tp_gbps, color=tp_colors, alpha=0.95)
    axes[1].set_xlabel('Mode index')
    axes[1].set_ylabel('Throughput (Gbps)')
    axes[1].set_title(f'Mode Throughput at distance {distance_m:.0f} m (B={bandwidth/1e6:.0f} MHz)')
    axes[1].grid(True, axis='y', alpha=0.3)
    for b in bars1:
        h = b.get_height()
        axes[1].text(b.get_x() + b.get_width()/2.0, h, f'{h:.1f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Visualize OAM modes: crosstalk, coupling, comparison')
    p.add_argument('--config', type=str, default='config/base_config_new.yaml', help='Path to YAML config')
    p.add_argument('--distance', type=float, default=1000.0, help='Evaluation distance in meters')
    p.add_argument('--output-dir', type=str, default='plots/oam_modes', help='Output directory for plots')
    p.add_argument('--min-mode', type=int, default=None, help='Minimum OAM mode to include')
    p.add_argument('--max-mode', type=int, default=None, help='Maximum OAM mode to include')
    return p.parse_args()


def main():
    args = parse_args()
    sim, cfg = build_simulator(args.config)

    # Determine mode indices from config/simulator
    cfg_oam = cfg.get('oam', {}) if isinstance(cfg, dict) else {}
    min_mode = args.min_mode if args.min_mode is not None else int(cfg_oam.get('min_mode', 1))
    # default to 8 modes if max not specified in the config
    max_mode = args.max_mode if args.max_mode is not None else int(cfg_oam.get('max_mode', 8))
    mode_indices = list(range(min_mode, max_mode + 1))

    os.makedirs(args.output_dir, exist_ok=True)

    # 1) Crosstalk matrix heatmap
    x_talk = compute_crosstalk_matrix(sim, args.distance)
    heatmap_path = os.path.join(args.output_dir, 'crosstalk_heatmap.png')
    plot_crosstalk_heatmap(x_talk, mode_indices, heatmap_path)

    # 2) Mode coupling bars
    coupling_path = os.path.join(args.output_dir, 'mode_coupling_bars.png')
    plot_mode_coupling_bars(x_talk, mode_indices, coupling_path)

    # 3) Mode comparison (SINR and throughput)
    compare_path = os.path.join(args.output_dir, 'mode_comparison_sinr_throughput.png')
    compare_modes_sinr_throughput(sim, mode_indices, args.distance, compare_path)

    print('\nVisualization complete. Files saved to:')
    print(f'  - {heatmap_path}')
    print(f'  - {coupling_path}')
    print(f'  - {compare_path}')


if __name__ == '__main__':
    main()


