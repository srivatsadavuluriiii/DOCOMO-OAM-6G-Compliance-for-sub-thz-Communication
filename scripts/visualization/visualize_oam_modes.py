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


def plot_crosstalk_heatmap(crosstalk: np.ndarray, mode_indices: List[int], save_path: str) -> None:
    plt.figure(figsize=(8, 6))
    im = plt.imshow(crosstalk, cmap='viridis', origin='lower', aspect='auto')
    plt.colorbar(im, label='Crosstalk Magnitude')
    plt.xticks(ticks=np.arange(len(mode_indices)), labels=mode_indices)
    plt.yticks(ticks=np.arange(len(mode_indices)), labels=mode_indices)
    plt.xlabel('Mode index')
    plt.ylabel('Mode index')
    plt.title('OAM Crosstalk Matrix (magnitude)')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_mode_coupling_bars(crosstalk: np.ndarray, mode_indices: List[int], save_path: str) -> None:
    # Coupling strength per mode as sum of off-diagonal entries on each row
    off_diag = crosstalk.copy()
    np.fill_diagonal(off_diag, 0.0)
    coupling = off_diag.sum(axis=1)

    plt.figure(figsize=(10, 5))
    plt.bar(mode_indices, coupling, color='steelblue', alpha=0.85)
    plt.xlabel('Mode index')
    plt.ylabel('Coupling Strength (sum of off-diagonals)')
    plt.title('OAM Mode Coupling Strengths')
    plt.grid(True, axis='y', alpha=0.3)
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

    # Plot side-by-side bars for SINR and throughput
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(mode_indices, sinrs, color='darkorange', alpha=0.9)
    axes[0].set_xlabel('Mode index')
    axes[0].set_ylabel('SINR (dB)')
    axes[0].set_title(f'Mode SINR at distance {distance_m:.0f} m')
    axes[0].grid(True, axis='y', alpha=0.3)

    axes[1].bar(mode_indices, np.array(throughputs) / 1e9, color='seagreen', alpha=0.9)
    axes[1].set_xlabel('Mode index')
    axes[1].set_ylabel('Throughput (Gbps)')
    axes[1].set_title(f'Mode Throughput at distance {distance_m:.0f} m (B={bandwidth/1e6:.0f} MHz)')
    axes[1].grid(True, axis='y', alpha=0.3)

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
    max_mode = args.max_mode if args.max_mode is not None else int(cfg_oam.get('max_mode', 6))
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


