#!/usr/bin/env python3
import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import torch

# Ensure project root on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from environment.docomo_6g_env import DOCOMO_6G_Environment
from models.agent import Agent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot distance-to-throughput and distance-to-handover (baseline vs trained)")
    p.add_argument("--config", type=str, required=True, help="Path to DOCOMO config YAML")
    p.add_argument("--model-dir", type=str, default=None, help="Directory with best_model.pth (for trained runs)")
    p.add_argument("--episodes", type=int, default=3, help="Number of episodes per run")
    p.add_argument("--steps", type=int, default=200, help="Max steps per episode")
    p.add_argument("--output", type=str, required=True, help="Output directory for plots and CSVs")
    p.add_argument("--baseline", action="store_true", help="Run a baseline random-policy pass")
    return p.parse_args()


def load_agent_for_env(env, model_dir: str, device: torch.device) -> Agent:
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = Agent(state_dim=state_dim, action_dim=action_dim, hidden_layers=[256, 256], device=device)
    if model_dir:
        ckpt_path = Path(model_dir) / "best_model.pth"
        if ckpt_path.exists():
            ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
            if 'policy_net_state_dict' in ckpt:
                agent.policy_net.load_state_dict(ckpt['policy_net_state_dict'])
            if 'target_net_state_dict' in ckpt:
                agent.target_net.load_state_dict(ckpt['target_net_state_dict'])
    agent.policy_net.to(device)
    agent.target_net.to(device)
    return agent


def collect_rollout(config_path: str, episodes: int, max_steps: int, device: torch.device,
                    model_dir: str = None, baseline: bool = False) -> List[Dict[str, Any]]:
    env = DOCOMO_6G_Environment(config_path=config_path)
    agent = None if baseline else load_agent_for_env(env, model_dir, device)
    data: List[Dict[str, Any]] = []

    for ep in range(episodes):
        state, _ = env.reset()
        for t in range(max_steps):
            if baseline or agent is None:
                action = env.action_space.sample()
            else:
                action = agent.choose_action(np.asarray(state, dtype=np.float32), epsilon=0.0)
            next_state, reward, done, truncated, info = env.step(action)
            perf = info.get('performance_metrics', {})
            action_info = info.get('action_info', {})
            data.append({
                'step': len(data),
                'distance_m': float(perf.get('distance_m', 0.0)),
                'throughput_gbps': float(perf.get('throughput_gbps', 0.0)),
                'handover': bool(action_info.get('handover_occurred', False)),
                'band': info.get('performance_metrics', {}).get('frequency_ghz', None),
            })
            state = next_state
            if done or truncated:
                break
    return data


def plot_distance_relationships(baseline_data: List[Dict[str, Any]], trained_data: List[Dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save CSVs
    def save_csv(name: str, rows: List[Dict[str, Any]]):
        import csv
        with open(out_dir / name, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['step', 'distance_m', 'throughput_gbps', 'handover', 'band'])
            w.writeheader()
            for r in rows:
                w.writerow(r)

    save_csv('baseline.csv', baseline_data)
    save_csv('trained.csv', trained_data)

    # Throughput vs Distance scatter (alpha blending)
    plt.figure(figsize=(10,6))
    if baseline_data:
        bd = np.array([[r['distance_m'], r['throughput_gbps']] for r in baseline_data], dtype=float)
        plt.scatter(bd[:,0], bd[:,1], s=8, alpha=0.3, label='Baseline')
    if trained_data:
        td = np.array([[r['distance_m'], r['throughput_gbps']] for r in trained_data], dtype=float)
        plt.scatter(td[:,0], td[:,1], s=8, alpha=0.3, label='Trained')
    plt.xlabel('Distance (m)')
    plt.ylabel('Throughput (Gbps)')
    plt.title('Throughput vs Distance')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / 'throughput_vs_distance.png', dpi=300)
    plt.close()

    # Handover rate vs Distance (bin)
    def handover_rate_by_bin(rows: List[Dict[str, Any]], bins: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not rows:
            return bins[:-1], np.zeros_like(bins[:-1], dtype=float)
        dist = np.array([r['distance_m'] for r in rows], dtype=float)
        ho = np.array([1.0 if r['handover'] else 0.0 for r in rows], dtype=float)
        idx = np.digitize(dist, bins) - 1
        rates = []
        for b in range(len(bins)-1):
            mask = idx == b
            if np.any(mask):
                rates.append(ho[mask].mean())
            else:
                rates.append(0.0)
        centers = 0.5 * (bins[:-1] + bins[1:])
        return centers, np.array(rates)

    # Define bins from combined distance range
    all_dist = np.array([r['distance_m'] for r in (baseline_data + trained_data)], dtype=float) if (baseline_data or trained_data) else np.array([0.0, 1.0])
    dmin, dmax = float(np.nanmin(all_dist)), float(np.nanmax(all_dist))
    if not np.isfinite(dmin) or not np.isfinite(dmax) or dmax <= dmin:
        dmin, dmax = 0.0, 1000.0
    bins = np.linspace(dmin, dmax, num=21)

    b_x, b_y = handover_rate_by_bin(baseline_data, bins)
    t_x, t_y = handover_rate_by_bin(trained_data, bins)

    plt.figure(figsize=(10,6))
    if baseline_data:
        plt.plot(b_x, b_y, label='Baseline', alpha=0.8)
    if trained_data:
        plt.plot(t_x, t_y, label='Trained', alpha=0.8)
    plt.xlabel('Distance (m)')
    plt.ylabel('Handover Rate (per step)')
    plt.title('Handover Rate vs Distance')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / 'handover_rate_vs_distance.png', dpi=300)
    plt.close()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    baseline_rows: List[Dict[str, Any]] = []
    trained_rows: List[Dict[str, Any]] = []

    if args.baseline:
        baseline_rows = collect_rollout(args.config, args.episodes, args.steps, device, model_dir=None, baseline=True)

    if args.model_dir:
        trained_rows = collect_rollout(args.config, args.episodes, args.steps, device, model_dir=args.model_dir, baseline=False)

    plot_distance_relationships(baseline_rows, trained_rows, out_dir)
    print(f"Saved plots to {out_dir}")


if __name__ == "__main__":
    main()


