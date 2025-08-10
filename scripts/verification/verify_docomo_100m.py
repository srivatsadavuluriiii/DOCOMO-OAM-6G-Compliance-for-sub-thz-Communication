#!/usr/bin/env python3
"""
Quick verification at 100 m for ~100 GHz bands against DOCOMO-like target.

Evaluates SINR and Shannon throughput for mmWave (28 GHz), 140 GHz, and 300 GHz
at a fixed 100 m distance using the current hybrid 100 Gbps configuration.
"""

import os
import sys
import numpy as np

# Ensure project root on path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from environment.hybrid_oam_env import HybridOAM_Env
from utils.config_utils import load_config, merge_configs


def load_config_100gbps():
    base = load_config('config/base_config_new.yaml')
    hybrid = load_config('config/hybrid_6g_config.yaml')
    cfg100 = load_config('config/hybrid_6g_100gbps_config.yaml')
    cfg = merge_configs(base, hybrid)
    cfg = merge_configs(cfg, cfg100)
    return cfg


def evaluate_band(env: HybridOAM_Env, band: str, distance_m: float = 100.0):
    # Ensure we can switch immediately for verification
    env.steps_since_last_band_switch = env.min_band_switch_interval
    # Switch band (updates simulator, physics, mobility, reward)
    env._switch_band(band)

    # Fix position at 100 m on x-axis, zero velocity
    env.position = np.array([distance_m, 0.0, 0.0], dtype=float)
    env.velocity = np.array([0.0, 0.0, 0.0], dtype=float)
    env.current_mode = (env.min_mode + env.max_mode) // 2

    # Run simulator
    _, sinr_db = env.simulator.run_step(env.position, env.current_mode)
    throughput_bps = env._calculate_throughput(sinr_db)

    freq_ghz = env.frequency_bands[band] / 1e9
    bw_ghz = env.bandwidth_bands[band] / 1e9
    return {
        'band': band,
        'frequency_GHz': freq_ghz,
        'bandwidth_GHz': bw_ghz,
        'distance_m': distance_m,
        'sinr_dB': float(sinr_db),
        'throughput_Gbps': float(throughput_bps / 1e9)
    }


def main():
    cfg = load_config_100gbps()
    env = HybridOAM_Env(cfg)
    env.reset()

    results = []
    for band in ['mmwave', 'sub_thz_low', 'sub_thz_high']:
        results.append(evaluate_band(env, band, 100.0))

    print("\n=== 100 m Verification (Shannon throughput) ===")
    for r in results:
        print(f"Band={r['band']:12s} f={r['frequency_GHz']:6.1f} GHz  B={r['bandwidth_GHz']:6.1f} GHz  "
              f"SINR={r['sinr_dB']:6.2f} dB  C={r['throughput_Gbps']:7.2f} Gbps")

    print("\nNote: Throughput is Shannon limit B*log2(1+SINR) with current per-band bandwidths.")
    print("      High throughputs at sub-THz rely on very large B and high effective EIRP.")


if __name__ == '__main__':
    main()

