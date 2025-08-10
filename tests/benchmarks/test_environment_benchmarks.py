"""
Benchmark tests for the OAM environment.
"""

import numpy as np
import pytest
import gymnasium as gym
from environment.oam_env import OAM_Env
from simulator.channel_simulator import ChannelSimulator


@pytest.fixture
def env():
    """Create an OAM environment for testing."""
    config = {
        'system': {
            'frequency': 28.0e9,
            'bandwidth': 100.0e6,
            'tx_power_dBm': 30.0,
            'noise_figure_dB': 5.0,
            'noise_temp': 290.0,
        },
        'oam': {
            'min_mode': 1,
            'max_mode': 8,
            'beam_width': 0.03,
        },
        'environment': {
            'humidity': 50.0,
            'temperature': 20.0,
            'pressure': 101325.0,
            'turbulence_strength': 1e-14,
            'rician_k_factor': 10.0,
            'pointing_error_sigma': 0.01,
            'max_distance': 1000.0,
            'max_steps': 1000,
            'max_speed': 5.0,
            'min_speed': 0.5,
        },
        'reward': {
            'throughput_factor': 1.0,
            'handover_penalty': 10.0,
        },
    }
    simulator = ChannelSimulator(config)
    return OAM_Env(config, simulator=simulator)


def test_environment_reset(benchmark, env):
    """Benchmark the reset method."""
    benchmark(env.reset)


def test_environment_step(benchmark, env):
    """Benchmark the step method."""
    env.reset()
    action = 1  # Choose the middle action
    
    benchmark(env.step, action)


def test_environment_calculate_throughput(benchmark, env):
    """Benchmark the _calculate_throughput method."""
    env.reset()
    position = np.array([100.0, 0.0, 0.0])
    mode = 1
    
    benchmark(env._calculate_throughput, position, mode)