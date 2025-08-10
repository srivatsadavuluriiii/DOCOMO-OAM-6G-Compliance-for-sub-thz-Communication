"""
Benchmark tests for the channel simulator.
"""

import numpy as np
import pytest
from simulator.channel_simulator import ChannelSimulator


@pytest.fixture
def simulator():
    """Create a channel simulator for testing."""
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
        }
    }
    return ChannelSimulator(config)


def test_simulator_run_step(benchmark, simulator):
    """Benchmark the run_step method."""
    position = np.array([100.0, 0.0, 0.0])
    velocity = np.array([1.0, 0.0, 0.0])
    mode = 1
    
    benchmark(simulator.run_step, position, velocity, mode)


def test_simulator_calculate_path_loss(benchmark, simulator):
    """Benchmark the _calculate_path_loss method."""
    distance = 100.0
    
    benchmark(simulator._calculate_path_loss, distance)


def test_simulator_calculate_atmospheric_absorption(benchmark, simulator):
    """Benchmark the _calculate_atmospheric_absorption method."""
    distance = 100.0
    
    benchmark(simulator._calculate_atmospheric_absorption, distance)


def test_simulator_generate_turbulence_screen(benchmark, simulator):
    """Benchmark the _generate_turbulence_screen method."""
    distance = 100.0
    
    benchmark(simulator._generate_turbulence_screen, distance)


def test_simulator_calculate_crosstalk(benchmark, simulator):
    """Benchmark the _calculate_crosstalk method."""
    distance = 100.0
    mode = 1
    
    benchmark(simulator._calculate_crosstalk, distance, mode)