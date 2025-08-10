#!/usr/bin/env python3
"""
Mock simulator for testing environment dependency injection.
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional
import os
import sys

# Ensure project root is on sys.path before importing project modules
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
from utils.path_utils import ensure_project_root_in_path
ensure_project_root_in_path()

# Support both package-relative and absolute imports
try:
    from .oam_env import SimulatorInterface
except ImportError:
    from environment.oam_env import SimulatorInterface


class MockChannelSimulator(SimulatorInterface):
    """
    Mock channel simulator for testing.
    
    This implements the SimulatorInterface and provides
    predictable, deterministic behavior for unit testing.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the mock simulator.
        
        Args:
            config: Configuration dictionary (optional)
        """
        self.config = config or {}
        self.step_count = 0
        
        # Mock parameters
        self.base_sinr = self.config.get('mock_base_sinr', 15.0)  # dB
        self.sinr_variation = self.config.get('mock_sinr_variation', 2.0)  # dB
        self.mode_penalty = self.config.get('mock_mode_penalty', 1.0)  # dB per mode
        
        # Required attributes that the environment expects
        self.bandwidth = self.config.get('bandwidth', 400e6)  # 400 MHz
        self.frequency = self.config.get('frequency', 28.0e9)  # 28 GHz
        self.wavelength = 3e8 / self.frequency  # Speed of light / frequency
        self.tx_power_dBm = self.config.get('tx_power_dBm', 30.0)  # dBm
        self.noise_figure_dB = self.config.get('noise_figure_dB', 8.0)  # dB
        self.noise_temp = self.config.get('noise_temp', 290.0)  # K
        self.min_mode = self.config.get('min_mode', 1)
        self.max_mode = self.config.get('max_mode', 8)
        self.beam_width = self.config.get('beam_width', 0.03)  # 30 mrad
        
    def run_step(self, position: np.ndarray, current_mode: int) -> Tuple[np.ndarray, float]:
        """
        Run a mock simulation step.
        
        Args:
            position: Current position [x, y, z]
            current_mode: Current OAM mode
            
        Returns:
            Tuple of (channel_matrix, sinr_dB)
        """
        self.step_count += 1
        
        # Calculate distance from origin
        distance = np.linalg.norm(position[:2])  # 2D distance
        
        # Mock SINR calculation
        # Base SINR decreases with distance
        distance_factor = max(0.1, 1.0 - distance / 1000.0)  # Distance penalty
        
        # Mode penalty (higher modes have lower SINR)
        mode_penalty = (current_mode - 1) * self.mode_penalty
        
        # Add some deterministic variation
        variation = self.sinr_variation * np.sin(self.step_count * 0.1)
        
        # Calculate final SINR
        sinr_dB = self.base_sinr * distance_factor - mode_penalty + variation
        
        # Ensure SINR is within reasonable bounds
        sinr_dB = np.clip(sinr_dB, -30.0, 50.0)
        
        # Mock channel matrix (identity matrix for simplicity)
        channel_matrix = np.eye(6, dtype=np.complex128)  # 6x6 identity matrix
        
        return channel_matrix, sinr_dB
    
    def reset(self):
        """Reset the mock simulator state."""
        self.step_count = 0


class DeterministicMockSimulator(SimulatorInterface):
    """
    Deterministic mock simulator for reproducible testing.
    
    This simulator always returns the same results for the same inputs,
    making it perfect for unit testing.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the deterministic mock simulator.
        
        Args:
            config: Configuration dictionary (optional)
        """
        self.config = config or {}
        
        # Fixed parameters for deterministic behavior
        self.fixed_sinr = self.config.get('fixed_sinr', 20.0)  # dB
        self.fixed_channel_matrix = np.eye(6, dtype=np.complex128)
        
        # Required attributes that the environment expects
        self.bandwidth = self.config.get('bandwidth', 400e6)  # 400 MHz
        self.frequency = self.config.get('frequency', 28.0e9)  # 28 GHz
        self.wavelength = 3e8 / self.frequency  # Speed of light / frequency
        self.tx_power_dBm = self.config.get('tx_power_dBm', 30.0)  # dBm
        self.noise_figure_dB = self.config.get('noise_figure_dB', 8.0)  # dB
        self.noise_temp = self.config.get('noise_temp', 290.0)  # K
        self.min_mode = self.config.get('min_mode', 1)
        self.max_mode = self.config.get('max_mode', 8)
        self.beam_width = self.config.get('beam_width', 0.03)  # 30 mrad
        
    def run_step(self, position: np.ndarray, current_mode: int) -> Tuple[np.ndarray, float]:
        """
        Run a deterministic simulation step.
        
        Args:
            position: Current position [x, y, z]
            current_mode: Current OAM mode
            
        Returns:
            Tuple of (channel_matrix, sinr_dB)
        """
        # Always return the same results for deterministic testing
        return self.fixed_channel_matrix, self.fixed_sinr
    
    def reset(self):
        """Reset the deterministic simulator state."""
        pass  # No state to reset


class FailingMockSimulator(SimulatorInterface):
    """
    Mock simulator that fails to test error handling.
    
    This simulator raises exceptions to test how the environment
    handles simulator failures.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, fail_after: int = 5):
        """
        Initialize the failing mock simulator.
        
        Args:
            config: Configuration dictionary (optional)
            fail_after: Number of steps before failing
        """
        self.config = config or {}
        self.fail_after = fail_after
        self.step_count = 0
        
        # Required attributes that the environment expects
        self.bandwidth = self.config.get('bandwidth', 400e6)  # 400 MHz
        self.frequency = self.config.get('frequency', 28.0e9)  # 28 GHz
        self.wavelength = 3e8 / self.frequency  # Speed of light / frequency
        self.tx_power_dBm = self.config.get('tx_power_dBm', 30.0)  # dBm
        self.noise_figure_dB = self.config.get('noise_figure_dB', 8.0)  # dB
        self.noise_temp = self.config.get('noise_temp', 290.0)  # K
        self.min_mode = self.config.get('min_mode', 1)
        self.max_mode = self.config.get('max_mode', 8)
        self.beam_width = self.config.get('beam_width', 0.03)  # 30 mrad
        
    def run_step(self, position: np.ndarray, current_mode: int) -> Tuple[np.ndarray, float]:
        """
        Run a simulation step that may fail.
        
        Args:
            position: Current position [x, y, z]
            current_mode: Current OAM mode
            
        Returns:
            Tuple of (channel_matrix, sinr_dB)
            
        Raises:
            RuntimeError: After fail_after steps
        """
        self.step_count += 1
        
        if self.step_count > self.fail_after:
            raise RuntimeError(f"Mock simulator failed after {self.fail_after} steps")
        
        # Return normal results before failing
        channel_matrix = np.eye(6, dtype=np.complex128)
        sinr_dB = 15.0
        
        return channel_matrix, sinr_dB
    
    def reset(self):
        """Reset the failing simulator state."""
        self.step_count = 0 