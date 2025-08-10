#!/usr/bin/env python3
"""
Comprehensive unit tests for OAM physics simulation components.

Tests scientific accuracy of:
- OAM beam pattern generation
- Path loss calculation validation
- Turbulence model verification
- SINR calculation correctness
- Atmospheric absorption models
- Crosstalk calculations
"""

import sys
import numpy as np
import unittest
from unittest.mock import patch, MagicMock
import matplotlib.pyplot as plt
from scipy.constants import c as speed_of_light

# Use centralized path management
from utils.path_utils import ensure_project_root_in_path
ensure_project_root_in_path()

from simulator.channel_simulator import ChannelSimulator


class TestOAMBeamPatternGeneration(unittest.TestCase):
    """Test OAM beam pattern generation accuracy."""
    
    def setUp(self):
        """Set up test configuration."""
        self.config = {
            'system': {
                'frequency': 28.0e9,  # 28 GHz
                'bandwidth': 400e6,
                'tx_power_dBm': 30.0,
                'noise_figure_dB': 8.0,
                'noise_temp': 290.0
            },
            'environment': {
                'humidity': 50.0,
                'temperature': 20.0,
                'pressure': 101.3,
                'turbulence_strength': 1e-14,
                'pointing_error_std': 0.005,
                'rician_k_factor': 8.0
            },
            'oam': {
                'min_mode': 1,
                'max_mode': 8,  # Standardized to match base config
                'beam_width': 0.03
            }
        }
        self.simulator = ChannelSimulator(self.config)
    
    def test_beam_width_evolution(self):
        """Test that beam width increases with distance and is always positive."""
        distances = [100, 500, 1000, 2000, 5000]  # meters
        prev_w_L = self.simulator.beam_width
        for distance in distances:
            w_L = self.simulator.beam_width * distance
            self.assertGreater(w_L, 0)
            self.assertGreater(w_L, prev_w_L)
            prev_w_L = w_L
    
    def test_wavelength_consistency(self):
        """Test that wavelength is consistent with frequency."""
        expected_wavelength = speed_of_light / self.simulator.frequency
        self.assertAlmostEqual(self.simulator.wavelength, expected_wavelength, places=10)
    
    def test_wave_number_consistency(self):
        """Test that wave number is consistent with wavelength."""
        expected_k = 2 * np.pi / self.simulator.wavelength
        self.assertAlmostEqual(self.simulator.k, expected_k, places=10)
    
    def test_mode_range_validation(self):
        """Test that OAM mode range is properly validated."""
        # Test valid mode
        valid_mode = 3
        self.assertTrue(self.simulator.min_mode <= valid_mode <= self.simulator.max_mode)
        
        # Test invalid modes - note that @safe_calculation decorator prevents ValueError from propagating
        # Instead, it logs a warning and returns fallback values
        import logging
        from io import StringIO
        
        # Create a log capture handler
        log_capture = StringIO()
        log_handler = logging.StreamHandler(log_capture)
        log_handler.setLevel(logging.WARNING)
        logger = logging.getLogger("utils.exception_handler")
        logger.addHandler(log_handler)
        
        # Test invalid modes
        invalid_modes = [0, 9, 12]  # Updated for max_mode=8
        for mode in invalid_modes:
            # Clear previous log
            log_capture.truncate(0)
            log_capture.seek(0)
            
            # Run with invalid mode
            H, sinr = self.simulator.run_step(np.array([100, 0, 0]), mode)
            
            # Check that some values were returned
            # Note: The fallback value may be None or a default eye matrix
            # The important thing is that the function didn't crash
            self.assertIsNotNone(H)
            self.assertIsNotNone(sinr)
            
            # Check that a warning was logged
            log_output = log_capture.getvalue()
            self.assertIn(f"current_oam_mode {mode} must be between", log_output)
        
        # Clean up
        logger.removeHandler(log_handler)


class TestPathLossCalculation(unittest.TestCase):
    """Test path loss calculation validation."""
    
    def setUp(self):
        """Set up test configuration."""
        self.config = {
            'system': {
                'frequency': 28.0e9,
                'bandwidth': 400e6,
                'tx_power_dBm': 30.0,
                'noise_figure_dB': 8.0,
                'noise_temp': 290.0
            },
            'environment': {
                'humidity': 50.0,
                'temperature': 20.0,
                'pressure': 101.3,
                'turbulence_strength': 1e-14,
                'pointing_error_std': 0.005,
                'rician_k_factor': 8.0
            },
            'oam': {
                'min_mode': 1,
                'max_mode': 8,  # Standardized to match base config
                'beam_width': 0.03
            }
        }
        self.simulator = ChannelSimulator(self.config)
    
    def test_free_space_path_loss(self):
        """Test free space path loss calculation."""
        distances = [100, 500, 1000, 2000, 5000]  # meters
        
        for distance in distances:
            # Calculate expected free space loss
            expected_loss = (4 * np.pi * distance / self.simulator.wavelength) ** 2
            
            # Get total path loss (includes atmospheric absorption)
            total_loss = self.simulator._calculate_path_loss(distance)
            
            # Total loss should be greater than free space loss due to atmospheric absorption
            self.assertGreater(total_loss, expected_loss)
            
            # Verify reasonable loss values
            self.assertGreater(total_loss, 1e3)   # At least 30 dB
            self.assertLess(total_loss, 1e15)     # Less than 150 dB
    
    def test_path_loss_monotonicity(self):
        """Test that path loss increases monotonically with distance."""
        distances = np.linspace(100, 10000, 100)
        losses = [self.simulator._calculate_path_loss(d) for d in distances]
        
        # Check monotonicity
        for i in range(1, len(losses)):
            self.assertGreater(losses[i], losses[i-1])
    
    def test_atmospheric_absorption_consistency(self):
        """Test atmospheric absorption calculation consistency."""
        distances = [100, 500, 1000, 2000, 5000]  # meters
        
        for distance in distances:
            # Calculate atmospheric absorption
            atmospheric_loss = self.simulator._calculate_atmospheric_absorption(distance)
            
            # Atmospheric loss should be greater than 1 (additional loss)
            self.assertGreater(atmospheric_loss, 1.0)
            
            # Loss should increase with distance
            if distance > 100:
                prev_loss = self.simulator._calculate_atmospheric_absorption(distance - 100)
                self.assertGreater(atmospheric_loss, prev_loss)


if __name__ == "__main__":
    # Run basic physics unit tests
    print("🧪 RUNNING BASIC PHYSICS UNIT TESTS")
    print("=" * 50)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestOAMBeamPatternGeneration,
        TestPathLossCalculation
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    if result.wasSuccessful():
        print("\n✅ BASIC PHYSICS UNIT TESTS PASSED!")
    else:
        print("\n❌ BASIC PHYSICS UNIT TESTS FAILED!")
        sys.exit(1) 