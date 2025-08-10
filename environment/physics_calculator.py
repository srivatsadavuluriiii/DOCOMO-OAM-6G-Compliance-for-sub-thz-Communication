#!/usr/bin/env python3
"""
Physics calculator for OAM environment.

This module handles all physics-related calculations, separating them
from the RL environment logic to follow the Single Responsibility Principle.
"""

import numpy as np
import math
from typing import Tuple, Optional
import os
import sys

# Ensure project root is on sys.path before importing project modules
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
from utils.path_utils import ensure_project_root_in_path
ensure_project_root_in_path()
from utils.exception_handler import safe_calculation, graceful_degradation, get_exception_handler


class PhysicsCalculator:
    """
    Handles physics calculations for the OAM environment.
    
    This class is responsible for all physics-related calculations,
    including throughput calculation, signal processing, and other
    physical phenomena calculations.
    """
    
    def __init__(self, bandwidth: float = 400e6):
        """
        Initialize the physics calculator.
        
        Args:
            bandwidth: Channel bandwidth in Hz
        """
        self.bandwidth = float(bandwidth)
        self.max_sinr_dB = 60.0
        self.min_sinr_dB = -40.0
        
        # OPTIMIZED: Pre-compute frequently used constants
        self._max_throughput = None
        self._throughput_cache = {}
        self._sinr_linear_cache = {}
        
        # Pre-compute max throughput once
        self._precompute_constants()
    
    def _precompute_constants(self):
        """Pre-compute frequently used constants to avoid redundant calculations."""
        # Pre-compute max throughput: B * log2(1 + 10^(max_sinr/10))
        max_sinr_linear = 10 ** (self.max_sinr_dB / 10)
        self._max_throughput = self.bandwidth * math.log2(1 + max_sinr_linear)
        
        # Pre-compute common SINR linear values for caching
        common_sinr_values = np.arange(self.min_sinr_dB, self.max_sinr_dB + 1, 0.1)
        for sinr_dB in common_sinr_values:
            sinr_linear = 10 ** (sinr_dB / 10)
            self._sinr_linear_cache[round(sinr_dB, 1)] = sinr_linear
    
    def calculate_throughput(self, sinr_dB: float) -> float:
        """
        Calculate throughput using Shannon's formula with enhanced error handling and caching.
        
        Args:
            sinr_dB: Signal-to-Interference-plus-Noise Ratio in dB
            
        Returns:
            Throughput in bits per second
        """
        # Input validation
        if not isinstance(sinr_dB, (int, float)):
            return 0.0
            
        # Handle NaN or infinity
        if np.isnan(sinr_dB) or np.isinf(sinr_dB):
            return 0.0
        
        # OPTIMIZED: Check cache first
        sinr_rounded = round(sinr_dB, 1)
        if sinr_rounded in self._throughput_cache:
            return self._throughput_cache[sinr_rounded]
        
        # Clamp SINR to reasonable bounds
        sinr_dB = max(min(sinr_dB, self.max_sinr_dB), self.min_sinr_dB)
        
        # OPTIMIZED: Use cached SINR linear value if available
        if sinr_rounded in self._sinr_linear_cache:
            sinr_linear = self._sinr_linear_cache[sinr_rounded]
        else:
            # Convert SINR from dB to linear
            sinr_linear = 10 ** (sinr_dB / 10)
            # Cache for future use
            self._sinr_linear_cache[sinr_rounded] = sinr_linear
        
        # Shannon's formula: C = B * log2(1 + SINR)
        try:
            # Add small epsilon to avoid log(1) = 0 issues
            sinr_for_log = max(sinr_linear, 1e-10)
            throughput = self.bandwidth * math.log2(1 + sinr_for_log)
            
            # Validate result
            if np.isnan(throughput) or np.isinf(throughput) or throughput < 0:
                return 0.0
            
            # OPTIMIZED: Use pre-computed max throughput
            throughput = min(throughput, self._max_throughput)
            
            # Cache the result
            self._throughput_cache[sinr_rounded] = throughput
            
            return throughput
            
        except (ValueError, TypeError, OverflowError):
            return 0.0
    
    def calculate_sinr_from_power(self, signal_power_dBm: float, interference_power_dBm: float, 
                                 noise_power_dBm: float) -> float:
        """
        Calculate SINR from power measurements.
        
        Args:
            signal_power_dBm: Signal power in dBm
            interference_power_dBm: Interference power in dBm
            noise_power_dBm: Noise power in dBm
            
        Returns:
            SINR in dB
        """
        try:
            # Convert to linear scale
            signal_power_linear = 10 ** (signal_power_dBm / 10)
            interference_power_linear = 10 ** (interference_power_dBm / 10)
            noise_power_linear = 10 ** (noise_power_dBm / 10)
            
            # Calculate total interference + noise
            total_interference = interference_power_linear + noise_power_linear
            
            # Avoid division by zero
            if total_interference <= 0:
                return self.min_sinr_dB
            
            # Calculate SINR
            sinr_linear = signal_power_linear / total_interference
            
            # Convert to dB
            sinr_dB = 10 * math.log10(sinr_linear)
            
            # Clamp to reasonable bounds
            return max(min(sinr_dB, self.max_sinr_dB), self.min_sinr_dB)
            
        except (ValueError, TypeError, OverflowError):
            return self.min_sinr_dB
    
    def calculate_path_loss(self, distance: float, frequency: float = 28e9) -> float:
        """
        Calculate free-space path loss.
        
        Args:
            distance: Distance in meters
            frequency: Frequency in Hz
            
        Returns:
            Path loss in dB
        """
        try:
            # Speed of light
            c = 3e8
            
            # Wavelength
            wavelength = c / frequency
            
            # Free-space path loss: L = (4πd/λ)²
            path_loss_linear = (4 * math.pi * distance / wavelength) ** 2
            
            # Convert to dB
            path_loss_dB = 10 * math.log10(path_loss_linear)
            
            return path_loss_dB
            
        except (ValueError, TypeError, OverflowError):
            return 200.0  # Default high loss
    
    def calculate_received_power(self, tx_power_dBm: float, path_loss_dB: float, 
                                antenna_gain_dB: float = 0.0) -> float:
        """
        Calculate received power.
        
        Args:
            tx_power_dBm: Transmit power in dBm
            path_loss_dB: Path loss in dB
            antenna_gain_dB: Antenna gain in dB
            
        Returns:
            Received power in dBm
        """
        try:
            rx_power_dBm = tx_power_dBm - path_loss_dB + antenna_gain_dB
            return max(rx_power_dBm, -200.0)  # Minimum reasonable power
            
        except (ValueError, TypeError, OverflowError):
            return -200.0
    
    def calculate_distance_from_position(self, position: np.ndarray) -> float:
        """
        Calculate distance from position vector.
        
        Args:
            position: 3D position vector [x, y, z]
            
        Returns:
            Distance in meters
        """
        try:
            return float(np.linalg.norm(position))
        except (ValueError, TypeError, OverflowError):
            return 0.0
    
    def validate_physics_parameters(self, sinr_dB: float, throughput: float) -> Tuple[bool, str]:
        """
        Validate physics calculation results.
        
        Args:
            sinr_dB: SINR in dB
            throughput: Throughput in bps
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        errors = []
        
        # Validate SINR
        if not isinstance(sinr_dB, (int, float)):
            errors.append("SINR must be numeric")
        elif np.isnan(sinr_dB) or np.isinf(sinr_dB):
            errors.append("SINR is NaN or infinite")
        elif sinr_dB < self.min_sinr_dB or sinr_dB > self.max_sinr_dB:
            errors.append(f"SINR {sinr_dB} dB outside valid range [{self.min_sinr_dB}, {self.max_sinr_dB}]")
        
        # Validate throughput
        if not isinstance(throughput, (int, float)):
            errors.append("Throughput must be numeric")
        elif np.isnan(throughput) or np.isinf(throughput):
            errors.append("Throughput is NaN or infinite")
        elif throughput < 0:
            errors.append("Throughput cannot be negative")
        elif throughput > self.bandwidth * math.log2(1 + 10**(self.max_sinr_dB/10)):
            errors.append("Throughput exceeds theoretical maximum")
        
        if errors:
            return False, "; ".join(errors)
        else:
            return True, "Physics parameters valid"
    
    def get_physics_info(self) -> dict:
        """
        Get information about physics parameters.
        
        Returns:
            Dictionary with physics parameters
        """
        return {
            'bandwidth': self.bandwidth,
            'max_sinr_dB': self.max_sinr_dB,
            'min_sinr_dB': self.min_sinr_dB,
            'max_throughput': self.bandwidth * math.log2(1 + 10**(self.max_sinr_dB/10))
        } 
    
    def clear_cache(self):
        """Clear the throughput and SINR caches."""
        self._throughput_cache.clear()
        self._sinr_linear_cache.clear()
    
    def get_cache_stats(self) -> dict:
        """Get cache statistics for monitoring performance."""
        return {
            'throughput_cache_size': len(self._throughput_cache),
            'sinr_cache_size': len(self._sinr_linear_cache),
            'max_throughput': self._max_throughput,
            'bandwidth': self.bandwidth,
            'max_sinr_dB': self.max_sinr_dB,
            'min_sinr_dB': self.min_sinr_dB
        }
    
    def reset_cache(self):
        """Reset caches and re-precompute constants."""
        self.clear_cache()
        self._precompute_constants() 