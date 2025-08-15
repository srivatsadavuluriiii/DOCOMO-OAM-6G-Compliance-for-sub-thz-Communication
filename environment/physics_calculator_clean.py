#!/usr/bin/env python3
"""
Physics calculator for OAM environment.

This module handles all physics-related calculations, separating them
from the RL environment logic to follow the Single Responsibility Principle.
"""

import numpy as np
import math
from typing import Tuple, Optional, Dict, Any
import os
import sys
from .unified_physics_engine import UnifiedPhysicsEngine

                                                                     
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
    
    def __init__(self, config: dict, bandwidth: float = 400e6):
        """
        Initialize the physics calculator.
        
        Args:
            config: Global configuration dictionary
            bandwidth: Channel bandwidth in Hz
        """
        self.config = config
        self.bandwidth = float(bandwidth)
        self.max_sinr_dB = 60.0
        self.min_sinr_dB = -40.0
        
        # Initialize unified physics engine for consistent performance
        self.unified_engine = UnifiedPhysicsEngine(config)

        # Clutter model configuration
        clutter_cfg = self.config.get('physics', {}).get('clutter', {})
        self.clutter_enabled = clutter_cfg.get('enabled', False)
        self.clutter_model = clutter_cfg.get('model', 'UMa_NLOS')
        self.clutter_params = {
            'A': clutter_cfg.get('loss_factor_A', 26.0),
            'B': clutter_cfg.get('loss_factor_B', 20.0),
            'C': clutter_cfg.get('loss_constant_C', 22.7)
        }

        # Dynamic pointing error configuration
        pe_cfg = self.config.get('physics', {}).get('pointing_error', {})
        self.pe_model = pe_cfg.get('model', 'dynamic')
        self.pe_params = {
            'tracking_speed_deg_s': pe_cfg.get('tracking_speed_deg_s', 180.0),
            'high_speed_threshold_kmh': pe_cfg.get('high_speed_threshold_kmh', 300.0),
            'loss_of_lock_base_prob': pe_cfg.get('loss_of_lock_base_prob', 0.001),
            'loss_of_lock_speed_factor': pe_cfg.get('loss_of_lock_speed_factor', 5.0),
            'loss_of_lock_penalty_db': pe_cfg.get('loss_of_lock_penalty_db', 10.0)
        }

        # Stochastic blockage model configuration
        blockage_cfg = self.config.get('physics', {}).get('blockage', {})
        self.blockage_enabled = blockage_cfg.get('enabled', False)
        self.blockage_model = blockage_cfg.get('model', 'urban')
        self.blockage_profiles = blockage_cfg.get('profiles', {})
        
        self._max_throughput = None
        self._throughput_cache = {}
        self._sinr_linear_cache = {}
        
                                         
        self._precompute_constants()
    
    def _precompute_constants(self):
        """
        Precompute constants for efficiency
        """
        pass

    def calculate_throughput(self, sinr_dB: float) -> float:
        """
        Calculate Shannon throughput.
        
        Args:
            sinr_dB: Signal-to-Interference-plus-Noise Ratio in dB
            
        Returns:
            Throughput in bits per second
        """
        try:
            # Validate input
            if np.isnan(sinr_dB) or np.isinf(sinr_dB):
                return 0.0
            
            # Clip to reasonable range
            sinr_dB = np.clip(sinr_dB, self.min_sinr_dB, self.max_sinr_dB)
            
            # Convert to linear scale
            sinr_linear = 10 ** (sinr_dB / 10.0)
            
            # Shannon capacity
            throughput = self.bandwidth * math.log2(1 + sinr_linear)
            
            return float(throughput)
            
        except (ValueError, OverflowError):
            return 0.0

    def calculate_enhanced_throughput(self, sinr_dB: float, frequency: float = 28e9,
                                    modulation_scheme: str = "adaptive",
                                    coding_rate: float = 0.8,
                                    distance_m: float = 100.0,
                                    oam_modes: int = 1) -> Dict[str, float]:
        """
        Calculate enhanced throughput using unified physics engine.
        Ensures consistent performance across all scenarios and frequencies.
        
        Args:
            sinr_dB: Signal-to-Interference-plus-Noise Ratio in dB
            frequency: Operating frequency in Hz
            modulation_scheme: Modulation scheme (kept for compatibility)
            coding_rate: Coding rate (kept for compatibility)
            distance_m: Distance for scenario detection
            oam_modes: Number of OAM modes for spatial multiplexing
            
        Returns:
            Dictionary with throughput analysis
        """
        # Use the unified physics engine for consistent results
        result = self.unified_engine.calculate_realistic_throughput(
            frequency_hz=frequency,
            distance_m=distance_m,
            sinr_db=sinr_dB,
            bandwidth_hz=self.bandwidth,
            oam_modes=oam_modes
        )
        
        # Convert to expected format for compatibility
        return {
            'shannon_throughput': result['shannon_throughput'],
            'practical_throughput': result['practical_throughput'], 
            'modulation': modulation_scheme,
            'spectral_efficiency': 8.0,  # Reasonable default
            'coding_rate': coding_rate,
            'coding_gain_db': 2.0,  # Reasonable default
            'effective_sinr_db': result['effective_sinr'],
            'link_margin_db': 3.0,  # Reasonable default
            'link_status': 'active',
            'scenario': result['scenario'],
            'frequency_ghz': result['frequency_ghz'],
            'enhancement_factor': result['enhancement_factor']
        }

    def get_physics_info(self) -> dict:
        """Get physics information for debugging"""
        return {
            'bandwidth': self.bandwidth,
            'max_sinr_dB': self.max_sinr_dB,
            'min_sinr_dB': self.min_sinr_dB,
            'clutter_enabled': self.clutter_enabled,
            'blockage_enabled': self.blockage_enabled,
        } 
    
    def clear_cache(self):
        """Clear the throughput and SINR caches."""
        self._throughput_cache.clear()
        self._sinr_linear_cache.clear()

    def calculate_beam_divergence(self, frequency: float, distance: float, 
                                 initial_beam_width: float) -> Dict[str, float]:
        """Calculate beam divergence parameters"""
        try:
            # Wavelength
            wavelength = 3e8 / frequency
            
            # Rayleigh range
            rayleigh_range = (math.pi * initial_beam_width**2) / wavelength
            
            # Beam width at distance
            if distance <= rayleigh_range:
                beam_width = initial_beam_width * math.sqrt(1 + (distance / rayleigh_range)**2)
            else:
                beam_width = initial_beam_width * (distance / rayleigh_range)
            
            # Radius of curvature
            if distance == 0:
                radius_of_curvature = float('inf')
            else:
                radius_of_curvature = distance * (1 + (rayleigh_range / distance)**2)
            
            # Divergence angle
            divergence_angle = wavelength / (math.pi * initial_beam_width)
            
            # Gouy phase
            gouy_phase = math.atan(distance / rayleigh_range)
            
            return {
                'beam_width': beam_width,
                'rayleigh_range': rayleigh_range,
                'radius_of_curvature': radius_of_curvature,
                'divergence_angle': divergence_angle,
                'gouy_phase': gouy_phase,
                'beam_area': math.pi * beam_width**2,
                'beam_expansion_factor': beam_width / initial_beam_width
            }
            
        except (ValueError, TypeError, OverflowError, ZeroDivisionError):
            # Return safe defaults for invalid inputs
            return {
                'beam_width': initial_beam_width,
                'rayleigh_range': 1000.0,
                'radius_of_curvature': float('inf'),
                'divergence_angle': 0.001,
                'gouy_phase': 0.0,
                'beam_area': math.pi * initial_beam_width**2,
                'beam_expansion_factor': 1.0
            }
