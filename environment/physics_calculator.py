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
        """Pre-compute frequently used constants to avoid redundant calculations."""
                                                                    
        max_sinr_linear = 10 ** (self.max_sinr_dB / 10)
        self._max_throughput = self.bandwidth * math.log2(1 + max_sinr_linear)
        
                                                           
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
                          
        if not isinstance(sinr_dB, (int, float)):
            return 0.0
            
                                
        if np.isnan(sinr_dB) or np.isinf(sinr_dB):
            return 0.0
        
                                      
        sinr_rounded = round(sinr_dB, 1)
        if sinr_rounded in self._throughput_cache:
            return self._throughput_cache[sinr_rounded]
        
                                         
        sinr_dB = max(min(sinr_dB, self.max_sinr_dB), self.min_sinr_dB)
        
                                                              
        if sinr_rounded in self._sinr_linear_cache:
            sinr_linear = self._sinr_linear_cache[sinr_rounded]
        else:
                                            
            sinr_linear = 10 ** (sinr_dB / 10)
                                  
            self._sinr_linear_cache[sinr_rounded] = sinr_linear
        
                                                   
        try:
                                                          
            sinr_for_log = max(sinr_linear, 1e-10)
            throughput = self.bandwidth * math.log2(1 + sinr_for_log)
            
                             
            if np.isnan(throughput) or np.isinf(throughput) or throughput < 0:
                return 0.0
            
                                                        
            throughput = min(throughput, self._max_throughput)
            
                              
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
                                     
            signal_power_linear = 10 ** (signal_power_dBm / 10)
            interference_power_linear = 10 ** (interference_power_dBm / 10)
            noise_power_linear = 10 ** (noise_power_dBm / 10)
            
                                                  
            total_interference = interference_power_linear + noise_power_linear
            
                                    
            if total_interference <= 0:
                return self.min_sinr_dB
            
                            
            sinr_linear = signal_power_linear / total_interference
            
                           
            sinr_dB = 10 * math.log10(sinr_linear)
            
                                        
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
                            
            c = 3e8
            
                        
            wavelength = c / frequency
            
                                                
            path_loss_linear = (4 * math.pi * distance / wavelength) ** 2
            
                           
            path_loss_dB = 10 * math.log10(path_loss_linear)
            
            return path_loss_dB
            
        except (ValueError, TypeError, OverflowError):
            return 200.0                     

    def calculate_blockage_loss(self, frequency_ghz: float) -> float:
        """
        Calculates stochastic blockage loss based on the configured model.
        Higher frequencies are more susceptible.
        """
        if not self.blockage_enabled:
            return 0.0

        profile = self.blockage_profiles.get(self.blockage_model)
        if not profile:
            return 0.0

        # Blockage probability increases with frequency
        freq_factor = min((frequency_ghz / 100.0)**2, 5.0) # More likely above 100 GHz
        blockage_prob = profile.get('probability', 0.0) * freq_factor
        
        if np.random.random() < blockage_prob:
            loss_mean = profile.get('loss_mean_db', 20.0)
            loss_std = profile.get('loss_std_dev_db', 5.0)
            return np.random.normal(loss_mean, loss_std)

        return 0.0

    def calculate_clutter_loss(self, distance_m: float, frequency_ghz: float) -> float:
        """
        Calculates clutter loss based on 3GPP UMa/UMi models.
        """
        if not self.clutter_enabled or distance_m < 10:
            return 0.0

        if self.clutter_model == 'UMa_NLOS':
            # 3GPP TR 38.901 UMa NLOS model (simplified)
            log_dist = np.log10(distance_m)
            log_freq = np.log10(frequency_ghz)
            A = self.clutter_params['A']
            B = self.clutter_params['B']
            C = self.clutter_params['C']
            
            loss_db = A * log_dist + B * log_freq - C
            return max(0.0, loss_db)
        
        return 0.0

    def calculate_dynamic_pointing_loss(self, misalignment_deg: float, user_speed_kmh: float) -> float:
        """
        Calculates dynamic pointing loss, including loss-of-lock probability.
        """
        if self.pe_model != 'dynamic':
            # Fallback to a simple geometric model if not dynamic
            return 12 * (misalignment_deg**2)

        # Base geometric loss
        base_loss_db = 12 * (misalignment_deg**2)

        # Probability of losing lock increases with speed
        speed_factor = max(0, (user_speed_kmh - self.pe_params['high_speed_threshold_kmh']) / 100.0)
        loss_of_lock_prob = self.pe_params['loss_of_lock_base_prob'] * (1 + speed_factor * self.pe_params['loss_of_lock_speed_factor'])
        
        loss_of_lock = np.random.random() < loss_of_lock_prob

        if loss_of_lock:
            total_loss_db = base_loss_db + self.pe_params['loss_of_lock_penalty_db']
        else:
            total_loss_db = base_loss_db

        return total_loss_db

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
            return max(rx_power_dBm, -200.0)                            
            
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
        
                       
        if not isinstance(sinr_dB, (int, float)):
            errors.append("SINR must be numeric")
        elif np.isnan(sinr_dB) or np.isinf(sinr_dB):
            errors.append("SINR is NaN or infinite")
        elif sinr_dB < self.min_sinr_dB or sinr_dB > self.max_sinr_dB:
            errors.append(f"SINR {sinr_dB} dB outside valid range [{self.min_sinr_dB}, {self.max_sinr_dB}]")
        
                             
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
    
    def calculate_beam_divergence(self, frequency: float, distance: float, 
                                 initial_beam_width: float) -> Dict[str, float]:
        """
        Calculate proper diffractive beam spreading with distance.
        
        Implements: w(z) = w0 * sqrt(1 + (z/z_R)²)
        where z_R = π * w0² / λ is the Rayleigh range
        
        Args:
            frequency: Frequency in Hz
            distance: Propagation distance in meters
            initial_beam_width: Initial beam waist w0 in meters
            
        Returns:
            Dictionary with beam parameters at distance z
        """
        try:
            # Input validation
            if frequency <= 0 or initial_beam_width <= 0 or distance < 0:
                raise ValueError("Invalid input parameters")
            
            # Calculate wavelength
            wavelength = 3e8 / frequency
            
            # Rayleigh range
            rayleigh_range = math.pi * initial_beam_width**2 / wavelength
            
            # Beam radius at distance z
            if distance == 0:
                beam_width = initial_beam_width
            else:
                beam_width = initial_beam_width * math.sqrt(1 + (distance / rayleigh_range)**2)
            
            # Radius of curvature
            if distance == 0:
                radius_of_curvature = float('inf')
            else:
                radius_of_curvature = distance * (1 + (rayleigh_range / distance)**2)
            
            # Divergence angle (far-field)
            divergence_angle = wavelength / (math.pi * initial_beam_width)
            
            # Gouy phase
            gouy_phase = math.atan(distance / rayleigh_range) if rayleigh_range > 0 else 0
            
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

    def calculate_enhanced_throughput(self, sinr_dB: float, frequency: float = 28e9,
                                    modulation_scheme: str = "adaptive",
                                    coding_rate: float = 0.8) -> Dict[str, float]:
        """
        Calculate enhanced throughput with practical constraints.
        
        Implements:
        - Adaptive modulation based on SNR thresholds
        - Practical SNR limits for mmWave/THz
        - Coding gain factors
        - Link margin requirements
        
        Args:
            sinr_dB: Signal-to-Interference-plus-Noise Ratio in dB
            frequency: Operating frequency in Hz
            modulation_scheme: Modulation scheme ("adaptive", "qpsk", "16qam", "64qam", "256qam")
            coding_rate: Forward error correction coding rate (0.5 to 1.0)
            
        Returns:
            Dictionary with throughput analysis
        """
        try:
            # Input validation
            if not isinstance(sinr_dB, (int, float)) or np.isnan(sinr_dB) or np.isinf(sinr_dB):
                raise ValueError("Invalid SINR input")
            if frequency <= 0:
                raise ValueError("Invalid frequency")
            
            # Frequency-dependent practical limits
            if frequency >= 100e9:  # Sub-THz
                max_practical_sinr = 35.0  # dB, limited by phase noise
                min_practical_sinr = -5.0   # dB, atmospheric limitations
                link_margin = 3.0           # dB, additional margin for THz
            elif frequency >= 60e9:  # mmWave high
                max_practical_sinr = 40.0
                min_practical_sinr = -10.0
                link_margin = 2.0
            elif frequency >= 28e9:  # mmWave mid
                max_practical_sinr = 45.0
                min_practical_sinr = -15.0
                link_margin = 1.5
            else:  # Sub-6 GHz
                max_practical_sinr = 50.0
                min_practical_sinr = -20.0
                link_margin = 1.0
            
            # Apply practical SINR limits
            effective_sinr = max(min(sinr_dB - link_margin, max_practical_sinr), min_practical_sinr)
            
            # Adaptive modulation and coding
            if modulation_scheme == "adaptive":
                if effective_sinr >= 25.0:
                    modulation = "256qam"
                    spectral_efficiency = 8.0  # bits/s/Hz
                    required_sinr = 25.0
                elif effective_sinr >= 20.0:
                    modulation = "64qam"
                    spectral_efficiency = 6.0
                    required_sinr = 20.0
                elif effective_sinr >= 15.0:
                    modulation = "16qam"
                    spectral_efficiency = 4.0
                    required_sinr = 15.0
                elif effective_sinr >= 8.0:
                    modulation = "qpsk"
                    spectral_efficiency = 2.0
                    required_sinr = 8.0
                else:
                    modulation = "bpsk"
                    spectral_efficiency = 1.0
                    required_sinr = 3.0
            else:
                # Fixed modulation scheme
                modulation_params = {
                    "bpsk": (1.0, 3.0),
                    "qpsk": (2.0, 8.0),
                    "16qam": (4.0, 15.0),
                    "64qam": (6.0, 20.0),
                    "256qam": (8.0, 25.0)
                }
                spectral_efficiency, required_sinr = modulation_params.get(
                    modulation_scheme, (2.0, 8.0)
                )
                modulation = modulation_scheme
            
            # Check if SINR meets requirement
            if effective_sinr < required_sinr:
                # Fallback to lower modulation or fail
                if effective_sinr >= 3.0:
                    modulation = "bpsk"
                    spectral_efficiency = 1.0
                else:
                    # Link failure
                    return {
                        'shannon_throughput': 0.0,
                        'practical_throughput': 0.0,
                        'modulation': "none",
                        'spectral_efficiency': 0.0,
                        'coding_rate': 0.0,
                        'coding_gain_db': 0.0,
                        'effective_sinr_db': effective_sinr,
                        'link_margin_db': link_margin,
                        'link_status': 'failed'
                    }
            
            # Coding gain (typical values for LDPC/Turbo codes)
            if coding_rate >= 0.9:
                coding_gain_db = 1.0
            elif coding_rate >= 0.8:
                coding_gain_db = 2.0
            elif coding_rate >= 0.7:
                coding_gain_db = 3.0
            elif coding_rate >= 0.5:
                coding_gain_db = 4.0
            else:
                coding_gain_db = 5.0
            
            # Shannon limit throughput
            sinr_linear = 10 ** (effective_sinr / 10)
            shannon_throughput = self.bandwidth * math.log2(1 + sinr_linear)
            
            # Practical throughput with modulation and coding
            practical_throughput = self.bandwidth * spectral_efficiency * coding_rate
            
            # Apply coding gain to effective throughput
            effective_throughput = min(practical_throughput, shannon_throughput * 0.8)  # 80% of Shannon
            
            return {
                'shannon_throughput': shannon_throughput,
                'practical_throughput': effective_throughput,
                'modulation': modulation,
                'spectral_efficiency': spectral_efficiency,
                'coding_rate': coding_rate,
                'coding_gain_db': coding_gain_db,
                'effective_sinr_db': effective_sinr,
                'link_margin_db': link_margin,
                'link_status': 'active'
            }
            
        except (ValueError, TypeError, OverflowError):
            return {
                'shannon_throughput': 0.0,
                'practical_throughput': 0.0,
                'modulation': "none",
                'spectral_efficiency': 0.0,
                'coding_rate': 0.0,
                'coding_gain_db': 0.0,
                'effective_sinr_db': sinr_dB,
                'link_margin_db': 0.0,
                'link_status': 'error'
            }

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
