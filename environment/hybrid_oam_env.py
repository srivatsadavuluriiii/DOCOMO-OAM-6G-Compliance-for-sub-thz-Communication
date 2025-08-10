#!/usr/bin/env python3
"""
Hybrid OAM Environment for 6G Multi-Band Operation

This environment extends the base OAM environment to support multiple frequency bands
(mmWave, 140 GHz, 300 GHz) with intelligent band selection and switching.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Tuple, List, Optional, Protocol
import math
import os
import sys

# Ensure project root is on sys.path before importing project modules
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
from utils.path_utils import ensure_project_root_in_path
ensure_project_root_in_path()
from utils.exception_handler import safe_calculation, graceful_degradation, get_exception_handler

from environment.oam_env import OAM_Env, SimulatorInterface
from environment.physics_calculator import PhysicsCalculator
from environment.mobility_model import MobilityModel
from environment.reward_calculator import RewardCalculator


class HybridOAM_Env(OAM_Env):
    """
    Hybrid OAM environment supporting multiple frequency bands.
    
    This environment extends the base OAM environment to support:
    - Multiple frequency bands (mmWave, 140 GHz, 300 GHz)
    - Intelligent band selection based on distance, performance, and environment
    - Band switching with appropriate penalties and bonuses
    - Band-specific distance optimization
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, simulator: Optional[SimulatorInterface] = None):
        """
        Initialize the hybrid OAM environment.
        
        Args:
            config: Configuration dictionary with hybrid parameters
            simulator: Channel simulator instance
        """
        # Preserve config for internal use
        self._config = config if config is not None else {}
        super().__init__(config, simulator)
        
        # Initialize hybrid-specific parameters
        self._init_hybrid_parameters(config)
        
        # Initialize band selection logic
        self._init_band_selection()
        
        # Initialize band-specific components
        self._init_band_components()
        
        # Update state space for hybrid operation
        self._update_state_space()
    
    def _init_hybrid_parameters(self, config: Optional[Dict[str, Any]] = None):
        """Initialize hybrid-specific parameters."""
        if config is None:
            config = {}
        
        # Band configuration
        frequency_bands_raw = config.get('system', {}).get('frequency_bands', {
            'mmwave': 28.0e9,
            'sub_thz_low': 140.0e9,
            'sub_thz_high': 300.0e9
        })
        self.frequency_bands = {band: float(freq) for band, freq in frequency_bands_raw.items()}
        
        bandwidth_bands_raw = config.get('system', {}).get('bandwidth_bands', {
            'mmwave': 400e6,
            'sub_thz_low': 4e9,
            'sub_thz_high': 10e9
        })
        self.bandwidth_bands = {band: float(bw) for band, bw in bandwidth_bands_raw.items()}
        
        self.tx_power_bands = config.get('system', {}).get('tx_power_dBm', {
            'mmwave': 30.0,
            'sub_thz_low': 40.0,
            'sub_thz_high': 45.0
        })

        # Antenna gains per band (optional)
        system_cfg = config.get('system', {})
        self.tx_gain_bands = system_cfg.get('tx_antenna_gain_dBi', {
            'mmwave': 20.0,
            'sub_thz_low': 30.0,
            'sub_thz_high': 40.0
        })
        self.rx_gain_bands = system_cfg.get('rx_antenna_gain_dBi', {
            'mmwave': 20.0,
            'sub_thz_low': 30.0,
            'sub_thz_high': 40.0
        })

        # Band-specific beam widths
        self.beam_width_bands = config.get('oam', {}).get('beam_width_bands', {
            'mmwave': 0.03,
            'sub_thz_low': 0.006,
            'sub_thz_high': 0.003
        })
        
        self.max_throughput_bands = config.get('system', {}).get('max_throughput_gbps', {
            'mmwave': 2.0,
            'sub_thz_low': 20.0,
            'sub_thz_high': 100.0
        })
        
        # Current band state
        self.current_band = 'mmwave'  # Start with mmWave
        self.previous_band = 'mmwave'
        self.band_switch_count = 0
        self.band_switch_history = []
        
        # Band-specific area sizes
        self.area_size_bands = config.get('mobility', {}).get('area_size_bands', {
            'mmwave': np.array([500.0, 500.0]),
            'sub_thz_low': np.array([200.0, 200.0]),
            'sub_thz_high': np.array([100.0, 100.0])
        })
        
        # Update area size based on current band
        self.area_size = self.area_size_bands[self.current_band]
    
    def _init_band_selection(self):
        """Initialize band selection logic."""
        # Band selection strategy
        self.band_selection_strategy = 'adaptive'  # adaptive, distance_based, performance_based
        
        # Distance thresholds for band selection
        self.distance_thresholds = {
            'mmwave_max_distance': 300.0,
            'sub_thz_low_max_distance': 100.0,
            'sub_thz_high_max_distance': 50.0
        }
        
        # Performance thresholds
        self.performance_thresholds = {
            'throughput_threshold_gbps': 5.0,
            'sinr_threshold_dB': -10.0
        }
        
        # Band switching parameters
        self.min_band_switch_interval = 10
        self.band_switch_hysteresis = 0.2
        self.steps_since_last_band_switch = 0
    
    def _init_band_components(self):
        """Initialize band-specific components."""
        # Band-specific physics calculators
        self.physics_calculators = {}
        for band in self.frequency_bands.keys():
            self.physics_calculators[band] = PhysicsCalculator(self.bandwidth_bands[band])
        
        # Band-specific mobility models
        self.mobility_models = {}
        for band in self.frequency_bands.keys():
            self.mobility_models[band] = MobilityModel({
                'mobility': {
                    'area_size': self.area_size_bands[band]
                }
            })
        
        # Band-specific reward calculators
        self.reward_calculators = {}
        for band in self.frequency_bands.keys():
            self.reward_calculators[band] = RewardCalculator({
                'reward': {
                    'throughput_factor': 1.0,
                    'handover_penalty': 0.2,
                    'outage_penalty': 1.0,
                    'sinr_threshold': -5.0
                }
            })

        # Physics overrides per band from config (optional)
        physics_cfg = self._config.get('physics', {}) if self._config else {}
        self.atmo_absorption_bands = physics_cfg.get('atmospheric_absorption', {
            'mmwave': 0.1,
            'sub_thz_low': 2.0,
            'sub_thz_high': 12.0
        })
    
    def _update_state_space(self):
        """Update state space to include band information."""
        # Original state: [SINR, distance, vx, vy, vz, current_mode, min_mode, max_mode]
        # New state: [SINR, distance, vx, vy, vz, current_mode, min_mode, max_mode, 
        #            current_band, band_throughput]
        
        self.observation_space = spaces.Box(
            low=np.array([
                -30.0,                 # Minimum SINR in dB
                self.distance_min,     # Minimum distance
                -self.velocity_max,    # Minimum velocity in x
                -self.velocity_max,    # Minimum velocity in y
                -self.velocity_max,    # Minimum velocity in z
                self.min_mode,         # Minimum OAM mode
                self.min_mode,         # Minimum possible mode
                self.min_mode,         # Minimum of max mode
                0,                     # Current band (encoded)
                0.0                    # Band throughput (Gbps)
            ], dtype=np.float32),
            high=np.array([
                30.0,                  # Maximum SINR in dB
                self.distance_max,     # Maximum distance
                self.velocity_max,     # Maximum velocity in x
                self.velocity_max,     # Maximum velocity in y
                self.velocity_max,     # Maximum velocity in z
                self.max_mode,         # Maximum OAM mode
                self.max_mode,         # Maximum possible mode
                self.max_mode,         # Maximum of max mode
                2,                     # Current band (encoded: 0=mmwave, 1=sub_thz_low, 2=sub_thz_high)
                100.0                  # Band throughput (Gbps)
            ], dtype=np.float32),
            dtype=np.float32
        )
        
        # Action space: [STAY, UP, DOWN, SWITCH_BAND_UP, SWITCH_BAND_DOWN]
        self.action_space = spaces.Discrete(5)
    
    def _get_band_encoding(self, band: str) -> int:
        """Convert band string to integer encoding."""
        band_encodings = {
            'mmwave': 0,
            'sub_thz_low': 1,
            'sub_thz_high': 2
        }
        return band_encodings.get(band, 0)
    
    def _get_band_from_encoding(self, encoding: int) -> str:
        """Convert integer encoding to band string."""
        band_decodings = {
            0: 'mmwave',
            1: 'sub_thz_low',
            2: 'sub_thz_high'
        }
        return band_decodings.get(encoding, 'mmwave')
    
    def _select_optimal_band(self, distance: float, current_throughput: float, 
                            current_sinr: float) -> str:
        """
        Select optimal frequency band based on distance, performance, and environment.
        
        Args:
            distance: Current distance in meters
            current_throughput: Current throughput in Gbps
            current_sinr: Current SINR in dB
            
        Returns:
            Optimal band string
        """
        # Distance-based selection
        if distance <= self.distance_thresholds['sub_thz_high_max_distance']:
            distance_band = 'sub_thz_high'
        elif distance <= self.distance_thresholds['sub_thz_low_max_distance']:
            distance_band = 'sub_thz_low'
        else:
            distance_band = 'mmwave'
        
        # Performance-based selection
        performance_band = self.current_band  # Default to current band
        if current_throughput < self.performance_thresholds['throughput_threshold_gbps']:
            # Try to switch to higher frequency for better throughput
            if self.current_band == 'mmwave':
                performance_band = 'sub_thz_low'
            elif self.current_band == 'sub_thz_low':
                performance_band = 'sub_thz_high'
        
        if current_sinr < self.performance_thresholds['sinr_threshold_dB']:
            # Try to switch to lower frequency for better SINR
            if self.current_band == 'sub_thz_high':
                performance_band = 'sub_thz_low'
            elif self.current_band == 'sub_thz_low':
                performance_band = 'mmwave'
        
        # Combine distance and performance considerations
        if distance_band == performance_band:
            return distance_band
        elif distance_band == 'sub_thz_high' and performance_band == 'sub_thz_low':
            return 'sub_thz_low'  # Prefer 140 GHz over 300 GHz for stability
        elif distance_band == 'sub_thz_low' and performance_band == 'mmwave':
            return 'sub_thz_low'  # Prefer 140 GHz over mmWave for throughput
        else:
            return self.current_band  # Keep current band if unclear
    
    def _switch_band(self, new_band: str) -> bool:
        """
        Switch to a new frequency band.
        
        Args:
            new_band: Target band string
            
        Returns:
            True if band switch was successful, False otherwise
        """
        if new_band == self.current_band:
            return False
        
        # Check if enough time has passed since last band switch
        if self.steps_since_last_band_switch < self.min_band_switch_interval:
            return False
        
        # Update band state
        self.previous_band = self.current_band
        self.current_band = new_band
        self.band_switch_count += 1
        self.steps_since_last_band_switch = 0
        
        # Update area size for new band
        self.area_size = self.area_size_bands[self.current_band]
        
        # Update mobility model for new band
        self.mobility_model = self.mobility_models[self.current_band]
        
        # Update physics calculator for new band
        self.physics_calculator = self.physics_calculators[self.current_band]
        
        # Update reward calculator for new band
        self.reward_calculator = self.reward_calculators[self.current_band]

        # Update simulator with band-specific parameters (frequency, bandwidth, tx power, beam width)
        try:
            self.simulator._update_config({
                'system': {
                    'frequency': self.frequency_bands[self.current_band],
                    'bandwidth': self.bandwidth_bands[self.current_band],
                    'tx_power_dBm': self.tx_power_bands[self.current_band],
                    'tx_antenna_gain_dBi': self.tx_gain_bands.get(self.current_band, 0.0),
                    'rx_antenna_gain_dBi': self.rx_gain_bands.get(self.current_band, 0.0)
                },
                'oam': {
                    'beam_width': float(self.beam_width_bands.get(self.current_band, 0.03))
                },
                'physics': {
                    'atmospheric_absorption_dB_per_km': float(self.atmo_absorption_bands.get(self.current_band, 0.1))
                }
            })
        except Exception:
            # Fall back gracefully if simulator does not support dynamic updates
            pass

        # Clear throughput caches to avoid cross-band cache pollution
        self.clear_throughput_cache()
        if hasattr(self.physics_calculator, 'reset_cache'):
            self.physics_calculator.reset_cache()
        
        # Record band switch
        self.band_switch_history.append({
            'step': self.steps,
            'from_band': self.previous_band,
            'to_band': self.current_band,
            'distance': np.linalg.norm(self.position),
            'throughput': self._calculate_throughput(self.current_sinr)
        })
        
        return True
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the hybrid environment.
        
        Args:
            action: Action to take (0: STAY, 1: UP, 2: DOWN, 3: SWITCH_BAND_UP, 4: SWITCH_BAND_DOWN)
            
        Returns:
            Tuple of (next_state, reward, done, truncated, info)
        """
        self.steps += 1
        self.steps_since_last_band_switch += 1
        
        # Handle band switching actions
        if action == 3:  # SWITCH_BAND_UP
            band_order = ['mmwave', 'sub_thz_low', 'sub_thz_high']
            current_index = band_order.index(self.current_band)
            if current_index < len(band_order) - 1:
                new_band = band_order[current_index + 1]
                band_switched = self._switch_band(new_band)
            else:
                band_switched = False
        elif action == 4:  # SWITCH_BAND_DOWN
            band_order = ['mmwave', 'sub_thz_low', 'sub_thz_high']
            current_index = band_order.index(self.current_band)
            if current_index > 0:
                new_band = band_order[current_index - 1]
                band_switched = self._switch_band(new_band)
            else:
                band_switched = False
        else:
            band_switched = False
        
        # Execute base step with mode actions (0: STAY, 1: UP, 2: DOWN)
        if action < 3:
            next_state, reward, done, truncated, info = super().step(action)
        else:
            # For band switching actions, just update position and recalculate
            self._update_position()
            _, self.current_sinr = self.simulator.run_step(self.position, self.current_mode)
            throughput = self._calculate_throughput(self.current_sinr)
            reward = self.reward_calculator.calculate_reward(throughput, self.current_sinr, False)
            
            # Construct state
            next_state = np.array([
                self.current_sinr,
                np.linalg.norm(self.position),
                self.velocity[0],
                self.velocity[1],
                self.velocity[2],
                self.current_mode,
                self.min_mode,
                self.max_mode,
                self._get_band_encoding(self.current_band),
                throughput / 1e9  # Convert to Gbps
            ], dtype=np.float32)
            
            info = {
                'position': self.position,
                'velocity': self.velocity,
                'throughput': throughput,
                'handovers': 0,
                'sinr': self.current_sinr,
                'mode': self.current_mode,
                'band': self.current_band,
                'band_switched': band_switched
            }
            
            done = False
            truncated = (self.steps >= self.max_steps)
        
        # Add band-specific information to info
        info.update({
            'band': self.current_band,
            'band_switched': band_switched,
            'band_switch_count': self.band_switch_count,
            'frequency': self.frequency_bands[self.current_band],
            'bandwidth': self.bandwidth_bands[self.current_band],
            'max_throughput': self.max_throughput_bands[self.current_band]
        })
        
        # Update state with band information
        if action < 3:  # Only for mode actions, not band switching
            next_state = np.array([
                next_state[0],  # SINR
                next_state[1],  # distance
                next_state[2],  # vx
                next_state[3],  # vy
                next_state[4],  # vz
                next_state[5],  # current_mode
                next_state[6],  # min_mode
                next_state[7],  # max_mode
                self._get_band_encoding(self.current_band),  # current_band
                info['throughput'] / 1e9  # band_throughput in Gbps
            ], dtype=np.float32)
        
        return next_state, reward, done, truncated, info
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the hybrid environment.
        
        Args:
            seed: Random seed
            options: Additional options
            
        Returns:
            Tuple of (initial state, info dictionary)
        """
        # Reset base environment
        state, info = super().reset(seed=seed, options=options)
        
        # Reset hybrid-specific state
        self.current_band = 'mmwave'
        self.previous_band = 'mmwave'
        self.band_switch_count = 0
        self.steps_since_last_band_switch = 0
        self.band_switch_history.clear()
        
        # Update components for current band
        self.area_size = self.area_size_bands[self.current_band]
        self.mobility_model = self.mobility_models[self.current_band]
        self.physics_calculator = self.physics_calculators[self.current_band]
        self.reward_calculator = self.reward_calculators[self.current_band]

        # Ensure simulator is aligned to current band's parameters at reset
        try:
            self.simulator._update_config({
                'system': {
                    'frequency': self.frequency_bands[self.current_band],
                    'bandwidth': self.bandwidth_bands[self.current_band],
                    'tx_power_dBm': self.tx_power_bands[self.current_band],
                    'tx_antenna_gain_dBi': self.tx_gain_bands.get(self.current_band, 0.0),
                    'rx_antenna_gain_dBi': self.rx_gain_bands.get(self.current_band, 0.0)
                },
                'oam': {
                    'beam_width': float(self.beam_width_bands.get(self.current_band, 0.03))
                },
                'physics': {
                    'atmospheric_absorption_dB_per_km': float(self.atmo_absorption_bands.get(self.current_band, 0.1))
                }
            })
        except Exception:
            pass
        
        # Update state with band information
        state = np.array([
            state[0],  # SINR
            state[1],  # distance
            state[2],  # vx
            state[3],  # vy
            state[4],  # vz
            state[5],  # current_mode
            state[6],  # min_mode
            state[7],  # max_mode
            self._get_band_encoding(self.current_band),  # current_band
            info['throughput'] / 1e9  # band_throughput in Gbps
        ], dtype=np.float32)
        
        # Add band information to info
        info.update({
            'band': self.current_band,
            'frequency': self.frequency_bands[self.current_band],
            'bandwidth': self.bandwidth_bands[self.current_band],
            'max_throughput': self.max_throughput_bands[self.current_band]
        })
        
        # Clear caches at reset to ensure correct per-band throughput
        self.clear_throughput_cache()
        if hasattr(self.physics_calculator, 'reset_cache'):
            self.physics_calculator.reset_cache()

        return state, info
    
    def get_hybrid_stats(self) -> Dict[str, Any]:
        """
        Get hybrid system statistics.
        
        Returns:
            Dictionary containing hybrid system statistics
        """
        stats = {
            'current_band': self.current_band,
            'band_switch_count': self.band_switch_count,
            'band_switch_history': self.band_switch_history,
            'frequency_bands': self.frequency_bands,
            'bandwidth_bands': self.bandwidth_bands,
            'max_throughput_bands': self.max_throughput_bands
        }
        
        return stats 