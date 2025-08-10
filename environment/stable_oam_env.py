import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Tuple, List, Optional
import math
import os
import sys

# Use centralized path management instead of sys.path.append
from utils.path_utils import ensure_project_root_in_path
ensure_project_root_in_path()

# Support both package-relative and absolute imports for direct execution
try:
    from .oam_env import OAM_Env
except ImportError:
    from environment.oam_env import OAM_Env
from simulator.channel_simulator import ChannelSimulator
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from simulator.channel_simulator import ChannelSimulator as SimulatorInterface


class StableOAM_Env(OAM_Env):
    """
    Stable OAM Environment with enhanced reward function.
    
    This environment extends the base OAM_Env with a more stable reward function
    that includes throughput normalization, SINR scaling, and relative improvement tracking.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, simulator: Optional['SimulatorInterface'] = None):
        """
        Initialize the stable OAM environment.
        
        Args:
            config: Dictionary containing environment parameters
            simulator: Channel simulator instance (for dependency injection)
        """
        super().__init__(config, simulator)
        
        # Initialize stable reward parameters with defaults
        self.reward_scale = 1.0
        self.reward_min = -10.0
        self.reward_max = 10.0
        self.throughput_window = 10
        self.throughput_factor = 1.0
        self.sinr_scaling_factor = 0.5
        self.handover_penalty = 0.5
        self.outage_penalty = 10.0
        self.sinr_threshold = 0.0
        self.reward_smoothing_factor = 0.1
        
        # Initialize stable reward tracking variables
        self.throughput_history = []
        self.baseline_throughput = None
        self.previous_reward = 0.0
        self.previous_mode = None  # Track previous mode for handover detection
        
        # Update with provided config
        if config:
            self._update_config(config)
        
    def _update_config(self, config: Dict[str, Any]) -> None:
        """
        Update stable reward parameters from configuration.
        
        Args:
            config: Dictionary containing stable reward parameters
        """
        if 'stable_reward' in config:
            stable_config = config['stable_reward']
            
            # Reward scaling parameters
            self.reward_scale = stable_config.get('reward_scale', 1.0)
            self.reward_min = stable_config.get('reward_min', -10.0)
            self.reward_max = stable_config.get('reward_max', 10.0)
            
            # Throughput normalization parameters
            self.throughput_window = stable_config.get('throughput_window', 10)
            self.throughput_factor = stable_config.get('throughput_factor', 1.0)
            
            # SINR scaling parameters
            self.sinr_scaling_factor = stable_config.get('sinr_scaling_factor', 0.5)
            
            # Penalty parameters
            self.handover_penalty = stable_config.get('handover_penalty', 0.5)
            self.outage_penalty = stable_config.get('outage_penalty', 10.0)
            self.sinr_threshold = stable_config.get('sinr_threshold', 0.0)
            
            # Smoothing parameters
            self.reward_smoothing_factor = stable_config.get('reward_smoothing_factor', 0.1)
        
        # Ensure OAM mode values are preserved from parent
        if 'oam' in config:
            oam_config = config['oam']
            if 'min_mode' in oam_config:
                self.min_mode = int(oam_config['min_mode'])
            if 'max_mode' in oam_config:
                self.max_mode = int(oam_config['max_mode'])
        
        # Set default OAM parameters if not provided in config
        if self.min_mode is None:
            self.min_mode = 1
        if self.max_mode is None:
            self.max_mode = 8
            
    def _get_moving_avg_throughput(self, current_throughput: float) -> float:
        """
        Calculate moving average throughput.
        
        Args:
            current_throughput: Current step throughput
            
        Returns:
            Moving average throughput
        """
        self.throughput_history.append(current_throughput)
        
        # Keep only the last N values
        if len(self.throughput_history) > self.throughput_window:
            self.throughput_history.pop(0)
        
        # Calculate moving average
        if len(self.throughput_history) > 0:
            return np.mean(self.throughput_history)
        else:
            return current_throughput
    
    def _normalize_throughput(self, throughput: float) -> float:
        """
        Normalize throughput to a reasonable range.
        
        Args:
            throughput: Raw throughput in bps
            
        Returns:
            Normalized throughput
        """
        # Normalize to Gbps for numerical stability
        throughput_gbps = throughput / 1e9
        
        # Apply log scaling to handle large dynamic range
        if throughput_gbps > 0:
            return np.log10(1 + throughput_gbps)
        else:
            return 0.0
    
    def _calculate_stable_reward(self, throughput: float, sinr_dB: float, handover_occurred: bool) -> float:
        """
        Calculate stable reward with enhanced components.
        
        Args:
            throughput: Current throughput
            sinr_dB: Current SINR in dB
            handover_occurred: Whether a handover occurred in this step
            
        Returns:
            Calculated reward
        """
        # Calculate moving average throughput
        avg_throughput = self._get_moving_avg_throughput(throughput)
        
        # Initialize baseline throughput if not set
        if self.baseline_throughput is None:
            self.baseline_throughput = avg_throughput
        
        # Normalize throughput
        normalized_throughput = self._normalize_throughput(avg_throughput)
        
        # Calculate relative improvement over baseline
        if self.baseline_throughput > 0:
            relative_improvement = (avg_throughput - self.baseline_throughput) / self.baseline_throughput
            relative_improvement = np.clip(relative_improvement, -1.0, 1.0)  # Clip to reasonable range
        else:
            relative_improvement = 0.0
        
        # Calculate base reward
        base_reward = self.throughput_factor * normalized_throughput
        
        # Add SINR contribution (scaled)
        sinr_contribution = self.sinr_scaling_factor * (sinr_dB / 30.0)  # Normalize SINR to reasonable range
        
        # Add relative improvement contribution
        improvement_contribution = 2.0 * relative_improvement  # Scale factor of 2.0
        
        # Apply handover penalty if mode was changed
        handover_penalty = self.handover_penalty if handover_occurred else 0.0
        
        # Apply outage penalty if SINR is below threshold
        outage_penalty = self.outage_penalty if sinr_dB < self.sinr_threshold else 0.0
        
        # Combine components
        raw_reward = base_reward + sinr_contribution + improvement_contribution - handover_penalty - outage_penalty
        
        # Scale the reward
        scaled_reward = self.reward_scale * raw_reward
        
        # Clip to min/max range
        clipped_reward = np.clip(scaled_reward, self.reward_min, self.reward_max)
        
        # Apply exponential smoothing
        smoothed_reward = (1 - self.reward_smoothing_factor) * clipped_reward + self.reward_smoothing_factor * self.previous_reward
        
        # Update previous reward for next step
        final_reward = smoothed_reward
        self.previous_reward = smoothed_reward
        
        # Gradually update baseline (slow tracking)
        self.baseline_throughput = 0.99 * self.baseline_throughput + 0.01 * avg_throughput
        
        return final_reward
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to an initial state.
        
        Args:
            seed: Random seed
            options: Additional options
            
        Returns:
            Tuple of (initial state, info dictionary)
        """
        # Reset stable reward variables
        self.throughput_history = []
        self.baseline_throughput = None
        self.previous_reward = 0.0
        self.previous_mode = None  # Reset previous mode tracking
        
        # Call parent reset
        return super().reset(seed=seed, options=options)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment with a more stable reward function.
        
        Args:
            action: Action to take (0: STAY, 1: UP, 2: DOWN)
            
        Returns:
            Tuple of (next state, reward, done, truncated, info)
        """
        # Store the previous mode before taking action
        self.previous_mode = self.current_mode
        
        # Call parent step method to handle common functionality
        next_state, _, done, truncated, info = super().step(action)
        
        # Extract values from info dictionary
        throughput = info['throughput']
        
        # FIXED: Proper per-step handover detection
        # Compare current mode with previous mode to detect handover in this step
        current_mode = info['mode']
        handover_occurred = (self.previous_mode is not None and 
                           self.previous_mode != current_mode)
        
        # Calculate stable reward
        reward = self._calculate_stable_reward(throughput, self.current_sinr, handover_occurred)
            
        # Handle NaN or infinity in reward
        if np.isnan(reward) or np.isinf(reward):
            reward = -self.outage_penalty  # Default to penalty value
        
        # Add additional info
        smoothed = self._get_moving_avg_throughput(throughput)
        info['avg_throughput'] = smoothed
        info['smoothed_throughput'] = float(smoothed)
        info['raw_reward'] = float(reward)
        info['handover_occurred'] = handover_occurred  # Add handover info for debugging
        
        return next_state, reward, done, truncated, info