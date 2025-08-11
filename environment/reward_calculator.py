#!/usr/bin/env python3
"""
Reward calculator for OAM environment.

This module handles reward calculations and logic, separating them
from the RL environment logic to follow the Single Responsibility Principle.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any


class RewardCalculator:
    """
    Handles reward calculations for the OAM environment.
    
    This class is responsible for all reward-related calculations,
    including throughput rewards, handover penalties, and outage penalties.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the reward calculator.
        
        Args:
            config: Configuration dictionary with reward parameters
        """
        # Default reward parameters
        self.throughput_factor = 1.0
        self.handover_penalty = 1.0
        self.outage_penalty = 10.0
        self.sinr_threshold = 0.0  # dB
        
        # Update with provided config
        if config and 'reward' in config:
            reward_config = config['reward']
            self.throughput_factor = reward_config.get('throughput_factor', self.throughput_factor)
            self.handover_penalty = reward_config.get('handover_penalty', self.handover_penalty)
            self.outage_penalty = reward_config.get('outage_penalty', self.outage_penalty)
            self.sinr_threshold = reward_config.get('sinr_threshold', self.sinr_threshold)
        
        # Multi-objective (optional)
        self.multi_objective_enabled = False
        self.mo_weights = {
            'throughput': 0.0,
            'stability': 0.0,
            'energy': 0.0,
            'handover': 0.0,
        }

        # Episode tracking
        self.episode_throughput = 0.0
        self.episode_handovers = 0

        # Read optional multi-objective weights
        if isinstance(config, dict):
            # Prefer rl_base.reward.multi_objective, also allow reward.multi_objective
            mo_cfg = (
                config.get('rl_base', {}).get('reward', {}).get('multi_objective')
                if isinstance(config.get('rl_base', {}), dict) else None
            )
            if mo_cfg is None:
                mo_cfg = config.get('reward', {}).get('multi_objective')
            if isinstance(mo_cfg, dict) and any(k in mo_cfg for k in self.mo_weights.keys()):
                self.multi_objective_enabled = True
                for k in self.mo_weights.keys():
                    try:
                        self.mo_weights[k] = float(mo_cfg.get(k, self.mo_weights[k]))
                    except Exception:
                        pass
        
    def calculate_reward(self, throughput: float, sinr_dB: float, handover_occurred: bool) -> float:
        """
        Calculate reward based on throughput, SINR, and handover events.
        
        Args:
            throughput: Current throughput in bps
            sinr_dB: Current SINR in dB
            handover_occurred: Whether a handover occurred in this step
            
        Returns:
            Calculated reward
        """
        if self.multi_objective_enabled:
            # Components
            throughput_gbps = throughput / 1e9
            stability_score = 0.0 if handover_occurred else 1.0
            energy_cost = 0.0  # Placeholder; extend when energy metrics available

            reward = (
                self.mo_weights['throughput'] * throughput_gbps
                + self.mo_weights['stability'] * stability_score
                - self.mo_weights['handover'] * (1.0 if handover_occurred else 0.0)
                - self.mo_weights['energy'] * energy_cost
            )
        else:
            # Start with throughput reward (normalized to Gbps)
            reward = self.throughput_factor * (throughput / 1e9)
            # Apply handover penalty if mode was changed
            if handover_occurred:
                # Cap per-step handover penalty to be consistent
                reward -= min(self.handover_penalty, 2.0)
        
        # Track handovers consistently
        if handover_occurred:
            self.episode_handovers += 1
        
        # Apply outage penalty if SINR is below threshold (applies to both paths)
        if sinr_dB < self.sinr_threshold:
            reward -= min(self.outage_penalty, 10.0)
            
        # Handle NaN or infinity in reward
        if np.isnan(reward) or np.isinf(reward):
            reward = -self.outage_penalty  # Default to penalty value
        
        # Update episode tracking
        self.episode_throughput += throughput
        
        # Final per-step clipping to avoid runaway values
        return max(min(reward, 10.0), -10.0)
    
    def calculate_throughput_reward(self, throughput: float) -> float:
        """
        Calculate reward component from throughput.
        
        Args:
            throughput: Current throughput in bps
            
        Returns:
            Throughput reward component
        """
        # Normalize throughput to Gbps for numerical stability
        throughput_gbps = throughput / 1e9
        return self.throughput_factor * throughput_gbps
    
    def calculate_handover_penalty(self, handover_occurred: bool) -> float:
        """
        Calculate handover penalty.
        
        Args:
            handover_occurred: Whether a handover occurred
            
        Returns:
            Handover penalty (negative value)
        """
        if handover_occurred:
            self.episode_handovers += 1
            return -self.handover_penalty
        return 0.0
    
    def calculate_outage_penalty(self, sinr_dB: float) -> float:
        """
        Calculate outage penalty based on SINR.
        
        Args:
            sinr_dB: Current SINR in dB
            
        Returns:
            Outage penalty (negative value)
        """
        if sinr_dB < self.sinr_threshold:
            return -self.outage_penalty
        return 0.0
    
    def calculate_combined_reward(self, throughput: float, sinr_dB: float, 
                                handover_occurred: bool) -> Tuple[float, Dict[str, float]]:
        """
        Calculate combined reward with detailed breakdown.
        
        Args:
            throughput: Current throughput in bps
            sinr_dB: Current SINR in dB
            handover_occurred: Whether a handover occurred
            
        Returns:
            Tuple of (total_reward, reward_breakdown)
        """
        # Calculate individual components
        throughput_reward = self.calculate_throughput_reward(throughput)
        handover_penalty = self.calculate_handover_penalty(handover_occurred)
        outage_penalty = self.calculate_outage_penalty(sinr_dB)
        
        # Calculate total reward
        total_reward = throughput_reward + handover_penalty + outage_penalty
        
        # Handle NaN or infinity
        if np.isnan(total_reward) or np.isinf(total_reward):
            total_reward = -self.outage_penalty
        
        # Update episode tracking
        self.episode_throughput += throughput
        
        # Create reward breakdown
        reward_breakdown = {
            'throughput_reward': throughput_reward,
            'handover_penalty': handover_penalty,
            'outage_penalty': outage_penalty,
            'total_reward': total_reward
        }
        
        return total_reward, reward_breakdown
    
    def reset_episode(self) -> None:
        """
        Reset episode tracking variables.
        """
        self.episode_throughput = 0.0
        self.episode_handovers = 0
    
    def get_episode_stats(self) -> Dict[str, float]:
        """
        Get episode statistics.
        
        Returns:
            Dictionary with episode statistics
        """
        return {
            'episode_throughput': self.episode_throughput,
            'episode_handovers': self.episode_handovers,
            'avg_throughput': self.episode_throughput / max(1, self.episode_handovers + 1)
        }
    
    def validate_reward_parameters(self) -> Tuple[bool, str]:
        """
        Validate reward calculator parameters.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        errors = []
        
        # Validate throughput factor
        if not isinstance(self.throughput_factor, (int, float)):
            errors.append("Throughput factor must be numeric")
        elif self.throughput_factor < 0:
            errors.append("Throughput factor cannot be negative")
        elif self.throughput_factor > 100:
            errors.append("Throughput factor too high")
        
        # Validate handover penalty
        if not isinstance(self.handover_penalty, (int, float)):
            errors.append("Handover penalty must be numeric")
        elif self.handover_penalty < 0:
            errors.append("Handover penalty cannot be negative")
        elif self.handover_penalty > 1000:
            errors.append("Handover penalty too high")
        
        # Validate outage penalty
        if not isinstance(self.outage_penalty, (int, float)):
            errors.append("Outage penalty must be numeric")
        elif self.outage_penalty < 0:
            errors.append("Outage penalty cannot be negative")
        elif self.outage_penalty > 10000:
            errors.append("Outage penalty too high")
        
        # Validate SINR threshold
        if not isinstance(self.sinr_threshold, (int, float)):
            errors.append("SINR threshold must be numeric")
        elif self.sinr_threshold < -100 or self.sinr_threshold > 100:
            errors.append("SINR threshold outside reasonable range")
        
        if errors:
            return False, "; ".join(errors)
        else:
            return True, "Reward parameters valid"
    
    def get_reward_info(self) -> dict:
        """
        Get information about reward parameters.
        
        Returns:
            Dictionary with reward parameters
        """
        return {
            'throughput_factor': self.throughput_factor,
            'handover_penalty': self.handover_penalty,
            'outage_penalty': self.outage_penalty,
            'sinr_threshold': self.sinr_threshold,
            'episode_throughput': self.episode_throughput,
            'episode_handovers': self.episode_handovers
        } 