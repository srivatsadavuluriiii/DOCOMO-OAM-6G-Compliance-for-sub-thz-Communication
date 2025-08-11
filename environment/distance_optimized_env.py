#!/usr/bin/env python3
"""
Distance-Optimized OAM Environment

This environment extends the base OAM environment with intelligent
distance-aware optimization for improved performance.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Tuple, List, Optional
import os
import sys

# Use centralized path management instead of sys.path.append
from utils.path_utils import ensure_project_root_in_path
ensure_project_root_in_path()

from .oam_env import OAM_Env
from .distance_optimizer import DistanceOptimizer, DistanceOptimizationConfig, DistanceAwareRewardCalculator
from simulator.channel_simulator import ChannelSimulator
from utils.exception_handler import safe_calculation


class DistanceOptimizedEnv(OAM_Env):
    """
    Distance-optimized OAM environment with intelligent mode selection.
    
    This environment extends the base OAM environment with distance-aware
    optimization strategies for improved performance and stability.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, simulator: Optional['SimulatorInterface'] = None):
        """
        Initialize the distance-optimized OAM environment.
        
        Args:
            config: Dictionary containing environment parameters
            simulator: Channel simulator instance (for dependency injection)
        """
        super().__init__(config, simulator)
        
        # Initialize distance optimization components
        self._init_distance_optimization(config)
        
        # Distance optimization tracking
        self.distance_optimization_history = []
        self.mode_change_history = []
        self.optimization_scores_history = []
        
        # Performance metrics
        self.total_distance_optimizations = 0
        self.successful_distance_optimizations = 0
        
    def _init_distance_optimization(self, config: Optional[Dict[str, Any]] = None):
        """Initialize distance optimization components."""
        # Create distance optimization configuration
        distance_config = DistanceOptimizationConfig()
        
        # Update with provided config if available
        if config and 'distance_optimization' in config:
            dist_opt_config = config['distance_optimization']
            
            # Update thresholds
            if 'distance_thresholds' in dist_opt_config:
                thresholds = dist_opt_config['distance_thresholds']
                distance_config.near_threshold = thresholds.get('near_threshold', distance_config.near_threshold)
                distance_config.medium_threshold = thresholds.get('medium_threshold', distance_config.medium_threshold)
                distance_config.far_threshold = thresholds.get('far_threshold', distance_config.far_threshold)
            
            # Update mode preferences
            if 'mode_preferences' in dist_opt_config:
                mode_prefs = dist_opt_config['mode_preferences']
                distance_config.near_modes = mode_prefs.get('near_modes', distance_config.near_modes)
                distance_config.medium_modes = mode_prefs.get('medium_modes', distance_config.medium_modes)
                distance_config.far_modes = mode_prefs.get('far_modes', distance_config.far_modes)
            
            # Update optimization weights
            if 'optimization_weights' in dist_opt_config:
                weights = dist_opt_config['optimization_weights']
                distance_config.distance_weight = weights.get('distance_weight', distance_config.distance_weight)
                distance_config.throughput_weight = weights.get('throughput_weight', distance_config.throughput_weight)
                distance_config.stability_weight = weights.get('stability_weight', distance_config.stability_weight)
            
            # Update adaptive parameters
            if 'adaptive_parameters' in dist_opt_config:
                adaptive = dist_opt_config['adaptive_parameters']
                distance_config.adaptive_threshold = adaptive.get('enabled', distance_config.adaptive_threshold)
                distance_config.learning_rate = adaptive.get('learning_rate', distance_config.learning_rate)
        
        # Initialize distance optimizer (optionally ML-based)
        use_ml = False
        model_path = None
        feature_list = None
        if config and 'distance_optimization' in config:
            ml_cfg = config['distance_optimization']
            use_ml = bool(ml_cfg.get('enabled_ml', False))
            model_path = ml_cfg.get('model_path')
            feature_list = ml_cfg.get('features')

        if use_ml:
            from .distance_optimizer import MLDistanceOptimizer
            self.distance_optimizer = MLDistanceOptimizer(
                distance_config,
                model_path=model_path,
                feature_list=feature_list,
                min_mode=self.min_mode,
                max_mode=self.max_mode,
            )
        else:
            self.distance_optimizer = DistanceOptimizer(distance_config)
        
        # Initialize distance-aware reward calculator
        self.distance_aware_reward_calculator = DistanceAwareRewardCalculator(
            self.reward_calculator, self.distance_optimizer
        )
        
        # Distance optimization parameters
        self.distance_optimization_enabled = True
        self.optimization_threshold = 0.7
        self.min_handover_interval = 5
        self.handover_hysteresis = 0.1
        
        # Update parameters from config
        if config and 'distance_optimization' in config:
            dist_opt_config = config['distance_optimization']
            
            if 'distance_reward' in dist_opt_config:
                reward_config = dist_opt_config['distance_reward']
                self.optimization_threshold = reward_config.get('optimization_threshold', self.optimization_threshold)
            
            if 'handover_optimization' in dist_opt_config:
                handover_config = dist_opt_config['handover_optimization']
                self.min_handover_interval = handover_config.get('min_handover_interval', self.min_handover_interval)
                self.handover_hysteresis = handover_config.get('handover_hysteresis', self.handover_hysteresis)

        # Analytics config
        self.analytics_enabled = bool(config.get('analytics', {}).get('enable', False)) if config else False
        self.analytics_backends = config.get('analytics', {}).get('backends', ["jsonl"]) if config else ["jsonl"]
        self.analytics_interval = int(config.get('analytics', {}).get('interval', 1)) if config else 1
        self._metrics_logger = None
        if self.analytics_enabled:
            from utils.visualization_unified import MetricsLogger
            log_dir = os.path.join('results', 'analytics')
            try:
                self._metrics_logger = MetricsLogger(log_dir, backends=self.analytics_backends, flush_interval=self.analytics_interval)
            except Exception:
                self._metrics_logger = None
    
    @safe_calculation("distance_optimization", fallback_value=1)
    def _get_distance_optimized_mode(self, distance: float, current_throughput: float,
                                   current_mode: int) -> Tuple[int, Dict[str, Any]]:
        """
        Get distance-optimized mode selection.
        
        Args:
            distance: Current distance in meters
            current_throughput: Current throughput in bps
            current_mode: Current OAM mode
            
        Returns:
            Tuple of (optimized_mode, optimization_info)
        """
        self.total_distance_optimizations += 1
        
        # Get available modes
        available_modes = list(range(self.min_mode, self.max_mode + 1))
        
        # Get mode changes in recent history
        recent_mode_changes = len([m for m in self.mode_change_history[-10:] if m])
        
        # Get distance-optimized mode
        # Provide SINR to ML optimizer if used
        sinr_db = getattr(self, 'current_sinr', 0.0)
        try:
            optimal_mode, optimization_scores = self.distance_optimizer.optimize_mode_selection(
                distance, current_throughput, current_mode, available_modes, recent_mode_changes, sinr_db=sinr_db
            )
        except TypeError:
            # Backward compatibility with base optimizer signature
            optimal_mode, optimization_scores = self.distance_optimizer.optimize_mode_selection(
                distance, current_throughput, current_mode, available_modes, recent_mode_changes
            )
        
        # Determine if mode change should occur
        should_change = False
        if optimal_mode != current_mode:
            # Check optimization threshold
            if optimization_scores['optimization_score'] > self.optimization_threshold:
                # Check handover interval constraint
                if len(self.mode_change_history) == 0 or not any(self.mode_change_history[-self.min_handover_interval:]):
                    should_change = True
        
        final_mode = optimal_mode if should_change else current_mode
        
        # Track optimization
        optimization_info = {
            'optimal_mode': optimal_mode,
            'final_mode': final_mode,
            'should_change': should_change,
            'optimization_scores': optimization_scores,
            'distance_category': self.distance_optimizer.get_distance_category(distance),
            'recent_mode_changes': recent_mode_changes
        }
        
        if should_change:
            self.successful_distance_optimizations += 1
        
        return final_mode, optimization_info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment with distance optimization.
        
        Args:
            action: Action to take (0: STAY, 1: UP, 2: DOWN)
            
        Returns:
            Tuple of (next_state, reward, done, truncated, info)
        """
        # Store previous state for optimization
        previous_mode = self.current_mode
        previous_distance = np.linalg.norm(self.position)
        
        # Execute base step
        next_state, reward, done, truncated, info = super().step(action)
        
        # Get current distance and throughput
        current_distance = np.linalg.norm(self.position)
        current_throughput = info.get('throughput', 0.0)
        
        # Apply distance optimization if enabled
        if self.distance_optimization_enabled:
            optimized_mode, optimization_info = self._get_distance_optimized_mode(
                current_distance, current_throughput, self.current_mode
            )
            
            # Apply mode change if recommended
            if optimized_mode != self.current_mode:
                self.current_mode = optimized_mode
                info['distance_optimization_mode_change'] = True
                info['previous_mode'] = previous_mode
                info['optimized_mode'] = optimized_mode
            else:
                info['distance_optimization_mode_change'] = False
            
            # Update reward with distance optimization
            distance_aware_reward, distance_optimization_info = self.distance_aware_reward_calculator.calculate_distance_aware_reward(
                current_throughput, self.current_sinr, info.get('handovers', 0) > 0,
                current_distance, self.current_mode, list(range(self.min_mode, self.max_mode + 1))
            )
            
            # Use distance-aware reward
            reward = distance_aware_reward
            
            # Add optimization info to info dictionary
            info.update({
                'distance_optimization_info': distance_optimization_info,
                'optimization_scores': optimization_info['optimization_scores'],
                'distance_category': optimization_info['distance_category']
            })
        
        # Track distance optimization history
        self.distance_optimization_history.append({
            'distance': current_distance,
            'mode': self.current_mode,
            'throughput': current_throughput,
            'optimization_scores': info.get('optimization_scores', {}),
            'mode_change': info.get('distance_optimization_mode_change', False)
        })
        
        # Track mode changes
        mode_changed = (self.current_mode != previous_mode)
        self.mode_change_history.append(mode_changed)
        
        # Keep history within reasonable bounds
        if len(self.distance_optimization_history) > 1000:
            self.distance_optimization_history = self.distance_optimization_history[-500:]
        if len(self.mode_change_history) > 100:
            self.mode_change_history = self.mode_change_history[-50:]
        
        # Analytics logging
        if self._metrics_logger is not None and (self.steps % self.analytics_interval == 0):
            try:
                self._metrics_logger.log_scalar('sinr_db', float(self.current_sinr), self.steps)
                self._metrics_logger.log_scalar('throughput', float(current_throughput), self.steps)
                self._metrics_logger.log_scalar('mode', float(self.current_mode), self.steps)
                self._metrics_logger.log_scalar('handovers', float(self.episode_handovers), self.steps)
            except Exception:
                pass

        return next_state, reward, done, truncated, info
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment with distance optimization tracking.
        
        Args:
            seed: Random seed
            options: Additional options
            
        Returns:
            Tuple of (initial state, info dictionary)
        """
        # Reset distance optimization tracking
        self.distance_optimization_history.clear()
        self.mode_change_history.clear()
        self.optimization_scores_history.clear()
        
        # Reset distance optimizer stats
        self.distance_optimizer.reset_stats()
        
        # Reset performance metrics
        self.total_distance_optimizations = 0
        self.successful_distance_optimizations = 0
        
        # Call parent reset
        state, info = super().reset(seed=seed, options=options)
        # Emit episode summary (previous episode), if logger exists and steps>0
        if self._metrics_logger is not None and self.steps == 0:
            try:
                stats = self.reward_calculator.get_episode_stats()
                summary = {
                    'episode_reward': float(stats.get('episode_reward', 0.0)),
                    'episode_throughput': float(stats.get('episode_throughput', 0.0)),
                    'episode_handovers': int(stats.get('episode_handovers', 0)),
                }
                # Use a simple episode counter proxy
                self._episode_counter = getattr(self, '_episode_counter', 0) + 1
                self._metrics_logger.log_episode_summary(summary, self._episode_counter)
            except Exception:
                pass
        return state, info
    
    def get_distance_optimization_stats(self) -> Dict[str, Any]:
        """
        Get distance optimization statistics.
        
        Returns:
            Dictionary containing distance optimization statistics
        """
        base_stats = self.distance_optimizer.get_distance_optimization_stats()
        
        # Add environment-specific stats
        stats = {
            **base_stats,
            'total_distance_optimizations': self.total_distance_optimizations,
            'successful_distance_optimizations': self.successful_distance_optimizations,
            'optimization_success_rate': (
                self.successful_distance_optimizations / max(self.total_distance_optimizations, 1)
            ),
            'average_distance': np.mean([h['distance'] for h in self.distance_optimization_history]) if self.distance_optimization_history else 0.0,
            'average_throughput': np.mean([h['throughput'] for h in self.distance_optimization_history]) if self.distance_optimization_history else 0.0,
            'mode_change_frequency': np.mean(self.mode_change_history) if self.mode_change_history else 0.0
        }
        
        return stats
    
    def get_distance_category_performance(self) -> Dict[str, Dict[str, float]]:
        """
        Get performance statistics by distance category.
        
        Returns:
            Dictionary containing performance by distance category
        """
        if not self.distance_optimization_history:
            return {}
        
        # Group by distance category
        category_data = {'near': [], 'medium': [], 'far': []}
        
        for entry in self.distance_optimization_history:
            distance = entry['distance']
            category = self.distance_optimizer.get_distance_category(distance)
            if category in category_data:
                category_data[category].append(entry)
        
        # Calculate statistics for each category
        performance_stats = {}
        
        for category, data in category_data.items():
            if data:
                performance_stats[category] = {
                    'count': len(data),
                    'avg_throughput': np.mean([d['throughput'] for d in data]),
                    'avg_distance': np.mean([d['distance'] for d in data]),
                    'avg_optimization_score': np.mean([d['optimization_scores'].get('optimization_score', 0) for d in data]),
                    'mode_change_rate': np.mean([d['mode_change'] for d in data])
                }
        
        return performance_stats
    
    def set_distance_optimization_enabled(self, enabled: bool):
        """
        Enable or disable distance optimization.
        
        Args:
            enabled: Whether to enable distance optimization
        """
        self.distance_optimization_enabled = enabled
    
    def set_optimization_threshold(self, threshold: float):
        """
        Set the optimization threshold for mode changes.
        
        Args:
            threshold: Optimization threshold (0-1)
        """
        self.optimization_threshold = np.clip(threshold, 0.0, 1.0) 