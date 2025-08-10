#!/usr/bin/env python3
"""
Distance Optimization Module for OAM 6G Systems

This module provides intelligent distance-aware optimization strategies
for OAM mode selection and system performance optimization.
"""

import numpy as np
from typing import Dict, Any, Tuple, List, Optional
from dataclasses import dataclass
import math


@dataclass
class DistanceThreshold:
    """Distance threshold configuration for mode optimization."""
    distance: float
    mode: int
    priority: float
    description: str


@dataclass
class DistanceOptimizationConfig:
    """Configuration for distance optimization strategies."""
    # Distance thresholds for mode selection
    near_threshold: float = 50.0      # meters
    medium_threshold: float = 150.0    # meters
    far_threshold: float = 300.0       # meters
    
    # Mode preferences by distance
    near_modes: List[int] = None       # Preferred modes for near distances
    medium_modes: List[int] = None     # Preferred modes for medium distances
    far_modes: List[int] = None        # Preferred modes for far distances
    
    # Optimization parameters
    distance_weight: float = 0.3       # Weight of distance in optimization
    throughput_weight: float = 0.5     # Weight of throughput in optimization
    stability_weight: float = 0.2      # Weight of stability in optimization
    
    # Adaptive parameters
    adaptive_threshold: bool = True     # Enable adaptive threshold adjustment
    learning_rate: float = 0.01        # Learning rate for adaptive thresholds
    
    def __post_init__(self):
        """Initialize default mode preferences if not provided."""
        if self.near_modes is None:
            self.near_modes = [1, 2, 3]  # Lower modes for near distances
        if self.medium_modes is None:
            self.medium_modes = [4, 5, 6]  # Medium modes for medium distances
        if self.far_modes is None:
            self.far_modes = [7, 8]  # Higher modes for far distances


class DistanceOptimizer:
    """
    Intelligent distance-aware optimization for OAM mode selection.
    
    This class provides various strategies for optimizing OAM mode selection
    based on distance, throughput, and system stability.
    """
    
    def __init__(self, config: Optional[DistanceOptimizationConfig] = None):
        """
        Initialize the distance optimizer.
        
        Args:
            config: Distance optimization configuration
        """
        self.config = config or DistanceOptimizationConfig()
        
        # Performance tracking
        self.distance_history = []
        self.throughput_history = []
        self.mode_history = []
        self.optimization_scores = []
        
        # Adaptive thresholds
        self.adaptive_near_threshold = self.config.near_threshold
        self.adaptive_medium_threshold = self.config.medium_threshold
        self.adaptive_far_threshold = self.config.far_threshold
        
        # Performance metrics
        self.total_optimizations = 0
        self.successful_optimizations = 0
        
    def get_distance_category(self, distance: float) -> str:
        """
        Categorize distance into near, medium, or far.
        
        Args:
            distance: Distance in meters
            
        Returns:
            Distance category string
        """
        if distance <= self.adaptive_near_threshold:
            return "near"
        elif distance <= self.adaptive_medium_threshold:
            return "medium"
        else:
            return "far"
    
    def get_optimal_mode_by_distance(self, distance: float, current_mode: int,
                                   available_modes: List[int]) -> int:
        """
        Get optimal mode based on distance using threshold-based strategy.
        
        Args:
            distance: Current distance in meters
            current_mode: Current OAM mode
            available_modes: List of available modes
            
        Returns:
            Optimal mode for the given distance
        """
        category = self.get_distance_category(distance)
        
        # Get preferred modes for this distance category
        if category == "near":
            preferred_modes = self.config.near_modes
        elif category == "medium":
            preferred_modes = self.config.medium_modes
        else:  # far
            preferred_modes = self.config.far_modes
        
        # Filter to available modes
        available_preferred = [m for m in preferred_modes if m in available_modes]
        
        if available_preferred:
            # Return the first available preferred mode
            return available_preferred[0]
        else:
            # Fallback to current mode if no preferred modes available
            return current_mode
    
    def calculate_distance_score(self, distance: float, target_distance: float) -> float:
        """
        Calculate distance optimization score.
        
        Args:
            distance: Current distance
            target_distance: Target optimal distance
            
        Returns:
            Distance score (0-1, higher is better)
        """
        distance_diff = abs(distance - target_distance)
        max_acceptable_diff = 100.0  # meters
        
        # Normalize to 0-1 range
        score = max(0.0, 1.0 - (distance_diff / max_acceptable_diff))
        return score
    
    def calculate_throughput_score(self, throughput: float, max_throughput: float) -> float:
        """
        Calculate throughput optimization score.
        
        Args:
            throughput: Current throughput
            max_throughput: Maximum possible throughput
            
        Returns:
            Throughput score (0-1, higher is better)
        """
        if max_throughput <= 0:
            return 0.0
        
        # Normalize to 0-1 range
        score = min(1.0, throughput / max_throughput)
        return score
    
    def calculate_stability_score(self, mode_changes: int, max_changes: int) -> float:
        """
        Calculate stability optimization score.
        
        Args:
            mode_changes: Number of mode changes
            max_changes: Maximum acceptable changes
            
        Returns:
            Stability score (0-1, higher is better)
        """
        if max_changes <= 0:
            return 1.0
        
        # Normalize to 0-1 range
        score = max(0.0, 1.0 - (mode_changes / max_changes))
        return score
    
    def optimize_mode_selection(self, distance: float, current_throughput: float,
                              current_mode: int, available_modes: List[int],
                              mode_changes: int = 0) -> Tuple[int, Dict[str, float]]:
        """
        Optimize mode selection using multi-criteria optimization.
        
        Args:
            distance: Current distance in meters
            current_throughput: Current throughput in bps
            current_mode: Current OAM mode
            available_modes: List of available modes
            mode_changes: Number of recent mode changes
            
        Returns:
            Tuple of (optimal_mode, optimization_scores)
        """
        self.total_optimizations += 1
        
        # Calculate individual scores
        distance_score = self.calculate_distance_score(distance, self.adaptive_medium_threshold)
        throughput_score = self.calculate_throughput_score(current_throughput, 1e9)  # 1 Gbps max
        stability_score = self.calculate_stability_score(mode_changes, 5)  # Max 5 changes
        
        # Calculate weighted optimization score
        optimization_score = (
            self.config.distance_weight * distance_score +
            self.config.throughput_weight * throughput_score +
            self.config.stability_weight * stability_score
        )
        
        # Get optimal mode using distance-based strategy
        optimal_mode = self.get_optimal_mode_by_distance(distance, current_mode, available_modes)
        
        # Determine if mode change is beneficial
        should_change = False
        if optimal_mode != current_mode:
            # Only change if optimization score is significantly better
            score_threshold = 0.7
            if optimization_score > score_threshold:
                should_change = True
        
        final_mode = optimal_mode if should_change else current_mode
        
        # Track optimization metrics
        scores = {
            'distance_score': distance_score,
            'throughput_score': throughput_score,
            'stability_score': stability_score,
            'optimization_score': optimization_score,
            'should_change': should_change
        }
        
        # Store optimization score for statistics
        self.optimization_scores.append(optimization_score)
        
        if should_change:
            self.successful_optimizations += 1
        
        # Update adaptive thresholds if enabled
        if self.config.adaptive_threshold:
            self._update_adaptive_thresholds(distance, optimization_score)
        
        return final_mode, scores
    
    def _update_adaptive_thresholds(self, distance: float, optimization_score: float):
        """
        Update adaptive thresholds based on performance.
        
        Args:
            distance: Current distance
            optimization_score: Current optimization score
        """
        # Update thresholds based on performance
        if optimization_score > 0.8:  # Good performance
            # Slightly expand the medium range
            self.adaptive_medium_threshold += self.config.learning_rate * 10
        elif optimization_score < 0.4:  # Poor performance
            # Contract the medium range
            self.adaptive_medium_threshold -= self.config.learning_rate * 10
        
        # Ensure thresholds remain reasonable
        self.adaptive_medium_threshold = np.clip(
            self.adaptive_medium_threshold, 
            50.0,  # Minimum
            400.0  # Maximum
        )
    
    def get_distance_optimization_stats(self) -> Dict[str, Any]:
        """
        Get distance optimization statistics.
        
        Returns:
            Dictionary containing optimization statistics
        """
        if self.total_optimizations == 0:
            return {
                'total_optimizations': 0,
                'success_rate': 0.0,
                'adaptive_thresholds': {
                    'near': self.adaptive_near_threshold,
                    'medium': self.adaptive_medium_threshold,
                    'far': self.adaptive_far_threshold
                }
            }
        
        success_rate = self.successful_optimizations / self.total_optimizations
        
        return {
            'total_optimizations': self.total_optimizations,
            'successful_optimizations': self.successful_optimizations,
            'success_rate': success_rate,
            'adaptive_thresholds': {
                'near': self.adaptive_near_threshold,
                'medium': self.adaptive_medium_threshold,
                'far': self.adaptive_far_threshold
            },
            'average_optimization_score': np.mean(self.optimization_scores) if self.optimization_scores else 0.0
        }
    
    def reset_stats(self):
        """Reset optimization statistics."""
        self.total_optimizations = 0
        self.successful_optimizations = 0
        self.optimization_scores.clear()
        self.distance_history.clear()
        self.throughput_history.clear()
        self.mode_history.clear()


class DistanceAwareRewardCalculator:
    """
    Distance-aware reward calculator that incorporates distance optimization.
    
    This class extends the standard reward calculation with distance-based
    optimization considerations.
    """
    
    def __init__(self, base_reward_calculator, distance_optimizer: DistanceOptimizer):
        """
        Initialize distance-aware reward calculator.
        
        Args:
            base_reward_calculator: Base reward calculator
            distance_optimizer: Distance optimizer instance
        """
        self.base_calculator = base_reward_calculator
        self.distance_optimizer = distance_optimizer
        
        # Distance-specific reward parameters
        self.distance_bonus_factor = 0.1
        self.distance_penalty_factor = 0.05
        
    def calculate_distance_aware_reward(self, throughput: float, sinr_dB: float,
                                      handover_occurred: bool, distance: float,
                                      current_mode: int, available_modes: List[int]) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate reward with distance optimization considerations.
        
        Args:
            throughput: Current throughput
            sinr_dB: Current SINR in dB
            handover_occurred: Whether a handover occurred
            distance: Current distance
            current_mode: Current OAM mode
            available_modes: Available modes
            
        Returns:
            Tuple of (reward, optimization_info)
        """
        # Get base reward
        base_reward = self.base_calculator.calculate_reward(throughput, sinr_dB, handover_occurred)
        
        # Get distance optimization recommendation
        optimal_mode, optimization_scores = self.distance_optimizer.optimize_mode_selection(
            distance, throughput, current_mode, available_modes
        )
        
        # Calculate distance-based reward adjustment
        distance_adjustment = 0.0
        
        if optimal_mode == current_mode:
            # Bonus for being at optimal mode
            distance_adjustment = self.distance_bonus_factor * optimization_scores['optimization_score']
        else:
            # Penalty for not being at optimal mode
            distance_adjustment = -self.distance_penalty_factor * (1.0 - optimization_scores['optimization_score'])
        
        # Apply distance adjustment
        final_reward = base_reward + distance_adjustment
        
        # Prepare optimization info
        optimization_info = {
            'base_reward': base_reward,
            'distance_adjustment': distance_adjustment,
            'final_reward': final_reward,
            'optimal_mode': optimal_mode,
            'optimization_scores': optimization_scores,
            'distance_category': self.distance_optimizer.get_distance_category(distance)
        }
        
        return final_reward, optimization_info 