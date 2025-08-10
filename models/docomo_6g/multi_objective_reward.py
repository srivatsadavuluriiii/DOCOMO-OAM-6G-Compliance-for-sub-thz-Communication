#!/usr/bin/env python3
"""
DOCOMO Multi-Objective Reward System
Implements DOCOMO 6G KPI priorities with physics-informed rewards
Optimizes throughput, latency, energy, reliability, and mobility simultaneously
"""

import numpy as np
import math
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from enum import Enum
import warnings

class RewardObjective(Enum):
    """DOCOMO reward objectives"""
    THROUGHPUT = "throughput"        # Peak throughput maximization
    LATENCY = "latency"             # Ultra-low latency minimization
    ENERGY = "energy"               # Energy efficiency improvement
    RELIABILITY = "reliability"     # Ultra-high reliability
    MOBILITY = "mobility"           # High-speed mobility support
    SPECTRUM = "spectrum"           # Spectrum efficiency
    STABILITY = "stability"         # System stability
    HANDOVER = "handover"          # Handover optimization

@dataclass
class DOCOMOObjectiveWeights:
    """DOCOMO objective weights (must sum to 1.0)"""
    throughput: float = 0.25        # 25% - Peak throughput priority
    latency: float = 0.25           # 25% - Ultra-low latency priority
    energy: float = 0.20            # 20% - Energy efficiency
    reliability: float = 0.15       # 15% - Ultra-high reliability  
    mobility: float = 0.10          # 10% - High-speed mobility
    spectrum: float = 0.05          # 5% - Spectrum efficiency
    
    def __post_init__(self):
        """Validate weights sum to 1.0"""
        total = (self.throughput + self.latency + self.energy + 
                self.reliability + self.mobility + self.spectrum)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Objective weights must sum to 1.0, got {total}")

@dataclass
class RewardComponents:
    """Individual reward components"""
    throughput_reward: float = 0.0
    latency_reward: float = 0.0
    energy_reward: float = 0.0
    reliability_reward: float = 0.0
    mobility_reward: float = 0.0
    spectrum_reward: float = 0.0
    stability_reward: float = 0.0
    handover_penalty: float = 0.0
    physics_bonus: float = 0.0
    total_reward: float = 0.0

class MultiObjectiveReward:
    """
    DOCOMO-aligned multi-objective reward system with physics constraints
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize multi-objective reward system
        
        Args:
            config: DOCOMO configuration dictionary
        """
        self.config = config
        
        # Load DOCOMO KPI targets
        kpi_targets = config.get('docomo_6g_system', {}).get('kpi_targets', {})
        self.target_throughput_gbps = kpi_targets.get('user_data_rate_gbps', 100.0)
        self.target_latency_ms = kpi_targets.get('latency_ms', 0.1)
        self.target_reliability = kpi_targets.get('reliability', 0.9999999)
        self.target_mobility_kmh = kpi_targets.get('mobility_kmh', 500.0)
        self.target_energy_improvement = kpi_targets.get('energy_efficiency_improvement', 100.0)
        self.target_spectrum_improvement = kpi_targets.get('spectrum_efficiency_improvement', 10.0)
        
        # Load objective weights
        rl_config = config.get('docomo_6g_system', {}).get('reinforcement_learning', {})
        objectives = rl_config.get('objectives', {})
        
        self.weights = DOCOMOObjectiveWeights(
            throughput=objectives.get('throughput_weight', 0.25),
            latency=objectives.get('latency_weight', 0.25),
            energy=objectives.get('energy_weight', 0.20),
            reliability=objectives.get('reliability_weight', 0.15),
            mobility=objectives.get('mobility_weight', 0.10),
            spectrum=objectives.get('spectrum_weight', 0.05)
        )
        
        # Reward scaling parameters (calibrated for stability-first learning)
        self.reward_scale = 3.0   # Global scale reduced to avoid runaway magnitudes
        self.penalty_scale = 1.5  # Penalties closer to rewards for balance
        
        # Physics-informed parameters
        self.physics_bonus_enabled = True
        self.atmospheric_penalty_enabled = True
        self.oam_crosstalk_penalty_enabled = True
        
        # Adaptive parameters
        self.adaptive_weights = False  # Enable adaptive weighting
        self.performance_history = []
        
        # Tracking for prolonged low-band use under good SINR
        self.low_band_good_sinr_streak = 0
        self.low_band_good_sinr_threshold_db = 10.0
        
        # Baseline values for normalization
        self.baseline_energy_w = 1.0        # Baseline energy consumption
        self.baseline_spectrum_efficiency = 1.0  # Baseline spectrum efficiency
        self.baseline_handover_rate = 0.1   # Baseline handover rate
        
    def calculate(self, 
                 state: np.ndarray,
                 action: int,
                 next_state: np.ndarray,
                 info: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        """
        Calculate multi-objective reward
        
        Args:
            state: Current state
            action: Action taken
            next_state: Next state after action
            info: Additional information dictionary
            
        Returns:
            Tuple of (total_reward, reward_breakdown)
        """
        # Extract state information
        sinr_db = next_state[0] if len(next_state) > 0 else 0.0
        throughput_gbps = info.get('throughput_gbps', 0.0)
        latency_ms = info.get('latency_ms', 1.0)
        distance_m = next_state[3] if len(next_state) > 3 else 100.0
        velocity_kmh = np.sqrt(next_state[4]**2 + next_state[5]**2 + next_state[6]**2) * 3.6 if len(next_state) > 6 else 0.0
        energy_w = info.get('energy_consumption_w', 1.0)
        reliability_score = info.get('reliability_score', 1.0)
        handover_occurred = info.get('handover_occurred', False)
        
        # Store action info for reward calculation
        self._current_action_info = info
        # Ensure sinr is available to downstream logic
        try:
            self._current_action_info['sinr_db'] = float(sinr_db)
        except Exception:
            pass
        
        # Calculate individual reward components
        components = RewardComponents()
        
        # 1. Throughput reward (DOCOMO: 100 Gbps user experience)
        components.throughput_reward = self._calculate_throughput_reward(
            throughput_gbps, sinr_db, distance_m, info
        )
        
        # 2. Latency reward (DOCOMO: 0.1 ms target)
        components.latency_reward = self._calculate_latency_reward(
            latency_ms, action, info
        )
        
        # 3. Energy efficiency reward (DOCOMO: 100x improvement)
        components.energy_reward = self._calculate_energy_reward(
            energy_w, throughput_gbps, info
        )
        
        # 4. Reliability reward (DOCOMO: 99.99999% target)
        components.reliability_reward = self._calculate_reliability_reward(
            reliability_score, sinr_db, info
        )
        
        # 5. Mobility reward (DOCOMO: 500 km/h support)
        components.mobility_reward = self._calculate_mobility_reward(
            velocity_kmh, throughput_gbps, latency_ms, info
        )
        
        # 6. Spectrum efficiency reward (DOCOMO: 10x improvement)
        components.spectrum_reward = self._calculate_spectrum_reward(
            throughput_gbps, info
        )
        
        # 7. System stability reward
        components.stability_reward = self._calculate_stability_reward(
            state, next_state, info
        )
        
        # 8. Handover penalty (increased)
        components.handover_penalty = self._calculate_handover_penalty(
            handover_occurred, velocity_kmh, info
        )
        
        # 9. Physics-informed bonus
        components.physics_bonus = self._calculate_physics_bonus(
            state, action, next_state, info
        )
        
        # Calculate weighted total reward
        components.total_reward = self._calculate_total_reward(components)
        
        # Apply constraints and clipping
        components.total_reward = self._apply_constraints(
            components.total_reward, state, next_state, info
        )
        
        # Update performance history
        self.performance_history.append({
            'components': components,
            'throughput_gbps': throughput_gbps,
            'latency_ms': latency_ms,
            'mobility_kmh': velocity_kmh
        })
        
        # Keep history bounded
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-500:]
        
        # Return reward and breakdown
        reward_breakdown = {
            'throughput_reward': components.throughput_reward,
            'latency_reward': components.latency_reward,
            'energy_reward': components.energy_reward,
            'reliability_reward': components.reliability_reward,
            'mobility_reward': components.mobility_reward,
            'spectrum_reward': components.spectrum_reward,
            'stability_reward': components.stability_reward,
            'handover_penalty': components.handover_penalty,
            'physics_bonus': components.physics_bonus,
            'total_reward': components.total_reward,
            'docomo_compliance': self._calculate_docomo_compliance(components, info)
        }
        
        return components.total_reward, reward_breakdown
    
    def _calculate_throughput_reward(self, 
                                   throughput_gbps: float, 
                                   sinr_db: float,
                                   distance_m: float,
                                   info: Dict[str, Any] = None) -> float:
        """Calculate throughput reward with DOCOMO 100 Gbps target and high-frequency band bonuses"""
        if throughput_gbps <= 0:
            return -1.0  # Penalty for zero throughput
        
        # Base reward: normalized to DOCOMO target with enhanced scaling
        base_reward = min(throughput_gbps / self.target_throughput_gbps, 3.0)  # Increased cap to 3x target
        
        # Distance-aware bonus (closer distances can achieve higher throughput)
        distance_bonus = max(0.0, (500.0 - distance_m) / 500.0) * 0.3  # Increased bonus
        
        # SINR bonus with enhanced scaling (good channel conditions)
        sinr_bonus = max(0.0, (sinr_db - 5.0) / 25.0) * 0.5  # Enhanced SINR bonus
        
        # High-frequency band bonus - encourage Sub-THz and THz usage (softened)
        current_band = info.get('current_band', 'mmwave_28') if info else 'mmwave_28'
        high_freq_bands = ['sub_thz_100', 'sub_thz_140', 'sub_thz_220', 'sub_thz_300', 'thz_600']
        if current_band in high_freq_bands:
            band_multipliers = {
                'sub_thz_100': 1.20,  # +0.10
                'sub_thz_140': 1.30,  # +0.10
                'sub_thz_220': 1.40,  # +0.10  
                'sub_thz_300': 1.50,  # +0.10
                'thz_600': 1.60       # +0.10 for THz
            }
            high_freq_bonus = band_multipliers.get(current_band, 1.0) - 1.0
        else:
            high_freq_bonus = 0.0
        
        # Non-linear scaling for exceptional throughput
        if throughput_gbps > self.target_throughput_gbps:
            excellence_bonus = math.log(throughput_gbps / self.target_throughput_gbps) * 0.8  # Increased bonus
        else:
            excellence_bonus = 0.0
        
        # Apply throughput scaling with high-frequency emphasis
        total_reward = (base_reward + distance_bonus + sinr_bonus + excellence_bonus) * (1.0 + high_freq_bonus)
        
        return min(total_reward, 3.0)  # Softer cap
    
    def _calculate_latency_reward(self, 
                                latency_ms: float, 
                                action: int,
                                info: Dict[str, Any]) -> float:
        """Calculate latency reward with DOCOMO 0.1 ms target"""
        if latency_ms <= 0:
            return -1.0  # Invalid latency
        
        # Exponential reward for meeting/exceeding latency target
        if latency_ms <= self.target_latency_ms:
            # Bonus for exceeding target
            reward = 1.0 + (self.target_latency_ms - latency_ms) / self.target_latency_ms
        else:
            # Exponential penalty for exceeding target
            excess_factor = latency_ms / self.target_latency_ms
            reward = 1.0 / excess_factor  # Exponential decay
        
        # Action-specific latency impacts
        handover_latency_penalty = 0.0
        if action in [3, 4]:  # Band switching actions
            handover_latency_penalty = -0.2  # Switching adds latency
        
        # Beam tracking latency bonus
        beam_tracking_active = info.get('beam_tracking_active', False)
        if beam_tracking_active:
            reward += 0.1  # Bonus for predictive tracking
        
        total_reward = reward + handover_latency_penalty
        
        return max(total_reward, -2.0)  # Floor at -2.0
    
    def _calculate_energy_reward(self,
                               energy_w: float,
                               throughput_gbps: float,
                               info: Dict[str, Any]) -> float:
        """Calculate energy efficiency reward (DOCOMO: 100x improvement)"""
        if energy_w <= 0 or throughput_gbps <= 0:
            return -1.0
        
        # Energy efficiency: bits per joule
        energy_efficiency = throughput_gbps / energy_w  # Gbps per Watt
        baseline_efficiency = self.target_throughput_gbps / self.baseline_energy_w
        
        # Normalized efficiency (target is 100x improvement)
        efficiency_ratio = energy_efficiency / baseline_efficiency
        
        # Logarithmic reward for efficiency improvements
        if efficiency_ratio > 1.0:
            reward = math.log(efficiency_ratio) / math.log(self.target_energy_improvement)
        else:
            reward = -1.0 + efficiency_ratio  # Linear penalty below baseline
        
        # Bonus for using optimal frequency bands
        optimal_band = info.get('using_optimal_band', False)
        if optimal_band:
            reward += 0.2
        
        # Penalty for excessive power
        if energy_w > 5.0 * self.baseline_energy_w:  # 5x baseline is excessive
            reward -= 0.5
        
        return max(min(reward, 2.0), -2.0)  # Clip to [-2, 2]
    
    def _calculate_reliability_reward(self,
                                    reliability_score: float,
                                    sinr_db: float,
                                    info: Dict[str, Any]) -> float:
        """Calculate reliability reward (DOCOMO: 99.99999% target)"""
        # Base reliability reward
        reliability_ratio = reliability_score / self.target_reliability
        
        if reliability_ratio >= 1.0:
            reward = 1.0  # Full reward for meeting target
        else:
            # Steep penalty for missing reliability target
            reward = (reliability_ratio - 0.99) / 0.01  # Scale 0.99-1.0 to 0-1
            reward = max(reward, -2.0)  # Floor at -2.0
        
        # SINR contribution to reliability
        if sinr_db > 15.0:  # Good SINR supports high reliability
            reward += 0.3
        elif sinr_db < 5.0:  # Poor SINR hurts reliability
            reward -= 0.5
        
        # Error rate bonus/penalty
        error_rate = info.get('error_rate', 0.0)
        if error_rate < 1e-7:  # Better than target
            reward += 0.2
        elif error_rate > 1e-5:  # Much worse than target
            reward -= 1.0
        
        # Link stability bonus
        link_stable = info.get('link_stable', True)
        if not link_stable:
            reward -= 0.5
        
        return max(min(reward, 2.0), -3.0)  # Clip to [-3, 2]
    
    def _calculate_mobility_reward(self,
                                 velocity_kmh: float,
                                 throughput_gbps: float,
                                 latency_ms: float,
                                 info: Dict[str, Any]) -> float:
        """Calculate mobility reward (DOCOMO: 500 km/h target)"""
        # Base mobility support reward
        mobility_ratio = min(velocity_kmh / self.target_mobility_kmh, 1.0)
        base_reward = mobility_ratio * 1.0
        
        # Performance at high mobility bonus
        if velocity_kmh > 100.0:  # High mobility scenario
            # Throughput maintenance bonus
            if throughput_gbps > 0.5 * self.target_throughput_gbps:
                base_reward += 0.5
            
            # Latency maintenance bonus  
            if latency_ms < 2.0 * self.target_latency_ms:
                base_reward += 0.3
            
            # Handover efficiency bonus
            handover_success_rate = info.get('handover_success_rate', 1.0)
            base_reward += handover_success_rate * 0.2
        
        # Predictive tracking bonus
        prediction_accuracy = info.get('mobility_prediction_accuracy', 0.0)
        base_reward += prediction_accuracy * 0.3
        
        # Doppler compensation bonus
        doppler_compensated = info.get('doppler_compensated', False)
        if doppler_compensated and velocity_kmh > 50.0:
            base_reward += 0.2
        
        # Extreme mobility bonus (above 350 km/h)
        if velocity_kmh > 350.0:
            extreme_bonus = min((velocity_kmh - 350.0) / 150.0, 1.0) * 0.5
            base_reward += extreme_bonus
        
        return min(base_reward, 2.0)  # Cap at 2.0
    
    def _calculate_spectrum_reward(self,
                                 throughput_gbps: float,
                                 info: Dict[str, Any]) -> float:
        """Calculate spectrum efficiency reward (DOCOMO: 10x improvement)"""
        bandwidth_mhz = info.get('bandwidth_mhz', 100.0)
        
        if bandwidth_mhz <= 0:
            return -1.0
        
        # Spectrum efficiency: bps/Hz
        spectrum_efficiency = (throughput_gbps * 1e9) / (bandwidth_mhz * 1e6)  # bps/Hz
        
        # Baseline spectrum efficiency (typical 4G/5G)
        baseline_efficiency = self.baseline_spectrum_efficiency  # ~1 bps/Hz
        
        # Efficiency improvement ratio
        improvement_ratio = spectrum_efficiency / baseline_efficiency
        
        # Logarithmic reward
        if improvement_ratio > 1.0:
            reward = math.log(improvement_ratio) / math.log(self.target_spectrum_improvement)
        else:
            reward = improvement_ratio - 1.0  # Linear penalty below baseline
        
        # Bonus for using high-efficiency techniques
        oam_efficiency_bonus = info.get('oam_mode_efficiency', 0.0) * 0.2
        multi_band_bonus = info.get('multi_band_active', False) * 0.1
        
        total_reward = reward + oam_efficiency_bonus + multi_band_bonus
        
        return max(min(total_reward, 1.5), -1.5)  # Clip to [-1.5, 1.5]
    
    def _calculate_stability_reward(self,
                                  state: np.ndarray,
                                  next_state: np.ndarray,
                                  info: Dict[str, Any]) -> float:
        """Calculate system stability reward"""
        # SINR stability
        if len(state) > 0 and len(next_state) > 0:
            sinr_change = abs(next_state[0] - state[0])
            sinr_stability = max(0.0, 1.0 - sinr_change / 10.0)  # Penalty for >10 dB changes
        else:
            sinr_stability = 0.0
        
        # Throughput stability
        current_throughput = info.get('throughput_gbps', 0.0)
        prev_throughput = info.get('prev_throughput_gbps', current_throughput)
        
        if prev_throughput > 0:
            throughput_change = abs(current_throughput - prev_throughput) / prev_throughput
            throughput_stability = max(0.0, 1.0 - throughput_change / 0.2)  # Penalty for >20% changes
        else:
            throughput_stability = 0.0
        
        # Mode switching stability (penalty for frequent switching)
        mode_switches = info.get('recent_mode_switches', 0)
        switching_penalty = min(mode_switches * 0.2, 1.0)  # Stronger penalty
        
        # Overall stability score
        stability_score = (sinr_stability + throughput_stability) / 2.0 - switching_penalty
        
        return max(min(stability_score, 1.0), -1.0)  # Clip to [-1, 1]
    
    def _calculate_handover_penalty(self,
                                  handover_occurred: bool,
                                  velocity_kmh: float,
                                  info: Dict[str, Any]) -> float:
        """Calculate handover penalty with velocity consideration"""
        if not handover_occurred:
            return 0.0
        
        # Base handover penalty
        base_penalty = -1.0
        
        # Velocity-adjusted penalty (higher speeds may need more handovers)
        if velocity_kmh > 50.0:
            velocity_adjustment = (velocity_kmh - 50.0) / 450.0  # Scale 50-500 to 0-1
            base_penalty *= (1.0 - velocity_adjustment * 0.5)  # Reduce penalty at high speeds
        
        # Handover success bonus
        handover_success = info.get('handover_successful', True)
        if not handover_success:
            base_penalty *= 2.0  # Double penalty for failed handovers
        
        # Handover timing penalty (premature or late handovers)
        handover_timing_score = info.get('handover_timing_score', 1.0)  # 1.0 = optimal
        base_penalty *= (2.0 - handover_timing_score)  # Penalty for suboptimal timing
        
        return max(base_penalty, -2.0)  # Cap penalty
    
    def _calculate_physics_bonus(self,
                               state: np.ndarray,
                               action: int,
                               next_state: np.ndarray,
                               info: Dict[str, Any]) -> float:
        """Calculate physics-informed bonus/penalty"""
        if not self.physics_bonus_enabled:
            return 0.0
        
        total_bonus = 0.0
        
        # OAM mode selection physics bonus
        distance_m = next_state[3] if len(next_state) > 3 else 100.0
        current_mode = info.get('current_oam_mode', 1)
        
        # Distance-mode matching bonus
        if distance_m < 50.0 and current_mode <= 3:  # Low modes for short distance
            total_bonus += 0.2
        elif 50.0 <= distance_m <= 150.0 and 3 <= current_mode <= 6:  # Mid modes for medium distance
            total_bonus += 0.2
        elif distance_m > 150.0 and current_mode >= 6:  # High modes for long distance
            total_bonus += 0.2
        
        # Atmospheric window bonus
        frequency_ghz = info.get('frequency_ghz', 28.0)
        in_atmospheric_window = info.get('in_atmospheric_window', False)
        if in_atmospheric_window:
            total_bonus += 0.1
        
        # Beam alignment bonus
        beam_alignment_error = info.get('beam_alignment_error_deg', 0.0)
        if beam_alignment_error < 0.1:  # Very good alignment
            total_bonus += 0.15
        elif beam_alignment_error > 2.0:  # Poor alignment
            total_bonus -= 0.3
        
        # Crosstalk minimization bonus
        crosstalk_db = info.get('oam_crosstalk_db', 0.0)
        if crosstalk_db < -20.0:  # Good mode isolation
            total_bonus += 0.1
        elif crosstalk_db > -10.0:  # Poor isolation
            total_bonus -= 0.2
        
        return max(min(total_bonus, 0.5), -0.5)  # Clip to [-0.5, 0.5]
    
    def _calculate_total_reward(self, components: RewardComponents) -> float:
        """Calculate weighted total reward with enhanced high-frequency band incentives"""
        # Base weighted reward
        total_reward = (
            self.weights.throughput * components.throughput_reward +
            self.weights.latency * components.latency_reward +
            self.weights.energy * components.energy_reward +
            self.weights.reliability * components.reliability_reward +
            self.weights.mobility * components.mobility_reward +
            self.weights.spectrum * components.spectrum_reward +
            0.1 * components.stability_reward +  # Additional stability weight
            components.handover_penalty +
            components.physics_bonus
        )
        
        # Get current system info
        action_info = getattr(self, '_current_action_info', {})
        current_band = action_info.get('current_band', 'mmwave_28')
        throughput_gbps = action_info.get('throughput_gbps', 0.0)
        
        # Frequency incentives (moderate additive, applied only when delivering throughput)
        freq_bands_priority = {
            'thz_600': 1.0,
            'sub_thz_300': 0.9,
            'sub_thz_220': 0.7,
            'sub_thz_140': 0.5,
            'sub_thz_100': 0.3,
            'mmwave_60': 0.1,
            'mmwave_39': 0.0,
            'mmwave_28': 0.0,
            'sub_6ghz': -0.3
        }
        frequency_bonus = freq_bands_priority.get(current_band, 0.0)
        # Apply bonus only if actual throughput exceeds a minimal threshold to avoid blind bias
        if throughput_gbps < 5.0:
            frequency_bonus = min(frequency_bonus, 0.0)
        
        # Stronger compliance bonus shaping to push >100 Gbps consistently
        compliance_bonus = 0.0
        if throughput_gbps >= 100.0:
            # Scale faster above target but cap contribution
            compliance_bonus = min(3.0, 2.0 + 0.02 * (throughput_gbps - 100.0))
        elif throughput_gbps >= 50.0:
            compliance_bonus = 0.6 + 0.6 * (throughput_gbps - 50.0) / 50.0
        elif throughput_gbps >= 20.0:
            compliance_bonus = 0.2 * (throughput_gbps / 100.0)
        
        # Band switching action bonuses (capped for per-step consistency)
        band_switch_bonus = 0.0
        if action_info.get('high_freq_bonus', 0.0) > 0:
            band_switch_bonus = min(action_info['high_freq_bonus'] * 0.5, 0.5)
            
        # Beam optimization bonus 
        beam_bonus = 0.0
        if action_info.get('beam_optimization_bonus', 0.0) > 0:
            beam_bonus = action_info['beam_optimization_bonus'] * 1.0  # Increased
            
        # Prediction accuracy bonus
        prediction_bonus = 0.0
        if action_info.get('prediction_bonus', 0.0) > 0:
            prediction_bonus = action_info['prediction_bonus'] * 0.5
        
        # Apply all bonuses
        # Penalty for prolonged low-band use under good SINR
        low_band_penalty = 0.0
        try:
            sinr_db_val = float(action_info.get('sinr_db', 0.0))
            is_low_band = current_band in ['sub_6ghz', 'mmwave_28', 'mmwave_39']
            good_sinr = sinr_db_val > self.low_band_good_sinr_threshold_db
            if is_low_band and good_sinr and throughput_gbps < 80.0:
                # Increase streak and apply escalating penalty (capped)
                self.low_band_good_sinr_streak = min(self.low_band_good_sinr_streak + 1, 50)
                scale = min(0.02 * self.low_band_good_sinr_streak, 0.6)  # up to -0.6 pre-scale
                low_band_penalty = -scale
            else:
                # Decay streak when condition not met
                self.low_band_good_sinr_streak = max(self.low_band_good_sinr_streak - 1, 0)
        except Exception:
            # Conservative fallbacks
            if current_band == 'sub_6ghz':
                low_band_penalty = -0.05
        
        # Small per-step penalty for mmWave usage when SINR is already good
        mmwave_penalty = 0.0
        try:
            if current_band in ['mmwave_28', 'mmwave_39', 'mmwave_60'] and float(action_info.get('sinr_db', 0.0)) > 10.0:
                mmwave_penalty = -0.05
        except Exception:
            pass

        enhanced_reward = (
            total_reward + 
            frequency_bonus + 
            compliance_bonus + 
            band_switch_bonus + 
            beam_bonus + 
            prediction_bonus +
            low_band_penalty +
            mmwave_penalty
        )
        
        # Apply calibrated global scale
        return enhanced_reward * self.reward_scale
    
    def _apply_constraints(self,
                         reward: float,
                         state: np.ndarray,
                         next_state: np.ndarray,
                         info: Dict[str, Any]) -> float:
        """Apply final constraints and clipping"""
        # Clip to a tighter per-step range to avoid domination
        reward = max(min(reward, 10.0), -10.0)
        
        # Terminal state penalties
        if info.get('terminal_failure', False):
            reward = -25.0  # Large penalty for system failure
        
        # Physics violation penalties
        if info.get('physics_violation', False):
            reward -= 10.0
        
        return reward
    
    def _calculate_docomo_compliance(self,
                                   components: RewardComponents,
                                   info: Dict[str, Any]) -> float:
        """Calculate overall DOCOMO compliance score"""
        # Extract key metrics
        throughput_gbps = info.get('throughput_gbps', 0.0)
        latency_ms = info.get('latency_ms', 1.0)
        reliability = info.get('reliability_score', 1.0)
        mobility_kmh = info.get('mobility_kmh', 0.0)
        
        # Individual compliance scores
        throughput_compliance = min(throughput_gbps / self.target_throughput_gbps, 1.0)
        latency_compliance = min(self.target_latency_ms / latency_ms, 1.0) if latency_ms > 0 else 0.0
        reliability_compliance = reliability / self.target_reliability
        mobility_compliance = min(mobility_kmh / self.target_mobility_kmh, 1.0)
        
        # Weighted compliance score
        overall_compliance = (
            self.weights.throughput * throughput_compliance +
            self.weights.latency * latency_compliance +
            self.weights.reliability * reliability_compliance +
            self.weights.mobility * mobility_compliance +
            0.2 * (components.energy_reward + 1.0) / 2.0  # Normalize energy reward
        )
        
        return min(overall_compliance, 1.0)
    
    def adapt_weights(self, performance_metrics: Dict[str, float]):
        """Adapt objective weights based on performance (optional)"""
        if not self.adaptive_weights:
            return
        
        # Simple adaptation logic - boost weights for underperforming objectives
        if performance_metrics.get('throughput_achievement_rate', 1.0) < 0.8:
            self.weights.throughput = min(self.weights.throughput * 1.1, 0.4)
        
        if performance_metrics.get('latency_achievement_rate', 1.0) < 0.8:
            self.weights.latency = min(self.weights.latency * 1.1, 0.4)
        
        # Renormalize weights
        total_weight = (self.weights.throughput + self.weights.latency + 
                       self.weights.energy + self.weights.reliability +
                       self.weights.mobility + self.weights.spectrum)
        
        if total_weight > 0:
            self.weights.throughput /= total_weight
            self.weights.latency /= total_weight
            self.weights.energy /= total_weight
            self.weights.reliability /= total_weight
            self.weights.mobility /= total_weight
            self.weights.spectrum /= total_weight
    
    def get_reward_statistics(self) -> Dict[str, Any]:
        """Get reward system statistics"""
        if not self.performance_history:
            return {}
        
        recent_performance = self.performance_history[-100:]  # Last 100 episodes
        
        return {
            'avg_total_reward': np.mean([p['components'].total_reward for p in recent_performance]),
            'avg_throughput_reward': np.mean([p['components'].throughput_reward for p in recent_performance]),
            'avg_latency_reward': np.mean([p['components'].latency_reward for p in recent_performance]),
            'avg_energy_reward': np.mean([p['components'].energy_reward for p in recent_performance]),
            'avg_reliability_reward': np.mean([p['components'].reliability_reward for p in recent_performance]),
            'avg_mobility_reward': np.mean([p['components'].mobility_reward for p in recent_performance]),
            'current_weights': {
                'throughput': self.weights.throughput,
                'latency': self.weights.latency,
                'energy': self.weights.energy,
                'reliability': self.weights.reliability,
                'mobility': self.weights.mobility,
                'spectrum': self.weights.spectrum
            },
            'performance_trends': {
                'throughput_trend': np.mean([p['throughput_gbps'] for p in recent_performance[-20:]]) if len(recent_performance) >= 20 else 0,
                'latency_trend': np.mean([p['latency_ms'] for p in recent_performance[-20:]]) if len(recent_performance) >= 20 else 0,
                'mobility_trend': np.mean([p['mobility_kmh'] for p in recent_performance[-20:]]) if len(recent_performance) >= 20 else 0
            }
        }
