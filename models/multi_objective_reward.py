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
    THROUGHPUT = "throughput"                                      
    LATENCY = "latency"                                             
    ENERGY = "energy"                                              
    RELIABILITY = "reliability"                             
    MOBILITY = "mobility"                                        
    SPECTRUM = "spectrum"                                
    STABILITY = "stability"                           
    HANDOVER = "handover"                                 

@dataclass
class DOCOMOObjectiveWeights:
    """DOCOMO objective weights (must sum to 1.0)"""
    throughput: float = 0.25                                        
    latency: float = 0.25                                             
    energy: float = 0.20                                     
    reliability: float = 0.15                                       
    mobility: float = 0.10                                     
    spectrum: float = 0.05                                    
    
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
        
                                 
        kpi_targets = config.get('docomo_6g_system', {}).get('kpi_targets', {})
        self.target_throughput_gbps = kpi_targets.get('user_data_rate_gbps', 10.0)  # Realistic 10 Gbps base target
        self.target_latency_ms = kpi_targets.get('latency_ms', 0.1)
        self.target_reliability = kpi_targets.get('reliability', 0.9999999)
        self.target_mobility_kmh = kpi_targets.get('mobility_kmh', 500.0)
        self.target_energy_improvement = kpi_targets.get('energy_efficiency_improvement', 100.0)
        self.target_spectrum_improvement = kpi_targets.get('spectrum_efficiency_improvement', 10.0)
        
                                
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
        
        # Dynamic QoS targets and reward weights for network slicing
        self.current_qos_targets = {}
        self.current_reward_weights = {}
                                                                             
        self.reward_scale = 1.0  # Reduce reward scale to stabilize learning                                                     
        self.penalty_scale = 0.5  # Reduce penalty scale                                           
        
                                     
        self.physics_bonus_enabled = True
        self.atmospheric_penalty_enabled = True
        self.oam_crosstalk_penalty_enabled = True
        
                             
        self.adaptive_weights = False                             
        self.performance_history = []
        
                                                             
        self.low_band_good_sinr_streak = 0
        self.low_band_good_sinr_threshold_db = 10.0
        
                                           
        self.baseline_energy_w = 1.0                                     
        self.baseline_spectrum_efficiency = 1.0                                
        self.baseline_handover_rate = 0.1                           
        
    def set_qos_targets(self, qos_targets: Dict[str, float]):
        """Set dynamic QoS targets based on the current network slice."""
        self.current_qos_targets = qos_targets

    def set_reward_weights(self, reward_weights: Dict[str, float]):
        """Set dynamic reward weights based on the current network slice."""
        self.current_reward_weights = reward_weights

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
                                   
        sinr_db = next_state[0] if len(next_state) > 0 else 0.0
        throughput_gbps = info.get('throughput_gbps', 0.0)
        latency_ms = info.get('latency_ms', 1.0)
        distance_m = next_state[3] if len(next_state) > 3 else 100.0
        velocity_kmh = np.sqrt(next_state[4]**2 + next_state[5]**2 + next_state[6]**2) * 3.6 if len(next_state) > 6 else 0.0
        energy_w = info.get('energy_consumption_w', 1.0)
        reliability_score = info.get('reliability_score', 1.0)
        handover_occurred = info.get('handover_occurred', False)
        
                                                  
        self._current_action_info = info
                                                      
        try:
            self._current_action_info['sinr_db'] = float(sinr_db)
        except Exception:
            pass
        
                                                
        components = RewardComponents()
        
                                                                 
        components.throughput_reward = self._calculate_throughput_reward(
            throughput_gbps, sinr_db, distance_m, info
        )
        
                                                   
        components.latency_reward = self._calculate_latency_reward(
            latency_ms, action, info
        )
        
                                                                
        components.energy_reward = self._calculate_energy_reward(
            energy_w, throughput_gbps, info
        )
        
                                                          
        components.reliability_reward = self._calculate_reliability_reward(
            reliability_score, sinr_db, info
        )
        
                                                       
        components.mobility_reward = self._calculate_mobility_reward(
            velocity_kmh, throughput_gbps, latency_ms, info
        )
        
                                                                 
        components.spectrum_reward = self._calculate_spectrum_reward(
            throughput_gbps, info
        )
        
                                    
        components.stability_reward = self._calculate_stability_reward(
            state, next_state, info
        )
        
                                         
        components.handover_penalty = self._calculate_handover_penalty(
            handover_occurred, velocity_kmh, info
        )
        
                                   
        components.physics_bonus = self._calculate_physics_bonus(
            state, action, next_state, info
        )
        
                                         
        components.total_reward = self._calculate_total_reward(components)
        
                                        
        components.total_reward = self._apply_constraints(
            components.total_reward, state, next_state, info
        )
        
                                    
        self.performance_history.append({
            'components': components,
            'throughput_gbps': throughput_gbps,
            'latency_ms': latency_ms,
            'mobility_kmh': velocity_kmh
        })
        
                              
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-500:]
        
                                     
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
            return -1.0                               
        
                                                                        
        target_tp = self.current_qos_targets.get('min_throughput_gbps', self.target_throughput_gbps)
        # Enhanced reward for lab THz - no cap to encourage exploration
        base_reward = throughput_gbps / target_tp
        
        # Apply lab THz exploration boost
        current_band = info.get('current_band', 'mmwave_28') if info else 'mmwave_28'
        if 'thz_' in current_band and distance_m <= 10:  # Lab THz conditions
            # Strong incentive for breaking through 100+ Gbps
            if throughput_gbps >= 100.0:
                base_reward *= 2.0  # 2x reward for achieving 100+ Gbps 
            elif throughput_gbps >= 80.0:
                base_reward *= 1.5  # 1.5x reward for 80+ Gbps
        else:
            # Keep reasonable cap for non-lab scenarios
            base_reward = min(base_reward, 3.0)                              
        
                                                                               
        distance_bonus = max(0.0, (500.0 - distance_m) / 500.0) * 0.3                   
        
                                                                    
        sinr_bonus = max(0.0, (sinr_db - 5.0) / 25.0) * 0.5                       
        
                                                                                
        # Enhanced THz band rewards for lab exploration
        high_freq_bands = ['sub_thz_100', 'sub_thz_140', 'sub_thz_220', 'sub_thz_300', 'thz_300', 'thz_400', 'thz_600']
        if current_band in high_freq_bands:
            band_multipliers = {
                'sub_thz_100': 1.20,         
                'sub_thz_140': 1.30,         
                'sub_thz_220': 1.40,           
                'sub_thz_300': 1.50,
                'thz_300': 2.0,    # Strong bonus for 300 GHz lab
                'thz_400': 2.5,    # Even stronger for 400 GHz
                'thz_600': 3.0     # Maximum bonus for 600 GHz                     
            }
            high_freq_bonus = band_multipliers.get(current_band, 1.0) - 1.0
            
            # Extra lab THz exploration bonus
            if 'thz_' in current_band and distance_m <= 10:
                high_freq_bonus *= 1.5  # 1.5x boost for lab THz
        else:
            high_freq_bonus = 0.0
        
                                                       
        if throughput_gbps > target_tp:
            excellence_bonus = math.log(throughput_gbps / target_tp) * 0.8                   
        else:
            excellence_bonus = 0.0
        
                                                               
        total_reward = (base_reward + distance_bonus + sinr_bonus + excellence_bonus) * (1.0 + high_freq_bonus)
        
        return min(total_reward, 3.0)              
    
    def _calculate_latency_reward(self, 
                                latency_ms: float, 
                                action: int,
                                info: Dict[str, Any]) -> float:
        """Calculate latency reward with DOCOMO 0.1 ms target"""
        if latency_ms <= 0:
            return -1.0                   
        
                                                                 
        target_lat = self.current_qos_targets.get('max_latency_ms', self.target_latency_ms)
        if latency_ms <= target_lat:
                                        
            reward = 1.0 + (target_lat - latency_ms) / target_lat
        else:
                                                      
            excess_factor = latency_ms / target_lat
            reward = 1.0 / excess_factor                     
        
                                         
        handover_latency_penalty = 0.0
        if action in [3, 4]:                          
            handover_latency_penalty = -0.2                          
        
                                     
        beam_tracking_active = info.get('beam_tracking_active', False)
        if beam_tracking_active:
            reward += 0.1                                 
        
        total_reward = reward + handover_latency_penalty
        
        return max(total_reward, -2.0)                 
    
    def _calculate_energy_reward(self,
                               energy_w: float,
                               throughput_gbps: float,
                               info: Dict[str, Any]) -> float:
        """Calculate energy efficiency reward (DOCOMO: 100x improvement)"""
        if energy_w <= 0 or throughput_gbps <= 0:
            return -1.0
        
                                           
        energy_efficiency = throughput_gbps / energy_w                 
        baseline_efficiency = self.target_throughput_gbps / self.baseline_energy_w
        
                                                            
        efficiency_ratio = energy_efficiency / baseline_efficiency
        
                                                        
        if efficiency_ratio > 1.0:
            reward = math.log(efficiency_ratio) / math.log(self.target_energy_improvement)
        else:
            reward = -1.0 + efficiency_ratio                                 
        
                                                 
        optimal_band = info.get('using_optimal_band', False)
        if optimal_band:
            reward += 0.2
        
                                     
        if energy_w > 5.0 * self.baseline_energy_w:                            
            reward -= 0.5
        
        return max(min(reward, 2.0), -2.0)                   
    
    def _calculate_reliability_reward(self,
                                    reliability_score: float,
                                    sinr_db: float,
                                    info: Dict[str, Any]) -> float:
        """Calculate reliability reward (DOCOMO: 99.99999% target)"""
                                 
        target_rel = self.current_qos_targets.get('min_reliability', self.target_reliability)
        reliability_ratio = reliability_score / target_rel
        
        if reliability_ratio >= 1.0:
            reward = 1.0                                  
        else:
                                                          
            reward = (reliability_ratio - 0.99) / 0.01                         
            reward = max(reward, -2.0)                 
        
                                          
        if sinr_db > 15.0:                                       
            reward += 0.3
        elif sinr_db < 5.0:                               
            reward -= 0.5
        
                                  
        error_rate = info.get('error_rate', 0.0)
        if error_rate < 1e-7:                      
            reward += 0.2
        elif error_rate > 1e-5:                          
            reward -= 1.0
        
                              
        link_stable = info.get('link_stable', True)
        if not link_stable:
            reward -= 0.5
        
        return max(min(reward, 2.0), -3.0)                   
    
    def _calculate_mobility_reward(self,
                                 velocity_kmh: float,
                                 throughput_gbps: float,
                                 latency_ms: float,
                                 info: Dict[str, Any]) -> float:
        """Calculate mobility reward (DOCOMO: 500 km/h target)"""
                                      
        target_mob = self.current_qos_targets.get('max_mobility_kmh', self.target_mobility_kmh)
        mobility_ratio = min(velocity_kmh / target_mob, 1.0)
        base_reward = mobility_ratio * 1.0
        
                                            
        if velocity_kmh > 100.0:                          
                                          
            if throughput_gbps > 0.5 * self.target_throughput_gbps:
                base_reward += 0.5
            
                                         
            if latency_ms < 2.0 * self.target_latency_ms:
                base_reward += 0.3
            
                                       
            handover_success_rate = info.get('handover_success_rate', 1.0)
            base_reward += handover_success_rate * 0.2
        
                                   
        prediction_accuracy = info.get('mobility_prediction_accuracy', 0.0)
        base_reward += prediction_accuracy * 0.3
        
                                    
        doppler_compensated = info.get('doppler_compensated', False)
        if doppler_compensated and velocity_kmh > 50.0:
            base_reward += 0.2
        
                                                 
        if velocity_kmh > 350.0:
            extreme_bonus = min((velocity_kmh - 350.0) / 150.0, 1.0) * 0.5
            base_reward += extreme_bonus
        
        return min(base_reward, 2.0)              
    
    def _calculate_spectrum_reward(self,
                                 throughput_gbps: float,
                                 info: Dict[str, Any]) -> float:
        """Calculate spectrum efficiency reward (DOCOMO: 10x improvement)"""
        bandwidth_mhz = info.get('bandwidth_mhz', 100.0)
        
        if bandwidth_mhz <= 0:
            return -1.0
        
                                     
        spectrum_efficiency = (throughput_gbps * 1e9) / (bandwidth_mhz * 1e6)          
        
                                                      
        baseline_efficiency = self.baseline_spectrum_efficiency             
        
                                      
        improvement_ratio = spectrum_efficiency / baseline_efficiency
        
                            
        if improvement_ratio > 1.0:
            reward = math.log(improvement_ratio) / math.log(self.target_spectrum_improvement)
        else:
            reward = improvement_ratio - 1.0                                 
        
                                                    
        oam_efficiency_bonus = info.get('oam_mode_efficiency', 0.0) * 0.2
        multi_band_bonus = info.get('multi_band_active', False) * 0.1
        
        total_reward = reward + oam_efficiency_bonus + multi_band_bonus
        
        return max(min(total_reward, 1.5), -1.5)                       
    
    def _calculate_stability_reward(self,
                                  state: np.ndarray,
                                  next_state: np.ndarray,
                                  info: Dict[str, Any]) -> float:
        """Calculate system stability reward"""
                        
        if len(state) > 0 and len(next_state) > 0:
            sinr_change = abs(next_state[0] - state[0])
            sinr_stability = max(0.0, 1.0 - sinr_change / 10.0)                              
        else:
            sinr_stability = 0.0
        
                              
        current_throughput = info.get('throughput_gbps', 0.0)
        prev_throughput = info.get('prev_throughput_gbps', current_throughput)
        
        if prev_throughput > 0:
            throughput_change = abs(current_throughput - prev_throughput) / prev_throughput
            throughput_stability = max(0.0, 1.0 - throughput_change / 0.2)                            
        else:
            throughput_stability = 0.0
        
                                                                   
        mode_switches = info.get('recent_mode_switches', 0)
        switching_penalty = min(mode_switches * 0.2, 1.0)                    
        
                                 
        stability_score = (sinr_stability + throughput_stability) / 2.0 - switching_penalty
        
        return max(min(stability_score, 1.0), -1.0)                   
    
    def _calculate_handover_penalty(self,
                                  handover_occurred: bool,
                                  velocity_kmh: float,
                                  info: Dict[str, Any]) -> float:
        """Calculate handover penalty with velocity consideration"""
        if not handover_occurred:
            return 0.0
        
                               
        base_penalty = -1.0
        
                                                                           
        if velocity_kmh > 50.0:
            velocity_adjustment = (velocity_kmh - 50.0) / 450.0                       
            base_penalty *= (1.0 - velocity_adjustment * 0.5)                                 
        
                                
        handover_success = info.get('handover_successful', True)
        if not handover_success:
            base_penalty *= 2.0                                       
        
                                                               
        handover_timing_score = info.get('handover_timing_score', 1.0)                 
        base_penalty *= (2.0 - handover_timing_score)                                 
        
        return max(base_penalty, -2.0)               
    
    def _calculate_physics_bonus(self,
                               state: np.ndarray,
                               action: int,
                               next_state: np.ndarray,
                               info: Dict[str, Any]) -> float:
        """Calculate physics-informed bonus/penalty"""
        if not self.physics_bonus_enabled:
            return 0.0
        
        total_bonus = 0.0
        
                                          
        distance_m = next_state[3] if len(next_state) > 3 else 100.0
        current_mode = info.get('current_oam_mode', 1)
        
                                      
        if distance_m < 50.0 and current_mode <= 3:                                
            total_bonus += 0.2
        elif 50.0 <= distance_m <= 150.0 and 3 <= current_mode <= 6:                                 
            total_bonus += 0.2
        elif distance_m > 150.0 and current_mode >= 6:                                
            total_bonus += 0.2
        
                                  
        frequency_ghz = info.get('frequency_ghz', 28.0)
        in_atmospheric_window = info.get('in_atmospheric_window', False)
        if in_atmospheric_window:
            total_bonus += 0.1
        
                              
        beam_alignment_error = info.get('beam_alignment_error_deg', 0.0)
        if beam_alignment_error < 0.1:                       
            total_bonus += 0.15
        elif beam_alignment_error > 2.0:                  
            total_bonus -= 0.3
        
                                      
        crosstalk_db = info.get('oam_crosstalk_db', 0.0)
        if crosstalk_db < -20.0:                       
            total_bonus += 0.1
        elif crosstalk_db > -10.0:                  
            total_bonus -= 0.2
        
        return max(min(total_bonus, 0.5), -0.5)                       
    
    def _calculate_total_reward(self, components: RewardComponents) -> float:
        """Calculate weighted total reward with enhanced high-frequency band incentives"""
                              
        # Use dynamic weights if available, otherwise fallback to default configured weights
        weights = self.current_reward_weights if self.current_reward_weights else self.weights.__dict__

        total_reward = (
            weights.get('throughput_bonus', weights.get('throughput', 0.25)) * components.throughput_reward +
            weights.get('latency_penalty', weights.get('latency', 0.25)) * components.latency_reward +
            weights.get('energy_efficiency_bonus', weights.get('energy', 0.20)) * components.energy_reward +
            weights.get('reliability_bonus', weights.get('reliability', 0.15)) * components.reliability_reward +
            weights.get('mobility_bonus', weights.get('mobility', 0.10)) * components.mobility_reward +
            weights.get('spectrum', 0.05) * components.spectrum_reward +
            0.1 * components.stability_reward +                               
            components.handover_penalty +
            components.physics_bonus
        )
        
                                 
        action_info = getattr(self, '_current_action_info', {})
        current_band = action_info.get('current_band', 'mmwave_28')
        throughput_gbps = action_info.get('throughput_gbps', 0.0)
        
                                                                                           
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
                                                                                               
        if throughput_gbps < 5.0:
            frequency_bonus = min(frequency_bonus, 0.0)
        
                                                                          
        compliance_bonus = 0.0
        # Strong compliance bonus for consistent high performance                                                           
        if throughput_gbps >= 50.0:
            compliance_bonus = 10.0  # Excellent - 50+ Gbps is realistic peak
        elif throughput_gbps >= 20.0:
            compliance_bonus = 5.0 + 5.0 * (throughput_gbps - 20.0) / 30.0
        elif throughput_gbps >= 10.0:
            compliance_bonus = 2.0 + 3.0 * (throughput_gbps - 10.0) / 10.0
        elif throughput_gbps >= 5.0:
            compliance_bonus = 1.0 + 1.0 * (throughput_gbps - 5.0) / 5.0
        elif throughput_gbps >= 1.0:
            compliance_bonus = 0.5 * (throughput_gbps / 5.0)
        else:
            compliance_bonus = -2.0  # Penalty for very low throughput
        
        # Strong incentive for using optimal THz bands                                                                         
        thz_band_bonus = 0.0
        if current_band in ['sub_thz_300', 'thz_600']:
            thz_band_bonus = 2.0  # Strong bonus for using optimal bands
        elif current_band in ['sub_thz_100', 'sub_thz_140', 'sub_thz_220']:
            thz_band_bonus = 1.0  # Medium bonus for sub-THz bands
        
        band_switch_bonus = 0.0
        if action_info.get('high_freq_bonus', 0.0) > 0:
            band_switch_bonus = min(action_info['high_freq_bonus'] * 0.5, 0.5)
            
                                  
        beam_bonus = 0.0
        if action_info.get('beam_optimization_bonus', 0.0) > 0:
            beam_bonus = action_info['beam_optimization_bonus'] * 1.0             
            
                                   
        prediction_bonus = 0.0
        if action_info.get('prediction_bonus', 0.0) > 0:
            prediction_bonus = action_info['prediction_bonus'] * 0.5
        
                           
                                                            
        low_band_penalty = 0.0
        try:
            sinr_db_val = float(action_info.get('sinr_db', 0.0))
            is_low_band = current_band in ['sub_6ghz', 'mmwave_28', 'mmwave_39']
            good_sinr = sinr_db_val > self.low_band_good_sinr_threshold_db
            if is_low_band and good_sinr and throughput_gbps < 400.0:
                                                                       
                self.low_band_good_sinr_streak = min(self.low_band_good_sinr_streak + 1, 50)
                scale = min(0.02 * self.low_band_good_sinr_streak, 0.6)                        
                low_band_penalty = -scale
            else:
                                                     
                self.low_band_good_sinr_streak = max(self.low_band_good_sinr_streak - 1, 0)
        except Exception:
                                    
            if current_band == 'sub_6ghz':
                low_band_penalty = -0.05
        
                                                                           
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
            thz_band_bonus +    # Add THz band incentive
            band_switch_bonus + 
            beam_bonus + 
            prediction_bonus +
            low_band_penalty +
            mmwave_penalty
        )
        
                                       
        return enhanced_reward * self.reward_scale
    
    def _apply_constraints(self,
                         reward: float,
                         state: np.ndarray,
                         next_state: np.ndarray,
                         info: Dict[str, Any]) -> float:
        """Apply final constraints and clipping"""
                                                              
        reward = max(min(reward, 10.0), -10.0)
        
                                  
        if info.get('terminal_failure', False):
            reward = -25.0                                    
        
                                     
        if info.get('physics_violation', False):
            reward -= 10.0
        
        return reward
    
    def _calculate_docomo_compliance(self,
                                   components: RewardComponents,
                                   info: Dict[str, Any]) -> float:
        """Calculate overall DOCOMO compliance score"""
                             
        throughput_gbps = info.get('throughput_gbps', 0.0)
        latency_ms = info.get('latency_ms', 1.0)
        reliability = info.get('reliability_score', 1.0)
        mobility_kmh = info.get('mobility_kmh', 0.0)
        
                                      
        throughput_compliance = min(throughput_gbps / self.target_throughput_gbps, 1.0)
        latency_compliance = min(self.target_latency_ms / latency_ms, 1.0) if latency_ms > 0 else 0.0
        reliability_compliance = reliability / self.target_reliability
        mobility_compliance = min(mobility_kmh / self.target_mobility_kmh, 1.0)
        
                                   
        # Use dynamic weights for compliance calculation as well
        weights = self.current_reward_weights if self.current_reward_weights else self.weights.__dict__

        overall_compliance = (
            weights.get('throughput_bonus', weights.get('throughput', 0.25)) * throughput_compliance +
            weights.get('latency_penalty', weights.get('latency', 0.25)) * latency_compliance +
            weights.get('reliability_bonus', weights.get('reliability', 0.15)) * reliability_compliance +
            weights.get('mobility_bonus', weights.get('mobility', 0.10)) * mobility_compliance +
            weights.get('energy_efficiency_bonus', weights.get('energy', 0.20)) * (components.energy_reward + 1.0) / 2.0                           
        )
        
        return min(overall_compliance, 1.0)
    
    def adapt_weights(self, performance_metrics: Dict[str, float]):
        """Adapt objective weights based on performance (optional)"""
        if not self.adaptive_weights:
            return
        
                                                                                
        if performance_metrics.get('throughput_achievement_rate', 1.0) < 0.8:
            self.weights.throughput = min(self.weights.throughput * 1.1, 0.4)
        
        if performance_metrics.get('latency_achievement_rate', 1.0) < 0.8:
            self.weights.latency = min(self.weights.latency * 1.1, 0.4)
        
                             
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
        
        recent_performance = self.performance_history[-100:]                     
        
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
