#!/usr/bin/env python3
"""
Unified Physics Engine for DOCOMO 6G
Handles all frequency ranges seamlessly: Sub-6, mmWave, Sub-THz, THz
Ensures realistic performance across lab, indoor, and outdoor scenarios
"""

import numpy as np
from typing import Dict, Any, Tuple
from .advanced_oam_physics import AdvancedOAMPhysics


class UnifiedPhysicsEngine:
    """
    Unified physics engine that works across all frequency ranges and scenarios
    Replaces the problematic physics_calculator for consistent performance
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Scenario detection thresholds
        self.lab_distance_threshold = 15.0     # <= 15m = lab (extended for mobility)
        self.indoor_distance_threshold = 100.0 # <= 100m = indoor
        
        # Handover reduction settings
        self.handover_penalty = 0.9  # Reduce handover frequency
        
        # Initialize advanced OAM physics engine
        self.advanced_oam = AdvancedOAMPhysics(config)
        
    def calculate_realistic_throughput(self, 
                                     frequency_hz: float,
                                     distance_m: float, 
                                     sinr_db: float,
                                     bandwidth_hz: float,
                                     oam_modes: int = 1) -> Dict[str, Any]:
        """
        Calculate realistic throughput across all scenarios
        """
        
        # Detect scenario based on distance
        scenario = self._detect_scenario(distance_m)
        frequency_ghz = frequency_hz / 1e9
        
        # Enhanced SINR for THz in lab conditions
        effective_sinr = self._enhance_sinr_for_scenario(sinr_db, frequency_ghz, scenario)
        
        # Calculate base Shannon throughput
        sinr_linear = 10 ** (effective_sinr / 10.0)
        shannon_capacity = bandwidth_hz * np.log2(1 + sinr_linear)
        
        # Apply realistic modulation and coding
        practical_efficiency = self._get_practical_efficiency(effective_sinr, frequency_ghz, scenario)
        base_throughput = shannon_capacity * practical_efficiency
        
        # Apply OAM spatial multiplexing (realistic physics)
        # OAM provides spatial channels, but with practical limitations
        if scenario == "lab" and frequency_ghz >= 300:
            # Best case lab: some spatial multiplexing gain
            oam_multiplier = min(1.0 + np.log2(1 + oam_modes * 0.3), 3.0)  # Max 3x total
        elif scenario == "indoor":
            # Indoor: limited by reflections and mode coupling  
            oam_multiplier = min(1.0 + np.log2(1 + oam_modes * 0.2), 2.0)  # Max 2x total
        else:  # outdoor
            # Outdoor: severely limited by atmospheric effects
            oam_multiplier = min(1.0 + np.log2(1 + oam_modes * 0.1), 1.5)  # Max 1.5x total
        base_throughput *= oam_multiplier
        
        # Apply scenario-specific enhancements
        enhanced_throughput = self._apply_scenario_enhancements(
            base_throughput, frequency_ghz, scenario, distance_m
        )
        
        # Apply realistic caps based on current technology
        final_throughput = self._apply_realistic_caps(
            enhanced_throughput, frequency_ghz, scenario, distance_m
        )
        
        # For high-performance scenarios, use advanced OAM optimizations with realistic caps
        if (scenario == "lab" and frequency_ghz >= 100 and oam_modes >= 5):
            advanced_result = self.advanced_oam.calculate_optimized_throughput(
                frequency_hz, bandwidth_hz, distance_m, sinr_db, oam_modes, scenario
            )
            
            # Apply realistic system limits
            optimized_throughput = advanced_result['optimized_throughput']
            
            # Apply realistic technology limits
            # 1. Shannon-based limit: 5× is very optimistic, 10× is theoretical maximum
            shannon_cap = shannon_capacity * 5.0  # Use base Shannon, not advanced
            
            # 2. Absolute technology limit: 1 Tbps for current lab systems
            absolute_cap = 1000e9  # 1 Tbps maximum for any real system
            
            # Apply the most restrictive limit
            realistic_cap = min(shannon_cap, absolute_cap)
            if optimized_throughput > realistic_cap:
                optimized_throughput = realistic_cap
            
            # Use advanced result if it's significantly better but realistic
            if optimized_throughput > final_throughput * 1.2:
                return {
                    'practical_throughput': optimized_throughput,
                    'shannon_throughput': advanced_result['base_shannon'],
                    'scenario': scenario,
                    'effective_sinr': effective_sinr,
                    'frequency_ghz': frequency_ghz,
                    'enhancement_factor': optimized_throughput / max(advanced_result['base_shannon'], 1e-9),
                    'advanced_optimizations': True,
                    'bandwidth_optimization': advanced_result['bandwidth_optimization'],
                    'mode_orthogonality': advanced_result['mode_orthogonality'],
                    'effective_modes': advanced_result['effective_modes'],
                    'shannon_limited': optimized_throughput == realistic_cap,
                    'cap_type': 'shannon' if optimized_throughput == shannon_cap else 'absolute'
                }
        
        return {
            'practical_throughput': final_throughput,
            'shannon_throughput': shannon_capacity,
            'scenario': scenario,
            'effective_sinr': effective_sinr,
            'frequency_ghz': frequency_ghz,
            'enhancement_factor': final_throughput / max(base_throughput, 1e-9),
            'advanced_optimizations': False
        }
    
    def _detect_scenario(self, distance_m: float) -> str:
        """Detect deployment scenario based on distance"""
        if distance_m <= self.lab_distance_threshold:
            return "lab"
        elif distance_m <= self.indoor_distance_threshold:
            return "indoor"
        else:
            return "outdoor"
    
    def _enhance_sinr_for_scenario(self, sinr_db: float, frequency_ghz: float, scenario: str) -> float:
        """Enhance SINR based on scenario and frequency (realistic values)"""
        enhanced_sinr = sinr_db
        
        if scenario == "lab":
            # Lab conditions: controlled environment, minimal realistic improvement
            if frequency_ghz >= 300:  # THz bands
                enhanced_sinr += 3.0   # +3dB for clean lab environment 
            elif frequency_ghz >= 100:  # Sub-THz
                enhanced_sinr += 2.0   # +2dB for controlled environment
            else:  # mmWave and below
                enhanced_sinr += 1.0   # +1dB for lab
                
        elif scenario == "indoor":
            # Indoor: some reduction in interference
            if frequency_ghz >= 100:
                enhanced_sinr += 1.0   # +1dB for indoor sub-THz/THz
            else:
                enhanced_sinr += 0.5   # +0.5dB for indoor mmWave
                
        # Outdoor gets no SINR enhancement (realistic conditions)
        
        return min(enhanced_sinr, sinr_db + 5.0)  # Max 5dB total enhancement
    
    def _get_practical_efficiency(self, sinr_db: float, frequency_ghz: float, scenario: str) -> float:
        """Get practical efficiency based on SINR and conditions"""
        
        if sinr_db >= 40:
            base_efficiency = 0.9  # Excellent conditions
        elif sinr_db >= 30:
            base_efficiency = 0.8  # Very good conditions  
        elif sinr_db >= 20:
            base_efficiency = 0.7  # Good conditions
        elif sinr_db >= 10:
            base_efficiency = 0.5  # Fair conditions
        elif sinr_db >= 0:
            base_efficiency = 0.3  # Poor conditions
        else:
            base_efficiency = 0.1  # Very poor conditions
            
        # Small frequency-dependent adjustments (realistic, not multipliers)
        if frequency_ghz >= 300:
            # THz has better potential but also more implementation challenges
            base_efficiency += 0.05  # Small +5% additive bonus
        elif frequency_ghz >= 100:
            # Sub-THz modest improvement
            base_efficiency += 0.02  # Small +2% additive bonus
            
        return min(base_efficiency, 0.85)  # Realistic cap at 85% efficiency
    
    def _apply_scenario_enhancements(self, throughput: float, frequency_ghz: float, 
                                   scenario: str, distance_m: float) -> float:
        """Apply scenario-specific throughput enhancements"""
        
        # Apply minimal, realistic scenario adjustments (not multipliers)
        enhanced = throughput
        
        if scenario == "lab":
            # Lab: controlled conditions provide small improvements
            enhanced *= 1.1  # 10% improvement for controlled lab environment
                
        elif scenario == "indoor":
            # Indoor: some environmental control
            enhanced *= 1.05  # 5% improvement for indoor vs outdoor
                
        # Outdoor gets no enhancement (baseline realistic conditions)
                
        return enhanced
    
    def _apply_realistic_caps(self, throughput: float, frequency_ghz: float,
                            scenario: str, distance_m: float) -> float:
        """Apply realistic throughput caps based on current technology"""
        
        # PHYSICS-BASED approach: No arbitrary caps, just realistic physics effects
        
        # Apply atmospheric and environmental effects instead of hard caps
        physics_limited = throughput
        
        # Atmospheric absorption effects (frequency dependent)
        if frequency_ghz >= 300:  # THz frequencies
            if scenario == "outdoor":
                # High atmospheric absorption for THz outdoor
                if distance_m > 100:
                    atm_loss = 0.05  # 95% loss beyond 100m
                elif distance_m > 50:
                    atm_loss = 0.2   # 80% loss at 50-100m
                else:
                    atm_loss = 0.5   # 50% loss under 50m
                physics_limited *= atm_loss
            elif scenario == "indoor":
                # Indoor reflections and wall penetration
                physics_limited *= 0.8  # 20% loss indoors
            # Lab: minimal atmospheric effects (controlled environment)
            
        elif frequency_ghz >= 60:  # mmWave
            if scenario == "outdoor" and distance_m > 200:
                # mmWave outdoor propagation limits
                physics_limited *= 0.6  # 40% loss at long range
                
        # Technology readiness factor (frequency dependent)
        if frequency_ghz >= 600:
            tech_factor = 0.85  # 600+ GHz still experimental
        elif frequency_ghz >= 300:
            tech_factor = 0.95  # 300+ GHz achievable with current tech
        else:
            tech_factor = 1.0   # Lower frequencies are mature
            
        return physics_limited * tech_factor
    
    def should_reduce_handovers(self, current_performance: float, 
                              target_performance: float, scenario: str) -> bool:
        """Determine if handovers should be reduced to maximize throughput"""
        
        # Strong preference to avoid handovers when performance is good
        performance_ratio = current_performance / max(target_performance, 1.0)
        
        if scenario == "lab":
            return performance_ratio > 0.5  # Avoid handovers if >50% of target
        elif scenario == "indoor":
            return performance_ratio > 0.7  # Avoid handovers if >70% of target  
        else:  # outdoor
            return performance_ratio > 0.8  # Avoid handovers if >80% of target
    
    def get_optimal_band_for_scenario(self, available_bands: list, 
                                    scenario: str, distance_m: float) -> str:
        """Get optimal frequency band for given scenario"""
        
        if scenario == "lab":
            # Prefer highest frequency for maximum throughput
            return max(available_bands, key=lambda b: float(b.get('frequency', 0)))
        elif scenario == "indoor":
            # Balance frequency and range
            suitable_bands = [b for b in available_bands 
                            if float(b.get('frequency', 0)) <= 300e9]  # Up to 300 GHz
            return max(suitable_bands or available_bands, 
                      key=lambda b: float(b.get('frequency', 0)))
        else:  # outdoor
            # Prefer lower frequencies for better range
            suitable_bands = [b for b in available_bands 
                            if float(b.get('frequency', 0)) <= 100e9]  # Up to 100 GHz
            return max(suitable_bands or available_bands,
                      key=lambda b: float(b.get('frequency', 0)))
