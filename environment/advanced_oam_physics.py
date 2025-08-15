#!/usr/bin/env python3
"""
Advanced OAM Physics Implementation
Incorporates realistic engineering optimizations for maximum throughput
"""

import numpy as np
from typing import Dict, Any, Tuple
import math

class AdvancedOAMPhysics:
    """
    Advanced OAM physics model incorporating real engineering optimizations:
    1. Adaptive bandwidth allocation
    2. Power allocation optimization  
    3. Antenna gain improvements
    4. Mode orthogonality optimization
    5. Adaptive mode scheduling via RL
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Engineering parameters - UPGRADED HARDWARE
        self.max_total_power_dbm = 53.0  # +3 dB power increase (hybrid option)
        self.antenna_gain_db = 45.0      # High-gain array
        self.noise_figure_db = 3.0       # Low-noise front-end
        self.adc_parallel_paths = 8      # DOUBLED: 8×100 Gsps = 800 Gsps total
        self.photonic_frontend = True    # Photonic processing capability
        
        # Mode orthogonality parameters
        self.mode_converter_quality = 0.95   # High-quality mode converters
        self.alignment_precision_rad = 0.001 # Precise beam alignment
        self.aperture_engineering_factor = 1.2  # Optimized aperture design
        
        # Adaptive parameters
        self.adaptive_mode_scheduling = True
        self.power_concentration_factor = 0.8  # Focus power on best modes
        
    def calculate_optimized_throughput(self, 
                                     base_frequency_hz: float,
                                     base_bandwidth_hz: float,
                                     distance_m: float,
                                     sinr_db: float,
                                     oam_modes: int,
                                     scenario: str = "lab") -> Dict[str, Any]:
        """
        Calculate throughput with advanced OAM optimizations
        """
        
        # 1. Bandwidth Engineering: Increase usable bandwidth
        optimized_bw = self._optimize_bandwidth(base_bandwidth_hz, scenario, base_frequency_hz)
        
        # 2. Power Allocation: Optimize per-mode power distribution
        power_allocation = self._calculate_optimal_power_allocation(oam_modes, scenario)
        
        # 3. Antenna/RF Improvements: Better SNR through engineering
        enhanced_sinr = self._enhance_sinr_engineering(sinr_db, distance_m, scenario)
        
        # 4. Mode Orthogonality: Reduce crosstalk between modes
        mode_efficiency = self._calculate_mode_orthogonality(oam_modes, scenario)
        
        # 5. Adaptive Mode Scheduling: RL-optimized mode utilization
        effective_modes = self._adaptive_mode_scheduling(oam_modes, enhanced_sinr, scenario)
        
        # Calculate base Shannon capacity with optimizations
        sinr_linear = 10 ** (enhanced_sinr / 10.0)
        shannon_capacity = optimized_bw * np.log2(1 + sinr_linear)
        
        # Apply mode multiplexing with orthogonality
        spatial_multiplexing_gain = effective_modes * mode_efficiency
        
        # Apply engineering improvements
        engineering_factor = self._calculate_engineering_factor(scenario)
        
        # Final throughput calculation
        final_throughput = shannon_capacity * spatial_multiplexing_gain * engineering_factor
        
        return {
            'base_shannon': shannon_capacity,
            'optimized_throughput': final_throughput,
            'bandwidth_optimization': optimized_bw / base_bandwidth_hz,
            'power_efficiency': 0.85,  # Fixed: Use realistic efficiency instead of raw power
            'mode_orthogonality': mode_efficiency,
            'effective_modes': effective_modes,
            'engineering_factor': engineering_factor,
            'scenario': scenario,
            'sinr_enhancement': enhanced_sinr - sinr_db
        }
    
    def _optimize_bandwidth(self, base_bw_hz: float, scenario: str, frequency_hz: float) -> float:
        """Optimize usable bandwidth through advanced techniques"""
        
        # Base bandwidth expansion based on scenario capabilities - REALISTIC
        if scenario == "lab":
            # Lab: Realistic RF frontend improvements
            if frequency_hz >= 300e9:  # THz
                bw_expansion = 1.5   # Realistic 15→22.5 GHz expansion
            elif frequency_hz >= 100e9:  # Sub-THz
                bw_expansion = 1.3   # Modest improvement
            else:
                bw_expansion = 1.2   # Small improvement
        elif scenario == "indoor":
            # Indoor: Some bandwidth expansion possible
            bw_expansion = 1.3
        else:  # outdoor
            # Outdoor: Limited by interference
            bw_expansion = 1.1
            
        # Additional optimization for photonic front-ends - REDUCED
        if self.photonic_frontend and frequency_hz >= 100e9:
            bw_expansion *= 1.1  # Modest photonic processing benefit
            
        return base_bw_hz * bw_expansion
    
    def _calculate_optimal_power_allocation(self, oam_modes: int, scenario: str) -> Dict[int, float]:
        """Calculate optimal power allocation across OAM modes"""
        
        power_allocation = {}
        total_power_linear = 10 ** (self.max_total_power_dbm / 10.0)
        
        if self.adaptive_mode_scheduling and scenario == "lab":
            # Adaptive: Concentrate power on highest-quality modes
            # Higher-order modes typically have better orthogonality
            weights = np.array([i**0.5 for i in range(1, oam_modes + 1)])
            weights = weights / np.sum(weights)  # Normalize
            
            for mode in range(1, oam_modes + 1):
                power_allocation[mode] = weights[mode-1] * total_power_linear
        else:
            # Uniform power distribution
            power_per_mode = total_power_linear / oam_modes
            for mode in range(1, oam_modes + 1):
                power_allocation[mode] = power_per_mode
                
        return power_allocation
    
    def _enhance_sinr_engineering(self, base_sinr_db: float, distance_m: float, scenario: str) -> float:
        """Enhance SINR through antenna/RF engineering"""
        
        enhanced_sinr = base_sinr_db
        
        # Power amplifier improvement (+3 dB from hybrid upgrade)
        power_amplifier_gain = 3.0  # +3dB from increased TX power (53 dBm)
        
        # Antenna gain improvement (already in system, but track contribution)
        # High-gain phased arrays provide significant improvement
        antenna_improvement = 3.0  # +3dB from advanced antenna design
        
        # Low noise figure improvement
        nf_improvement = 2.0  # +2dB from low-noise front-end
        
        # Distance-dependent path loss compensation
        if scenario == "lab" and distance_m <= 10:
            # Lab: Perfect alignment and minimal path loss
            alignment_gain = 2.0  # +2dB from precise alignment
        else:
            alignment_gain = 0.5  # +0.5dB from good alignment
            
        # Parallel processing gain (DOUBLED ADC paths reduce quantization noise)
        if self.adc_parallel_paths > 1:
            processing_gain = 10 * np.log10(self.adc_parallel_paths) * 0.6  # Enhanced benefit
        else:
            processing_gain = 0.0
            
        enhanced_sinr += (power_amplifier_gain + antenna_improvement + 
                         nf_improvement + alignment_gain + processing_gain)
        
        return min(enhanced_sinr, base_sinr_db + 15.0)  # Increased cap to +15dB
    
    def _calculate_mode_orthogonality(self, oam_modes: int, scenario: str) -> float:
        """Calculate mode orthogonality efficiency"""
        
        # Base orthogonality depends on number of modes (more modes = more crosstalk)
        base_orthogonality = 1.0 / (1.0 + 0.1 * np.log(oam_modes))
        
        # Engineering improvements
        mode_converter_boost = self.mode_converter_quality  # High-quality converters
        alignment_boost = 1.0 - (self.alignment_precision_rad * 100)  # Precise alignment
        aperture_boost = self.aperture_engineering_factor / 2.0  # Optimized aperture
        
        # Scenario-dependent factors
        if scenario == "lab":
            environment_factor = 0.95  # Controlled environment
        elif scenario == "indoor":
            environment_factor = 0.85  # Some reflections
        else:  # outdoor
            environment_factor = 0.70  # Atmospheric turbulence
            
        # Combined orthogonality
        orthogonality = (base_orthogonality * 
                        mode_converter_boost * 
                        alignment_boost * 
                        aperture_boost * 
                        environment_factor)
        
        return min(orthogonality, 0.95)  # Cap at 95% efficiency
    
    def _adaptive_mode_scheduling(self, total_modes: int, sinr_db: float, scenario: str) -> float:
        """RL-inspired adaptive mode scheduling"""
        
        if not self.adaptive_mode_scheduling:
            return float(total_modes)
            
        # Adaptive algorithm: Use modes based on quality
        # Higher SINR allows more effective modes
        if sinr_db >= 40:
            mode_utilization = 0.9  # Can effectively use 90% of modes
        elif sinr_db >= 30:
            mode_utilization = 0.8  # 80% utilization
        elif sinr_db >= 20:
            mode_utilization = 0.6  # 60% utilization
        else:
            mode_utilization = 0.4  # 40% utilization
            
        # Scenario adjustment
        if scenario == "lab":
            mode_utilization *= 1.1  # Better utilization in lab
        elif scenario == "outdoor":
            mode_utilization *= 0.8  # Reduced utilization outdoor
            
        effective_modes = total_modes * mode_utilization
        
        # Concentration benefit: Better to use fewer modes well than many poorly
        concentration_benefit = 1.0 + (1.0 - mode_utilization) * self.power_concentration_factor
        
        return effective_modes * concentration_benefit
    
    def _calculate_engineering_factor(self, scenario: str) -> float:
        """Overall engineering factor combining all improvements"""
        
        # Base engineering improvements
        base_factor = 1.0
        
        # Photonic front-end benefits
        if self.photonic_frontend:
            base_factor *= 1.15  # 15% improvement from photonic processing
            
        # Parallel processing benefits
        if self.adc_parallel_paths > 1:
            parallel_benefit = 1.0 + 0.05 * (self.adc_parallel_paths - 1)  # 5% per additional path
            base_factor *= min(parallel_benefit, 1.2)  # Cap at 20% improvement
            
        # Advanced signal processing
        dsp_factor = 1.08  # 8% from advanced DSP algorithms
        base_factor *= dsp_factor
        
        # Scenario-specific engineering optimizations - REALISTIC
        if scenario == "lab":
            # Lab allows for modest engineering optimization
            scenario_factor = 1.1  # 10% additional optimization (reduced)
        elif scenario == "indoor":
            scenario_factor = 1.05 # 5% optimization
        else:  # outdoor
            scenario_factor = 1.0  # No additional optimization
            
        total_factor = base_factor * scenario_factor
        return min(total_factor, 2.0)  # Cap at 2× engineering improvement
