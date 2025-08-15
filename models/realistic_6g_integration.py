#!/usr/bin/env python3
"""
Realistic 6G Integration Layer
Integrates all realistic models to replace optimistic assumptions in the original system.

This module serves as the main interface to replace:
- Unrealistic atmospheric absorption
- Optimistic hardware capabilities  
- Continuous spectrum assumptions
- Fixed performance targets
- Simple propagation models
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

# Import all realistic models
from models.atmospheric.itu_p676_model import ITUR_P676_AtmosphericModel, AtmosphericConditions
from models.atmospheric.weather_effects import WeatherEffectsModel, WeatherParameters, WeatherCondition
from models.hardware.thz_hardware_model import THzHardwareModel, AmplifierType
from models.spectrum.thz_spectrum_windows import THzSpectrumModel, SpectrumWindow
from models.channel.ray_tracing_engine import RayTracingEngine, MaterialType, Obstacle
from models.performance.adaptive_targets import AdaptivePerformanceModel, ServiceClass, EnvironmentType, EnvironmentConditions

@dataclass
class RealisticSystemConfig:
    """Configuration for realistic 6G system"""
    frequency_ghz: float
    environment_type: EnvironmentType
    weather_condition: WeatherCondition
    service_class: ServiceClass
    enable_ray_tracing: bool = True
    enable_weather_effects: bool = True
    enable_hardware_constraints: bool = True

class Realistic6GSystem:
    """
    Main integration class for realistic 6G THz modeling.
    
    Replaces optimistic models with realistic physics-based calculations:
    - ITU-R P.676-13 atmospheric absorption
    - Realistic hardware power/noise limits
    - Fragmented spectrum windows
    - Context-aware performance targets
    - 3D ray tracing propagation
    """
    
    def __init__(self, config: RealisticSystemConfig):
        self.config = config
        
        # Initialize all realistic models
        self.atmospheric_model = ITUR_P676_AtmosphericModel()
        self.weather_model = WeatherEffectsModel()
        self.hardware_model = THzHardwareModel()
        self.spectrum_model = THzSpectrumModel()
        self.ray_tracing_engine = RayTracingEngine() if config.enable_ray_tracing else None
        self.performance_model = AdaptivePerformanceModel()
        
        # Get current atmospheric conditions
        self.atmospheric_conditions = AtmosphericConditions()
        self.weather_params = self.weather_model.get_weather_scenario(config.weather_condition)
        
        # Get hardware specifications for frequency
        self.hardware_specs = self.hardware_model.get_hardware_specifications(config.frequency_ghz)
        
        # Get usable spectrum
        freq_range = (config.frequency_ghz - 50, config.frequency_ghz + 50)
        self.spectrum_windows = self.spectrum_model.get_usable_spectrum(freq_range, 'mobile')
        
    def calculate_realistic_path_loss(self, distance_m: float, 
                                    environment_obstacles: List[Obstacle] = None) -> Dict[str, float]:
        """
        Calculate realistic path loss including all physical effects.
        
        Returns:
            Dictionary with loss breakdown in dB
        """
        results = {
            'free_space_loss_db': 0.0,
            'atmospheric_absorption_db': 0.0,
            'weather_loss_db': 0.0,
            'multipath_loss_db': 0.0,
            'total_loss_db': 0.0
        }
        
        # Free space path loss
        fspl_db = 20 * np.log10(distance_m) + 20 * np.log10(self.config.frequency_ghz * 1e9) + 20 * np.log10(4 * np.pi / 3e8)
        results['free_space_loss_db'] = fspl_db
        
        # Realistic atmospheric absorption (ITU-R P.676-13)
        gamma_o2, gamma_h2o, gamma_total = self.atmospheric_model.calculate_specific_attenuation(
            self.config.frequency_ghz, self.atmospheric_conditions
        )
        distance_km = distance_m / 1000.0
        atmospheric_loss_db = gamma_total * distance_km
        results['atmospheric_absorption_db'] = atmospheric_loss_db
        
        # Weather effects
        if self.config.enable_weather_effects:
            weather_effects = self.weather_model.calculate_total_weather_attenuation(
                self.config.frequency_ghz, self.weather_params
            )
            weather_loss_db = weather_effects['total_db_km'] * distance_km
            results['weather_loss_db'] = weather_loss_db
        
        # Multipath effects from ray tracing
        if self.config.enable_ray_tracing and self.ray_tracing_engine and environment_obstacles:
            # Simplified: assume some multipath fading
            results['multipath_loss_db'] = 5.0  # Typical THz multipath fading
        
        # Total realistic path loss
        results['total_loss_db'] = (results['free_space_loss_db'] + 
                                  results['atmospheric_absorption_db'] +
                                  results['weather_loss_db'] +
                                  results['multipath_loss_db'])
        
        return results
    
    def calculate_realistic_throughput(self, distance_m: float, 
                                     environment_obstacles: List[Obstacle] = None) -> Dict[str, float]:
        """
        Calculate realistic achievable throughput.
        
        Returns:
            Dictionary with throughput analysis
        """
        # Get path loss
        path_loss = self.calculate_realistic_path_loss(distance_m, environment_obstacles)
        
        # Get realistic hardware constraints
        max_tx_power_dbm = self.hardware_specs.max_tx_power_dbm
        noise_figure_db = self.hardware_specs.noise_figure_db
        
        # Calculate SNR
        noise_power_dbm = -174 + 10 * np.log10(self.hardware_specs.bandwidth_ghz * 1e9) + noise_figure_db
        rx_power_dbm = max_tx_power_dbm - path_loss['total_loss_db']
        snr_db = rx_power_dbm - noise_power_dbm
        
        # Get usable bandwidth (realistic spectrum windows)
        usable_bandwidth_ghz = sum(w.usable_bandwidth_ghz for w in self.spectrum_windows)
        
        # Shannon capacity with realistic implementation efficiency
        if snr_db <= 0:
            shannon_capacity_bps = 0.0
            practical_throughput_bps = 0.0
        else:
            snr_linear = 10**(snr_db / 10)
            shannon_capacity_bps = usable_bandwidth_ghz * 1e9 * np.log2(1 + snr_linear)
            
            # Apply realistic implementation efficiency
            implementation_efficiency = 0.3 if self.config.frequency_ghz > 300 else 0.6
            practical_throughput_bps = shannon_capacity_bps * implementation_efficiency
        
        return {
            'snr_db': snr_db,
            'usable_bandwidth_ghz': usable_bandwidth_ghz,
            'shannon_capacity_gbps': shannon_capacity_bps / 1e9,
            'practical_throughput_gbps': practical_throughput_bps / 1e9,
            'implementation_efficiency': implementation_efficiency,
            'max_tx_power_dbm': max_tx_power_dbm,
            'path_loss_breakdown': path_loss
        }
    
    def get_adaptive_performance_targets(self, distance_m: float, mobility_kmh: float = 0.0) -> Dict[str, float]:
        """Get realistic performance targets for current conditions"""
        
        # Create environment conditions
        conditions = EnvironmentConditions(
            environment_type=self.config.environment_type,
            distance_m=distance_m,
            frequency_ghz=self.config.frequency_ghz,
            los_probability=0.9 if distance_m < 10 else 0.6,  # Simplified
            weather_condition=self.config.weather_condition.value,
            interference_level_db=5.0,  # Typical
            mobility_kmh=mobility_kmh,
            temperature_c=self.atmospheric_conditions.temperature_K - 273.15
        )
        
        return self.performance_model.calculate_adaptive_targets(self.config.service_class, conditions)
    
    def compare_with_original_system(self, distance_m: float) -> Dict[str, Dict[str, float]]:
        """Compare realistic model with original optimistic assumptions"""
        
        # Calculate realistic performance
        realistic = self.calculate_realistic_throughput(distance_m)
        
        # Original system assumptions (from your existing code)
        original_assumptions = {
            'atmospheric_absorption_db_km': 0.3,  # Original unrealistic value
            'max_tx_power_dbm': 40.0,             # Original unrealistic for THz
            'continuous_bandwidth_ghz': 50.0 if self.config.frequency_ghz < 400 else 100.0,
            'implementation_efficiency': 0.95     # Overly optimistic
        }
        
        # Calculate what original system would predict
        original_atmos_loss = original_assumptions['atmospheric_absorption_db_km'] * (distance_m / 1000)
        original_fspl = 20 * np.log10(distance_m) + 20 * np.log10(self.config.frequency_ghz * 1e9) + 20 * np.log10(4 * np.pi / 3e8)
        original_total_loss = original_fspl + original_atmos_loss
        
        original_rx_power = original_assumptions['max_tx_power_dbm'] - original_total_loss
        original_noise_power = -174 + 10 * np.log10(original_assumptions['continuous_bandwidth_ghz'] * 1e9) + 10
        original_snr = original_rx_power - original_noise_power
        
        if original_snr > 0:
            original_shannon = original_assumptions['continuous_bandwidth_ghz'] * 1e9 * np.log2(1 + 10**(original_snr/10))
            original_throughput = original_shannon * original_assumptions['implementation_efficiency'] / 1e9
        else:
            original_throughput = 0.0
        
        return {
            'realistic': {
                'throughput_gbps': realistic['practical_throughput_gbps'],
                'atmospheric_loss_db': realistic['path_loss_breakdown']['atmospheric_absorption_db'],
                'max_tx_power_dbm': realistic['max_tx_power_dbm'],
                'usable_bandwidth_ghz': realistic['usable_bandwidth_ghz'],
                'snr_db': realistic['snr_db']
            },
            'original': {
                'throughput_gbps': original_throughput,
                'atmospheric_loss_db': original_atmos_loss,
                'max_tx_power_dbm': original_assumptions['max_tx_power_dbm'],
                'usable_bandwidth_ghz': original_assumptions['continuous_bandwidth_ghz'],
                'snr_db': original_snr
            },
            'differences': {
                'throughput_ratio': original_throughput / max(realistic['practical_throughput_gbps'], 0.001),
                'atmos_loss_ratio': original_atmos_loss / max(realistic['path_loss_breakdown']['atmospheric_absorption_db'], 0.001),
                'power_difference_db': original_assumptions['max_tx_power_dbm'] - realistic['max_tx_power_dbm'],
                'bandwidth_ratio': original_assumptions['continuous_bandwidth_ghz'] / max(realistic['usable_bandwidth_ghz'], 0.001)
            }
        }
    
    def get_system_recommendations(self) -> Dict[str, Any]:
        """Get recommendations for realistic 6G system deployment"""
        
        recommendations = {
            'optimal_frequency_ghz': None,
            'max_recommended_range_m': None,
            'required_tx_power_dbm': None,
            'recommended_bandwidth_ghz': None,
            'deployment_challenges': [],
            'mitigation_strategies': []
        }
        
        # Find optimal frequency window
        best_window = None
        best_score = 0
        
        for window in self.spectrum_windows:
            # Score based on bandwidth vs atmospheric loss
            score = window.usable_bandwidth_ghz / max(window.atmospheric_loss_db_km, 1.0)
            if score > best_score:
                best_score = score
                best_window = window
        
        if best_window:
            recommendations['optimal_frequency_ghz'] = best_window.center_freq_ghz
            recommendations['recommended_bandwidth_ghz'] = best_window.usable_bandwidth_ghz
        
        # Calculate maximum practical range
        max_range = 1.0  # Start with 1m
        while max_range < 100.0:
            throughput = self.calculate_realistic_throughput(max_range)
            if throughput['practical_throughput_gbps'] < 0.1:  # Below 100 Mbps
                break
            max_range += 1.0
        
        recommendations['max_recommended_range_m'] = max_range - 1.0
        recommendations['required_tx_power_dbm'] = self.hardware_specs.max_tx_power_dbm
        
        # Identify deployment challenges
        if self.config.frequency_ghz > 300:
            recommendations['deployment_challenges'].extend([
                'Very high atmospheric absorption at THz frequencies',
                'Limited TX power available at THz',
                'Severe range limitations (<10m typical)',
                'Weather sensitivity'
            ])
            
            recommendations['mitigation_strategies'].extend([
                'Indoor/controlled environment deployment',
                'Intelligent reflecting surfaces (IRS)',
                'Ultra-dense small cell networks',
                'Adaptive beamforming and tracking'
            ])
        elif self.config.frequency_ghz > 100:
            recommendations['deployment_challenges'].extend([
                'Moderate atmospheric absorption',
                'Spectrum fragmentation',
                'Line-of-sight requirements'
            ])
            
            recommendations['mitigation_strategies'].extend([
                'Smart relay networks',
                'Dynamic spectrum aggregation',
                'Predictive beam management'
            ])
        
        return recommendations

def test_realistic_integration():
    """Test the integrated realistic 6G system"""
    
    # Test configurations
    configs = [
        RealisticSystemConfig(
            frequency_ghz=140.0,
            environment_type=EnvironmentType.INDOOR_OFFICE,
            weather_condition=WeatherCondition.CLEAR,
            service_class=ServiceClass.EMBB
        ),
        RealisticSystemConfig(
            frequency_ghz=300.0,
            environment_type=EnvironmentType.INDOOR_OFFICE,
            weather_condition=WeatherCondition.CLEAR,
            service_class=ServiceClass.HOLOGRAPHIC
        ),
        RealisticSystemConfig(
            frequency_ghz=600.0,
            environment_type=EnvironmentType.INDOOR_OFFICE,
            weather_condition=WeatherCondition.LIGHT_RAIN,
            service_class=ServiceClass.EMBB
        )
    ]
    
    print("=== REALISTIC 6G SYSTEM INTEGRATION TEST ===")
    print("Freq | Distance | Realistic T'put | Original T'put | Reduction | Atmos Loss | Range")
    print("(GHz)|    (m)   |     (Gbps)      |     (Gbps)     |  Factor   |    (dB)    |  (m)")
    print("-" * 85)
    
    for config in configs:
        system = Realistic6GSystem(config)
        
        for distance in [5.0, 20.0]:
            comparison = system.compare_with_original_system(distance)
            recommendations = system.get_system_recommendations()
            
            realistic_throughput = comparison['realistic']['throughput_gbps']
            original_throughput = comparison['original']['throughput_gbps']
            reduction_factor = comparison['differences']['throughput_ratio']
            atmos_loss = comparison['realistic']['atmospheric_loss_db']
            max_range = recommendations['max_recommended_range_m']
            
            print(f"{config.frequency_ghz:4.0f} | {distance:6.1f}   | {realistic_throughput:13.2f}   | "
                  f"{original_throughput:12.2f}   | {reduction_factor:7.1f}   | {atmos_loss:8.1f}   | {max_range:4.1f}")
    
    # Test performance targets
    print(f"\n=== ADAPTIVE PERFORMANCE TARGETS ===")
    system = Realistic6GSystem(configs[1])  # 300 GHz system
    
    for distance in [2.0, 10.0, 30.0]:
        targets = system.get_adaptive_performance_targets(distance, mobility_kmh=5.0)
        throughput = system.calculate_realistic_throughput(distance)
        
        print(f"Distance {distance:4.1f}m: Target={targets['target_throughput_mbps']/1000:5.1f} Gbps, "
              f"Achievable={targets['achievable_throughput_mbps']/1000:5.1f} Gbps, "
              f"Actual={throughput['practical_throughput_gbps']:5.1f} Gbps")

if __name__ == "__main__":
    test_realistic_integration()
