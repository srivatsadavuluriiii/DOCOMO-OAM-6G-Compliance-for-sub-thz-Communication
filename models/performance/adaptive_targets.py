#!/usr/bin/env python3
"""
Adaptive Performance Targets for Realistic 6G THz Systems
Context-aware performance targets that adapt to environmental conditions,
distance, frequency, and service requirements.

This replaces fixed unrealistic targets with dynamic achievable goals.
"""

import numpy as np
import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class ServiceClass(Enum):
    URLC = "ultra_reliable_low_latency"  # Critical applications
    EMBB = "enhanced_mobile_broadband"   # High data rate
    MMTC = "massive_machine_type"        # IoT sensors
    HOLOGRAPHIC = "holographic_media"    # AR/VR/XR
    INDUSTRIAL = "industrial_automation" # Industry 4.0
    VEHICULAR = "vehicular_communications" # V2X

class EnvironmentType(Enum):
    INDOOR_OFFICE = "indoor_office"
    INDOOR_FACTORY = "indoor_factory"
    OUTDOOR_URBAN = "outdoor_urban"
    OUTDOOR_SUBURBAN = "outdoor_suburban"
    VEHICLE_INTERIOR = "vehicle_interior"
    DENSE_URBAN = "dense_urban"

@dataclass
class ServiceRequirements:
    """Service-specific requirements and priorities"""
    min_throughput_mbps: float
    max_latency_ms: float
    min_reliability: float
    max_range_m: float
    mobility_tolerance_kmh: float
    priority_weight: float

@dataclass
class EnvironmentConditions:
    """Current environment conditions affecting performance"""
    environment_type: EnvironmentType
    distance_m: float
    frequency_ghz: float
    los_probability: float
    weather_condition: str
    interference_level_db: float
    mobility_kmh: float
    temperature_c: float

class AdaptivePerformanceModel:
    """
    Adaptive performance model that sets realistic targets based on:
    - Physical propagation constraints
    - Service requirements
    - Environmental conditions
    - Technology limitations
    """
    
    def __init__(self):
        # Service class definitions
        self.service_classes = self._initialize_service_classes()
        
        # Environment-specific parameters
        self.environment_params = self._initialize_environment_params()
        
        # Technology-dependent maximum achievable rates
        self.technology_limits = self._initialize_technology_limits()
    
    def _initialize_service_classes(self) -> Dict[ServiceClass, ServiceRequirements]:
        """Initialize realistic service class requirements"""
        return {
            ServiceClass.URLC: ServiceRequirements(
                min_throughput_mbps=10.0,      # Moderate data rate
                max_latency_ms=1.0,            # Ultra-low latency
                min_reliability=0.99999,       # Five 9s
                max_range_m=10.0,              # Short range for reliability
                mobility_tolerance_kmh=30.0,   # Low mobility
                priority_weight=0.9            # High priority
            ),
            ServiceClass.EMBB: ServiceRequirements(
                min_throughput_mbps=1000.0,    # High data rate
                max_latency_ms=10.0,           # Moderate latency
                min_reliability=0.99,          # Two 9s
                max_range_m=50.0,              # Medium range
                mobility_tolerance_kmh=60.0,   # Medium mobility
                priority_weight=0.7            # Medium priority
            ),
            ServiceClass.MMTC: ServiceRequirements(
                min_throughput_mbps=0.1,       # Very low data rate
                max_latency_ms=100.0,          # High latency tolerance
                min_reliability=0.9,           # One 9
                max_range_m=100.0,             # Long range preferred
                mobility_tolerance_kmh=0.0,    # Stationary
                priority_weight=0.3            # Low priority
            ),
            ServiceClass.HOLOGRAPHIC: ServiceRequirements(
                min_throughput_mbps=10000.0,   # Very high data rate
                max_latency_ms=5.0,            # Low latency
                min_reliability=0.999,         # Three 9s
                max_range_m=20.0,              # Short range for quality
                mobility_tolerance_kmh=5.0,    # Near-stationary
                priority_weight=0.8            # High priority
            ),
            ServiceClass.INDUSTRIAL: ServiceRequirements(
                min_throughput_mbps=100.0,     # Medium-high data rate
                max_latency_ms=1.0,            # Ultra-low latency
                min_reliability=0.9999,        # Four 9s
                max_range_m=30.0,              # Factory coverage
                mobility_tolerance_kmh=10.0,   # Slow AGV movement
                priority_weight=0.85           # High priority
            ),
            ServiceClass.VEHICULAR: ServiceRequirements(
                min_throughput_mbps=50.0,      # Medium data rate
                max_latency_ms=5.0,            # Low latency
                min_reliability=0.999,         # Three 9s
                max_range_m=200.0,             # V2V communication
                mobility_tolerance_kmh=120.0,  # High mobility
                priority_weight=0.75           # High priority
            )
        }
    
    def _initialize_environment_params(self) -> Dict[EnvironmentType, Dict]:
        """Initialize environment-specific propagation parameters"""
        return {
            EnvironmentType.INDOOR_OFFICE: {
                'path_loss_exponent': 2.0,
                'shadow_fading_std_db': 3.0,
                'los_probability_at_10m': 0.9,
                'max_practical_range_m': 50.0,
                'typical_blockage_loss_db': 5.0,
                'mobility_penalty_factor': 1.0
            },
            EnvironmentType.INDOOR_FACTORY: {
                'path_loss_exponent': 2.2,
                'shadow_fading_std_db': 5.0,
                'los_probability_at_10m': 0.7,
                'max_practical_range_m': 100.0,
                'typical_blockage_loss_db': 10.0,
                'mobility_penalty_factor': 1.2
            },
            EnvironmentType.OUTDOOR_URBAN: {
                'path_loss_exponent': 3.5,
                'shadow_fading_std_db': 8.0,
                'los_probability_at_10m': 0.6,
                'max_practical_range_m': 200.0,
                'typical_blockage_loss_db': 20.0,
                'mobility_penalty_factor': 2.0
            },
            EnvironmentType.VEHICLE_INTERIOR: {
                'path_loss_exponent': 1.8,
                'shadow_fading_std_db': 2.0,
                'los_probability_at_10m': 0.95,
                'max_practical_range_m': 10.0,
                'typical_blockage_loss_db': 2.0,
                'mobility_penalty_factor': 1.5
            },
            EnvironmentType.DENSE_URBAN: {
                'path_loss_exponent': 4.0,
                'shadow_fading_std_db': 10.0,
                'los_probability_at_10m': 0.3,
                'max_practical_range_m': 100.0,
                'typical_blockage_loss_db': 30.0,
                'mobility_penalty_factor': 3.0
            }
        }
    
    def _initialize_technology_limits(self) -> Dict[str, Dict]:
        """Initialize technology-dependent achievable rates"""
        return {
            'sub_100ghz': {
                'max_spectral_efficiency_bps_hz': 8.0,   # Realistic MIMO
                'max_practical_snr_db': 30.0,
                'implementation_efficiency': 0.7
            },
            '100_300ghz': {
                'max_spectral_efficiency_bps_hz': 6.0,   # THz challenges
                'max_practical_snr_db': 25.0,
                'implementation_efficiency': 0.5
            },
            'above_300ghz': {
                'max_spectral_efficiency_bps_hz': 4.0,   # Severe limitations
                'max_practical_snr_db': 20.0,
                'implementation_efficiency': 0.3
            }
        }
    
    def calculate_adaptive_targets(self, service: ServiceClass, 
                                 conditions: EnvironmentConditions) -> Dict[str, float]:
        """
        Calculate adaptive performance targets based on service and conditions.
        
        Args:
            service: Service class with requirements
            conditions: Current environment conditions
            
        Returns:
            Dictionary with achievable performance targets
        """
        service_req = self.service_classes[service]
        env_params = self.environment_params[conditions.environment_type]
        
        # Get technology limits for frequency
        if conditions.frequency_ghz < 100:
            tech_limits = self.technology_limits['sub_100ghz']
        elif conditions.frequency_ghz < 300:
            tech_limits = self.technology_limits['100_300ghz']
        else:
            tech_limits = self.technology_limits['above_300ghz']
        
        # Calculate achievable throughput
        achievable_throughput = self._calculate_achievable_throughput(
            conditions, tech_limits, env_params
        )
        
        # Calculate achievable latency
        achievable_latency = self._calculate_achievable_latency(
            conditions, env_params
        )
        
        # Calculate achievable reliability
        achievable_reliability = self._calculate_achievable_reliability(
            conditions, env_params
        )
        
        # Calculate achievable range
        achievable_range = self._calculate_achievable_range(
            conditions, env_params, tech_limits
        )
        
        # Apply service-specific adjustments
        targets = self._apply_service_constraints(
            service_req, achievable_throughput, achievable_latency,
            achievable_reliability, achievable_range
        )
        
        return targets
    
    def _calculate_achievable_throughput(self, conditions: EnvironmentConditions,
                                       tech_limits: Dict, env_params: Dict) -> float:
        """Calculate achievable throughput considering all constraints"""
        
        # Start with Shannon limit
        # Estimate SNR based on distance and environment
        fspl_db = 20 * math.log10(conditions.distance_m) + 20 * math.log10(conditions.frequency_ghz) + 92.45
        path_loss_db = fspl_db + env_params['path_loss_exponent'] * 10 * math.log10(conditions.distance_m / 1.0)
        
        # Add environment-specific losses
        blockage_loss = env_params['typical_blockage_loss_db'] * (1 - conditions.los_probability)
        weather_loss = self._get_weather_loss(conditions.weather_condition, conditions.frequency_ghz)
        
        total_loss_db = path_loss_db + blockage_loss + weather_loss + conditions.interference_level_db
        
        # Assume 30 dBm TX power, -90 dBm noise floor
        snr_db = 30 - total_loss_db - (-90)
        snr_db = min(snr_db, tech_limits['max_practical_snr_db'])
        
        if snr_db <= 0:
            return 0.0
        
        # Shannon capacity with practical limits
        snr_linear = 10**(snr_db / 10)
        spectral_efficiency = min(
            math.log2(1 + snr_linear),
            tech_limits['max_spectral_efficiency_bps_hz']
        )
        
        # Estimate usable bandwidth (realistic spectrum windows)
        if conditions.frequency_ghz < 60:
            usable_bandwidth_ghz = 0.1  # 100 MHz typical
        elif conditions.frequency_ghz < 100:
            usable_bandwidth_ghz = 0.8  # 800 MHz for mmWave
        elif conditions.frequency_ghz < 200:
            usable_bandwidth_ghz = 2.0  # 2 GHz for lower sub-THz
        elif conditions.frequency_ghz < 400:
            usable_bandwidth_ghz = 5.0  # 5 GHz for THz windows
        else:
            usable_bandwidth_ghz = 1.0  # Very limited at high THz
        
        # Calculate throughput
        throughput_bps = spectral_efficiency * usable_bandwidth_ghz * 1e9
        throughput_bps *= tech_limits['implementation_efficiency']
        
        # Apply mobility penalty
        mobility_penalty = 1.0 / (1 + 0.01 * conditions.mobility_kmh * env_params['mobility_penalty_factor'])
        throughput_bps *= mobility_penalty
        
        return throughput_bps / 1e6  # Convert to Mbps
    
    def _calculate_achievable_latency(self, conditions: EnvironmentConditions,
                                    env_params: Dict) -> float:
        """Calculate achievable latency including processing and propagation"""
        
        # Propagation delay
        prop_delay_ms = conditions.distance_m / 3e8 * 1000
        
        # Processing delay (depends on complexity)
        if conditions.frequency_ghz > 300:
            processing_delay_ms = 2.0  # Higher processing for THz
        else:
            processing_delay_ms = 1.0
        
        # Queueing delay (depends on mobility and environment)
        base_queueing_ms = 0.5
        mobility_factor = 1 + 0.01 * conditions.mobility_kmh
        environment_factor = env_params['mobility_penalty_factor']
        queueing_delay_ms = base_queueing_ms * mobility_factor * environment_factor
        
        total_latency_ms = prop_delay_ms + processing_delay_ms + queueing_delay_ms
        
        return total_latency_ms
    
    def _calculate_achievable_reliability(self, conditions: EnvironmentConditions,
                                        env_params: Dict) -> float:
        """Calculate achievable reliability based on environment"""
        
        # Base reliability depends on environment
        base_reliability = 0.99
        
        # LOS probability impact
        los_impact = 0.005 * (1 - conditions.los_probability)
        
        # Distance impact (longer distance = lower reliability)
        distance_impact = 0.001 * (conditions.distance_m / 100.0)
        
        # Mobility impact
        mobility_impact = 0.0001 * conditions.mobility_kmh
        
        # Weather impact
        weather_impact = self._get_weather_reliability_impact(conditions.weather_condition)
        
        reliability = base_reliability - los_impact - distance_impact - mobility_impact - weather_impact
        
        return max(reliability, 0.5)  # Minimum 50% reliability
    
    def _calculate_achievable_range(self, conditions: EnvironmentConditions,
                                  env_params: Dict, tech_limits: Dict) -> float:
        """Calculate maximum achievable range for reliable communication"""
        
        # Start with environment limit
        max_range = env_params['max_practical_range_m']
        
        # Frequency penalty (higher frequency = shorter range)
        freq_penalty = min(1.0, 100.0 / conditions.frequency_ghz)
        max_range *= freq_penalty
        
        # Weather penalty
        weather_penalty = self._get_weather_range_penalty(conditions.weather_condition)
        max_range *= weather_penalty
        
        return max_range
    
    def _apply_service_constraints(self, service_req: ServiceRequirements,
                                 achievable_throughput: float, achievable_latency: float,
                                 achievable_reliability: float, achievable_range: float) -> Dict[str, float]:
        """Apply service-specific constraints to achievable performance"""
        
        return {
            'target_throughput_mbps': min(achievable_throughput, service_req.min_throughput_mbps * 10),  # Allow 10x headroom
            'target_latency_ms': max(achievable_latency, service_req.max_latency_ms),
            'target_reliability': min(achievable_reliability, service_req.min_reliability),
            'target_range_m': min(achievable_range, service_req.max_range_m),
            'service_priority': service_req.priority_weight,
            'achievable_throughput_mbps': achievable_throughput,
            'achievable_latency_ms': achievable_latency,
            'achievable_reliability': achievable_reliability,
            'achievable_range_m': achievable_range
        }
    
    def _get_weather_loss(self, weather: str, freq_ghz: float) -> float:
        """Get weather-dependent loss in dB"""
        weather_losses = {
            'clear': 0.0,
            'light_rain': 2.0 * (freq_ghz / 100),
            'moderate_rain': 8.0 * (freq_ghz / 100),
            'heavy_rain': 20.0 * (freq_ghz / 100),
            'fog': 5.0 * (freq_ghz / 100),
            'snow': 3.0 * (freq_ghz / 100)
        }
        return weather_losses.get(weather, 0.0)
    
    def _get_weather_reliability_impact(self, weather: str) -> float:
        """Get weather impact on reliability"""
        impacts = {
            'clear': 0.0,
            'light_rain': 0.01,
            'moderate_rain': 0.05,
            'heavy_rain': 0.15,
            'fog': 0.02,
            'snow': 0.03
        }
        return impacts.get(weather, 0.0)
    
    def _get_weather_range_penalty(self, weather: str) -> float:
        """Get weather penalty on range (multiplicative factor)"""
        penalties = {
            'clear': 1.0,
            'light_rain': 0.9,
            'moderate_rain': 0.7,
            'heavy_rain': 0.4,
            'fog': 0.8,
            'snow': 0.85
        }
        return penalties.get(weather, 1.0)
    
    def get_compliance_score(self, actual_performance: Dict[str, float],
                           targets: Dict[str, float]) -> float:
        """Calculate compliance score based on achieved vs target performance"""
        
        # Weighted compliance for each metric
        throughput_compliance = min(1.0, actual_performance.get('throughput_mbps', 0) / 
                                  max(targets['target_throughput_mbps'], 0.1))
        
        latency_compliance = min(1.0, targets['target_latency_ms'] / 
                               max(actual_performance.get('latency_ms', float('inf')), 0.1))
        
        reliability_compliance = min(1.0, actual_performance.get('reliability', 0) / 
                                   targets['target_reliability'])
        
        # Weighted average
        total_compliance = (
            0.4 * throughput_compliance +
            0.3 * latency_compliance +
            0.3 * reliability_compliance
        )
        
        return min(total_compliance * 100, 100.0)  # Return as percentage

def test_adaptive_targets():
    """Test adaptive performance targeting"""
    model = AdaptivePerformanceModel()
    
    # Test scenarios
    scenarios = [
        {
            'name': 'Indoor Office - Short Range',
            'service': ServiceClass.EMBB,
            'conditions': EnvironmentConditions(
                environment_type=EnvironmentType.INDOOR_OFFICE,
                distance_m=5.0,
                frequency_ghz=140.0,
                los_probability=0.9,
                weather_condition='clear',
                interference_level_db=3.0,
                mobility_kmh=3.0,
                temperature_c=20.0
            )
        },
        {
            'name': 'Outdoor Urban - Medium Range',
            'service': ServiceClass.EMBB,
            'conditions': EnvironmentConditions(
                environment_type=EnvironmentType.OUTDOOR_URBAN,
                distance_m=50.0,
                frequency_ghz=300.0,
                los_probability=0.6,
                weather_condition='light_rain',
                interference_level_db=8.0,
                mobility_kmh=30.0,
                temperature_c=15.0
            )
        },
        {
            'name': 'THz Laboratory - Ultra High Rate',
            'service': ServiceClass.HOLOGRAPHIC,
            'conditions': EnvironmentConditions(
                environment_type=EnvironmentType.INDOOR_OFFICE,
                distance_m=2.0,
                frequency_ghz=600.0,
                los_probability=0.99,
                weather_condition='clear',
                interference_level_db=1.0,
                mobility_kmh=0.0,
                temperature_c=22.0
            )
        }
    ]
    
    print("=== ADAPTIVE PERFORMANCE TARGETS ===")
    print("Scenario                    | Freq  | Target T'put | Achievable | Target Lat | Compliance")
    print("                           | (GHz) |    (Mbps)    |   (Mbps)   |    (ms)    |    (%)")
    print("-" * 95)
    
    for scenario in scenarios:
        targets = model.calculate_adaptive_targets(scenario['service'], scenario['conditions'])
        
        # Simulate actual performance (slightly below achievable)
        actual_performance = {
            'throughput_mbps': targets['achievable_throughput_mbps'] * 0.8,
            'latency_ms': targets['achievable_latency_ms'] * 1.1,
            'reliability': targets['achievable_reliability'] * 0.95
        }
        
        compliance = model.get_compliance_score(actual_performance, targets)
        
        print(f"{scenario['name'][:26]:26s} | {scenario['conditions'].frequency_ghz:5.0f} | "
              f"{targets['target_throughput_mbps']:10.0f} | {targets['achievable_throughput_mbps']:9.0f} | "
              f"{targets['target_latency_ms']:8.1f} | {compliance:8.1f}")
    
    print(f"\n=== COMPARISON: Old vs New Targets ===")
    old_targets = {'sub_thz_300': 1000000, 'thz_600': 2000000}  # Old unrealistic Mbps
    
    for scenario in scenarios[:2]:  # Test first two scenarios
        targets = model.calculate_adaptive_targets(scenario['service'], scenario['conditions'])
        freq_key = 'sub_thz_300' if scenario['conditions'].frequency_ghz < 400 else 'thz_600'
        old_target = old_targets.get(freq_key, 1000000)
        
        reduction_factor = old_target / max(targets['achievable_throughput_mbps'], 1)
        
        print(f"{scenario['name'][:20]:20s} | Old: {old_target/1000:6.0f} Gbps | "
              f"New: {targets['achievable_throughput_mbps']/1000:6.1f} Gbps | "
              f"Reduction: {reduction_factor:4.0f}x")

if __name__ == "__main__":
    test_adaptive_targets()
