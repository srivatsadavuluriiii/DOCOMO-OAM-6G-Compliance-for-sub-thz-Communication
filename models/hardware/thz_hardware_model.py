#!/usr/bin/env python3
"""
Realistic THz Hardware Constraints Model
Implementation of frequency-dependent power limits, phase noise, thermal effects,
and other hardware impairments for 6G THz systems.

Based on current state-of-the-art THz technology and projected 2030 capabilities.
"""

import numpy as np
import math
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class AmplifierType(Enum):
    SILICON_CMOS = "silicon_cmos"      # Up to ~100 GHz
    SILICON_BICMOS = "silicon_bicmos"  # Up to ~200 GHz  
    GAAS_PHEMT = "gaas_phemt"         # Up to ~300 GHz
    INP_HEMT = "inp_hemt"             # Up to ~500 GHz
    UTC_PD = "utc_photodiode"         # Photonic, >500 GHz
    QCL = "quantum_cascade_laser"     # >1000 GHz

@dataclass
class HardwareSpecs:
    """Hardware specifications for THz transceiver"""
    amplifier_type: AmplifierType
    max_tx_power_dbm: float
    noise_figure_db: float
    phase_noise_1khz_dbc_hz: float
    bandwidth_ghz: float
    efficiency_percent: float
    cost_factor: float  # Relative cost (1.0 = baseline)

class THzHardwareModel:
    """
    Realistic THz hardware constraints model.
    
    Key constraints modeled:
    - Frequency-dependent TX power limitations
    - Phase noise scaling with frequency
    - Power amplifier efficiency rolloff  
    - Thermal noise and drift
    - Technology-specific limitations
    """
    
    def __init__(self):
        # Technology roadmap for THz hardware (2024-2030)
        self.technology_specs = self._initialize_technology_specs()
        
        # Physical constants
        self.boltzmann_k = 1.380649e-23  # J/K
        self.T0 = 290.0  # Standard temperature (K)
    
    def _initialize_technology_specs(self) -> Dict[AmplifierType, Dict]:
        """Initialize realistic hardware specifications by technology"""
        return {
            AmplifierType.SILICON_CMOS: {
                'freq_range_ghz': (1, 110),
                'max_power_dbm_at_min_freq': 25,
                'power_rolloff_db_per_decade': 15,
                'noise_figure_db': 8.0,
                'phase_noise_floor_dbc_hz': -100,
                'efficiency_percent': 15,
                'cost_factor': 1.0
            },
            AmplifierType.SILICON_BICMOS: {
                'freq_range_ghz': (50, 220),
                'max_power_dbm_at_min_freq': 22,
                'power_rolloff_db_per_decade': 12,
                'noise_figure_db': 7.0,
                'phase_noise_floor_dbc_hz': -105,
                'efficiency_percent': 20,
                'cost_factor': 2.0
            },
            AmplifierType.GAAS_PHEMT: {
                'freq_range_ghz': (100, 350),
                'max_power_dbm_at_min_freq': 20,
                'power_rolloff_db_per_decade': 10,
                'noise_figure_db': 6.0,
                'phase_noise_floor_dbc_hz': -110,
                'efficiency_percent': 25,
                'cost_factor': 5.0
            },
            AmplifierType.INP_HEMT: {
                'freq_range_ghz': (200, 600),
                'max_power_dbm_at_min_freq': 15,
                'power_rolloff_db_per_decade': 8,
                'noise_figure_db': 5.0,
                'phase_noise_floor_dbc_hz': -115,
                'efficiency_percent': 30,
                'cost_factor': 10.0
            },
            AmplifierType.UTC_PD: {
                'freq_range_ghz': (300, 1000),
                'max_power_dbm_at_min_freq': 10,
                'power_rolloff_db_per_decade': 6,
                'noise_figure_db': 15.0,  # Higher for photonic
                'phase_noise_floor_dbc_hz': -120,
                'efficiency_percent': 5,   # Low efficiency
                'cost_factor': 20.0
            },
            AmplifierType.QCL: {
                'freq_range_ghz': (600, 3000),
                'max_power_dbm_at_min_freq': 5,
                'power_rolloff_db_per_decade': 4,
                'noise_figure_db': 20.0,
                'phase_noise_floor_dbc_hz': -125,
                'efficiency_percent': 2,
                'cost_factor': 50.0
            }
        }
    
    def get_optimal_technology(self, freq_ghz: float) -> AmplifierType:
        """Select optimal technology for given frequency"""
        best_tech = AmplifierType.SILICON_CMOS
        best_power = -float('inf')
        
        for tech, specs in self.technology_specs.items():
            freq_min, freq_max = specs['freq_range_ghz']
            
            if freq_min <= freq_ghz <= freq_max:
                # Calculate available power for this technology
                power = self.get_max_tx_power_dbm(freq_ghz, tech)
                
                # Consider efficiency and cost (simplified metric)
                performance_metric = power + 0.1 * specs['efficiency_percent'] - 0.01 * specs['cost_factor']
                
                if performance_metric > best_power:
                    best_power = performance_metric
                    best_tech = tech
        
        return best_tech
    
    def get_max_tx_power_dbm(self, freq_ghz: float, tech: Optional[AmplifierType] = None) -> float:
        """
        Get maximum realistic transmit power for given frequency.
        
        Args:
            freq_ghz: Frequency in GHz
            tech: Specific technology (auto-select if None)
            
        Returns:
            Maximum TX power in dBm
        """
        if tech is None:
            tech = self.get_optimal_technology(freq_ghz)
        
        specs = self.technology_specs[tech]
        freq_min, freq_max = specs['freq_range_ghz']
        
        # Check if frequency is in range
        if not (freq_min <= freq_ghz <= freq_max):
            return -float('inf')  # Technology can't operate at this frequency
        
        # Calculate power with frequency rolloff
        base_power = specs['max_power_dbm_at_min_freq']
        rolloff = specs['power_rolloff_db_per_decade']
        
        # Power decreases with frequency
        freq_decades = math.log10(freq_ghz / freq_min)
        max_power_dbm = base_power - rolloff * freq_decades
        
        # Add temperature derating (1 dB reduction per 10°C above 25°C)
        temperature_derating = 0.0  # Assume 25°C operation
        
        # Add process variation (±2 dB typical)
        process_variation = 0.0  # Use nominal values
        
        return max_power_dbm + temperature_derating + process_variation
    
    def calculate_phase_noise(self, freq_ghz: float, offset_hz: float, 
                            tech: Optional[AmplifierType] = None) -> float:
        """
        Calculate phase noise at given offset frequency.
        
        Args:
            freq_ghz: Carrier frequency in GHz
            offset_hz: Offset frequency in Hz
            tech: Technology type
            
        Returns:
            Phase noise in dBc/Hz
        """
        if tech is None:
            tech = self.get_optimal_technology(freq_ghz)
        
        specs = self.technology_specs[tech]
        phase_noise_floor = specs['phase_noise_floor_dbc_hz']
        
        # Phase noise model: L(f) = L_floor + 20*log10(fc/10GHz) - 20*log10(f_offset/1kHz)
        # Higher carrier frequency increases phase noise
        carrier_factor = 20 * math.log10(freq_ghz / 10.0)
        
        # 1/f² rolloff from carrier
        offset_factor = -20 * math.log10(max(offset_hz, 1.0) / 1000.0)
        
        phase_noise_dbc_hz = phase_noise_floor + carrier_factor + offset_factor
        
        return phase_noise_dbc_hz
    
    def calculate_noise_figure(self, freq_ghz: float, temp_c: float = 25.0,
                             tech: Optional[AmplifierType] = None) -> float:
        """
        Calculate receiver noise figure including temperature effects.
        
        Args:
            freq_ghz: Frequency in GHz
            temp_c: Temperature in Celsius
            tech: Technology type
            
        Returns:
            Noise figure in dB
        """
        if tech is None:
            tech = self.get_optimal_technology(freq_ghz)
        
        specs = self.technology_specs[tech]
        base_nf = specs['noise_figure_db']
        
        # Noise figure increases with frequency (approximately)
        freq_factor = 0.01 * freq_ghz  # 0.01 dB per GHz
        
        # Temperature coefficient (typically +0.02 dB/°C)
        temp_factor = 0.02 * (temp_c - 25.0)
        
        total_nf = base_nf + freq_factor + temp_factor
        
        return max(total_nf, 0.5)  # Minimum 0.5 dB NF
    
    def calculate_power_consumption(self, tx_power_dbm: float, freq_ghz: float,
                                  tech: Optional[AmplifierType] = None) -> Dict[str, float]:
        """
        Calculate realistic power consumption breakdown.
        
        Args:
            tx_power_dbm: Transmit power in dBm
            freq_ghz: Frequency in GHz
            tech: Technology type
            
        Returns:
            Power consumption breakdown in Watts
        """
        if tech is None:
            tech = self.get_optimal_technology(freq_ghz)
        
        specs = self.technology_specs[tech]
        efficiency = specs['efficiency_percent'] / 100.0
        
        # Output power in Watts
        tx_power_w = 10**(tx_power_dbm / 10) / 1000.0
        
        # PA power consumption
        pa_power_w = tx_power_w / efficiency if efficiency > 0 else 100.0
        
        # Frequency synthesis (LO + PLLs)
        # Power scales with frequency due to higher division ratios
        lo_power_w = 0.5 + 0.001 * freq_ghz
        
        # DSP and baseband processing
        # Scales with bandwidth
        bandwidth_estimate = min(freq_ghz * 0.1, 10.0)  # Estimate 10% fractional BW
        dsp_power_w = 2.0 + 0.1 * bandwidth_estimate
        
        # Control and bias circuits
        control_power_w = 1.0
        
        # Cooling (for higher power systems)
        cooling_power_w = max(0, (pa_power_w - 1.0) * 0.5)  # 50% overhead for >1W PA
        
        return {
            'pa_power_w': pa_power_w,
            'lo_power_w': lo_power_w,
            'dsp_power_w': dsp_power_w,
            'control_power_w': control_power_w,
            'cooling_power_w': cooling_power_w,
            'total_power_w': pa_power_w + lo_power_w + dsp_power_w + control_power_w + cooling_power_w
        }
    
    def calculate_thermal_noise_power(self, bandwidth_hz: float, temp_k: float = 290.0,
                                    noise_figure_db: float = 10.0) -> float:
        """
        Calculate thermal noise power.
        
        Args:
            bandwidth_hz: Noise bandwidth in Hz
            temp_k: Temperature in Kelvin
            noise_figure_db: Receiver noise figure in dB
            
        Returns:
            Noise power in Watts
        """
        # Thermal noise power: N = k*T*B*NF
        noise_factor = 10**(noise_figure_db / 10)
        noise_power_w = self.boltzmann_k * temp_k * bandwidth_hz * noise_factor
        
        return noise_power_w
    
    def get_hardware_specifications(self, freq_ghz: float) -> HardwareSpecs:
        """Get complete hardware specifications for given frequency"""
        tech = self.get_optimal_technology(freq_ghz)
        specs = self.technology_specs[tech]
        
        return HardwareSpecs(
            amplifier_type=tech,
            max_tx_power_dbm=self.get_max_tx_power_dbm(freq_ghz, tech),
            noise_figure_db=self.calculate_noise_figure(freq_ghz, tech=tech),
            phase_noise_1khz_dbc_hz=self.calculate_phase_noise(freq_ghz, 1000.0, tech),
            bandwidth_ghz=min(freq_ghz * 0.1, 20.0),  # Conservative estimate
            efficiency_percent=specs['efficiency_percent'],
            cost_factor=specs['cost_factor']
        )

def test_hardware_constraints():
    """Test realistic hardware constraints"""
    model = THzHardwareModel()
    
    test_frequencies = [28, 60, 100, 140, 220, 300, 400, 600]
    
    print("=== REALISTIC THz HARDWARE CONSTRAINTS ===")
    print("Freq | Tech      | Max Pwr | NF  | Phase Noise | Efficiency | Total Power")
    print("(GHz)|           | (dBm)   | (dB)| @1kHz(dBc/Hz)|    (%)     |    (W)")
    print("-" * 80)
    
    for freq in test_frequencies:
        specs = model.get_hardware_specifications(freq)
        power_breakdown = model.calculate_power_consumption(specs.max_tx_power_dbm, freq)
        
        tech_short = specs.amplifier_type.value.replace('_', ' ')[:9]
        
        print(f"{freq:4.0f} | {tech_short:9s} | {specs.max_tx_power_dbm:7.1f} | "
              f"{specs.noise_figure_db:3.1f} | {specs.phase_noise_1khz_dbc_hz:11.1f} | "
              f"{specs.efficiency_percent:8.1f} | {power_breakdown['total_power_w']:9.1f}")
    
    print(f"\n=== COMPARISON: Current vs Previous Implementation ===")
    print("Frequency | Old Power | New Power | Difference")
    print("   (GHz)  |   (dBm)   |   (dBm)   |    (dB)")
    print("-" * 45)
    
    old_powers = {28: 40, 60: 35, 100: 30, 220: 25, 300: 40, 600: 40}  # Previous unrealistic values
    
    for freq in [28, 60, 100, 220, 300, 600]:
        new_power = model.get_max_tx_power_dbm(freq)
        old_power = old_powers[freq]
        difference = new_power - old_power
        
        print(f"{freq:7.0f} | {old_power:9.0f} | {new_power:9.1f} | {difference:8.1f}")

if __name__ == "__main__":
    test_hardware_constraints()
