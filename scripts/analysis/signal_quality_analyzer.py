#!/usr/bin/env python3
"""
DOCOMO 6G Signal Quality Analyzer
=================================

Comprehensive analysis of:
- Per-mode SINR measurements
- Error Vector Magnitude (EVM) per constellation
- Constellation diagrams
- Spectrum occupancy (waterfall/FFT)

Author: DOCOMO 6G Research
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from scipy import signal
from scipy.fft import fft, fftfreq
import yaml
import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from environment.docomo_6g_env import DOCOMO_6G_Environment
from environment.unified_physics_engine import UnifiedPhysicsEngine

class SignalQualityAnalyzer:
    """Comprehensive signal quality and spectrum analysis for DOCOMO 6G systems"""
    
    def __init__(self, config_path: str):
        """Initialize analyzer with configuration"""
        self.config_path = config_path
        self.load_config()
        self.physics_engine = UnifiedPhysicsEngine(self.config)
        
        # Analysis parameters
        self.constellation_points = 1024  # Points per constellation
        self.fft_size = 4096             # FFT analysis size
        self.waterfall_duration = 100    # Time samples for waterfall
        
        # Modulation schemes and their theoretical EVM limits
        self.modulation_schemes = {
            'BPSK': {'order': 2, 'evm_limit': 32.0},     # % RMS EVM
            'QPSK': {'order': 4, 'evm_limit': 17.5},
            '16QAM': {'order': 16, 'evm_limit': 12.5},
            '64QAM': {'order': 64, 'evm_limit': 8.0},
            '256QAM': {'order': 256, 'evm_limit': 3.5},
            '1024QAM': {'order': 1024, 'evm_limit': 1.8}
        }
        
    def load_config(self):
        """Load DOCOMO 6G configuration"""
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Extract frequency bands
        self.frequency_bands = self.config['docomo_6g_system']['frequency_bands']
        
    def calculate_per_mode_sinr(self, frequency_hz: float, distance_m: float, 
                               base_sinr_db: float, num_modes: int) -> Dict:
        """Calculate SINR for each OAM mode including crosstalk effects"""
        
        results = {
            'frequency_ghz': frequency_hz / 1e9,
            'distance_m': distance_m,
            'base_sinr_db': base_sinr_db,
            'num_modes': num_modes,
            'mode_analysis': []
        }
        
        # Mode-dependent parameters
        for mode_idx in range(1, num_modes + 1):
            # Mode-specific degradation factors
            mode_crosstalk = self._calculate_mode_crosstalk(mode_idx, num_modes)
            pointing_error = self._calculate_pointing_error(mode_idx, frequency_hz)
            atmospheric_fade = self._calculate_atmospheric_fade(mode_idx, frequency_hz, distance_m)
            
            # Effective SINR for this mode
            mode_sinr_db = base_sinr_db - mode_crosstalk - pointing_error - atmospheric_fade
            
            # EVM calculation based on SINR
            evm_rms = self._sinr_to_evm(mode_sinr_db)
            
            # Determine optimal modulation for this SINR
            optimal_mod = self._select_modulation(mode_sinr_db, evm_rms)
            
            mode_data = {
                'mode_number': mode_idx,
                'sinr_db': float(mode_sinr_db),
                'evm_rms_percent': float(evm_rms),
                'crosstalk_db': float(mode_crosstalk),
                'pointing_error_db': float(pointing_error),
                'atmospheric_fade_db': float(atmospheric_fade),
                'optimal_modulation': optimal_mod,
                'constellation_snr': float(mode_sinr_db + 3)  # Processing gain
            }
            
            results['mode_analysis'].append(mode_data)
        
        # Summary statistics
        sinr_values = [m['sinr_db'] for m in results['mode_analysis']]
        evm_values = [m['evm_rms_percent'] for m in results['mode_analysis']]
        
        results['summary'] = {
            'mean_sinr_db': float(np.mean(sinr_values)),
            'std_sinr_db': float(np.std(sinr_values)),
            'min_sinr_db': float(np.min(sinr_values)),
            'max_sinr_db': float(np.max(sinr_values)),
            'mean_evm_percent': float(np.mean(evm_values)),
            'modes_above_20db': int(sum(1 for s in sinr_values if s > 20)),
            'usable_modes': int(sum(1 for s in sinr_values if s > 10))
        }
        
        return results
    
    def _calculate_mode_crosstalk(self, mode_idx: int, total_modes: int) -> float:
        """Calculate crosstalk between OAM modes"""
        # Higher modes and more total modes increase crosstalk
        base_crosstalk = 0.5 * np.log10(mode_idx + 1)  # 0.5 dB per decade
        density_penalty = 0.2 * np.log10(total_modes)   # Mode density effect
        
        # Random variations due to alignment and atmospheric effects
        alignment_error = np.random.normal(0, 0.3)  # ±0.3 dB std
        
        return base_crosstalk + density_penalty + abs(alignment_error)
    
    def _calculate_pointing_error(self, mode_idx: int, frequency_hz: float) -> float:
        """Calculate pointing error impact on specific mode"""
        # Higher modes are more sensitive to pointing errors
        mode_sensitivity = 0.1 * mode_idx  # 0.1 dB per mode
        
        # Frequency-dependent beam width (smaller at higher freq)
        freq_factor = 0.5 * np.log10(frequency_hz / 28e9)  # vs 28 GHz reference
        
        return mode_sensitivity + freq_factor
    
    def _calculate_atmospheric_fade(self, mode_idx: int, frequency_hz: float, distance_m: float) -> float:
        """Calculate atmospheric fade for specific mode"""
        # Base atmospheric loss
        if frequency_hz >= 300e9:      # THz
            base_loss = 2.0 + 0.1 * distance_m
        elif frequency_hz >= 100e9:    # Sub-THz  
            base_loss = 1.0 + 0.05 * distance_m
        else:                          # mmWave
            base_loss = 0.5 + 0.02 * distance_m
        
        # Mode-dependent scattering
        mode_factor = 0.05 * mode_idx
        
        return base_loss + mode_factor
    
    def _sinr_to_evm(self, sinr_db: float) -> float:
        """Convert SINR to EVM (Error Vector Magnitude)"""
        # Theoretical relationship: EVM% ≈ 100 / sqrt(10^(SINR/10))
        sinr_linear = 10 ** (sinr_db / 10.0)
        evm_rms_fraction = 1.0 / np.sqrt(sinr_linear)
        evm_rms_percent = evm_rms_fraction * 100
        
        # Add implementation losses (typically 1-3 dB)
        implementation_loss = 2.0  # 2 dB typical
        practical_evm = evm_rms_percent * 10**(implementation_loss/20)
        
        return min(practical_evm, 100)  # Cap at 100%
    
    def _select_modulation(self, sinr_db: float, evm_percent: float) -> str:
        """Select optimal modulation based on SINR and EVM"""
        for mod_name, mod_info in reversed(list(self.modulation_schemes.items())):
            if sinr_db >= self._required_sinr(mod_name) and evm_percent <= mod_info['evm_limit']:
                return mod_name
        return 'BPSK'  # Fallback to most robust
    
    def _required_sinr(self, modulation: str) -> float:
        """Required SINR for modulation scheme (at 1e-3 BER)"""
        sinr_requirements = {
            'BPSK': 7.0,     # dB
            'QPSK': 10.0,
            '16QAM': 16.0,
            '64QAM': 22.0,
            '256QAM': 28.0,
            '1024QAM': 34.0
        }
        return sinr_requirements.get(modulation, 50.0)
    
    def generate_constellation_diagram(self, sinr_db: float, modulation: str, 
                                     mode_idx: int = 1) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Generate constellation diagram with realistic noise and impairments"""
        
        mod_info = self.modulation_schemes[modulation]
        order = mod_info['order']
        
        # Generate ideal constellation points
        if modulation == 'BPSK':
            ideal_i = np.array([-1, 1])
            ideal_q = np.array([0, 0])
        elif modulation == 'QPSK':
            ideal_i = np.array([-1, 1, -1, 1]) / np.sqrt(2)
            ideal_q = np.array([-1, -1, 1, 1]) / np.sqrt(2)
        else:  # QAM
            sqrt_order = int(np.sqrt(order))
            levels = np.linspace(-1, 1, sqrt_order)
            ideal_i, ideal_q = np.meshgrid(levels, levels)
            ideal_i = ideal_i.flatten()
            ideal_q = ideal_q.flatten()
        
        # Generate received constellation with impairments
        num_symbols = self.constellation_points
        symbol_indices = np.random.randint(0, len(ideal_i), num_symbols)
        
        # Start with ideal points
        rx_i = ideal_i[symbol_indices]
        rx_q = ideal_q[symbol_indices]
        
        # Add AWGN based on SINR
        noise_power = 10**(-sinr_db/10)
        noise_i = np.random.normal(0, np.sqrt(noise_power/2), num_symbols)
        noise_q = np.random.normal(0, np.sqrt(noise_power/2), num_symbols)
        
        # Add implementation impairments
        # 1. Phase noise (higher for higher frequencies)
        phase_noise_std = 0.02 + 0.01 * mode_idx  # radians
        phase_error = np.random.normal(0, phase_noise_std, num_symbols)
        
        # 2. Amplitude imbalance
        amp_imbalance = 1 + np.random.normal(0, 0.02)  # 2% std
        
        # 3. Quadrature error
        quad_error = np.random.normal(0, 0.01)  # radians
        
        # Apply impairments
        rx_i_impaired = (rx_i + noise_i) * amp_imbalance
        rx_q_impaired = (rx_q + noise_q) * np.cos(quad_error) + (rx_i + noise_i) * np.sin(quad_error)
        
        # Add phase noise rotation
        rx_i_final = rx_i_impaired * np.cos(phase_error) - rx_q_impaired * np.sin(phase_error)
        rx_q_final = rx_i_impaired * np.sin(phase_error) + rx_q_impaired * np.cos(phase_error)
        
        # Calculate actual EVM
        ideal_power = np.mean(ideal_i[symbol_indices]**2 + ideal_q[symbol_indices]**2)
        error_power = np.mean((rx_i_final - ideal_i[symbol_indices])**2 + 
                             (rx_q_final - ideal_q[symbol_indices])**2)
        actual_evm = 100 * np.sqrt(error_power / ideal_power)
        
        constellation_data = {
            'modulation': modulation,
            'order': order,
            'sinr_db': sinr_db,
            'evm_rms_percent': actual_evm,
            'num_symbols': num_symbols,
            'mode_index': mode_idx,
            'ideal_points': (ideal_i, ideal_q),
            'phase_noise_std_deg': np.degrees(phase_noise_std),
            'amplitude_imbalance_db': 20*np.log10(amp_imbalance)
        }
        
        return rx_i_final, rx_q_final, constellation_data
    
    def generate_spectrum_occupancy(self, frequency_bands: List[str], 
                                  duration_ms: float = 10.0) -> Dict:
        """Generate realistic spectrum occupancy data"""
        
        sample_rate = 100e9  # 100 GHz sampling rate
        samples = int(duration_ms * 1e-3 * sample_rate)
        time_axis = np.linspace(0, duration_ms, samples)
        
        spectrum_data = {
            'sample_rate_ghz': sample_rate / 1e9,
            'duration_ms': duration_ms,
            'time_axis_ms': time_axis * 1000,
            'bands': {}
        }
        
        for band_name in frequency_bands:
            if band_name not in self.frequency_bands:
                continue
                
            band_info = self.frequency_bands[band_name]
            center_freq = float(band_info['frequency'])
            bandwidth = float(band_info['bandwidth'])
            
            # Generate band signal
            band_signal = self._generate_band_signal(
                center_freq, bandwidth, sample_rate, samples
            )
            
            # Calculate FFT
            fft_data = fft(band_signal)
            freq_axis = fftfreq(samples, 1/sample_rate)
            
            # Generate waterfall (time-frequency)
            waterfall = self._generate_waterfall(band_signal, sample_rate)
            
            spectrum_data['bands'][band_name] = {
                'center_frequency_ghz': center_freq / 1e9,
                'bandwidth_ghz': bandwidth / 1e9,
                'time_signal': band_signal[:1000],  # First 1000 samples
                'fft_magnitude_db': 20 * np.log10(np.abs(fft_data) + 1e-12),
                'frequency_axis_ghz': freq_axis / 1e9,
                'waterfall_db': waterfall,
                'peak_power_dbm': float(np.max(20 * np.log10(np.abs(band_signal) + 1e-12))),
                'occupied_bandwidth_ghz': self._calculate_occupied_bandwidth(fft_data, freq_axis) / 1e9
            }
        
        return spectrum_data
    
    def _generate_band_signal(self, center_freq: float, bandwidth: float, 
                            sample_rate: float, samples: int) -> np.ndarray:
        """Generate realistic signal for frequency band"""
        
        # Create OFDM-like signal
        num_subcarriers = int(bandwidth / 1e6)  # 1 MHz per subcarrier
        subcarrier_data = np.random.randn(num_subcarriers) + 1j * np.random.randn(num_subcarriers)
        
        # Generate time-domain signal
        time_samples = np.arange(samples)
        signal_real = np.zeros(samples)
        
        for i, data in enumerate(subcarrier_data):
            freq_offset = (i - num_subcarriers//2) * 1e6  # Subcarrier frequency
            carrier_freq = center_freq + freq_offset
            
            # Generate subcarrier signal
            subcarrier_signal = np.real(data * np.exp(1j * 2 * np.pi * carrier_freq * time_samples / sample_rate))
            signal_real += subcarrier_signal
        
        # Add realistic impairments
        # 1. Phase noise
        phase_noise = np.cumsum(np.random.randn(samples) * 0.01)
        signal_real *= np.cos(phase_noise)
        
        # 2. Thermal noise
        noise_power = -100  # dBm
        noise = np.random.randn(samples) * 10**(noise_power/20)
        signal_real += noise
        
        return signal_real
    
    def _generate_waterfall(self, signal: np.ndarray, sample_rate: float) -> np.ndarray:
        """Generate waterfall (spectrogram) data"""
        
        # Parameters for STFT
        nperseg = 512      # FFT size
        noverlap = 256     # 50% overlap
        
        from scipy.signal import spectrogram
        frequencies, times, Sxx = spectrogram(
            signal, sample_rate, nperseg=nperseg, noverlap=noverlap
        )
        
        # Convert to dB
        Sxx_db = 10 * np.log10(Sxx + 1e-12)
        
        return Sxx_db
    
    def _calculate_occupied_bandwidth(self, fft_data: np.ndarray, freq_axis: np.ndarray) -> float:
        """Calculate 99% occupied bandwidth"""
        
        power_spectrum = np.abs(fft_data)**2
        total_power = np.sum(power_spectrum)
        
        # Sort by power and find 99% threshold
        sorted_indices = np.argsort(power_spectrum)[::-1]
        cumulative_power = np.cumsum(power_spectrum[sorted_indices])
        threshold_idx = np.where(cumulative_power >= 0.99 * total_power)[0][0]
        
        # Find frequency span of significant components
        significant_indices = sorted_indices[:threshold_idx]
        freq_span = np.max(freq_axis[significant_indices]) - np.min(freq_axis[significant_indices])
        
        return abs(freq_span)

def main():
    """Main analysis function"""
    
    if len(sys.argv) < 2:
        print("Usage: python signal_quality_analyzer.py <config_file> [output_dir]")
        sys.exit(1)
    
    config_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "results/signal_analysis"
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize analyzer
    analyzer = SignalQualityAnalyzer(config_file)
    
    print("🔬 DOCOMO 6G Signal Quality Analysis")
    print("=" * 50)
    
    # Analysis parameters
    test_scenarios = [
        {'distance_m': 2.0, 'base_sinr_db': 35.0, 'scenario': 'Lab Close'},
        {'distance_m': 10.0, 'base_sinr_db': 25.0, 'scenario': 'Lab Standard'},
        {'distance_m': 50.0, 'base_sinr_db': 15.0, 'scenario': 'Indoor'},
        {'distance_m': 200.0, 'base_sinr_db': 10.0, 'scenario': 'Outdoor'}
    ]
    
    frequency_bands = list(analyzer.frequency_bands.keys())
    
    # Per-mode SINR analysis
    print("\n📊 Per-Mode SINR Analysis:")
    all_results = {}
    
    for scenario in test_scenarios:
        print(f"\n  Scenario: {scenario['scenario']} ({scenario['distance_m']}m)")
        scenario_results = {}
        
        for band_name in frequency_bands[:2]:  # Analyze first 2 bands
            band_info = analyzer.frequency_bands[band_name]
            frequency_hz = float(band_info['frequency'])
            
            # Analyze with different numbers of modes
            for num_modes in [1, 5, 10, 15]:
                sinr_results = analyzer.calculate_per_mode_sinr(
                    frequency_hz, scenario['distance_m'], 
                    scenario['base_sinr_db'], num_modes
                )
                
                key = f"{band_name}_{num_modes}modes"
                scenario_results[key] = sinr_results
                
                summary = sinr_results['summary']
                print(f"    {band_name} ({num_modes} modes): "
                      f"SINR={summary['mean_sinr_db']:.1f}±{summary['std_sinr_db']:.1f}dB, "
                      f"EVM={summary['mean_evm_percent']:.1f}%, "
                      f"Usable={summary['usable_modes']}/{num_modes}")
        
        all_results[scenario['scenario']] = scenario_results
    
    # Save detailed results
    with open(f"{output_dir}/sinr_analysis.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_dir}/")
    print("   • sinr_analysis.json - Complete SINR/EVM data")
    
    print("\n✅ Signal Quality Analysis Complete!")

if __name__ == "__main__":
    main()
