#!/usr/bin/env python3
"""
DOCOMO 6G Spectrum Visualizer
============================

Advanced visualization of:
- Constellation diagrams with EVM analysis
- Spectrum waterfall displays
- FFT analysis with occupied bandwidth
- Per-mode SINR distributions

Author: DOCOMO 6G Research
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from pathlib import Path
import json
import sys
from typing import Dict, List, Tuple, Optional

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.analysis.signal_quality_analyzer import SignalQualityAnalyzer

class SpectrumVisualizer:
    """Advanced spectrum and signal quality visualization"""
    
    def __init__(self, output_dir: str = "results/spectrum_plots"):
        """Initialize visualizer"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        # Custom colormap for spectrum
        self.spectrum_cmap = LinearSegmentedColormap.from_list(
            'spectrum', ['#000080', '#0000FF', '#00FFFF', '#FFFF00', '#FF8000', '#FF0000']
        )
        
    def plot_comprehensive_analysis(self, analyzer: SignalQualityAnalyzer, 
                                  config_name: str) -> str:
        """Create comprehensive signal quality analysis plots"""
        
        # Create large figure with multiple subplots
        fig = plt.figure(figsize=(20, 16))
        gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.3, wspace=0.3)
        
        # Test parameters
        frequency_hz = 300e9  # THz band
        distance_m = 5.0      # Lab scenario
        base_sinr_db = 30.0   # Good SINR
        num_modes = 10        # Multiple modes
        
        # 1. Per-mode SINR analysis (top row)
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_per_mode_sinr(ax1, analyzer, frequency_hz, distance_m, base_sinr_db, num_modes)
        
        # 2. EVM vs Modulation (top row)
        ax2 = fig.add_subplot(gs[0, 2:])
        self._plot_evm_vs_modulation(ax2, analyzer)
        
        # 3. Constellation diagrams (second row)
        constellation_axes = [fig.add_subplot(gs[1, i]) for i in range(4)]
        self._plot_constellation_comparison(constellation_axes, analyzer, base_sinr_db)
        
        # 4. Spectrum occupancy (third row)
        ax5 = fig.add_subplot(gs[2, :2])
        ax6 = fig.add_subplot(gs[2, 2:])
        self._plot_spectrum_analysis(ax5, ax6, analyzer)
        
        # 5. Waterfall display (bottom row)
        ax7 = fig.add_subplot(gs[3, :])
        self._plot_spectrum_waterfall(ax7, analyzer)
        
        # Add title and metadata
        fig.suptitle(f'DOCOMO 6G Signal Quality Analysis - {config_name}', 
                    fontsize=16, fontweight='bold')
        
        # Add analysis timestamp
        timestamp = plt.figtext(0.99, 0.01, f'Generated: {np.datetime64("now")}',
                               ha='right', va='bottom', fontsize=8, alpha=0.7)
        
        # Save comprehensive plot
        output_file = self.output_dir / f"comprehensive_analysis_{config_name}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"📊 Comprehensive analysis saved: {output_file}")
        
        plt.close()
        return str(output_file)
    
    def _plot_per_mode_sinr(self, ax, analyzer: SignalQualityAnalyzer, 
                           frequency_hz: float, distance_m: float, 
                           base_sinr_db: float, num_modes: int):
        """Plot per-mode SINR with error bars"""
        
        # Analyze multiple scenarios
        scenarios = [5, 10, 15, 20]  # Different numbers of modes
        mode_data = {}
        
        for n_modes in scenarios:
            results = analyzer.calculate_per_mode_sinr(
                frequency_hz, distance_m, base_sinr_db, n_modes
            )
            
            mode_indices = [m['mode_number'] for m in results['mode_analysis']]
            sinr_values = [m['sinr_db'] for m in results['mode_analysis']]
            evm_values = [m['evm_rms_percent'] for m in results['mode_analysis']]
            
            mode_data[n_modes] = {
                'modes': mode_indices,
                'sinr': sinr_values,
                'evm': evm_values
            }
        
        # Plot SINR vs mode number for different total modes
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for i, n_modes in enumerate(scenarios):
            data = mode_data[n_modes]
            ax.plot(data['modes'], data['sinr'], 'o-', 
                   color=colors[i], label=f'{n_modes} total modes',
                   linewidth=2, markersize=6)
        
        # Add SINR thresholds
        ax.axhline(y=20, color='green', linestyle='--', alpha=0.7, label='Good SINR (20 dB)')
        ax.axhline(y=10, color='orange', linestyle='--', alpha=0.7, label='Min SINR (10 dB)')
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Link failure')
        
        ax.set_xlabel('OAM Mode Number')
        ax.set_ylabel('SINR (dB)')
        ax.set_title(f'Per-Mode SINR Analysis\n{frequency_hz/1e9:.0f} GHz, {distance_m:.1f}m')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-5, 35)
    
    def _plot_evm_vs_modulation(self, ax, analyzer: SignalQualityAnalyzer):
        """Plot EVM requirements vs modulation schemes"""
        
        modulations = list(analyzer.modulation_schemes.keys())
        evm_limits = [analyzer.modulation_schemes[mod]['evm_limit'] for mod in modulations]
        sinr_req = [analyzer._required_sinr(mod) for mod in modulations]
        
        # Create dual-axis plot
        ax2 = ax.twinx()
        
        # Plot EVM limits
        bars1 = ax.bar([i-0.2 for i in range(len(modulations))], evm_limits, 
                      width=0.4, label='EVM Limit (%)', color='skyblue', alpha=0.7)
        
        # Plot SINR requirements
        bars2 = ax2.bar([i+0.2 for i in range(len(modulations))], sinr_req,
                       width=0.4, label='SINR Req (dB)', color='orange', alpha=0.7)
        
        # Formatting
        ax.set_xlabel('Modulation Scheme')
        ax.set_ylabel('EVM Limit (%)', color='blue')
        ax2.set_ylabel('Required SINR (dB)', color='orange')
        ax.set_title('Modulation Requirements\nEVM Limits vs SINR Thresholds')
        
        ax.set_xticks(range(len(modulations)))
        ax.set_xticklabels(modulations, rotation=45)
        
        # Add value labels on bars
        for i, (evm, sinr) in enumerate(zip(evm_limits, sinr_req)):
            ax.text(i-0.2, evm+0.5, f'{evm:.1f}%', ha='center', va='bottom', fontsize=9)
            ax2.text(i+0.2, sinr+0.5, f'{sinr:.0f}dB', ha='center', va='bottom', fontsize=9)
        
        ax.set_ylim(0, max(evm_limits) * 1.2)
        ax2.set_ylim(0, max(sinr_req) * 1.2)
        ax.grid(True, alpha=0.3)
    
    def _plot_constellation_comparison(self, axes, analyzer: SignalQualityAnalyzer, base_sinr_db: float):
        """Plot constellation diagrams for different modulations"""
        
        modulations = ['QPSK', '16QAM', '64QAM', '256QAM']
        sinr_values = [base_sinr_db, base_sinr_db-5, base_sinr_db-10, base_sinr_db-15]
        
        for i, (ax, mod, sinr) in enumerate(zip(axes, modulations, sinr_values)):
            # Generate constellation
            rx_i, rx_q, const_data = analyzer.generate_constellation_diagram(sinr, mod)
            
            # Plot received constellation
            ax.scatter(rx_i, rx_q, s=1, alpha=0.6, c='blue', rasterized=True)
            
            # Plot ideal constellation
            ideal_i, ideal_q = const_data['ideal_points']
            ax.scatter(ideal_i, ideal_q, s=50, c='red', marker='x', linewidth=2, label='Ideal')
            
            # Formatting
            ax.set_title(f'{mod}\nSINR: {sinr:.1f}dB, EVM: {const_data["evm_rms_percent"]:.1f}%')
            ax.set_xlabel('In-phase')
            ax.set_ylabel('Quadrature')
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
            
            # Set axis limits based on modulation
            if mod == 'QPSK':
                ax.set_xlim(-2, 2)
                ax.set_ylim(-2, 2)
            else:
                ax.set_xlim(-1.5, 1.5)
                ax.set_ylim(-1.5, 1.5)
    
    def _plot_spectrum_analysis(self, ax1, ax2, analyzer: SignalQualityAnalyzer):
        """Plot spectrum analysis - FFT and occupied bandwidth"""
        
        # Get frequency bands for analysis
        band_names = list(analyzer.frequency_bands.keys())[:3]  # First 3 bands
        spectrum_data = analyzer.generate_spectrum_occupancy(band_names)
        
        # Plot 1: FFT magnitude
        colors = ['blue', 'red', 'green']
        
        for i, band_name in enumerate(band_names):
            if band_name in spectrum_data['bands']:
                band_data = spectrum_data['bands'][band_name]
                
                # Plot only positive frequencies
                freqs = band_data['frequency_axis_ghz']
                fft_mag = band_data['fft_magnitude_db']
                
                # Find positive frequency indices
                pos_indices = freqs >= 0
                freqs_pos = freqs[pos_indices]
                fft_pos = fft_mag[pos_indices]
                
                # Plot around center frequency
                center_freq = band_data['center_frequency_ghz']
                freq_range = band_data['bandwidth_ghz'] * 2
                
                freq_mask = (freqs_pos >= center_freq - freq_range) & (freqs_pos <= center_freq + freq_range)
                
                ax1.plot(freqs_pos[freq_mask], fft_pos[freq_mask], 
                        color=colors[i], label=f'{band_name} ({center_freq:.0f} GHz)',
                        linewidth=1.5)
        
        ax1.set_xlabel('Frequency (GHz)')
        ax1.set_ylabel('Magnitude (dB)')
        ax1.set_title('Spectrum Analysis - FFT')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Occupied bandwidth comparison
        band_names_clean = [name.replace('_', ' ').upper() for name in band_names]
        center_freqs = []
        occupied_bw = []
        allocated_bw = []
        
        for band_name in band_names:
            if band_name in spectrum_data['bands']:
                band_data = spectrum_data['bands'][band_name]
                center_freqs.append(band_data['center_frequency_ghz'])
                occupied_bw.append(band_data['occupied_bandwidth_ghz'])
                allocated_bw.append(band_data['bandwidth_ghz'])
        
        x = np.arange(len(band_names_clean))
        width = 0.35
        
        bars1 = ax2.bar(x - width/2, allocated_bw, width, label='Allocated BW', alpha=0.7)
        bars2 = ax2.bar(x + width/2, occupied_bw, width, label='Occupied BW (99%)', alpha=0.7)
        
        ax2.set_xlabel('Frequency Band')
        ax2.set_ylabel('Bandwidth (GHz)')
        ax2.set_title('Bandwidth Utilization')
        ax2.set_xticks(x)
        ax2.set_xticklabels(band_names_clean)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add efficiency percentages
        for i, (alloc, occup) in enumerate(zip(allocated_bw, occupied_bw)):
            efficiency = (occup / alloc) * 100
            ax2.text(i, max(alloc, occup) + 0.5, f'{efficiency:.0f}%', 
                    ha='center', va='bottom', fontweight='bold')
    
    def _plot_spectrum_waterfall(self, ax, analyzer: SignalQualityAnalyzer):
        """Plot spectrum waterfall (time-frequency) display"""
        
        # Generate waterfall data for main band
        band_names = list(analyzer.frequency_bands.keys())
        if not band_names:
            ax.text(0.5, 0.5, 'No frequency bands available', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        main_band = band_names[0]  # Use first band
        spectrum_data = analyzer.generate_spectrum_occupancy([main_band], duration_ms=50.0)
        
        if main_band not in spectrum_data['bands']:
            ax.text(0.5, 0.5, f'No data for {main_band}', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        band_data = spectrum_data['bands'][main_band]
        
        # Create simplified waterfall for visualization
        center_freq = band_data['center_frequency_ghz']
        bandwidth = band_data['bandwidth_ghz']
        
        # Generate synthetic waterfall data
        time_samples = 100
        freq_samples = 200
        
        time_axis = np.linspace(0, 50, time_samples)  # 50 ms
        freq_axis = np.linspace(center_freq - bandwidth, center_freq + bandwidth, freq_samples)
        
        # Create realistic waterfall pattern
        waterfall_data = np.zeros((freq_samples, time_samples))
        
        for t in range(time_samples):
            # Main signal around center frequency
            main_signal_indices = (freq_axis >= center_freq - bandwidth/4) & (freq_axis <= center_freq + bandwidth/4)
            waterfall_data[main_signal_indices, t] = -20 + np.random.randn(np.sum(main_signal_indices)) * 5
            
            # Noise floor
            noise_indices = ~main_signal_indices
            waterfall_data[noise_indices, t] = -60 + np.random.randn(np.sum(noise_indices)) * 10
            
            # Add some time-varying interference
            if t % 20 < 5:  # Periodic interference
                interference_freq = center_freq + bandwidth/3
                interference_idx = np.argmin(np.abs(freq_axis - interference_freq))
                waterfall_data[interference_idx-2:interference_idx+3, t] += 15
        
        # Plot waterfall
        im = ax.imshow(waterfall_data, aspect='auto', origin='lower', 
                      cmap=self.spectrum_cmap, interpolation='bilinear',
                      extent=[time_axis[0], time_axis[-1], freq_axis[0], freq_axis[-1]])
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Power (dBm)', rotation=270, labelpad=15)
        
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Frequency (GHz)')
        ax.set_title(f'Spectrum Waterfall - {main_band.upper()}\n{center_freq:.0f} GHz ± {bandwidth/2:.1f} GHz')
        
        # Add frequency markers
        ax.axhline(y=center_freq, color='white', linestyle='--', alpha=0.8, linewidth=1, label='Center')
        ax.axhline(y=center_freq - bandwidth/2, color='yellow', linestyle=':', alpha=0.6, linewidth=1)
        ax.axhline(y=center_freq + bandwidth/2, color='yellow', linestyle=':', alpha=0.6, linewidth=1)

def main():
    """Main visualization function"""
    
    if len(sys.argv) < 2:
        print("Usage: python spectrum_visualizer.py <config_file> [output_dir]")
        sys.exit(1)
    
    config_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "results/spectrum_plots"
    
    # Initialize components
    analyzer = SignalQualityAnalyzer(config_file)
    visualizer = SpectrumVisualizer(output_dir)
    
    # Extract config name
    config_name = Path(config_file).stem
    
    print("📊 DOCOMO 6G Spectrum Visualization")
    print("=" * 50)
    
    # Generate comprehensive analysis
    plot_file = visualizer.plot_comprehensive_analysis(analyzer, config_name)
    
    print(f"\n✅ Spectrum Visualization Complete!")
    print(f"📊 Comprehensive plots: {plot_file}")

if __name__ == "__main__":
    main()
