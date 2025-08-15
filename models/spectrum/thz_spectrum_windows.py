#!/usr/bin/env python3
"""
THz Spectrum Windows Model
Implementation of realistic atmospheric transmission windows for THz frequencies
based on molecular absorption nulls and regulatory allocations.

References:
- ITU-R Radio Regulations
- ITU-R SM.1842: Frequency ranges for THz applications
- CEPT Report 32: THz spectrum requirements
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class SpectrumAllocation(Enum):
    MOBILE = "mobile"
    FIXED = "fixed" 
    RADIOLOCATION = "radiolocation"
    RADIO_ASTRONOMY = "radio_astronomy"
    EARTH_EXPLORATION = "earth_exploration"
    SPACE_RESEARCH = "space_research"
    ISM = "ism"  # Industrial, Scientific, Medical
    PASSIVE = "passive"  # Passive services (protected)

@dataclass
class SpectrumWindow:
    """Atmospheric transmission window specification"""
    name: str
    center_freq_ghz: float
    bandwidth_ghz: float
    start_freq_ghz: float
    end_freq_ghz: float
    atmospheric_loss_db_km: float
    usable_bandwidth_ghz: float  # After guard bands and regulations
    allocation: SpectrumAllocation
    regulatory_constraints: Dict[str, str]
    interference_sources: List[str]

class THzSpectrumModel:
    """
    Realistic THz spectrum model with atmospheric windows and regulatory constraints.
    
    Models:
    - Atmospheric transmission windows between absorption peaks
    - ITU-R spectrum allocations and regulations
    - Guard bands for passive service protection
    - Regional variations in spectrum policy
    - Interference sources and protection requirements
    """
    
    def __init__(self):
        # Initialize realistic transmission windows
        self.transmission_windows = self._initialize_transmission_windows()
        
        # ITU-R allocations for THz bands
        self.itu_allocations = self._initialize_itu_allocations()
        
        # Protected frequencies (radio astronomy, Earth exploration)
        self.protected_frequencies = self._initialize_protected_frequencies()
    
    def _initialize_transmission_windows(self) -> Dict[str, SpectrumWindow]:
        """Initialize realistic atmospheric transmission windows"""
        windows = {}
        
        # Window 1: Lower sub-THz (relatively clear)
        windows['W1'] = SpectrumWindow(
            name="Sub-THz Window 1",
            center_freq_ghz=125.0,
            bandwidth_ghz=35.0,
            start_freq_ghz=107.5,
            end_freq_ghz=142.5,
            atmospheric_loss_db_km=2.5,  # Low loss window
            usable_bandwidth_ghz=25.0,   # After guard bands
            allocation=SpectrumAllocation.MOBILE,
            regulatory_constraints={
                'max_eirp_dbm': 55,
                'max_power_density_dbm_mhz': 13,
                'coordination_required': 'radio_astronomy'
            },
            interference_sources=['automotive_radar', 'industrial_heating']
        )
        
        # Window 2: Mid sub-THz (moderate loss)
        windows['W2'] = SpectrumWindow(
            name="Sub-THz Window 2", 
            center_freq_ghz=160.0,
            bandwidth_ghz=25.0,
            start_freq_ghz=147.5,
            end_freq_ghz=172.5,
            atmospheric_loss_db_km=5.0,
            usable_bandwidth_ghz=18.0,
            allocation=SpectrumAllocation.MOBILE,
            regulatory_constraints={
                'max_eirp_dbm': 50,
                'max_power_density_dbm_mhz': 10,
                'coordination_required': 'earth_exploration'
            },
            interference_sources=['weather_radar']
        )
        
        # Window 3: Higher sub-THz (higher loss but large bandwidth)
        windows['W3'] = SpectrumWindow(
            name="Sub-THz Window 3",
            center_freq_ghz=220.0,
            bandwidth_ghz=40.0,
            start_freq_ghz=200.0,
            end_freq_ghz=240.0,
            atmospheric_loss_db_km=8.0,
            usable_bandwidth_ghz=30.0,
            allocation=SpectrumAllocation.MOBILE,
            regulatory_constraints={
                'max_eirp_dbm': 45,
                'max_power_density_dbm_mhz': 8,
                'coordination_required': 'passive_services'
            },
            interference_sources=['satellite_communications']
        )
        
        # Window 4: Lower THz (between water vapor lines)
        windows['W4'] = SpectrumWindow(
            name="THz Window 1",
            center_freq_ghz=340.0,
            bandwidth_ghz=30.0,
            start_freq_ghz=325.0,
            end_freq_ghz=355.0,
            atmospheric_loss_db_km=15.0,  # Higher loss
            usable_bandwidth_ghz=20.0,
            allocation=SpectrumAllocation.ISM,
            regulatory_constraints={
                'max_eirp_dbm': 40,
                'max_power_density_dbm_mhz': 5,
                'duty_cycle_limit': 0.1
            },
            interference_sources=['medical_imaging', 'security_scanners']
        )
        
        # Window 5: Mid THz (narrow but useful)
        windows['W5'] = SpectrumWindow(
            name="THz Window 2",
            center_freq_ghz=410.0,
            bandwidth_ghz=20.0,
            start_freq_ghz=400.0,
            end_freq_ghz=420.0,
            atmospheric_loss_db_km=25.0,
            usable_bandwidth_ghz=12.0,
            allocation=SpectrumAllocation.FIXED,
            regulatory_constraints={
                'max_eirp_dbm': 35,
                'max_power_density_dbm_mhz': 3,
                'indoor_only': True
            },
            interference_sources=['industrial_heating']
        )
        
        # Window 6: Upper THz (experimental)
        windows['W6'] = SpectrumWindow(
            name="THz Window 3",
            center_freq_ghz=650.0,
            bandwidth_ghz=50.0,
            start_freq_ghz=625.0,
            end_freq_ghz=675.0,
            atmospheric_loss_db_km=50.0,  # Very high loss
            usable_bandwidth_ghz=35.0,
            allocation=SpectrumAllocation.ISM,
            regulatory_constraints={
                'max_eirp_dbm': 30,
                'max_power_density_dbm_mhz': 0,
                'experimental_only': True,
                'indoor_only': True,
                'max_range_m': 10
            },
            interference_sources=['space_research_passive']
        )
        
        return windows
    
    def _initialize_itu_allocations(self) -> Dict[Tuple[float, float], SpectrumAllocation]:
        """Initialize ITU-R spectrum allocations for THz bands"""
        return {
            (105.0, 116.0): SpectrumAllocation.RADIO_ASTRONOMY,
            (116.0, 148.5): SpectrumAllocation.MOBILE,
            (148.5, 151.5): SpectrumAllocation.RADIO_ASTRONOMY,  # Protected
            (151.5, 174.8): SpectrumAllocation.MOBILE,
            (174.8, 182.0): SpectrumAllocation.EARTH_EXPLORATION,
            (182.0, 185.0): SpectrumAllocation.RADIO_ASTRONOMY,  # Protected
            (200.0, 252.0): SpectrumAllocation.MOBILE,
            (275.0, 3000.0): SpectrumAllocation.ISM,  # Above 275 GHz mostly ISM
        }
    
    def _initialize_protected_frequencies(self) -> List[Tuple[float, float]]:
        """Initialize protected frequency ranges (passive services)"""
        return [
            (116.0, 120.0),   # Radio astronomy
            (148.5, 151.5),   # Radio astronomy  
            (164.0, 167.0),   # Earth exploration passive
            (182.0, 185.0),   # Radio astronomy
            (235.0, 238.0),   # Earth exploration passive
            (294.0, 300.0),   # Radio astronomy
            (326.0, 334.0),   # Radio astronomy
            (356.0, 364.0),   # Earth exploration passive
            (416.0, 434.0),   # Radio astronomy
            (442.0, 444.0),   # Radio astronomy
            (496.0, 506.0),   # Earth exploration passive
            (546.0, 568.0),   # Earth exploration passive
            (624.0, 629.0),   # Radio astronomy
            (634.0, 654.0),   # Radio astronomy
            (659.0, 661.0),   # Radio astronomy
            (684.0, 692.0),   # Earth exploration passive
            (730.0, 732.0),   # Radio astronomy
            (851.0, 853.0),   # Radio astronomy
        ]
    
    def get_usable_spectrum(self, freq_range_ghz: Tuple[float, float], 
                          application: str = 'mobile') -> List[SpectrumWindow]:
        """
        Get usable spectrum windows in frequency range for given application.
        
        Args:
            freq_range_ghz: (start_freq, end_freq) in GHz
            application: Application type ('mobile', 'fixed', 'ism')
            
        Returns:
            List of usable spectrum windows
        """
        start_freq, end_freq = freq_range_ghz
        usable_windows = []
        
        for window in self.transmission_windows.values():
            # Check if window overlaps with requested range
            if (window.start_freq_ghz <= end_freq and window.end_freq_ghz >= start_freq):
                
                # Check if allocation is compatible
                if self._is_allocation_compatible(window.allocation, application):
                    
                    # Check if not heavily protected
                    if not self._is_heavily_protected(window):
                        
                        # Adjust window to requested range
                        adjusted_window = self._adjust_window_to_range(window, (start_freq, end_freq))
                        usable_windows.append(adjusted_window)
        
        return sorted(usable_windows, key=lambda w: w.center_freq_ghz)
    
    def _is_allocation_compatible(self, allocation: SpectrumAllocation, application: str) -> bool:
        """Check if spectrum allocation is compatible with application"""
        compatibility_map = {
            'mobile': [SpectrumAllocation.MOBILE, SpectrumAllocation.ISM],
            'fixed': [SpectrumAllocation.FIXED, SpectrumAllocation.MOBILE, SpectrumAllocation.ISM],
            'ism': [SpectrumAllocation.ISM],
            'research': [SpectrumAllocation.ISM, SpectrumAllocation.FIXED]
        }
        
        return allocation in compatibility_map.get(application, [])
    
    def _is_heavily_protected(self, window: SpectrumWindow) -> bool:
        """Check if window overlaps with heavily protected frequencies"""
        for start_prot, end_prot in self.protected_frequencies:
            if (window.start_freq_ghz < end_prot and window.end_freq_ghz > start_prot):
                # Calculate overlap percentage
                overlap_start = max(window.start_freq_ghz, start_prot)
                overlap_end = min(window.end_freq_ghz, end_prot)
                overlap_bw = overlap_end - overlap_start
                
                if overlap_bw > window.bandwidth_ghz * 0.2:  # >20% overlap
                    return True
        
        return False
    
    def _adjust_window_to_range(self, window: SpectrumWindow, 
                               freq_range: Tuple[float, float]) -> SpectrumWindow:
        """Adjust window boundaries to fit within requested frequency range"""
        start_freq, end_freq = freq_range
        
        # Create adjusted copy
        adjusted = SpectrumWindow(
            name=window.name,
            center_freq_ghz=window.center_freq_ghz,
            bandwidth_ghz=window.bandwidth_ghz,
            start_freq_ghz=max(window.start_freq_ghz, start_freq),
            end_freq_ghz=min(window.end_freq_ghz, end_freq),
            atmospheric_loss_db_km=window.atmospheric_loss_db_km,
            usable_bandwidth_ghz=window.usable_bandwidth_ghz,
            allocation=window.allocation,
            regulatory_constraints=window.regulatory_constraints,
            interference_sources=window.interference_sources
        )
        
        # Recalculate adjusted parameters
        adjusted.bandwidth_ghz = adjusted.end_freq_ghz - adjusted.start_freq_ghz
        adjusted.center_freq_ghz = (adjusted.start_freq_ghz + adjusted.end_freq_ghz) / 2
        adjusted.usable_bandwidth_ghz = min(adjusted.bandwidth_ghz * 0.8, window.usable_bandwidth_ghz)
        
        return adjusted
    
    def calculate_aggregate_bandwidth(self, windows: List[SpectrumWindow]) -> Dict[str, float]:
        """Calculate aggregate bandwidth statistics for spectrum windows"""
        if not windows:
            return {
                'total_bandwidth_ghz': 0.0,
                'usable_bandwidth_ghz': 0.0,
                'efficiency_percent': 0.0,
                'avg_loss_db_km': 0.0
            }
        
        total_bw = sum(w.bandwidth_ghz for w in windows)
        usable_bw = sum(w.usable_bandwidth_ghz for w in windows)
        avg_loss = sum(w.atmospheric_loss_db_km * w.bandwidth_ghz for w in windows) / total_bw
        efficiency = (usable_bw / total_bw * 100) if total_bw > 0 else 0
        
        return {
            'total_bandwidth_ghz': total_bw,
            'usable_bandwidth_ghz': usable_bw,
            'efficiency_percent': efficiency,
            'avg_loss_db_km': avg_loss
        }
    
    def get_interference_assessment(self, window: SpectrumWindow) -> Dict[str, float]:
        """Assess interference environment for spectrum window"""
        interference_levels = {
            'automotive_radar': 15.0,      # dB above noise
            'weather_radar': 20.0,
            'satellite_communications': 10.0,
            'industrial_heating': 25.0,
            'medical_imaging': 5.0,
            'security_scanners': 8.0,
            'space_research_passive': 0.0  # Protected, very low interference
        }
        
        assessment = {}
        for source in window.interference_sources:
            assessment[source] = interference_levels.get(source, 0.0)
        
        # Calculate total interference (non-coherent sum)
        total_interference = 10 * np.log10(sum(10**(level/10) for level in assessment.values()))
        assessment['total_interference_db'] = total_interference
        
        return assessment

def test_spectrum_windows():
    """Test realistic spectrum windows model"""
    model = THzSpectrumModel()
    
    print("=== REALISTIC THz SPECTRUM WINDOWS ===")
    print("Window | Center  | Total BW | Usable BW | Loss    | Allocation")
    print("       | (GHz)   | (GHz)    | (GHz)     | (dB/km) |")
    print("-" * 65)
    
    for name, window in model.transmission_windows.items():
        print(f"{name:6s} | {window.center_freq_ghz:7.1f} | {window.bandwidth_ghz:8.1f} | "
              f"{window.usable_bandwidth_ghz:9.1f} | {window.atmospheric_loss_db_km:7.1f} | "
              f"{window.allocation.value}")
    
    # Test usable spectrum for mobile applications
    mobile_windows = model.get_usable_spectrum((100, 400), 'mobile')
    stats = model.calculate_aggregate_bandwidth(mobile_windows)
    
    print(f"\n=== MOBILE APPLICATION (100-400 GHz) ===")
    print(f"Available windows: {len(mobile_windows)}")
    print(f"Total bandwidth: {stats['total_bandwidth_ghz']:.1f} GHz")
    print(f"Usable bandwidth: {stats['usable_bandwidth_ghz']:.1f} GHz")
    print(f"Efficiency: {stats['efficiency_percent']:.1f}%")
    print(f"Average loss: {stats['avg_loss_db_km']:.1f} dB/km")
    
    # Compare with previous unrealistic model
    print(f"\n=== COMPARISON: Current vs Previous Implementation ===")
    old_continuous_bw = {
        'sub_thz_300': 50.0,   # Previous: 50 GHz continuous
        'thz_600': 100.0       # Previous: 100 GHz continuous  
    }
    
    realistic_bw_300 = sum(w.usable_bandwidth_ghz for w in mobile_windows if 250 <= w.center_freq_ghz <= 350)
    realistic_bw_600 = sum(w.usable_bandwidth_ghz for w in model.get_usable_spectrum((550, 700), 'ism'))
    
    print("Band      | Old BW  | New BW  | Reduction")
    print("          | (GHz)   | (GHz)   | Factor")
    print("-" * 40)
    print(f"300 GHz   | {old_continuous_bw['sub_thz_300']:7.1f} | {realistic_bw_300:7.1f} | {old_continuous_bw['sub_thz_300']/max(realistic_bw_300,1):7.1f}x")
    print(f"600 GHz   | {old_continuous_bw['thz_600']:7.1f} | {realistic_bw_600:7.1f} | {old_continuous_bw['thz_600']/max(realistic_bw_600,1):7.1f}x")

if __name__ == "__main__":
    test_spectrum_windows()
