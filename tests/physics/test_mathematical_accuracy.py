#!/usr/bin/env python3
"""
Test suite for mathematical accuracy improvements in physics calculations
Validates beam divergence, enhanced throughput, and practical constraints
"""

import pytest
import numpy as np
import sys
import os
from typing import Dict

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Import directly to avoid circular imports
import importlib.util
spec = importlib.util.spec_from_file_location(
    "physics_calculator", 
    os.path.join(os.path.dirname(__file__), '..', '..', 'environment', 'physics_calculator.py')
)
physics_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(physics_module)

PhysicsCalculator = physics_module.PhysicsCalculator

class TestMathematicalAccuracy:
    """Test class for mathematical accuracy improvements"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.config = {
            'physics': {
                'clutter': {'enabled': False},
                'pointing_error': {'model': 'static'},
                'blockage': {'enabled': False}
            }
        }
        self.physics = PhysicsCalculator(self.config, bandwidth=1e9)  # 1 GHz bandwidth
        
        # Test frequencies
        self.freq_sub6 = 6e9      # 6 GHz
        self.freq_mmwave = 28e9   # 28 GHz  
        self.freq_sub_thz = 140e9 # 140 GHz
        self.freq_thz = 300e9     # 300 GHz
    
    def test_beam_divergence_physics(self):
        """Test proper diffractive beam spreading implementation"""
        initial_width = 0.01  # 1 cm beam waist
        
        # Test at different frequencies and distances
        test_cases = [
            (self.freq_mmwave, 0.0, initial_width),      # At source
            (self.freq_mmwave, 100.0, initial_width),    # 100m  
            (self.freq_mmwave, 1000.0, initial_width),   # 1km
            (self.freq_sub_thz, 100.0, initial_width),   # Sub-THz
            (self.freq_thz, 50.0, initial_width),        # THz
        ]
        
        for frequency, distance, beam_width in test_cases:
            result = self.physics.calculate_beam_divergence(frequency, distance, beam_width)
            
            # Basic validation
            assert isinstance(result, dict)
            assert all(key in result for key in [
                'beam_width', 'rayleigh_range', 'radius_of_curvature',
                'divergence_angle', 'gouy_phase', 'beam_area'
            ])
            
            # Physics validation
            wavelength = 3e8 / frequency
            expected_rayleigh = np.pi * beam_width**2 / wavelength
            assert abs(result['rayleigh_range'] - expected_rayleigh) / expected_rayleigh < 0.01
            
            # At source, beam width should equal initial width
            if distance == 0:
                assert abs(result['beam_width'] - beam_width) < 1e-10
            else:
                # Beam should expand with distance
                expected_width = beam_width * np.sqrt(1 + (distance / expected_rayleigh)**2)
                assert abs(result['beam_width'] - expected_width) / expected_width < 0.01
                assert result['beam_width'] >= beam_width  # Always larger than initial
            
            # Beam area should be consistent
            expected_area = np.pi * result['beam_width']**2
            assert abs(result['beam_area'] - expected_area) / expected_area < 0.01
    
    def test_enhanced_throughput_adaptive_modulation(self):
        """Test enhanced throughput with adaptive modulation"""
        
        # Test different SINR values and expected modulation
        test_cases = [
            (-20.0, "none", 0.0),      # Too low SINR
            (5.0, "bpsk", 1.0),        # Low SINR -> BPSK
            (10.0, "qpsk", 2.0),       # Medium SINR -> QPSK
            (17.0, "16qam", 4.0),      # Good SINR -> 16QAM
            (22.0, "64qam", 6.0),      # High SINR -> 64QAM
            (30.0, "256qam", 8.0),     # Very high SINR -> 256QAM
        ]
        
        for sinr_db, expected_mod, expected_se in test_cases:
            result = self.physics.calculate_enhanced_throughput(
                sinr_db, self.freq_mmwave, "adaptive", 0.8
            )
            
            assert isinstance(result, dict)
            assert result['modulation'] == expected_mod
            
            if expected_mod != "none":
                assert result['spectral_efficiency'] == expected_se
                assert result['practical_throughput'] > 0
                assert result['link_status'] == 'active'
            else:
                assert result['practical_throughput'] == 0
                assert result['link_status'] == 'failed'
    
    def test_frequency_dependent_limits(self):
        """Test frequency-dependent practical SINR limits"""
        high_sinr = 50.0  # Very high SINR
        
        # Test different frequency bands
        frequencies = [
            (self.freq_sub6, 50.0),     # Sub-6 GHz: highest limit
            (self.freq_mmwave, 45.0),   # mmWave: medium limit  
            (70e9, 40.0),              # mmWave high: lower limit
            (self.freq_sub_thz, 35.0), # Sub-THz: lowest limit
        ]
        
        for frequency, expected_max_sinr in frequencies:
            result = self.physics.calculate_enhanced_throughput(
                high_sinr, frequency, "adaptive", 0.8
            )
            
            # Effective SINR should be capped by frequency limits
            assert result['effective_sinr_db'] <= expected_max_sinr + 1.0  # Small tolerance
            assert result['link_margin_db'] > 0  # Should have link margin
    
    def test_coding_gain_implementation(self):
        """Test coding gain factors for different coding rates"""
        sinr_db = 20.0  # Good SINR
        
        coding_rates = [0.5, 0.7, 0.8, 0.9, 1.0]
        expected_gains = [4.0, 3.0, 2.0, 1.0, 1.0]  # Approximate values
        
        for coding_rate, expected_gain in zip(coding_rates, expected_gains):
            result = self.physics.calculate_enhanced_throughput(
                sinr_db, self.freq_mmwave, "adaptive", coding_rate
            )
            
            assert result['coding_rate'] == coding_rate
            assert abs(result['coding_gain_db'] - expected_gain) <= 0.5
            
            # Lower coding rate should give higher gain but lower throughput  
            if coding_rate < 0.9:  # Only for significantly lower rates
                assert result['coding_gain_db'] > 1.0
    
    def test_shannon_vs_practical_throughput(self):
        """Test relationship between Shannon limit and practical throughput"""
        sinr_values = np.arange(5, 35, 5)  # 5 to 30 dB
        
        for sinr_db in sinr_values:
            result = self.physics.calculate_enhanced_throughput(
                sinr_db, self.freq_mmwave, "adaptive", 0.8
            )
            
            if result['link_status'] == 'active':
                # Practical throughput should be <= Shannon throughput
                assert result['practical_throughput'] <= result['shannon_throughput']
                
                # Practical throughput should be reasonable fraction of Shannon
                ratio = result['practical_throughput'] / result['shannon_throughput']
                assert 0.1 <= ratio <= 1.0, f"Unrealistic throughput ratio {ratio} at SINR {sinr_db} dB"
    
    def test_thz_specific_limitations(self):
        """Test THz-specific limitations and constraints"""
        sinr_db = 25.0  # High SINR
        
        # Compare mmWave vs THz performance
        mmwave_result = self.physics.calculate_enhanced_throughput(
            sinr_db, self.freq_mmwave, "adaptive", 0.8
        )
        thz_result = self.physics.calculate_enhanced_throughput(
            sinr_db, self.freq_thz, "adaptive", 0.8
        )
        
        # THz should have higher link margin
        assert thz_result['link_margin_db'] > mmwave_result['link_margin_db']
        
        # THz should have lower effective SINR due to practical limits
        assert thz_result['effective_sinr_db'] <= mmwave_result['effective_sinr_db']
        
        # Both should still achieve good performance
        assert thz_result['link_status'] == 'active'
        assert mmwave_result['link_status'] == 'active'
    
    def test_beam_divergence_frequency_scaling(self):
        """Test beam divergence scales correctly with frequency"""
        distances = [100.0, 500.0, 1000.0]
        initial_width = 0.01
        
        # Higher frequency should have smaller beam divergence
        mmwave_divergence = []
        thz_divergence = []
        
        for distance in distances:
            mmwave_result = self.physics.calculate_beam_divergence(
                self.freq_mmwave, distance, initial_width
            )
            thz_result = self.physics.calculate_beam_divergence(
                self.freq_thz, distance, initial_width
            )
            
            mmwave_divergence.append(mmwave_result['divergence_angle'])
            thz_divergence.append(thz_result['divergence_angle'])
            
            # THz should have smaller divergence angle
            assert thz_result['divergence_angle'] < mmwave_result['divergence_angle']
            
            # THz should have larger Rayleigh range (for same beam waist)
            assert thz_result['rayleigh_range'] > mmwave_result['rayleigh_range']
    
    def test_practical_vs_shannon_convergence(self):
        """Test that practical throughput approaches Shannon limit at high SINR"""
        high_sinr = 40.0
        
        # At high SINR with good coding, should approach Shannon limit
        result = self.physics.calculate_enhanced_throughput(
            high_sinr, self.freq_sub6, "256qam", 0.9  # High-efficiency setup
        )
        
        efficiency = result['practical_throughput'] / result['shannon_throughput']
        
        # Should achieve good efficiency at high SINR
        assert efficiency > 0.5, f"Low efficiency {efficiency} at high SINR"
        
        # But shouldn't exceed theoretical limit
        assert efficiency <= 1.0
    
    def test_modulation_fallback_behavior(self):
        """Test graceful fallback when SINR is insufficient"""
        low_sinr = 5.0  # Borderline SINR
        
        # Request high-order modulation
        result = self.physics.calculate_enhanced_throughput(
            low_sinr, self.freq_mmwave, "256qam", 0.8
        )
        
        # Should fallback to suitable modulation
        assert result['modulation'] in ["bpsk", "qpsk"]
        assert result['link_status'] == 'active'
        assert result['practical_throughput'] > 0
    
    def test_error_handling_robustness(self):
        """Test error handling for invalid inputs"""
        
        # Test beam divergence with invalid inputs
        result = self.physics.calculate_beam_divergence(0, 100, 0.01)  # Zero frequency
        assert result['beam_width'] == 0.01  # Should return safe defaults
        
        result = self.physics.calculate_beam_divergence(1e9, -100, 0.01)  # Negative distance
        assert result['beam_width'] == 0.01
        
        # Test enhanced throughput with invalid inputs
        result = self.physics.calculate_enhanced_throughput(float('inf'), 1e9)
        assert result['link_status'] == 'error'
        
        result = self.physics.calculate_enhanced_throughput(20, 0)  # Zero frequency
        assert result['link_status'] == 'error'
    
    def test_consistency_with_original_throughput(self):
        """Test that enhanced throughput is consistent with original for basic cases"""
        sinr_db = 20.0
        
        # Original Shannon calculation
        original = self.physics.calculate_throughput(sinr_db)
        
        # Enhanced calculation
        enhanced = self.physics.calculate_enhanced_throughput(sinr_db, self.freq_mmwave)
        
        # Shannon components should be reasonably close (allowing for link margin effects)
        assert abs(enhanced['shannon_throughput'] - original) / original < 0.15
        
        # Practical should be lower due to realistic constraints
        assert enhanced['practical_throughput'] <= enhanced['shannon_throughput']

if __name__ == "__main__":
    pytest.main([__file__])
