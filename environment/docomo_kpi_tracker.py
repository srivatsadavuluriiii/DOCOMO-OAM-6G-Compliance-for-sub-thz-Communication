#!/usr/bin/env python3
"""
DOCOMO KPI Tracker - Real-time Performance Monitoring
Tracks all DOCOMO 6G KPIs with compliance scoring and analytics
"""

import numpy as np
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
import json
from datetime import datetime, timedelta

@dataclass
class DOCOMOKPIs:
    """DOCOMO 6G Key Performance Indicators"""
                           
    peak_data_rate_tbps: float = 1.0                                  
    user_data_rate_gbps: float = 100.0                                   
    latency_ms: float = 0.1                                               
    reliability: float = 0.9999999                                    
    
                                  
    mobility_kmh: float = 500.0                                           
    connection_density_per_km2: float = 1e7                         
    coverage_improvement: float = 100.0                                
    
                     
    energy_efficiency_improvement: float = 100                         
    spectrum_efficiency_improvement: float = 10                        
    
                   
    sensing_accuracy_cm: float = 1.0                                  
    positioning_accuracy_cm: float = 10.0                        
    synchronization_accuracy_ns: float = 1.0                  

@dataclass 
class PerformanceMeasurement:
    """Single performance measurement point"""
    timestamp: datetime
    throughput_gbps: float
    latency_ms: float
    sinr_db: float
    distance_m: float
    mobility_kmh: float
    band: str
    oam_mode: int
    energy_consumption_w: float
    reliability_score: float
    handover_count: int = 0
    beam_alignment_error_deg: float = 0.0
    doppler_shift_hz: float = 0.0
    atmospheric_loss_db: float = 0.0

class DOCOMOKPITracker:
    """
    Real-time DOCOMO KPI tracking and compliance monitoring
    """
    
    def __init__(self, config: Dict[str, Any], window_size: int = 1000):
        """
        Initialize DOCOMO KPI tracker
        
        Args:
            config: DOCOMO configuration dictionary
            window_size: Rolling window size for statistics
        """
                                              
        self.config = config if isinstance(config, dict) else {}
        self.window_size = window_size
        self.docomo_targets = DOCOMOKPIs()
        
                                    
        kpi_targets = self.config.get('docomo_6g_system', {}).get('kpi_targets', {})
        for key, value in kpi_targets.items():
            if hasattr(self.docomo_targets, key):
                setattr(self.docomo_targets, key, value)
        
                                          
        self.measurements = deque(maxlen=window_size)
        self.throughput_history = deque(maxlen=window_size)
        self.latency_history = deque(maxlen=window_size)
        self.reliability_history = deque(maxlen=window_size)
        self.energy_history = deque(maxlen=window_size)
        self.mobility_history = deque(maxlen=window_size)
        
                             
        self.compliance_scores = {
            'throughput': deque(maxlen=100),
            'latency': deque(maxlen=100),
            'reliability': deque(maxlen=100),
            'mobility': deque(maxlen=100),
            'energy': deque(maxlen=100),
            'overall': deque(maxlen=100)
        }
        
                             
        self.session_stats = {
            'start_time': datetime.now(),
            'total_measurements': 0,
            'peak_throughput_gbps': 0.0,
            'min_latency_ms': float('inf'),
            'max_mobility_kmh': 0.0,
            'handover_count': 0,
            'error_count': 0
        }
        
                                            
        monitoring_cfg = self.config.get('docomo_6g_system', {}).get('monitoring', {})
        self.anomaly_detection_enabled = bool(monitoring_cfg.get('anomaly_detection', True))
        self.anomaly_threshold = float(monitoring_cfg.get('anomaly_threshold', 3.5))
        self.anomaly_warmup_samples = int(monitoring_cfg.get('anomaly_warmup_samples', 200))
        self.anomaly_log_enabled = bool(monitoring_cfg.get('anomaly_log', False))

        self.performance_trend = 'stable'
        self.last_update = datetime.now()
        
    def update(self, measurement: PerformanceMeasurement) -> Dict[str, float]:
        """
        Update KPI tracker with new measurement
        
        Args:
            measurement: Performance measurement
            
        Returns:
            Current compliance scores
        """
                                    
        self.measurements.append(measurement)
        self.throughput_history.append(measurement.throughput_gbps)
        self.latency_history.append(measurement.latency_ms)
        self.reliability_history.append(measurement.reliability_score)
        self.energy_history.append(measurement.energy_consumption_w)
        self.mobility_history.append(measurement.mobility_kmh)
        
                                   
        self._update_session_stats(measurement)
        
                                     
        compliance = self._calculate_compliance_scores(measurement)
        
                                   
        for metric, score in compliance.items():
            if metric in self.compliance_scores:
                self.compliance_scores[metric].append(score)
        
                          
        self._detect_anomalies(measurement)
        
                                  
        self._update_performance_trend()
        
        self.last_update = datetime.now()
        return compliance
    
    def _update_session_stats(self, measurement: PerformanceMeasurement):
        """Update session-level statistics"""
        self.session_stats['total_measurements'] += 1
        
                                  
        if measurement.throughput_gbps > self.session_stats['peak_throughput_gbps']:
            self.session_stats['peak_throughput_gbps'] = measurement.throughput_gbps
            
                                    
        if measurement.latency_ms < self.session_stats['min_latency_ms']:
            self.session_stats['min_latency_ms'] = measurement.latency_ms
            
                                   
        if measurement.mobility_kmh > self.session_stats['max_mobility_kmh']:
            self.session_stats['max_mobility_kmh'] = measurement.mobility_kmh
            
                           
        self.session_stats['handover_count'] += measurement.handover_count
    
    def _calculate_compliance_scores(self, measurement: PerformanceMeasurement) -> Dict[str, float]:
        """
        Calculate DOCOMO compliance scores
        
        Args:
            measurement: Current performance measurement
            
        Returns:
            Compliance scores for each KPI
        """
        compliance = {}
        
                                                      
        throughput_ratio = measurement.throughput_gbps / self.docomo_targets.user_data_rate_gbps
        compliance['throughput'] = min(throughput_ratio, 1.0)
        
                                            
        if measurement.latency_ms <= self.docomo_targets.latency_ms:
            compliance['latency'] = 1.0
        else:
                                                              
            excess_ratio = measurement.latency_ms / self.docomo_targets.latency_ms
            compliance['latency'] = max(0.0, 1.0 / excess_ratio)
        
                                
        compliance['reliability'] = measurement.reliability_score / self.docomo_targets.reliability
        
                                               
        if measurement.mobility_kmh <= self.docomo_targets.mobility_kmh:
            compliance['mobility'] = 1.0
        else:
                                                         
            excess_ratio = measurement.mobility_kmh / self.docomo_targets.mobility_kmh
            compliance['mobility'] = max(0.5, 1.0 / excess_ratio)
        
                                                                                           
        baseline_energy_w = 1.0                
        baseline_efficiency = self.docomo_targets.user_data_rate_gbps / baseline_energy_w              
        current_efficiency = 0.0
        if measurement.energy_consumption_w > 0:
            current_efficiency = measurement.throughput_gbps / measurement.energy_consumption_w
                                         
        efficiency_improvement = current_efficiency / max(baseline_efficiency, 1e-6)
        target_improvement = max(self.docomo_targets.energy_efficiency_improvement, 1e-6)
        compliance['energy'] = float(min(max(efficiency_improvement / target_improvement, 0.0), 1.0))
        
                                               
                                                                         
                                                                                                  
        weights = {
            'throughput': 0.35,
            'latency': 0.35,
            'reliability': 0.18,
            'mobility': 0.10,
            'energy': 0.02
        }
        
        compliance['overall'] = sum(
            weights[metric] * score for metric, score in compliance.items()
        )
        
        return compliance
    
    def _detect_anomalies(self, measurement: PerformanceMeasurement):
        """Detect performance anomalies"""
        if not self.anomaly_detection_enabled:
            return
                                                                  
        if (self.session_stats.get('total_measurements', 0) < self.anomaly_warmup_samples or
            len(self.throughput_history) < 30 or len(self.latency_history) < 30):
            return
            
                                      
        recent_throughput = np.array(list(self.throughput_history)[-30:])
        throughput_mean = np.mean(recent_throughput)
        throughput_std = np.std(recent_throughput)
        
                           
        if throughput_std > 1e-9:
            z_score = abs(measurement.throughput_gbps - throughput_mean) / throughput_std
            if z_score > self.anomaly_threshold:
                self._log_anomaly('throughput', measurement, z_score)
        
                                       
        recent_latency = np.array(list(self.latency_history)[-30:])
        latency_mean = np.mean(recent_latency)
        latency_std = np.std(recent_latency)
        
        if latency_std > 1e-9:
            z_score = abs(measurement.latency_ms - latency_mean) / latency_std
            if z_score > self.anomaly_threshold:
                self._log_anomaly('latency', measurement, z_score)
    
    def _log_anomaly(self, metric: str, measurement: PerformanceMeasurement, z_score: float):
        """Log detected anomaly"""
        self.session_stats['error_count'] += 1
        if self.anomaly_log_enabled:
            print(f" ANOMALY DETECTED: {metric} Z-score: {z_score:.2f} at {measurement.timestamp}")
    
    def _update_performance_trend(self):
        """Update overall performance trend"""
        if len(self.compliance_scores['overall']) < 10:
            self.performance_trend = 'insufficient_data'
            return
            
        recent_scores = list(self.compliance_scores['overall'])[-10:]
        older_scores = list(self.compliance_scores['overall'])[-20:-10] if len(self.compliance_scores['overall']) >= 20 else []
        
        if len(older_scores) == 0:
            self.performance_trend = 'stable'
            return
            
        recent_avg = np.mean(recent_scores)
        older_avg = np.mean(older_scores)
        
        if recent_avg > older_avg * 1.05:
            self.performance_trend = 'improving'
        elif recent_avg < older_avg * 0.95:
            self.performance_trend = 'degrading'
        else:
            self.performance_trend = 'stable'
    
    def get_current_kpis(self) -> Dict[str, Any]:
        """
        Get current KPI measurements
        
        Returns:
            Current KPI values and statistics
        """
        if not self.measurements:
                                                              
            return {
                'timestamp': datetime.now().isoformat(),
                'current_throughput_gbps': 0.0,
                'current_latency_ms': float('inf'),
                'current_reliability': 0.0,
                'current_mobility_kmh': 0.0,
                'current_energy_w': 0.0,
                'avg_throughput_gbps': 0.0,
                'avg_latency_ms': float('inf'),
                'avg_reliability': 0.0,
                'peak_throughput_gbps': 0.0,
                'min_latency_ms': float('inf'),
                'max_mobility_kmh': 0.0,
                'total_measurements': 0,
                'session_duration_min': 0.0,
                'handover_count': 0,
                'error_count': 0
            }
        
        latest = self.measurements[-1]
        
        return {
            'timestamp': latest.timestamp.isoformat(),
            'current_throughput_gbps': latest.throughput_gbps,
            'current_latency_ms': latest.latency_ms,
            'current_reliability': latest.reliability_score,
            'current_mobility_kmh': latest.mobility_kmh,
            'current_energy_w': latest.energy_consumption_w,
            
                              
            'avg_throughput_gbps': np.mean(self.throughput_history) if self.throughput_history else 0,
            'avg_latency_ms': np.mean(self.latency_history) if self.latency_history else 0,
            'avg_reliability': np.mean(self.reliability_history) if self.reliability_history else 0,
            
                         
            'peak_throughput_gbps': self.session_stats['peak_throughput_gbps'],
            'min_latency_ms': self.session_stats['min_latency_ms'],
            'max_mobility_kmh': self.session_stats['max_mobility_kmh'],
            
                                
            'total_measurements': self.session_stats['total_measurements'],
            'session_duration_min': (datetime.now() - self.session_stats['start_time']).total_seconds() / 60,
            'handover_count': self.session_stats['handover_count'],
            'error_count': self.session_stats['error_count']
        }
    
    def get_compliance_score(self) -> Dict[str, float]:
        """
        Get current compliance scores
        
        Returns:
            Compliance scores for all KPIs
        """
        if not self.compliance_scores['overall']:
            return {'overall': 0.0}
        
        compliance = {}
        for metric, scores in self.compliance_scores.items():
            if scores and len(scores) > 0:
                compliance[f'{metric}_current'] = scores[-1]
                compliance[f'{metric}_avg'] = float(np.mean(scores))
                compliance[f'{metric}_min'] = float(np.min(scores))
                compliance[f'{metric}_max'] = float(np.max(scores))
            else:
                compliance[f'{metric}_current'] = 0.0
                compliance[f'{metric}_avg'] = 0.0
                compliance[f'{metric}_min'] = 0.0
                compliance[f'{metric}_max'] = 0.0
        
                                      
                                               
        root_cfg = self.config if isinstance(self.config, dict) else {}
        try:
            docomo_threshold = float(root_cfg.get('docomo_6g_system', {}).get('validation', {}).get('kpi_compliance_threshold', 0.95))
        except Exception:
            docomo_threshold = 0.95
        
        compliance['docomo_compliant'] = compliance.get('overall_current', 0.0) >= docomo_threshold
        compliance['compliance_trend'] = self.performance_trend
        
        return compliance
    
    def get_docomo_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive DOCOMO compliance report
        
        Returns:
            Detailed compliance report
        """
        current_kpis = self.get_current_kpis()
        compliance = self.get_compliance_score()
        
                                     
        achievement_rates = {}
        if self.throughput_history:
            target_throughput = self.docomo_targets.user_data_rate_gbps
            achievements = [t >= target_throughput for t in self.throughput_history]
            achievement_rates['throughput_achievement_rate'] = np.mean(achievements)
        
        if self.latency_history:
            target_latency = self.docomo_targets.latency_ms
            achievements = [l <= target_latency for l in self.latency_history]
            achievement_rates['latency_achievement_rate'] = np.mean(achievements)
        
        return {
            'report_timestamp': datetime.now().isoformat(),
            'docomo_compliance': compliance.get('docomo_compliant', False),
            'overall_compliance_score': compliance.get('overall_current', 0.0),
            'performance_trend': self.performance_trend,
            
                                                   
            'performance_vs_targets': {
                'throughput': {
                    'current_gbps': current_kpis.get('current_throughput_gbps', 0),
                    'target_gbps': self.docomo_targets.user_data_rate_gbps,
                    'achievement_rate': achievement_rates.get('throughput_achievement_rate', 0)
                },
                'latency': {
                    'current_ms': current_kpis.get('current_latency_ms', 0),
                    'target_ms': self.docomo_targets.latency_ms,
                    'achievement_rate': achievement_rates.get('latency_achievement_rate', 0)
                },
                'mobility': {
                    'max_supported_kmh': current_kpis.get('max_mobility_kmh', 0),
                    'target_kmh': self.docomo_targets.mobility_kmh
                }
            },
            
                             
            'session_summary': {
                'duration_min': current_kpis.get('session_duration_min', 0),
                'total_measurements': current_kpis.get('total_measurements', 0),
                'peak_throughput_gbps': current_kpis.get('peak_throughput_gbps', 0),
                'handover_efficiency': self._calculate_handover_efficiency(),
                'system_stability': self._calculate_system_stability()
            },
            
                             
            'recommendations': self._generate_recommendations(compliance, current_kpis)
        }
    
    def _calculate_handover_efficiency(self) -> float:
        """Calculate handover efficiency metric"""
        if not self.measurements:
            return 0.0
            
        total_handovers = sum(m.handover_count for m in self.measurements)
        total_time = len(self.measurements)                                        
        
        if total_time == 0:
            return 0.0
            
                                                    
        handover_rate = total_handovers / total_time
        efficiency = max(0.0, 1.0 - handover_rate / 10.0)                    
        return efficiency
    
    def _calculate_system_stability(self) -> float:
        """Calculate overall system stability metric"""
        if len(self.throughput_history) < 10:
            return 0.0
        
                                                          
        throughput_std = np.std(self.throughput_history)
        throughput_mean = np.mean(self.throughput_history)
        
        if throughput_mean == 0:
            return 0.0
            
        cv = throughput_std / throughput_mean
        stability = max(0.0, 1.0 - cv)                               
        return stability
    
    def _generate_recommendations(self, compliance: Dict[str, float], kpis: Dict[str, Any]) -> List[str]:
        """Generate performance improvement recommendations"""
        recommendations = []
        
                                    
        if compliance.get('throughput_current', 0) < 0.8:
            recommendations.append("Consider using higher frequency bands (sub-THz) for increased throughput")
            recommendations.append("Optimize OAM mode selection for current distance and conditions")
        
                                   
        if compliance.get('latency_current', 0) < 0.8:
            recommendations.append("Enable predictive handover to reduce switching latency")
            recommendations.append("Optimize beam tracking algorithms for faster response")
        
                                  
        if kpis.get('max_mobility_kmh', 0) < self.docomo_targets.mobility_kmh * 0.8:
            recommendations.append("Enhance mobility prediction algorithms")
            recommendations.append("Implement advanced Doppler compensation")
        
                             
        if compliance.get('overall_current', 0) < 0.9:
            recommendations.append("Enable multi-objective optimization with DOCOMO priorities")
            recommendations.append("Consider adaptive frequency band selection")
        
                                
        if self._calculate_system_stability() < 0.8:
            recommendations.append("Implement more aggressive handover penalties for stability")
            recommendations.append("Enable atmospheric condition prediction")
        
        return recommendations if recommendations else ["System performing within DOCOMO targets"]

    def save_report(self, filepath: str):
        """Save DOCOMO compliance report to file"""
        report = self.get_docomo_report()
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f" DOCOMO compliance report saved to {filepath}")

    def reset_session(self):
        """Reset session tracking"""
        self.measurements.clear()
        self.throughput_history.clear()
        self.latency_history.clear() 
        self.reliability_history.clear()
        self.energy_history.clear()
        self.mobility_history.clear()
        
        for scores in self.compliance_scores.values():
            scores.clear()
        
        self.session_stats = {
            'start_time': datetime.now(),
            'total_measurements': 0,
            'peak_throughput_gbps': 0.0,
            'min_latency_ms': float('inf'),
            'max_mobility_kmh': 0.0,
            'handover_count': 0,
            'error_count': 0
        }
        
        print(" DOCOMO KPI tracker session reset")
