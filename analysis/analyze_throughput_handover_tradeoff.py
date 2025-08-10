#!/usr/bin/env python3
"""
Analysis of Throughput vs Handover Tradeoff in OAM 6G System

This script analyzes the fundamental tradeoff between throughput and handover frequency
in orbital angular momentum (OAM) based wireless communication systems.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment.oam_env import OAM_Env
from models.agent import Agent
import pandas as pd
from typing import Dict, List, Tuple

def analyze_throughput_handover_tradeoff():
    """Analyze the throughput vs handover tradeoff in OAM 6G system."""
    
    print("Analyzing Throughput vs Handover Tradeoff in OAM 6G System")
    print("=" * 60)
    
    # Initialize environment
    env = OAM_Env({'oam': {'min_mode': 1, 'max_mode': 8}})
    
    # Data collection
    data = {
        'throughput': [],
        'handovers': [],
        'modes': [],
        'sinr': [],
        'distance': [],
        'episode': []
    }
    
    # Run multiple episodes to collect data
    num_episodes = 50
    steps_per_episode = 100
    
    print(f"📊 Collecting data from {num_episodes} episodes...")
    
    for episode in range(num_episodes):
        state_tuple = env.reset()
        state = state_tuple[0] if isinstance(state_tuple, tuple) else state_tuple
        
        episode_handovers = 0
        episode_throughputs = []
        episode_modes = []
        episode_sinrs = []
        episode_distances = []
        
        for step in range(steps_per_episode):
            # Take random action to explore different scenarios
            action = np.random.randint(0, 3)  # 0: STAY, 1: UP, 2: DOWN
            
            next_state_tuple, reward, done, truncated, info = env.step(action)
            next_state = next_state_tuple[0] if isinstance(next_state_tuple, tuple) else next_state_tuple
            
            # Extract metrics
            throughput = info['throughput']
            handover_count = info['handovers']
            current_mode = info['mode']
            sinr = info['sinr']
            distance = np.linalg.norm(info['position'])
            
            # Track handovers (increment if mode changed)
            if step > 0 and current_mode != episode_modes[-1]:
                episode_handovers += 1
            
            episode_throughputs.append(throughput)
            episode_modes.append(current_mode)
            episode_sinrs.append(sinr)
            episode_distances.append(distance)
            
            state = next_state
            if done:
                break
        
        # Store episode data
        data['throughput'].extend(episode_throughputs)
        data['handovers'].extend([episode_handovers] * len(episode_throughputs))
        data['modes'].extend(episode_modes)
        data['sinr'].extend(episode_sinrs)
        data['distance'].extend(episode_distances)
        data['episode'].extend([episode] * len(episode_throughputs))
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(data)
    
    print("\n📈 ANALYSIS RESULTS:")
    print("-" * 40)
    
    # 1. Basic Statistics
    print(f"📊 Throughput Statistics:")
    print(f"   Mean: {df['throughput'].mean():.0f} bps")
    print(f"   Std:  {df['throughput'].std():.0f} bps")
    print(f"   Min:  {df['throughput'].min():.0f} bps")
    print(f"   Max:  {df['throughput'].max():.0f} bps")
    
    print(f"\n🔄 Handover Statistics:")
    print(f"   Mean handovers per episode: {df['handovers'].mean():.1f}")
    print(f"   Total handovers: {df['handovers'].sum()}")
    print(f"   Handover rate: {df['handovers'].sum() / num_episodes:.2f} per episode")
    
    # 2. Mode Analysis
    print(f"\n🎯 OAM Mode Analysis:")
    mode_counts = df['modes'].value_counts().sort_index()
    for mode, count in mode_counts.items():
        percentage = (count / len(df)) * 100
        print(f"   Mode {mode}: {count} times ({percentage:.1f}%)")
    
    # 3. Throughput vs Handover Correlation
    correlation = df['throughput'].corr(df['handovers'])
    print(f"\n📊 Throughput-Handover Correlation: {correlation:.3f}")
    
    # 4. Mode-specific Analysis
    print(f"\n🎯 Mode-Specific Performance:")
    for mode in sorted(df['modes'].unique()):
        mode_data = df[df['modes'] == mode]
        avg_throughput = mode_data['throughput'].mean()
        avg_sinr = mode_data['sinr'].mean()
        print(f"   Mode {mode}: Avg Throughput = {avg_throughput:.0f} bps, Avg SINR = {avg_sinr:.1f} dB")
    
    # 5. Distance Impact
    print(f"\n📏 Distance Impact:")
    distance_correlation = df['throughput'].corr(df['distance'])
    print(f"   Throughput-Distance Correlation: {distance_correlation:.3f}")
    
    # Create visualizations
    create_tradeoff_visualizations(df)
    
    return df

def create_tradeoff_visualizations(df: pd.DataFrame):
    """Create visualizations for the throughput-handover tradeoff analysis."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Throughput vs Handover Scatter
    axes[0, 0].scatter(df['handovers'], df['throughput'], alpha=0.6, s=20)
    axes[0, 0].set_xlabel('Handovers per Episode')
    axes[0, 0].set_ylabel('Throughput (bps)')
    axes[0, 0].set_title('Throughput vs Handover Frequency')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Throughput Distribution by Mode
    for mode in sorted(df['modes'].unique()):
        mode_data = df[df['modes'] == mode]
        axes[0, 1].hist(mode_data['throughput'], alpha=0.7, label=f'Mode {mode}', bins=20)
    axes[0, 1].set_xlabel('Throughput (bps)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Throughput Distribution by OAM Mode')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. SINR vs Throughput
    axes[1, 0].scatter(df['sinr'], df['throughput'], alpha=0.6, s=20)
    axes[1, 0].set_xlabel('SINR (dB)')
    axes[1, 0].set_ylabel('Throughput (bps)')
    axes[1, 0].set_title('SINR vs Throughput')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Distance vs Throughput
    axes[1, 1].scatter(df['distance'], df['throughput'], alpha=0.6, s=20)
    axes[1, 1].set_xlabel('Distance (m)')
    axes[1, 1].set_ylabel('Throughput (bps)')
    axes[1, 1].set_title('Distance vs Throughput')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('throughput_handover_tradeoff_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\n📊 Visualization saved as 'throughput_handover_tradeoff_analysis.png'")
    
    # Create detailed tradeoff analysis
    create_detailed_tradeoff_analysis(df)

def create_detailed_tradeoff_analysis(df: pd.DataFrame):
    """Create detailed analysis of the throughput-handover tradeoff."""
    
    print(f"\n🔍 DETAILED TRADEOFF ANALYSIS:")
    print("=" * 50)
    
    # 1. Handover Impact on Throughput
    handover_groups = df.groupby('handovers')['throughput'].agg(['mean', 'std', 'count'])
    print(f"\n📊 Handover Impact on Throughput:")
    for handovers, stats in handover_groups.iterrows():
        print(f"   {handovers} handovers: {stats['mean']:.0f} ± {stats['std']:.0f} bps (n={stats['count']})")
    
    # 2. Mode Switching Analysis
    print(f"\n🔄 Mode Switching Analysis:")
    mode_transitions = []
    for episode in df['episode'].unique():
        episode_data = df[df['episode'] == episode]
        modes = episode_data['modes'].values
        for i in range(1, len(modes)):
            if modes[i] != modes[i-1]:
                mode_transitions.append((modes[i-1], modes[i]))
    
    if mode_transitions:
        transition_counts = pd.Series(mode_transitions).value_counts().head(10)
        print(f"   Most common mode transitions:")
        for (from_mode, to_mode), count in transition_counts.items():
            print(f"     {from_mode} → {to_mode}: {count} times")
    
    # 3. Optimal Mode Selection
    print(f"\n🎯 Optimal Mode Selection:")
    mode_performance = df.groupby('modes').agg({
        'throughput': ['mean', 'std'],
        'sinr': 'mean',
        'handovers': 'mean'
    }).round(2)
    
    for mode in sorted(df['modes'].unique()):
        throughput_mean = mode_performance.loc[mode, ('throughput', 'mean')]
        throughput_std = mode_performance.loc[mode, ('throughput', 'std')]
        sinr_mean = mode_performance.loc[mode, ('sinr', 'mean')]
        print(f"   Mode {mode}: {throughput_mean:.0f} ± {throughput_std:.0f} bps, SINR: {sinr_mean:.1f} dB")
    
    # 4. Tradeoff Recommendations
    print(f"\n💡 TRADEOFF RECOMMENDATIONS:")
    print("-" * 40)
    
    # Calculate efficiency metric (throughput / handover_cost)
    handover_cost = 0.1  # Assume 10% throughput penalty per handover
    df['efficiency'] = df['throughput'] / (1 + df['handovers'] * handover_cost)
    
    best_efficiency = df['efficiency'].max()
    best_scenario = df.loc[df['efficiency'].idxmax()]
    
    print(f"   Best efficiency scenario:")
    print(f"     Throughput: {best_scenario['throughput']:.0f} bps")
    print(f"     Handovers: {best_scenario['handovers']} per episode")
    print(f"     Mode: {best_scenario['modes']}")
    print(f"     Efficiency: {best_scenario['efficiency']:.0f} bps")
    
    # 5. System Design Implications
    print(f"\n🏗️ SYSTEM DESIGN IMPLICATIONS:")
    print("-" * 40)
    
    avg_throughput = df['throughput'].mean()
    avg_handovers = df['handovers'].mean()
    
    print(f"   Current System Performance:")
    print(f"     Average Throughput: {avg_throughput:.0f} bps")
    print(f"     Average Handovers: {avg_handovers:.1f} per episode")
    print(f"     Throughput-Handover Ratio: {avg_throughput/avg_handovers:.0f} bps/handover")
    
    # Calculate correlation for recommendations
    correlation = df['throughput'].corr(df['handovers'])
    
    # Recommendations based on analysis
    if correlation > 0.1:
        print(f"   ⚠️  High handover frequency reduces throughput")
        print(f"   💡 Recommendation: Implement handover prediction and optimization")
    elif correlation < -0.1:
        print(f"   ✅ Handovers improve throughput (aggressive mode switching)")
        print(f"   💡 Recommendation: Encourage adaptive mode selection")
    else:
        print(f"   ➖ Minimal correlation between handovers and throughput")
        print(f"   💡 Recommendation: Focus on other optimization factors")

if __name__ == "__main__":
    # Run the analysis
    df = analyze_throughput_handover_tradeoff()
    
    print(f"\n✅ Analysis complete! Generated visualizations and recommendations.") 