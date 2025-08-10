#!/usr/bin/env python3
"""
Three-Way Analysis: Handover Count vs Distance vs Throughput

This script analyzes the complex three-way relationship between handover frequency,
distance, and throughput in OAM 6G systems.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment.oam_env import OAM_Env
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

def analyze_three_way_relationship():
    """Analyze the three-way relationship between handover count, distance, and throughput."""
    
    print("Analyzing Three-Way Relationship: Handover Count vs Distance vs Throughput")
    print("=" * 70)
    
    # Initialize environment
    env = OAM_Env({'oam': {'min_mode': 1, 'max_mode': 8}})
    
    # Data collection
    data = {
        'throughput': [],
        'handovers': [],
        'distance': [],
        'modes': [],
        'sinr': [],
        'episode': [],
        'avg_distance': [],
        'distance_variance': []
    }
    
    # Run multiple episodes to collect data
    num_episodes = 100
    steps_per_episode = 50
    
    print(f"Collecting data from {num_episodes} episodes...")
    
    for episode in range(num_episodes):
        state_tuple = env.reset()
        state = state_tuple[0] if isinstance(state_tuple, tuple) else state_tuple
        
        episode_handovers = 0
        episode_throughputs = []
        episode_distances = []
        episode_modes = []
        episode_sinrs = []
        
        for step in range(steps_per_episode):
            # Take random action to explore different scenarios
            action = np.random.randint(0, 3)  # 0: STAY, 1: UP, 2: DOWN
            
            next_state_tuple, reward, done, truncated, info = env.step(action)
            next_state = next_state_tuple[0] if isinstance(next_state_tuple, tuple) else next_state_tuple
            
            # Extract metrics
            throughput = info['throughput']
            current_mode = info['mode']
            sinr = info['sinr']
            distance = np.linalg.norm(info['position'])
            
            # Track handovers (increment if mode changed)
            if step > 0 and current_mode != episode_modes[-1]:
                episode_handovers += 1
            
            episode_throughputs.append(throughput)
            episode_distances.append(distance)
            episode_modes.append(current_mode)
            episode_sinrs.append(sinr)
            
            state = next_state
            if done:
                break
        
        # Calculate episode-level metrics
        avg_distance = np.mean(episode_distances)
        distance_variance = np.var(episode_distances)
        
        # Store episode data
        data['throughput'].extend(episode_throughputs)
        data['handovers'].extend([episode_handovers] * len(episode_throughputs))
        data['distance'].extend(episode_distances)
        data['modes'].extend(episode_modes)
        data['sinr'].extend(episode_sinrs)
        data['episode'].extend([episode] * len(episode_throughputs))
        data['avg_distance'].extend([avg_distance] * len(episode_throughputs))
        data['distance_variance'].extend([distance_variance] * len(episode_throughputs))
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(data)
    
    print("\nTHREE-WAY ANALYSIS RESULTS:")
    print("-" * 50)
    
    # 1. Correlation Analysis
    print("Correlation Analysis:")
    correlations = {
        'Throughput-Handover': df['throughput'].corr(df['handovers']),
        'Throughput-Distance': df['throughput'].corr(df['distance']),
        'Handover-Distance': df['handovers'].corr(df['distance']),
        'Throughput-AvgDistance': df['throughput'].corr(df['avg_distance']),
        'Handover-DistanceVariance': df['handovers'].corr(df['distance_variance'])
    }
    
    for name, corr in correlations.items():
        print(f"   {name}: {corr:.3f}")
    
    # 2. Distance Impact Analysis
    print(f"\nDistance Impact Analysis:")
    distance_bins = pd.cut(df['distance'], bins=5)
    distance_analysis = df.groupby(distance_bins).agg({
        'throughput': ['mean', 'std'],
        'handovers': 'mean',
        'sinr': 'mean'
    }).round(2)
    
    print("   Distance Range | Avg Throughput | Avg Handovers | Avg SINR")
    print("   " + "-" * 60)
    for distance_range, stats in distance_analysis.iterrows():
        throughput = stats[('throughput', 'mean')]
        handovers = stats[('handovers', 'mean')]
        sinr = stats[('sinr', 'mean')]
        print(f"   {distance_range} | {throughput:.0f} bps | {handovers:.1f} | {sinr:.1f} dB")
    
    # 3. Handover Impact Analysis
    print(f"\nHandover Impact Analysis:")
    handover_bins = pd.cut(df['handovers'], bins=5)
    handover_analysis = df.groupby(handover_bins).agg({
        'throughput': ['mean', 'std'],
        'distance': 'mean',
        'sinr': 'mean'
    }).round(2)
    
    print("   Handover Range | Avg Throughput | Avg Distance | Avg SINR")
    print("   " + "-" * 60)
    for handover_range, stats in handover_analysis.iterrows():
        throughput = stats[('throughput', 'mean')]
        distance = stats[('distance', 'mean')]
        sinr = stats[('sinr', 'mean')]
        print(f"   {handover_range} | {throughput:.0f} bps | {distance:.0f}m | {sinr:.1f} dB")
    
    # 4. Three-Way Interaction Analysis
    print(f"\nThree-Way Interaction Analysis:")
    
    # Create distance-handover groups
    df['distance_group'] = pd.cut(df['distance'], bins=3, labels=['Near', 'Medium', 'Far'])
    df['handover_group'] = pd.cut(df['handovers'], bins=3, labels=['Low', 'Medium', 'High'])
    
    interaction_analysis = df.groupby(['distance_group', 'handover_group'])['throughput'].agg(['mean', 'count']).round(0)
    
    print("   Distance | Handovers | Avg Throughput | Count")
    print("   " + "-" * 50)
    for (distance, handover), stats in interaction_analysis.iterrows():
        throughput = stats['mean']
        count = stats['count']
        print(f"   {distance} | {handover} | {throughput:.0f} bps | {count}")
    
    # Create visualizations
    create_three_way_visualizations(df)
    
    return df

def create_three_way_visualizations(df: pd.DataFrame):
    """Create comprehensive visualizations for the three-way relationship."""
    
    fig = plt.figure(figsize=(20, 15))
    
    # 1. 3D Scatter Plot
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    scatter = ax1.scatter(df['distance'], df['handovers'], df['throughput'], 
                          c=df['throughput'], cmap='viridis', alpha=0.6, s=20)
    ax1.set_xlabel('Distance (m)')
    ax1.set_ylabel('Handovers per Episode')
    ax1.set_zlabel('Throughput (bps)')
    ax1.set_title('3D Relationship: Distance vs Handovers vs Throughput')
    plt.colorbar(scatter, ax=ax1, label='Throughput (bps)')
    
    # 2. Distance vs Throughput (colored by handovers)
    ax2 = fig.add_subplot(2, 3, 2)
    scatter = ax2.scatter(df['distance'], df['throughput'], c=df['handovers'], 
                          cmap='plasma', alpha=0.6, s=20)
    ax2.set_xlabel('Distance (m)')
    ax2.set_ylabel('Throughput (bps)')
    ax2.set_title('Distance vs Throughput (colored by handovers)')
    plt.colorbar(scatter, ax=ax2, label='Handovers per Episode')
    
    # 3. Handovers vs Throughput (colored by distance)
    ax3 = fig.add_subplot(2, 3, 3)
    scatter = ax3.scatter(df['handovers'], df['throughput'], c=df['distance'], 
                          cmap='coolwarm', alpha=0.6, s=20)
    ax3.set_xlabel('Handovers per Episode')
    ax3.set_ylabel('Throughput (bps)')
    ax3.set_title('Handovers vs Throughput (colored by distance)')
    plt.colorbar(scatter, ax=ax3, label='Distance (m)')
    
    # 4. Heatmap of Distance-Handover Interaction
    ax4 = fig.add_subplot(2, 3, 4)
    pivot_table = df.pivot_table(values='throughput', 
                                index=pd.cut(df['distance'], bins=5), 
                                columns=pd.cut(df['handovers'], bins=5), 
                                aggfunc='mean')
    sns.heatmap(pivot_table, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax4)
    ax4.set_title('Distance-Handover Throughput Heatmap')
    ax4.set_xlabel('Handovers per Episode')
    ax4.set_ylabel('Distance (m)')
    
    # 5. Box plots by distance groups
    ax5 = fig.add_subplot(2, 3, 5)
    df['distance_group'] = pd.cut(df['distance'], bins=3, labels=['Near', 'Medium', 'Far'])
    df.boxplot(column='throughput', by='distance_group', ax=ax5)
    ax5.set_title('Throughput Distribution by Distance')
    ax5.set_xlabel('Distance Group')
    ax5.set_ylabel('Throughput (bps)')
    
    # 6. Box plots by handover groups
    ax6 = fig.add_subplot(2, 3, 6)
    df['handover_group'] = pd.cut(df['handovers'], bins=3, labels=['Low', 'Medium', 'High'])
    df.boxplot(column='throughput', by='handover_group', ax=ax6)
    ax6.set_title('Throughput Distribution by Handover Frequency')
    ax6.set_xlabel('Handover Group')
    ax6.set_ylabel('Throughput (bps)')
    
    plt.tight_layout()
    plt.savefig('three_way_relationship_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved as 'three_way_relationship_analysis.png'")
    
    # Create detailed analysis
    create_detailed_three_way_analysis(df)

def create_detailed_three_way_analysis(df: pd.DataFrame):
    """Create detailed analysis of the three-way relationship."""
    
    print(f"\nDETAILED THREE-WAY ANALYSIS:")
    print("=" * 50)
    
    # 1. Optimal Operating Regions
    print(f"\n🎯 Optimal Operating Regions:")
    
    # Find best performance combinations
    df['performance_score'] = df['throughput'] / (1 + df['distance'] / 1000)  # Normalize distance impact
    
    best_scenarios = df.nlargest(10, 'performance_score')
    print("   Top 10 Performance Scenarios:")
    print("   Rank | Throughput | Distance | Handovers | Performance Score")
    print("   " + "-" * 70)
    
    for i, (idx, row) in enumerate(best_scenarios.iterrows(), 1):
        throughput = row['throughput']
        distance = row['distance']
        handovers = row['handovers']
        score = row['performance_score']
        print(f"   {i:2d}   | {throughput:9.0f} | {distance:7.0f}m | {handovers:9.0f} | {score:15.0f}")
    
    # 2. Distance Threshold Analysis
    print(f"\nDistance Threshold Analysis:")
    
    distance_thresholds = [50, 100, 150, 200, 250]
    for threshold in distance_thresholds:
        near_data = df[df['distance'] <= threshold]
        far_data = df[df['distance'] > threshold]
        
        if len(near_data) > 0 and len(far_data) > 0:
            near_throughput = near_data['throughput'].mean()
            far_throughput = far_data['throughput'].mean()
            near_handovers = near_data['handovers'].mean()
            far_handovers = far_data['handovers'].mean()
            
            print(f"   Distance ≤ {threshold}m: {near_throughput:.0f} bps, {near_handovers:.1f} handovers")
            print(f"   Distance > {threshold}m: {far_throughput:.0f} bps, {far_handovers:.1f} handovers")
            print(f"   Throughput difference: {near_throughput - far_throughput:.0f} bps")
            print()
    
    # 3. Handover Efficiency Analysis
    print(f"\nHandover Efficiency Analysis:")
    
    # Calculate efficiency metric
    df['handover_efficiency'] = df['throughput'] / (1 + df['handovers'] * 0.01)  # 1% penalty per handover
    
    efficiency_by_distance = df.groupby(pd.cut(df['distance'], bins=5))['handover_efficiency'].agg(['mean', 'std'])
    print("   Distance Range | Avg Efficiency | Std Efficiency")
    print("   " + "-" * 50)
    
    for distance_range, stats in efficiency_by_distance.iterrows():
        mean_eff = stats['mean']
        std_eff = stats['std']
        print(f"   {distance_range} | {mean_eff:.0f} bps | {std_eff:.0f} bps")
    
    # 4. System Design Recommendations
    print(f"\nSYSTEM DESIGN RECOMMENDATIONS:")
    print("-" * 40)
    
    # Analyze the three-way tradeoff
    distance_impact = df['throughput'].corr(df['distance'])
    handover_impact = df['throughput'].corr(df['handovers'])
    distance_handover_interaction = df['distance'].corr(df['handovers'])
    
    print(f"   Distance Impact on Throughput: {distance_impact:.3f}")
    print(f"   Handover Impact on Throughput: {handover_impact:.3f}")
    print(f"   Distance-Handover Interaction: {distance_handover_interaction:.3f}")
    
    if abs(distance_impact) > 0.5:
        print(f"   ->  Strong distance impact on throughput")
        print(f"   ->  Recommendation: Implement distance-aware mode selection")
    
    if abs(handover_impact) > 0.1:
        print(f"   ->  Handover frequency affects throughput")
        print(f"   ->  Recommendation: Optimize handover strategy")
    else:
        print(f"   -> Handovers don't significantly impact throughput")
        print(f"   -> Recommendation: Use adaptive mode switching")
    
    if abs(distance_handover_interaction) > 0.1:
        print(f"   -> Distance and handovers are correlated")
        print(f"   ->  Recommendation: Implement distance-aware handover prediction")
    else:
        print(f"   -> Distance and handovers are independent")
        print(f"   ->  Recommendation: Optimize distance and handovers separately")
    
    # 5. Performance Optimization Strategy
    print(f"\n PERFORMANCE OPTIMIZATION STRATEGY:")
    print("-" * 40)
    
    # Find optimal operating points
    optimal_near = df[df['distance'] <= 100]['throughput'].max()
    optimal_far = df[df['distance'] > 100]['throughput'].max()
    
    print(f"   Optimal Near Range (≤100m): {optimal_near:.0f} bps")
    print(f"   Optimal Far Range (>100m): {optimal_far:.0f} bps")
    print(f"   Performance Gap: {optimal_near - optimal_far:.0f} bps")
    
    # Recommendations
    if optimal_near - optimal_far > 500000000:  # 500M bps difference
        print(f"   -> Large performance gap between near and far ranges")
        print(f"   -> Recommendation: Focus on distance optimization")
    else:
        print(f"   -> Relatively consistent performance across distances")
        print(f"   -> Recommendation: Focus on handover optimization")

if __name__ == "__main__":
    # Run the analysis
    df = analyze_three_way_relationship()
    
    print(f"\nThree-way analysis complete! Generated comprehensive visualizations and recommendations.") 