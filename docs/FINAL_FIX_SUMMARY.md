# FINAL COMPREHENSIVE FIX PLAN

## Issues Identified:
1. **Lab THz 0 Gbps Issue**: Physics calculator returning 0 for 300+ GHz frequencies
2. **Outdoor Low Performance**: Only 1.6 Gbps vs 5-6 Gbps target  
3. **Outdoor High Handovers**: H:19-59 vs target of minimal handovers
4. **Physics Gaps**: Core calculations not working for THz frequencies

## Root Causes:
1. **SINR Calculation**: Likely producing very low values for THz
2. **Atmospheric Losses**: May be too high for THz frequencies
3. **Handover Thresholds**: Still triggering too frequently
4. **Physics Multipliers**: Not applied correctly in calculation chain

## Comprehensive Solution:
1. **Fix THz Physics Chain**: Debug SINR → throughput calculation
2. **Optimize All Frequency Ranges**: Ensure realistic performance
3. **Reduce Handover Sensitivity**: Further threshold increases  
4. **Validate All 3 Configs**: Comprehensive testing

## Target Results:
- **Lab**: 100+ Gbps consistently with THz bands
- **Outdoor**: 5-6 Gbps with minimal handovers  
- **Indoor**: 10-15 Gbps (already working)
- **Handovers**: Lab H:0, Outdoor H:0-2, Indoor H:0-5

## Implementation Priority:
1. Fix THz physics (CRITICAL)
2. Boost outdoor caps  
3. Reduce handovers further
4. Final testing of all configs
