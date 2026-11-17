"""
Energy Optimization Recommendations & Tariff Impact Calculator
Parses facility recommendations, categorizes operational opportunities,
and provides transparent cost & demand scenario modeling.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any

RECOMMENDATION_CATEGORIES = {
    'Elevated overnight baseline': {
        'category': 'Baseload & Overnight Optimization',
        'priority': 'High',
        'badge_color': 'coral',
        'icon': '🌙',
        'action_summary': 'Audit after-hours HVAC setback, interior lighting, and standby plug loads.'
    },
    'Weekend consumption present': {
        'category': 'Occupancy & Schedule Alignment',
        'priority': 'Medium',
        'badge_color': 'amber',
        'icon': '📅',
        'action_summary': 'Verify building automation weekend scheduling and ventilation setback.'
    },
    'Peak Demand Alert': {
        'category': 'Peak Load Shifting',
        'priority': 'High',
        'badge_color': 'purple',
        'icon': '⚡',
        'action_summary': 'Investigate pre-cooling or staggering major electrical equipment start times.'
    }
}

def enrich_recommendations(df_rec: pd.DataFrame, df_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich raw recommendations with category metadata, priority, and building context.
    """
    if df_rec.empty:
        return pd.DataFrame()
    
    enriched = df_rec.copy()
    
    # Map pattern attributes
    categories, priorities, badge_colors, icons, action_summaries = [], [], [], [], []
    for _, row in enriched.iterrows():
        pattern = row.get('pattern', '')
        info = RECOMMENDATION_CATEGORIES.get(pattern, {
            'category': 'Operational Opportunity',
            'priority': 'Medium',
            'badge_color': 'blue',
            'icon': '💡',
            'action_summary': 'Investigate equipment runtime and controls.'
        })
        categories.append(info['category'])
        priorities.append(info['priority'])
        badge_colors.append(info['badge_color'])
        icons.append(info['icon'])
        action_summaries.append(info['action_summary'])
        
    enriched['category'] = categories
    enriched['priority'] = priorities
    enriched['badge_color'] = badge_colors
    enriched['icon'] = icons
    enriched['action_summary'] = action_summaries
    
    # Merge building statistics if available
    if not df_summary.empty and 'building_id' in df_summary.columns:
        enriched = enriched.merge(
            df_summary[['building_id', 'mean_consumption', 'maximum_consumption']],
            on='building_id',
            how='left'
        )
        
    return enriched

def calculate_tariff_scenarios(
    mean_hourly_consumption: float,
    peak_demand: float,
    rate_per_kwh: float = 0.14,
    peak_demand_charge_per_kw: float = 12.0,
    target_reduction_pct: float = 5.0
) -> Dict[str, Any]:
    """
    Transparent energy cost & demand scenario calculator.
    Uses explicitly labeled user-supplied tariff rates and assumptions.
    All calculations are deterministic and clearly distinguished from verified historical readings.
    """
    # Annual consumption estimate = mean hourly consumption * 8760 hours
    annual_kwh = mean_hourly_consumption * 8760.0
    annual_energy_cost = annual_kwh * rate_per_kwh
    
    # Monthly peak demand charge estimate = peak_demand * demand_rate * 12
    annual_demand_cost = peak_demand * peak_demand_charge_per_kw * 12.0
    total_annual_cost = annual_energy_cost + annual_demand_cost
    
    # Potential reduction scenario
    reduction_fraction = target_reduction_pct / 100.0
    potential_kwh_avoided = annual_kwh * reduction_fraction
    potential_energy_cost_avoided = potential_kwh_avoided * rate_per_kwh
    
    return {
        'annual_kwh_baseline': annual_kwh,
        'annual_energy_cost_baseline': annual_energy_cost,
        'annual_demand_cost_baseline': annual_demand_cost,
        'total_annual_cost_baseline': total_annual_cost,
        'target_reduction_pct': target_reduction_pct,
        'potential_kwh_avoided': potential_kwh_avoided,
        'potential_cost_avoided': potential_energy_cost_avoided,
        'assumptions': {
            'rate_per_kwh': rate_per_kwh,
            'peak_demand_charge_per_kw': peak_demand_charge_per_kw,
            'hours_per_year': 8760
        }
    }
