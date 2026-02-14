from typing import Any, Dict

# Global cached data
cached_data: Dict[str, Any] = {
    'df': None,
    'crime_clusters': None,
    'zone_safety_scores': None,
    'crime_severity': None,
    'zone_dominant_crimes': None,
    'amenities_df': None,
    'dbscan_clusters': None,
    'victim_demographic_zones': None,
    'demographic_feature_importance': None,
    'crime_density_zones': None,
    'cleaning_report': None,
    'crime_scaler': None,  # Added missing key from data_init.py
}
