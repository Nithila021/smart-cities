# app/services/analysis.py
from datetime import datetime, timedelta
import numpy as np
import polars as pl
from app.core.state import cached_data
from app.services.clustering import predict_dbscan_cluster, predict_demographic_zone, get_crime_density_classification
from app.utils.geo import find_nearby_points

# Try to import database functions (optional - falls back to in-memory if unavailable)
try:
    from db_config import (
        get_nearby_crimes as db_get_nearby_crimes,
        calculate_risk_score as db_calculate_risk_score,
        get_demographic_weights as db_get_demographic_weights
    )
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False

# Default demographic weights (fallback if database unavailable)
DEFAULT_DEMOGRAPHIC_WEIGHTS = {
    'general': {'sexual': 1.0, 'harassment': 1.0, 'assault': 1.0, 'theft': 1.0, 'robbery': 1.0},
    'women': {'sexual': 1.8, 'harassment': 1.5, 'assault': 1.3, 'theft': 1.0, 'robbery': 1.2},
    'children': {'sexual': 2.0, 'harassment': 1.3, 'assault': 1.5, 'theft': 0.8, 'robbery': 1.0},
    'elderly': {'sexual': 1.2, 'harassment': 1.2, 'assault': 1.4, 'theft': 1.5, 'robbery': 1.6},
}

# Flag to control whether to use database (can be toggled)
USE_DATABASE = False  # Set to True when ready to use PostgreSQL


def get_demographic_weights(demographic_group: str = 'general') -> dict:
    """Get demographic impact weights, preferring database if available."""
    if USE_DATABASE and DB_AVAILABLE:
        try:
            weights = db_get_demographic_weights(demographic_group)
            if weights:
                return weights
        except Exception:
            pass

    return DEFAULT_DEMOGRAPHIC_WEIGHTS.get(demographic_group, DEFAULT_DEMOGRAPHIC_WEIGHTS['general'])


def apply_demographic_weighting(base_score: float, crime_counts: dict, demographic_group: str = 'general') -> float:
    """Apply demographic-specific weighting to a safety score."""
    if demographic_group == 'general' or not crime_counts:
        return base_score

    weights = get_demographic_weights(demographic_group)

    # Calculate weighted adjustment
    total_crimes = sum(crime_counts.values())
    if total_crimes == 0:
        return base_score

    weighted_sum = sum(
        count * weights.get(category, 1.0)
        for category, count in crime_counts.items()
    )

    # Average weight applied
    avg_weight = weighted_sum / total_crimes

    # Adjust score: higher weight means lower safety for this demographic
    # Scale factor: if avg_weight > 1, reduce safety score proportionally
    adjustment = (avg_weight - 1.0) * 10  # 10 points per 1.0 weight increase
    adjusted_score = max(0, min(100, base_score - adjustment))

    return round(adjusted_score, 1)


def analyze_safety(lat, lon, use_db=None, demographic_group='general'):
    """Perform safety analysis for given coordinates using Polars."""
    # Import locally to avoid circular dependencies if any
    from app.services.data_loader import initialize_data
    
    # Determine whether to use database
    should_use_db = use_db if use_db is not None else (USE_DATABASE and DB_AVAILABLE)

    df = cached_data['df']

    if df is None:
        initialize_data()
        df = cached_data['df']

    # Get crime zone
    point = np.array([[lat, lon]])
    scaler = cached_data.get('crime_scaler')
    if scaler is not None:
        point_scaled = scaler.transform(point)
        zone = cached_data['crime_clusters'].predict(point_scaled)[0]
    else:
        zone = cached_data['crime_clusters'].predict(point)[0]

    # Get safety score - prefer database if available and enabled
    if should_use_db:
        try:
            safety_score = db_calculate_risk_score(lat, lon, radius_meters=3000)
            safety_score = 100 - safety_score
        except Exception:
            safety_score = cached_data['zone_safety_scores'].get(zone, 50)
    else:
        safety_score = cached_data['zone_safety_scores'].get(zone, 50)

    # Get dominant crime information
    dominant_crime_info = cached_data['zone_dominant_crimes'].get(zone, {
        'dominant_crime': 'Unknown',
        'common_crimes': {}
    })

    # Find nearby crimes (3km radius) - prefer database if available
    if should_use_db:
        try:
            nearby_list = db_get_nearby_crimes(lat, lon, radius_meters=3000)
            # If standard list of dicts, make Polars DF
            if nearby_list:
                nearby = pl.DataFrame(nearby_list)
            else:
                nearby = pl.DataFrame()
        except Exception:
            nearby = find_nearby_points(df, lat, lon, 3)
    else:
        nearby = find_nearby_points(df, lat, lon, 3)
    
    # Time of day analysis
    time_analysis = {}
    if 'occurred_at' in df.columns: # Was cmplnt_fr_datetime/parsed_date. New cleaner uses occurred_at
        # Check datetime column name
        dt_col = 'occurred_at' if 'occurred_at' in nearby.columns else 'cmplnt_fr_dt' 
        # Actually our cleaner standardized on 'occurred_at'.
        
        if dt_col in nearby.columns and nearby.height > 0:
            thirty_days_ago = datetime.now() - timedelta(days=30)
            
            # Filter recent
            # Polars datetime filter
            recent = nearby.filter(pl.col(dt_col) >= thirty_days_ago) # works if dt_col is proper Datetime type
            
            # Hour analysis
            # We have 'hour_of_day' from cleaner. simpler.
            if 'hour_of_day' in nearby.columns:
                 hours = nearby['hour_of_day']
                 
                 time_analysis = {
                     'total_recent': recent.height,
                     'time_of_day': {
                         'morning': nearby.filter((pl.col('hour_of_day') >= 6) & (pl.col('hour_of_day') < 12)).height,
                         'afternoon': nearby.filter((pl.col('hour_of_day') >= 12) & (pl.col('hour_of_day') < 18)).height,
                         'evening': nearby.filter((pl.col('hour_of_day') >= 18) & (pl.col('hour_of_day') < 24)).height,
                         'night': nearby.filter(pl.col('hour_of_day') < 6).height
                     }
                 }
    
    # Get DBSCAN cluster info
    dbscan_cluster = predict_dbscan_cluster(lat, lon)
    dbscan_info = None
    if dbscan_cluster is not None and dbscan_cluster != -1: # -1 is outlier
        dbscan_data = cached_data.get('dbscan_clusters')
        if dbscan_data and dbscan_cluster in dbscan_data['dominant_crimes']:
            dbscan_info = dbscan_data['dominant_crimes'][dbscan_cluster]
    
    # Get demographics for zone prediction
    demographics = {}
    if nearby.height > 0:
        for col in ['vic_age_group', 'vic_race', 'vic_sex']:
             if col in nearby.columns:
                 # Mode in Polars
                 modes = nearby[col].mode()
                 if len(modes) > 0:
                     demographics[col] = modes[0]
                 else:
                     demographics[col] = None

    demographic_zone = predict_demographic_zone(lat, lon, demographics)
    demographic_info = None
    if demographic_zone is not None:
        demo_data = cached_data.get('victim_demographic_zones')
        if demo_data and demographic_zone in demo_data['zones']:
            demographic_info = demo_data['zones'][demographic_zone]
    
    # Get crime density classification
    density_info = get_crime_density_classification(lat, lon)

    # Get crime category counts for demographic weighting
    crime_types = {}
    if nearby.height > 0:
        # Value counts to dict
        vc = nearby['crime_type'].value_counts()
        crime_types = {row['crime_type']: row['count'] for row in vc.iter_rows(named=True)}

    crime_categories = {}
    if 'crime_category' in nearby.columns and nearby.height > 0:
        vc = nearby['crime_category'].value_counts()
        crime_categories = {row['crime_category']: row['count'] for row in vc.iter_rows(named=True)}

    # Apply demographic-specific weighting to safety score
    adjusted_safety_score = apply_demographic_weighting(
        safety_score, crime_categories, demographic_group
    )

    # Prepare final analysis
    analysis = {
        'safety_score': round(adjusted_safety_score, 1),
        'base_safety_score': round(safety_score, 1),  # Unadjusted score for reference
        'demographic_group': demographic_group,
        'zone': int(zone),
        'dominant_crime': dominant_crime_info['dominant_crime'],
        'common_crimes': dominant_crime_info['common_crimes'],
        'nearby_crime_count': nearby.height,
        'crime_types': crime_types,
        'crime_categories': crime_categories,
        'lat': lat,
        'lon': lon,
        'density': density_info,
    }
    
    if time_analysis:
        analysis['time_analysis'] = time_analysis
    
    if dbscan_info and dbscan_cluster is not None:
        analysis['dbscan_cluster'] = {
            'cluster_id': int(dbscan_cluster),
            'dominant_crime': dbscan_info['dominant_crime'],
            'common_crimes': dbscan_info['common_crimes'],
            'crime_count': dbscan_info['crime_count']
        }

    if demographic_info and demographic_zone is not None:
        analysis['demographic_zone'] = {
            'zone_id': int(demographic_zone),
            'profiles': demographic_info['concentration_scores'],
            'crime_count': demographic_info['crime_count']
        }
    
    return analysis


def analyze_amenities(lat, lon, radius_km=1):
    """Analyze amenities near a location"""
    # Import locally to avoid circular dependencies
    from app.services.data_loader import load_amenity_data

    amenities_df = load_amenity_data()
    
    # Find nearby amenities
    nearby = find_nearby_points(amenities_df, lat, lon, radius_km)
    
    type_counts = {}
    if nearby.height > 0 and 'type' in nearby.columns:
        vc = nearby['type'].value_counts()
        type_counts = {row['type']: row['count'] for row in vc.iter_rows(named=True)}
    
    # Get closest amenities of each type
    closest_amenities = {}
    if nearby.height > 0 and 'type' in nearby.columns:
        # Sort by distance
        nearby_sorted = nearby.sort('distance')
        
        # Unique types
        unique_types = nearby_sorted['type'].unique().to_list()
        
        for amenity_type in unique_types:
            # Filter for type and take first
            closest = nearby_sorted.filter(pl.col('type') == amenity_type)[0] # Row as tuple/struct
            # or .row(0, named=True)
            closest_dict = nearby_sorted.filter(pl.col('type') == amenity_type).row(0, named=True)
            
            closest_amenities[amenity_type] = {
                'name': closest_dict.get('name', f"Unnamed {amenity_type}"),
                'distance_km': round(closest_dict['distance'], 2),
                'latitude': closest_dict['latitude'],
                'longitude': closest_dict['longitude']
            }
    
    return {
        'nearby_count': nearby.height,
        'type_counts': type_counts,
        'closest_amenities': closest_amenities
    }
