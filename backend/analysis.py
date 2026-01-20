# analysis.py
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from data_init import cached_data, initialize_data, load_amenity_data
from models import predict_dbscan_cluster, predict_demographic_zone, get_crime_density_classification
from geo_utils import find_nearby_points


def analyze_safety(lat, lon):
    """Perform safety analysis for given coordinates"""
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
    
    # Get zone safety information
    safety_score = cached_data['zone_safety_scores'].get(zone, 50)
    
    # Get dominant crime information
    dominant_crime_info = cached_data['zone_dominant_crimes'].get(zone, {
        'dominant_crime': 'Unknown',
        'common_crimes': {}
    })
    
    # Find nearby crimes (3km radius)
    nearby = find_nearby_points(df, lat, lon, 3)
    
    # Time of day analysis (if datetime column exists)
    time_analysis = {}
    if 'cmplnt_fr_datetime' in df.columns:
        thirty_days_ago = datetime.now() - timedelta(days=30)
        recent = nearby[nearby['cmplnt_fr_datetime'] >= thirty_days_ago]
        
        if not nearby.empty and 'cmplnt_fr_datetime' in nearby.columns:
            nearby['hour'] = nearby['cmplnt_fr_datetime'].dt.hour
            hour_groups = {
                'morning': range(6, 12),
                'afternoon': range(12, 18),
                'evening': range(18, 24),
                'night': list(range(0, 6))
            }
            
            time_analysis = {
                'total_recent': len(recent),
                'time_of_day': {
                    period: len(nearby[nearby['hour'].isin(hours)])
                    for period, hours in hour_groups.items()
                }
            }
    
    # Get DBSCAN cluster info
    dbscan_cluster = predict_dbscan_cluster(lat, lon)
    dbscan_info = None
    if dbscan_cluster is not None:
        dbscan_data = cached_data.get('dbscan_clusters')
        if dbscan_data and dbscan_cluster in dbscan_data['dominant_crimes']:
            dbscan_info = dbscan_data['dominant_crimes'][dbscan_cluster]
    
    # Get demographic zone info
    demographics = {}
    for col in ['vic_age_group', 'vic_race', 'vic_sex']:
        if col in nearby.columns:
            demographics[col] = nearby[col].mode().iloc[0] if not nearby.empty and not nearby[col].dropna().empty else None
    
    demographic_zone = predict_demographic_zone(lat, lon, demographics)
    demographic_info = None
    if demographic_zone is not None:
        demo_data = cached_data.get('victim_demographic_zones')
        if demo_data and demographic_zone in demo_data['zones']:
            demographic_info = demo_data['zones'][demographic_zone]
    
    # Get crime density classification
    density_info = get_crime_density_classification(lat, lon)
    
    # Prepare final analysis
    analysis = {
        'safety_score': round(safety_score, 1),
        'zone': int(zone),
        'dominant_crime': dominant_crime_info['dominant_crime'],
        'common_crimes': dominant_crime_info['common_crimes'],
        'nearby_crime_count': len(nearby),
        'crime_types': nearby['crime_type'].value_counts().to_dict(),
        'lat': lat,
        'lon': lon,
        'density': density_info,
    }
    
    if time_analysis:
        analysis['time_analysis'] = time_analysis
    
    if dbscan_info:
        analysis['dbscan_cluster'] = {
            'cluster_id': int(dbscan_cluster),
            'dominant_crime': dbscan_info['dominant_crime'],
            'common_crimes': dbscan_info['common_crimes'],
            'crime_count': dbscan_info['crime_count']
        }
    
    if demographic_info:
        analysis['demographic_zone'] = {
            'zone_id': int(demographic_zone),
            'profiles': demographic_info['concentration_scores'],
            'crime_count': demographic_info['crime_count']
        }
    
    return analysis


def analyze_amenities(lat, lon, radius_km=1):
    """Analyze amenities near a location"""
    amenities_df = load_amenity_data()
    
    # Find nearby amenities
    nearby = find_nearby_points(amenities_df, lat, lon, radius_km)
    
    # Group by type
    type_counts = nearby['type'].value_counts().to_dict() if 'type' in nearby.columns else {}
    
    # Get closest amenities of each type
    closest_amenities = {}
    if not nearby.empty and 'type' in nearby.columns:
        for amenity_type in nearby['type'].unique():
            type_df = nearby[nearby['type'] == amenity_type].sort_values('distance')
            if not type_df.empty:
                closest = type_df.iloc[0]
                closest_amenities[amenity_type] = {
                    'name': closest.get('name', f"Unnamed {amenity_type}"),
                    'distance_km': round(closest['distance'], 2),
                    'latitude': closest['latitude'],
                    'longitude': closest['longitude']
                }
    
    return {
        'nearby_count': len(nearby),
        'type_counts': type_counts,
        'closest_amenities': closest_amenities
    }