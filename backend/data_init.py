import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import re
from datetime import datetime
import os

# Global cached data
cached_data = {
    'df': None,
    'crime_clusters': None,
    'zone_safety_scores': None,
    'crime_severity': None,
    'zone_dominant_crimes': None,
    'amenities_df': None,
    'dbscan_clusters': None,
    'victim_demographic_zones': None,
    'demographic_feature_importance': None,
    'crime_density_zones': None
}


def prepare_data():
    """Load and preprocess raw NYPD crime data with enhanced cleaning.

    This function is responsible only for I/O and cleaning and returns a
    cleaned DataFrame. Model training and caching are handled separately
    in train_models().
    """
    # Load data with optimized dtypes
    try:
        dtypes = {
            'cmplnt_num': 'string',
            'rpt_dt': 'string',
            'pd_desc': 'category',
            'ofns_desc': 'category',
            'boro_nm': 'category',
            'prem_typ_desc': 'category'
        }
        df = pd.read_csv(
            'NYPD_Complaint_Data_YTD.csv',
            dtype=dtypes,
            parse_dates=['cmplnt_fr_dt', 'cmplnt_to_dt']
        )
    except FileNotFoundError:
        try:
            # Try alternate filename
            df = pd.read_csv(
                'NYPD_Complaint_Data_Historic.csv',
                dtype=dtypes,
                parse_dates=['cmplnt_fr_dt', 'cmplnt_to_dt']
            )
        except FileNotFoundError:
            raise RuntimeError("Data file not found. Ensure NYPD crime data CSV exists")

    # Clean coordinates
    def validate_nyc_coords(lat, lon):
        return (40.4 <= lat <= 41.0) and (-74.3 <= lon <= -73.7)

    coord_cols = ['lat_lon.latitude', 'lat_lon.longitude', 'latitude', 'longitude']
    for col in coord_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Use either lat_lon.latitude/longitude or latitude/longitude columns
    if 'lat_lon.latitude' in df.columns and 'lat_lon.longitude' in df.columns:
        df['clean_lat'] = df['lat_lon.latitude']
        df['clean_lon'] = df['lat_lon.longitude']
    else:
        df['clean_lat'] = df['latitude']
        df['clean_lon'] = df['longitude']

    valid_coords = df.apply(lambda x: validate_nyc_coords(x.clean_lat, x.clean_lon), axis=1)
    df = df[valid_coords].copy()

    # Clean victim demographics
    victim_cols = ['vic_age_group', 'vic_race', 'vic_sex']
    demographic_clean = {
        'vic_age_group': lambda x: re.sub(r'\D+', '-', str(x)).upper() if pd.notna(x) else np.nan,
        'vic_race': lambda x: 'UNKNOWN' if 'UNKNOWN' in str(x).upper() else str(x).upper(),
        'vic_sex': lambda x: x[0].upper() if pd.notna(x) else np.nan
    }

    for col, fn in demographic_clean.items():
        if col in df.columns:
            df[col] = df[col].apply(fn).str.strip().replace('', np.nan)

    # Clean crime types
    crime_field = 'ofns_desc' if 'ofns_desc' in df.columns else 'offense_description'
    crime_mappings = {
        r'ASSAULT.*3.*': 'ASSAULT_3',
        r'HARRASSMENT': 'HARASSMENT',
        r'DRIVING WHILE INTOXICATED': 'DWI',
        r'CRIMINAL MISCHIEF.*': 'CRIMINAL_MISCHIEF'
    }

    df['crime_type'] = df[crime_field].str.upper().str.strip()
    for pattern, replacement in crime_mappings.items():
        df['crime_type'] = df['crime_type'].str.replace(pattern, replacement, regex=True)

    # Temporal features
    date_col = 'cmplnt_fr_dt'
    time_col = 'cmplnt_fr_tm'

    def parse_dt(date_str, time_str):
        try:
            return datetime.strptime(f"{str(date_str)[:10]} {str(time_str)}", "%Y-%m-%d %H:%M:%S")
        except Exception:
            return pd.NaT

    if date_col in df.columns and time_col in df.columns:
        df['cmplnt_fr_datetime'] = df.apply(
            lambda x: parse_dt(x[date_col], x[time_col]), axis=1
        )

    # Final cleaning
    keep_cols = [
        'clean_lat', 'clean_lon', 'crime_type',
        'cmplnt_fr_datetime' if 'cmplnt_fr_datetime' in df.columns else date_col
    ]

    # Add demographic columns if available
    for col in victim_cols + ['boro_nm']:
        if col in df.columns:
            keep_cols.append(col)

    df = df[keep_cols].rename(columns={
        'clean_lat': 'latitude',
        'clean_lon': 'longitude'
    }).dropna(subset=['latitude', 'longitude', 'crime_type'])

    return df


def train_models(df):
    """Train clustering / density models and populate the global cache.

    Receives a cleaned DataFrame and is responsible for fitting models and
    updating cached_data. This keeps training concerns separated from
    data loading/cleaning.
    """
    # Create crime clusters - simplified to use only coordinates for easier prediction
    coords = df[['latitude', 'longitude']].values
    crime_scaler = StandardScaler()
    coords_scaled = crime_scaler.fit_transform(coords)

    crime_kmeans = KMeans(n_clusters=30, random_state=42, n_init=10)
    df['crime_zone'] = crime_kmeans.fit_predict(coords_scaled)

    # Create safety scores
    crime_severity = {
        'ASSAULT_3': 7, 'HARASSMENT': 4, 'DWI': 6, 'CRIMINAL_MISCHIEF': 3,
        'ROBBERY': 8, 'GRAND LARCENY': 5, 'BURGLARY': 6, 'RAPE': 9,
        'MURDER & NON-NEGL. MANSLAUGHTER': 10, 'FELONY ASSAULT': 8
    }

    zone_safety = {}
    zone_dominant_crimes = {}

    for zone in df['crime_zone'].unique():
        zone_df = df[df['crime_zone'] == zone]
        total_crimes = len(zone_df)
        severity_score = sum(
            (crime_severity.get(crime, 3) * count)
            for crime, count in zone_df['crime_type'].value_counts().items()
        )
        safety_score = 100 - ((severity_score / (total_crimes * 10)) * 100)
        zone_safety[zone] = max(0, min(100, safety_score))

        # Get dominant crimes for each zone
        crime_counts = zone_df['crime_type'].value_counts()
        zone_dominant_crimes[zone] = {
            'dominant_crime': crime_counts.idxmax() if not crime_counts.empty else "Unknown",
            'common_crimes': crime_counts.nlargest(3).to_dict()
        }

    # Update cached data
    cached_data.update({
        'df': df,
        'crime_clusters': crime_kmeans,
        'crime_scaler': crime_scaler,
        'zone_safety_scores': zone_safety,
        'crime_severity': crime_severity,
        'zone_dominant_crimes': zone_dominant_crimes
    })

    # Import moved from models.py to avoid circular imports
    from models import (
        initialize_dbscan_clusters,
        initialize_victim_demographic_zones,
        initialize_crime_density_zones,
    )

    # Initialize new clustering and zoning methods (added functionality)
    initialize_dbscan_clusters(df)
    initialize_victim_demographic_zones(df)
    initialize_crime_density_zones(df)


def initialize_data():
    """Top-level initializer that loads data and trains models.

    This preserves the existing public API while delegating to
    prepare_data() and train_models() for better structure.
    """
    print("Initializing data with enhanced cleaning...")
    df = prepare_data()
    train_models(df)
    print(f"Data initialization complete. {len(df)} records processed.")
    return df

def load_amenity_data():
    """Load amenity data if available"""
    if cached_data.get('amenities_df') is not None:
        return cached_data['amenities_df']
    
    # Check if amenity data file exists
    if not os.path.exists('NYC_Amenities.csv'):
        # Create dummy data if file doesn't exist
        print("Amenity data file not found. Creating dummy data.")
        amenities = []
        
        # Create some sample amenities across NYC
        amenity_types = ['park', 'school', 'restaurant', 'hospital', 'police', 'subway']
        
        # NYC borough center points
        borough_centers = {
            'Manhattan': (40.7831, -73.9712),
            'Brooklyn': (40.6782, -73.9442),
            'Queens': (40.7282, -73.7949),
            'Bronx': (40.8448, -73.8648),
            'Staten Island': (40.5795, -74.1502)
        }
        
        # Generate sample amenities around borough centers
        for borough, (lat, lon) in borough_centers.items():
            for i in range(20):  # 20 amenities per borough
                amenity_type = amenity_types[i % len(amenity_types)]
                lat_offset = np.random.uniform(-0.05, 0.05)
                lon_offset = np.random.uniform(-0.05, 0.05)
                
                amenities.append({
                    'name': f"{borough} {amenity_type.capitalize()} {i+1}",
                    'type': amenity_type,
                    'borough': borough,
                    'latitude': lat + lat_offset,
                    'longitude': lon + lon_offset
                })
        
        amenities_df = pd.DataFrame(amenities)
        
    else:
        # Load actual amenity data
        amenities_df = pd.read_csv('NYC_Amenities.csv')
    
    # Store in cache
    cached_data['amenities_df'] = amenities_df
    return amenities_df