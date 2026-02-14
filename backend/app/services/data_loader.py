import numpy as np
import polars as pl
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os
from typing import Any, Dict

# Import the data cleaner module
from app.services.cleaning import load_and_clean_csv, SEVERITY_WEIGHTS
from app.core.state import cached_data

# =============================================================================
# DATA LOADING AND TRAINING
# =============================================================================

def prepare_data() -> pl.DataFrame:
    """Load and preprocess raw NYPD crime data using the data_cleaner module.

    This function is responsible only for I/O and cleaning and returns a
    cleaned Polars DataFrame. Model training and caching are handled separately
    in train_models().
    """
    # Find available data file
    csv_files = [
        'NYPD_Complaint_Data_YTD.csv',
        'NYPD_Complaint_Data_Historic.csv',
        '../NYPD_Complaint_Data_YTD.csv',
        '../NYPD_Complaint_Data_Historic.csv',
    ]

    csv_path = None
    for f in csv_files:
        if os.path.exists(f):
            csv_path = f
            break

    if csv_path is None:
        raise RuntimeError(
            "Data file not found. Ensure NYPD crime data CSV exists in project root. "
            f"Tried: {csv_files}"
        )

    # Use the centralized data cleaner
    df, report = load_and_clean_csv(csv_path, verbose=True)

    # Store cleaning report for debugging/auditing
    cached_data['cleaning_report'] = report

    return df


def train_models(df: pl.DataFrame) -> None:
    """Train clustering / density models and populate the global cache.

    Receives a cleaned Polars DataFrame and is responsible for fitting models and
    updating cached_data. This keeps training concerns separated from
    data loading/cleaning.
    """
    # Create crime clusters - simplified to use only coordinates for easier prediction
    # Convert to numpy for sklearn
    coords = df.select(['latitude', 'longitude']).to_numpy()
    
    crime_scaler = StandardScaler()
    coords_scaled = crime_scaler.fit_transform(coords)

    crime_kmeans = KMeans(n_clusters=30, random_state=42, n_init=10)
    crime_labels = crime_kmeans.fit_predict(coords_scaled)
    
    # Add labels back to DataFrame
    df = df.with_columns(pl.Series(name="crime_zone", values=crime_labels))

    # Create safety scores using SEVERITY_WEIGHTS from data_cleaner (0-1 scale)
    # Convert to 1-10 scale for backward compatibility with existing formula
    crime_severity = {k: int(v * 10) for k, v in SEVERITY_WEIGHTS.items()}

    zone_safety = {}
    zone_dominant_crimes = {}
    
    # Aggregations in Polars
    # We want per-zone:
    # 1. Total crimes
    # 2. Weighted severity sum
    # 3. Crime type counts (for dominant/common)
    
    # It's often easier to iterate through unique zones if N is small (30 is small)
    # But let's use Polars aggregation power where possible or hybrid
    
    # Group by zone and aggregated stats
    unique_zones = df['crime_zone'].unique().to_list()
    
    for zone in unique_zones:
        zone_df = df.filter(pl.col('crime_zone') == zone)
        total_crimes = zone_df.height
        
        # Calculate severity score
        # Join with weights or apply map
        # Since we have 'severity_weight' column in the cleaned DF (0.0-1.0), we can just sum it * 10
        # Check if severity_weight exists
        if 'severity_weight' in zone_df.columns:
             severity_score_sum = zone_df['severity_weight'].sum() * 10
        else:
             # Fallback if column missing (shouldn't happen with updated cleaner)
             severity_score_sum = total_crimes * 5 # average
             
        safety_score = 100 - ((severity_score_sum / (total_crimes * 10)) * 100)
        zone_safety[zone] = max(0, min(100, safety_score))

        # Get dominant crimes
        crime_counts = zone_df['crime_type'].value_counts().sort('count', descending=True)
        
        dominant = "Unknown"
        common = {}
        
        if crime_counts.height > 0:
            dominant = crime_counts[0, 'crime_type']
            # Top 3
            top3 = crime_counts.head(3)
            common = {row['crime_type']: row['count'] for row in top3.iter_rows(named=True)}
            
        zone_dominant_crimes[zone] = {
            'dominant_crime': dominant,
            'common_crimes': common
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

    # Import specialized model initializers
    from app.services.clustering import (
        initialize_dbscan_clusters,
        initialize_victim_demographic_zones,
        initialize_crime_density_zones,
    )

    # Initialize new clustering and zoning methods (added functionality)
    initialize_dbscan_clusters(df)
    initialize_victim_demographic_zones(df)
    initialize_crime_density_zones(df)


def initialize_data() -> pl.DataFrame:
    """Top-level initializer that loads data and trains models."""
    print("Initializing data with enhanced cleaning (Polars)...")
    df = prepare_data()
    train_models(df)
    print(f"Data initialization complete. {df.height} records processed.")
    return df

def load_amenity_data() -> pl.DataFrame:
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
        
        amenities_df = pl.DataFrame(amenities)
        
    else:
        # Load actual amenity data
        amenities_df = pl.read_csv('NYC_Amenities.csv')
    
    # Store in cache
    cached_data['amenities_df'] = amenities_df
    return amenities_df
