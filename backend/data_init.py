import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os

# Import the data cleaner module
from data_cleaner import DataCleaner, load_and_clean_csv, SEVERITY_WEIGHTS

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
    'crime_density_zones': None,
    'cleaning_report': None  # Store the cleaning report
}


def prepare_data():
    """Load and preprocess raw NYPD crime data using the data_cleaner module.

    This function is responsible only for I/O and cleaning and returns a
    cleaned DataFrame. Model training and caching are handled separately
    in train_models().

    Uses data_cleaner.py for all cleaning operations to ensure consistency
    between CSV loading and database migration.
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

    # Create safety scores using SEVERITY_WEIGHTS from data_cleaner (0-1 scale)
    # Convert to 1-10 scale for backward compatibility with existing formula
    crime_severity = {k: int(v * 10) for k, v in SEVERITY_WEIGHTS.items()}

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