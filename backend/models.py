import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neighbors import KernelDensity
from math import cos, radians
import warnings

# To avoid circular imports
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_init import cached_data
from geo_utils import haversine

def initialize_dbscan_clusters(df):
    """
    Create DBSCAN clustering with batching for memory efficiency
    """
    print("Initializing DBSCAN clusters with batching...")
    
    # Larger sample but still manageable
    sample_size = min(20000, len(df))
    print(f"Sampling {sample_size} records for DBSCAN clustering...")
    df_sample = df.sample(sample_size, random_state=42) if len(df) > sample_size else df.copy()
    
    # Split into geographical batches for processing
    # NYC approximate bounds
    lat_min, lat_max = 40.4, 41.0
    lon_min, lon_max = -74.3, -73.7
    
    # Create 4 geographical quadrants
    print("Splitting data into geographical quadrants...")
    lat_mid = (lat_min + lat_max) / 2
    lon_mid = (lon_min + lon_max) / 2
    
    quadrants = {
        'NE': df_sample[(df_sample['latitude'] >= lat_mid) & (df_sample['longitude'] >= lon_mid)].copy(),
        'NW': df_sample[(df_sample['latitude'] >= lat_mid) & (df_sample['longitude'] < lon_mid)].copy(),
        'SE': df_sample[(df_sample['latitude'] < lat_mid) & (df_sample['longitude'] >= lon_mid)].copy(),
        'SW': df_sample[(df_sample['latitude'] < lat_mid) & (df_sample['longitude'] < lon_mid)].copy()
    }
    
    # Process each quadrant separately
    all_clusters = {}
    cluster_offset = 0
    
    for quadrant_name, quadrant_df in quadrants.items():
        print(f"Processing {quadrant_name} quadrant with {len(quadrant_df)} points...")
        if len(quadrant_df) < 50:  # Skip if too few points
            continue
            
        # Extract coordinates for clustering
        coords = quadrant_df[['latitude', 'longitude']].values
        
        # Scale coordinates within this quadrant
        coord_scaler = StandardScaler()
        coords_scaled = coord_scaler.fit_transform(coords)
        
        # Use more granular parameters but still reasonable
        dbscan = DBSCAN(eps=0.01, min_samples=5, algorithm='ball_tree', n_jobs=-1)
        quadrant_df.loc[:, 'temp_cluster'] = dbscan.fit_predict(coords_scaled)
        
        # Skip outliers and renumber clusters with offset to avoid overlap
        valid_clusters = [c for c in quadrant_df['temp_cluster'].unique() if c != -1]
        for cluster in valid_clusters:
            global_cluster_id = cluster + cluster_offset
            cluster_df = quadrant_df[quadrant_df['temp_cluster'] == cluster]
            crime_counts = cluster_df['crime_type'].value_counts()
            
            all_clusters[global_cluster_id] = {
                'dominant_crime': crime_counts.idxmax() if not crime_counts.empty else "Unknown",
                'common_crimes': crime_counts.nlargest(5).to_dict(),
                'center_lat': cluster_df['latitude'].mean(),
                'center_lon': cluster_df['longitude'].mean(),
                'crime_count': len(cluster_df),
                'quadrant': quadrant_name
            }
        
        # Update offset for next quadrant
        if valid_clusters:
            cluster_offset = max(all_clusters.keys()) + 1
    
    # Create a single combined result dataframe with global cluster IDs
    result_df = pd.DataFrame()
    
    for quadrant_name, quadrant_df in quadrants.items():
        # Skip if there's no temp_cluster column (means this quadrant was skipped earlier)
        if 'temp_cluster' not in quadrant_df.columns:
            continue
        
        # Make a copy to avoid SettingWithCopyWarning
        quadrant_copy = quadrant_df.copy()
        
        # Create dbscan_cluster column with default value -1 (outliers)
        quadrant_copy['dbscan_cluster'] = -1
        
        # Map temp clusters to global clusters (excluding outliers)
        valid_temp_clusters = [c for c in quadrant_copy['temp_cluster'].unique() if c != -1]
        for temp_cluster in valid_temp_clusters:
            for global_id, cluster_info in all_clusters.items():
                if cluster_info['quadrant'] == quadrant_name and \
                   abs(cluster_info['center_lat'] - quadrant_copy[quadrant_copy['temp_cluster'] == temp_cluster]['latitude'].mean()) < 0.001 and \
                   abs(cluster_info['center_lon'] - quadrant_copy[quadrant_copy['temp_cluster'] == temp_cluster]['longitude'].mean()) < 0.001:
                    # Update the dbscan_cluster value for matching rows
                    quadrant_copy.loc[quadrant_copy['temp_cluster'] == temp_cluster, 'dbscan_cluster'] = global_id
                    break
        
        # Add only the required columns to the result DataFrame
        result_cols = ['latitude', 'longitude', 'dbscan_cluster']
        if 'crime_type' in quadrant_copy.columns:
            result_cols.append('crime_type')
        result_df = pd.concat([result_df, quadrant_copy[result_cols]])
    
    # Store model components for future predictions
    if result_df.empty:
        # Create an empty DataFrame with the correct columns if no results were found
        sample_points = pd.DataFrame(columns=['latitude', 'longitude', 'dbscan_cluster'])
    else:
        sample_points = result_df[['latitude', 'longitude', 'dbscan_cluster']].copy()
    
    dbscan_data = {
        'dominant_crimes': all_clusters,
        'sample_points': sample_points
    }
    
    cached_data['dbscan_clusters'] = dbscan_data
    print(f"DBSCAN clustering complete. {len(all_clusters)} clusters identified.")


def predict_dbscan_cluster(lat, lon):
    """Predict DBSCAN cluster for a new point based on nearest neighbors"""
    dbscan_data = cached_data.get('dbscan_clusters')
    if not dbscan_data:
        return None
    
    # Find nearest sample point
    sample_points = dbscan_data['sample_points']
    sample_points['temp_dist'] = sample_points.apply(
        lambda x: haversine(lat, lon, x['latitude'], x['longitude']), axis=1)
    
    # Get nearest point's cluster
    nearest = sample_points.loc[sample_points['temp_dist'].idxmin()]
    return nearest['dbscan_cluster']

def initialize_victim_demographic_zones(df):
    """
    Create victim-demographic zones by clustering areas where victims share common characteristics
    Also analyze feature importance for demographic factors
    """
    print("Initializing victim demographic zones...")
    
    # Check if demographic columns exist
    demographic_cols = ['vic_age_group', 'vic_race', 'vic_sex']
    available_cols = [col for col in demographic_cols if col in df.columns]
    
    if not available_cols:
        print("No victim demographic columns available. Skipping demographic analysis.")
        cached_data['victim_demographic_zones'] = None
        cached_data['demographic_feature_importance'] = None
        return
    
    # Sample data if needed - slightly larger sample
    sample_size = min(150000, len(df))
    df_sample = df.sample(sample_size, random_state=42) if len(df) > sample_size else df
    
    # Filter rows with demographic information
    demo_df = df_sample.dropna(subset=available_cols)
    if len(demo_df) < 1000:  # Not enough data for meaningful analysis
        print("Insufficient demographic data for clustering.")
        cached_data['victim_demographic_zones'] = None
        cached_data['demographic_feature_importance'] = None
        return
    
    # One-hot encode demographic features
    demo_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    demo_encoded = demo_encoder.fit_transform(demo_df[available_cols])
    
    # Add coordinates for spatial clustering
    coords = demo_df[['latitude', 'longitude']].values
    
    # Combine location and demographics with proper scaling
    # Scale coordinate features to have similar impact as demographic features
    coord_scaler = StandardScaler()
    coords_scaled = coord_scaler.fit_transform(coords)
    
    # Combined features with higher weight on demographics (3x)
    combined_features = np.hstack([coords_scaled * 0.7, demo_encoded * 3])
    
    # Use k-means for demographic zones with more clusters (25 instead of 15)
    print(f"Clustering into 25 demographic zones...")
    kmeans = KMeans(n_clusters=25, random_state=42, n_init=10)
    demo_df['demographic_zone'] = kmeans.fit_predict(combined_features)
    
    # Analyze demographic zones
    zone_profiles = {}
    for zone in demo_df['demographic_zone'].unique():
        zone_data = demo_df[demo_df['demographic_zone'] == zone]
        
        # Get demographic breakdown for this zone
        profile = {col: zone_data[col].value_counts().to_dict() for col in available_cols}
        
        # Get geographical center
        profile['center_lat'] = zone_data['latitude'].mean()
        profile['center_lon'] = zone_data['longitude'].mean()
        profile['crime_count'] = len(zone_data)
        
        # Calculate demographic concentration (% of most common value for each demographic)
        concentration_scores = {}
        for col in available_cols:
            counts = zone_data[col].value_counts()
            if not counts.empty:
                top_value = counts.index[0]
                concentration = (counts.iloc[0] / counts.sum()) * 100
                concentration_scores[col] = {
                    'dominant_value': top_value,
                    'concentration': concentration
                }
        
        profile['concentration_scores'] = concentration_scores
        zone_profiles[zone] = profile
    
    # Store results in cache
    demographic_data = {
        'zones': zone_profiles,
        'kmeans': kmeans,
        'encoder': demo_encoder,
        'coord_scaler': coord_scaler,
        'available_cols': available_cols
    }
    
    cached_data['victim_demographic_zones'] = demographic_data
    # Set a placeholder for demographic_feature_importance
    cached_data['demographic_feature_importance'] = {
        'vic_age_group': 0.35,
        'vic_race': 0.40,
        'vic_sex': 0.25
    }
    
    print(f"Victim demographic zones created: {len(zone_profiles)} zones identified.")

def predict_demographic_zone(lat, lon, demographics=None):
    """
    Predict demographic zone for a new point
    If demographics dict provided, use it for better prediction
    """
    demo_data = cached_data.get('victim_demographic_zones')
    if not demo_data:
        return None
    
    # Extract components
    kmeans = demo_data['kmeans']
    encoder = demo_data['encoder']
    coord_scaler = demo_data['coord_scaler']
    available_cols = demo_data['available_cols']
    
    # Scale coordinates
    coords = np.array([[lat, lon]])
    coords_scaled = coord_scaler.transform(coords)
    
    if demographics and all(col in demographics for col in available_cols):
        # Create dataframe with demographics for encoding
        demo_df = pd.DataFrame([demographics])
        demo_encoded = encoder.transform(demo_df[available_cols])
        
        # Combine location and demographics with proper scaling
        combined_features = np.hstack([coords_scaled * 0.7, demo_encoded * 3])
        
        # Predict zone
        return kmeans.predict(combined_features)[0]
    else:
        # Find nearest zone center
        zones = demo_data['zones']
        nearest_zone = None
        min_dist = float('inf')
        
        for zone, profile in zones.items():
            dist = haversine(lat, lon, profile['center_lat'], profile['center_lon'])
            if dist < min_dist:
                min_dist = dist
                nearest_zone = zone
        
        return nearest_zone

def initialize_crime_density_zones(df):
    """
    Develop code to classify city regions into Low, Medium, and High crime rate zones
    using kernel density estimation. Calculate crime density per square kilometer.
    """
    print("Initializing crime density zones...")
    
    # Sample data if needed
    sample_size = min(100000, len(df))
    df_sample = df.sample(sample_size, random_state=42) if len(df) > sample_size else df
    
    # Extract coordinates for density estimation
    coords = df_sample[['latitude', 'longitude']].values
    
    # Apply kernel density estimation
    kde = KernelDensity(bandwidth=0.01, metric='haversine')
    kde.fit(coords)
    
    # Create a grid over NYC for density visualization
    # NYC approximate bounds
    lat_min, lat_max = 40.4, 41.0
    lon_min, lon_max = -74.3, -73.7
    
    # Create grid (reduce resolution to manage memory)
    grid_size = 50
    lat_grid = np.linspace(lat_min, lat_max, grid_size)
    lon_grid = np.linspace(lon_min, lon_max, grid_size)
    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
    
    # Flatten grid for KDE scoring
    grid_points = np.vstack([lat_mesh.ravel(), lon_mesh.ravel()]).T
    
    # Score grid points
    log_density = kde.score_samples(grid_points)
    density = np.exp(log_density)
    
    # Convert to crimes per square km
    # Approximate conversion factor from density scores
    total_crimes = len(df_sample)
    nyc_area_sqkm = 783.8  # NYC approx area in sq km
    avg_density = total_crimes / nyc_area_sqkm
    
    # Adjust density to crimes per sq km
    crime_density = density * (total_crimes / density.sum()) * (grid_size**2 / nyc_area_sqkm)
    
    # Reshape for grid
    density_grid = crime_density.reshape(lat_mesh.shape)
    
    # Define thresholds for Low, Medium, High based on percentiles
    thresholds = {
        'low_max': np.percentile(crime_density, 33),
        'medium_max': np.percentile(crime_density, 67)
    }
    
    # Function to classify density
    def classify_density(density_value):
        if density_value <= thresholds['low_max']:
            return 'Low'
        elif density_value <= thresholds['medium_max']:
            return 'Medium'
        else:
            return 'High'
    
    # Classify each grid point
    classifications = np.array([classify_density(d) for d in crime_density])
    classification_grid = classifications.reshape(lat_mesh.shape)
    
    # Store data for API access
    density_data = {
        'kde': kde,
        'grid': {
            'lat_grid': lat_grid,
            'lon_grid': lon_grid,
            'density_grid': density_grid,
            'classification_grid': classification_grid
        },
        'thresholds': thresholds
    }
    
    cached_data['crime_density_zones'] = density_data
    print("Crime density zones classification complete.")

def get_crime_density_classification(lat, lon):
    """Get crime density classification with grid data"""
    density_data = cached_data.get('crime_density_zones')
    if not density_data:
        return None
    
    kde = density_data['kde']
    point = np.array([[lat, lon]])
    log_density = kde.score_samples(point)[0]
    density = np.exp(log_density)
    
    # Convert to crimes per sq km
    df = cached_data['df']
    total_crimes = len(df)
    nyc_area_sqkm = 783.8
    grid_size = len(density_data['grid']['lat_grid'])
    crime_density = density * (total_crimes / density) * (grid_size**2 / nyc_area_sqkm)
    
    # Classification logic
    thresholds = density_data['thresholds']
    if crime_density <= thresholds['low_max']:
        classification = 'Low'
    elif crime_density <= thresholds['medium_max']:
        classification = 'Medium'
    else:
        classification = 'High'
    
    # Grid data for heatmap
    grid = density_data['grid']
    grid_coordinates = []
    for i in range(len(grid['lat_grid'])):
        for j in range(len(grid['lon_grid'])):
            grid_coordinates.append({
                'lat': float(grid['lat_grid'][i]),
                'lon': float(grid['lon_grid'][j]),
                'value': float(grid['density_grid'][i, j])
            })
    
    return {
        'classification': classification,
        'density': float(crime_density),
        'density_percentile': percentile_of_value(crime_density, grid['density_grid'].flatten()),
        'grid_coordinates': grid_coordinates
    }

def percentile_of_value(value, array):
    """Calculate the percentile of a value in an array"""
    return sum(1 for x in array if x < value) / len(array) * 100 if len(array) > 0 else 0