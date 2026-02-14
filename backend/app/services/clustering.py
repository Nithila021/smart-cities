import numpy as np
import polars as pl
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neighbors import KernelDensity
from math import cos, radians
# import warnings

# To avoid circular imports
import sys
import os

from app.core.state import cached_data
from app.utils.geo import haversine
from app.services.cleaning import CITY_BOUNDS_CONFIG

# =============================================================================
# CONSTANTS
# =============================================================================
# Get default city bounds (NYC for now, can be configured via ACTIVE_CITY env var)
_active_city = os.getenv('ACTIVE_CITY', 'nyc')
_bounds = CITY_BOUNDS_CONFIG.get(_active_city, CITY_BOUNDS_CONFIG['nyc'])

DEFAULT_LAT_MIN = _bounds['lat_min']
DEFAULT_LAT_MAX = _bounds['lat_max']
DEFAULT_LON_MIN = _bounds['lon_min']
DEFAULT_LON_MAX = _bounds['lon_max']

# NYC approximate area in square kilometers
NYC_AREA_SQKM = 783.8

# Grid resolution for density calculations
DENSITY_GRID_SIZE = 50

def initialize_dbscan_clusters(df: pl.DataFrame):
    """
    Create DBSCAN clustering with batching for memory efficiency
    """
    print("Initializing DBSCAN clusters with batching...")
    
    # Larger sample but still manageable
    sample_size = min(20000, df.height)
    print(f"Sampling {sample_size} records for DBSCAN clustering...")
    df_sample = df.sample(sample_size, seed=42) if df.height > sample_size else df.clone()
    
    # Split into geographical batches for processing
    # Use configured city bounds
    lat_min, lat_max = DEFAULT_LAT_MIN, DEFAULT_LAT_MAX
    lon_min, lon_max = DEFAULT_LON_MIN, DEFAULT_LON_MAX

    # Create 4 geographical quadrants
    print("Splitting data into geographical quadrants...")
    lat_mid = (lat_min + lat_max) / 2
    lon_mid = (lon_min + lon_max) / 2
    
    # Polars filtering
    quadrants = {
        'NE': df_sample.filter((pl.col('latitude') >= lat_mid) & (pl.col('longitude') >= lon_mid)),
        'NW': df_sample.filter((pl.col('latitude') >= lat_mid) & (pl.col('longitude') < lon_mid)),
        'SE': df_sample.filter((pl.col('latitude') < lat_mid) & (pl.col('longitude') >= lon_mid)),
        'SW': df_sample.filter((pl.col('latitude') < lat_mid) & (pl.col('longitude') < lon_mid))
    }
    
    # Process each quadrant separately
    all_clusters = {}
    cluster_offset = 0
    
    # We will collect result DataFrames
    result_dfs = []
    
    for quadrant_name, quadrant_df in quadrants.items():
        print(f"Processing {quadrant_name} quadrant with {quadrant_df.height} points...")
        if quadrant_df.height < 50:  # Skip if too few points
            continue
            
        # Extract coordinates for clustering
        coords = quadrant_df.select(['latitude', 'longitude']).to_numpy()
        
        # Scale coordinates within this quadrant
        coord_scaler = StandardScaler()
        coords_scaled = coord_scaler.fit_transform(coords)
        
        # Use more granular parameters but still reasonable
        dbscan = DBSCAN(eps=0.01, min_samples=5, algorithm='ball_tree', n_jobs=-1)
        # Fit predict returns numpy array
        temp_clusters = dbscan.fit_predict(coords_scaled)
        
        # Add to DataFrame
        quadrant_df = quadrant_df.with_columns(pl.Series(name='temp_cluster', values=temp_clusters))
        
        # Renumber clusters to avoid overlap and store profiles
        # Filter valid ones
        valid_clusters = [c for c in np.unique(temp_clusters) if c != -1]
        
        # Initialize dbscan_cluster with -1
        quadrant_df = quadrant_df.with_columns([
            pl.when(pl.col('temp_cluster') != -1)
            .then(pl.col('temp_cluster') + cluster_offset)
            .otherwise(pl.lit(-1))
            .alias('dbscan_cluster')
        ])
        
        # Analyze clusters
        # Group by 'temp_cluster' for efficiency
        grouped = quadrant_df.filter(pl.col('temp_cluster') != -1).group_by('temp_cluster')
        
        # We can analyze aggregation
        # But we need specific crime counts per cluster
        # Let's iterate valid clusters for now to build `all_clusters` dict structure expected by API
        
        for cluster in valid_clusters:
            global_id = cluster + cluster_offset
            cluster_df = quadrant_df.filter(pl.col('temp_cluster') == cluster)
            
            # Crime counts
            crime_counts = cluster_df['crime_type'].value_counts().sort('count', descending=True)
            
            dominant = "Unknown"
            common = {}
            if crime_counts.height > 0:
                dominant = crime_counts[0, 'crime_type']
                top5 = crime_counts.head(5)
                common = {row['crime_type']: row['count'] for row in top5.iter_rows(named=True)}
            
            all_clusters[global_id] = {
                'dominant_crime': dominant,
                'common_crimes': common,
                'center_lat': cluster_df['latitude'].mean(),
                'center_lon': cluster_df['longitude'].mean(),
                'crime_count': cluster_df.height,
                'quadrant': quadrant_name
            }
        
        # Add to results
        result_dfs.append(quadrant_df.select(['latitude', 'longitude', 'dbscan_cluster', 'crime_type']))
        
        # Update offset for next quadrant
        if valid_clusters:
            cluster_offset = max(all_clusters.keys()) + 1
    
    # Concatenate results
    if result_dfs:
        result_df = pl.concat(result_dfs)
    else:
        # Empty schema
        result_df = pl.DataFrame(schema={
            'latitude': pl.Float64, 
            'longitude': pl.Float64, 
            'dbscan_cluster': pl.Int64,
            'crime_type': pl.Utf8
        })
    
    # Store model components for future predictions
    if result_df.height == 0:
        sample_points = pl.DataFrame(schema={'latitude': pl.Float64, 'longitude': pl.Float64, 'dbscan_cluster': pl.Int64})
    else:
        sample_points = result_df.select(['latitude', 'longitude', 'dbscan_cluster'])
    
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
    
    # Calculate distance for all points
    # Polars expression for Haversine approximation or just Euclidean if local
    # But let's use the python function appyl vectorised if possible, or mapping
    
    # If sample_points is huge, this is slow. 
    # But keeping it similar to original implementation for now.
    
    # Creating a literal Lat/Lon to calculate distance against
    # Use map_elements for custom haversine function
    
    # Optimization: pre-filter? No, global search.
    
    dist_series = sample_points.map_rows(
        lambda row: haversine(lat, lon, row[0], row[1])
    ).to_series()
    
    # Add dist column
    # sample_points is likely large, creating a full distance column every request is expensive!
    # Original code was: sample_points.apply(...)
    
    # Optimization: Use KDTree if available? Or just simplistic
    # For now, replicate logic
    
    min_idx = dist_series.arg_min()
    if min_idx is None:
        return None
        
    return sample_points[min_idx, 'dbscan_cluster']

def initialize_victim_demographic_zones(df: pl.DataFrame):
    """
    Create victim-demographic zones by clustering areas where victims share common characteristics
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
    
    # Sample data if needed
    sample_size = min(150000, df.height)
    df_sample = df.sample(sample_size, seed=42) if df.height > sample_size else df
    
    # Filter rows with demographic information (drop nulls)
    demo_df = df_sample.drop_nulls(subset=available_cols)
    if demo_df.height < 1000:
        print("Insufficient demographic data for clustering.")
        cached_data['victim_demographic_zones'] = None
        cached_data['demographic_feature_importance'] = None
        return
    
    # One-hot encode demographic features
    # Scikit-Learn OneHotEncoder expects 2D array
    demo_data_np = demo_df.select(available_cols).to_numpy()
    
    demo_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    demo_encoded = demo_encoder.fit_transform(demo_data_np)
    
    # Add coordinates for spatial clustering
    coords = demo_df.select(['latitude', 'longitude']).to_numpy()
    
    # Scale coordinates
    coord_scaler = StandardScaler()
    coords_scaled = coord_scaler.fit_transform(coords)
    
    # Combined features with higher weight on demographics (3x)
    combined_features = np.hstack([coords_scaled * 0.7, demo_encoded * 3])
    
    # Use k-means
    print(f"Clustering into 25 demographic zones...")
    kmeans = KMeans(n_clusters=25, random_state=42, n_init=10)
    zone_labels = kmeans.fit_predict(combined_features)
    
    demo_df = demo_df.with_columns(pl.Series(name='demographic_zone', values=zone_labels))
    
    # Analyze demographic zones
    zone_profiles = {}
    
    # Iterating unique zones
    unique_zones = np.unique(zone_labels)
    
    for zone in unique_zones:
        zone_data = demo_df.filter(pl.col('demographic_zone') == zone)
        
        profile = {}
        # Demographic breakdown
        for col in available_cols:
             val_counts = zone_data[col].value_counts()
             profile[col] = {row[col]: row['count'] for row in val_counts.iter_rows(named=True)}
             
        # Geographical center
        profile['center_lat'] = zone_data['latitude'].mean()
        profile['center_lon'] = zone_data['longitude'].mean()
        profile['crime_count'] = zone_data.height
        
        # Concentration scores
        concentration_scores = {}
        for col in available_cols:
            counts = zone_data[col].value_counts().sort('count', descending=True)
            if counts.height > 0:
                top_value = counts[0, col]
                total_sum = counts['count'].sum()
                concentration = (counts[0, 'count'] / total_sum) * 100
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
        # Since encoder expects a certain structure, we need to respect it. 
        # Using pandas for intermediate easy encoding if needed, OR just numpy array construction
        # BUT encoder was fitted on DataFrame values mostly. 
        # Safest is to construct a DataFrame (pd or pl) and converting to numpy in same order
        
        # Scikit's encoder.transform expects 2D array-like
        row_values = [[demographics[col] for col in available_cols]]
        demo_encoded = encoder.transform(row_values)
        
        # Combined features
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

def initialize_crime_density_zones(df: pl.DataFrame):
    """
    Classify city regions into Low, Medium, and High crime rate zones
    """
    print("Initializing crime density zones...")
    
    # Sample data if needed
    sample_size = min(100000, df.height)
    df_sample = df.sample(sample_size, seed=42) if df.height > sample_size else df
    
    # Extract coordinates for density estimation
    coords = df_sample.select(['latitude', 'longitude']).to_numpy()
    
    # Apply kernel density estimation
    kde = KernelDensity(bandwidth=0.01, metric='haversine')
    kde.fit(coords)
    
    # Create a grid for density visualization using configured bounds
    lat_min, lat_max = DEFAULT_LAT_MIN, DEFAULT_LAT_MAX
    lon_min, lon_max = DEFAULT_LON_MIN, DEFAULT_LON_MAX

    # Create grid (reduce resolution to manage memory)
    lat_grid = np.linspace(lat_min, lat_max, DENSITY_GRID_SIZE)
    lon_grid = np.linspace(lon_min, lon_max, DENSITY_GRID_SIZE)
    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)

    # Flatten grid for KDE scoring
    grid_points = np.vstack([lat_mesh.ravel(), lon_mesh.ravel()]).T

    # Score grid points
    log_density = kde.score_samples(grid_points)
    density = np.exp(log_density)

    # Convert to crimes per square km
    total_crimes = df_sample.height
    avg_density = total_crimes / NYC_AREA_SQKM

    # Adjust density to crimes per sq km
    crime_density = density * (total_crimes / density.sum()) * (DENSITY_GRID_SIZE**2 / NYC_AREA_SQKM)
    
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
    
    # Store data for API access (Numpy arrays, no Polars conversion needed for storage)
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
    total_crimes = df.height
    grid_size = len(density_data['grid']['lat_grid'])
    crime_density = density * (total_crimes / density) * (grid_size**2 / NYC_AREA_SQKM)
    
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
    
    # This loop is slightly generic, but functional
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
