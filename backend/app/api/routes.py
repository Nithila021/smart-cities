from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import Optional, Dict, List, Any
from app.core.state import cached_data
from app.utils.geo import get_coordinates
from app.services.data_loader import initialize_data
from app.services.analysis import analyze_safety, analyze_amenities
from app.services.clustering import get_crime_density_classification
from app.utils.report import generate_safety_report
from app.services.demographics import parse_demographic_group, generate_custom_safety_recommendations
import polars as pl

router = APIRouter(prefix="/api")

# --------------------------
# Pydantic Models
# --------------------------
class AnalyzeRequest(BaseModel):
    location: str
    include_advanced: bool = False
    include_amenities: bool = False

class ChatRequest(BaseModel):
    message: str
    demographics: Optional[str] = None

# --------------------------
# Helper Functions
# --------------------------
def ensure_data_initialized():
    """Ensure crime data and models are loaded into the global cache."""
    if cached_data.get('df') is None:
        initialize_data()

def get_crime_heatmap():
    ensure_data_initialized()
    df = cached_data.get('df')
    if df is None or df.height == 0:
        return []
    
    sample_size = min(5000, df.height)
    heatmap_data = df.sample(sample_size, seed=42).select(['latitude', 'longitude', 'crime_type'])

    if 'severity_weight' in cached_data.get('df', pl.DataFrame()).columns: # Check original df columns safe way
        # Re-sample to include weight if it exists
         heatmap_data = df.sample(sample_size, seed=42).select(['latitude', 'longitude', 'crime_type', 'severity_weight'])
         heatmap_data = heatmap_data.rename({'severity_weight': 'weight'})
    else:
        crime_severity = cached_data.get('crime_severity', {})
        def get_weight(ctype):
            return crime_severity.get(ctype, 3) / 10.0
            
        heatmap_data = heatmap_data.with_columns(
            pl.col('crime_type').map_elements(get_weight, return_dtype=pl.Float64).alias('weight')
        )
        
    return heatmap_data.select(['latitude', 'longitude', 'weight']).to_dicts()

def get_demographic_analysis_data():
    ensure_data_initialized()
    feature_importance = cached_data.get('demographic_feature_importance')
    demographic_zones = cached_data.get('victim_demographic_zones')
    if not feature_importance or not demographic_zones:
        return {'error': 'Demographic analysis data not available'}

    zone_data = []
    for zone_id, profile in demographic_zones['zones'].items():
        zone_data.append({
            'zone_id': int(zone_id),
            'center_lat': profile['center_lat'],
            'center_lon': profile['center_lon'],
            'crime_count': profile['crime_count'],
            'concentration_scores': profile['concentration_scores']
        })

    return {'feature_importance': feature_importance, 'zones': zone_data}

def get_crime_density_map_data():
    ensure_data_initialized()
    density_data = cached_data.get('crime_density_zones')
    if not density_data:
        return {'error': 'Crime density data not available'}

    grid = density_data['grid']
    map_data = []
    for i in range(len(grid['lat_grid'])):
        for j in range(len(grid['lon_grid'])):
            map_data.append({
                'latitude': float(grid['lat_grid'][i]),
                'longitude': float(grid['lon_grid'][j]),
                'density': float(grid['density_grid'][i, j]),
                'classification': str(grid['classification_grid'][i, j])
            })

    return {
        'points': map_data,
        'thresholds': {
            'low_max': float(density_data['thresholds']['low_max']),
            'medium_max': float(density_data['thresholds']['medium_max'])
        }
    }

def get_dbscan_clusters_data():
    ensure_data_initialized()
    dbscan_data = cached_data.get('dbscan_clusters')
    if not dbscan_data:
        return []
    return [
        {
            "id": cluster_id,
            "lat": cluster['center_lat'],
            "lon": cluster['center_lon'],
            "crime_count": cluster['crime_count']
        }
        for cluster_id, cluster in dbscan_data['dominant_crimes'].items()
    ]

def get_demographic_zones_data():
    ensure_data_initialized()
    demo_data = cached_data.get('victim_demographic_zones')
    if not demo_data:
        return []
    return [
        {
            "id": zone_id,
            "lat": zone['center_lat'],
            "lon": zone['center_lon'],
            "dominant_demo": next(iter(zone['concentration_scores'].values()))['dominant_value']
        }
        for zone_id, zone in demo_data['zones'].items()
    ]


# --------------------------
# API Endpoints
# --------------------------

@router.post('/analyze_v1')
def analyze_endpoint_v1(request: AnalyzeRequest):
    location = request.location
    include_advanced = request.include_advanced
    
    if coords := get_coordinates(location):
        lat, lon = coords
        analysis = analyze_safety(lat, lon)
        
        if request.include_amenities:
            analysis['amenities'] = analyze_amenities(lat, lon)
            
        if include_advanced:
            demographic_data = get_demographic_analysis_data()
            if 'error' not in demographic_data:
                analysis['demographic_analysis'] = demographic_data
                
        return analysis
    
    raise HTTPException(status_code=400, detail="Invalid location")

@router.get('/heatmap')
def heatmap_endpoint():
    return get_crime_heatmap()

@router.get('/density_map')
def density_map_endpoint():
    return get_crime_density_map_data()

@router.get('/demographic_zones')
def demographic_zones_endpoint():
    return get_demographic_analysis_data()

@router.get('/dbscan_clusters')
def dbscan_clusters_endpoint():
    dbscan_data = cached_data.get('dbscan_clusters')
    if not dbscan_data:
         raise HTTPException(status_code=404, detail="DBSCAN clustering data not available")
    
    clusters = []
    for cluster_id, info in dbscan_data['dominant_crimes'].items():
        clusters.append({
            'cluster_id': int(cluster_id),
            'center_lat': float(info['center_lat']),
            'center_lon': float(info['center_lon']),
            'crime_count': int(info['crime_count']),
            'dominant_crime': info['dominant_crime'],
            'common_crimes': {str(k): int(v) for k, v in info['common_crimes'].items()}
        })
    
    return {'clusters': clusters}

@router.post('/analyze_v2')
def analyze_endpoint_v2(request: AnalyzeRequest):
    location = request.location
    
    if coords := get_coordinates(location):
        lat, lon = coords
        analysis = analyze_safety(lat, lon)
        amenities = analyze_amenities(lat, lon)
        
        response_data = {
            **analysis,
            "lat": lat,
            "lon": lon,
            "crime_types": analysis.get('crime_types', {}),
            "amenities": amenities,
            "density": get_crime_density_classification(lat, lon)
        }
        
        return response_data
    
    raise HTTPException(status_code=400, detail="Invalid location")

@router.get('/map_data')
def map_data_endpoint():
    return {
        "dbscan_clusters": get_dbscan_clusters_data(),
        "demographic_zones": get_demographic_zones_data(),
        "density_zones": get_crime_density_map_data()
    }

@router.post('/chat')
def chat_endpoint(request: ChatRequest):
    try:
        location = request.message
        demographics = request.demographics
        
        if not location:
            raise HTTPException(status_code=400, detail="Empty request")

        amenity_type = None
        if ',' in location:
            amenity_type = location.split(',')[0].strip()

        demographic_profile = None
        if demographics:
            demographic_profile = parse_demographic_group(demographics)

        coords = get_coordinates(location)
        
        if not coords and ',' in location:
            parts = location.split(',', 1)
            if len(parts) > 1:
                potential_address = parts[1].strip()
                print(f"Full string geocode failed. Retrying with potential address: '{potential_address}'")
                coords = get_coordinates(potential_address)

        if coords:
            lat, lon = coords
            analysis = analyze_safety(lat, lon)
            amenities = analyze_amenities(lat, lon)
            
            base_report = generate_safety_report(analysis, location, amenity_type)

            if demographic_profile:
                custom_recommendations = generate_custom_safety_recommendations(
                    demographic_profile,
                    analysis['safety_score'],
                    analysis.get('crime_types', {})
                )
                full_report = base_report + "\n" + custom_recommendations
            else:
                full_report = base_report

            return {
                "text": full_report,
                "lat": lat,
                "lon": lon,
                "crime_types": analysis.get('crime_types', {}),
                "demographic_profile": demographic_profile,
                "amenities": amenities,
                "graph": {
                    "type": "bar",
                    "data": {
                        "labels": list(analysis.get('crime_types', {}).keys()),
                        "datasets": [{
                            "data": list(analysis.get('crime_types', {}).values()),
                            "backgroundColor": "#ec4899"
                        }]
                    }
                }
            }
            
        raise HTTPException(status_code=400, detail="Invalid location")
        
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        print(f"Server Error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal server error")
