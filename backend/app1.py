from flask import Flask, request, jsonify
from flask_cors import CORS
from geo_utils import get_coordinates
from data_init import initialize_data, cached_data
from analysis import analyze_safety, analyze_amenities
from models import get_crime_density_classification
from utils import generate_safety_report
from demographic_parser import parse_demographic_group, generate_custom_safety_recommendations

app = Flask(__name__)

# Configure CORS
CORS(app, resources={
	    r"/*": {
	        "origins": [
	            "http://localhost:3000",  # Local development
	            "http://localhost:5000",  # Local development (if needed)
	            "https://your-frontend-domain.com"  # Replace with your actual frontend domain
	        ]
	    }
})


def ensure_data_initialized():
    """Ensure crime data and models are loaded into the global cache."""
    if cached_data.get('df') is None:
        initialize_data()


# --------------------------
# Helper Functions (Moved to analysis.py in production)
# --------------------------
def get_crime_heatmap():
    ensure_data_initialized()
    df = cached_data.get('df')
    if df is None or df.empty:
        return []
    sample_size = min(5000, len(df))
    heatmap_data = df.sample(sample_size)[['latitude', 'longitude', 'crime_type']].copy()

    # Use severity_weight column if available, otherwise fall back to cached severity dict
    if 'severity_weight' in df.columns:
        heatmap_data['weight'] = df.sample(sample_size)['severity_weight']
    else:
        crime_severity = cached_data.get('crime_severity', {})
        heatmap_data['weight'] = heatmap_data['crime_type'].map(
            lambda x: crime_severity.get(x, 3) / 10
        )
    return heatmap_data[['latitude', 'longitude', 'weight']].to_dict('records')


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
            "id": cluster_id,  # cluster_id is the dict key, not a field
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
@app.route('/api/analyze_v1', methods=['POST'])
def analyze_endpoint_v1():
    data = request.get_json()
    location = data.get('location', '')
    include_advanced = data.get('include_advanced', False)
    
    if coords := get_coordinates(location):
        lat, lon = coords
        analysis = analyze_safety(lat, lon)
        
        if data.get('include_amenities', False):
            analysis['amenities'] = analyze_amenities(lat, lon)
            
        if include_advanced:
            demographic_data = get_demographic_analysis_data()
            if 'error' not in demographic_data:
                analysis['demographic_analysis'] = demographic_data
                
        return jsonify(analysis)
    
    return jsonify({"error": "Invalid location"}), 400

@app.route('/api/heatmap', methods=['GET'])
def heatmap_endpoint():
    heatmap_data = get_crime_heatmap()
    return jsonify(heatmap_data)

@app.route('/api/density_map', methods=['GET'])
def density_map_endpoint():
    density_data = get_crime_density_map_data()
    return jsonify(density_data)

@app.route('/api/demographic_zones', methods=['GET'])
def demographic_zones_endpoint():
    demographic_data = get_demographic_analysis_data()
    return jsonify(demographic_data)

@app.route('/api/dbscan_clusters', methods=['GET'])
def dbscan_clusters_endpoint():
    dbscan_data = cached_data.get('dbscan_clusters')
    if not dbscan_data:
        return jsonify({"error": "DBSCAN clustering data not available"}), 404
    
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
    
    return jsonify({'clusters': clusters})

@app.route('/api/analyze_v2', methods=['POST'])
def analyze_endpoint_v2():
    data = request.get_json()
    location = data.get('location', '')
    
    if coords := get_coordinates(location):
        lat, lon = coords
        analysis = analyze_safety(lat, lon)
        amenities = analyze_amenities(lat, lon)
        
        # Add coordinates and raw crime data to response
        response_data = {
            **analysis,
            "lat": lat,
            "lon": lon,
            "crime_types": analysis.get('crime_types', {}),
            "amenities": amenities,
            "density": get_crime_density_classification(lat, lon)
        }
        
        return jsonify(response_data)
    
    return jsonify({"error": "Invalid location"}), 400

@app.route('/api/map_data', methods=['GET'])
def map_data_endpoint():
    return jsonify({
        "dbscan_clusters": get_dbscan_clusters_data(),
        "demographic_zones": get_demographic_zones_data(),
        "density_zones": get_crime_density_map_data()
    })


@app.route('/api/chat', methods=['POST'])
def chat_endpoint():
    try:
        data = request.get_json()
        location = data.get('message', '')
        demographics = data.get('demographics', '')  # Optional custom demographics
        
        if not location:
            return jsonify({"error": "Empty request"}), 400

        # Extract amenity type from location string (before first comma)
        amenity_type = None
        if ',' in location:
            amenity_type = location.split(',')[0].strip()

        # Parse custom demographics if provided
        demographic_profile = None
        if demographics:
            demographic_profile = parse_demographic_group(demographics)

        if coords := get_coordinates(location):
            lat, lon = coords
            analysis = analyze_safety(lat, lon)
            amenities = analyze_amenities(lat, lon)
            
            # Generate base report
            base_report = generate_safety_report(analysis, location, amenity_type)

            # Add custom demographic analysis if provided
            if demographic_profile:
                custom_recommendations = generate_custom_safety_recommendations(
                    demographic_profile,
                    analysis['safety_score'],
                    analysis.get('crime_types', {})
                )
                full_report = base_report + "\n" + custom_recommendations
            else:
                full_report = base_report

            return jsonify({
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
            })
            
        return jsonify({"error": "Invalid location"}), 400
        
    except Exception as e:  # REQUIRED EXCEPT CLAUSE
        print(f"Server Error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500
    
@app.route('/')
def health_check():
    return "Safety Analysis Service Running"

if __name__ == '__main__':
    initialize_data()
    app.run(host='0.0.0.0', port=5000, debug=True)