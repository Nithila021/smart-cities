# geo_utils.py
import re
import pandas as pd
from geopy.geocoders import Nominatim
from math import radians, cos, sin, sqrt, atan2

def haversine(lat1, lon1, lat2, lon2):
    """Calculate distance between two points in km using Haversine formula"""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c
def get_coordinates(address):
    print(f"\n--- Geocoding attempt for: '{address}' ---")
    coord_pattern = r'(-?\d+\.?\d*),\s*(-?\d+\.?\d*)'
    
    # Check for direct coordinate input
    if match := re.search(coord_pattern, address):
        lat, lon = float(match.group(1)), float(match.group(2))
        print(f"Direct coordinates found: {lat}, {lon}")
        if 40.4 <= lat <= 41.0 and -74.3 <= lon <= -73.7:
            return lat, lon
        print("Coordinates outside NYC area")
        return None
    
    # Clean up address - remove business types and extra commas
    # This helps the geocoder focus on the location part
    cleaned_address = re.sub(r'^(restaurant|cafe|park|store|shop|mall),\s*', '', address, flags=re.IGNORECASE)
    
    # Geocode address
    geolocator = Nominatim(user_agent="safety_app", timeout=15)
    try:
        print(f"Geocoding cleaned address: '{cleaned_address}'")
        location = geolocator.geocode(f"{cleaned_address}, New York City")
        
        if location:
            print(f"Geocoding result: {location.address}")
            print(f"Coordinates: {location.latitude}, {location.longitude}")
            if 40.4 <= location.latitude <= 41.0 and -74.3 <= location.longitude <= -73.7:
                return location.latitude, location.longitude
            print("Geocoded coordinates outside NYC bounds")
            # Consider returning coordinates anyway with a warning
            return location.latitude, location.longitude
        print("No geocoding results found")
        return None
        
    except Exception as e:
        print(f"Geocoding error: {str(e)}")
        return None
def find_nearby_points(df, lat, lon, distance_km=3):
    """Find points within specified distance using haversine"""
    # First filter with bounding box (faster)
    # 0.01 degrees is roughly 1.1km at NYC's latitude
    nearby = df[
        (df['latitude'].between(lat - distance_km/111, lat + distance_km/111)) &
        (df['longitude'].between(lon - distance_km/(111 * cos(radians(lat))), 
                                lon + distance_km/(111 * cos(radians(lat)))))
    ]
    
    # Then apply precise distance calculation
    nearby['distance'] = nearby.apply(
        lambda row: haversine(lat, lon, row['latitude'], row['longitude']), axis=1)
    return nearby[nearby['distance'] <= distance_km]
