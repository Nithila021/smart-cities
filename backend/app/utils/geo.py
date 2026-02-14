# app/utils/geo.py
import re
from geopy.geocoders import Nominatim
from math import radians, cos, sin, sqrt, atan2
from typing import Tuple, Optional, Dict
import polars as pl
import numpy as np

# City name mappings to standardized codes
CITY_NAME_MAPPINGS = {
    # New York variations
    'new york': 'nyc',
    'new york city': 'nyc',
    'new york county': 'nyc',
    'nyc': 'nyc',
    'manhattan': 'nyc',
    'brooklyn': 'nyc',
    'kings county': 'nyc',  # Brooklyn's county name
    'queens': 'nyc',
    'queens county': 'nyc',
    'bronx': 'nyc',
    'bronx county': 'nyc',
    'staten island': 'nyc',
    'richmond county': 'nyc',  # Staten Island's county name
    # London variations
    'london': 'london',
    'greater london': 'london',
    'city of london': 'london',
    'city of westminster': 'london',
    'westminster': 'london',
    'tower hamlets': 'london',
    'camden': 'london',
    'islington': 'london',
    'hackney': 'london',
    'southwark': 'london',
    'lambeth': 'london',
    # Chicago variations
    'chicago': 'chicago',
    'cook county': 'chicago',
}


def detect_city_from_address(address_components: Dict) -> Optional[str]:
    """
    Extract standardized city code from geocoded address components.
    """
    city_fields = ['city', 'town', 'municipality', 'village', 'county', 'state']

    for field in city_fields:
        if field in address_components:
            city_name = address_components[field].lower().strip()
            if city_name in CITY_NAME_MAPPINGS:
                return CITY_NAME_MAPPINGS[city_name]

    # Check borough/district for NYC (Manhattan, Brooklyn, etc.)
    for field in ['borough', 'suburb', 'district', 'neighbourhood']:
        if field in address_components:
            area_name = address_components[field].lower().strip()
            if area_name in CITY_NAME_MAPPINGS:
                return CITY_NAME_MAPPINGS[area_name]

    return None


def detect_city_from_coordinates(lat: float, lon: float) -> Optional[str]:
    """
    Detect city from GPS coordinates using reverse geocoding.
    """
    geolocator = Nominatim(user_agent="safety_app", timeout=15)
    try:
        location = geolocator.reverse(f"{lat}, {lon}", language='en', addressdetails=True)
        if location and 'address' in location.raw:
            return detect_city_from_address(location.raw['address'])
    except Exception as e:
        print(f"Reverse geocoding error: {e}")

    return None


def haversine(lat1, lon1, lat2, lon2):
    """Calculate distance between two points in km using Haversine formula"""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c


def get_coordinates_with_city(address: str) -> Optional[Tuple[float, float, str]]:
    """
    Geocode an address and detect the city.
    """
    print(f"\n--- Geocoding attempt for: '{address}' ---")
    coord_pattern = r'(-?\d+\.?\d*),\s*(-?\d+\.?\d*)'

    # Check for direct coordinate input
    if match := re.search(coord_pattern, address):
        lat, lon = float(match.group(1)), float(match.group(2))
        print(f"Direct coordinates found: {lat}, {lon}")
        # Detect city from coordinates
        city = detect_city_from_coordinates(lat, lon) or 'unknown'
        print(f"Detected city: {city}")
        return lat, lon, city

    # Clean up address - remove business types and extra commas
    cleaned_address = re.sub(r'^(restaurant|cafe|park|store|shop|mall),\s*', '', address, flags=re.IGNORECASE)

    # Geocode address (without forcing NYC)
    geolocator = Nominatim(user_agent="safety_app", timeout=15)
    try:
        print(f"Geocoding cleaned address: '{cleaned_address}'")
        # First try without city suffix for flexibility
        location = geolocator.geocode(cleaned_address, addressdetails=True)

        if location:
            print(f"Geocoding result: {location.address}")
            print(f"Coordinates: {location.latitude}, {location.longitude}")

            # Detect city from address components
            city = 'unknown'
            if 'address' in location.raw:
                city = detect_city_from_address(location.raw['address']) or 'unknown'
            print(f"Detected city: {city}")

            return location.latitude, location.longitude, city

        print("No geocoding results found")
        return None

    except Exception as e:
        print(f"Geocoding error: {str(e)}")
        return None


def get_coordinates(address: str) -> Optional[Tuple[float, float]]:
    """
    Geocode an address to coordinates (backward compatible).
    """
    result = get_coordinates_with_city(address)
    if result:
        return result[0], result[1]  # Just lat, lon
    return None


def find_nearby_points(df: pl.DataFrame, lat: float, lon: float, distance_km: float = 3.0) -> pl.DataFrame:
    """Find points within specified distance using haversine with Polars."""
    # First filter with bounding box (faster)
    # 1 degree lat approx 111 km
    lat_diff = distance_km / 111.0
    lon_diff = distance_km / (111.0 * cos(radians(lat)))
    
    # Bounding box filter
    nearby = df.filter(
        (pl.col('latitude') >= lat - lat_diff) &
        (pl.col('latitude') <= lat + lat_diff) &
        (pl.col('longitude') >= lon - lon_diff) &
        (pl.col('longitude') <= lon + lon_diff)
    )
    
    if nearby.height == 0:
        return nearby.with_columns(pl.lit(0.0).alias('distance'))
        
    # Apply precise distance calculation
    # Polars map_elements is slow for large data, but we filtered significantly
    # Ideally use a vectorized custom expression or UDF if possible, but map_elements is easiest port
    
    # We can use apply (map_rows) on select columns
    # Or struct mapping
    
    # Function to apply
    def calc_dist(struct):
        return haversine(lat, lon, struct['latitude'], struct['longitude'])
        
    nearby = nearby.with_columns(
        pl.struct(['latitude', 'longitude'])
        .map_elements(calc_dist, return_dtype=pl.Float64)
        .alias('distance')
    )
    
    return nearby.filter(pl.col('distance') <= distance_km)
