# geo_utils.py
import re
# import os
# import pandas as pd
from geopy.geocoders import Nominatim
from math import radians, cos, sin, sqrt, atan2
from typing import Tuple, Optional, Dict

# City name mappings to standardized codes
# Add more mappings as you scale to new cities
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
    # Add more cities as needed
}


def detect_city_from_address(address_components: Dict) -> Optional[str]:
    """
    Extract standardized city code from geocoded address components.

    Args:
        address_components: Dict from Nominatim's raw address data

    Returns:
        Standardized city code ('nyc', 'london', etc.) or None if unknown
    """
    # Nominatim returns address in 'raw' with 'address' dict containing:
    # city, town, village, municipality, county, state, country, etc.

    # Priority order for city detection
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

    Args:
        lat: Latitude
        lon: Longitude

    Returns:
        Standardized city code or None
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

    Args:
        address: User-provided address string

    Returns:
        Tuple of (latitude, longitude, city_code) or None if geocoding fails
        city_code is standardized ('nyc', 'london', etc.) or 'unknown'
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

    For new code, prefer get_coordinates_with_city() which also returns city.

    Args:
        address: User-provided address string

    Returns:
        Tuple of (latitude, longitude) or None if geocoding fails
    """
    result = get_coordinates_with_city(address)
    if result:
        return result[0], result[1]  # Just lat, lon
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
