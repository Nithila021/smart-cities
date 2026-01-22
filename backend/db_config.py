"""
Database configuration and connection management for Smart Cities Crime Safety Analysis.

This module provides PostgreSQL + PostGIS connection handling for the Flask backend.
"""

import os
from contextlib import contextmanager
from typing import Optional, Dict, Any, List
import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor

# Database configuration from environment variables with defaults
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', 5432)),
    'database': os.getenv('DB_NAME', 'smart_cities_db'),
    'user': os.getenv('DB_USER', os.getenv('USER', 'postgres')),
    'password': os.getenv('DB_PASSWORD', ''),
}

# Connection pool (initialized lazily)
_connection_pool: Optional[pool.ThreadedConnectionPool] = None


def get_connection_pool() -> pool.ThreadedConnectionPool:
    """Get or create the database connection pool."""
    global _connection_pool
    if _connection_pool is None:
        _connection_pool = pool.ThreadedConnectionPool(
            minconn=2,
            maxconn=10,
            **DB_CONFIG
        )
    return _connection_pool


@contextmanager
def get_db_connection():
    """Context manager for database connections.
    
    Usage:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM crime_incidents LIMIT 10")
                results = cur.fetchall()
    """
    conn = None
    try:
        conn = get_connection_pool().getconn()
        yield conn
    finally:
        if conn:
            get_connection_pool().putconn(conn)


@contextmanager
def get_db_cursor(dict_cursor: bool = True):
    """Context manager for database cursors with automatic commit.
    
    Args:
        dict_cursor: If True, returns results as dictionaries
        
    Usage:
        with get_db_cursor() as cur:
            cur.execute("SELECT * FROM crime_incidents LIMIT 10")
            results = cur.fetchall()
    """
    with get_db_connection() as conn:
        cursor_factory = RealDictCursor if dict_cursor else None
        cur = conn.cursor(cursor_factory=cursor_factory)
        try:
            yield cur
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cur.close()


def execute_query(query: str, params: tuple = None, fetch: bool = True) -> Optional[List[Dict]]:
    """Execute a query and optionally fetch results.
    
    Args:
        query: SQL query string
        params: Query parameters (for parameterized queries)
        fetch: If True, fetch and return results
        
    Returns:
        List of dictionaries if fetch=True, None otherwise
    """
    with get_db_cursor() as cur:
        cur.execute(query, params)
        if fetch:
            return cur.fetchall()
    return None


def get_nearby_crimes(lat: float, lon: float, radius_meters: int = 500) -> List[Dict]:
    """Get crimes within a radius of a location using PostGIS.
    
    Args:
        lat: Latitude
        lon: Longitude
        radius_meters: Search radius in meters
        
    Returns:
        List of crime incidents with distance
    """
    query = """
        SELECT * FROM get_nearby_crimes(%s, %s, %s)
    """
    return execute_query(query, (lat, lon, radius_meters))


def calculate_risk_score(lat: float, lon: float, radius_meters: int = 500) -> float:
    """Calculate risk score for a location.
    
    Args:
        lat: Latitude
        lon: Longitude
        radius_meters: Analysis radius in meters
        
    Returns:
        Risk score (0-100)
    """
    query = "SELECT calculate_location_risk(%s, %s, %s) as risk_score"
    result = execute_query(query, (lat, lon, radius_meters))
    if result and len(result) > 0:
        return float(result[0]['risk_score'] or 0)
    return 0.0


def get_demographic_weights(demographic_group: str = 'general') -> Dict[str, float]:
    """Get demographic impact weights for risk calculation.
    
    Args:
        demographic_group: One of 'children', 'women', 'elderly', 'general'
        
    Returns:
        Dictionary mapping crime_category to impact_weight
    """
    query = """
        SELECT crime_category, impact_weight 
        FROM demographic_impact_weights 
        WHERE demographic_group = %s
    """
    results = execute_query(query, (demographic_group,))
    return {r['crime_category']: float(r['impact_weight']) for r in results}


def close_pool():
    """Close the connection pool (call on application shutdown)."""
    global _connection_pool
    if _connection_pool:
        _connection_pool.closeall()
        _connection_pool = None


def test_connection() -> bool:
    """Test database connection."""
    try:
        with get_db_cursor() as cur:
            cur.execute("SELECT PostGIS_Version()")
            result = cur.fetchone()
            print(f"Connected to PostgreSQL with PostGIS: {result['postgis_version']}")
            return True
    except Exception as e:
        print(f"Database connection failed: {e}")
        return False


if __name__ == "__main__":
    # Test the connection when run directly
    test_connection()

