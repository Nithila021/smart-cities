-- Smart Cities Crime Safety Analysis System
-- PostgreSQL + PostGIS Database Schema
-- Created: 2026-01-20

-- Enable PostGIS extension (if not already enabled)
CREATE EXTENSION IF NOT EXISTS postgis;

-- ============================================================================
-- CORE TABLES
-- ============================================================================

-- Core crime incidents table
CREATE TABLE IF NOT EXISTS crime_incidents (
    id SERIAL PRIMARY KEY,
    location GEOGRAPHY(POINT, 4326),
    crime_type VARCHAR(100),
    crime_category VARCHAR(50),  -- sexual, harassment, assault, theft, property
    severity_weight DECIMAL(3,2) DEFAULT 1.0,
    occurred_at TIMESTAMP,
    day_of_week INTEGER,         -- 0=Sunday, 6=Saturday
    hour_of_day INTEGER,         -- 0-23
    victim_demographics JSONB,   -- {age_group, sex, race}
    precinct VARCHAR(20),
    borough VARCHAR(50),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Spatial index for fast geospatial queries
CREATE INDEX IF NOT EXISTS idx_crime_location ON crime_incidents USING GIST(location);
CREATE INDEX IF NOT EXISTS idx_crime_time ON crime_incidents(occurred_at);
CREATE INDEX IF NOT EXISTS idx_crime_type ON crime_incidents(crime_category);
CREATE INDEX IF NOT EXISTS idx_crime_hour ON crime_incidents(hour_of_day);
CREATE INDEX IF NOT EXISTS idx_crime_dow ON crime_incidents(day_of_week);

-- Real amenities/POIs from Overpass API
CREATE TABLE IF NOT EXISTS amenities (
    id SERIAL PRIMARY KEY,
    location GEOGRAPHY(POINT, 4326),
    name VARCHAR(255),
    amenity_type VARCHAR(50),    -- bar, restaurant, park, transit_station
    operating_hours JSONB,       -- {open: "09:00", close: "22:00"}
    neighborhood VARCHAR(100),
    borough VARCHAR(50),
    osm_id BIGINT UNIQUE,        -- OpenStreetMap ID for deduplication
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_amenity_location ON amenities USING GIST(location);
CREATE INDEX IF NOT EXISTS idx_amenity_type ON amenities(amenity_type);

-- Pre-computed risk scores (cached for performance)
CREATE TABLE IF NOT EXISTS risk_cache (
    id SERIAL PRIMARY KEY,
    grid_cell_id VARCHAR(50),    -- H3 or custom grid cell identifier
    center_point GEOGRAPHY(POINT, 4326),
    crime_category VARCHAR(50),
    hour_bucket INTEGER,         -- 0-23
    day_type VARCHAR(10),        -- weekday, weekend
    risk_score DECIMAL(5,2),
    confidence DECIMAL(3,2),
    sample_size INTEGER,
    computed_at TIMESTAMP DEFAULT NOW(),
    expires_at TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_risk_grid ON risk_cache(grid_cell_id);
CREATE INDEX IF NOT EXISTS idx_risk_location ON risk_cache USING GIST(center_point);
CREATE INDEX IF NOT EXISTS idx_risk_hour ON risk_cache(hour_bucket);

-- Demographic impact weights (configurable based on research)
CREATE TABLE IF NOT EXISTS demographic_impact_weights (
    id SERIAL PRIMARY KEY,
    crime_category VARCHAR(50),
    demographic_group VARCHAR(50),  -- children, women, elderly, general
    impact_weight DECIMAL(3,2),
    source VARCHAR(255),            -- research citation
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_demo_unique 
    ON demographic_impact_weights(crime_category, demographic_group);

-- ============================================================================
-- SEED DATA: Default demographic impact weights
-- ============================================================================

INSERT INTO demographic_impact_weights (crime_category, demographic_group, impact_weight, source) VALUES
    ('sexual', 'women', 1.8, 'FBI UCR 2023'),
    ('sexual', 'children', 2.0, 'FBI UCR 2023'),
    ('sexual', 'elderly', 1.2, 'FBI UCR 2023'),
    ('sexual', 'general', 1.0, 'FBI UCR 2023'),
    ('harassment', 'women', 1.5, 'NCVS 2023'),
    ('harassment', 'children', 1.3, 'NCVS 2023'),
    ('harassment', 'elderly', 1.1, 'NCVS 2023'),
    ('harassment', 'general', 1.0, 'NCVS 2023'),
    ('assault', 'women', 1.3, 'NCVS 2023'),
    ('assault', 'children', 1.5, 'NCVS 2023'),
    ('assault', 'elderly', 1.6, 'NCVS 2023'),
    ('assault', 'general', 1.0, 'NCVS 2023'),
    ('theft', 'women', 1.1, 'NCVS 2023'),
    ('theft', 'children', 1.0, 'NCVS 2023'),
    ('theft', 'elderly', 1.4, 'NCVS 2023'),
    ('theft', 'general', 1.0, 'NCVS 2023'),
    ('robbery', 'women', 1.2, 'NCVS 2023'),
    ('robbery', 'children', 1.3, 'NCVS 2023'),
    ('robbery', 'elderly', 1.5, 'NCVS 2023'),
    ('robbery', 'general', 1.0, 'NCVS 2023')
ON CONFLICT (crime_category, demographic_group) DO NOTHING;

-- ============================================================================
-- HELPER FUNCTIONS
-- ============================================================================

-- Function to get nearby crimes within a radius (in meters)
CREATE OR REPLACE FUNCTION get_nearby_crimes(
    lat DOUBLE PRECISION,
    lon DOUBLE PRECISION,
    radius_meters INTEGER DEFAULT 500
)
RETURNS TABLE (
    id INTEGER,
    crime_type VARCHAR,
    crime_category VARCHAR,
    distance_meters DOUBLE PRECISION,
    occurred_at TIMESTAMP
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        c.id,
        c.crime_type,
        c.crime_category,
        ST_Distance(c.location, ST_SetSRID(ST_MakePoint(lon, lat), 4326)::geography) as distance_meters,
        c.occurred_at
    FROM crime_incidents c
    WHERE ST_DWithin(
        c.location,
        ST_SetSRID(ST_MakePoint(lon, lat), 4326)::geography,
        radius_meters
    )
    ORDER BY distance_meters;
END;
$$ LANGUAGE plpgsql;

-- Function to calculate risk score for a location
CREATE OR REPLACE FUNCTION calculate_location_risk(
    lat DOUBLE PRECISION,
    lon DOUBLE PRECISION,
    radius_meters INTEGER DEFAULT 500,
    time_window_days INTEGER DEFAULT 365
)
RETURNS DECIMAL AS $$
DECLARE
    crime_count INTEGER;
    weighted_score DECIMAL;
BEGIN
    SELECT 
        COUNT(*),
        COALESCE(SUM(severity_weight), 0)
    INTO crime_count, weighted_score
    FROM crime_incidents
    WHERE ST_DWithin(
        location,
        ST_SetSRID(ST_MakePoint(lon, lat), 4326)::geography,
        radius_meters
    )
    AND occurred_at >= NOW() - (time_window_days || ' days')::INTERVAL;
    
    -- Normalize to 0-100 scale (adjust divisor based on your data)
    RETURN LEAST(100, (weighted_score / GREATEST(crime_count, 1)) * 10);
END;
$$ LANGUAGE plpgsql;

