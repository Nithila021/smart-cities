"""
Data Cleaning Pipeline for Smart Cities Crime Safety Analysis.

This module provides reusable, testable cleaning functions for NYPD crime data.
Each function is pure (no side effects) and can be used independently.

Usage:
    from data_cleaner import DataCleaner
    
    cleaner = DataCleaner()
    clean_df, report = cleaner.clean(raw_df)
    print(report.summary())
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from datetime import datetime


@dataclass
class CleaningReport:
    """Report of what was cleaned/removed during data processing."""
    initial_rows: int = 0
    final_rows: int = 0
    steps: List[Dict] = field(default_factory=list)
    
    def add_step(self, name: str, removed: int, details: str = ""):
        """Record a cleaning step."""
        self.steps.append({
            'name': name,
            'removed': removed,
            'details': details,
            'timestamp': datetime.now().isoformat()
        })
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 60,
            "DATA CLEANING REPORT",
            "=" * 60,
            f"Initial rows: {self.initial_rows:,}",
            f"Final rows:   {self.final_rows:,}",
            f"Total removed: {self.initial_rows - self.final_rows:,} ({self.removal_percentage:.1f}%)",
            "-" * 60,
            "STEPS:",
        ]
        for step in self.steps:
            lines.append(f"  • {step['name']}: -{step['removed']:,} rows")
            if step['details']:
                lines.append(f"    └─ {step['details']}")
        lines.append("=" * 60)
        return "\n".join(lines)
    
    @property
    def removal_percentage(self) -> float:
        if self.initial_rows == 0:
            return 0.0
        return ((self.initial_rows - self.final_rows) / self.initial_rows) * 100


# =============================================================================
# CLEANING CONFIGURATION
# =============================================================================

# City bounds configuration (optional override for auto-detection)
# Set ACTIVE_CITY env var to use a preset, or leave empty for auto-detection
CITY_BOUNDS_CONFIG = {
    'nyc': {
        'name': 'New York City',
        'lat_min': 40.4, 'lat_max': 41.0,
        'lon_min': -74.3, 'lon_max': -73.7
    },
    'london': {
        'name': 'London',
        'lat_min': 51.28, 'lat_max': 51.69,
        'lon_min': -0.51, 'lon_max': 0.33
    },
    'chicago': {
        'name': 'Chicago',
        'lat_min': 41.64, 'lat_max': 42.02,
        'lon_min': -87.94, 'lon_max': -87.52
    },
    # Add more cities as needed
}

def auto_detect_bounds(df: pd.DataFrame,
                       lat_col: str = 'latitude',
                       lon_col: str = 'longitude',
                       outlier_percentile: float = 1.0) -> Dict:
    """
    Automatically detect geographic bounds from data.
    Removes outliers using percentile filtering.

    Works for ANY city without configuration.

    Args:
        df: DataFrame with coordinate columns
        lat_col: Name of latitude column
        lon_col: Name of longitude column
        outlier_percentile: Percentile to trim from each end (default 1%)

    Returns:
        Dict with lat_min, lat_max, lon_min, lon_max
    """
    lower = outlier_percentile / 100
    upper = 1 - lower

    # Get numeric coordinates only
    lat_series = pd.to_numeric(df[lat_col], errors='coerce').dropna()
    lon_series = pd.to_numeric(df[lon_col], errors='coerce').dropna()

    bounds = {
        'lat_min': float(lat_series.quantile(lower)),
        'lat_max': float(lat_series.quantile(upper)),
        'lon_min': float(lon_series.quantile(lower)),
        'lon_max': float(lon_series.quantile(upper)),
    }

    return bounds


def get_bounds(df: pd.DataFrame = None, city: str = None) -> Dict:
    """
    Get geographic bounds - either from config or auto-detected.

    Priority:
    1. Explicit city parameter
    2. ACTIVE_CITY environment variable
    3. Auto-detect from data (if df provided)
    4. Default to NYC (fallback)

    Args:
        df: Optional DataFrame for auto-detection
        city: Optional city code ('nyc', 'london', etc.)

    Returns:
        Dict with lat_min, lat_max, lon_min, lon_max
    """
    import os

    # Check explicit city parameter
    if city and city in CITY_BOUNDS_CONFIG:
        config = CITY_BOUNDS_CONFIG[city]
        return {k: v for k, v in config.items() if k != 'name'}

    # Check environment variable
    env_city = os.getenv('ACTIVE_CITY', '').lower()
    if env_city and env_city in CITY_BOUNDS_CONFIG:
        config = CITY_BOUNDS_CONFIG[env_city]
        return {k: v for k, v in config.items() if k != 'name'}

    # Auto-detect from data
    if df is not None and len(df) > 0:
        return auto_detect_bounds(df)

    # Fallback to NYC
    config = CITY_BOUNDS_CONFIG['nyc']
    return {k: v for k, v in config.items() if k != 'name'}

# Crime type standardization mappings
CRIME_TYPE_MAPPINGS = {
    r'ASSAULT.*3.*': 'ASSAULT_3',
    r'HARRASSMENT': 'HARASSMENT',  # Fix common misspelling
    r'DRIVING WHILE INTOXICATED': 'DWI',
    r'CRIMINAL MISCHIEF.*': 'CRIMINAL_MISCHIEF',
    r'SEX CRIMES': 'SEX_CRIMES',
    r'PETIT LARCENY': 'PETIT_LARCENY',
    r'GRAND LARCENY.*': 'GRAND_LARCENY',
}

# Crime category mappings (for demographic weighting)
CRIME_CATEGORY_MAP = {
    'RAPE': 'sexual',
    'SEX_CRIMES': 'sexual',
    'HARASSMENT': 'harassment',
    'ASSAULT': 'assault',
    'ASSAULT_3': 'assault',
    'FELONY ASSAULT': 'assault',
    'ROBBERY': 'robbery',
    'GRAND_LARCENY': 'theft',
    'PETIT_LARCENY': 'theft',
    'BURGLARY': 'property',
    'CRIMINAL_MISCHIEF': 'property',
}

# Severity weights for risk scoring
SEVERITY_WEIGHTS = {
    'MURDER & NON-NEGL. MANSLAUGHTER': 1.0,
    'RAPE': 0.95,
    'FELONY ASSAULT': 0.85,
    'ROBBERY': 0.80,
    'ASSAULT_3': 0.70,
    'BURGLARY': 0.60,
    'DWI': 0.60,
    'GRAND_LARCENY': 0.50,
    'PETIT_LARCENY': 0.40,
    'HARASSMENT': 0.40,
    'CRIMINAL_MISCHIEF': 0.30,
}


# =============================================================================
# PURE CLEANING FUNCTIONS
# =============================================================================

def clean_coordinates(df: pd.DataFrame,
                      lat_col: str = 'latitude',
                      lon_col: str = 'longitude',
                      bounds: Dict = None,
                      city: str = None) -> Tuple[pd.DataFrame, int, Dict]:
    """
    Validate and filter coordinates to geographic bounds.

    Uses auto-detection or config-based bounds.

    Args:
        df: DataFrame with coordinate columns
        lat_col: Name of latitude column
        lon_col: Name of longitude column
        bounds: Optional explicit bounds dict
        city: Optional city code for config lookup

    Returns:
        Tuple of (cleaned DataFrame, number of rows removed, bounds used)
    """
    initial_count = len(df)

    # Convert to numeric, coercing errors to NaN
    df[lat_col] = pd.to_numeric(df[lat_col], errors='coerce')
    df[lon_col] = pd.to_numeric(df[lon_col], errors='coerce')

    # Get bounds (explicit → config → auto-detect → NYC fallback)
    if bounds is None:
        bounds = get_bounds(df=df, city=city)

    # Filter to bounds
    valid_mask = (
        (df[lat_col] >= bounds['lat_min']) &
        (df[lat_col] <= bounds['lat_max']) &
        (df[lon_col] >= bounds['lon_min']) &
        (df[lon_col] <= bounds['lon_max'])
    )

    df_clean = df[valid_mask].copy()
    removed = initial_count - len(df_clean)

    return df_clean, removed, bounds


def clean_crime_types(df: pd.DataFrame,
                      crime_col: str = 'crime_type') -> Tuple[pd.DataFrame, Dict]:
    """
    Standardize crime type names using regex mappings.
    
    Args:
        df: DataFrame with crime type column
        crime_col: Name of crime type column
        
    Returns:
        Tuple of (cleaned DataFrame, mapping statistics)
    """
    df = df.copy()
    stats = {'mappings_applied': 0, 'unique_before': 0, 'unique_after': 0}
    
    if crime_col not in df.columns:
        return df, stats
    
    # Uppercase and strip
    df[crime_col] = df[crime_col].astype(str).str.upper().str.strip()
    stats['unique_before'] = df[crime_col].nunique()
    
    # Apply regex mappings
    for pattern, replacement in CRIME_TYPE_MAPPINGS.items():
        mask = df[crime_col].str.contains(pattern, regex=True, na=False)
        stats['mappings_applied'] += mask.sum()
        df.loc[mask, crime_col] = replacement
    
    stats['unique_after'] = df[crime_col].nunique()

    return df, stats


def add_crime_category(df: pd.DataFrame,
                       crime_col: str = 'crime_type',
                       category_col: str = 'crime_category') -> pd.DataFrame:
    """
    Add crime category based on crime type for demographic weighting.

    Args:
        df: DataFrame with crime type column
        crime_col: Name of crime type column
        category_col: Name for new category column

    Returns:
        DataFrame with added category column
    """
    df = df.copy()

    def get_category(crime_type: str) -> str:
        if not crime_type or pd.isna(crime_type):
            return 'other'
        crime_upper = str(crime_type).upper()
        for pattern, category in CRIME_CATEGORY_MAP.items():
            if pattern in crime_upper:
                return category
        return 'other'

    df[category_col] = df[crime_col].apply(get_category)
    return df


def add_severity_weight(df: pd.DataFrame,
                        crime_col: str = 'crime_type',
                        weight_col: str = 'severity_weight') -> pd.DataFrame:
    """
    Add severity weight for risk scoring.

    Args:
        df: DataFrame with crime type column
        crime_col: Name of crime type column
        weight_col: Name for new weight column

    Returns:
        DataFrame with added weight column
    """
    df = df.copy()

    def get_weight(crime_type: str) -> float:
        if not crime_type or pd.isna(crime_type):
            return 0.5
        crime_upper = str(crime_type).upper()
        for pattern, weight in SEVERITY_WEIGHTS.items():
            if pattern in crime_upper:
                return weight
        return 0.5  # Default weight

    df[weight_col] = df[crime_col].apply(get_weight)
    return df


def clean_demographics(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Clean victim demographic columns.

    Args:
        df: DataFrame with vic_age_group, vic_race, vic_sex columns

    Returns:
        Tuple of (cleaned DataFrame, cleaning statistics)
    """
    df = df.copy()
    stats = {'cleaned_columns': []}

    # Clean age group - standardize format
    if 'vic_age_group' in df.columns:
        df['vic_age_group'] = (
            df['vic_age_group']
            .astype(str)
            .str.upper()
            .str.strip()
            .replace({'NAN': np.nan, 'UNKNOWN': np.nan, '': np.nan})
        )
        stats['cleaned_columns'].append('vic_age_group')

    # Clean race - standardize
    if 'vic_race' in df.columns:
        df['vic_race'] = (
            df['vic_race']
            .astype(str)
            .str.upper()
            .str.strip()
        )
        # Mark unknowns
        unknown_mask = df['vic_race'].str.contains('UNKNOWN', na=False)
        df.loc[unknown_mask, 'vic_race'] = 'UNKNOWN'
        stats['cleaned_columns'].append('vic_race')

    # Clean sex - take first character
    if 'vic_sex' in df.columns:
        df['vic_sex'] = (
            df['vic_sex']
            .astype(str)
            .str.strip()
            .str[0]
            .str.upper()
            .replace({'N': np.nan, 'U': np.nan, '': np.nan})
        )
        stats['cleaned_columns'].append('vic_sex')

    return df, stats


def clean_timestamps(df: pd.DataFrame,
                     date_col: str = 'cmplnt_fr_dt',
                     output_col: str = 'occurred_at') -> Tuple[pd.DataFrame, int]:
    """
    Parse date column into a datetime and add derived time features.

    Args:
        df: DataFrame with date column
        date_col: Name of date column
        output_col: Name for datetime column

    Returns:
        Tuple of (DataFrame with datetime column, number of parse failures)
    """
    df = df.copy()
    parse_failures = 0

    if date_col in df.columns:
        # Try pandas native parsing first (faster)
        df[output_col] = pd.to_datetime(df[date_col], errors='coerce')
        parse_failures = df[output_col].isna().sum()

        # Add derived time features only if datetime column exists
        if output_col in df.columns:
            df['hour_of_day'] = df[output_col].dt.hour.fillna(12).astype(int)
            df['day_of_week'] = df[output_col].dt.dayofweek.fillna(0).astype(int)

    return df, parse_failures


def remove_nulls(df: pd.DataFrame,
                 required_cols: List[str]) -> Tuple[pd.DataFrame, int]:
    """
    Remove rows with null values in required columns.

    Args:
        df: DataFrame to clean
        required_cols: List of column names that cannot be null

    Returns:
        Tuple of (cleaned DataFrame, number of rows removed)
    """
    initial_count = len(df)
    existing_cols = [c for c in required_cols if c in df.columns]
    df_clean = df.dropna(subset=existing_cols)
    removed = initial_count - len(df_clean)
    return df_clean, removed


# =============================================================================
# MAIN CLEANER CLASS
# =============================================================================

class DataCleaner:
    """
    Orchestrates the data cleaning pipeline.

    Usage:
        cleaner = DataCleaner()
        clean_df, report = cleaner.clean(raw_df)
        print(report.summary())
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize cleaner with optional configuration.

        Args:
            config: Optional dict to override default settings
        """
        self.config = config or {}

    def normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize column names to handle different CSV formats."""
        df = df.copy()

        # Map alternative column names to standard names
        column_mappings = {
            'lat_lon.latitude': 'latitude',
            'lat_lon.longitude': 'longitude',
            'Latitude': 'latitude',
            'Longitude': 'longitude',
            'ofns_desc': 'crime_type',
            'offense_description': 'crime_type',
            'OFNS_DESC': 'crime_type',
            'boro_nm': 'borough',
            'BORO_NM': 'borough',
            'VIC_AGE_GROUP': 'vic_age_group',
            'VIC_RACE': 'vic_race',
            'VIC_SEX': 'vic_sex',
            'CMPLNT_FR_DT': 'cmplnt_fr_dt',
            'CMPLNT_FR_TM': 'cmplnt_fr_tm',
        }

        # Apply mappings
        for old_name, new_name in column_mappings.items():
            if old_name in df.columns and new_name not in df.columns:
                df[new_name] = df[old_name]

        return df

    def clean(self, df: pd.DataFrame,
              verbose: bool = True) -> Tuple[pd.DataFrame, CleaningReport]:
        """
        Run the full cleaning pipeline.

        Args:
            df: Raw DataFrame to clean
            verbose: Print progress messages

        Returns:
            Tuple of (cleaned DataFrame, CleaningReport)
        """
        report = CleaningReport(initial_rows=len(df))

        if verbose:
            print(f"Starting cleaning pipeline with {len(df):,} rows...")

        # Step 0: Normalize column names
        df = self.normalize_columns(df)

        # Step 1: Clean coordinates
        if verbose:
            print("  [1/6] Cleaning coordinates...")
        df, removed, bounds_used = clean_coordinates(df)
        report.add_step("Coordinate validation", removed,
                       f"Filtered to bounds ({bounds_used})")

        # Step 2: Clean crime types
        if verbose:
            print("  [2/6] Standardizing crime types...")
        df, stats = clean_crime_types(df)
        report.add_step("Crime type standardization", 0,
                       f"{stats['unique_before']} → {stats['unique_after']} unique types")

        # Step 3: Add crime category
        if verbose:
            print("  [3/6] Adding crime categories...")
        df = add_crime_category(df)

        # Step 4: Add severity weights
        if verbose:
            print("  [4/6] Adding severity weights...")
        df = add_severity_weight(df)

        # Step 5: Clean demographics
        if verbose:
            print("  [5/6] Cleaning demographics...")
        df, demo_stats = clean_demographics(df)
        report.add_step("Demographic cleaning", 0,
                       f"Cleaned columns: {demo_stats['cleaned_columns']}")

        # Step 6: Clean timestamps
        if verbose:
            print("  [6/6] Parsing timestamps...")
        df, parse_failures = clean_timestamps(df)
        report.add_step("Timestamp parsing", 0,
                       f"{parse_failures:,} parse failures (kept with NaT)")

        # Final: Remove rows with missing essential data
        df, removed = remove_nulls(df, ['latitude', 'longitude', 'crime_type'])
        report.add_step("Remove incomplete rows", removed,
                       "Missing latitude, longitude, or crime_type")

        report.final_rows = len(df)

        if verbose:
            print(f"Cleaning complete: {report.final_rows:,} rows remaining")

        return df, report

    def clean_for_database(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, CleaningReport]:
        """
        Clean data specifically for database insertion.

        Returns DataFrame with columns matching the database schema.
        """
        df, report = self.clean(df)

        # Select and rename columns to match database schema
        db_columns = {
            'latitude': 'latitude',
            'longitude': 'longitude',
            'crime_type': 'crime_type',
            'crime_category': 'crime_category',
            'severity_weight': 'severity_weight',
            'occurred_at': 'occurred_at',
            'hour_of_day': 'hour_of_day',
            'day_of_week': 'day_of_week',
            'borough': 'borough',
            'vic_age_group': 'vic_age_group',
            'vic_race': 'vic_race',
            'vic_sex': 'vic_sex',
        }

        # Keep only columns that exist
        keep_cols = [c for c in db_columns.keys() if c in df.columns]
        df = df[keep_cols].copy()

        return df, report


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def load_and_clean_csv(filepath: str,
                       verbose: bool = True) -> Tuple[pd.DataFrame, CleaningReport]:
    """
    Load a CSV file and run the cleaning pipeline.

    Args:
        filepath: Path to CSV file
        verbose: Print progress messages

    Returns:
        Tuple of (cleaned DataFrame, CleaningReport)
    """
    if verbose:
        print(f"Loading {filepath}...")

    # Optimized dtypes for memory efficiency
    dtypes = {
        'cmplnt_num': 'string',
        'rpt_dt': 'string',
        'pd_desc': 'category',
        'ofns_desc': 'category',
        'boro_nm': 'category',
        'prem_typ_desc': 'category'
    }

    df = pd.read_csv(filepath, dtype=dtypes, low_memory=False)

    if verbose:
        print(f"Loaded {len(df):,} rows")

    cleaner = DataCleaner()
    return cleaner.clean(df, verbose=verbose)


# =============================================================================
# CLI INTERFACE
# =============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python data_cleaner.py <csv_file>")
        print("Example: python data_cleaner.py NYPD_Complaint_Data_YTD.csv")
        sys.exit(1)

    filepath = sys.argv[1]
    clean_df, report = load_and_clean_csv(filepath)

    print("\n" + report.summary())

    # Show sample of cleaned data
    print("\nSample of cleaned data:")
    print(clean_df.head())

