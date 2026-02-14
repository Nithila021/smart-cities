"""
Data Cleaning Pipeline for Smart Cities Crime Safety Analysis (Polars Edition).

This module provides reusable, testable cleaning functions for NYPD crime data using Polars.
"""

import polars as pl
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import os


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

def auto_detect_bounds(df: pl.DataFrame,
                       lat_col: str = 'latitude',
                       lon_col: str = 'longitude',
                       outlier_percentile: float = 1.0) -> Dict:
    """
    Automatically detect geographic bounds from data with Polars
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

    # Cast to float, drop nulls
    valid_coords = df.select([
        pl.col(lat_col).cast(pl.Float64, strict=False),
        pl.col(lon_col).cast(pl.Float64, strict=False)
    ]).drop_nulls()

    stats = valid_coords.select([
        pl.col(lat_col).quantile(lower).alias('lat_min'),
        pl.col(lat_col).quantile(upper).alias('lat_max'),
        pl.col(lon_col).quantile(lower).alias('lon_min'),
        pl.col(lon_col).quantile(upper).alias('lon_max')
    ]).to_dict(as_series=False)
    
    return {k: v[0] for k, v in stats.items()}


def get_bounds(df: Optional[pl.DataFrame] = None, city: Optional[str] = None) -> Dict:
    """Get geographic bounds."""
    if city and city in CITY_BOUNDS_CONFIG:
        config = CITY_BOUNDS_CONFIG[city]
        return {k: v for k, v in config.items() if k != 'name'}

    env_city = os.getenv('ACTIVE_CITY', '').lower()
    if env_city and env_city in CITY_BOUNDS_CONFIG:
        config = CITY_BOUNDS_CONFIG[env_city]
        return {k: v for k, v in config.items() if k != 'name'}

    if df is not None and df.height > 0:
        return auto_detect_bounds(df)

    config = CITY_BOUNDS_CONFIG['nyc']
    return {k: v for k, v in config.items() if k != 'name'}

# Mappings (Same as before)
CRIME_TYPE_MAPPINGS = {
    r'ASSAULT.*3.*': 'ASSAULT_3',
    r'HARRASSMENT': 'HARASSMENT',
    r'DRIVING WHILE INTOXICATED': 'DWI',
    r'CRIMINAL MISCHIEF.*': 'CRIMINAL_MISCHIEF',
    r'SEX CRIMES': 'SEX_CRIMES',
    r'PETIT LARCENY': 'PETIT_LARCENY',
    r'GRAND LARCENY.*': 'GRAND_LARCENY',
}

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
# PURE CLEANING FUNCTIONS (POLARS)
# =============================================================================

def clean_coordinates(df: pl.DataFrame,
                      lat_col: str = 'latitude',
                      lon_col: str = 'longitude',
                      bounds: Optional[Dict] = None,
                      city: Optional[str] = None) -> Tuple[pl.DataFrame, int, Dict]:
    """Validate and filter coordinates."""
    initial_count = df.height

    # Cast coordinates to Float64
    df = df.with_columns([
        pl.col(lat_col).cast(pl.Float64, strict=False),
        pl.col(lon_col).cast(pl.Float64, strict=False)
    ])

    if bounds is None:
        bounds = get_bounds(df=df, city=city)

    df_clean = df.filter(
        (pl.col(lat_col) >= bounds['lat_min']) &
        (pl.col(lat_col) <= bounds['lat_max']) &
        (pl.col(lon_col) >= bounds['lon_min']) &
        (pl.col(lon_col) <= bounds['lon_max'])
    )

    removed = initial_count - df_clean.height
    return df_clean, removed, bounds


def clean_crime_types(df: pl.DataFrame,
                      crime_col: str = 'crime_type') -> Tuple[pl.DataFrame, Dict]:
    """Standardize crime types."""
    if crime_col not in df.columns:
        return df, {'mappings_applied': 0, 'unique_before': 0, 'unique_after': 0}

    stats = {'unique_before': df[crime_col].n_unique()}
    
    # Upper case and strip
    df = df.with_columns(
        pl.col(crime_col).cast(pl.Utf8).str.to_uppercase().str.strip_chars()
    )

    # Apply Regex Mappings
    # Construct a big `when-then-otherwise` expression or apply sequentially
    expr = pl.col(crime_col)
    for pattern, replacement in CRIME_TYPE_MAPPINGS.items():
        expr = expr.map_elements(lambda x: replacement if x and pattern in x else x, return_dtype=pl.Utf8) 
        # Note: Polars regex replacement is more complex for "contains" logic in bulk without specific regex features
        # For simplicity and speed in Polars, direct string matching is preferred, but for regex patterns we use:
        # pl.col(col).str.replace_all(pattern, replacement) if it's a replacement
        # But here we are mapping *if contains*.
        # We can use `pl.when(pl.col(crime_col).str.contains(pattern)).then(pl.lit(replacement)).otherwise(...)` chaining.
    
    # Efficient Chaining for standardizing
    clean_expr = pl.col(crime_col)
    for pattern, replacement in CRIME_TYPE_MAPPINGS.items():
        clean_expr = pl.when(clean_expr.str.contains(pattern)).then(pl.lit(replacement)).otherwise(clean_expr)

    df = df.with_columns(clean_expr.alias(crime_col))
    
    stats['unique_after'] = df[crime_col].n_unique()
    stats['mappings_applied'] = 0 # specific count hard to get efficiently without extra pass
    
    return df, stats


def add_crime_category(df: pl.DataFrame,
                       crime_col: str = 'crime_type',
                       category_col: str = 'crime_category') -> pl.DataFrame:
    """Add crime category."""
    # Build expression chain
    expr = pl.lit('other')
    
    # We loop in reverse or standard order (doesn't matter much if maps are disjoint)
    # But Polars `when` chains need `otherwise` at the end
    for pattern, category in CRIME_CATEGORY_MAP.items():
        # Check if pattern is in crime_type
        expr = pl.when(pl.col(crime_col).str.contains(pattern)).then(pl.lit(category)).otherwise(expr)
        
    return df.with_columns(expr.alias(category_col))


def add_severity_weight(df: pl.DataFrame,
                        crime_col: str = 'crime_type',
                        weight_col: str = 'severity_weight') -> pl.DataFrame:
    """Add severity weight."""
    expr = pl.lit(0.5) # Default
    
    for pattern, weight in SEVERITY_WEIGHTS.items():
         expr = pl.when(pl.col(crime_col).str.contains(pattern)).then(pl.lit(weight)).otherwise(expr)
         
    return df.with_columns(expr.alias(weight_col))


def clean_demographics(df: pl.DataFrame) -> Tuple[pl.DataFrame, Dict]:
    """Clean demographics."""
    stats = {'cleaned_columns': []}
    
    cols = df.columns
    
    exprs = []
    
    if 'vic_age_group' in cols:
        exprs.append(
            pl.col('vic_age_group').cast(pl.Utf8).str.to_uppercase().str.strip_chars()
            .replace("NAN", None).replace("UNKNOWN", None).replace("", None)
        )
        stats['cleaned_columns'].append('vic_age_group')
        
    if 'vic_race' in cols:
        exprs.append(
            pl.col('vic_race').cast(pl.Utf8).str.to_uppercase().str.strip_chars()
            .replace("UNKNOWN", "UNKNOWN") # Ensure existing UNKNOWN stays
        )
        # We can also do the partial match for unknown if needed, but strict replace is safer for now
        stats['cleaned_columns'].append('vic_race')

    if 'vic_sex' in cols:
        exprs.append(
            pl.col('vic_sex').cast(pl.Utf8).str.strip_chars().str.slice(0, 1).str.to_uppercase()
            .replace("N", None).replace("U", None).replace("", None)
        )
        stats['cleaned_columns'].append('vic_sex')
        
    if exprs:
        df = df.with_columns(exprs)
        
    return df, stats


def clean_timestamps(df: pl.DataFrame,
                     date_col: str = 'cmplnt_fr_dt',
                     output_col: str = 'occurred_at') -> Tuple[pl.DataFrame, int]:
    """Parse timestamps."""
    if date_col not in df.columns:
        return df, 0

    # Polars robust date parsing
    # Typically MM/DD/YYYY in NYPD data
    # We use `str.to_date` or `strptime`
    # We'll try a common format. If raw data has mixed formats, this is tricky.
    # Assuming standard NYPD format: '12/31/2023'
    
    try:
        df = df.with_columns(
            pl.col(date_col).str.strptime(pl.Date, "%m/%d/%Y", strict=False).alias("parsed_date")
        )
    except Exception:
         # Fallback or different format try
         df = df.with_columns(pl.lit(None).cast(pl.Date).alias("parsed_date"))

    # If simple date, cast to datetime? Or just keep date. 
    # Original used to_datetime which gives Datetime[ns]. 
    # Let's make `occurred_at` a Datetime
    
    df = df.with_columns(
        pl.col("parsed_date").cast(pl.Datetime).alias(output_col)
    )
    
    # Calculate failures
    failures = df.filter(pl.col(output_col).is_null()).height

    # Features
    df = df.with_columns([
        pl.col(output_col).dt.hour().fill_null(12).alias("hour_of_day"),
        pl.col(output_col).dt.weekday().fill_null(0).alias("day_of_week") # 1-7 in polars usually? no 0-6 or 1-7 depending.
        # Polars .dt.weekday() is 1(Mon)-7(Sun). Pandas dayofweek is 0-6.
        # Let's adjust to match pandas 0-6 for compat if needed, or just warn.
        # Pandas: 0 is Monday. Polars: 1 is Monday.
        # So Polars - 1 = Pandas
    ])
    
    df = df.with_columns(
        (pl.col("day_of_week") - 1).alias("day_of_week")
    )
    
    return df.drop("parsed_date"), failures


def remove_nulls(df: pl.DataFrame, required_cols: List[str]) -> Tuple[pl.DataFrame, int]:
    """Remove rows with nulls."""
    initial = df.height
    existing = [c for c in required_cols if c in df.columns]
    df_clean = df.drop_nulls(subset=existing)
    return df_clean, initial - df_clean.height


# =============================================================================
# MAIN CLEANER CLASS
# =============================================================================

class DataCleaner:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}

    def normalize_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        mapping = {
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
        
        rename_map = {}
        for old, new in mapping.items():
            if old in df.columns and new not in df.columns:
                rename_map[old] = new
                
        if rename_map:
            df = df.rename(rename_map)
            
        return df

    def clean(self, df: pl.DataFrame, verbose: bool = True) -> Tuple[pl.DataFrame, CleaningReport]:
        report = CleaningReport(initial_rows=df.height)
        if verbose: print(f"Starting cleaning with {df.height} rows...")

        df = self.normalize_columns(df)
        
        if verbose: print("  [1/6] Cleaning coordinates...")
        df, removed, bounds = clean_coordinates(df)
        report.add_step("Coordinate validation", removed, str(bounds))
        
        if verbose: print("  [2/6] Standardizing crime types...")
        df, stats = clean_crime_types(df)
        report.add_step("Crime standardization", 0, f"{stats['unique_before']} -> {stats['unique_after']}")
        
        if verbose: print("  [3/6] Adding categories...")
        df = add_crime_category(df)
        
        if verbose: print("  [4/6] Adding weights...")
        df = add_severity_weight(df)
        
        if verbose: print("  [5/6] Cleaning demographics...")
        df, _ = clean_demographics(df)
        
        if verbose: print("  [6/6] Parsing timestamps...")
        df, fails = clean_timestamps(df)
        report.add_step("Timestamp parsing", 0, f"{fails} failures")
        
        df, removed = remove_nulls(df, ['latitude', 'longitude', 'crime_type'])
        report.add_step("Remove incomplete", removed)
        
        report.final_rows = df.height
        if verbose: print(f"Complete: {df.height} rows")
        
        return df, report

def load_and_clean_csv(filepath: str, verbose: bool = True) -> Tuple[pl.DataFrame, CleaningReport]:
    if verbose: print(f"Loading {filepath} using Polars...")
    
    # Polars scan_csv is lazy, read_csv is eager.
    # infer_schema_length=10000 to be safe
    df = pl.read_csv(filepath, infer_schema_length=10000, ignore_errors=True, null_values=["(null)", "NULL", ""])
    
    cleaner = DataCleaner()
    return cleaner.clean(df, verbose=verbose)
