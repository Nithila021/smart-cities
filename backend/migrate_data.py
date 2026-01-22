"""
Data Migration Script: CSV to PostgreSQL + PostGIS

Migrates NYPD crime data from CSV files to the PostgreSQL database.
Uses data_cleaner.py for all data cleaning operations.
"""

import os
import sys
import json
import pandas as pd
from typing import Optional

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from db_config import get_db_cursor, test_connection
from data_cleaner import DataCleaner, load_and_clean_csv


def migrate_to_database(df: pd.DataFrame, batch_size: int = 1000) -> int:
    """Migrate cleaned DataFrame to PostgreSQL database."""
    print(f"Migrating {len(df):,} records to database...")

    insert_query = """
        INSERT INTO crime_incidents
        (location, crime_type, crime_category, severity_weight, occurred_at,
         day_of_week, hour_of_day, victim_demographics, borough)
        VALUES (
            ST_SetSRID(ST_MakePoint(%s, %s), 4326)::geography,
            %s, %s, %s, %s, %s, %s, %s::jsonb, %s
        )
    """

    total_inserted = 0
    errors = 0

    with get_db_cursor(dict_cursor=False) as cur:
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size]
            batch_data = []

            for _, row in batch.iterrows():
                try:
                    # Build victim demographics JSON from separate columns
                    vic_demo = json.dumps({
                        'age_group': str(row.get('vic_age_group', '')) if pd.notna(row.get('vic_age_group')) else '',
                        'sex': str(row.get('vic_sex', '')) if pd.notna(row.get('vic_sex')) else '',
                        'race': str(row.get('vic_race', '')) if pd.notna(row.get('vic_race')) else ''
                    })

                    occurred = row.get('occurred_at')
                    if pd.isna(occurred):
                        occurred = None

                    batch_data.append((
                        float(row['longitude']),
                        float(row['latitude']),
                        str(row['crime_type'])[:100],
                        str(row['crime_category'])[:50],
                        float(row['severity_weight']),
                        occurred,
                        int(row['day_of_week']),
                        int(row['hour_of_day']),
                        vic_demo,
                        str(row.get('borough', 'UNKNOWN'))[:50]
                    ))
                except Exception as e:
                    errors += 1
                    continue

            # Execute batch insert
            from psycopg2.extras import execute_batch
            execute_batch(cur, insert_query, batch_data)
            total_inserted += len(batch_data)

            if (i + batch_size) % 10000 == 0:
                print(f"  Inserted {total_inserted:,} records...")

    print(f"Migration complete: {total_inserted:,} records inserted, {errors} errors")
    return total_inserted


def get_record_count() -> int:
    """Get current record count in database."""
    with get_db_cursor() as cur:
        cur.execute("SELECT COUNT(*) as count FROM crime_incidents")
        result = cur.fetchone()
        return result['count'] if result else 0


def main():
    """Main migration function."""
    print("=" * 60)
    print("Smart Cities Crime Data Migration")
    print("=" * 60)

    # Test database connection
    if not test_connection():
        print("ERROR: Cannot connect to database. Exiting.")
        sys.exit(1)

    # Check current record count
    current_count = get_record_count()
    print(f"Current records in database: {current_count}")

    if current_count > 0:
        response = input("Database already has data. Clear and reimport? (y/N): ")
        if response.lower() == 'y':
            with get_db_cursor() as cur:
                cur.execute("TRUNCATE crime_incidents RESTART IDENTITY")
            print("Cleared existing data.")
        else:
            print("Keeping existing data. Appending new records.")

    # Find CSV file
    csv_files = [
        'NYPD_Complaint_Data_YTD.csv',
        'NYPD_Complaint_Data_Historic.csv',
        '../NYPD_Complaint_Data_YTD.csv',
        '../NYPD_Complaint_Data_Historic.csv',
    ]

    csv_path = None
    for f in csv_files:
        if os.path.exists(f):
            csv_path = f
            break

    if not csv_path:
        print("ERROR: No NYPD crime data CSV found.")
        print("Please ensure one of these files exists:")
        for f in csv_files:
            print(f"  - {f}")
        sys.exit(1)

    # Load and clean data using the data_cleaner module
    df, report = load_and_clean_csv(csv_path)
    print("\n" + report.summary() + "\n")

    # Migrate to database
    migrate_to_database(df)

    # Verify
    final_count = get_record_count()
    print(f"\nFinal record count: {final_count}")
    print("Migration complete!")


if __name__ == "__main__":
    main()

