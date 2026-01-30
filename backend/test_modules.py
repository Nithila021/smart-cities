#!/usr/bin/env python3
"""
Comprehensive module test for Smart Cities backend.
Tests all modules can be imported and key symbols exist.
"""

import sys

print("=" * 70)
print("COMPREHENSIVE MODULE TEST")
print("=" * 70)
print()

modules_to_test = [
    ('data_cleaner', ['DataCleaner', 'SEVERITY_WEIGHTS', 'CITY_BOUNDS_CONFIG']),
    ('demographic_parser', ['parse_demographic_group']),
    ('geo_utils', ['haversine', 'get_coordinates']),
    ('models', ['initialize_dbscan_clusters', 'initialize_crime_density_zones']),
    ('utils', []),  # Just test import
    ('data_init', ['initialize_data']),
    ('analysis', ['analyze_safety']),
    ('app1', ['app']),
]

results = []

for module_name, symbols in modules_to_test:
    try:
        print(f"Testing {module_name}...", end=" ")
        module = __import__(module_name)
        
        # Check if specific symbols exist
        for symbol in symbols:
            if not hasattr(module, symbol):
                raise AttributeError(f"Missing symbol: {symbol}")
        
        print(f"OK ({len(symbols)} symbols)")
        results.append((module_name, True, None))
    except Exception as e:
        print(f"FAILED")
        print(f"  Error: {str(e)}")
        results.append((module_name, False, str(e)))

print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)

passed = sum(1 for _, success, _ in results if success)
failed = sum(1 for _, success, _ in results if not success)

print(f"Passed: {passed}/{len(results)}")
print(f"Failed: {failed}/{len(results)}")

if failed > 0:
    print()
    print("Failed modules:")
    for module_name, success, error in results:
        if not success:
            print(f"  - {module_name}: {error}")
    sys.exit(1)
else:
    print()
    print("All modules working!")
    sys.exit(0)

