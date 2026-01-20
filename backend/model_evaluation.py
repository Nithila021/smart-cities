"""
Model Evaluation and Accuracy Testing Module
Tests the accuracy and performance of ML models used in the safety analysis system
"""

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from data_init import cached_data, initialize_data
from models import (
    initialize_dbscan_clusters, 
    initialize_victim_demographic_zones,
    initialize_crime_density_zones
)
import warnings
warnings.filterwarnings('ignore')


def evaluate_clustering_quality():
    """
    Evaluate the quality of clustering algorithms using multiple metrics
    """
    print("\n" + "="*60)
    print("CLUSTERING QUALITY EVALUATION")
    print("="*60)
    
    df = cached_data.get('df')
    if df is None:
        print("Error: Data not initialized")
        return None
    
    results = {}
    
    # 1. DBSCAN Clustering Evaluation
    print("\n1. DBSCAN Clustering Metrics:")
    print("-" * 40)
    dbscan_data = cached_data.get('dbscan_clusters')
    if dbscan_data and 'sample_points' in dbscan_data:
        sample_points = dbscan_data['sample_points']
        valid_points = sample_points[sample_points['dbscan_cluster'] != -1]
        
        if len(valid_points) > 0:
            coords = valid_points[['latitude', 'longitude']].values
            labels = valid_points['dbscan_cluster'].values
            
            # Calculate clustering metrics
            silhouette = silhouette_score(coords, labels)
            davies_bouldin = davies_bouldin_score(coords, labels)
            calinski = calinski_harabasz_score(coords, labels)
            
            n_clusters = len(np.unique(labels))
            n_noise = len(sample_points[sample_points['dbscan_cluster'] == -1])
            
            results['dbscan'] = {
                'n_clusters': n_clusters,
                'n_noise_points': n_noise,
                'silhouette_score': silhouette,
                'davies_bouldin_score': davies_bouldin,
                'calinski_harabasz_score': calinski
            }
            
            print(f"Number of Clusters: {n_clusters}")
            print(f"Noise Points: {n_noise}")
            print(f"Silhouette Score: {silhouette:.4f} (Range: -1 to 1, higher is better)")
            print(f"Davies-Bouldin Score: {davies_bouldin:.4f} (Lower is better)")
            print(f"Calinski-Harabasz Score: {calinski:.2f} (Higher is better)")
            
            # Interpretation
            if silhouette > 0.5:
                print("✓ DBSCAN shows GOOD cluster separation")
            elif silhouette > 0.25:
                print("⚠ DBSCAN shows MODERATE cluster separation")
            else:
                print("✗ DBSCAN shows WEAK cluster separation")
    
    # 2. Demographic Zone Clustering Evaluation
    print("\n2. Demographic Zone Clustering Metrics:")
    print("-" * 40)
    demo_data = cached_data.get('victim_demographic_zones')
    if demo_data and 'zones' in demo_data:
        zones = demo_data['zones']
        n_zones = len(zones)
        
        # Calculate average concentration scores
        avg_concentrations = []
        for zone_id, profile in zones.items():
            if 'concentration_scores' in profile:
                concentrations = [v['concentration'] for v in profile['concentration_scores'].values()]
                avg_concentrations.append(np.mean(concentrations))
        
        avg_concentration = np.mean(avg_concentrations) if avg_concentrations else 0
        
        results['demographic'] = {
            'n_zones': n_zones,
            'avg_concentration': avg_concentration
        }
        
        print(f"Number of Demographic Zones: {n_zones}")
        print(f"Average Demographic Concentration: {avg_concentration:.2f}%")
        
        if avg_concentration > 60:
            print("✓ Demographic zones show STRONG characteristic patterns")
        elif avg_concentration > 40:
            print("⚠ Demographic zones show MODERATE characteristic patterns")
        else:
            print("✗ Demographic zones show WEAK characteristic patterns")
    
    # 3. Crime Density Classification Evaluation
    print("\n3. Crime Density Classification:")
    print("-" * 40)
    density_data = cached_data.get('crime_density_zones')
    if density_data and 'grid' in density_data:
        grid = density_data['grid']
        classifications = grid['classification_grid'].flatten()
        
        unique, counts = np.unique(classifications, return_counts=True)
        total = len(classifications)
        
        results['density'] = {
            'total_grid_points': total,
            'distribution': dict(zip(unique, counts))
        }
        
        print(f"Total Grid Points: {total}")
        print("Crime Density Distribution:")
        for cls, count in zip(unique, counts):
            percentage = (count / total) * 100
            print(f"  {cls}: {count} points ({percentage:.1f}%)")
        
        # Check for balanced distribution
        if max(counts) / min(counts) < 3:
            print("✓ Crime density shows BALANCED distribution across zones")
        else:
            print("⚠ Crime density shows IMBALANCED distribution")
    
    return results


def evaluate_prediction_accuracy():
    """
    Evaluate prediction accuracy using cross-validation on zone predictions
    """
    print("\n" + "="*60)
    print("PREDICTION ACCURACY EVALUATION")
    print("="*60)
    
    df = cached_data.get('df')
    if df is None:
        print("Error: Data not initialized")
        return None
    
    results = {}
    
    # Test KMeans zone prediction accuracy
    print("\n1. Crime Zone Classification Accuracy:")
    print("-" * 40)
    
    crime_clusters = cached_data.get('crime_clusters')
    if crime_clusters:
        # Sample test data
        sample_size = min(5000, len(df))
        test_df = df.sample(sample_size, random_state=123)
        
        coords = test_df[['latitude', 'longitude']].values
        predicted_zones = crime_clusters.predict(coords)
        
        # Calculate zone distribution
        unique_zones, counts = np.unique(predicted_zones, return_counts=True)
        
        results['zone_prediction'] = {
            'n_test_samples': sample_size,
            'n_zones_predicted': len(unique_zones),
            'zone_distribution': dict(zip(unique_zones.tolist(), counts.tolist()))
        }
        
        print(f"Test Samples: {sample_size}")
        print(f"Unique Zones Predicted: {len(unique_zones)}")
        print(f"Average Samples per Zone: {sample_size / len(unique_zones):.1f}")
        
        # Check coverage
        if len(unique_zones) >= 8:
            print("✓ Model predicts across MULTIPLE zones (good coverage)")
        else:
            print("⚠ Model predicts LIMITED zones (may need retraining)")
    
    # Test safety score consistency
    print("\n2. Safety Score Consistency:")
    print("-" * 40)
    
    zone_safety_scores = cached_data.get('zone_safety_scores', {})
    if zone_safety_scores:
        scores = list(zone_safety_scores.values())
        
        results['safety_scores'] = {
            'mean': np.mean(scores),
            'std': np.std(scores),
            'min': np.min(scores),
            'max': np.max(scores),
            'range': np.max(scores) - np.min(scores)
        }
        
        print(f"Mean Safety Score: {np.mean(scores):.2f}")
        print(f"Std Deviation: {np.std(scores):.2f}")
        print(f"Score Range: {np.min(scores):.2f} - {np.max(scores):.2f}")
        
        if np.std(scores) > 10:
            print("✓ Safety scores show GOOD variance (distinguishes zones)")
        else:
            print("⚠ Safety scores show LOW variance (limited differentiation)")
    
    return results


def generate_evaluation_report():
    """
    Generate a comprehensive evaluation report
    """
    print("\n" + "="*60)
    print("SMART CITIES SAFETY ANALYSIS - MODEL EVALUATION REPORT")
    print("="*60)
    
    # Initialize data if not already done
    if cached_data.get('df') is None:
        print("\nInitializing data and models...")
        initialize_data()
    
    # Run evaluations
    clustering_results = evaluate_clustering_quality()
    prediction_results = evaluate_prediction_accuracy()
    
    # Overall Assessment
    print("\n" + "="*60)
    print("OVERALL MODEL ASSESSMENT")
    print("="*60)
    
    scores = []
    
    # DBSCAN assessment
    if clustering_results and 'dbscan' in clustering_results:
        silhouette = clustering_results['dbscan']['silhouette_score']
        if silhouette > 0.5:
            scores.append(('DBSCAN Clustering', 'EXCELLENT', silhouette))
        elif silhouette > 0.25:
            scores.append(('DBSCAN Clustering', 'GOOD', silhouette))
        else:
            scores.append(('DBSCAN Clustering', 'FAIR', silhouette))
    
    # Demographic zones assessment
    if clustering_results and 'demographic' in clustering_results:
        concentration = clustering_results['demographic']['avg_concentration']
        if concentration > 60:
            scores.append(('Demographic Zones', 'EXCELLENT', concentration/100))
        elif concentration > 40:
            scores.append(('Demographic Zones', 'GOOD', concentration/100))
        else:
            scores.append(('Demographic Zones', 'FAIR', concentration/100))
    
    # Print assessment
    print("\nModel Performance Summary:")
    for model, rating, score in scores:
        print(f"  {model}: {rating} (Score: {score:.3f})")
    
    # Calculate overall score
    if scores:
        overall_score = np.mean([s[2] for s in scores])
        print(f"\nOverall System Score: {overall_score:.3f}")
        
        if overall_score > 0.6:
            print("✓ System shows STRONG performance across all models")
        elif overall_score > 0.4:
            print("⚠ System shows ACCEPTABLE performance")
        else:
            print("✗ System may need model improvements")
    
    print("\n" + "="*60)
    print("Evaluation Complete!")
    print("="*60 + "\n")
    
    return {
        'clustering': clustering_results,
        'prediction': prediction_results,
        'overall_score': overall_score if scores else None
    }


if __name__ == '__main__':
    # Run full evaluation
    report = generate_evaluation_report()
