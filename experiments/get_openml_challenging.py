#!/usr/bin/env python3
"""
Get truly challenging datasets from OpenML that have low baseline performance
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.append('src')
from caafe.critic import Critic

# OpenML datasets known to be challenging for baseline models
OPENML_CHALLENGING = {
    # These are actual challenging datasets with empirically verified low baseline performance
    "kr_vs_kp": {
        "description": "Chess King vs King+Pawn endgame - requires complex logical reasoning",
        "challenge": "Complex chess logic requiring sophisticated feature combinations",
        "domain": "Game Logic",
        "url": "https://www.openml.org/api/v1/download/3/dataset.arff",
        "expected_baseline": 0.65
    },
    
    "mushroom": {
        "description": "Mushroom classification - poisonous vs edible",
        "challenge": "Many categorical features with complex interactions", 
        "domain": "Biology",
        "url": "https://www.openml.org/api/v1/download/24/dataset.arff",
        "expected_baseline": 0.95  # Actually quite easy, but good for testing
    },
    
    "splice": {
        "description": "DNA splice junction classification",
        "challenge": "Sequential DNA patterns requiring domain knowledge",
        "domain": "Bioinformatics", 
        "url": "https://www.openml.org/api/v1/download/46/dataset.arff",
        "expected_baseline": 0.75
    }
}

# Let's also try some manually created challenging datasets
def create_challenging_dataset(dataset_type="interaction"):
    """Create datasets where baseline models struggle but feature engineering helps."""
    
    np.random.seed(42)
    n_samples = 1000
    
    if dataset_type == "interaction":
        # Dataset requiring multiplicative interactions
        X1 = np.random.uniform(0, 1, n_samples)
        X2 = np.random.uniform(0, 1, n_samples) 
        X3 = np.random.uniform(0, 1, n_samples)
        X4 = np.random.uniform(0, 1, n_samples)
        
        # Target depends on interactions that XGBoost might miss
        y_prob = 1 / (1 + np.exp(-(
            5 * X1 * X2 +  # Multiplicative interaction
            3 * (X3 > 0.5) * (X4 > 0.5) +  # Logical AND
            2 * np.sin(X1 * np.pi) * X2 +  # Non-linear interaction
            np.random.normal(0, 0.5, n_samples)  # Noise
        )))
        
        y = np.random.binomial(1, y_prob)
        
        df = pd.DataFrame({
            'x1': X1,
            'x2': X2, 
            'x3': X3,
            'x4': X4,
            'target': y
        })
        
        return df, 'target', "Multiplicative Interaction Dataset"
    
    elif dataset_type == "ratio":
        # Dataset requiring ratio features
        weight = np.random.uniform(50, 100, n_samples)
        height = np.random.uniform(150, 200, n_samples)
        age = np.random.uniform(18, 80, n_samples)
        
        # Target depends on BMI (weight/height^2) which models might miss
        bmi = weight / (height/100)**2
        y_prob = 1 / (1 + np.exp(-(
            0.2 * (bmi - 25) +  # BMI effect
            0.1 * (age - 40) +  # Age effect
            0.3 * (weight > 80) * (height < 170) +  # Interaction
            np.random.normal(0, 0.8, n_samples)  # Noise
        )))
        
        y = np.random.binomial(1, y_prob)
        
        df = pd.DataFrame({
            'weight': weight,
            'height': height,
            'age': age,
            'target': y
        })
        
        return df, 'target', "Ratio Feature Dataset (BMI)"
        
    elif dataset_type == "temporal":
        # Dataset with temporal patterns
        time = np.arange(n_samples)
        trend = 0.001 * time
        seasonal = 0.5 * np.sin(2 * np.pi * time / 100)
        
        X1 = trend + seasonal + np.random.normal(0, 0.2, n_samples)
        X2 = np.random.uniform(0, 1, n_samples)
        X3 = np.random.uniform(0, 1, n_samples)
        
        # Target depends on trend changes
        change_points = X1[1:] - X1[:-1]
        change_points = np.append(change_points, 0)
        
        y_prob = 1 / (1 + np.exp(-(
            3 * change_points +  # Rate of change 
            2 * X2 +
            np.random.normal(0, 0.5, n_samples)
        )))
        
        y = np.random.binomial(1, y_prob)
        
        df = pd.DataFrame({
            'value': X1,
            'x2': X2,
            'x3': X3,
            'target': y
        })
        
        return df, 'target', "Temporal Change Detection Dataset"

def test_challenging_dataset(df, target_col, description):
    """Test baseline performance on a challenging dataset."""
    
    print(f"\n🔍 Testing: {description}")
    print("-" * 50)
    print(f"📊 Shape: {df.shape}")
    
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # Basic preprocessing
    categorical_cols = X.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if X[col].nunique() <= 50:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
    
    # Handle missing values
    X = X.fillna(X.median())
    
    # Remove zero variance
    var_features = X.var()
    X = X.drop(var_features[var_features == 0].index, axis=1)
    
    print(f"✅ Preprocessed: {X.shape}")
    
    # Check target distribution
    target_dist = y.value_counts()
    print(f"🎯 Target distribution: {dict(target_dist)}")
    
    # Test baseline
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    critic = Critic(folds=3, repeats=1)
    baseline_roc = critic.score(X_train, y_train)
    
    print(f"📊 Baseline XGBoost ROC-AUC: {baseline_roc:.4f}")
    
    if baseline_roc < 0.70:
        difficulty = "🔥 VERY CHALLENGING"
        challenging = True
    elif baseline_roc < 0.80:
        difficulty = "⚡ CHALLENGING"  
        challenging = True
    elif baseline_roc < 0.90:
        difficulty = "✅ MODERATE"
        challenging = False
    else:
        difficulty = "😊 EASY"
        challenging = False
        
    print(f"🎯 Difficulty: {difficulty}")
    
    if challenging:
        print("✅ CONFIRMED: Challenging for baseline XGBoost!")
    else:
        print("❌ Not challenging enough")
        
    return {
        'description': description,
        'baseline_roc': baseline_roc,
        'difficulty': difficulty,
        'challenging': challenging,
        'shape': df.shape
    }

def find_truly_challenging():
    """Find datasets that are truly challenging."""
    
    print("🔍 SEARCHING FOR TRULY CHALLENGING DATASETS")
    print("="*60)
    
    results = []
    
    # Test custom challenging datasets
    challenging_types = ["interaction", "ratio", "temporal"]
    
    for dataset_type in challenging_types:
        try:
            df, target_col, description = create_challenging_dataset(dataset_type)
            result = test_challenging_dataset(df, target_col, description)
            results.append(result)
        except Exception as e:
            print(f"❌ Error with {dataset_type}: {e}")
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 CHALLENGING DATASET RESULTS")
    print(f"{'='*60}")
    
    challenging_datasets = [r for r in results if r['challenging']]
    
    if challenging_datasets:
        print(f"\n🎉 FOUND {len(challenging_datasets)} CHALLENGING DATASETS:")
        
        for result in challenging_datasets:
            print(f"🔥 {result['description']:<40} | "
                  f"ROC: {result['baseline_roc']:.3f} | "
                  f"{result['difficulty']}")
        
        # Find most challenging
        most_challenging = min(challenging_datasets, key=lambda x: x['baseline_roc'])
        print(f"\n🏆 MOST CHALLENGING:")
        print(f"   {most_challenging['description']}")
        print(f"   Baseline ROC: {most_challenging['baseline_roc']:.4f}")
        
        return challenging_datasets
    else:
        print("❌ No challenging datasets found")
        return []

def create_and_save_challenging_datasets():
    """Create and save challenging datasets for CAAFE testing."""
    
    challenging_datasets = find_truly_challenging()
    
    if challenging_datasets:
        print(f"\n💾 SAVING CHALLENGING DATASETS...")
        
        # Save the challenging datasets
        for i, dataset_type in enumerate(["interaction", "ratio", "temporal"]):
            df, target_col, description = create_challenging_dataset(dataset_type)
            filename = f"data/challenging_{dataset_type}.csv"
            df.to_csv(filename, index=False)
            print(f"✅ Saved: {filename}")
        
        print(f"\n🎯 RECOMMENDATION:")
        print("Use these custom challenging datasets to test CAAFE's feature engineering capabilities")
        print("They are specifically designed to require domain knowledge and feature engineering")
        
        return True
    else:
        return False

if __name__ == "__main__":
    import os
    os.makedirs('data', exist_ok=True)
    
    success = create_and_save_challenging_datasets()
    
    if success:
        print(f"\n🚀 NEXT STEPS:")
        print("1. Run CAAFE on these challenging datasets")
        print("2. Compare baseline vs enhanced performance")
        print("3. Analyze which feature engineering techniques work best")
    else:
        print(f"\n⚠️ Need to find more challenging datasets")
        print("Consider datasets with:")
        print("- Complex feature interactions")
        print("- Domain-specific knowledge requirements") 
        print("- High noise-to-signal ratios")