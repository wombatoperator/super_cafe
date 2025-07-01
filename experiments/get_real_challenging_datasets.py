#!/usr/bin/env python3
"""
Download real challenging datasets that are known to have poor baseline performance
"""

import pandas as pd
import numpy as np
import urllib.request
import zipfile
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.append('src')
from caafe.critic import Critic

# Real challenging datasets from research and competitions
REAL_CHALLENGING_DATASETS = {
    "bioresponse": {
        "url": "https://www.kaggle.com/c/bioresponse/download/train.csv.zip", 
        "description": "Boehringer Ingelheim Bioresponse - predict biological response to chemical compounds",
        "challenge": "High-dimensional molecular data, many uninformative features, requires chemistry domain knowledge",
        "domain": "Computational Chemistry",
        "format": "zip",
        "target": "Activity",
        "notes": "3751 molecular descriptors, known to be very challenging"
    },
    
    "kdd_appetency": {
        "url": "https://www.kdd.org/cupfiles/KDDCupData/2009/orange_small_train_appetency.labels.txt",
        "data_url": "https://www.kdd.org/cupfiles/KDDCupData/2009/orange_small_train.data.zip",
        "description": "KDD Cup 2009 Appetency - predict customer propensity to buy product",
        "challenge": "Marketing/customer behavior with many sparse categorical features",
        "domain": "Customer Analytics", 
        "format": "zip",
        "target": "appetency",
        "notes": "Known for poor baseline performance, heavy feature engineering needed"
    },
    
    "gisette": {
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/gisette/GISETTE/gisette_train.data",
        "labels_url": "https://archive.ics.uci.edu/ml/machine-learning-databases/gisette/GISETTE/gisette_train.labels",
        "description": "GISETTE dataset - handwritten digit recognition with many irrelevant features",
        "challenge": "5000 features, only few relevant, designed to test feature selection",
        "domain": "Computer Vision/Feature Selection",
        "target": "digit",
        "notes": "Specifically designed to be challenging for feature selection algorithms"
    },
    
    "madelon": {
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/madelon/MADELON/madelon_train.data",
        "labels_url": "https://archive.ics.uci.edu/ml/machine-learning-databases/madelon/MADELON/madelon_train.labels", 
        "description": "MADELON dataset - artificial but difficult binary classification",
        "challenge": "500 features, only 20 informative, many redundant/irrelevant features",
        "domain": "Feature Selection Challenge",
        "target": "class",
        "notes": "Part of NIPS 2003 feature selection challenge, very difficult"
    },
    
    "isolet": {
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/isolet/isolet1+2+3+4.data.Z",
        "description": "ISOLET Spoken Letter Recognition - audio features of spoken letters",
        "challenge": "617 audio features, requires signal processing domain knowledge",
        "domain": "Audio Signal Processing",
        "target": "letter",
        "notes": "Multi-class (26 letters), challenging for audio feature engineering"
    },
    
    "secom": {
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom.data",
        "labels_url": "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom_labels.data",
        "description": "SECOM semiconductor manufacturing - predict pass/fail from manufacturing data",
        "challenge": "Highly imbalanced (6% failure rate), 590 sensor measurements, many missing values",
        "domain": "Manufacturing/Process Control", 
        "target": "pass_fail",
        "notes": "Real semiconductor manufacturing data, very challenging imbalanced classification"
    }
}

def download_dataset(dataset_name, dataset_info):
    """Download a challenging dataset."""
    
    print(f"\n📥 Downloading {dataset_name}...")
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    
    try:
        if dataset_name == "bioresponse":
            print("⚠️ Bioresponse requires Kaggle API - skipping for now")
            return None
            
        elif dataset_name == "kdd_appetency":
            print("⚠️ KDD Cup data requires special handling - skipping for now")
            return None
            
        elif dataset_name == "gisette":
            # Download data and labels
            data_file = f"{data_dir}/gisette_data.txt"
            labels_file = f"{data_dir}/gisette_labels.txt"
            
            urllib.request.urlretrieve(dataset_info['url'], data_file)
            urllib.request.urlretrieve(dataset_info['labels_url'], labels_file)
            
            # Load and combine
            X = pd.read_csv(data_file, sep=' ', header=None)
            X = X.iloc[:, :-1]  # Remove last empty column
            y = pd.read_csv(labels_file, header=None)[0]
            
            df = X.copy()
            df['target'] = y
            
            output_file = f"{data_dir}/{dataset_name}.csv"
            df.to_csv(output_file, index=False)
            
            # Clean up
            os.remove(data_file)
            os.remove(labels_file)
            
            print(f"✅ Downloaded and processed GISETTE: {df.shape}")
            return output_file
            
        elif dataset_name == "madelon":
            # Download data and labels  
            data_file = f"{data_dir}/madelon_data.txt"
            labels_file = f"{data_dir}/madelon_labels.txt"
            
            urllib.request.urlretrieve(dataset_info['url'], data_file)
            urllib.request.urlretrieve(dataset_info['labels_url'], labels_file)
            
            # Load and combine
            X = pd.read_csv(data_file, sep=' ', header=None)
            X = X.iloc[:, :-1]  # Remove last empty column
            y = pd.read_csv(labels_file, header=None)[0]
            
            df = X.copy()
            df['target'] = y
            # Convert -1/1 to 0/1
            df['target'] = (df['target'] == 1).astype(int)
            
            output_file = f"{data_dir}/{dataset_name}.csv"
            df.to_csv(output_file, index=False)
            
            # Clean up
            os.remove(data_file)
            os.remove(labels_file)
            
            print(f"✅ Downloaded and processed MADELON: {df.shape}")
            return output_file
            
        elif dataset_name == "secom":
            # Download data and labels
            data_file = f"{data_dir}/secom_data.txt"
            labels_file = f"{data_dir}/secom_labels.txt"
            
            urllib.request.urlretrieve(dataset_info['url'], data_file)
            urllib.request.urlretrieve(dataset_info['labels_url'], labels_file)
            
            # Load and combine
            X = pd.read_csv(data_file, sep=' ', header=None)
            y = pd.read_csv(labels_file, header=None, sep=' ')[0]
            
            df = X.copy()
            df['target'] = y
            
            output_file = f"{data_dir}/{dataset_name}.csv"
            df.to_csv(output_file, index=False)
            
            # Clean up
            os.remove(data_file)
            os.remove(labels_file)
            
            print(f"✅ Downloaded and processed SECOM: {df.shape}")
            return output_file
            
        elif dataset_name == "isolet":
            print("⚠️ ISOLET is compressed format - skipping for now")
            return None
            
    except Exception as e:
        print(f"❌ Error downloading {dataset_name}: {e}")
        return None

def test_dataset_difficulty(file_path, dataset_name, dataset_info):
    """Test how challenging a dataset actually is."""
    
    print(f"\n🔍 Testing difficulty: {dataset_name.upper()}")
    print("-" * 50)
    
    try:
        # Load dataset
        df = pd.read_csv(file_path)
        print(f"📊 Shape: {df.shape}")
        print(f"📝 Description: {dataset_info['description']}")
        print(f"🏷️ Domain: {dataset_info['domain']}")
        
        # Prepare data
        X = df.drop('target', axis=1)
        y = df['target']
        
        # Handle missing values
        print(f"⚠️ Missing values: {X.isnull().sum().sum()} ({X.isnull().sum().sum() / X.size * 100:.1f}%)")
        X = X.fillna(X.median())
        
        # Remove zero variance features
        var_features = X.var()
        zero_var = (var_features == 0).sum()
        if zero_var > 0:
            print(f"⚠️ Removing {zero_var} zero-variance features")
            X = X.drop(var_features[var_features == 0].index, axis=1)
        
        print(f"✅ Final shape: {X.shape}")
        
        # Check target distribution
        target_dist = y.value_counts()
        print(f"🎯 Target distribution: {dict(target_dist)}")
        
        # Check for class imbalance
        if len(target_dist) == 2:
            minority_pct = min(target_dist.values) / len(y) * 100
            print(f"⚖️ Class imbalance: {minority_pct:.1f}% minority class")
            
            if minority_pct < 10:
                print("🔥 HIGHLY IMBALANCED dataset!")
        
        # Test baseline XGBoost
        if len(X) < 100:
            print("⚠️ Dataset too small for reliable testing")
            return None
            
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, 
            stratify=y if len(np.unique(y)) <= 10 else None
        )
        
        print(f"🔍 Testing XGBoost baseline...")
        critic = Critic(folds=3, repeats=1)
        baseline_roc = critic.score(X_train, y_train)
        
        print(f"📊 Baseline XGBoost ROC-AUC: {baseline_roc:.4f}")
        
        # Determine difficulty
        if baseline_roc < 0.55:
            difficulty = "🔥 EXTREMELY CHALLENGING"
            challenging = True
        elif baseline_roc < 0.65:
            difficulty = "🔥 VERY CHALLENGING"
            challenging = True
        elif baseline_roc < 0.75:
            difficulty = "⚡ CHALLENGING"
            challenging = True
        elif baseline_roc < 0.85:
            difficulty = "✅ MODERATE"
            challenging = False
        else:
            difficulty = "😊 EASY"
            challenging = False
            
        print(f"🎯 Actual difficulty: {difficulty}")
        
        if challenging:
            print("✅ CONFIRMED: This is a challenging real dataset!")
        else:
            print("❌ Not as challenging as expected")
            
        return {
            'dataset': dataset_name,
            'file_path': file_path,
            'baseline_roc': baseline_roc,
            'difficulty': difficulty,
            'challenging': challenging,
            'shape': df.shape,
            'domain': dataset_info['domain'],
            'description': dataset_info['description']
        }
        
    except Exception as e:
        print(f"❌ Error testing {dataset_name}: {e}")
        return None

def find_real_challenging_datasets():
    """Download and test real challenging datasets."""
    
    print("🔍 DOWNLOADING REAL CHALLENGING DATASETS")
    print("="*60)
    print("Testing datasets known to have poor baseline performance")
    print("="*60)
    
    results = []
    challenging_datasets = []
    
    for dataset_name, dataset_info in REAL_CHALLENGING_DATASETS.items():
        file_path = download_dataset(dataset_name, dataset_info)
        
        if file_path and os.path.exists(file_path):
            result = test_dataset_difficulty(file_path, dataset_name, dataset_info)
            if result:
                results.append(result)
                if result['challenging']:
                    challenging_datasets.append(result)
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 REAL CHALLENGING DATASET RESULTS")
    print(f"{'='*60}")
    
    if challenging_datasets:
        print(f"\n🎉 FOUND {len(challenging_datasets)} TRULY CHALLENGING REAL DATASETS:")
        print("-" * 60)
        
        for result in challenging_datasets:
            print(f"🔥 {result['dataset'].upper():<15} | "
                  f"{result['domain']:<25} | "
                  f"ROC: {result['baseline_roc']:.3f} | "
                  f"Shape: {result['shape']}")
        
        print(f"\n💡 RECOMMENDATIONS:")
        print("These are real datasets where feature engineering can make a difference!")
        
        # Show most challenging
        most_challenging = min(challenging_datasets, key=lambda x: x['baseline_roc'])
        print(f"\n🏆 MOST CHALLENGING REAL DATASET:")
        print(f"   {most_challenging['dataset'].upper()}")
        print(f"   Domain: {most_challenging['domain']}")
        print(f"   Baseline ROC: {most_challenging['baseline_roc']:.4f}")
        print(f"   File: {most_challenging['file_path']}")
        print(f"   Description: {most_challenging['description']}")
        
        return challenging_datasets
    else:
        print("❌ No challenging real datasets found or downloaded")
        return []

if __name__ == "__main__":
    challenging = find_real_challenging_datasets()
    
    if challenging:
        print(f"\n🚀 NEXT STEP:")
        print("Run CAAFE on these real challenging datasets!")
        print("These datasets require genuine domain expertise and feature engineering.")
    else:
        print(f"\n⚠️ Try downloading more datasets manually or check network connectivity")