#!/usr/bin/env python3
"""
Real Dataset Loader for CAAFE Evaluation
========================================

Loads and preprocesses real datasets for meaningful CAAFE testing.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sys
sys.path.append('src')
from caafe import CAAFE
from caafe.critic import Critic

def load_titanic():
    """Load and preprocess Titanic dataset."""
    df = pd.read_csv('data/titanic.csv')
    
    # Clean and prepare features (let CAAFE handle encoding)
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['Embarked'] = df['Embarked'].fillna('S')
    
    # Select features (avoid obvious leaks like Name, Ticket) - keep categorical as-is
    feature_cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    X = df[feature_cols]
    y = df['Survived']
    y.name = 'Survived'
    
    description = """Titanic survival dataset. Predict passenger survival based on demographics, ticket class, family size, and fare. 
    Historical patterns: higher class passengers, women, children, and those with higher fares had better survival rates."""
    
    return X, y, description

def load_diabetes():
    """Load and preprocess Diabetes dataset."""
    df = pd.read_csv('data/diabetes.csv')
    
    # Features are already clean
    feature_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    X = df[feature_cols]
    y = df['Outcome']
    y.name = 'Outcome'
    
    description = """Diabetes prediction dataset. Predict diabetes diagnosis based on medical indicators.
    Key relationships: glucose levels, BMI, age, and family history are strong predictors."""
    
    return X, y, description

def load_privatead_clean():
    """Load and clean PrivateAd dataset, removing obvious leakage."""
    df = pd.read_parquet('/Volumes/T7/criterio_data/raw/privatead_subset.parquet')
    
    print(f"Original PrivateAd shape: {df.shape}")
    
    # Sample for manageable size
    if len(df) > 5000:
        df = df.sample(5000, random_state=42)
    
    # Identify and remove obvious leaking columns
    leak_patterns = [
        'click', 'sale', 'visit', 'conversion', 'landed', 'delay_after'
    ]
    
    leaking_cols = []
    for col in df.columns:
        if any(pattern in col.lower() for pattern in leak_patterns):
            leaking_cols.append(col)
    
    print(f"Removing potential leaking columns: {leaking_cols}")
    
    # Keep only feature columns and create target
    feature_cols = [col for col in df.columns if col not in leaking_cols and not col.startswith('id')]
    
    # Use 'is_clicked' as target if available, otherwise create synthetic
    if 'is_clicked' in df.columns:
        target = df['is_clicked']
        target_name = 'is_clicked'
    else:
        # Create synthetic target based on display_order (early displays more likely to be effective)
        target = (df['display_order'] <= 3).astype(int)
        target_name = 'early_display'
    
    # Select reasonable number of features
    feature_cols = [col for col in feature_cols if col != target_name][:20]
    
    X = df[feature_cols].fillna(0).infer_objects(copy=False)
    
    # Let CAAFE handle all encoding - keep object columns as-is
    # The Critic's _clean_df() will handle categorical encoding properly
    
    y = pd.Series(target, name=target_name)
    
    print(f"Clean PrivateAd shape: X={X.shape}, y distribution: {y.value_counts().to_dict()}")
    
    description = f"""Private advertising dataset. Predict {target_name} based on user and campaign features.
    Contains browser features, campaign context, and user behavior patterns."""
    
    return X, y, description

def evaluate_real_dataset(dataset_name, max_iterations=3):
    """Evaluate CAAFE on a real dataset."""
    print(f"\n{'='*60}")
    print(f"🧪 REAL DATA EVALUATION: {dataset_name.upper()}")
    print(f"{'='*60}")
    
    # Load dataset
    if dataset_name == "titanic":
        X, y, description = load_titanic()
    elif dataset_name == "diabetes":
        X, y, description = load_diabetes()
    elif dataset_name == "privatead":
        X, y, description = load_privatead_clean()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    print(f"📊 Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"📊 Target distribution: {y.value_counts().to_dict()}")
    
    # Use Critic for baseline evaluation (same as CAAFE uses internally)
    critic = Critic(folds=3, repeats=2, n_jobs=-1)
    baseline_roc = critic.score(X, y)
    
    print(f"📊 Baseline ROC-AUC: {baseline_roc:.4f}")
    
    # CAAFE evaluation - use same Critic instance for consistency
    caafe = CAAFE(
        provider="ollama",
        model="llama3:8b-instruct-q4_K_M",
        ollama_models_path="/Volumes/T7/Ollama",
        max_iterations=max_iterations,
        scorer=critic  # Use same critic instance for consistent evaluation
    )
    
    print(f"\n🚀 Starting CAAFE feature generation...")
    X_enhanced = caafe.generate_features(X, y, description)
    
    # Evaluate enhanced features using same Critic
    if X_enhanced.shape[1] > X.shape[1]:
        enhanced_roc = critic.score(X_enhanced, y)
    else:
        enhanced_roc = baseline_roc
    
    # Results
    improvement = enhanced_roc - baseline_roc
    generated_features = caafe.get_generated_features()
    
    print(f"\n📈 RESULTS:")
    print(f"   Baseline ROC-AUC: {baseline_roc:.4f}")
    print(f"   Enhanced ROC-AUC: {enhanced_roc:.4f}")
    print(f"   Improvement: {improvement:+.4f} ({improvement/baseline_roc*100:+.1f}%)")
    print(f"   Generated features: {len(generated_features)}")
    
    if generated_features:
        print(f"   Feature names: {generated_features}")
        
        # Add feature importance analysis if features were generated
        if X_enhanced.shape[1] > X.shape[1]:
            try:
                impact_analysis = critic.analyze_feature_impact(X, X_enhanced, y)
                print(f"\n🔍 Feature Importance Analysis:")
                print(f"   Top 3 overall features:")
                for feat in impact_analysis['top_overall_features'][:3]:
                    print(f"     {feat['feature']}: {feat['importance']:.3f}")
                if impact_analysis['new_feature_importance']:
                    print(f"   New feature importance:")
                    for feat in impact_analysis['new_feature_importance']:
                        print(f"     {feat['feature']}: {feat['importance']:.3f}")
            except Exception as e:
                print(f"   ⚠️  Feature importance analysis failed: {e}")
    
    return {
        'dataset': dataset_name,
        'baseline_roc': baseline_roc,
        'enhanced_roc': enhanced_roc,
        'improvement': improvement,
        'improvement_pct': improvement/baseline_roc*100,
        'generated_features': generated_features,
        'success': improvement > 0
    }

def main():
    """Test CAAFE on real datasets."""
    datasets = ["titanic", "diabetes", "privatead"]
    
    results = []
    for dataset in datasets:
        try:
            result = evaluate_real_dataset(dataset, max_iterations=2)
            results.append(result)
        except Exception as e:
            print(f"❌ Failed on {dataset}: {e}")
            results.append({
                'dataset': dataset,
                'success': False,
                'error': str(e)
            })
    
    # Summary
    print(f"\n{'='*60}")
    print("📋 REAL DATA EVALUATION SUMMARY")
    print(f"{'='*60}")
    
    successful = [r for r in results if r.get('success', False)]
    improved = [r for r in successful if r.get('improvement', 0) > 0]
    
    print(f"Datasets evaluated: {len(results)}")
    print(f"Successful runs: {len(successful)}")
    print(f"Datasets improved: {len(improved)}")
    
    for result in successful:
        print(f"\n📊 {result['dataset'].upper()}:")
        print(f"   Improvement: {result['improvement']:+.4f} ({result['improvement_pct']:+.1f}%)")
        print(f"   Features: {len(result['generated_features'])}")

if __name__ == "__main__":
    main()