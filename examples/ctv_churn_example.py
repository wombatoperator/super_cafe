"""
CAAFE CTV Churn Example
=======================

A comprehensive example showing how to use CAAFE to improve 
CTV (Connected TV) customer churn prediction with automated feature engineering.

This example demonstrates:
1. Loading and preprocessing a large real-world dataset (125K customers)
2. Using CAAFE to generate features for customer churn prediction
3. Comparing model performance before/after feature engineering
4. Testing cache consistency with dataset-specific features
"""

import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path to import caafe
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import caafe

def load_and_preprocess_ctv_churn():
    """Load CTV churn dataset and do comprehensive, legitimate preprocessing."""
    print("Loading CTV churn dataset...")
    
    # Load the data
    data_path = project_root / "data" / "ctv_churn.csv"
    df = pd.read_csv(data_path)
    
    print(f"Original dataset: {len(df):,} customers, {len(df.columns)} columns")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    # Initial data quality check
    print(f"\nData Quality Assessment:")
    print(f"  Missing values: {df.isnull().sum().sum():,} total")
    print(f"  Duplicate rows: {df.duplicated().sum():,}")
    
    # Remove duplicates if any
    if df.duplicated().sum() > 0:
        df = df.drop_duplicates()
        print(f"  Removed {df.duplicated().sum():,} duplicate rows")
    
    # Separate features and target BEFORE any preprocessing
    target = df['churned'].copy()
    features = df.drop(['churned', 'customer_id'], axis=1)
    
    print(f"\nTarget distribution:")
    print(f"  Churn rate: {target.mean():.1%}")
    print(f"  Churned: {target.sum():,} customers")
    print(f"  Retained: {(~target.astype(bool)).sum():,} customers")
    
    # Detailed missing value analysis
    print(f"\nMissing Value Analysis:")
    missing_counts = features.isnull().sum()
    total_rows = len(features)
    
    for col, count in missing_counts[missing_counts > 0].items():
        pct = count / total_rows * 100
        print(f"  {col}: {count:,} ({pct:.1f}%)")
        
        # Check if missing values are related to churn (potential data leakage)
        missing_mask = features[col].isnull()
        if missing_mask.sum() > 0:
            churn_rate_missing = target[missing_mask].mean()
            churn_rate_not_missing = target[~missing_mask].mean()
            print(f"    Churn rate when {col} missing: {churn_rate_missing:.1%}")
            print(f"    Churn rate when {col} present: {churn_rate_not_missing:.1%}")
    
    # Handle missing values appropriately
    print(f"\nHandling missing values with domain knowledge:")
    
    # Age: Missing could indicate privacy concerns or data collection issues
    # Use median imputation but create a flag
    if 'age' in features.columns and features['age'].isnull().sum() > 0:
        features['age_missing'] = features['age'].isnull().astype(int)
        features['age'] = features['age'].fillna(features['age'].median())
        print(f"  Age: Imputed with median ({features['age'].median():.1f}), created missing flag")
    
    # Weekly hours: Missing could indicate inactive users
    if 'weekly_hours' in features.columns and features['weekly_hours'].isnull().sum() > 0:
        features['weekly_hours_missing'] = features['weekly_hours'].isnull().astype(int)
        # For usage metrics, 0 might be more appropriate than median for missing values
        features['weekly_hours'] = features['weekly_hours'].fillna(0)
        print(f"  Weekly hours: Imputed with 0 (assuming inactive), created missing flag")
    
    # Location and subscription_type: Use mode or create 'Unknown' category
    for col in ['location', 'subscription_type']:
        if col in features.columns and features[col].isnull().sum() > 0:
            # Create 'Unknown' category instead of using mode to avoid bias
            features[col] = features[col].fillna('Unknown')
            print(f"  {col}: Created 'Unknown' category for missing values")
    
    # Data validation and cleaning
    print(f"\nData Validation:")
    
    # Check for unrealistic values
    numeric_cols = features.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col in features.columns:
            min_val, max_val = features[col].min(), features[col].max()
            print(f"  {col}: range [{min_val:.2f}, {max_val:.2f}]")
            
            # Flag potential outliers (beyond 3 standard deviations)
            if features[col].std() > 0:
                z_scores = np.abs((features[col] - features[col].mean()) / features[col].std())
                outliers = (z_scores > 3).sum()
                if outliers > 0:
                    print(f"    Potential outliers (|z| > 3): {outliers} ({outliers/len(features)*100:.1f}%)")
    
    # Encode categorical variables properly
    print(f"\nEncoding categorical variables:")
    categorical_cols = features.select_dtypes(include=['object']).columns
    
    # Track encoding for interpretability
    encoding_info = {}
    
    for col in categorical_cols:
        unique_vals = features[col].nunique()
        print(f"  {col}: {unique_vals} unique values")
        
        if unique_vals <= 10:  # One-hot encode low cardinality
            # Use drop_first=True to avoid multicollinearity
            dummies = pd.get_dummies(features[col], prefix=col, drop_first=True)
            encoding_info[col] = f"One-hot encoded ({len(dummies.columns)} features)"
            features = pd.concat([features.drop(col, axis=1), dummies], axis=1)
        else:  # High cardinality - use target encoding or frequency encoding
            # For this example, we'll use frequency encoding
            freq_encoding = features[col].value_counts(normalize=True)
            features[col + '_frequency'] = features[col].map(freq_encoding)
            features = features.drop(col, axis=1)
            encoding_info[col] = "Frequency encoded"
    
    # Feature engineering based on domain knowledge
    print(f"\nDomain-specific feature engineering:")
    
    # Create meaningful derived features
    if all(col in features.columns for col in ['weekly_songs_played', 'weekly_unique_songs']):
        # Song diversity ratio
        features['song_diversity_ratio'] = (
            features['weekly_unique_songs'] / (features['weekly_songs_played'] + 1)
        )
        print(f"  Created song_diversity_ratio")
    
    if all(col in features.columns for col in ['weekly_hours', 'average_session_length']):
        # Sessions per week
        features['sessions_per_week'] = (
            features['weekly_hours'] * 60 / (features['average_session_length'] + 1)
        )
        print(f"  Created sessions_per_week")
    
    if 'signup_date' in features.columns:
        # Convert signup_date to days since signup (more interpretable)
        # Assuming negative values are days before some reference point
        features['days_since_signup'] = -features['signup_date']  # Make positive
        features = features.drop('signup_date', axis=1)
        print(f"  Converted signup_date to days_since_signup")
    
    # Final data quality check
    print(f"\nFinal preprocessing summary:")
    print(f"  Features: {len(features.columns)}")
    print(f"  Samples: {len(features):,}")
    print(f"  Missing values remaining: {features.isnull().sum().sum()}")
    print(f"  Memory usage: {features.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    # Ensure no data leakage by checking correlations with target
    print(f"\nChecking for potential data leakage:")
    correlations = {}
    for col in features.select_dtypes(include=[np.number]).columns:
        corr = np.corrcoef(features[col], target)[0, 1]
        if not np.isnan(corr):
            correlations[col] = abs(corr)
    
    # Sort by correlation strength
    sorted_corr = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
    print(f"  Top 5 correlations with target:")
    for col, corr in sorted_corr[:5]:
        print(f"    {col}: {corr:.3f}")
        
    # Flag suspiciously high correlations (potential leakage)
    high_corr_features = [col for col, corr in correlations.items() if corr > 0.8]
    if high_corr_features:
        print(f"  WARNING: High correlation features (>0.8): {high_corr_features}")
        print(f"  These may indicate data leakage - review carefully!")
    
    # Ensure target has a name
    target.name = 'churned'
    
    return features, target

def evaluate_baseline_model(X, y):
    """Evaluate baseline model using CAAFE's Critic."""
    print("\n" + "="*60)
    print("Baseline Model Evaluation")
    print("="*60)
    
    # Use CAAFE's Critic for consistent evaluation
    critic = caafe.Critic(n_folds=5, random_state=42)
    baseline_auc, baseline_std = critic.evaluate(X, y)
    
    print(f"Baseline Performance (CAAFE Critic):") 
    print(f"  ROC-AUC: {baseline_auc:.4f} ± {baseline_std:.4f}")
    print(f"  Range: {baseline_auc - baseline_std:.4f} to {baseline_auc + baseline_std:.4f}")
    
    # Show adaptive epsilon threshold
    epsilon = critic.get_epsilon(baseline_auc)
    print(f"  Adaptive threshold (ε): {epsilon:.4f}")
    print(f"  (Features need >{epsilon:.4f} improvement to be accepted)")
    
    # Show data characteristics
    print(f"\nDataset characteristics:")
    print(f"  Samples: {len(X):,}")
    print(f"  Features: {len(X.columns)}")
    print(f"  Churn rate: {y.mean():.1%}")
    print(f"  Class balance: {(~y.astype(bool)).sum():,} retained, {y.sum():,} churned")
    
    return baseline_auc, baseline_std

def run_caafe_feature_engineering(X, y):
    """Use CAAFE to generate and validate new features for CTV churn."""
    print("\n" + "="*60)
    print("CAAFE Feature Engineering for CTV Churn")
    print("="*60)
    
    # Comprehensive dataset description for better LLM understanding
    dataset_description = """
    Connected TV (CTV) customer churn prediction dataset with 125,000 customers.
    
    Customer Demographics:
    - Age and geographic location (US states)
    
    Subscription Details:
    - Subscription type (Free, Premium, Student, Family)
    - Payment plan (Monthly, Yearly) and method (Credit Card, PayPal, Apple Pay)
    - Subscription pause history and signup recency
    
    Usage Patterns:
    - Weekly listening hours and session lengths
    - Song engagement (skip rates, songs played, unique songs)
    - Social features (favorite artists, platform friends, playlists)
    - Interaction patterns (shared playlists, notifications clicked)
    
    Customer Service:
    - Customer service inquiry levels (Low, Medium, High)
    
    Goal: Predict which customers will churn (cancel their subscription)
    """
    
    try:
        # Use CAAFE to validate if feature engineering can help
        print("Starting CAAFE analysis...")
        print(f"Dataset: {len(X):,} samples, {len(X.columns)} features")
        
        results = caafe.validate_feature_engineering(
            X=X,
            y=y,
            description=dataset_description.strip(),
            provider="openai",  # Change to "gemini" if you prefer
            model="gpt-4o-mini",  # Fast and cost-effective
            max_iterations=8,  # More iterations for complex dataset
            patience=4,  # Allow more consecutive failures
            random_state=42
        )
        
        return results
        
    except Exception as e:
        print(f"CAAFE failed: {e}")
        print("\nMake sure you have set your API key:")
        print("  export OPENAI_API_KEY='sk-...'")
        print("  # or")
        print("  export GEMINI_API_KEY='...'")
        return None

def analyze_feature_engineering_results(results, X_original):
    """Analyze and display CAAFE results in detail."""
    print("\n" + "="*60)
    print("CAAFE Feature Engineering Results")
    print("="*60)
    
    print(f"Performance Summary:")
    print(f"  Baseline ROC-AUC: {results['baseline_roc']:.4f}")
    print(f"  Final ROC-AUC: {results['final_roc']:.4f}")
    print(f"  Total Improvement: {results['total_improvement']:+.4f}")
    print(f"  Relative Improvement: {(results['total_improvement']/results['baseline_roc']*100):+.1f}%")
    
    print(f"\nFeature Generation:")
    print(f"  Features accepted: {results['features_accepted']}")
    print(f"  Features rejected: {results['features_rejected']}")
    print(f"  Success rate: {results['features_accepted']/(results['features_accepted']+results['features_rejected'])*100:.1f}%")
    
    print(f"\nRecommendation:")
    print(f"  {results['recommendation']}")
    
    return results['features_accepted'] > 0 and results['final_code']

def apply_and_compare_features(results, X_original, y):
    """Apply generated features and compare performance."""
    print("\n" + "="*60)
    print("Feature Application and Performance Comparison")
    print("="*60)
    
    # Apply features to get enhanced dataset
    print("Applying generated features to full dataset...")
    caafe_instance = caafe.SuperCAAFE(provider="openai", random_state=42)
    
    # Test our cache consistency - should use dataset-specific cache
    dataset_context = "CTV customer churn prediction with subscription and usage patterns"
    dataset_columns = list(X_original.columns)
    
    print(f"Using dataset-specific caching with context and {len(dataset_columns)} columns...")
    
    X_enhanced = caafe_instance._execute_feature_code(
        results['final_code'], X_original.copy(), 'churned'
    )
    
    # Show what features were created
    new_features = [col for col in X_enhanced.columns if col not in X_original.columns]
    print(f"Generated {len(new_features)} new features: {new_features}")
    
    # Compare performance using CAAFE's Critic for consistency
    print("\nEvaluating enhanced model...")
    critic = caafe.Critic(n_folds=5, random_state=42)
    
    # Evaluate original features
    print("Re-evaluating original features...")
    auc_orig, std_orig = critic.evaluate(X_original, y)
    
    # Evaluate enhanced features  
    print("Evaluating enhanced features...")
    auc_enh, std_enh = critic.evaluate(X_enhanced, y)
    
    # Calculate improvement
    improvement = auc_enh - auc_orig
    relative_improvement = (improvement / auc_orig) * 100
    
    # Show results
    print(f"\nPerformance Comparison:")
    print(f"Original Model ({X_original.shape[1]} features):")
    print(f"  ROC-AUC: {auc_orig:.4f} ± {std_orig:.4f}")
    
    print(f"\nEnhanced Model ({X_enhanced.shape[1]} features):")
    print(f"  ROC-AUC: {auc_enh:.4f} ± {std_enh:.4f}")
    
    print(f"\nImprovement:")
    print(f"  Absolute: {improvement:+.4f}")
    print(f"  Relative: {relative_improvement:+.1f}%")
    
    # Use CAAFE's acceptance criteria
    epsilon = critic.get_epsilon(auc_orig)
    accept, _, reason = critic.should_accept_feature(auc_orig, auc_enh, std_orig, std_enh)
    
    print(f"\nCAAFE Evaluation:")
    print(f"  Acceptance threshold (ε): {epsilon:.4f}")
    print(f"  Decision: {'ACCEPT' if accept else 'REJECT'}")
    print(f"  Reason: {reason}")
    
    # Test apply_cached_features with our new parameters
    print(f"\nTesting apply_cached_features with dataset-specific parameters...")
    try:
        # This should use our enhanced caching logic
        X_cached = caafe_instance.apply_cached_features(
            X_original.sample(1000, random_state=42),  # Use subset for speed
            y.sample(1000, random_state=42), 
            dataset_context=dataset_context,
            dataset_columns=dataset_columns
        )
        print(f"✓ apply_cached_features worked with dataset-specific parameters")
        print(f"  Original subset: {X_original.sample(1000, random_state=42).shape[1]} features")
        print(f"  Cached result: {X_cached.shape[1]} features")
    except Exception as e:
        print(f"✗ apply_cached_features failed: {e}")
    
    # Feature importance analysis
    if len(new_features) > 0:
        print(f"\nFeature Importance Analysis:")
        from sklearn.model_selection import train_test_split
        import xgboost as xgb
        
        # Sample data for faster analysis
        sample_size = min(10000, len(X_enhanced))
        sample_idx = np.random.choice(len(X_enhanced), sample_size, replace=False)
        X_sample = X_enhanced.iloc[sample_idx]
        y_sample = y.iloc[sample_idx]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_sample, y_sample, test_size=0.3, random_state=42, stratify=y_sample
        )
        
        model = xgb.XGBClassifier(random_state=42, verbosity=0, n_estimators=100)
        model.fit(X_train, y_train)
        
        # Show top feature importances
        importance_df = pd.DataFrame({
            'feature': X_enhanced.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"Top 15 Feature Importances:")
        for i, (_, row) in enumerate(importance_df.head(15).iterrows()):
            is_new = row['feature'] in new_features
            marker = "NEW" if is_new else "   "
            print(f"  {i+1:2d}. {marker} {row['feature']:<25} {row['importance']:.3f}")

def test_cache_consistency():
    """Test that our cache consistency improvements work correctly."""
    print("\n" + "="*60)
    print("Testing Cache Consistency Implementation")
    print("="*60)
    
    # Check cache directory
    cache_dir = Path(".caafe_cache")
    if cache_dir.exists():
        cache_files = list(cache_dir.glob("*.json"))
        print(f"Cache directory contains {len(cache_files)} files:")
        for cache_file in cache_files:
            print(f"  - {cache_file.name}")
            
        # Check if we have dataset-specific cache
        dataset_specific_files = [f for f in cache_files if "features_" in f.name and "global" not in f.name]
        if dataset_specific_files:
            print(f"\n✓ Dataset-specific cache files created: {len(dataset_specific_files)}")
            
            # Read one cache file to verify structure
            with open(dataset_specific_files[0], 'r') as f:
                import json
                cache_data = json.load(f)
                print(f"  - Version: {cache_data.get('version', 'unknown')}")
                print(f"  - Dataset hash: {cache_data.get('dataset_hash', 'unknown')}")
                print(f"  - Features cached: {len(cache_data.get('features', []))}")
        else:
            print("No dataset-specific cache files found")
    else:
        print("No cache directory found")

def main():
    """Main execution function."""
    print("CAAFE CTV Customer Churn Prediction Example")
    print("=" * 60)
    
    # Check for API key
    if not os.getenv('OPENAI_API_KEY') and not os.getenv('GEMINI_API_KEY'):
        print("Warning: No API key found!")
        print("Please set one of:")
        print("  export OPENAI_API_KEY='sk-...'")
        print("  export GEMINI_API_KEY='...'")
        print("\nContinuing with data preparation and baseline evaluation only...\n")
    
    # Step 1: Load and preprocess data
    X, y = load_and_preprocess_ctv_churn()
    
    # Step 2: Evaluate baseline model using CAAFE's Critic
    baseline_auc, baseline_std = evaluate_baseline_model(X, y)
    
    # Step 3: Run CAAFE (if API key available)
    if os.getenv('OPENAI_API_KEY') or os.getenv('GEMINI_API_KEY'):
        results = run_caafe_feature_engineering(X, y)
        
        if results:
            # Analyze results
            has_features = analyze_feature_engineering_results(results, X)
            
            # Apply and compare features if any were generated
            if has_features:
                apply_and_compare_features(results, X, y)
            else:
                print(f"\nNo features were accepted - baseline model is already strong!")
                
        # Test cache consistency regardless of feature generation
        test_cache_consistency()
        
    else:
        print(f"\nSkipping CAAFE feature generation (no API key)")
        print(f"Set an API key to see automated feature engineering in action!")
    
    print(f"\n" + "="*60)
    print("CTV Churn Example Completed!")
    print("="*60)

if __name__ == "__main__":
    main()