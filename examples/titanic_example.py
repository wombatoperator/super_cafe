"""
CAAFE Titanic Example
====================

A simple, complete example showing how to use CAAFE to improve 
Titanic survival prediction with automated feature engineering.

This example demonstrates:
1. Loading and preprocessing real data
2. Using CAAFE to generate features 
3. Comparing model performance before/after
"""

import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# Add src to path to import caafe
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import caafe

def load_and_preprocess_titanic():
    """Load Titanic dataset and do basic preprocessing."""
    print("Loading Titanic dataset...")
    
    # Load the data
    data_path = project_root / "data" / "titanic.csv"
    df = pd.read_csv(data_path)
    
    print(f"Original dataset: {len(df)} passengers, {len(df.columns)} columns")
    
    # Separate features and target
    target = df['Survived'].copy()
    features = df.drop(['Survived', 'PassengerId', 'Name', 'Ticket'], axis=1)
    
    # Handle missing values
    features['Age'] = features['Age'].fillna(features['Age'].median())
    features['Embarked'] = features['Embarked'].fillna(features['Embarked'].mode()[0])
    features['Fare'] = features['Fare'].fillna(features['Fare'].median())
    
    # Convert categorical variables to numeric
    features['Sex'] = features['Sex'].map({'male': 0, 'female': 1})
    features['Embarked'] = features['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    
    # Create a simple cabin feature (has cabin or not)
    features['HasCabin'] = features['Cabin'].notna().astype(int)
    features = features.drop('Cabin', axis=1)
    
    # Ensure target has a name
    target.name = 'Survived'
    
    print(f"Preprocessed: {len(features)} passengers, {len(features.columns)} features")
    print(f"Features: {list(features.columns)}")
    print(f"Survival rate: {target.mean():.1%}")
    
    return features, target

def evaluate_baseline_model(X, y):
    """Evaluate baseline model using CAAFE's Critic."""
    print("Evaluating baseline model with CAAFE Critic...")
    
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
    
    return baseline_auc, baseline_std

def run_caafe_feature_engineering(X, y):
    """Use CAAFE to generate and validate new features."""
    print("\n" + "="*50)
    print("Running CAAFE Feature Engineering")
    print("="*50)
    
    # Dataset description to help the LLM understand the problem
    dataset_description = """
    Titanic passenger survival prediction dataset. 
    Features include passenger class, demographics (sex, age), 
    family information (siblings/spouses, parents/children), 
    ticket fare, and embarkation port.
    Goal: Predict whether a passenger survived the disaster.
    """
    
    try:
        # Use CAAFE to validate if feature engineering can help
        results = caafe.validate_feature_engineering(
            X=X,
            y=y,
            description=dataset_description.strip(),
            provider="openai",  # Change to "gemini" if you prefer
            model="gpt-4o-mini",  # Fast and cost-effective
            max_iterations=5,  # Keep it reasonable for demo
            patience=3,  # Stop after 3 consecutive failures
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

def compare_performance(X_original, X_enhanced, y):
    """Compare model performance using CAAFE's Critic for consistency."""
    print("\n" + "="*50)
    print("Performance Comparison (Using CAAFE Critic)")
    print("="*50)
    
    # Use CAAFE's Critic for consistent evaluation methodology
    critic = caafe.Critic(n_folds=5, random_state=42)
    
    # Evaluate original features
    print("Evaluating original features...")
    auc_orig, std_orig = critic.evaluate(X_original, y)
    
    # Evaluate enhanced features  
    print("Evaluating enhanced features...")
    auc_enh, std_enh = critic.evaluate(X_enhanced, y)
    
    # Calculate improvement
    improvement = auc_enh - auc_orig
    relative_improvement = (improvement / auc_orig) * 100
    
    # Show results
    print(f"\nResults:")
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
    
    # Show feature comparison
    if X_enhanced.shape[1] > X_original.shape[1]:
        new_features = [col for col in X_enhanced.columns if col not in X_original.columns]
        print(f"\nGenerated Features:")
        for feature in new_features:
            print(f"  • {feature}")
        
        # Quick feature importance using a simple model
        from sklearn.model_selection import train_test_split
        import xgboost as xgb
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_enhanced, y, test_size=0.3, random_state=42, stratify=y
        )
        
        model = xgb.XGBClassifier(random_state=42, verbosity=0)
        model.fit(X_train, y_train)
        
        # Show top feature importances
        importance_df = pd.DataFrame({
            'feature': X_enhanced.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop Feature Importances:")
        for _, row in importance_df.head(8).iterrows():
            is_new = row['feature'] not in X_original.columns
            marker = "NEW" if is_new else "   "
            print(f"  {marker} {row['feature']:<15} {row['importance']:.3f}")

def main():
    """Main execution function."""
    print("CAAFE Titanic Survival Prediction Example")
    print("=" * 50)
    
    # Check for API key
    if not os.getenv('OPENAI_API_KEY') and not os.getenv('GEMINI_API_KEY'):
        print("Warning: No API key found!")
        print("Please set one of:")
        print("  export OPENAI_API_KEY='sk-...'")
        print("  export GEMINI_API_KEY='...'")
        print("\nContinuing with data preparation and baseline evaluation only...\n")
    
    # Step 1: Load and preprocess data
    X, y = load_and_preprocess_titanic()
    
    # Step 2: Evaluate baseline model using CAAFE's Critic
    evaluate_baseline_model(X, y)
    
    # Step 3: Run CAAFE (if API key available)
    if os.getenv('OPENAI_API_KEY') or os.getenv('GEMINI_API_KEY'):
        results = run_caafe_feature_engineering(X, y)
        
        if results:
            print(f"\nCAAFE Results:")
            print(f"  Baseline ROC-AUC: {results['baseline_roc']:.4f}")
            print(f"  Final ROC-AUC: {results['final_roc']:.4f}")
            print(f"  Improvement: {results['total_improvement']:+.4f}")
            print(f"  Features accepted: {results['features_accepted']}")
            print(f"  Features rejected: {results['features_rejected']}")
            
            print(f"\nRecommendation:")
            print(f"  {results['recommendation']}")
            
            # If features were generated, compare performance
            if results['features_accepted'] > 0 and results['final_code']:
                print(f"\nApplying generated features to full dataset...")
                
                # Apply features to get enhanced dataset
                caafe_instance = caafe.SuperCAAFE(provider="openai", random_state=42)
                X_enhanced = caafe_instance._execute_feature_code(
                    results['final_code'], X.copy(), 'Survived'
                )
                
                # Show what features were created
                new_features = [col for col in X_enhanced.columns if col not in X.columns]
                print(f"Generated features: {new_features}")
                
                # Compare performance
                compare_performance(X, X_enhanced, y)
            else:
                print(f"\nNo features were accepted - baseline model is already strong!")
        
    else:
        print(f"\nSkipping CAAFE feature generation (no API key)")
        print(f"Set an API key to see automated feature engineering in action!")
    
    print(f"\n" + "="*50)
    print("Example completed!")
    print("="*50)

if __name__ == "__main__":
    main()