"""
Modern CAAFE Example

This demonstrates the XGBoost-optimized CAAFE in action with minimal code.
"""

import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import caafe

def create_sample_data():
    """Create a simple dataset for demonstration."""
    np.random.seed(42)
    
    # Create features
    n_samples = 500
    X = pd.DataFrame({
        'age': np.random.randint(18, 80, n_samples),
        'income': np.random.normal(50000, 20000, n_samples),
        'debt': np.random.normal(10000, 5000, n_samples),
        'credit_score': np.random.randint(300, 850, n_samples),
        'employment_years': np.random.randint(0, 40, n_samples),
        'location': np.random.choice(['urban', 'suburban', 'rural'], n_samples)
    })
    
    # Create realistic target (loan default)
    default_prob = (
        0.1 + 
        0.2 * (X['debt'] / X['income']).clip(0, 1) +
        0.15 * ((850 - X['credit_score']) / 550) +
        0.05 * (X['age'] < 25).astype(int)
    )
    y = pd.Series(np.random.binomial(1, default_prob), name='default')
    
    return X, y

def main():
    """Run the simple example."""
    print("Modern CAAFE Example")
    print("====================")
    
    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ Set OPENAI_API_KEY environment variable")
        print("Example: export OPENAI_API_KEY='sk-...'")
        return
    
    # Create sample data
    print("Creating sample loan default dataset...")
    X, y = create_sample_data()
    
    print(f"Dataset: {len(X)} samples, {len(X.columns)} features")
    print(f"Features: {list(X.columns)}")
    print(f"Target: {y.value_counts().to_dict()}")
    
    # Dataset description for the LLM
    description = """
    Loan default prediction dataset. Features include borrower demographics,
    financial information, and location. The goal is to predict whether
    someone will default on their loan (binary classification).
    """
    
    print("\nRunning Modern CAAFE feature generation...")
    print("This will take 1-3 minutes with XGBoost optimization.")
    
    try:
        # Generate features - this is the main API!
        caafe_model = caafe.CAAFE(max_iterations=3, model="gpt-4o-mini")
        X_enhanced = caafe_model.generate_features(
            X=X,
            y=y, 
            description=description.strip()
        )
        
        print(f"\n✅ Success!")
        print(f"Original features: {X.shape[1]}")
        print(f"Enhanced features: {X_enhanced.shape[1]}")
        print(f"New features added: {X_enhanced.shape[1] - X.shape[1]}")
        
        # Show new features
        new_features = [col for col in X_enhanced.columns if col not in X.columns]
        if new_features:
            print(f"\nGenerated features:")
            for feature in new_features:
                print(f"  - {feature}")
        
        # Quick performance comparison
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import roc_auc_score
        import xgboost as xgb
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        X_enh_train = X_enhanced.loc[X_train.index]
        X_enh_test = X_enhanced.loc[X_test.index]
        
        # Original model
        model_orig = xgb.XGBClassifier(random_state=42, verbosity=0)
        model_orig.fit(X_train, y_train)
        pred_orig = model_orig.predict_proba(X_test)[:, 1]
        auc_orig = roc_auc_score(y_test, pred_orig)
        
        # Enhanced model
        model_enh = xgb.XGBClassifier(random_state=42, verbosity=0)
        model_enh.fit(X_enh_train, y_train)
        pred_enh = model_enh.predict_proba(X_enh_test)[:, 1]
        auc_enh = roc_auc_score(y_test, pred_enh)
        
        print(f"\nPerformance Comparison:")
        print(f"Original XGBoost AUC: {auc_orig:.4f}")
        print(f"Enhanced XGBoost AUC: {auc_enh:.4f}")
        print(f"Improvement: {auc_enh - auc_orig:+.4f}")
        
        if auc_enh > auc_orig:
            print("🎉 Modern CAAFE improved XGBoost performance!")
        else:
            print("📊 No improvement this time (common with synthetic data)")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure your OpenAI API key is valid and you have internet access.")

if __name__ == "__main__":
    main()