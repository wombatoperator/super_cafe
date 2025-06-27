"""
CAAFE evaluation utilities - Legacy compatibility wrapper.

This module provides backward compatibility for the original CAAFE evaluation methodology.
All core evaluation logic has been moved to the optimized Critic class in critic.py.
"""

from typing import Tuple, Dict
import pandas as pd
from .critic import Critic


def evaluate_dataset(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    target_name: str,
    method: str = "xgb",
    seed: int = 0
) -> Dict[str, float]:
    """
    Legacy compatibility wrapper for evaluate_dataset.
    
    Args:
        df_train: Training dataframe with target
        df_test: Test dataframe with target  
        target_name: Name of target column
        method: Method to use for evaluation (ignored - always uses optimized XGBoost)
        seed: Random seed
        
    Returns:
        Dictionary with roc and acc scores
    """
    # Extract features and target
    X_train = df_train.drop(columns=[target_name])
    y_train = df_train[target_name]
    X_test = df_test.drop(columns=[target_name])
    y_test = df_test[target_name]
    
    # Use optimized Critic for ROC evaluation
    critic = Critic(seed=seed, holdout=0.2)  # Use holdout since we have separate train/test
    
    # Train on training set, evaluate on test set
    # Combine for Critic which handles its own splitting
    X_combined = pd.concat([X_train, X_test], axis=0, ignore_index=True)
    y_combined = pd.concat([y_train, y_test], axis=0, ignore_index=True)
    
    roc_score = critic.score(X_combined, y_combined)
    
    # Calculate true accuracy using the trained model
    from .critic import _clean_df, _clean_target
    y_test_clean = _clean_target(y_test)
    X_test_clean = _clean_df(X_test.loc[y_test_clean.index])
    
    # Use the critic's trained model to get predictions
    y_pred_proba = critic._model.predict_proba(X_test_clean.values)[:, 1]
    y_pred_labels = (y_pred_proba > 0.5).astype(int)
    accuracy = (y_pred_labels == y_test_clean.values).mean()
    
    return {
        "roc": float(roc_score),
        "acc": float(accuracy)
    }


def execute_and_evaluate_code_block(
    df: pd.DataFrame,
    target_name: str,
    full_code: str,
    new_code: str,
    n_splits: int = 3,
    n_repeats: int = 2,
    method: str = "xgb"
) -> Tuple[Exception, list, list, list, list]:
    """
    Legacy compatibility wrapper - delegates to the main CAAFE evaluation loop.
    
    This function is deprecated in favor of the optimized Critic class.
    It's kept for backward compatibility only.
    
    Args:
        df: Full dataframe with target
        target_name: Name of target column
        full_code: Previously accumulated code
        new_code: New code to test
        n_splits: Number of CV splits
        n_repeats: Number of CV repeats
        method: Evaluation method (ignored)
        
    Returns:
        Tuple of (error, new_rocs, new_accs, old_rocs, old_accs)
    """
    try:
        from .core import execute_code_safely
        
        # Extract features and target
        X = df.drop(columns=[target_name])
        y = df[target_name]
        
        # Create Critic instance
        critic = Critic(folds=n_splits, repeats=n_repeats)
        
        # Evaluate old features
        if full_code.strip():
            X_old = execute_code_safely(full_code, X.copy(), target_name)
        else:
            X_old = X.copy()
        
        old_roc = critic.score(X_old, y)
        
        # Evaluate new features
        combined_code = full_code + "\n" + new_code if full_code.strip() else new_code
        X_new = execute_code_safely(combined_code, X.copy(), target_name)
        new_roc = critic.score(X_new, y)
        
        # Return in legacy format (lists for compatibility)
        # For binary classification, we'll use ROC as a proxy for accuracy in the legacy wrapper
        return None, [new_roc], [new_roc], [old_roc], [old_roc]
        
    except Exception as e:
        return e, None, None, None, None


def execute_code_safely(code: str, df: pd.DataFrame, target_column=None) -> pd.DataFrame:
    """
    Legacy wrapper - imports the secure version from core to avoid duplication.
    
    Args:
        code: Python code to execute
        df: Dataframe to execute on
        target_column: Target column name (for leakage prevention)
        
    Returns:
        Modified dataframe
    """
    from .core import execute_code_safely as secure_execute
    return secure_execute(code, df, target_column)