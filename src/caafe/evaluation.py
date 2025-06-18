"""
CAAFE evaluation utilities - separated from core for clarity.
"""

import copy
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, accuracy_score
from typing import Tuple, Dict

def evaluate_dataset(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    target_name: str,
    method: str = "xgb",
    seed: int = 0
) -> Dict[str, float]:
    """
    Evaluate a dataset with a specific method.
    
    Args:
        df_train: Training dataframe with target
        df_test: Test dataframe with target
        target_name: Name of target column
        method: Method to use for evaluation
        seed: Random seed
        
    Returns:
        Dictionary with roc and acc scores
    """
    # Split features and target
    X_train = df_train.drop(columns=[target_name])
    y_train = df_train[target_name]
    X_test = df_test.drop(columns=[target_name])
    y_test = df_test[target_name]
    
    # Handle missing values
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)
    
    # Train model
    if method == "xgb":
        model = xgb.XGBClassifier(random_state=seed, verbosity=0)
    else:
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(random_state=seed, max_iter=1000)
    
    model.fit(X_train, y_train)
    
    # Get predictions
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        roc = roc_auc_score(y_test, y_pred_proba)
    else:
        roc = 0.0
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    return {
        "roc": float(roc),
        "acc": float(acc)
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
    Execute code and evaluate performance using cross-validation.
    Following the original CAAFE evaluation methodology.
    
    Args:
        df: Full dataframe with target
        target_name: Name of target column
        full_code: Previously accumulated code
        new_code: New code to test
        n_splits: Number of CV splits
        n_repeats: Number of CV repeats
        method: Evaluation method
        
    Returns:
        Tuple of (error, new_rocs, new_accs, old_rocs, old_accs)
    """
    old_rocs, old_accs, new_rocs, new_accs = [], [], [], []
    
    # Set up cross-validation
    ss = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=0)
    
    for train_idx, valid_idx in ss.split(df):
        df_train, df_valid = df.iloc[train_idx].copy(), df.iloc[valid_idx].copy()
        
        # Extract target and remove from dataframes (following reference code pattern)
        target_train = df_train[target_name]
        target_valid = df_valid[target_name]
        df_train_features = df_train.drop(columns=[target_name])
        df_valid_features = df_valid.drop(columns=[target_name])
        
        try:
            # Apply old code (accumulated features so far)
            df_train_old = df_train_features.copy()
            df_valid_old = df_valid_features.copy()
            
            if full_code.strip():
                df_train_old = execute_code_safely(full_code, df_train_old)
                df_valid_old = execute_code_safely(full_code, df_valid_old)
            
            # Apply old + new code
            df_train_new = df_train_features.copy()
            df_valid_new = df_valid_features.copy()
            
            combined_code = full_code + "\n" + new_code if full_code.strip() else new_code
            df_train_new = execute_code_safely(combined_code, df_train_new)
            df_valid_new = execute_code_safely(combined_code, df_valid_new)
            
            # Add target back (following reference code pattern)
            df_train_old[target_name] = target_train
            df_valid_old[target_name] = target_valid
            df_train_new[target_name] = target_train
            df_valid_new[target_name] = target_valid
            
        except Exception as e:
            return e, None, None, None, None
        
        # Evaluate old features
        result_old = evaluate_dataset(df_train_old, df_valid_old, target_name, method)
        old_rocs.append(result_old["roc"])
        old_accs.append(result_old["acc"])
        
        # Evaluate new features
        result_new = evaluate_dataset(df_train_new, df_valid_new, target_name, method)
        new_rocs.append(result_new["roc"])
        new_accs.append(result_new["acc"])
    
    return None, new_rocs, new_accs, old_rocs, old_accs

def execute_code_safely(code: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Safely execute code on a dataframe.
    
    Args:
        code: Python code to execute
        df: Dataframe to execute on
        
    Returns:
        Modified dataframe
    """
    if not code.strip():
        return df
    
    # Create safe execution environment
    local_vars = {
        'df': df.copy(),
        'pd': pd,
        'np': np
    }
    
    # Execute code
    exec(code, {"__builtins__": {}, "pd": pd, "np": np}, local_vars)
    
    return local_vars['df']