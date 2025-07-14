"""
SUPER CAAFE Critic: Deterministic, High-Performance Feature Validation
=====================================================================

A stateless, deterministic XGBoost critic that provides consistent evaluation
signals for feature engineering decisions. Designed for speed and reproducibility.
"""

from typing import Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.base import clone
from xgboost import XGBClassifier


class Critic:
    """
    Deterministic XGBoost-based feature evaluator.
    
    Key principles:
    - Complete statelessness: fresh models for every evaluation
    - Deterministic results: fixed seeds and single-threaded execution
    - Fast evaluation: 'hist' tree method for speed
    - Proper data handling: consistent encoding and imputation
    """
    
    def __init__(
        self,
        n_folds: int = 3,
        epsilon: float = 0.001,
        adaptive_epsilon: bool = True,
        random_state: int = 42
    ):
        """
        Initialize the critic with evaluation parameters.
        
        Args:
            n_folds: Number of cross-validation folds
            epsilon: Minimum improvement threshold (adaptive if enabled)
            adaptive_epsilon: Whether to adjust epsilon based on baseline performance
            random_state: Seed for reproducibility
        """
        self.n_folds = n_folds
        self.epsilon = epsilon
        self.adaptive_epsilon = adaptive_epsilon
        self.random_state = random_state
        
        # Fixed XGBoost parameters for speed and determinism
        self.model_params = {
            'tree_method': 'hist',  # Fast histogram-based method
            'n_estimators': 50,     # Reduced for speed
            'max_depth': 3,         # Shallow trees for generalization
            'learning_rate': 0.3,   # Higher LR with fewer trees
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 5,
            'gamma': 2.0,
            'random_state': random_state,
            'n_jobs': 1,            # Single thread for determinism
            'verbosity': 0,
            'eval_metric': 'auc',
            'use_label_encoder': False
        }
        
    def _prepare_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for XGBoost with consistent encoding and imputation.
        
        Args:
            X: Feature dataframe
            y: Target series
            
        Returns:
            Tuple of (prepared features, encoded target)
        """
        # Create a copy to avoid modifying original
        X_prep = X.copy()
        
        # Handle numeric columns
        numeric_cols = X_prep.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            # Vectorized median imputation
            medians = X_prep[numeric_cols].median()
            X_prep[numeric_cols] = X_prep[numeric_cols].fillna(medians)
        
        # Handle categorical columns
        categorical_cols = X_prep.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            # Simple ordinal encoding with NaN handling
            X_prep[col] = pd.Categorical(X_prep[col]).codes + 1  # +1 to reserve 0 for NaN
            
        # Ensure all values are float32 for XGBoost
        X_prep = X_prep.astype(np.float32)
        
        # Encode target if necessary
        if y.dtype == 'object' or y.dtype.name == 'category':
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            # Verify binary classification
            if len(np.unique(y_encoded)) > 2:
                raise ValueError("Critic only supports binary classification")
        else:
            y_encoded = y.values
            
        return X_prep.values, y_encoded
    
    def get_epsilon(self, baseline_score: float) -> float:
        """
        Get adaptive epsilon threshold based on baseline performance.
        
        Implements the insight from SUPER CAAFE: harder to improve strong baselines.
        
        Args:
            baseline_score: Current model's ROC-AUC
            
        Returns:
            Adjusted epsilon threshold
        """
        if not self.adaptive_epsilon:
            return self.epsilon
            
        if baseline_score < 0.6:
            return 0.01    # 1% for random-level baselines
        elif baseline_score < 0.7:
            return 0.005   # 0.5% for weak baselines
        elif baseline_score < 0.85:
            return 0.002   # 0.2% for moderate baselines
        elif baseline_score < 0.95:
            return 0.001   # 0.1% for strong baselines
        else:
            return 0.0005  # 0.05% for very strong baselines
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Tuple[float, float]:
        """
        Evaluate features using stratified k-fold cross-validation.
        
        Args:
            X: Feature dataframe
            y: Target series
            
        Returns:
            Tuple of (mean ROC-AUC, standard deviation)
        """
        # Prepare data
        X_prep, y_prep = self._prepare_data(X, y)
        
        # Create fresh model instance
        model = XGBClassifier(**self.model_params)
        
        # Stratified K-Fold with fixed random state
        cv = StratifiedKFold(
            n_splits=self.n_folds,
            shuffle=True,
            random_state=self.random_state
        )
        
        # Get cross-validation scores
        scores = cross_val_score(
            estimator=model,
            X=X_prep,
            y=y_prep,
            cv=cv,
            scoring='roc_auc',
            n_jobs=1  # Ensure determinism
        )
        
        return float(scores.mean()), float(scores.std())
    
    def should_accept_feature(
        self,
        baseline_score: float,
        new_score: float,
        baseline_std: float = 0.0,
        new_std: float = 0.0
    ) -> Tuple[bool, float, str]:
        """
        Determine if a feature should be accepted based on improvement.
        
        Args:
            baseline_score: ROC-AUC without the feature
            new_score: ROC-AUC with the feature
            baseline_std: Standard deviation of baseline
            new_std: Standard deviation with feature
            
        Returns:
            Tuple of (accept decision, improvement, reason)
        """
        improvement = new_score - baseline_score
        epsilon = self.get_epsilon(baseline_score)
        
        # Accept if improvement exceeds threshold
        if improvement > epsilon:
            reason = f"Improvement {improvement:.4f} > ε={epsilon:.4f}"
            return True, improvement, reason
        
        # Reject if performance degrades
        elif improvement < -epsilon/2:  # More tolerant of small decreases
            reason = f"Performance degraded by {-improvement:.4f}"
            return False, improvement, reason
        
        # Reject if no meaningful change
        else:
            reason = f"Improvement {improvement:.4f} ≤ ε={epsilon:.4f}"
            return False, improvement, reason
    
    def evaluate_delta(
        self,
        X_baseline: pd.DataFrame,
        X_enhanced: pd.DataFrame,
        y: pd.Series
    ) -> Tuple[bool, float, dict]:
        """
        Evaluate whether enhanced features improve over baseline.
        
        Args:
            X_baseline: Original features
            X_enhanced: Features with additions
            y: Target variable
            
        Returns:
            Tuple of (accept, improvement, metrics_dict)
        """
        # Evaluate baseline
        baseline_mean, baseline_std = self.evaluate(X_baseline, y)
        
        # Evaluate enhanced
        enhanced_mean, enhanced_std = self.evaluate(X_enhanced, y)
        
        # Make acceptance decision
        accept, improvement, reason = self.should_accept_feature(
            baseline_mean, enhanced_mean, baseline_std, enhanced_std
        )
        
        # Compile metrics
        metrics = {
            'baseline_roc': baseline_mean,
            'baseline_std': baseline_std,
            'enhanced_roc': enhanced_mean,
            'enhanced_std': enhanced_std,
            'improvement': improvement,
            'relative_improvement': improvement / baseline_mean if baseline_mean > 0 else 0,
            'epsilon': self.get_epsilon(baseline_mean),
            'accept': accept,
            'reason': reason
        }
        
        return accept, improvement, metrics
    
    def validate_holdout(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_holdout: pd.DataFrame,
        y_holdout: pd.Series
    ) -> float:
        """
        Train on full training set and evaluate on holdout.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_holdout: Holdout features
            y_holdout: Holdout target
            
        Returns:
            ROC-AUC score on holdout set
        """
        # Prepare data
        X_train_prep, y_train_prep = self._prepare_data(X_train, y_train)
        X_holdout_prep, y_holdout_prep = self._prepare_data(X_holdout, y_holdout)
        
        # Create fresh model and train
        model = XGBClassifier(**self.model_params)
        model.fit(X_train_prep, y_train_prep)
        
        # Predict probabilities
        y_pred_proba = model.predict_proba(X_holdout_prep)[:, 1]
        
        # Calculate ROC-AUC
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y_holdout_prep, y_pred_proba))