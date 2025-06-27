# src/caafe/critic.py – OPTIMIZED XGBoost Critic
"""
[OPTIMIZED] Critic module for *super_caafe*, using XGBoost for maximum stability and performance.

This version is a hybrid of best practices, designed to be a robust, drop-in replacement
for the TabPFN critic, especially on hardware where TabPFN is unstable (e.g., Apple Silicon).

Key characteristics
-------------------
* Uses **XGBoost >= 2.0** with `tree_method="hist"` for native Apple Silicon (arm64) performance.
* Deterministic, tuned hyperparameters chosen for 10k-100k row tables to avoid tuning in the loop.
* Implements a **manual K-fold cross-validation loop** for maximum stability and transparency,
  avoiding the high memory overhead of `cross_val_predict`.
* Parallel training within XGBoost is enabled via its internal `n_jobs` parameter.

Typical runtimes (M1 Pro)
-----------------------------------
| rows × cols | time/feature |
|-------------|-------------|
| 10k × 150   |  ~5-6 s      |
| 50k × 150   |  ~15-18 s    |
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold

try:
    from xgboost import XGBClassifier
    import xgboost as xgb
    # Check version compatibility
    xgb_version = tuple(map(int, xgb.__version__.split('.')[:2]))
    if xgb_version < (2, 0):
        raise ImportError(f"XGBoost version {xgb.__version__} detected. Please install XGBoost>=2.0 for optimal performance: `pip install xgboost>=2.0`")
except ModuleNotFoundError as exc:
    raise ImportError("XGBoost not installed. Please install with: `pip install xgboost>=2.0`") from exc

# ---------------------------------------------------------------------------
# Helpers --------------------------------------------------------------------
# ---------------------------------------------------------------------------

def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """[OPTIMIZED] One-pass numeric-ise & impute, with robust NaN handling for categoricals."""
    clean = df.copy()
    for c in clean.columns:
        if pd.api.types.is_numeric_dtype(clean[c]):
            clean[c] = clean[c].fillna(clean[c].median())
        else:
            # This maps NaNs to a new category (0), which is more robust
            # than collapsing them with the first category.
            clean[c] = pd.Categorical(clean[c]).codes + 1
    return clean.astype(np.float32)


def _clean_target(y: pd.Series) -> pd.Series:
    """[ROBUST] Cleans the target variable and ensures it is binary."""
    y = y.dropna()
    codes, _ = pd.factorize(y, sort=True)
    if codes.max() > 1:
        raise ValueError("Critic expects a binary target (0/1) after cleaning.")
    return pd.Series(codes, index=y.index, name=y.name)

# ---------------------------------------------------------------------------
# Critic class ----------------------------------------------------------------
# ---------------------------------------------------------------------------

@dataclass
class Critic:
    """ROC-AUC scorer based on a fixed, tuned XGBoost model."""
    folds: int = 3
    repeats: int = 2  
    holdout: float | None = None
    n_jobs: int = -1  # Use all available CPU cores by default
    seed: int = 42

    _model: XGBClassifier = field(init=False, repr=False)
    fold_scores: list = field(init=False, repr=False, default_factory=list)

    def __post_init__(self) -> None:
        if self.holdout is not None:
            assert 0 < self.holdout < 1, "holdout must be between 0 and 1"
        
        # Initialize feature importance storage
        self.feature_importances_ = []
        
        # [OPTIMIZED] Using well-tuned hyperparameters for speed and accuracy
        self._model = XGBClassifier(
            tree_method="hist",      # native ARM backend
            n_estimators=120, 
            max_depth=3, 
            learning_rate=0.20,
            gamma=2.0, 
            min_child_weight=5,
            subsample=0.80, 
            colsample_bytree=0.80,
            objective="binary:logistic", 
            eval_metric="auc",
            n_jobs=self.n_jobs,  # Use the parameter instead of hardcoded value
            random_state=42
        )
        print(f"✅ XGBoost critic ready (CPU hist) — folds={self.folds}, jobs={self.n_jobs}")

    def score(self, X: pd.DataFrame, y: pd.Series) -> float:
        """Cleans data and returns the out-of-sample ROC AUC score."""
        y_bin = _clean_target(y)
        X_sync = _clean_df(X.loc[y_bin.index])

        # Auto-switch to holdout for large datasets (≥10k rows) - use local flag, don't mutate instance
        use_holdout = len(X_sync) >= 10000 or self.holdout is not None
        if use_holdout:
            # Use configured holdout or default 0.2 for large datasets
            holdout_size = self.holdout if self.holdout is not None else 0.2
            return self._holdout_auc(X_sync, y_bin, holdout_size)
        return self._cv_auc(X_sync, y_bin)

    def score_delta(self, X: pd.DataFrame, y: pd.Series, baseline_auc: float) -> float:
        """Calculates the performance improvement over a baseline."""
        return self.score(X, y) - baseline_auc
    
    def get_feature_importance(self, X: pd.DataFrame, y: pd.Series, method: str = "gain") -> pd.DataFrame:
        """Get feature importance analysis after scoring.
        
        Args:
            X: Feature dataframe
            y: Target series
            method: Importance type ('gain', 'weight', 'cover')
            
        Returns:
            DataFrame with features and their importance scores
        """
        y_bin = _clean_target(y)
        X_sync = _clean_df(X.loc[y_bin.index])
        
        # Train model on full dataset for feature importance
        self._model.fit(X_sync.values, y_bin.values)
        
        # Get importance scores
        if method == "gain":
            importance_scores = self._model.feature_importances_
        else:
            importance_dict = self._model.get_booster().get_score(importance_type=method)
            # Map back to feature indices
            importance_scores = [importance_dict.get(f"f{i}", 0) for i in range(len(X_sync.columns))]
        
        # Store for later access
        self.feature_importances_ = importance_scores.copy() if hasattr(importance_scores, 'copy') else list(importance_scores)
        
        # Create DataFrame with feature names and importance
        importance_df = pd.DataFrame({
            'feature': X_sync.columns,
            'importance': importance_scores
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def analyze_feature_impact(self, X_baseline: pd.DataFrame, X_enhanced: pd.DataFrame, y: pd.Series) -> dict:
        """Analyze the impact of new features on model performance.
        
        Args:
            X_baseline: Original features
            X_enhanced: Features with new additions
            y: Target variable
            
        Returns:
            Dictionary with feature impact analysis
        """
        # Get baseline performance and importance
        baseline_score = self.score(X_baseline, y)
        baseline_importance = self.get_feature_importance(X_baseline, y)
        
        # Get enhanced performance and importance  
        enhanced_score = self.score(X_enhanced, y)
        enhanced_importance = self.get_feature_importance(X_enhanced, y)
        
        # Identify new features
        new_features = [col for col in X_enhanced.columns if col not in X_baseline.columns]
        
        # Get importance of new features
        new_feature_importance = enhanced_importance[
            enhanced_importance['feature'].isin(new_features)
        ].copy()
        
        return {
            'baseline_score': baseline_score,
            'enhanced_score': enhanced_score,
            'score_improvement': enhanced_score - baseline_score,
            'new_features': new_features,
            'new_feature_importance': new_feature_importance.to_dict('records'),
            'top_overall_features': enhanced_importance.head(10).to_dict('records'),
            'baseline_top_features': baseline_importance.head(10).to_dict('records')
        }

    def _cv_auc(self, X: pd.DataFrame, y: pd.Series) -> float:
        """[STABILITY] Uses RepeatedStratifiedKFold for more robust evaluation."""
        cv = RepeatedStratifiedKFold(n_splits=self.folds, n_repeats=self.repeats, random_state=self.seed)
        X_np, y_np = X.values, y.values
        oof_preds = np.zeros_like(y_np, dtype=float)
        self.fold_scores = []

        for i, (train_idx, test_idx) in enumerate(cv.split(X_np, y_np)):
            X_train, X_test = X_np[train_idx], X_np[test_idx]
            y_train, y_test = y_np[train_idx], y_np[test_idx]
            
            self._model.fit(X_train, y_train)
            probas = self._model.predict_proba(X_test)[:, 1]
            oof_preds[test_idx] = probas
            self.fold_scores.append(roc_auc_score(y_test, probas))
            
            # Store feature importance from first fold for analysis
            if i == 0:
                self.feature_importances_ = self._model.feature_importances_.tolist()

        return roc_auc_score(y_np, oof_preds)

    def _holdout_auc(self, X: pd.DataFrame, y: pd.Series, holdout_size: float = 0.2) -> float:
        """Calculates ROC AUC on a single train-test split."""
        X_train, X_test, y_train, y_test = train_test_split(
            X.values, y.values, test_size=holdout_size, random_state=self.seed, stratify=y.values
        )
        self._model.fit(X_train, y_train)
        prob = self._model.predict_proba(X_test)[:, 1]
        self.fold_scores = [roc_auc_score(y_test, prob)] # Keep API consistent
        
        # Store feature importance for analysis
        self.feature_importances_ = self._model.feature_importances_.tolist()
        
        return self.fold_scores[0]