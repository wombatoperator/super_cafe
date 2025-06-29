
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold

try:
    from xgboost import XGBClassifier
except ModuleNotFoundError as exc:
    raise ImportError("XGBoost not installed – `pip install xgboost>=2.0`. ") from exc

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



@dataclass(slots=True)
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
        
        # [OPTIMIZED] Using well-tuned hyperparameters for speed and accuracy
        self._model = XGBClassifier(
            tree_method="hist",      # native ARM backend
            n_estimators=60, 
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

        # Auto-switch to holdout for large datasets (≥10k rows)
        if len(X_sync) >= 10000:
            self.holdout = 0.2
            return self._holdout_auc(X_sync, y_bin)
        elif self.holdout:
            return self._holdout_auc(X_sync, y_bin)
        return self._cv_auc(X_sync, y_bin)

    def score_delta(self, X: pd.DataFrame, y: pd.Series, baseline_auc: float) -> float:
        """Calculates the performance improvement over a baseline."""
        return self.score(X, y) - baseline_auc

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

        return roc_auc_score(y_np, oof_preds)

    def _holdout_auc(self, X: pd.DataFrame, y: pd.Series) -> float:
        """Calculates ROC AUC on a single train-test split."""
        X_train, X_test, y_train, y_test = train_test_split(
            X.values, y.values, test_size=self.holdout, random_state=self.seed, stratify=y.values
        )
        self._model.fit(X_train, y_train)
        prob = self._model.predict_proba(X_test)[:, 1]
        self.fold_scores = [roc_auc_score(y_test, prob)] # Keep API consistent
        return self.fold_scores[0]