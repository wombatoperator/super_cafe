"""
Modern CAAFE: Streamlined LLM-Powered Feature Engineering for XGBoost

A modern implementation focused on XGBoost optimization with 
streamlined architecture and practical performance.

Usage:
    # Simple usage (backward compatible)
    from caafe import CAAFE
    caafe = CAAFE()
    X_enhanced = caafe.generate_features(X, y, "Dataset description")
    
    # New modern architecture
    from caafe.feature_loop import generate_features
    X_enhanced, features = generate_features(X, y, "Dataset description")
"""

__version__ = "0.1.0"

# Backward compatibility
from .core import CAAFE, generate_features as legacy_generate_features

# New modern architecture components
from .feature_loop import FeatureLoop, generate_features
from .critic import Critic
from .sandbox import SafeExecutor, SandboxError
from .manifest import Manifest, ManifestManager
from .subset import SubsetScanner, run_subset_scan

__all__ = [
    "CAAFE", "legacy_generate_features",  # Legacy compatibility
    "FeatureLoop", "generate_features",   # New architecture
    "Critic", "SafeExecutor", "SandboxError",
    "Manifest", "ManifestManager", 
    "SubsetScanner", "run_subset_scan"
]