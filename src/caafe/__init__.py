"""
CAAFE: Context-Aware Automated Feature Engineering
==================================================

A powerful Python library that uses Large Language Models (LLMs) to automatically 
generate high-quality features for machine learning models.

Main implementation: SUPER CAAFE
- Modern validation-focused approach
- Determines if feature engineering is worthwhile
- Adaptive thresholds based on baseline strength
- Intelligent caching and learning
- Production-ready security validation

Usage:
    # Main API (recommended)
    from caafe import validate_feature_engineering
    
    results = validate_feature_engineering(
        X, y,
        description="Customer churn prediction dataset", 
        provider="openai"
    )
    
    # Class-based API for advanced usage
    from caafe import SuperCAAFE
    
    caafe = SuperCAAFE(provider="openai", model="gpt-4o")
    results = caafe.probe_performance_ceiling(X, y, description)
"""

__version__ = "2.0.0"
__author__ = "CAAFE Contributors"

# Main CAAFE implementation (streamlined_core)
from .streamlined_core import (
    SuperCAAFE,
    validate_feature_engineering,
    FeatureGenerator,
    SecurityValidator
)

# Core components
from .critic import Critic
from .feature_cache import FeatureCache, FeatureMetadata

__all__ = [
    # Main interfaces
    "validate_feature_engineering",  # Main convenience function
    "SuperCAAFE",                   # Main CAAFE class
    
    # Core components
    "Critic",
    "FeatureCache", 
    "FeatureMetadata",
    "FeatureGenerator",
    "SecurityValidator",
]