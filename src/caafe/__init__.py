"""
Modern CAAFE: Streamlined LLM-Powered Feature Engineering

Clean implementation of Context-Aware Automated Feature Engineering
following the original methodology with modern improvements.

Usage:
    from caafe import CAAFE
    
    caafe = CAAFE(model="gpt-4o-mini", max_iterations=5)
    X_enhanced = caafe.generate_features(X, y, "Dataset description")
    
    # Or use the simple function interface
    from caafe import generate_features
    X_enhanced = generate_features(X, y, "Dataset description")
"""

__version__ = "0.1.0"

from .core import CAAFE, generate_features
from .evaluation import evaluate_dataset

__all__ = ["CAAFE", "generate_features", "evaluate_dataset"]