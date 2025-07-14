"""
SUPER CAAFE Feature Cache: Intelligent Feature Learning System
==============================================================

Implements in-context learning through intelligent caching of successful features.
Provides atomic operations, semantic matching, and cross-dataset learning.
"""

import json
import os
import hashlib
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
from dataclasses import dataclass, asdict


@dataclass
class FeatureMetadata:
    """Metadata for a successfully validated feature."""
    
    code: str
    description: str
    feature_name: str
    improvement: float
    baseline_roc: float
    enhanced_roc: float
    dataset_hash: str
    dataset_context: str
    dataset_size: int
    dataset_features: int
    timestamp: str
    feature_type: str
    complexity: str  # simple, moderate, complex
    
    @property
    def relative_improvement(self) -> float:
        """Calculate relative improvement percentage."""
        if self.baseline_roc > 0:
            return (self.improvement / self.baseline_roc) * 100
        return 0.0
    
    @property
    def impact_score(self) -> float:
        """
        Compute an impact score combining absolute and relative improvement.
        Higher scores indicate more valuable features.
        """
        # Weight absolute improvement more for weak baselines
        # Weight relative improvement more for strong baselines
        baseline_weight = 1 - self.baseline_roc
        relative_weight = self.baseline_roc
        
        return (baseline_weight * self.improvement * 10 + 
                relative_weight * self.relative_improvement / 10)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data['relative_improvement'] = self.relative_improvement
        data['impact_score'] = self.impact_score
        return data
    
    @classmethod
    def from_dict(cls, data: dict) -> 'FeatureMetadata':
        """Create from dictionary."""
        # Remove computed properties if present
        data.pop('relative_improvement', None)
        data.pop('impact_score', None)
        return cls(**data)


class FeatureCache:
    """
    Intelligent cache for storing and retrieving successful features.
    
    Implements:
    - Atomic file operations for crash safety
    - Dataset-specific caching with semantic matching
    - Cross-dataset pattern learning
    - In-context few-shot examples for LLMs
    """
    
    def __init__(
        self,
        cache_dir: str = ".caafe_cache",
        dataset_context: Optional[str] = None,
        dataset_columns: Optional[List[str]] = None
    ):
        """
        Initialize cache with optional dataset-specific storage.
        
        Args:
            cache_dir: Directory for cache files
            dataset_context: Description of the dataset
            dataset_columns: List of column names
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Determine cache file
        if dataset_context and dataset_columns:
            self.dataset_hash = self._compute_dataset_hash(dataset_context, dataset_columns)
            self.cache_file = self.cache_dir / f"features_{self.dataset_hash}.json"
            self.is_dataset_specific = True
        else:
            self.cache_file = self.cache_dir / "features_global.json"
            self.is_dataset_specific = False
            self.dataset_hash = "global"
        
        self.features: List[FeatureMetadata] = []
        self.load()
    
    def _compute_dataset_hash(self, context: str, columns: List[str]) -> str:
        """Generate a unique hash for dataset identification."""
        # Normalize and combine dataset identifiers
        canonical_cols = sorted([col.lower().strip() for col in columns])
        signature = f"{context.lower().strip()}|{'|'.join(canonical_cols)}"
        
        # Generate short hash
        return hashlib.sha256(signature.encode()).hexdigest()[:12]
    
    def _classify_feature_type(self, code: str, description: str) -> str:
        """Classify the type of feature transformation."""
        code_lower = code.lower()
        desc_lower = description.lower()
        
        # Check for different feature types
        if any(op in code for op in ['*', '/', '+', '-']) and code.count('df[') >= 2:
            if '/' in code:
                return 'ratio'
            elif '*' in code:
                return 'interaction'
            else:
                return 'arithmetic'
        elif 'kmeans' in code_lower or 'dbscan' in code_lower:
            return 'clustering'
        elif any(func in code_lower for func in ['log', 'sqrt', 'exp', '**']):
            return 'mathematical'
        elif 'groupby' in code_lower or 'transform' in code_lower:
            return 'aggregation'
        elif 'pd.cut' in code_lower or 'pd.qcut' in code_lower:
            return 'binning'
        elif any(time in desc_lower for time in ['time', 'date', 'hour', 'day', 'month']):
            return 'temporal'
        else:
            return 'other'
    
    def _assess_complexity(self, code: str) -> str:
        """Assess the complexity of the feature code."""
        lines = [l.strip() for l in code.split('\n') if l.strip() and not l.strip().startswith('#')]
        
        if len(lines) == 1:
            return 'simple'
        elif len(lines) <= 3:
            return 'moderate'
        else:
            return 'complex'
    
    def add_feature(
        self,
        code: str,
        description: str,
        feature_name: str,
        improvement: float,
        baseline_roc: float,
        enhanced_roc: float,
        dataset_context: str,
        dataset_size: int,
        dataset_features: int
    ) -> FeatureMetadata:
        """
        Add a validated feature to the cache.
        
        Args:
            code: Python code that generates the feature
            description: Human-readable description
            feature_name: Name of the generated column
            improvement: ROC-AUC improvement
            baseline_roc: Baseline model ROC-AUC
            enhanced_roc: Enhanced model ROC-AUC
            dataset_context: Dataset description
            dataset_size: Number of samples
            dataset_features: Number of features
            
        Returns:
            Created FeatureMetadata object
        """
        # Check for duplicates by code similarity
        code_hash = hashlib.md5(code.encode()).hexdigest()
        for existing in self.features:
            existing_hash = hashlib.md5(existing.code.encode()).hexdigest()
            if existing_hash == code_hash:
                # Update if this version performed better
                if improvement > existing.improvement:
                    self.features.remove(existing)
                    break
                else:
                    return existing
        
        # Create metadata
        metadata = FeatureMetadata(
            code=code,
            description=description,
            feature_name=feature_name,
            improvement=improvement,
            baseline_roc=baseline_roc,
            enhanced_roc=enhanced_roc,
            dataset_hash=self.dataset_hash,
            dataset_context=dataset_context,
            dataset_size=dataset_size,
            dataset_features=dataset_features,
            timestamp=datetime.now().isoformat(),
            feature_type=self._classify_feature_type(code, description),
            complexity=self._assess_complexity(code)
        )
        
        self.features.append(metadata)
        self.save()
        
        return metadata
    
    def get_relevant_features(
        self,
        dataset_context: str,
        baseline_roc: float,
        limit: int = 5,
        min_impact: float = 0.5
    ) -> List[FeatureMetadata]:
        """
        Get features relevant to the current problem.
        
        Args:
            dataset_context: Current dataset description
            baseline_roc: Current baseline performance
            limit: Maximum features to return
            min_impact: Minimum impact score
            
        Returns:
            List of relevant features sorted by impact
        """
        relevant = []
        context_words = set(dataset_context.lower().split())
        
        for feature in self.features:
            # Skip low-impact features
            if feature.impact_score < min_impact:
                continue
            
            # Calculate relevance score
            relevance = 0.0
            
            # Semantic similarity
            feature_words = set(feature.dataset_context.lower().split())
            overlap = len(context_words & feature_words) / max(len(context_words), 1)
            relevance += overlap * 0.3
            
            # Similar baseline performance (features that worked at similar difficulty)
            baseline_diff = abs(feature.baseline_roc - baseline_roc)
            if baseline_diff < 0.1:
                relevance += 0.3
            elif baseline_diff < 0.2:
                relevance += 0.1
            
            # Impact score contribution
            relevance += min(feature.impact_score / 10, 0.4)
            
            if relevance > 0.2:  # Minimum relevance threshold
                relevant.append((relevance, feature))
        
        # Sort by relevance and return top features
        relevant.sort(key=lambda x: x[0], reverse=True)
        return [f[1] for f in relevant[:limit]]
    
    def get_feature_patterns(self) -> Dict[str, List[str]]:
        """
        Extract successful patterns grouped by type.
        
        Returns:
            Dictionary mapping feature types to example descriptions
        """
        patterns = {}
        
        # Group by feature type
        for feature in self.features:
            if feature.impact_score > 1.0:  # Only successful features
                if feature.feature_type not in patterns:
                    patterns[feature.feature_type] = []
                
                # Add unique patterns
                pattern_desc = f"{feature.description} (Δ={feature.improvement:.3f})"
                if pattern_desc not in patterns[feature.feature_type]:
                    patterns[feature.feature_type].append(pattern_desc)
        
        # Limit to top 3 per type
        for feat_type in patterns:
            patterns[feat_type] = patterns[feat_type][:3]
        
        return patterns
    
    def generate_prompt_context(
        self,
        dataset_context: str,
        baseline_roc: float,
        current_features: List[str]
    ) -> str:
        """
        Generate context for LLM prompt based on cached knowledge.
        
        Args:
            dataset_context: Current dataset description
            baseline_roc: Current baseline performance
            current_features: Already generated feature names
            
        Returns:
            Formatted prompt context
        """
        context_parts = []
        
        # Get relevant successful features
        relevant = self.get_relevant_features(dataset_context, baseline_roc, limit=3)
        
        if relevant:
            context_parts.append("## Successful Patterns from Similar Problems\n")
            for i, feature in enumerate(relevant, 1):
                context_parts.append(
                    f"{i}. **{feature.feature_type.title()}**: {feature.description}\n"
                    f"   - Impact: {feature.relative_improvement:.1f}% relative improvement\n"
                    f"   - Baseline: {feature.baseline_roc:.3f} → {feature.enhanced_roc:.3f}\n"
                )
        
        # Add pattern summary if available
        patterns = self.get_feature_patterns()
        if patterns:
            context_parts.append("\n## Proven Feature Types\n")
            for feat_type, examples in patterns.items():
                if examples:
                    context_parts.append(f"- **{feat_type.title()}**: {examples[0]}\n")
        
        # Add guidance to avoid repetition
        if current_features:
            context_parts.append(f"\n## Already Generated (Do Not Repeat)\n")
            for feat in current_features[:5]:  # Show recent 5
                context_parts.append(f"- {feat}\n")
        
        return "".join(context_parts)
    
    def save(self) -> None:
        """Atomically save cache to disk."""
        cache_data = {
            'version': '2.0',
            'dataset_hash': self.dataset_hash,
            'updated': datetime.now().isoformat(),
            'features': [f.to_dict() for f in self.features]
        }
        
        # Write to temporary file first
        temp_fd, temp_path = tempfile.mkstemp(dir=self.cache_dir, suffix='.tmp')
        try:
            with os.fdopen(temp_fd, 'w') as f:
                json.dump(cache_data, f, indent=2)
            
            # Atomic rename
            os.replace(temp_path, self.cache_file)
        except Exception:
            # Clean up temp file on error
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise
    
    def load(self) -> None:
        """Load cache from disk."""
        if not self.cache_file.exists():
            return
        
        try:
            with open(self.cache_file, 'r') as f:
                cache_data = json.load(f)
            
            self.features = [
                FeatureMetadata.from_dict(f) 
                for f in cache_data.get('features', [])
            ]
        except Exception as e:
            print(f"Warning: Failed to load cache: {e}")
            self.features = []
    
    def merge_with_global_cache(self) -> None:
        """
        Merge the current cache with the global cache to access cross-dataset features.
        
        This method loads features from the global cache and adds them to the current
        cache instance for enhanced feature retrieval. Features from the dataset-specific
        cache take precedence over global cache features with the same code.
        """
        if not self.is_dataset_specific:
            return  # Already using global cache
        
        global_cache_file = self.cache_dir / "features_global.json"
        if not global_cache_file.exists():
            return
        
        try:
            with open(global_cache_file, 'r') as f:
                global_cache_data = json.load(f)
            
            global_features = [
                FeatureMetadata.from_dict(f) 
                for f in global_cache_data.get('features', [])
            ]
            
            # Create set of existing codes to avoid duplicates
            existing_codes = {hashlib.md5(f.code.encode()).hexdigest() for f in self.features}
            
            # Add global features that don't already exist
            for global_feature in global_features:
                global_code_hash = hashlib.md5(global_feature.code.encode()).hexdigest()
                if global_code_hash not in existing_codes:
                    self.features.append(global_feature)
            
        except Exception as e:
            print(f"Warning: Failed to merge with global cache: {e}")
    
    def get_statistics(self) -> dict:
        """Get cache statistics."""
        if not self.features:
            return {
                'total_features': 0,
                'datasets': 0,
                'avg_improvement': 0,
                'best_improvement': 0,
                'feature_types': {}
            }
        
        # Compute statistics
        improvements = [f.improvement for f in self.features]
        datasets = len(set(f.dataset_hash for f in self.features))
        
        # Count feature types
        type_counts = {}
        for f in self.features:
            type_counts[f.feature_type] = type_counts.get(f.feature_type, 0) + 1
        
        return {
            'total_features': len(self.features),
            'datasets': datasets,
            'avg_improvement': np.mean(improvements),
            'best_improvement': max(improvements),
            'avg_impact_score': np.mean([f.impact_score for f in self.features]),
            'feature_types': type_counts
        }