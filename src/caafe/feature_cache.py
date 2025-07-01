"""
CAAFE Feature Cache System
==========================

Intelligent caching system that stores successful features and learns from past experiments
to make future feature generation more effective.
"""

import json
import os
import hashlib
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
from pathlib import Path
import portalocker
from tempfile import NamedTemporaryFile


class FeatureEntry:
    """Represents a single cached feature with its metadata and performance."""
    
    def __init__(
        self,
        code: str,
        description: str,
        feature_name: str,
        improvement_score: float,
        baseline_score: float,
        dataset_context: str,
        dataset_size: int,
        dataset_features: int,
        timestamp: Optional[str] = None,
        tags: Optional[List[str]] = None
    ):
        self.code = code
        self.description = description
        self.feature_name = feature_name
        self.improvement_score = improvement_score
        self.baseline_score = baseline_score
        self.dataset_context = dataset_context
        self.dataset_size = dataset_size
        self.dataset_features = dataset_features
        self.timestamp = timestamp or datetime.now().isoformat()
        self.tags = tags or []
        
        # Calculate derived metrics - [FIX] Stop squaring the reward
        self.relative_improvement = improvement_score / baseline_score if baseline_score > 0 else 0
        self.success_rank = self._calculate_success_rank()
        self.code_hash = hashlib.md5(code.encode()).hexdigest()[:8]
    
    def _calculate_success_rank(self) -> str:
        """Categorize the success level of this feature."""
        if self.improvement_score >= 0.02:
            return "exceptional"  # 2%+ improvement
        elif self.improvement_score >= 0.01:
            return "excellent"   # 1-2% improvement
        elif self.improvement_score >= 0.005:
            return "good"        # 0.5-1% improvement
        elif self.improvement_score >= 0.001:
            return "moderate"    # 0.1-0.5% improvement
        else:
            return "minimal"     # <0.1% improvement
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'code': self.code,
            'description': self.description,
            'feature_name': self.feature_name,
            'improvement_score': self.improvement_score,
            'baseline_score': self.baseline_score,
            'dataset_context': self.dataset_context,
            'dataset_size': self.dataset_size,
            'dataset_features': self.dataset_features,
            'timestamp': self.timestamp,
            'tags': self.tags,
            'relative_improvement': self.relative_improvement,
            'success_rank': self.success_rank,
            'code_hash': self.code_hash
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'FeatureEntry':
        """Create FeatureEntry from dictionary."""
        return cls(
            code=data['code'],
            description=data['description'],
            feature_name=data['feature_name'],
            improvement_score=data['improvement_score'],
            baseline_score=data['baseline_score'],
            dataset_context=data['dataset_context'],
            dataset_size=data['dataset_size'],
            dataset_features=data['dataset_features'],
            timestamp=data.get('timestamp'),
            tags=data.get('tags', [])
        )


class FeatureCache:
    """Intelligent cache for storing and retrieving successful features."""
    
    def __init__(self, cache_file: str = "caafe_feature_cache.json", dataset_context: str = None, dataset_columns: List[str] = None):
        # Generate dataset-specific cache file if context provided
        if dataset_context and dataset_columns:
            dataset_hash = self._generate_dataset_hash(dataset_context, dataset_columns)
            cache_dir = Path(cache_file).parent
            cache_name = f"caafe_cache_{dataset_hash}.json"
            self.cache_file = cache_dir / cache_name
            print(f"🗂️  Using dataset-specific cache: {cache_name}")
        else:
            self.cache_file = Path(cache_file)
        
        self.features: List[FeatureEntry] = []
        self.load_cache()
    
    def _generate_dataset_hash(self, dataset_context: str, dataset_columns: List[str]) -> str:
        """Generate a unique hash for the dataset based on context and columns."""
        # Create a consistent string representation
        columns_str = "_".join(sorted(dataset_columns))
        dataset_signature = f"{dataset_context}_{columns_str}_{len(dataset_columns)}"
        
        # Generate short hash
        hash_obj = hashlib.md5(dataset_signature.encode())
        return hash_obj.hexdigest()[:8]
    
    def add_feature(
        self,
        code: str,
        description: str,
        feature_name: str,
        improvement_score: float,
        baseline_score: float,
        dataset_context: str,
        dataset_size: int,
        dataset_features: int,
        tags: Optional[List[str]] = None
    ) -> FeatureEntry:
        """Add a new feature to the cache."""
        
        # Check for duplicates by code hash
        code_hash = hashlib.md5(code.encode()).hexdigest()[:8]
        for existing in self.features:
            if existing.code_hash == code_hash:
                print(f"⚠️  Feature with similar code already exists, updating performance...")
                # Update if this is a better performance
                if improvement_score > existing.improvement_score:
                    self.features.remove(existing)
                    break
                else:
                    return existing
        
        feature = FeatureEntry(
            code=code,
            description=description,
            feature_name=feature_name,
            improvement_score=improvement_score,
            baseline_score=baseline_score,
            dataset_context=dataset_context,
            dataset_size=dataset_size,
            dataset_features=dataset_features,
            tags=tags
        )
        
        self.features.append(feature)
        self.save_cache()
        
        print(f"💾 Cached feature '{feature_name}' with {improvement_score:+.4f} improvement ({feature.success_rank})")
        return feature
    
    def get_successful_features(
        self,
        min_improvement: float = 0.001,
        success_ranks: Optional[List[str]] = None,
        dataset_context: Optional[str] = None,
        limit: int = 10
    ) -> List[FeatureEntry]:
        """Get successful features matching criteria."""
        
        filtered = []
        for feature in self.features:
            # Filter by improvement threshold
            if feature.improvement_score < min_improvement:
                continue
            
            # Filter by success rank
            if success_ranks and feature.success_rank not in success_ranks:
                continue
            
            # Filter by dataset context (fuzzy matching)
            if dataset_context:
                context_lower = dataset_context.lower()
                feature_context_lower = feature.dataset_context.lower()
                
                # Check for common keywords
                context_keywords = set(context_lower.split())
                feature_keywords = set(feature_context_lower.split())
                
                if not context_keywords.intersection(feature_keywords):
                    continue
            
            filtered.append(feature)
        
        # Sort by improvement score (descending)
        filtered.sort(key=lambda x: x.improvement_score, reverse=True)
        
        return filtered[:limit]
    
    def get_top_universal_features(self, top_n: int = 5) -> List[FeatureEntry]:
        """Get the most universally successful features across different datasets."""
        
        # Group features by code similarity and take the best performer
        feature_groups = {}
        for feature in self.features:
            # Simple grouping by first few words of description
            group_key = ' '.join(feature.description.split()[:3]).lower()
            
            if group_key not in feature_groups:
                feature_groups[group_key] = feature
            elif feature.improvement_score > feature_groups[group_key].improvement_score:
                feature_groups[group_key] = feature
        
        # Sort by improvement and return top N
        top_features = sorted(feature_groups.values(), 
                            key=lambda x: x.improvement_score, reverse=True)
        
        return top_features[:top_n]
    
    def get_feature_patterns(self, min_success_rank: str = "good") -> Dict[str, List[str]]:
        """Extract common patterns from successful features."""
        
        rank_order = ["minimal", "moderate", "good", "excellent", "exceptional"]
        min_rank_idx = rank_order.index(min_success_rank)
        
        patterns = {
            "interaction_features": [],
            "mathematical_transforms": [],
            "aggregation_features": [],
            "time_features": [],
            "ratio_features": []
        }
        
        for feature in self.features:
            if rank_order.index(feature.success_rank) < min_rank_idx:
                continue
            
            code_lower = feature.code.lower()
            desc_lower = feature.description.lower()
            
            # Categorize feature types
            if "*" in feature.code or "interaction" in desc_lower:
                patterns["interaction_features"].append(feature.description)
            
            if any(op in code_lower for op in ["log", "sqrt", "**", "^"]):
                patterns["mathematical_transforms"].append(feature.description)
            
            if any(agg in code_lower for agg in ["groupby", "transform", "mean", "sum", "count"]):
                patterns["aggregation_features"].append(feature.description)
            
            if any(time_word in desc_lower for time_word in ["time", "hour", "day", "temporal"]):
                patterns["time_features"].append(feature.description)
            
            if "/" in feature.code or "ratio" in desc_lower:
                patterns["ratio_features"].append(feature.description)
        
        return patterns
    
    def generate_intelligent_prompt_addition(self, dataset_context: str, current_dataset_features: List[str] = None) -> str:
        """Generate enhanced prompt content based on cached features and avoid repetition."""
        
        if not self.features:
            return ""
        
        # Separate current dataset features from other datasets
        current_dataset_features_cache = [f for f in self.features if f.dataset_context == dataset_context]
        other_dataset_features = [f for f in self.features if f.dataset_context != dataset_context and f.improvement_score > 0.001]
        
        prompt_addition = "\n**🧠 INTELLIGENT CACHE GUIDANCE:**\n"
        
        # 1. Features already generated for THIS exact dataset
        if current_dataset_features_cache:
            prompt_addition += "**ALREADY GENERATED FOR THIS DATASET (DO NOT REPEAT):**\n"
            for feature in current_dataset_features_cache:
                feature_code = feature.code.split('\n')[-1]  # Get the actual code line
                prompt_addition += f"❌ AVOID: {feature.feature_name} = {feature_code}\n"
                prompt_addition += f"   (Already tested, improvement: {feature.improvement_score:+.4f})\n"
            prompt_addition += "\n**❗ CRITICAL: Do NOT generate any variation of the above features. They have already been tested.**\n\n"
        
        # 2. Successful patterns from other similar datasets (inspiration)
        if other_dataset_features:
            prompt_addition += "**SUCCESSFUL PATTERNS FROM OTHER DATASETS (for inspiration):**\n"
            unique_patterns = {}
            for feature in other_dataset_features[:5]:
                pattern_key = feature.feature_name.lower().replace('_', ' ')
                if pattern_key not in unique_patterns:
                    unique_patterns[pattern_key] = feature
            
            for feature in list(unique_patterns.values())[:3]:
                prompt_addition += f"✅ Pattern: {feature.description}\n"
                prompt_addition += f"   Type: {self._categorize_feature_type(feature)}\n"
                prompt_addition += f"   Improvement: {feature.improvement_score:+.4f}\n\n"
        
        # 3. Suggested new directions
        attempted_types = [self._categorize_feature_type(f) for f in current_dataset_features_cache]
        untried_types = [t for t in ["interaction", "ratio", "aggregation", "mathematical", "binning"] if t not in attempted_types]
        
        if untried_types:
            prompt_addition += f"**SUGGESTED NEW DIRECTIONS (not yet tried):**\n"
            for feat_type in untried_types[:3]:
                prompt_addition += f"💡 Try {feat_type} features\n"
            prompt_addition += "\n"
        
        prompt_addition += "**🎯 YOUR TASK: Generate a NEW type of feature that hasn't been tried yet on this dataset.**\n"
        
        return prompt_addition
    
    def _categorize_feature_type(self, feature: FeatureEntry) -> str:
        """Categorize a feature by its type for better guidance."""
        code_lower = feature.code.lower()
        desc_lower = feature.description.lower()
        
        if "+" in feature.code and any(x in desc_lower for x in ["family", "total", "sum"]):
            return "aggregation"
        elif "*" in feature.code or "interaction" in desc_lower:
            return "interaction"
        elif "/" in feature.code or "ratio" in desc_lower:
            return "ratio"
        elif any(op in code_lower for op in ["log", "sqrt", "**", "exp"]):
            return "mathematical"
        elif any(op in code_lower for op in ["cut", "qcut", "bin"]):
            return "binning"
        elif any(op in code_lower for op in ["groupby", "transform"]):
            return "grouping"
        else:
            return "other"
    
    def save_cache(self):
        """Save cache to JSON file with atomic write."""
        cache_data = {
            'version': '1.0',
            'created': datetime.now().isoformat(),
            'features': [feature.to_dict() for feature in self.features]
        }
        
        try:
            # [FIX] Atomic write to prevent corruption during parallel runs
            cache_dir = self.cache_file.parent if hasattr(self.cache_file, 'parent') else os.path.dirname(self.cache_file)
            with NamedTemporaryFile('w', delete=False, dir=cache_dir, suffix='.tmp') as tmp:
                json.dump(cache_data, tmp, indent=2)
                tmp.flush()
                os.fsync(tmp.fileno())
                temp_path = tmp.name
            
            # Atomic move
            os.replace(temp_path, self.cache_file)
        except Exception as e:
            print(f"⚠️  Failed to save feature cache: {e}")
            # Clean up temp file if it exists
            try:
                if 'temp_path' in locals():
                    os.unlink(temp_path)
            except:
                pass
    
    def load_cache(self):
        """Load cache from JSON file."""
        if not self.cache_file.exists():
            print(f"📂 No existing cache found, starting fresh: {self.cache_file}")
            return
        
        try:
            with open(self.cache_file, 'r') as f:
                cache_data = json.load(f)
            
            self.features = [
                FeatureEntry.from_dict(feature_data) 
                for feature_data in cache_data.get('features', [])
            ]
            
            print(f"📂 Loaded {len(self.features)} cached features from {self.cache_file}")
            
            # Print summary
            if self.features:
                success_counts = {}
                for feature in self.features:
                    success_counts[feature.success_rank] = success_counts.get(feature.success_rank, 0) + 1
                
                print(f"   Cache summary: {dict(success_counts)}")
                
        except Exception as e:
            print(f"⚠️  Failed to load feature cache: {e}")
            self.features = []
    
    def get_cache_stats(self) -> Dict:
        """Get statistics about the cache."""
        if not self.features:
            return {"total_features": 0}
        
        improvements = [f.improvement_score for f in self.features]
        success_ranks = [f.success_rank for f in self.features]
        
        return {
            "total_features": len(self.features),
            "avg_improvement": np.mean(improvements),
            "best_improvement": max(improvements),
            "success_distribution": {rank: success_ranks.count(rank) for rank in set(success_ranks)},
            "dataset_contexts": list(set(f.dataset_context for f in self.features))
        }
    
    def export_best_features(self, output_file: str = "best_features.py", min_rank: str = "good"):
        """Export the best features as a Python module for easy reuse."""
        
        rank_order = ["minimal", "moderate", "good", "excellent", "exceptional"]
        min_rank_idx = rank_order.index(min_rank)
        
        best_features = [
            f for f in self.features 
            if rank_order.index(f.success_rank) >= min_rank_idx
        ]
        
        best_features.sort(key=lambda x: x.improvement_score, reverse=True)
        
        with open(output_file, 'w') as f:
            f.write('"""\nCAFE Best Features Library\n')
            f.write(f'Generated on {datetime.now().isoformat()}\n')
            f.write(f'Contains {len(best_features)} high-performing features\n"""\n\n')
            f.write('import pandas as pd\nimport numpy as np\n\n')
            
            for i, feature in enumerate(best_features):
                f.write(f'def feature_{i+1}_{feature.feature_name.replace(" ", "_")}(df):\n')
                f.write(f'    """\n    {feature.description}\n')
                f.write(f'    Improvement: {feature.improvement_score:+.4f} ({feature.success_rank})\n')
                f.write(f'    Dataset: {feature.dataset_context}\n    """\n')
                
                # Clean up the code for function format
                code_lines = feature.code.strip().split('\n')
                for line in code_lines:
                    if line.strip() and not line.strip().startswith('#'):
                        f.write(f'    {line}\n')
                
                f.write(f'    return df\n\n')
        
        print(f"📄 Exported {len(best_features)} best features to {output_file}")