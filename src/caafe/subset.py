"""
Subset scanning for finding optimal feature combinations.
"""

import itertools
import warnings
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd

from .critic import Critic


class SubsetScanner:
    """
    Scans through feature subsets to find optimal combinations.
    
    Given a set of generated features, this scanner evaluates different
    combinations to find the subset that provides the best performance.
    """
    
    def __init__(
        self,
        critic: Optional[Critic] = None,
        max_combinations: int = 14,
        min_subset_size: int = 3,
        max_subset_size: int = 4
    ):
        """
        Initialize subset scanner.
        
        Args:
            critic: XGBoost critic for evaluation (creates new one if None)
            max_combinations: Maximum number of combinations to try (≤14 to keep workload manageable)
            min_subset_size: Minimum subset size to consider
            max_subset_size: Maximum subset size to consider
        """
        self.critic = critic or Critic()
        self.max_combinations = max_combinations
        self.min_subset_size = min_subset_size
        self.max_subset_size = max_subset_size
        self.scan_results = []
    
    def run_scan(
        self,
        df_baseline: pd.DataFrame,
        df_augmented: pd.DataFrame,
        y: pd.Series,
        extra_features: List[str]
    ) -> List[str]:
        """
        Run subset scan to find optimal feature combination.
        
        Args:
            df_baseline: Original DataFrame with baseline features
            df_augmented: Augmented DataFrame with all features
            y: Target variable
            extra_features: List of generated feature names to scan
            
        Returns:
            List of feature names in the best subset
        """
        if not extra_features:
            return []
        
        # Rank features by individual performance first
        ranked_features = self._rank_features_individually(
            df_baseline, df_augmented, y, extra_features
        )
        
        if not ranked_features:
            return []
        
        # Limit features to scan based on individual performance - top 4 only
        top_features = ranked_features[:min(4, len(ranked_features))]
        
        print(f"Scanning subsets from top {len(top_features)} features: {top_features}")
        
        best_auc = -np.inf
        best_subset = None
        combinations_tried = 0
        
        # Try different subset sizes (only 3- and 4-way combos)
        for k in range(self.min_subset_size, min(self.max_subset_size + 1, len(top_features) + 1)):
            if combinations_tried >= self.max_combinations:
                break
            
            print(f"Trying subsets of size {k}...")
            
            # Generate combinations
            for combo in itertools.combinations(top_features, k):
                if combinations_tried >= self.max_combinations:
                    break
                
                # Create subset DataFrame
                baseline_cols = df_baseline.columns.tolist()
                subset_cols = baseline_cols + list(combo)
                df_subset = df_augmented[subset_cols].copy()
                
                # Evaluate subset
                try:
                    auc = self.critic.score(df_subset, y)
                    
                    # Store result
                    result = {
                        'subset': list(combo),
                        'size': k,
                        'auc': auc,
                        'features': combo,
                        'fold_scores': self.critic.fold_scores.copy()
                    }
                    self.scan_results.append(result)
                    
                    # Check if this is the best so far
                    if auc > best_auc:
                        best_auc = auc
                        best_subset = combo
                        print(f"  New best: {combo} -> AUC: {auc:.4f}")
                    
                    combinations_tried += 1
                    
                except Exception as e:
                    warnings.warn(f"Error evaluating subset {combo}: {e}")
                    continue
        
        print(f"Subset scan completed. Tried {combinations_tried} combinations.")
        
        if best_subset:
            print(f"Best subset: {best_subset} with AUC: {best_auc:.4f}")
            return list(best_subset)
        else:
            print("No beneficial subset found, returning top individual feature")
            return [ranked_features[0]] if ranked_features else []
    
    def _rank_features_individually(
        self,
        df_baseline: pd.DataFrame,
        df_augmented: pd.DataFrame,
        y: pd.Series,
        extra_features: List[str]
    ) -> List[str]:
        """
        Rank features by their individual contribution to performance.
        
        Args:
            df_baseline: Original DataFrame
            df_augmented: Augmented DataFrame
            y: Target variable
            extra_features: List of feature names to rank
            
        Returns:
            List of feature names ranked by performance (best first)
        """
        baseline_auc = self.critic.score(df_baseline, y)
        feature_scores = []
        
        print(f"Ranking {len(extra_features)} features individually...")
        
        for feature in extra_features:
            try:
                # Create DataFrame with baseline + this feature
                df_with_feature = df_baseline.copy()
                df_with_feature[feature] = df_augmented[feature]
                
                # Evaluate performance
                auc = self.critic.score(df_with_feature, y)
                delta = auc - baseline_auc
                
                feature_scores.append((feature, delta, auc))
                print(f"  {feature}: Δ={delta:+.4f} (AUC: {auc:.4f})")
                
            except Exception as e:
                warnings.warn(f"Error evaluating feature {feature}: {e}")
                continue
        
        # Sort by delta (improvement) descending
        feature_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return only features with positive contribution
        positive_features = [f[0] for f in feature_scores if f[1] > 0]
        
        print(f"Found {len(positive_features)} beneficial features out of {len(extra_features)}")
        
        return positive_features
    
    def get_scan_summary(self) -> Dict[str, Any]:
        """
        Get summary of the last scan.
        
        Returns:
            Dictionary with scan statistics
        """
        if not self.scan_results:
            return {"total_combinations": 0, "best_auc": None}
        
        best_result = max(self.scan_results, key=lambda x: x['auc'])
        
        return {
            "total_combinations": len(self.scan_results),
            "best_auc": best_result['auc'],
            "best_subset": best_result['subset'],
            "best_subset_size": best_result['size'],
            "auc_distribution": {
                "min": min(r['auc'] for r in self.scan_results),
                "max": max(r['auc'] for r in self.scan_results),
                "mean": np.mean([r['auc'] for r in self.scan_results]),
                "std": np.std([r['auc'] for r in self.scan_results])
            },
            "subset_sizes_tried": list(set(r['size'] for r in self.scan_results))
        }
    
    def get_top_subsets(self, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Get the top k performing subsets.
        
        Args:
            top_k: Number of top subsets to return
            
        Returns:
            List of subset results sorted by AUC (best first)
        """
        if not self.scan_results:
            return []
        
        sorted_results = sorted(self.scan_results, key=lambda x: x['auc'], reverse=True)
        return sorted_results[:top_k]
    
    def export_results(self) -> pd.DataFrame:
        """
        Export scan results as DataFrame for analysis.
        
        Returns:
            DataFrame with all scan results
        """
        if not self.scan_results:
            return pd.DataFrame()
        
        df_data = []
        for result in self.scan_results:
            df_data.append({
                'subset_size': result['size'],
                'auc': result['auc'],
                'features': ', '.join(result['features']),
                'feature_count': len(result['features']),
                'mean_fold_score': np.mean(result['fold_scores']) if result['fold_scores'] else None,
                'std_fold_score': np.std(result['fold_scores']) if result['fold_scores'] else None
            })
        
        return pd.DataFrame(df_data).sort_values('auc', ascending=False)


def run_subset_scan(
    df_baseline: pd.DataFrame,
    df_augmented: pd.DataFrame,
    y: pd.Series,
    extra_features: List[str],
    critic: Optional[Critic] = None
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Convenience function to run subset scan.
    
    Args:
        df_baseline: Original DataFrame
        df_augmented: Augmented DataFrame with new features
        y: Target variable
        extra_features: List of generated feature names
        critic: Optional XGBoost critic (creates new one if None)
        
    Returns:
        Tuple of (best_subset_features, scan_summary)
    """
    scanner = SubsetScanner(critic=critic)
    best_subset = scanner.run_scan(df_baseline, df_augmented, y, extra_features)
    summary = scanner.get_scan_summary()
    
    return best_subset, summary


def evaluate_feature_combinations(
    df: pd.DataFrame,
    y: pd.Series,
    feature_combinations: List[List[str]],
    critic: Optional[Critic] = None
) -> List[Tuple[List[str], float]]:
    """
    Evaluate specific feature combinations.
    
    Args:
        df: DataFrame with all features
        y: Target variable
        feature_combinations: List of feature combinations to evaluate
        critic: Optional XGBoost critic
        
    Returns:
        List of (feature_combination, auc_score) tuples
    """
    if critic is None:
        critic = Critic()
    
    results = []
    
    for combo in feature_combinations:
        try:
            df_subset = df[combo].copy()
            auc = critic.score(df_subset, y)
            results.append((combo, auc))
        except Exception as e:
            warnings.warn(f"Error evaluating combination {combo}: {e}")
            results.append((combo, 0.0))
    
    # Sort by AUC descending
    results.sort(key=lambda x: x[1], reverse=True)
    
    return results