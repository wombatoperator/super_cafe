"""
Feature generation loop for super_caafe with XGBoost-based evaluation.
"""

import hashlib
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .critic import Critic
from .llm_interface import LLMInterface
from .manifest import Manifest
from .sandbox import SafeExecutor, SandboxError


class FeatureLoop:
    """
    Main feature generation loop for super_caafe.
    
    Implements the iterative LLM → Sandbox → Critic → Feedback cycle
    with XGBoost-based evaluation and comprehensive tracking.
    """
    
    def __init__(
        self,
        llm_cfg: Dict,
        loop_cfg: Dict,
        cache_dir: Optional[Path] = None
    ):
        """
        Initialize feature loop.
        
        Args:
            llm_cfg: LLM configuration (model, temperature, etc.)
            loop_cfg: Loop configuration (iterations, thresholds, etc.)
            cache_dir: Directory for caching LLM responses
        """
        # Initialize components
        self.llm = LLMInterface(
            model=llm_cfg.get("model", "gpt-4o"),
            api_key=llm_cfg.get("api_key")
        )
        self.critic = Critic(
            folds=loop_cfg.get("cv_folds", 3),
            repeats=loop_cfg.get("cv_repeats", 2),
            holdout=loop_cfg.get("holdout", None),
        )
        self.executor = SafeExecutor(
            cpu_seconds=loop_cfg.get("cpu_seconds", 3),
            mem_mb=loop_cfg.get("mem_mb", 500)
        )
        
        # Configuration
        self.iter_max = loop_cfg.get("iter_max", 5)
        self.max_consec_fails = loop_cfg.get("max_consec_fails", 2)
        self.delta_thresh = loop_cfg.get("delta_thresh", 0.003)
        self.temperature = llm_cfg.get("temperature", 0.2)
        self.max_tokens = llm_cfg.get("max_tokens", 600)
        self.seed = llm_cfg.get("seed", 1234)
        
        # Cache setup
        self.cache_dir = cache_dir
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # State
        self.manifest = Manifest()
        self.baseline_auc = None
        self.current_df = None
        self.accepted_features = []
    
    def generate(
        self,
        df: pd.DataFrame,
        y: pd.Series,
        dataset_description: str
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        [UPGRADED] Main entry point for feature generation with improved feedback and logic.
        """
        print(f"Starting super_caafe feature generation...")
        print(f"Dataset: {len(df)} rows, {len(df.columns)} features")
        print(f"Target: {y.name} ({y.nunique()} unique values)")
        
        # Initialize state
        self.current_df = df.copy()
        self.accepted_features = []
        
        # Calculate baseline performance
        print("\nCalculating baseline performance...")
        start_time = time.time()
        self.baseline_auc = self.critic.score(self.current_df, y)
        baseline_time = time.time() - start_time
        print(f"Baseline AUC: {self.baseline_auc:.4f} (took {baseline_time:.1f}s)")
        
        # Main generation loop
        failure_count = 0
        accepted_count = 0
        last_feedback = ""
        
        # [NEW] Stop after 3 accepted features OR 2 consecutive fails  
        while accepted_count < 3 and failure_count < self.max_consec_fails:
            iteration = self.manifest.get_summary_stats().get("total_iterations", 0) + 1
            print(f"\n{'='*50}")
            print(f"ITERATION {iteration} (Accepted: {accepted_count}, Fails: {failure_count})")
            print(f"{'='*50}")
            
            # Generate feature candidate
            try:
                candidate = self._prompt(
                    cols=list(self.current_df.columns),
                    dataset_description=dataset_description,
                    feedback=last_feedback
                )
                
                if not candidate or not candidate.get("code"):
                    print("No valid candidate generated, skipping iteration")
                    failure_count += 1
                    last_feedback = "The last response was not a valid JSON with a 'code' key. Please try again."
                    continue
                
                print(f"Generated candidate:")
                print(f"Code: {candidate['code'][:100]}...")
                print(f"Explanation: {candidate['explanation']}")
                
            except Exception as e:
                print(f"LLM generation failed: {e}")
                failure_count += 1
                last_feedback = f"LLM error: {str(e)}"
                continue
            
            # Execute code safely
            try:
                start_time = time.time()
                new_feature = self.executor.run(candidate["code"], self.current_df)
                execution_time = time.time() - start_time
                
                print(f"Code executed successfully in {execution_time:.2f}s")
                print(f"Generated feature: {new_feature.name}")
                
            except SandboxError as e:
                print(f"Code execution failed: {e.short_msg}")
                self.manifest.add_iteration(iteration=iteration, code=candidate["code"], explanation=candidate["explanation"], status="failed", error_message=e.short_msg)
                last_feedback = f"Code rejected: {e.short_msg}. Please generate a new, valid function."
                failure_count += 1
                continue
            
            # Evaluate feature
            try:
                df_with_feature = self.current_df.copy()
                df_with_feature[new_feature.name] = new_feature
                
                start_time = time.time()
                current_baseline = self.critic.score(self.current_df, y) # Re-score baseline each time
                new_auc = self.critic.score(df_with_feature, y)
                eval_time = time.time() - start_time
                
                delta_auc = new_auc - current_baseline
                
                print(f"Evaluation completed in {eval_time:.2f}s")
                print(f"Baseline AUC for this round: {current_baseline:.4f}")
                print(f"AUC with new feature: {new_auc:.4f}")
                print(f"Delta AUC: {delta_auc:+.4f}")

                self.manifest.add_iteration(
                    iteration=iteration, code=candidate["code"], explanation=candidate["explanation"], delta_auc=delta_auc, 
                    fold_scores=getattr(self.critic, 'fold_scores', []), feature_name=new_feature.name, 
                    execution_time=execution_time + eval_time, status="evaluated"
                )
                
            except Exception as e:
                print(f"Feature evaluation failed: {e}")
                self.manifest.add_iteration(iteration=iteration, code=candidate["code"], explanation=candidate["explanation"], feature_name=new_feature.name, execution_time=execution_time, status="failed", error_message=str(e))
                last_feedback = f"Evaluation failed: {str(e)}. Please try again."
                failure_count += 1
                continue
            
            # [UPGRADED] Accept or reject feature
            # Use the delta threshold (0.003) to account for single-fold noise
            if delta_auc > self.delta_thresh: 
                # Accept feature
                print(f"✅ ACCEPTED: {new_feature.name} (Δ={delta_auc:+.4f})")
                self.current_df[new_feature.name] = new_feature
                self.accepted_features.append(new_feature.name)
                
                last_feedback = f"Feature '{new_feature.name}' was accepted with Δ_AUC={delta_auc:+.4f}. Please propose a new, different feature."
                failure_count = 0  # Reset failure count on success
                accepted_count += 1

            else:
                # Reject feature and provide specific feedback
                print(f"❌ REJECTED: Δ={delta_auc:+.4f}")
                # [NEW] Extract features used in rejected code to give better feedback
                import re
                used_cols = re.findall(r"df\[['\"](.*?)['\"]\]", candidate['code'])
                if used_cols:
                    last_feedback = f"The feature using columns {list(set(used_cols))} was rejected with Δ_AUC={delta_auc:+.4f}. Try a completely different combination of columns or a different type of transformation (e.g., a ratio, polynomial, or aggregation)."
                else:
                    last_feedback = f"The last feature was rejected with Δ_AUC={delta_auc:+.4f}. Try an alternative logic."
                
                failure_count += 1
        
        # Final summary
        print(f"\n{'='*50}")
        print(f"GENERATION COMPLETE")
        print(f"{'='*50}")
        print(f"Total Iterations: {iteration}")
        print(f"Accepted features: {len(self.accepted_features)}")
        print(f"Final AUC with new features: {self.critic.score(self.current_df, y):.4f}")
        
        if self.accepted_features:
            print(f"Generated features: {self.accepted_features}")
        
        return self.current_df, self.accepted_features
    
    def _prompt(
        self,
        cols: List[str],
        dataset_description: str,
        feedback: str = ""
    ) -> Optional[Dict]:
        """
        Generate feature candidate using LLM.
        
        Args:
            cols: Current column names
            dataset_description: Dataset description
            feedback: Feedback from previous iteration
            
        Returns:
            Dictionary with 'code' and 'explanation' keys, or None if failed
        """
        # Build prompt
        prompt = self._build_prompt(cols, dataset_description, feedback)
        
        # Check cache
        if self.cache_dir:
            cache_key = hashlib.sha256(prompt.encode()).hexdigest()
            cache_file = self.cache_dir / f"{cache_key[:8]}.json"
            
            if cache_file.exists():
                print(f"Using cached response: {cache_key[:8]}")
                with open(cache_file, 'r') as f:
                    return json.load(f)
        
        # Generate response
        try:
            response = self.llm.client.chat.completions.create(
                model=self.llm.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                seed=self.seed,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            result = json.loads(content)
            
            # Cache response
            if self.cache_dir and cache_file:
                with open(cache_file, 'w') as f:
                    json.dump(result, f, indent=2)
            
            return result
            
        except Exception as e:
            print(f"LLM prompt failed: {e}")
            return None
    
    def _build_prompt(
        self,
        cols: List[str],
        dataset_description: str,
        feedback: str
    ) -> str:
        """Build prompt for feature generation."""
        base_prompt = f"""You are engineering features for a Gradient-Boosted Tree (XGBoost, depth ≤ 3). Prefer log/ratio/z-score transforms; avoid high-degree products.

TASK: Propose exactly ONE new feature as a valid Python function named `make_feature(df)` that returns a `pd.Series` aligned to `df.index`.

DATASET: {dataset_description}
EXISTING_COLS: {', '.join(cols)}
ALLOWED_IMPORTS: pandas as pd, numpy as np

{f"FEEDBACK: {feedback}" if feedback else ""}

Return **only** JSON:
{{
 "code": "<python code>",
 "explanation": "<why this helps>"
}}

Focus on creating features that will improve XGBoost performance:
- Log/ratio/z-score transforms that trees can exploit
- Non-linear interactions that create clean splits
- Aggregations and statistical features  
- Handle missing values appropriately
- Create features with meaningful variance"""

        return base_prompt
    
    def _accept_feature(self, delta_auc: float) -> bool:
        """
        Decide whether to accept a feature based on performance improvement.
        
        Args:
            delta_auc: AUC improvement
            
        Returns:
            True if feature should be accepted
        """
        return delta_auc > self.delta_thresh
    
    def get_manifest(self) -> Manifest:
        """Get the current manifest."""
        return self.manifest
    
    def get_generation_summary(self) -> Dict:
        """Get summary of the generation process."""
        return self.manifest.get_summary_stats()


# Configuration helpers
def create_default_llm_config(
    model: str = "gpt-4o",
    temperature: float = 0.2,
    max_tokens: int = 600,
    seed: int = 1234,
    api_key: Optional[str] = None
) -> Dict:
    """Create default LLM configuration."""
    return {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "seed": seed,
        "api_key": api_key,
        "device": "auto"
    }


def create_default_loop_config(
    iter_max: int = 5,
    max_consec_fails: int = 2,
    delta_thresh: float = 0.003,
    cv_folds: int = 3,
    cv_repeats: int = 2,
    holdout: float = None,
    cpu_seconds: int = 3,
    mem_mb: int = 500
) -> Dict:
    """Create default loop configuration."""
    return {
        "iter_max": iter_max,
        "max_consec_fails": max_consec_fails,
        "delta_thresh": delta_thresh,
        "cv_folds": cv_folds,
        "cv_repeats": cv_repeats,
        "holdout": holdout,
        "cpu_seconds": cpu_seconds,
        "mem_mb": mem_mb
    }


# Convenience function for simple usage
def generate_features(
    df: pd.DataFrame,
    y: pd.Series,
    dataset_description: str,
    **kwargs
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Generate features using super_caafe.
    
    Args:
        df: Input DataFrame
        y: Target Series
        dataset_description: Dataset description
        **kwargs: Configuration overrides
        
    Returns:
        Tuple of (enhanced_dataframe, feature_names)
    """
    llm_cfg = create_default_llm_config(**{k: v for k, v in kwargs.items() if k in ["model", "temperature", "max_tokens", "seed", "api_key"]})
    loop_cfg = create_default_loop_config(**{k: v for k, v in kwargs.items() if k in ["iter_max", "max_consec_fails", "delta_thresh", "cv_folds", "cv_repeats", "holdout", "cpu_seconds", "mem_mb"]})
    
    loop = FeatureLoop(llm_cfg, loop_cfg)
    return loop.generate(df, y, dataset_description)