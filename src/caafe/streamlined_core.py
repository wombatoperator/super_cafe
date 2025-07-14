"""
SUPER CAAFE: Self-Improving Validation Framework for Automated Feature Engineering
==================================================================================

A production-ready implementation that treats automated feature engineering as a
validation problem: determining whether further feature engineering is worthwhile
for a given model and dataset.
"""

import ast
import os
import re
import json
import warnings
from typing import Optional, Dict, List, Tuple, Any
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif

from .critic import Critic
from .feature_cache import FeatureCache, FeatureMetadata

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)

# Try importing LLM libraries
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class SecurityValidator:
    """Validates generated code for safety."""
    
    # Safe built-ins that can't access filesystem or network
    SAFE_BUILTINS = {
        # Type conversions
        'int', 'float', 'str', 'bool', 'complex', 'bytes',
        # Data structures
        'list', 'tuple', 'dict', 'set', 'frozenset',
        # Math and logic
        'abs', 'round', 'min', 'max', 'sum', 'pow', 'divmod',
        'all', 'any', 'len', 'sorted', 'reversed',
        # Iteration helpers
        'range', 'enumerate', 'zip', 'map', 'filter',
        # Safe introspection
        'type', 'isinstance', 'hasattr', 'getattr',
        # Constants
        'None', 'True', 'False',
    }
    
    # Allowed AST node types
    ALLOWED_NODES = {
        # Structure
        ast.Module, ast.Interactive, ast.Expression, ast.FunctionDef,
        ast.Return, ast.Pass, ast.Expr,
        # Variables and assignment
        ast.Assign, ast.AugAssign, ast.AnnAssign,
        ast.Name, ast.Load, ast.Store, ast.Del,
        # Operators
        ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare,
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
        ast.LShift, ast.RShift, ast.BitOr, ast.BitXor, ast.BitAnd,
        ast.And, ast.Or, ast.Not, ast.Invert, ast.UAdd, ast.USub,
        ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
        ast.Is, ast.IsNot, ast.In, ast.NotIn,
        # Data access
        ast.Subscript, ast.Slice, ast.Index, ast.Attribute,
        # Control flow
        ast.If, ast.For, ast.While, ast.Break, ast.Continue,
        ast.IfExp, ast.comprehension,
        # Data structures
        ast.List, ast.Tuple, ast.Set, ast.Dict,
        ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp,
        # Function calls
        ast.Call, ast.keyword, ast.Starred,
        # Lambda functions and arguments
        ast.Lambda, ast.arguments, ast.arg,
        # Constants
        ast.Constant, ast.Num, ast.Str, ast.Bytes,
        ast.NameConstant, ast.Ellipsis,
    }
    
    @classmethod
    def validate(cls, code: str) -> Tuple[bool, Optional[str]]:
        """
        Validate code for security issues.
        
        Args:
            code: Python code to validate
            
        Returns:
            Tuple of (is_safe, error_message)
        """
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, f"Syntax error: {e}"
        
        # Check all nodes
        for node in ast.walk(tree):
            # Check node type
            if type(node) not in cls.ALLOWED_NODES:
                return False, f"Forbidden AST node: {type(node).__name__}"
            
            # Check for imports
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                return False, "Import statements are not allowed"
            
            # Check function calls
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                    # Block dangerous functions
                    if func_name in ['eval', 'exec', 'compile', 'open', '__import__']:
                        return False, f"Forbidden function: {func_name}"
        
        return True, None


class FeatureGenerator:
    """Handles LLM-based feature generation."""
    
    def __init__(
        self,
        provider: str = "gemini",
        model: str = "gemini-2.0-flash",
        api_key: Optional[str] = None,
        temperature: float = 0.4
    ):
        """
        Initialize feature generator.
        
        Args:
            provider: LLM provider ("gemini", "openai", etc.)
            model: Model name
            api_key: API key (uses environment variable if not provided)
            temperature: Generation temperature
        """
        self.provider = provider.lower()
        self.model = model
        self.temperature = temperature
        
        # Initialize based on provider
        if self.provider == "gemini":
            if not GEMINI_AVAILABLE:
                raise ImportError("google-generativeai not installed")
            
            api_key = api_key or os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("Gemini API key required")
            
            genai.configure(api_key=api_key)
            self.client = genai.GenerativeModel(model)
            
        elif self.provider == "openai":
            if not OPENAI_AVAILABLE:
                raise ImportError("openai not installed")
            
            api_key = api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key required")
            
            self.client = openai.OpenAI(api_key=api_key)
            
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    def _build_prompt(
        self,
        df_sample: pd.DataFrame,
        target_name: str,
        dataset_context: str,
        feature_importance: Dict[str, float],
        cache_context: str = "",
        iteration: int = 1,
        iteration_history: List[Dict] = None
    ) -> str:
        """Build the feature generation prompt."""
        
        # Column information with importance scores
        column_info = []
        for col in df_sample.columns:
            if col == target_name:
                continue
            
            dtype = df_sample[col].dtype
            nan_pct = df_sample[col].isna().mean() * 100
            importance = feature_importance.get(col, 0.0)
            
            # Sample values
            non_null = df_sample[col].dropna()
            if len(non_null) > 0:
                if pd.api.types.is_numeric_dtype(dtype):
                    samples = non_null.sample(min(3, len(non_null))).round(2).tolist()
                else:
                    samples = non_null.sample(min(3, len(non_null))).tolist()
            else:
                samples = ["All NaN"]
            
            column_info.append(
                f"{col} ({dtype}): MI={importance:.3f}, NaN={nan_pct:.1f}%, "
                f"Samples={samples}"
            )
        
        column_text = "\n".join(column_info)
        
        # Build iteration history context
        history_context = ""
        if iteration_history:
            history_context = "\n## PREVIOUS ITERATION RESULTS:\n"
            for i, hist in enumerate(iteration_history[-5:], 1):  # Last 5 iterations
                status = "✅ ACCEPTED" if hist['accepted'] else "❌ REJECTED"
                history_context += f"Attempt {len(iteration_history)-5+i}: {status}\n"
                history_context += f"  Feature: {hist.get('feature_name', 'N/A')}\n"
                history_context += f"  Method: {hist.get('method', 'N/A')}\n"
                history_context += f"  Hypothesis: {hist.get('hypothesis', 'N/A')[:80]}...\n"
                history_context += f"  Result: {hist.get('reason', 'N/A')}\n\n"
            
            history_context += "LEARN FROM THESE ATTEMPTS: Avoid similar approaches that failed. Build on successful patterns.\n"

        # Build prompt
        prompt = f"""You are a world-class data scientist specializing in ADVANCED feature engineering for binary classification.

DATASET CONTEXT: {dataset_context}
TARGET: Predict '{target_name}'
ITERATION: {iteration}

AVAILABLE COLUMNS:
{column_text}

FEATURE IMPORTANCE GUIDE:
Columns with higher Mutual Information (MI) scores contain more signal about the target.
Look for non-obvious interactions and domain-specific patterns.

{cache_context}

{history_context}

YOUR MISSION:
Create ONE extremely sophisticated, novel feature that goes beyond basic feature engineering.
We need ADVANCED techniques that capture complex, non-linear patterns in the data.

SOPHISTICATION REQUIREMENTS:
❌ NO basic ratios, simple interactions, or obvious aggregations
❌ NO features that are just column1 * column2 or column1 / column2
❌ NO simple binning or basic mathematical transforms

✅ YES to advanced techniques like:
- Multi-dimensional clustering with domain meaning
- Complex statistical measures (entropy, mutual information, percentile ranks)
- Sophisticated temporal patterns and trend analysis
- Advanced interaction detection (3+ way interactions, conditional relationships)
- Domain-specific transformations that capture business logic
- Statistical features that encode distributional properties
- Non-linear transformations that reveal hidden patterns

TECHNICAL REQUIREMENTS:
1. Use pandas/numpy operations only (no imports needed)
2. Handle NaN values gracefully
3. Create exactly one new column in 'df'
4. Code must be deterministic (set random_state=42 where applicable)
5. Feature must be semantically meaningful for the business domain

AVAILABLE TOOLS (pre-imported):
- pandas as pd
- numpy as np
- KMeans, DBSCAN (for clustering)
- StandardScaler (for normalization)

OUTPUT FORMAT:
```python
# HYPOTHESIS: [Deep scientific hypothesis about the business/domain reason this advanced feature matters]
# METHOD: [clustering|statistical|multi_interaction|temporal_advanced|distributional|domain_specific]
# COMPLEXITY: [complex|very_complex]
# NOVELTY: [Brief explanation of why this approach is sophisticated and non-obvious]

[Your advanced pandas code to create the feature]
```

Generate a sophisticated feature that captures complex, domain-specific patterns:
```python"""
        
        return prompt
    
    def generate(
        self,
        df_sample: pd.DataFrame,
        target_name: str,
        dataset_context: str,
        feature_importance: Dict[str, float],
        cache_context: str = "",
        iteration: int = 1,
        iteration_history: List[Dict] = None
    ) -> Tuple[str, Dict[str, str]]:
        """
        Generate a feature using the LLM.
        
        Returns:
            Tuple of (code, metadata_dict)
        """
        prompt = self._build_prompt(
            df_sample, target_name, dataset_context,
            feature_importance, cache_context, iteration, iteration_history
        )
        
        # Generate based on provider
        if self.provider == "gemini":
            response = self.client.generate_content(
                prompt,
                generation_config={
                    'temperature': self.temperature,
                    'max_output_tokens': 1000,
                }
            )
            generated_text = response.text
            
        elif self.provider == "openai":
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=1000
            )
            generated_text = response.choices[0].message.content
        
        # Extract code and metadata
        code, metadata = self._parse_response(generated_text)
        
        return code, metadata
    
    def _parse_response(self, text: str) -> Tuple[str, Dict[str, str]]:
        """Parse LLM response to extract code and metadata."""
        # Extract code block
        code_match = re.search(r'```(?:python)?\n(.*?)\n```', text, re.DOTALL)
        if not code_match:
            # Fallback: assume everything after first # is code
            lines = text.strip().split('\n')
            code_start = 0
            for i, line in enumerate(lines):
                if line.strip().startswith('#'):
                    code_start = i
                    break
            code = '\n'.join(lines[code_start:])
        else:
            code = code_match.group(1)
        
        # Extract metadata from comments
        metadata = {
            'hypothesis': '',
            'method': 'unknown',
            'complexity': 'moderate',
            'novelty': ''
        }
        
        for line in code.split('\n'):
            if '# HYPOTHESIS:' in line:
                metadata['hypothesis'] = line.split('# HYPOTHESIS:')[1].strip()
            elif '# METHOD:' in line:
                metadata['method'] = line.split('# METHOD:')[1].strip().lower()
            elif '# COMPLEXITY:' in line:
                metadata['complexity'] = line.split('# COMPLEXITY:')[1].strip().lower()
            elif '# NOVELTY:' in line:
                metadata['novelty'] = line.split('# NOVELTY:')[1].strip()
        
        # Clean code (remove metadata comments)
        clean_lines = []
        for line in code.split('\n'):
            if not any(marker in line for marker in ['# HYPOTHESIS:', '# METHOD:', '# COMPLEXITY:', '# NOVELTY:']):
                clean_lines.append(line)
        
        clean_code = '\n'.join(clean_lines).strip()
        
        return clean_code, metadata


class SuperCAAFE:
    """
    Main SUPER CAAFE implementation.
    
    A self-improving validation framework that determines whether
    further feature engineering is worthwhile for a given model.
    """
    
    def __init__(
        self,
        provider: str = "gemini",
        model: str = "gemini-2.0-flash",
        api_key: Optional[str] = None,
        max_iterations: int = 20,
        patience: int = 5,
        cache_dir: str = ".caafe_cache",
        random_state: int = 42
    ):
        """
        Initialize SUPER CAAFE.
        
        Args:
            provider: LLM provider
            model: Model name
            api_key: API key
            max_iterations: Maximum feature generation attempts
            patience: Stop after this many consecutive rejections
            cache_dir: Directory for feature cache
            random_state: Random seed for reproducibility
        """
        self.max_iterations = max_iterations
        self.patience = patience
        self.random_state = random_state
        
        # Initialize components
        self.critic = Critic(n_folds=3, adaptive_epsilon=True, random_state=random_state)
        self.generator = FeatureGenerator(provider, model, api_key)
        self.validator = SecurityValidator()
        
        # Cache will be initialized per dataset
        self.cache_dir = cache_dir
        self.cache: Optional[FeatureCache] = None
        
        # State tracking
        self.accepted_features: List[Dict[str, Any]] = []
        self.rejected_features: List[Dict[str, Any]] = []
        self.full_code = ""
        self.consecutive_rejections = 0
    
    def _detect_target_leakage(self, code: str, target_name: str) -> bool:
        """Check if code potentially leaks target information."""
        # Only check for direct references to actual target variable names
        patterns = [
            rf'\b{re.escape(target_name)}\b',  # Direct reference to target column
            rf'["\']]{re.escape(target_name)}["\']',  # String reference to target
            rf'\.{re.escape(target_name)}(?:\s|$|[^\w])',  # Attribute access to target
            r'\b(target|label|y_true|y_train|y_test|y_pred)\b',  # Common ML variable names
        ]
        
        # Remove comments before checking
        code_no_comments = re.sub(r'#.*$', '', code, flags=re.MULTILINE)
        
        for pattern in patterns:
            if re.search(pattern, code_no_comments, re.IGNORECASE):
                return True
        
        return False
    
    def _execute_feature_code(
        self,
        code: str,
        df: pd.DataFrame,
        target_name: str
    ) -> pd.DataFrame:
        """
        Safely execute feature generation code.
        
        Args:
            code: Python code to execute
            df: Input dataframe
            target_name: Target column name (for leak detection)
            
        Returns:
            Modified dataframe
        """
        # Validate security
        is_safe, error = self.validator.validate(code)
        if not is_safe:
            raise ValueError(f"Security validation failed: {error}")
        
        # Check for target leakage
        if self._detect_target_leakage(code, target_name):
            raise ValueError(f"Potential target leakage detected")
        
        # Prepare execution environment
        df_work = df.copy()
        
        # Safe namespace with limited builtins
        namespace = {
            '__builtins__': {k: __builtins__[k] for k in self.validator.SAFE_BUILTINS},
            'pd': pd,
            'np': np,
            'df': df_work,
        }
        
        # Add sklearn tools if available
        try:
            from sklearn.cluster import KMeans, DBSCAN
            from sklearn.preprocessing import StandardScaler
            namespace.update({
                'KMeans': KMeans,
                'DBSCAN': DBSCAN,
                'StandardScaler': StandardScaler,
            })
        except ImportError:
            pass
        
        # Execute code
        exec(code, namespace)
        
        # Return modified dataframe
        return namespace['df']
    
    def _compute_feature_importance(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Dict[str, float]:
        """Compute mutual information scores for features."""
        # Prepare data (handle categoricals)
        X_mi = X.copy()
        for col in X_mi.columns:
            if X_mi[col].dtype == 'object' or X_mi[col].dtype.name == 'category':
                X_mi[col] = pd.Categorical(X_mi[col]).codes
        
        # Compute MI scores
        mi_scores = mutual_info_classif(X_mi, y, random_state=self.random_state)
        
        # Create importance dict
        importance = {}
        for col, score in zip(X.columns, mi_scores):
            importance[col] = float(score)
        
        return importance
    
    def probe_performance_ceiling(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        dataset_context: str = ""
    ) -> Dict[str, Any]:
        """
        Main entry point: Probe whether feature engineering can improve the model.
        
        Args:
            X: Feature dataframe
            y: Target series
            dataset_context: Description of the dataset
            
        Returns:
            Dictionary with results and recommendations
        """
        print(f"SUPER CAAFE: Validating feature engineering potential")
        print(f"Dataset: {len(X)} samples, {len(X.columns)} features")
        print(f"Context: {dataset_context}\n")
        
        # Initialize cache for this dataset
        self.cache = FeatureCache(
            cache_dir=self.cache_dir,
            dataset_context=dataset_context,
            dataset_columns=list(X.columns)
        )
        
        # Split data
        X_dev, X_holdout, y_dev, y_holdout = train_test_split(
            X, y, test_size=0.3, random_state=self.random_state, stratify=y
        )
        
        # Get baseline performance
        baseline_score, baseline_std = self.critic.evaluate(X_dev, y_dev)
        print(f"Baseline ROC-AUC: {baseline_score:.4f} (±{baseline_std:.4f})")
        
        # Compute feature importance
        feature_importance = self._compute_feature_importance(X_dev, y_dev)
        
        # Initialize tracking
        self.accepted_features = []
        self.rejected_features = []
        self.full_code = ""
        self.consecutive_rejections = 0
        self.iteration_history = []  # Track all attempts for learning
        
        target_name = y.name or 'target'
        current_features = list(X.columns)
        best_score = baseline_score
        
        # Main iteration loop
        for iteration in range(1, self.max_iterations + 1):
            print(f"\n--- Iteration {iteration} ---")
            
            # Check patience
            if self.consecutive_rejections >= self.patience:
                print(f"Stopping: {self.patience} consecutive rejections")
                break
            
            # Generate cache context
            cache_context = self.cache.generate_prompt_context(
                dataset_context, best_score, 
                [f['name'] for f in self.accepted_features]
            )
            
            # Generate feature
            try:
                code, metadata = self.generator.generate(
                    X_dev.head(20),  # Sample for context
                    target_name,
                    dataset_context,
                    feature_importance,
                    cache_context,
                    iteration,
                    self.iteration_history
                )
                
                print(f"Generated: {metadata.get('method', 'unknown')} feature")
                print(f"Hypothesis: {metadata.get('hypothesis', 'None')[:100]}...")
                
            except Exception as e:
                print(f"Generation failed: {e}")
                continue
            
            # Execute and evaluate
            try:
                # Apply existing features + new feature
                X_enhanced = X_dev.copy()
                if self.full_code:
                    X_enhanced = self._execute_feature_code(self.full_code, X_enhanced, target_name)
                
                X_new = self._execute_feature_code(code, X_enhanced, target_name)
                
                # Find new column(s)
                new_cols = [col for col in X_new.columns if col not in X_enhanced.columns]
                if not new_cols:
                    print("No new features created")
                    self.consecutive_rejections += 1
                    continue
                
                # Evaluate
                accept, improvement, metrics = self.critic.evaluate_delta(
                    X_enhanced, X_new, y_dev
                )
                
                print(f"New feature: {new_cols[0]}")
                print(f"Performance: {metrics['enhanced_roc']:.4f} "
                      f"({improvement:+.4f} vs ε={metrics['epsilon']:.4f})")
                print(f"Decision: {metrics['reason']}")
                
            except Exception as e:
                print(f"Execution failed: {e}")
                self.consecutive_rejections += 1
                self.rejected_features.append({
                    'iteration': iteration,
                    'error': str(e),
                    'code': code
                })
                
                # Add to iteration history
                self.iteration_history.append({
                    'iteration': iteration,
                    'accepted': False,
                    'feature_name': 'execution_failed',
                    'method': metadata.get('method', ''),
                    'hypothesis': metadata.get('hypothesis', ''),
                    'novelty': metadata.get('novelty', ''),
                    'improvement': 0.0,
                    'reason': f"Execution error: {str(e)}"
                })
                continue
            
            # Handle acceptance/rejection
            if accept:
                self.consecutive_rejections = 0
                self.full_code = self.full_code + "\n\n" + code if self.full_code else code
                best_score = metrics['enhanced_roc']
                
                # Track accepted feature
                feature_info = {
                    'name': new_cols[0],
                    'iteration': iteration,
                    'improvement': improvement,
                    'code': code,
                    'metadata': metadata,
                    'metrics': metrics
                }
                self.accepted_features.append(feature_info)
                
                # Cache successful feature
                self.cache.add_feature(
                    code=code,
                    description=metadata.get('hypothesis', ''),
                    feature_name=new_cols[0],
                    improvement=improvement,
                    baseline_roc=metrics['baseline_roc'],
                    enhanced_roc=metrics['enhanced_roc'],
                    dataset_context=dataset_context,
                    dataset_size=len(X_dev),
                    dataset_features=len(X_dev.columns)
                )
                
                # Update feature importance for next iteration
                feature_importance = self._compute_feature_importance(X_new, y_dev)
                
                # Add to iteration history
                self.iteration_history.append({
                    'iteration': iteration,
                    'accepted': True,
                    'feature_name': new_cols[0],
                    'method': metadata.get('method', ''),
                    'hypothesis': metadata.get('hypothesis', ''),
                    'novelty': metadata.get('novelty', ''),
                    'improvement': improvement,
                    'reason': f"Accepted: {metrics['reason']}"
                })
                
            else:
                self.consecutive_rejections += 1
                self.rejected_features.append({
                    'iteration': iteration,
                    'reason': metrics['reason'],
                    'code': code
                })
                
                # Add to iteration history
                self.iteration_history.append({
                    'iteration': iteration,
                    'accepted': False,
                    'feature_name': new_cols[0] if new_cols else 'unknown',
                    'method': metadata.get('method', ''),
                    'hypothesis': metadata.get('hypothesis', ''),
                    'novelty': metadata.get('novelty', ''),
                    'improvement': improvement,
                    'reason': f"Rejected: {metrics['reason']}"
                })
        
        # Final holdout validation
        print(f"\n--- Holdout Validation ---")
        
        if self.accepted_features:
            # Apply all features to both training and holdout sets
            X_dev_enhanced = self._execute_feature_code(
                self.full_code, X_dev.copy(), target_name
            )
            X_holdout_base = X_holdout.copy()
            X_holdout_enhanced = self._execute_feature_code(
                self.full_code, X_holdout_base, target_name
            )
            
            # Evaluate on holdout
            holdout_base = self.critic.validate_holdout(
                X_dev, y_dev, X_holdout_base, y_holdout
            )
            holdout_enhanced = self.critic.validate_holdout(
                X_dev_enhanced, y_dev, X_holdout_enhanced, y_holdout
            )
            
            holdout_improvement = holdout_enhanced - holdout_base
            print(f"Holdout: {holdout_base:.4f} → {holdout_enhanced:.4f} "
                  f"({holdout_improvement:+.4f})")
            
        else:
            holdout_base = self.critic.validate_holdout(
                X_dev, y_dev, X_holdout, y_holdout
            )
            holdout_enhanced = holdout_base
            holdout_improvement = 0.0
            print(f"Holdout: {holdout_base:.4f} (no features added)")
        
        # Generate recommendation
        total_improvement = best_score - baseline_score
        recommendation = self._generate_recommendation(
            baseline_score, best_score, holdout_improvement,
            len(self.accepted_features), len(self.rejected_features)
        )
        
        # Prepare results
        results = {
            'recommendation': recommendation,
            'worth_further_investment': total_improvement > 0.005,
            'baseline_roc': baseline_score,
            'final_roc': best_score,
            'holdout_roc': holdout_enhanced,
            'total_improvement': total_improvement,
            'relative_improvement': (total_improvement / baseline_score * 100 
                                     if baseline_score > 0 else 0),
            'features_accepted': len(self.accepted_features),
            'features_rejected': len(self.rejected_features),
            'accepted_features': self.accepted_features,
            'final_code': self.full_code,
            'cache_stats': self.cache.get_statistics()
        }
        
        return results
    
    def _generate_recommendation(
        self,
        baseline: float,
        final: float,
        holdout_delta: float,
        n_accepted: int,
        n_rejected: int
    ) -> str:
        """Generate actionable recommendation based on results."""
        improvement = final - baseline
        
        if improvement < 0.001:
            return (
                "🎯 STRONG BASELINE CONFIRMED: No meaningful improvements found. "
                "The model is likely near its performance ceiling. "
                "Consider: (1) Collecting more training data, "
                "(2) Addressing class imbalance, or (3) Sourcing novel feature types."
            )
        
        elif improvement < 0.005:
            return (
                f"✓ MARGINAL GAINS: Found {n_accepted} features with +{improvement:.3f} ROC. "
                "While statistically significant, the practical impact is limited. "
                "Feature engineering has reached diminishing returns."
            )
        
        elif improvement < 0.01:
            return (
                f"📈 MODERATE SUCCESS: Added {n_accepted} features improving +{improvement:.3f} ROC. "
                f"Holdout confirms {holdout_delta:+.3f} gain. "
                "Further gains possible but may require domain expertise or external data."
            )
        
        else:
            return (
                f"🚀 SIGNIFICANT IMPROVEMENT: Generated {n_accepted} high-impact features "
                f"with +{improvement:.3f} ROC ({improvement/baseline*100:.1f}% relative). "
                f"Holdout validates {holdout_delta:+.3f} gain. "
                "Continue feature engineering - substantial headroom remains."
            )
    
    def apply_cached_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        dataset_context: str = "",
        dataset_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Apply the best cached features to a dataset."""
        # Use dataset columns from DataFrame if not provided
        if dataset_columns is None:
            dataset_columns = list(X.columns)
        
        # Initialize cache with dataset-specific mode if context/columns provided
        if dataset_context and dataset_columns:
            cache = FeatureCache(
                cache_dir=self.cache_dir,
                dataset_context=dataset_context,
                dataset_columns=dataset_columns
            )
            # Merge with global cache for cross-dataset learning
            cache.merge_with_global_cache()
        else:
            cache = FeatureCache(self.cache_dir)
        
        # Get baseline for reference
        baseline, _ = self.critic.evaluate(X, y)
        
        # Get relevant features
        features = cache.get_relevant_features(dataset_context, baseline, limit=5)
        
        if not features:
            print("No relevant cached features found")
            return X
        
        # Apply features
        X_enhanced = X.copy()
        applied = []
        
        target_name = y.name or 'target'
        
        for feature in features:
            try:
                X_new = self._execute_feature_code(feature.code, X_enhanced, target_name)
                
                # Check if feature was created
                new_cols = [col for col in X_new.columns if col not in X_enhanced.columns]
                if new_cols:
                    X_enhanced = X_new
                    applied.append(feature.feature_name)
                    print(f"Applied: {feature.feature_name} "
                          f"(expected Δ≈{feature.improvement:.3f})")
                    
            except Exception as e:
                print(f"Failed to apply {feature.feature_name}: {e}")
        
        print(f"\nApplied {len(applied)} cached features")
        return X_enhanced


# Convenience function
def validate_feature_engineering(
    X: pd.DataFrame,
    y: pd.Series,
    description: str = "",
    **kwargs
) -> Dict[str, Any]:
    """
    Validate whether feature engineering can improve model performance.
    
    Args:
        X: Feature dataframe
        y: Target series
        description: Dataset description
        **kwargs: Additional arguments for SuperCAAFE
        
    Returns:
        Results dictionary with recommendation
    """
    caafe = SuperCAAFE(**kwargs)
    return caafe.probe_performance_ceiling(X, y, description)