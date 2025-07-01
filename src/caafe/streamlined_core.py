"""
CAAFE: Context-Aware Automated Feature Engineering - Streamlined Gemini Implementation
Clean, optimized implementation focusing exclusively on Gemini 2.5 Pro.
"""

import pandas as pd
import numpy as np
import os
import ast
import re
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from .critic import Critic, _clean_df
from .feature_cache import FeatureCache
from typing import Optional
from dotenv import load_dotenv

# Gemini imports
try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Load environment variables
load_dotenv()


def check_ast(node):
    """
    Security validation for AST nodes to prevent dangerous operations.
    Based on CodeB's check_ast() implementation.
    """
    # Allowed node types for safe pandas/numpy operations
    ALLOWED_NODES = {
        ast.Module, ast.Expr, ast.Assign, ast.AugAssign, ast.Name, ast.Load, ast.Store,
        ast.Attribute, ast.Subscript, ast.Slice, ast.Call, ast.keyword,
        ast.BinOp, ast.UnaryOp, ast.Compare, ast.BoolOp, ast.IfExp,
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow, ast.FloorDiv,
        ast.And, ast.Or, ast.Not, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
        ast.Is, ast.IsNot, ast.In, ast.NotIn,
        ast.BitAnd, ast.BitOr, ast.BitXor,  # Bitwise operators for pandas boolean operations
        ast.Constant,  # Literals (covers old Num, Str, NameConstant in Python 3.8+)
        # Backward compatibility for Python < 3.8
        getattr(ast, 'Num', type(None)), getattr(ast, 'Str', type(None)), getattr(ast, 'NameConstant', type(None)),
        ast.List, ast.Tuple, ast.Dict, ast.Set,  # Containers
        ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp,  # Comprehensions
    }
    
    # Forbidden imports and operations
    FORBIDDEN_NAMES = {
        'import', 'exec', 'eval', '__import__', 'open', 'file', 'input', 'raw_input',
        'compile', 'reload', 'vars', 'locals', 'globals', 'dir', 'help', 'quit', 'exit',
        'os', 'sys', 'subprocess', 'shutil', 'urllib', 'requests', 'socket',
    }
    
    if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
        raise ValueError("Import statements are not allowed for security reasons")
    
    if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
        raise ValueError(f"Forbidden operation: {node.id}")
    
    if isinstance(node, ast.Call):
        # Check function calls
        if isinstance(node.func, ast.Name):
            if node.func.id in FORBIDDEN_NAMES:
                raise ValueError(f"Forbidden function call: {node.func.id}")
        elif isinstance(node.func, ast.Attribute):
            # Check for dangerous method calls
            full_name = ""
            temp = node.func
            while isinstance(temp, ast.Attribute):
                full_name = "." + temp.attr + full_name
                temp = temp.value
            if isinstance(temp, ast.Name):
                full_name = temp.id + full_name
                
            # Block dangerous methods
            if any(danger in full_name.lower() for danger in ['system', 'popen', 'shell', 'exec']):
                raise ValueError(f"Forbidden method call: {full_name}")
    
    if type(node) not in ALLOWED_NODES:
        raise ValueError(f"Forbidden AST node type: {type(node).__name__}")
    
    # Recursively check all child nodes
    for child in ast.iter_child_nodes(node):
        check_ast(child)


def build_gemini_prompt(df, target_name, description, cache_intelligence=""):
    """Build optimized prompt specifically for Gemini 2.5 Pro with cache intelligence."""

    # Create dataframe WITHOUT target column to prevent data leakage
    df_features_only = df.drop(columns=[target_name] if target_name in df.columns else [])

    # Generate column info with samples
    samples = ""
    # Get a sample, handling cases with fewer than 10 rows
    sample_size = min(10, len(df_features_only))
    df_sample = df_features_only.head(sample_size)
    for col in df_sample.columns:
        nan_freq = f"{df_features_only[col].isna().mean() * 100:.1f}"
        sample_values = df_sample[col].tolist()

        if np.issubdtype(df_features_only[col].dtype, np.floating):
            sample_values = [round(sample, 2) for sample in sample_values]

        samples += f"{col} ({df_features_only[col].dtype}): NaN-freq [{nan_freq}%], Samples {sample_values}\n"

    # Gemini-optimized prompt following original CAAFE methodology
    base_prompt = f"""You are an expert data scientist and researcher specializing in identifying subtle, non-obvious patterns in data. Your task is to generate a single, highly predictive feature to improve a model's ability to predict "{target_name}".

The dataframe 'df' is loaded and in memory.
Dataset Description: "{description}"

The model has already been enhanced with basic interaction features. Your goal is to find something more creative.

**Available columns in 'df':**
{samples}**Your Thought Process (Chain of Thought):**
1.  **Hypothesize:** State a novel hypothesis about a hidden grouping or relationship in the data.
2.  **Choose Method:** Select the best method: a mathematical transformation (ratio, polynomial) OR a clustering algorithm (KMeans, DBSCAN) to test your hypothesis.
3.  **Explain Usefulness:** Justify why this feature adds new semantic information.
4.  **Implement:** Provide the Python code.

**RULES:**
-   Your code must be a block of Python creating a new column on the 'df' DataFrame.
-   **CRITICAL: NO IMPORT STATEMENTS ALLOWED! All tools are pre-loaded in your environment.**
-   **Available tools (already imported): `KMeans`, `DBSCAN`, `StandardScaler`, `pd`, `np`**
-   When clustering, always apply `StandardScaler` to the data first.
-   Your code should handle fitting the models and assigning the resulting labels to a new column.
-   Example for clustering (NO imports needed):
    `scaler = StandardScaler()`
    `features_to_cluster = df[['Age', 'Fare']]`
    `scaled_features = scaler.fit_transform(features_to_cluster)`
    `kmeans = KMeans(n_clusters=4, random_state=42, n_init='auto')`
    `df['Geo_Cluster'] = kmeans.fit_predict(scaled_features)`
-   **DO NOT** use import statements like `from sklearn.cluster import KMeans` - these tools are already available!
-   **DO NOT** generate simple sum/product features that have already been created.

**Code Formatting:**
```python
# HYPOTHESIS: (Your brief hypothesis about a hidden grouping)
# METHOD: (e.g., "KMeans Clustering" or "Ratio Transformation")
# USEFULNESS: (Your explanation of why this feature is valuable)
# Input samples: (Samples of columns used, e.g. 'Age': [22.0, 38.0, 26.0])
(Your block of pandas/sklearn code to generate the new column)
```end

**Your Task:**
Generate one new feature based on your expert analysis.

{cache_intelligence}

Codeblock:
"""
    
    return base_prompt


def encode_categorical_columns(df):
    """Automatically encode categorical/object columns for XGBoost compatibility.
    
    NOTE: We skip encoding here since the Critic's _clean_df() will handle it.
    This prevents double-encoding which was causing features to be corrupted.
    """
    # Let the Critic handle categorical encoding to prevent double-encoding issues
    return df


class FeaturePipeline:
    """Lightweight pipeline for caching fitted transforms across train/validation splits."""
    
    def __init__(self, code: str, target_column: str = None):
        self.code = code
        self.target_column = target_column
        self.fitted_objects = {}
        self.is_fitted = False
    
    def fit(self, df: pd.DataFrame) -> 'FeaturePipeline':
        """Fit any transforms in the code on training data."""
        if not self.code.strip():
            self.is_fitted = True
            return self
        
        # Execute code and cache any fitted objects
        self._execute_code(df, fit_mode=True)
        self.is_fitted = True
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply cached transforms to new data."""
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before transform")
        
        if not self.code.strip():
            return df
        
        return self._execute_code(df, fit_mode=False)
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(df).transform(df)
    
    def _execute_code(self, df: pd.DataFrame, fit_mode: bool = True) -> pd.DataFrame:
        """Execute code with caching support."""
        # Create a copy of the dataframe WITHOUT the target column to prevent data leakage
        df_safe = df.copy()
        if self.target_column and self.target_column in df_safe.columns:
            df_safe = df_safe.drop(columns=[self.target_column])
        
        # Enhanced target leakage detection - check before execution
        if self.target_column:
            # Strip comments before checking for target column name to avoid false positives
            clean_code = re.sub(r"#.*?$", "", self.code, flags=re.MULTILINE)
            
            # Use word boundaries to avoid false positives with substrings
            # Look for target column used as a pandas column access pattern
            target_patterns = [
                rf"\['{self.target_column}'\]",  # df['target_col']
                rf'\["{self.target_column}"\]',  # df["target_col"]
                rf"\.{self.target_column}\b",    # df.target_col (if valid identifier)
                rf"\b{re.escape(self.target_column)}\b(?=\s*[=,\)\]])"  # target_col as variable/parameter
            ]
            target_patterns += [
                rf"loc\[[^\]]*,\s*['\"]" + re.escape(self.target_column) + rf"['\"]\]",  # df.loc[:, 'target']
                rf"iloc\[[^\]]*\]\s*\.?\s*" + re.escape(self.target_column) + rf"?",  # positional slices
            ]
            
            for pattern in target_patterns:
                if re.search(pattern, clean_code):
                    raise ValueError(f"Data leakage detected: code attempts to use target column '{self.target_column}'")
        
        # Check for drop operations (discourage dropping columns)
        if '.drop(' in self.code or 'drop(' in self.code:
            print("⚠️ Warning: Code contains drop operations which may cause validation issues")
        
        # AST security validation
        try:
            parsed = ast.parse(self.code, mode="exec")
            check_ast(parsed)
        except (SyntaxError, ValueError) as e:
            raise ValueError(f"Code security validation failed: {e}")
        
        # Define freq_encode helper for safe frequency encoding
        def freq_encode(col):
            """Helper function for safe frequency encoding."""
            vc = df_safe[col].value_counts()
            return df_safe[col].map(vc)
        
        # Prepare local variables with cached objects
        local_vars = {
            'df': df_safe,
            'pd': pd,
            'np': np,
            'KMeans': KMeans,
            'DBSCAN': DBSCAN,
            'StandardScaler': StandardScaler,
            '_fitted_objects': self.fitted_objects if not fit_mode else {},
        }
        
        safe_builtins = {
            'int': int, 'float': float, 'str': str, 'bool': bool,
            'len': len, 'range': range, 'abs': abs, 'max': max, 
            'min': min, 'sum': sum, 'round': round,
            'freq_encode': freq_encode,  # Add frequency encoding helper
        }
        
        try:
            exec(compile(parsed, "", "exec"), {"__builtins__": safe_builtins, "pd": pd, "np": np}, local_vars)
            result_df = local_vars['df']
            
            # ---- AUTO-CACHE ESTIMATORS ----------------------------------
            if fit_mode:
                for name, obj in local_vars.items():
                    if hasattr(obj, "fit") and hasattr(obj, "transform"):
                        # mark as fitted once; store in pipeline cache
                        self.fitted_objects[name] = obj
            else:
                # forbid re-fitting on new data
                for obj in self.fitted_objects.values():
                    def _no_fit(*_args, **_kwargs):  # shadow .fit
                        raise RuntimeError("Refit blocked – transformer cached from train split")
                    obj.fit = _no_fit
            
            # Cache fitted objects in fit mode
            if fit_mode:
                self.fitted_objects = local_vars.get('_fitted_objects', {})
            
            # Automatically encode any new categorical columns created by the code
            result_df = encode_categorical_columns(result_df)
            
            # Check for problematic values that could crash XGBoost
            for col in result_df.columns:
                if pd.api.types.is_numeric_dtype(result_df[col]):
                    # Only check numeric columns for inf/large values
                    if np.isinf(result_df[col]).any():
                        raise ValueError(f"Column '{col}' contains infinite values (likely division by zero)")
                    if result_df[col].isna().all():
                        raise ValueError(f"Column '{col}' contains all NaN values")
                    # Check for extremely large values that could cause numerical issues
                    if result_df[col].abs().max() > 1e10:
                        raise ValueError(f"Column '{col}' contains extremely large values")
                    # Check for constant columns (no variance)
                    if result_df[col].nunique() <= 1:
                        raise ValueError(f"Column '{col}' has no variance (constant values)")
            
            return result_df
            
        except Exception as e:
            # Re-raise with more context
            raise ValueError(f"Code execution error: {str(e)}")


def execute_code_safely(code, df, target_column=None):
    """Execute LLM code safely without access to target column."""
    if not code.strip():
        return df
    
    # Use FeaturePipeline for consistent execution
    pipeline = FeaturePipeline(code, target_column)
    return pipeline.fit_transform(df)
    


class StreamlinedCAAFE:
    """Streamlined Context-Aware Automated Feature Engineering - Gemini Only Implementation with Intelligence Cache."""
    
    def __init__(
        self,
        model: str = "gemini-2.5-pro",
        api_key: Optional[str] = None,
        max_iterations: int = 5,
        n_splits: int = 3,
        n_repeats: int = 2,
        scorer=None,  # Optional scorer object with .score(X, y) method
        use_cache: bool = True,
        cache_file: str = "caafe_feature_cache.json"
    ):
        """
        Initialize Streamlined CAAFE with Intelligence Cache.
        
        Args:
            model: Gemini model to use (default: gemini-2.5-pro)
            api_key: Google AI API key (if not set via GEMINI_API_KEY env var)
            max_iterations: Number of feature generation attempts
            n_splits: Cross-validation splits
            n_repeats: Cross-validation repeats
            scorer: Optional scorer object with .score(X, y) method
            use_cache: Whether to use intelligent feature caching
            cache_file: Path to cache file
        """
        if not GEMINI_AVAILABLE:
            raise ImportError("google-genai package not installed. Install with: pip install google-genai")
        
        self.model = model
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.max_iterations = max_iterations
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        
        if not self.api_key:
            raise ValueError("Gemini API key not provided. Set GEMINI_API_KEY environment variable or pass api_key parameter.")
        
        # Use optimized Critic by default (matching original CAAFE design)
        self.scorer = scorer or Critic(folds=n_splits, repeats=n_repeats, n_jobs=-1)
        
        # Store cache settings for later initialization with dataset info
        self.use_cache = use_cache
        self.cache_file = cache_file
        self.cache = None  # Will be initialized in generate_features with dataset info
        
        self.generated_features = []
        self.full_code = ""
        self.messages = []
        self.consecutive_rejections = 0
        self.MAX_REJECTIONS_BEFORE_META_PROMPT = 2
    
    def _generate_gemini_code(self, messages):
        """Generate code using Gemini 2.5 Pro with structured output."""
        # Set environment variable for Gemini client
        os.environ['GEMINI_API_KEY'] = self.api_key
        
        client = genai.Client()
        
        # Extract key info from the original prompt (user message, not system message)
        original_prompt = ""
        for msg in messages:
            if msg['role'] == 'user' and len(msg['content']) > 500:  # Find the main user prompt
                original_prompt = msg['content']
                break
        
        if not original_prompt and messages:
            original_prompt = messages[-1]['content']  # Fallback to last message
        
        # Extract cache intelligence from the prompt
        cache_intelligence = ""
        if "🧠 INTELLIGENT CACHE GUIDANCE:" in original_prompt:
            cache_start = original_prompt.find("🧠 INTELLIGENT CACHE GUIDANCE:")
            cache_end = original_prompt.find("Codeblock:", cache_start)
            if cache_end > cache_start:
                cache_intelligence = original_prompt[cache_start:cache_end].strip()
        
        # Extract dataset description and sample count from original prompt
        description_start = original_prompt.find('Dataset Description: "') + 22
        description_end = original_prompt.find('"', description_start)
        description = original_prompt[description_start:description_end] if description_start > 21 else "Feature engineering task"
        
        # Extract sample count for more accurate context
        sample_count_match = re.search(r'(\d+)\s+samples', original_prompt)
        sample_count = sample_count_match.group(1) if sample_count_match else "Unknown"
        
        # Extract column information from the original prompt more reliably
        column_lines = []
        
        # Find the section with available columns
        sections = original_prompt.split("**Available columns in 'df':**")
        if len(sections) > 1:
            # Extract the section after "Available columns"
            columns_section = sections[1]
            # Split on next major section (like "Your Thought Process")
            if "**Your Thought Process" in columns_section:
                columns_section = columns_section.split("**Your Thought Process")[0]
            elif "**RULES:**" in columns_section:
                columns_section = columns_section.split("**RULES:**")[0]
            
            # Extract column lines
            lines = columns_section.split('\n')
            for line in lines:
                line = line.strip()
                # Look for pattern: "column_name (dtype): NaN-freq [x%], Samples [values]"
                if ':' in line and 'NaN-freq' in line and 'Samples' in line:
                    column_lines.append(line)
        
        # Fallback: look through all lines for column definitions
        if not column_lines:
            for line in original_prompt.split('\n'):
                line = line.strip()
                if re.match(r'^\w+.*\([^)]+\):\s*NaN-freq.*Samples', line):
                    column_lines.append(line)
        
        if column_lines:
            columns_info = '\n'.join(column_lines[:10])  # Limit for better context
        else:
            columns_info = "Available columns in dataset"
        
        print(f"🧠 Extracted {len(column_lines)} column definitions from prompt")
        
        # Create comprehensive Gemini prompt following original CAAFE methodology
        gemini_prompt = f"""The dataframe 'df' is loaded and in memory. Columns are also named attributes.

Description of the dataset in 'df' (column dtypes might be inaccurate):
"{description}"

Columns in 'df' (true feature dtypes listed here, categoricals encoded as int):
{columns_info}

This code was written by an expert data scientist working to improve predictions. It is a snippet of code that adds new columns to the dataset.

Number of samples (rows) in training dataset: {sample_count}

This code generates additional columns that are useful for a downstream classification algorithm (such as XGBoost) predicting the target variable.

Additional columns add new semantic information, that is they use real world knowledge on the dataset. They can e.g. be feature combinations, transformations, aggregations where the new column is a function of the existing columns.

The scale of columns and offset does not matter. Make sure all used columns exist. Follow the above description of columns closely and consider the data types and meanings of classes.

IMPORTANT: Focus on ADDING new columns only. Do NOT drop existing columns as this creates validation issues. The goal is to enhance the dataset with additional predictive features while keeping all original columns.

The classifier will be trained on the dataset with the generated columns and evaluated on a holdout set. The evaluation metric is ROC-AUC (area under the ROC curve). The best performing code will be selected.

You have access to advanced tools for sophisticated feature engineering:
- Standard operations: pd (pandas), np (numpy)
- Clustering: KMeans, DBSCAN (already imported, no import needed)
- Preprocessing: StandardScaler (already imported, no import needed)
- Mathematical functions: All numpy functions (log, sqrt, exp, sin, cos, etc.)

Generate sophisticated features that capture complex patterns:
- Mathematical transformations (ratios, polynomials, logarithms)
- Statistical aggregations and interactions
- Clustering-based features using domain knowledge
- Feature combinations that reveal hidden relationships

Each code block generates exactly one useful column. Focus on creating NEW features, not replacing existing ones.

{cache_intelligence}

Important: DO NOT use import statements - all tools are pre-loaded. Use column names exactly as provided above."""
        
        # Log the full prompt being sent to Gemini
        print("\n" + "="*80)
        print("🔍 FULL PROMPT SENT TO GEMINI:")
        print("="*80)
        print(gemini_prompt)
        print("="*80 + "\n")
        
        # Define structured output schema following original CAAFE format
        feature_schema = {
            "type": "object",
            "properties": {
                "feature_name_and_description": {
                    "type": "string",
                    "description": "Brief name and description of the feature"
                },
                "usefulness": {
                    "type": "string",
                    "description": "Description why this adds useful real world knowledge to classify the target according to dataset description and attributes"
                },
                "input_samples": {
                    "type": "string", 
                    "description": "Three samples of the columns used in the following code, e.g. 'column1': [val1, val2, val3], 'column2': [val4, val5, val6]"
                },
                "code": {
                    "type": "string",
                    "description": "Pandas code using the available columns to add exactly ONE new column to df. Do NOT drop any existing columns. Only create new features."
                }
            },
            "required": ["feature_name_and_description", "usefulness", "input_samples", "code"]
        }
        
        # Use Gemini 2.5 Pro with structured output
        response = client.models.generate_content(
            model=self.model,
            contents=gemini_prompt,
            config=types.GenerateContentConfig(
                # Reduced thinking budget to leave room for output
                thinking_config=types.ThinkingConfig(thinking_budget=512),
                temperature=0.4,
                max_output_tokens=1500,
                response_mime_type="application/json",
                response_schema=feature_schema
            )
        )
        
        # Parse structured JSON response
        response_text = response.text or ""
        
        # Log the full response from Gemini
        print("\n" + "="*80)
        print("📥 FULL RESPONSE FROM GEMINI:")
        print("="*80)
        print(response_text)
        print("="*80 + "\n")
        
        print(f"🧠 Structured response: {response_text}")
        
        if response_text:
            try:
                import json
                feature_data = json.loads(response_text)
                
                # Extract structured fields following original CAAFE format
                feature_name_desc = feature_data.get('feature_name_and_description', 'New Feature')
                usefulness = feature_data.get('usefulness', 'Adds predictive value')
                input_samples = feature_data.get('input_samples', 'Various columns used')
                code_block = feature_data.get('code', '')
                
                # Format exactly like original CAAFE prompt format
                formatted_code = f"""# ({feature_name_desc})
# Usefulness: ({usefulness})
# Input samples: ({input_samples})
{code_block}"""
                
                print(f"🧠 Formatted feature code: {formatted_code}")
                return formatted_code
                
            except json.JSONDecodeError as e:
                print(f"🧠 JSON parsing failed: {e}")
                # Fallback to original text if JSON parsing fails
                return response_text
        
        return response_text
    
    def generate_features(self, X: pd.DataFrame, y: pd.Series, description: str = "") -> pd.DataFrame:
        """
        Generate features using streamlined CAAFE methodology with Gemini.
        
        Args:
            X: Input features
            y: Target variable
            description: Dataset description for the LLM
            
        Returns:
            DataFrame with original + generated features
        """
        print(f"*Dataset description:* {description}")
        print(f"Starting CAAFE with {len(X)} samples, {len(X.columns)} features\n")
        
        # Initialize dataset-specific cache
        if self.use_cache and self.cache is None:
            dataset_columns = list(X.columns)
            self.cache = FeatureCache(
                cache_file=self.cache_file,
                dataset_context=description[:100],  # Use first 100 chars of description
                dataset_columns=dataset_columns
            )
            if self.cache:
                stats = self.cache.get_cache_stats()
                if stats.get("total_features", 0) > 0:
                    print(f"🧠 Loaded dataset-specific cache with {stats['total_features']} features")
                    print(f"   Best improvement: {stats.get('best_improvement', 0):+.4f}")
                else:
                    print(f"🧠 Created new dataset-specific cache")
        
        # Combine X and y
        df = X.copy()
        df[y.name] = y
        target_name = y.name
        
        # Hold-out confirmation split (following original CAAFE methodology)
        df_dev, df_hold = train_test_split(
            df, test_size=0.3, stratify=y, random_state=42
        )
        y_dev = df_dev[target_name]
        y_hold = df_hold[target_name]
        
        print(f"Split: {len(df_dev)} dev samples, {len(df_hold)} hold-out samples")
        print(f"Dev target distribution: {dict(y_dev.value_counts())}")
        print(f"Hold target distribution: {dict(y_hold.value_counts())}")
        
        # Use dev set for iteration (df becomes df_dev)
        df = df_dev
        
        def execute_and_evaluate(full_code, new_code):
            """Execute code and evaluate with optimized Critic."""
            try:
                # Apply code to feature data (excluding target)
                df_features = df.drop(columns=[target_name])
                print(f"🔧 Executing code on {len(df_features)} samples with {len(df_features.columns)} features")
                
                # Apply old code
                if full_code.strip():
                    df_old = execute_code_safely(full_code, df_features, target_name)
                    print(f"🔧 After old code: {len(df_old.columns)} features")
                else:
                    df_old = df_features.copy()
                
                # Apply old + new code
                combined_code = full_code + "\n" + new_code if full_code.strip() else new_code
                print(f"🔧 Executing combined code (old + new)")
                df_new = execute_code_safely(combined_code, df_features, target_name)
                print(f"🔧 After new code: {len(df_new.columns)} features (added {len(df_new.columns) - len(df_old.columns)})")
                
                # Use optimized Critic for evaluation (handles its own CV internally)
                try:
                    old_roc = self.scorer.score(df_old, y_dev)
                    new_roc = self.scorer.score(df_new, y_dev)
                    
                    # Return consistent format with single scores (Critic handles CV internally)
                    # Use the same score for both ROC and ACC for consistency with Critic
                    old_rocs = [old_roc]
                    new_rocs = [new_roc]
                    old_accs = [old_roc]  # Critic uses ROC-AUC as primary metric
                    new_accs = [new_roc]  # Critic uses ROC-AUC as primary metric
                    
                    return None, new_rocs, new_accs, old_rocs, old_accs
                    
                except Exception as e:
                    # Re-raise the exception - streamlined approach uses Critic only
                    raise e
                    
            except Exception as e:
                return e, None, None, None, None
        
        # Initialize conversation (using dev set)
        # Generate cache intelligence if available
        cache_intelligence = ""
        if self.cache:
            try:
                # Use the exact dataset context that matches the cache
                dataset_context_for_cache = description[:100]  # Match the cache initialization
                cache_intelligence = self.cache.generate_intelligent_prompt_addition(
                    dataset_context_for_cache, 
                    list(X.columns)
                )
                if cache_intelligence:
                    print(f"🧠 Generated cache intelligence with {len(self.cache.features)} cached features")
            except Exception as e:
                print(f"⚠️  Failed to generate cache intelligence: {e}")
        
        prompt = build_gemini_prompt(df, target_name, description, cache_intelligence)
        
        self.messages = [
            {
                "role": "system",
                "content": "You are an expert datascientist assistant solving Kaggle problems. You answer only by generating code. Answer as concisely as possible."
            },
            {
                "role": "user", 
                "content": prompt
            }
        ]
        
        self.full_code = ""
        
        # Main iteration loop
        for i in range(self.max_iterations):
            print(f"*Iteration {i+1}*")
            
            # === META-PROMPT INTERVENTION LOGIC ===
            if self.consecutive_rejections >= self.MAX_REJECTIONS_BEFORE_META_PROMPT:
                print("\n🧠 System is stuck in a loop. Injecting a meta-prompt to force a new strategy...")
                
                # Get the last error message for context
                last_error_msg = "Your last suggestion failed."
                if len(self.messages) > 1 and "Code execution failed" in self.messages[-1]['content']:
                    last_error_msg = self.messages[-1]['content']

                meta_prompt = f"""
CRITICAL FEEDBACK: Your last several suggestions have failed with the error: '{last_error_msg}'

This means your current strategy is fundamentally flawed. DO NOT try the same method again (e.g., KMeans on similar features).

You must propose a COMPLETELY DIFFERENT STRATEGY.

Since the feature names are anonymized, focus on robust mathematical transformations. Propose a new approach:
-   **New Idea 1:** Create a ratio of two features with high variance.
-   **New Idea 2:** Create a polynomial interaction term (`feat_A * feat_B + feat_C**2`).
-   **New Idea 3:** Create a feature that counts how many of a small subset of features are non-zero.

State your new strategy clearly in the HYPOTHESIS, then provide the code.
"""
                self.messages.append({"role": "user", "content": meta_prompt})
                self.consecutive_rejections = 0  # Reset counter after intervention
            
            try:
                code = self._generate_gemini_code(self.messages)
            except Exception as e:
                print(f"LLM API error: {e}")
                continue
            
            if not code.strip():
                continue
                
            print(f"```python\n{code}\n```")
            
            # Execute and evaluate
            error, new_rocs, _, old_rocs, _ = execute_and_evaluate(self.full_code, code)
            
            if error is not None:
                print(f"Code execution failed with error: {error}")
                
                # Get current available columns for better error guidance
                try:
                    if self.full_code.strip():
                        df_current = execute_code_safely(self.full_code, df.drop(columns=[target_name]), target_name)
                        available_vars = list(df_current.columns)
                    else:
                        available_vars = list(df.drop(columns=[target_name]).columns)
                except:
                    available_vars = list(df.drop(columns=[target_name]).columns)
                
                # Add special handling for import errors
                if "Import statements are not allowed" in str(error):
                    error_message = f"""Import error: {error}

CRITICAL: DO NOT use any import statements! The following tools are already available in your environment:
- KMeans (from sklearn.cluster)
- DBSCAN (from sklearn.cluster) 
- StandardScaler (from sklearn.preprocessing)
- pd (pandas)
- np (numpy)

Example of correct clustering code (NO IMPORTS NEEDED):
```python
scaler = StandardScaler()
features_to_cluster = df[['feature1', 'feature2']]
scaled_features = scaler.fit_transform(features_to_cluster)
kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
df['cluster_feature'] = kmeans.fit_predict(scaled_features)
```

Available variables: {available_vars}
Generate next feature WITHOUT any imports:
```python"""
                else:
                    error_message = f"Code execution failed with error: {type(error).__name__}: {error}.\nThis likely means you referenced a variable that doesn't exist or was previously rejected.\nAvailable variables: {available_vars}\nGenerate next feature using only available variables:\n```python"
                
                self.messages += [
                    {"role": "assistant", "content": code},
                    {"role": "user", "content": error_message}
                ]
                continue
            
            # Calculate improvements - use Critic (ROC-only) evaluation
            roc_improvement = np.nanmean(new_rocs) - np.nanmean(old_rocs)
            
            # Use ROC only with threshold matching original CAAFE paper expectations
            epsilon = 0.001  # Fixed low threshold from original CAAFE methodology
            decision_metric = roc_improvement
            
            # Display results
            print(f"Performance before adding features ROC {np.nanmean(old_rocs):.3f}.")
            print(f"Performance after adding features ROC {np.nanmean(new_rocs):.3f}.")
            print(f"Improvement ROC {roc_improvement:.3f} (req. ≥{epsilon:.3f}).", end=" ")
            
            # Acceptance gate following original CAAFE methodology
            add_feature = roc_improvement > epsilon
            add_feature_sentence = (
                "The code was executed and changes to ´df´ were kept." if add_feature
                else f"The last code changes to ´df´ were discarded. (Improvement: {decision_metric:.3f})"
            )
            print(add_feature_sentence)
            
            # Update conversation with available variables context
            if len(code) > 10:
                # Get current available columns (original + accepted features)
                try:
                    if self.full_code.strip():
                        df_current = execute_code_safely(self.full_code, df.drop(columns=[target_name]), target_name)
                        available_vars = list(df_current.columns)
                        print(f"🔧 Available columns after iteration {i+1}: {available_vars}")
                    else:
                        available_vars = list(df.drop(columns=[target_name]).columns)
                        print(f"🔧 Original columns: {available_vars}")
                except:
                    available_vars = list(df.drop(columns=[target_name]).columns)
                    print(f"🔧 Fallback columns: {available_vars}")
                
                if add_feature:
                    perf_message = f"Performance after adding feature ROC {np.nanmean(new_rocs):.3f}. {add_feature_sentence}\nAvailable variables: {available_vars}\nNext codeblock:"
                else:
                    perf_message = f"Performance after adding feature ROC {np.nanmean(new_rocs):.3f}. {add_feature_sentence}\nAvailable variables: {available_vars}\nGenerate a different feature using only available variables:\nNext codeblock:"
                
                self.messages += [
                    {"role": "assistant", "content": code},
                    {"role": "user", "content": perf_message}
                ]
            
            # Keep feature if it helps
            if add_feature:
                self.full_code += "\n" + code
                self.consecutive_rejections = 0  # Reset on success
                
                # Track new features (with target column protection)
                try:
                    df_temp = execute_code_safely(self.full_code, X.copy(), target_name)
                    new_features = [col for col in df_temp.columns if col not in X.columns]
                    for feat in new_features:
                        if feat not in self.generated_features:
                            self.generated_features.append(feat)
                except:
                    pass
                
                # Save successful feature to cache
                if self.cache and roc_improvement > 0.001:  # Only cache meaningful improvements
                    try:
                        # Extract feature description from code (look for comments)
                        feature_description = "Generated feature"
                        feature_name = "unknown_feature"
                        
                        # Try to extract feature name from code
                        code_lines = code.strip().split('\n')
                        for line in code_lines:
                            if "df['" in line and "=" in line:
                                # Extract feature name from assignment
                                parts = line.split("df['")
                                if len(parts) > 1:
                                    feature_name = parts[1].split("'")[0]
                                    break
                        
                        # Try to extract description from comments
                        for line in code_lines:
                            if line.strip().startswith('# ') and not line.strip().startswith('# Input'):
                                feature_description = line.strip()[2:]  # Remove '# '
                                break
                        
                        # Determine dataset context keywords
                        desc_words = description.lower().split()
                        context_keywords = []
                        for word in ['click', 'ad', 'advertising', 'content', 'recommendation', 
                                   'titanic', 'passenger', 'survival', 'diabetes', 'medical',
                                   'wine', 'quality', 'rating', 'finance', 'fraud', 'customer']:
                            if word in desc_words:
                                context_keywords.append(word)
                        
                        dataset_context = ' '.join(context_keywords) if context_keywords else description[:50]
                        
                        # Get baseline score from old_rocs
                        baseline_score = np.nanmean(old_rocs) if old_rocs else 0.6
                        
                        # Save to cache
                        self.cache.add_feature(
                            code=code,
                            description=feature_description,
                            feature_name=feature_name,
                            improvement_score=roc_improvement,
                            baseline_score=baseline_score,
                            dataset_context=dataset_context,
                            dataset_size=len(X),
                            dataset_features=len(X.columns),
                            tags=['gemini', 'streamlined_caafe']
                        )
                        
                    except Exception as e:
                        print(f"⚠️  Failed to cache feature: {e}")
            else:
                self.consecutive_rejections += 1  # Increment on failure
            
            print(f"Consecutive rejections: {self.consecutive_rejections}")
            
            print()  # Empty line
        
        print(f"Completed! Generated {len(self.generated_features)} useful features")
        
        # Hold-out confirmation pass to prevent overfitting to dev CV
        if self.full_code.strip():
            print(f"\n🔍 Hold-out confirmation pass...")
            
            # Apply features to both dev and hold-out sets
            df_dev_features = df_dev.drop(columns=[target_name])
            df_hold_features = df_hold.drop(columns=[target_name])
            
            # Create pipeline fitted ONLY on dev set
            pipeline = FeaturePipeline(self.full_code, target_name)
            pipeline.fit(df_dev_features)  # Fit ONLY on dev set
            
            # Transform both dev and hold-out with same fitted pipeline
            df_hold_old = df_hold_features.copy()
            df_hold_new = pipeline.transform(df_hold_features)  # Apply fitted transforms
            
            # Decouple training from hold-out scoring using existing Critic
            # Create local hold-out configuration (don't mutate self.scorer)
            tmp_holdout = 0.0  # No internal holdout since we're managing it manually
            
            # Save original holdout setting and temporarily override
            original_holdout = getattr(self.scorer, 'holdout', 0.2)
            self.scorer.holdout = tmp_holdout
            
            try:
                # Use Critic for hold-out evaluation to maintain consistency with original CAAFE
                # Create separate Critic instances for clean evaluation without contamination
                holdout_critic = Critic(folds=2, repeats=1, holdout=None, n_jobs=-1)
                
                # Evaluate baseline on hold-out
                roc_hold_old = holdout_critic.score(df_hold_old, y_hold)
                
                # Evaluate enhanced on hold-out
                roc_hold_new = holdout_critic.score(df_hold_new, y_hold)
                
            finally:
                # Restore original holdout setting
                self.scorer.holdout = original_holdout
            
            hold_improvement = roc_hold_new - roc_hold_old
            
            # Verify no target leakage in generated features
            for new_col in df_hold_new.columns:
                if new_col not in df_hold_old.columns:
                    # Check correlation with target (should be < 0.95 to avoid perfect leakage)
                    corr = abs(np.corrcoef(df_hold_new[new_col], y_hold)[0, 1])
                    if corr > 0.95:
                        raise ValueError(f"Potential target leakage detected: feature '{new_col}' has correlation {corr:.3f} with target")
            
            print(f"Hold-out validation: {roc_hold_old:.3f} → {roc_hold_new:.3f} (Δ{hold_improvement:+.3f})")
            
            # Keep features if they improve on hold-out OR don't hurt too much (tolerant check)
            # Allow small negative changes to account for holdout variance
            keep_features = hold_improvement >= -0.005
            
            if keep_features:
                print(f"✅ Features confirmed on hold-out set")
                enhanced_df = execute_code_safely(self.full_code, X.copy(), target_name)
                return enhanced_df
            else:
                print(f"❌ Features rejected - overfitted to dev set")
                return X
        else:
            return X
    
    def get_generated_features(self):
        """Get list of successfully generated features."""
        return self.generated_features
    
    def get_conversation_history(self):
        """Get the full conversation history with the LLM."""
        return self.messages
    
    def get_cache_stats(self):
        """Get statistics about the feature cache."""
        if not self.cache:
            return {"cache_enabled": False}
        return self.cache.get_cache_stats()
    
    def get_successful_features_for_context(self, dataset_context: str, limit: int = 10):
        """Get successful features from cache for a specific dataset context."""
        if not self.cache:
            return []
        return self.cache.get_successful_features(
            min_improvement=0.001,
            dataset_context=dataset_context,
            limit=limit
        )
    
    def apply_top_universal_features(self, X: pd.DataFrame, target_name: str, top_n: int = 3):
        """Apply the top universal features from cache to the dataset."""
        if not self.cache:
            print("📂 No cache available for universal features")
            return X
        
        top_features = self.cache.get_top_universal_features(top_n)
        if not top_features:
            print("📂 No universal features found in cache")
            return X
        
        enhanced_df = X.copy()
        applied_features = []
        
        print(f"🧠 Applying {len(top_features)} top universal features from cache...")
        
        for feature in top_features:
            try:
                # Apply the feature code
                enhanced_df = execute_code_safely(feature.code, enhanced_df, target_name)
                applied_features.append(feature.feature_name)
                print(f"   ✅ Applied '{feature.feature_name}' (original improvement: {feature.improvement_score:+.4f})")
            except Exception as e:
                print(f"   ❌ Failed to apply '{feature.feature_name}': {e}")
        
        if applied_features:
            print(f"🧠 Successfully applied {len(applied_features)} universal features: {applied_features}")
        
        return enhanced_df
    
    def export_cache_best_features(self, output_file: str = "best_caafe_features.py", min_rank: str = "good"):
        """Export the best cached features as a reusable Python module."""
        if not self.cache:
            print("❌ No cache available for export")
            return
        
        self.cache.export_best_features(output_file, min_rank)
        print(f"📄 Exported best features to {output_file}")


# Simple function for streamlined usage
def generate_features_gemini(
    X: pd.DataFrame, 
    y: pd.Series,
    description: str = "",
    max_iterations: int = 5,
    **kwargs
) -> pd.DataFrame:
    """
    Simple function to generate features using Streamlined CAAFE with Gemini.
    Each dataset gets its own cache file based on description and columns.
    
    Args:
        X: Input features DataFrame
        y: Target Series
        description: Dataset description
        max_iterations: Number of generation attempts
        **kwargs: Additional StreamlinedCAAFE arguments
        
    Returns:
        Enhanced DataFrame with generated features
    """
    caafe = StreamlinedCAAFE(max_iterations=max_iterations, **kwargs)
    return caafe.generate_features(X, y, description)