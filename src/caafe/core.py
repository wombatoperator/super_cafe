"""
CAAFE: Context-Aware Automated Feature Engineering
Clean implementation following original methodology.
"""

import pandas as pd
import numpy as np
import openai
import ollama
import os
import ast
from sklearn.model_selection import RepeatedKFold, train_test_split
from .evaluation import evaluate_dataset
from .critic import Critic
from typing import Optional
from dotenv import load_dotenv

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
        ast.Attribute, ast.Subscript, ast.Index, ast.Slice, ast.Call, ast.keyword,
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
        ast.Lambda, ast.arguments, ast.arg,  # Lambda functions and arguments for pandas operations
    }
    
    # Forbidden imports and operations
    FORBIDDEN_NAMES = {
        'import', 'exec', 'eval', '__import__', 'open', 'file', 'input', 'raw_input',
        'compile', 'reload', 'vars', 'locals', 'globals', 'dir', 'help', 'quit', 'exit',
        'os', 'sys', 'subprocess', 'shutil', 'urllib', 'requests', 'socket',
    }
    
    # Allowed function/attribute prefixes for pandas and numpy
    ALLOWED_PREFIXES = ['pd.', 'np.', 'df.', 'df[']
    
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


def build_prompt(df, target_name, description, iterative=1):
    """Build prompt exactly matching CAAFE paper format."""
    how_many = (
        "exactly one useful column"  
    )
    
    # Create dataframe WITHOUT target column to prevent data leakage
    df_features_only = df.drop(columns=[target_name] if target_name in df.columns else [])
    
    # Generate column info with samples (exactly like CAAFE format)
    samples = ""
    df_sample = df_features_only.head(10)
    for col in df_sample.columns:
        nan_freq = f"{df_features_only[col].isna().mean() * 100:.1f}"
        sample_values = df_sample[col].tolist()
        
        if str(df_features_only[col].dtype) == "float64":
            sample_values = [round(sample, 2) for sample in sample_values]
        
        # Format exactly like CAAFE paper
        samples += f"{col} ({df_features_only[col].dtype}): NaN-freq [{nan_freq}%], Samples {sample_values}\n"
    
    # Use CAAFE paper prompt format exactly
    return f"""The dataframe 'df' is loaded and in memory. Columns are also named attributes.
Description of the dataset in 'df' (column dtypes might be inaccurate):
"{description}"

Columns in 'df' (true feature dtypes listed here, categoricals encoded as int):
{samples}

This code was written by an expert datascientist working to improve predictions. It is a snippet of code that adds new columns to the dataset.
Number of samples (rows) in training dataset: {len(df_features_only)}

This code generates additional columns that are useful for a downstream classification algorithm (such as XGBoost) predicting "{target_name}".
Additional columns add new semantic information, that is they use real world knowledge on the dataset. They can e.g. be feature combinations, transformations, aggregations where the new column is a function of the existing columns.
The scale of columns and offset does not matter. Make sure all used columns exist. Follow the above description of columns closely and consider the datatypes and meanings of classes.
This code also drops columns, if these may be redundant and hurt the predictive performance of the downstream classifier (Feature selection). Dropping columns may help as the chance of overfitting is lower, especially if the dataset is small.
The classifier will be trained on the dataset with the generated columns and evaluated on a holdout set. The evaluation metric is accuracy. The best performing code will be selected.
Added columns can be used in other codeblocks, dropped columns are not available anymore.

Code formatting for each added column:
```python
# (Feature name and description)
# Usefulness: (Description why this adds useful real world knowledge to classify "{target_name}" according to dataset description and attributes.)
# Input samples: (Three samples of the columns used in the following code, e.g. '{df_features_only.columns[0] if len(df_features_only.columns) > 0 else "col1"}': {list(df_features_only.iloc[:3, 0].values) if len(df_features_only) > 0 else [1,2,3]}, '{df_features_only.columns[1] if len(df_features_only.columns) > 1 else "col2"}': {list(df_features_only.iloc[:3, 1].values) if len(df_features_only) > 1 else [4,5,6]}, ...)
(Some pandas code using '{df_features_only.columns[0] if len(df_features_only.columns) > 0 else "col1"}', '{df_features_only.columns[1] if len(df_features_only.columns) > 1 else "col2"}', ... to add a new column for each row in df)
```end

Code formatting for dropping columns:
```python
# Explanation why the column XX is dropped
df.drop(columns=['XX'], inplace=True)
```end

Each codeblock generates {how_many} and can drop unused columns (Feature selection).
Each codeblock ends with ```end and starts with "```python"

Codeblock:
"""



def execute_code_safely(code, df, target_column=None):
    """Execute LLM code safely without access to target column."""
    if not code.strip():
        return df
    
    # Create a copy of the dataframe WITHOUT the target column to prevent data leakage
    df_safe = df.copy()
    if target_column and target_column in df_safe.columns:
        df_safe = df_safe.drop(columns=[target_column])
    
    # Check if code tries to access the target column (strip comments first)
    if target_column:
        # Strip comments before checking for target column name to avoid false positives
        import re
        clean_code = re.sub(r"#.*?$", "", code, flags=re.MULTILINE)
        if target_column in clean_code:
            raise ValueError(f"Data leakage detected: code attempts to use target column '{target_column}'")
    
    # AST security validation
    try:
        parsed = ast.parse(code, mode="exec")
        check_ast(parsed)
    except (SyntaxError, ValueError) as e:
        raise ValueError(f"Code security validation failed: {e}")
    
    # Define freq_encode helper for safe frequency encoding
    def freq_encode(col):
        """Helper function for safe frequency encoding."""
        vc = df_safe[col].value_counts()
        return df_safe[col].map(vc)
    
    local_vars = {
        'df': df_safe,
        'pd': pd,
        'np': np
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


class CAAFE:
    """Context-Aware Automated Feature Engineering."""
    
    def __init__(
        self,
        provider: str = "openai",  # "openai" or "ollama"
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        ollama_base_url: str = "http://localhost:11434",
        ollama_models_path: Optional[str] = None,
        max_iterations: int = 5,
        n_splits: int = 3,
        n_repeats: int = 2,
        method: str = "xgb",
        scorer=None  # Optional scorer object with .score(X, y) method
    ):
        """
        Initialize CAAFE.
        
        Args:
            provider: LLM provider ("openai" or "ollama")
            model: LLM model to use
            api_key: OpenAI API key (if using OpenAI)
            ollama_base_url: Ollama server URL
            ollama_models_path: Path to Ollama models directory
            max_iterations: Number of feature generation attempts
            n_splits: Cross-validation splits
            n_repeats: Cross-validation repeats
            method: Evaluation method (xgb, logistic) - used only for fallback
            scorer: Optional scorer object with .score(X, y) method
        """
        self.provider = provider
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.ollama_base_url = ollama_base_url
        self.ollama_models_path = ollama_models_path
        self.max_iterations = max_iterations
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.method = method
        
        # Use optimized Critic by default, fallback to legacy evaluation
        self.scorer = scorer or Critic(folds=n_splits, repeats=n_repeats, n_jobs=-1)
        
        self.generated_features = []
        self.full_code = ""
        self.messages = []
    
    def generate_features(self, X: pd.DataFrame, y: pd.Series, description: str = "") -> pd.DataFrame:
        """
        Generate features using CAAFE methodology.
        
        Args:
            X: Input features
            y: Target variable
            description: Dataset description for the LLM
            
        Returns:
            DataFrame with original + generated features
        """
        print(f"*Dataset description:* {description}")
        print(f"Starting CAAFE with {len(X)} samples, {len(X.columns)} features\n")
        
        # Combine X and y
        df = X.copy()
        df[y.name] = y
        target_name = y.name
        
        # Hold-out confirmation split (50/50 like CAAFE paper)
        df_dev, df_hold = train_test_split(
            df, test_size=0.5, stratify=y, random_state=42
        )
        y_dev = df_dev[target_name]
        y_hold = df_hold[target_name]
        
        print(f"Split: {len(df_dev)} dev samples, {len(df_hold)} hold-out samples")
        
        # Use dev set for iteration (df becomes df_dev)
        df = df_dev
        
        def generate_code(messages):
            """Call LLM API (OpenAI or Ollama)."""
            if self.provider == "ollama":
                # Set OLLAMA_MODELS environment variable if provided
                if self.ollama_models_path:
                    os.environ['OLLAMA_MODELS'] = self.ollama_models_path
                
                client = ollama.Client(host=self.ollama_base_url)
                
                # Define structured output schema for consistent code generation
                schema = {
                    "type": "object",
                    "properties": {
                        "feature_name": {
                            "type": "string",
                            "description": "Short descriptive name for the feature"
                        },
                        "explanation": {
                            "type": "string", 
                            "description": "Why this feature adds useful real world knowledge"
                        },
                        "code": {
                            "type": "string",
                            "description": "Single line of pandas code to generate the feature"
                        }
                    },
                    "required": ["feature_name", "explanation", "code"]
                }
                
                # Add structured output instruction to the last message
                structured_messages = messages.copy()
                last_message = structured_messages[-1]['content']
                structured_messages[-1]['content'] = f"""{last_message}

Return your response as JSON with this exact format:
{{
    "feature_name": "descriptive_name", 
    "explanation": "Why this helps classify the target",
    "code": "df['new_feature'] = # single line pandas expression"
}}

Generate exactly ONE feature. Use only pandas operations on existing columns."""
                
                response = client.chat(
                    model=self.model,
                    messages=structured_messages,
                    format="json",  # Enable structured output
                    options={
                        'temperature': 0.1,  # Lower for more consistent outputs
                        'num_predict': 300
                    }
                )
                
                try:
                    import json
                    result = json.loads(response['message']['content'])
                    # Format as expected by the rest of the pipeline
                    code = f"# ({result['feature_name']}: {result['explanation']})\n{result['code']}"
                    return code
                except (json.JSONDecodeError, KeyError) as e:
                    # Fallback to raw content if JSON parsing fails
                    return response['message']['content']
                    
            else:  # OpenAI
                client = openai.OpenAI(api_key=self.api_key)
                completion = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    stop=["```end"],
                    temperature=0.5,
                    max_tokens=500
                )
                code = completion.choices[0].message.content
                return code.replace("```python", "").replace("```", "").replace("```end", "")
        
        def execute_and_evaluate(full_code, new_code):
            """Execute code and evaluate with optimized Critic."""
            try:
                # Apply code to feature data (excluding target)
                df_features = df.drop(columns=[target_name])
                
                # Apply old code
                if full_code.strip():
                    df_old = execute_code_safely(full_code, df_features, target_name)
                else:
                    df_old = df_features.copy()
                
                # Apply old + new code
                combined_code = full_code + "\n" + new_code if full_code.strip() else new_code
                df_new = execute_code_safely(combined_code, df_features, target_name)
                
                # Use optimized Critic for evaluation (handles its own CV internally)
                try:
                    old_roc = self.scorer.score(df_old, y_dev)
                    new_roc = self.scorer.score(df_new, y_dev)
                    
                    # Return consistent format with single scores (Critic handles CV internally)
                    old_rocs = [old_roc]
                    new_rocs = [new_roc]
                    old_accs = [old_roc]  # Minimal for API compatibility
                    new_accs = [new_roc]  # Minimal for API compatibility
                    
                    return None, new_rocs, new_accs, old_rocs, old_accs
                    
                except Exception as e:
                    # Fallback to legacy evaluation if Critic fails
                    print(f"⚠️  Critic failed, falling back to legacy evaluation: {e}")
                    
                    # Legacy CV loop as fallback
                    old_rocs, old_accs, new_rocs, new_accs = [], [], [], []
                    ss = RepeatedKFold(n_splits=self.n_splits, n_repeats=self.n_repeats, random_state=0)
                    
                    for train_idx, valid_idx in ss.split(df):
                        df_train, df_valid = df.iloc[train_idx].copy(), df.iloc[valid_idx].copy()
                        
                        # Extract target
                        target_train = df_train[target_name]
                        target_valid = df_valid[target_name]
                        df_train_features = df_train.drop(columns=[target_name])
                        df_valid_features = df_valid.drop(columns=[target_name])
                        
                        # Apply transformations
                        df_train_old = execute_code_safely(full_code, df_train_features, target_name)
                        df_valid_old = execute_code_safely(full_code, df_valid_features, target_name)
                        df_train_new = execute_code_safely(combined_code, df_train_features, target_name)
                        df_valid_new = execute_code_safely(combined_code, df_valid_features, target_name)
                        
                        # Add target back for legacy evaluation
                        df_train_old[target_name] = target_train
                        df_valid_old[target_name] = target_valid
                        df_train_new[target_name] = target_train
                        df_valid_new[target_name] = target_valid
                        
                        # Legacy evaluation
                        result_old = evaluate_dataset(df_train_old, df_valid_old, target_name, self.method)
                        result_new = evaluate_dataset(df_train_new, df_valid_new, target_name, self.method)
                        
                        old_rocs.append(result_old["roc"])
                        old_accs.append(result_old["acc"])
                        new_rocs.append(result_new["roc"])
                        new_accs.append(result_new["acc"])
                    
                    return None, new_rocs, new_accs, old_rocs, old_accs
                    
            except Exception as e:
                return e, None, None, None, None
        
        # Initialize conversation (using dev set)
        prompt = build_prompt(df, target_name, description, self.max_iterations)
        
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
            
            try:
                code = generate_code(self.messages)
            except Exception as e:
                print(f"LLM API error: {e}")
                continue
            
            if not code.strip():
                continue
                
            print(f"```python\n{code}\n```")
            
            # Execute and evaluate
            error, new_rocs, new_accs, old_rocs, old_accs = execute_and_evaluate(self.full_code, code)
            
            if error is not None:
                print(f"Code execution failed with error: {error}")
                self.messages += [
                    {"role": "assistant", "content": code},
                    {"role": "user", "content": f"Code execution failed with error: {type(error)} {error}.\nCode: ```python\n{code}\n```\nGenerate next feature (fixing error?):\n```python"}
                ]
                continue
            
            # Calculate improvements - handle Critic vs legacy evaluation differently
            roc_improvement = np.nanmean(new_rocs) - np.nanmean(old_rocs)
            
            # Check if we're using Critic (ROC-only) or legacy evaluation (ROC + ACC)
            using_critic = isinstance(self.scorer, Critic)
            
            if using_critic:
                # Critic mode: use ROC only with dynamic acceptance threshold
                # Calculate dynamic threshold based on sample size (more lenient for real data)
                n_holdout = len(y_dev)
                epsilon = 0.001  # Fixed low threshold matching CAAFE paper expectations
                acc_improvement = 0.0
                decision_metric = roc_improvement
                
                # Display results (show that ACC is same as ROC for Critic)
                print(f"Performance before adding features ROC {np.nanmean(old_rocs):.3f}.")
                print(f"Performance after adding features ROC {np.nanmean(new_rocs):.3f}.")
                print(f"Improvement ROC {roc_improvement:.3f} (req. ≥{epsilon:.3f}).", end=" ")
                
                # Tightened acceptance gate
                add_feature = roc_improvement > epsilon
            else:
                # Legacy mode: use both ROC and ACC
                acc_improvement = np.nanmean(new_accs) - np.nanmean(old_accs)
                decision_metric = roc_improvement + acc_improvement
                
                # Display results
                print(f"Performance before adding features ROC {np.nanmean(old_rocs):.3f}, ACC {np.nanmean(old_accs):.3f}.")
                print(f"Performance after adding features ROC {np.nanmean(new_rocs):.3f}, ACC {np.nanmean(new_accs):.3f}.")
                print(f"Improvement ROC {roc_improvement:.3f}, ACC {acc_improvement:.3f}.", end=" ")
                
                # Legacy acceptance gate
                add_feature = decision_metric > 0
            add_feature_sentence = (
                "The code was executed and changes to ´df´ were kept." if add_feature
                else f"The last code changes to ´df´ were discarded. (Improvement: {decision_metric:.3f})"
            )
            print(add_feature_sentence)
            
            # Update conversation
            if len(code) > 10:
                if using_critic:
                    perf_message = f"Performance after adding feature ROC {np.nanmean(new_rocs):.3f}. {add_feature_sentence}\nNext codeblock:"
                else:
                    perf_message = f"Performance after adding feature ROC {np.nanmean(new_rocs):.3f}, ACC {np.nanmean(new_accs):.3f}. {add_feature_sentence}\nNext codeblock:"
                
                self.messages += [
                    {"role": "assistant", "content": code},
                    {"role": "user", "content": perf_message}
                ]
            
            # Keep feature if it helps
            if add_feature:
                self.full_code += "\n" + code
                # Track new features (with target column protection)
                try:
                    df_temp = execute_code_safely(self.full_code, X.copy(), target_name)
                    new_features = [col for col in df_temp.columns if col not in X.columns]
                    for feat in new_features:
                        if feat not in self.generated_features:
                            self.generated_features.append(feat)
                except:
                    pass
            
            print()  # Empty line
        
        print(f"Completed! Generated {len(self.generated_features)} useful features")
        
        # Hold-out confirmation pass to prevent overfitting to dev CV
        if self.full_code.strip():
            print(f"\n🔍 Hold-out confirmation pass...")
            
            # Apply features to both dev and hold-out sets
            df_dev_features = df_dev.drop(columns=[target_name])
            df_hold_features = df_hold.drop(columns=[target_name])
            
            # Original features (no generated features)
            df_dev_old = df_dev_features.copy()
            df_hold_old = df_hold_features.copy()
            
            # Enhanced features (with generated features)
            df_dev_new = execute_code_safely(self.full_code, df_dev_features, target_name)
            df_hold_new = execute_code_safely(self.full_code, df_hold_features, target_name)
            
            # Evaluate on hold-out set
            roc_hold_old = self.scorer.score(df_hold_old, y_hold)
            roc_hold_new = self.scorer.score(df_hold_new, y_hold)
            hold_improvement = roc_hold_new - roc_hold_old
            
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


# Legacy function for backward compatibility
def generate_features(
    X: pd.DataFrame, 
    y: pd.Series,
    description: str = "",
    max_iterations: int = 5,
    **kwargs
) -> pd.DataFrame:
    """
    Simple function to generate features using CAAFE.
    
    Args:
        X: Input features DataFrame
        y: Target Series
        description: Dataset description
        max_iterations: Number of generation attempts
        **kwargs: Additional CAAFE arguments
        
    Returns:
        Enhanced DataFrame with generated features
    """
    caafe = CAAFE(max_iterations=max_iterations, **kwargs)
    return caafe.generate_features(X, y, description)