"""
CAAFE: Context-Aware Automated Feature Engineering
Clean implementation following original methodology.
"""

import pandas as pd
import numpy as np
import openai
import os
from sklearn.model_selection import RepeatedKFold
from .evaluation import evaluate_dataset
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def build_prompt(df, target_name, description, iterative=1):
    """Build prompt with live DataFrame context."""
    how_many = (
        "up to 10 useful columns. Generate as many features as useful for downstream classifier, but as few as necessary to reach good performance."
        if iterative == 1
        else "exactly one useful column"
    )
    
    # Generate column info with samples
    samples = ""
    df_sample = df.head(10)
    for col in df_sample.columns:
        nan_freq = f"{df[col].isna().mean() * 100:.1f}"
        sample_values = df_sample[col].tolist()
        
        if str(df[col].dtype) == "float64":
            sample_values = [round(sample, 2) for sample in sample_values]
        
        samples += f"{col} ({df[col].dtype}): NaN-freq [{nan_freq}%], Samples {sample_values}\n"
    
    return f"""The dataframe `df` is loaded and in memory. Columns are also named attributes.
Description of the dataset in `df` (column dtypes might be inaccurate):
"{description}"

Columns in `df` (true feature dtypes listed here, categoricals encoded as int):
{samples}

This code was written by an expert datascientist working to improve predictions. It is a snippet of code that adds new columns to the dataset.
Number of samples (rows) in training dataset: {len(df)}

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
# Input samples: (Three samples of the columns used in the following code, e.g. '{df.columns[0]}': {list(df.iloc[:3, 0].values)}, '{df.columns[1]}': {list(df.iloc[:3, 1].values)}, ...)
(Some pandas code using '{df.columns[0]}', '{df.columns[1]}', ... to add a new column for each row in df)
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


def execute_code_safely(code, df):
    """Execute LLM code safely."""
    if not code.strip():
        return df
    
    local_vars = {
        'df': df.copy(),
        'pd': pd,
        'np': np
    }
    
    safe_builtins = {
        'int': int, 'float': float, 'str': str, 'bool': bool,
        'len': len, 'range': range, 'abs': abs, 'max': max, 
        'min': min, 'sum': sum, 'round': round,
    }
    
    exec(code, {"__builtins__": safe_builtins, "pd": pd, "np": np}, local_vars)
    return local_vars['df']


class CAAFE:
    """Context-Aware Automated Feature Engineering."""
    
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        max_iterations: int = 5,
        n_splits: int = 3,
        n_repeats: int = 2,
        method: str = "xgb"
    ):
        """
        Initialize CAAFE.
        
        Args:
            model: OpenAI model to use
            api_key: OpenAI API key (uses env var if None)
            max_iterations: Number of feature generation attempts
            n_splits: Cross-validation splits
            n_repeats: Cross-validation repeats
            method: Evaluation method (xgb, logistic)
        """
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.max_iterations = max_iterations
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.method = method
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
        
        def generate_code(messages):
            """Call OpenAI API."""
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
            """Execute code and evaluate with cross-validation."""
            old_rocs, old_accs, new_rocs, new_accs = [], [], [], []
            
            ss = RepeatedKFold(n_splits=self.n_splits, n_repeats=self.n_repeats, random_state=0)
            
            for train_idx, valid_idx in ss.split(df):
                df_train, df_valid = df.iloc[train_idx].copy(), df.iloc[valid_idx].copy()
                
                # Extract target
                target_train = df_train[target_name]
                target_valid = df_valid[target_name]
                df_train_features = df_train.drop(columns=[target_name])
                df_valid_features = df_valid.drop(columns=[target_name])
                
                try:
                    # Apply old code
                    df_train_old = execute_code_safely(full_code, df_train_features)
                    df_valid_old = execute_code_safely(full_code, df_valid_features)
                    
                    # Apply old + new code
                    combined_code = full_code + "\n" + new_code if full_code.strip() else new_code
                    df_train_new = execute_code_safely(combined_code, df_train_features)
                    df_valid_new = execute_code_safely(combined_code, df_valid_features)
                    
                    # Add target back
                    df_train_old[target_name] = target_train
                    df_valid_old[target_name] = target_valid
                    df_train_new[target_name] = target_train
                    df_valid_new[target_name] = target_valid
                    
                except Exception as e:
                    return e, None, None, None, None
                
                # Evaluate
                result_old = evaluate_dataset(
                    df_train=df_train_old,
                    df_test=df_valid_old,
                    target_name=target_name,
                    method=self.method
                )
                
                result_new = evaluate_dataset(
                    df_train=df_train_new,
                    df_test=df_valid_new,
                    target_name=target_name,
                    method=self.method
                )
                
                old_rocs.append(result_old["roc"])
                old_accs.append(result_old["acc"])
                new_rocs.append(result_new["roc"])
                new_accs.append(result_new["acc"])
            
            return None, new_rocs, new_accs, old_rocs, old_accs
        
        # Initialize conversation
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
            
            # Calculate improvements
            roc_improvement = np.nanmean(new_rocs) - np.nanmean(old_rocs)
            acc_improvement = np.nanmean(new_accs) - np.nanmean(old_accs)
            
            # Display results
            print(f"Performance before adding features ROC {np.nanmean(old_rocs):.3f}, ACC {np.nanmean(old_accs):.3f}.")
            print(f"Performance after adding features ROC {np.nanmean(new_rocs):.3f}, ACC {np.nanmean(new_accs):.3f}.")
            print(f"Improvement ROC {roc_improvement:.3f}, ACC {acc_improvement:.3f}.", end=" ")
            
            # Decide whether to keep
            add_feature = (roc_improvement + acc_improvement) > 0
            add_feature_sentence = (
                "The code was executed and changes to ´df´ were kept." if add_feature
                else f"The last code changes to ´df´ were discarded. (Improvement: {roc_improvement + acc_improvement:.3f})"
            )
            print(add_feature_sentence)
            
            # Update conversation
            if len(code) > 10:
                self.messages += [
                    {"role": "assistant", "content": code},
                    {"role": "user", "content": f"Performance after adding feature ROC {np.nanmean(new_rocs):.3f}, ACC {np.nanmean(new_accs):.3f}. {add_feature_sentence}\nNext codeblock:"}
                ]
            
            # Keep feature if it helps
            if add_feature:
                self.full_code += "\n" + code
                # Track new features
                try:
                    df_temp = execute_code_safely(self.full_code, X.copy())
                    new_features = [col for col in df_temp.columns if col not in X.columns]
                    for feat in new_features:
                        if feat not in self.generated_features:
                            self.generated_features.append(feat)
                except:
                    pass
            
            print()  # Empty line
        
        print(f"Completed! Generated {len(self.generated_features)} useful features")
        
        # Apply final code and return enhanced DataFrame
        if self.full_code.strip():
            enhanced_df = execute_code_safely(self.full_code, X.copy())
            return enhanced_df
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