"""
CAAFE implementation following the original paper methodology.
"""

import pandas as pd
import numpy as np
import os
from typing import Optional, List
from dotenv import load_dotenv

from .evaluation import execute_and_evaluate_code_block
from .llm_interface import LLMInterface, build_caafe_prompt

# Load environment variables
load_dotenv()

class CAAFE:
    """Context-Aware Automated Feature Engineering following original paper."""
    
    def __init__(self, 
                 model: str = "gpt-4o",
                 api_key: Optional[str] = None,
                 max_iterations: int = 5,
                 n_splits: int = 3,
                 n_repeats: int = 2,
                 method: str = "xgb"):
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
        self.llm = LLMInterface(model=model, api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.max_iterations = max_iterations
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.method = method
        self.generated_features = []
        self.full_code = ""
        self.messages = []
        
    def generate_features(self, 
                         X: pd.DataFrame, 
                         y: pd.Series,
                         description: str = "") -> pd.DataFrame:
        """
        Generate features using LLM and evaluation following original CAAFE.
        
        Args:
            X: Input features
            y: Target variable
            description: Dataset description for the LLM
            
        Returns:
            DataFrame with original + generated features
        """
        print(f"*Dataset description:* {description}")
        print(f"Starting CAAFE with {len(X)} samples, {len(X.columns)} features\n")
        
        # Combine X and y into a single dataframe for processing
        df = X.copy()
        df[y.name] = y
        target_name = y.name
        
        # Initialize conversation
        self.messages = [
            {
                "role": "system", 
                "content": "You are an expert datascientist assistant solving Kaggle problems. You answer only by generating code. Answer as concisely as possible."
            }
        ]
        
        # Main iteration loop
        for i in range(self.max_iterations):
            print(f"*Iteration {i+1}*")
            
            # Build prompt
            prompt = build_caafe_prompt(
                df, 
                target_name, 
                description, 
                iteration=i+1,
                previous_features=self.generated_features
            )
            
            # Add to conversation
            self.messages.append({"role": "user", "content": prompt})
            
            # Generate code
            try:
                code = self.llm.generate_code(self.messages)
                if not code.strip():
                    print("No code generated, skipping iteration")
                    continue
                    
                print(f"```python\n{code}\n```")
                
            except Exception as e:
                print(f"LLM error: {e}")
                continue
            
            # Execute and evaluate
            error, new_rocs, new_accs, old_rocs, old_accs = execute_and_evaluate_code_block(
                df, target_name, self.full_code, code, 
                self.n_splits, self.n_repeats, self.method
            )
            
            if error is not None:
                print(f"Code execution failed with error: {error}")
                # Add error feedback to conversation
                self.messages.extend([
                    {"role": "assistant", "content": code},
                    {"role": "user", "content": f"Code execution failed with error: {type(error)} {error}.\nCode: ```python{code}```\nGenerate next feature (fixing error?):\n```python"}
                ])
                continue
            
            # Calculate improvements
            roc_improvement = np.nanmean(new_rocs) - np.nanmean(old_rocs)
            acc_improvement = np.nanmean(new_accs) - np.nanmean(old_accs)
            
            # Display results in CAAFE format
            print(f"Performance before adding features ROC {np.nanmean(old_rocs):.3f}, ACC {np.nanmean(old_accs):.3f}.")
            print(f"Performance after adding features ROC {np.nanmean(new_rocs):.3f}, ACC {np.nanmean(new_accs):.3f}.")
            print(f"Improvement ROC {roc_improvement:.3f}, ACC {acc_improvement:.3f}.", end=" ")
            
            # Decide whether to keep feature
            add_feature = (roc_improvement + acc_improvement) > 0
            
            if add_feature:
                self.full_code += "\n" + code
                # Apply the code to our working dataframe
                try:
                    from .evaluation import execute_code_safely
                    df_temp = execute_code_safely(self.full_code, X.copy())
                    df_temp[target_name] = y  # Re-add target
                    
                    # Track new features
                    new_features = [col for col in df_temp.columns if col not in X.columns and col != target_name]
                    for feat in new_features:
                        if feat not in self.generated_features:
                            self.generated_features.append(feat)
                    
                    # Update working dataframe
                    df = df_temp
                    
                except Exception as e:
                    print(f"Error applying code to working dataframe: {e}")
                    add_feature = False
            
            add_feature_sentence = "The code was executed and changes to ´df´ were kept." if add_feature else f"The last code changes to ´df´ were discarded. (Improvement: {roc_improvement + acc_improvement:.3f})"
            print(add_feature_sentence)
            
            # Add to conversation for next iteration
            if len(code) > 10:
                self.messages.extend([
                    {"role": "assistant", "content": code},
                    {"role": "user", "content": f"Performance after adding feature ROC {np.nanmean(new_rocs):.3f}, ACC {np.nanmean(new_accs):.3f}. {add_feature_sentence}\nNext codeblock:"}
                ])
            
            print()  # Empty line for readability
        
        print(f"Completed! Generated {len(self.generated_features)} useful features")
        
        # Return the final enhanced dataframe (without target)
        final_df = df.drop(columns=[target_name])
        return final_df
    
    def get_generated_features(self) -> List[str]:
        """Get list of successfully generated features."""
        return self.generated_features
    
    def get_conversation_history(self) -> List[dict]:
        """Get the full conversation history with the LLM."""
        return self.messages
    
# Legacy functions - keeping for compatibility but using new architecture
def generate_features(X: pd.DataFrame, 
                     y: pd.Series,
                     description: str = "",
                     max_iterations: int = 5,
                     **kwargs) -> pd.DataFrame:
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
    
