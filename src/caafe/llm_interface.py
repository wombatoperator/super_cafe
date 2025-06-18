"""
LLM interface for CAAFE - handles prompt generation and LLM calls.
"""

import openai
import pandas as pd
import re
from typing import Optional, List

class LLMInterface:
    """Interface for interacting with LLMs for feature generation."""
    
    def __init__(self, model: str = "gpt-4o", api_key: Optional[str] = None):
        """
        Initialize LLM interface.
        
        Args:
            model: Model name to use
            api_key: API key (uses environment if None)
        """
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
    
    def generate_code(self, messages: List[dict]) -> str:
        """
        Generate code using the LLM.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Generated code string
        """
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stop=["```end"],
            temperature=0.5,
            max_tokens=500
        )
        
        code = completion.choices[0].message.content
        return self._clean_code(code)
    
    def _clean_code(self, code: str) -> str:
        """
        Clean and extract executable code from LLM response.
        
        Args:
            code: Raw LLM response
            
        Returns:
            Cleaned executable code
        """
        # Remove markdown formatting
        code = code.replace("```python", "").replace("```end", "").replace("```", "")
        
        # Clean up and return all non-comment lines
        lines = code.split('\n')
        clean_lines = []
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                clean_lines.append(line)
        
        return '\n'.join(clean_lines)

def build_caafe_prompt(
    df,
    target_name: str,
    data_description: str,
    iteration: int = 1,
    previous_features: List[str] = None
) -> str:
    """
    Build a sophisticated CAAFE-style prompt.
    
    Args:
        df: Dataframe to analyze
        target_name: Target column name
        data_description: Dataset description
        iteration: Current iteration number
        previous_features: List of previously generated features
        
    Returns:
        Formatted prompt string
    """
    # Sample data for the prompt
    samples = ""
    df_sample = df.head(10)
    for col in list(df_sample.columns):
        if col == target_name:
            continue
        nan_freq = f"{df[col].isna().mean() * 100:.1f}"
        sample_values = df_sample[col].tolist()
        
        # Format sample values
        if df[col].dtype == 'float64':
            sample_values = [round(val, 2) if pd.notna(val) else 'nan' for val in sample_values]
        
        samples += f"{col} ({df[col].dtype}): NaN-freq [{nan_freq}%], Samples {sample_values}\n"
    
    # Build the main prompt 
    how_many = "exactly one useful column"
    
    prompt = f"""The dataframe `df` is loaded and in memory. Columns are also named attributes.
Description of the dataset in `df` (column dtypes might be inaccurate):
"{data_description}"

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
(Some pandas code using {df.columns[0]}', '{df.columns[1]}', ... to add a new column for each row in df)
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
    
    return prompt