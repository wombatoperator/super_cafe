# Modern CAAFE: Multi-LLM Feature Engineering with Local Model Support

A streamlined implementation of Context-Aware Automated Feature Engineering (CAAFE) that uses Large Language Models to generate high-quality features optimized for XGBoost. Supports both cloud APIs (OpenAI) and local models (Ollama) with robust data leakage protection and optimized evaluation.

## 🚀 Quick Start

```python
from caafe import CAAFE

# OpenAI (cloud-based)
caafe_openai = CAAFE(provider="openai", model="gpt-4o-mini")
X_enhanced = caafe_openai.generate_features(
    X=your_dataframe,
    y=your_target, 
    description="Customer churn prediction with demographic and behavioral features"
)

# Ollama (local model)
caafe_local = CAAFE(
    provider="ollama",
    model="llama3:8b-instruct-q4_K_M",
    ollama_models_path="/path/to/ollama/models"
)
X_enhanced = caafe_local.generate_features(X, y, description)

# Train XGBoost with the enhanced features
import xgboost as xgb
model = xgb.XGBClassifier()
model.fit(X_enhanced, y)
```

## ✨ Key Features

- **Multi-LLM Support**: Use OpenAI (cloud) or Ollama (local) models
- **XGBoost Optimized**: Prompts designed specifically for gradient boosting
- **Data Leakage Protection**: Prevents target column access during feature generation
- **Optimized Evaluation**: Fast XGBoost Critic with pre-tuned hyperparameters
- **Structured Outputs**: JSON-based responses for consistent local model performance
- **Original CAAFE Spirit**: Maintains the elegance of the original implementation

## 📦 Installation

```bash
# Install dependencies
pip install pandas xgboost scikit-learn openai ollama python-dotenv

# Clone and use
git clone <this-repo>
cd caafe_2

# For OpenAI (optional)
export OPENAI_API_KEY="your-openai-api-key"

# For Ollama (install and start server)
ollama serve
ollama pull llama3:8b-instruct-q4_K_M
```

## 💡 Complete Example

```python
import pandas as pd
import numpy as np
from caafe import CAAFE

# Create sample data
np.random.seed(42)
X = pd.DataFrame({
    'age': np.random.randint(18, 80, 1000),
    'income': np.random.normal(50000, 20000, 1000),
    'debt': np.random.normal(10000, 5000, 1000),
    'credit_score': np.random.randint(300, 850, 1000)
})

# Create target (loan default)
default_prob = 0.1 + 0.2 * (X['debt'] / X['income']).clip(0, 1)
y = pd.Series(np.random.binomial(1, default_prob), name='default')

# Example 1: OpenAI
caafe_openai = CAAFE(provider="openai", model="gpt-4o-mini", max_iterations=3)
X_enhanced_openai = caafe_openai.generate_features(
    X=X, y=y, description="Loan default prediction with financial and demographic data"
)

# Example 2: Local Ollama
caafe_local = CAAFE(
    provider="ollama", 
    model="llama3:8b-instruct-q4_K_M",
    max_iterations=3
)
X_enhanced_local = caafe_local.generate_features(
    X=X, y=y, description="Loan default prediction with financial and demographic data"
)

print(f"Original: {X.shape[1]} features")
print(f"OpenAI enhanced: {X_enhanced_openai.shape[1]} features")
print(f"Ollama enhanced: {X_enhanced_local.shape[1]} features")
print(f"OpenAI generated: {caafe_openai.get_generated_features()}")
print(f"Ollama generated: {caafe_local.get_generated_features()}")
```

## 🔧 API Reference

### `CAAFE(provider="openai", model="gpt-4o-mini", **kwargs)`

**Core Args:**
- `provider`: LLM provider ("openai" or "ollama")
- `model`: Model name (e.g., "gpt-4o-mini", "llama3:8b-instruct-q4_K_M")
- `max_iterations`: Number of generation attempts (default: 5)

**OpenAI Args:**
- `api_key`: OpenAI API key (uses env var if None)

**Ollama Args:**
- `ollama_base_url`: Ollama server URL (default: "http://localhost:11434")
- `ollama_models_path`: Path to Ollama models directory

**Evaluation Args:**
- `n_splits`: Cross-validation splits (default: 3)
- `n_repeats`: Cross-validation repeats (default: 2)
- `scorer`: Custom scorer object (defaults to optimized XGBoost Critic)

### `generate_features(X, y, description)`

**Args:**
- `X`: Input features DataFrame  
- `y`: Target variable Series (must have .name attribute)
- `description`: Dataset description for the LLM

**Returns:**
- Enhanced DataFrame with original + generated features

## 🏗️ How It Works

1. **Data Leakage Protection**: Removes target column from LLM prompts
2. **LLM Feature Generation**: Creates XGBoost-optimized features based on dataset description
3. **Structured Outputs**: Uses JSON schema for consistent local model responses
4. **Optimized Evaluation**: Fast XGBoost Critic with pre-tuned hyperparameters
5. **Cross-Validation**: RepeatedStratifiedKFold for robust performance assessment  
6. **Iterative Improvement**: Keeps features that improve ROC-AUC + accuracy
7. **Safety Checks**: Validates generated code and handles edge cases

## 📊 Generated Feature Examples

CAAFE generates XGBoost-optimized features like:

```python
# Log/ratio transforms that trees can exploit
df['debt_to_income_ratio'] = df['debt'] / df['income']
df['log_income'] = np.log(df['income'] + 1)

# Z-score normalizations for clean splits
df['credit_score_zscore'] = (df['credit_score'] - df['credit_score'].mean()) / df['credit_score'].std()

# Non-linear interactions that create clean splits
df['high_debt_young'] = ((df['debt'] > df['debt'].median()) & (df['age'] < 30)).astype(int)

# Tree-friendly binning
df['income_quartile'] = pd.qcut(df['income'], 4, labels=False)
```

## ⚡ Performance

**OpenAI (Cloud):**
- **Generation Time**: ≈6s per iteration (500 rows), ≈18s (50k rows)
- **Cost**: ~$0.01-0.05 per dataset with gpt-4o-mini
- **Success Rate**: 70-85% of attempts generate useful features

**Ollama (Local):**
- **Generation Time**: ≈15-25s per iteration (llama3:8b on M-series Mac)
- **Cost**: Free after initial download (~4GB model)
- **Success Rate**: 60-75% with structured JSON outputs

**Both:**
- **Improvement**: Typically +1-3pp ROC-AUC on tabular datasets
- **XGBoost Critic**: 50-70% faster evaluation vs legacy cross-validation
- **Data Safety**: Zero data leakage with target column protection

## 🎯 When to Use CAAFE

**Good for:**
- Tabular datasets with clear feature meanings
- When you need automatic feature engineering
- XGBoost model optimization
- Rapid prototyping and experimentation

**Not ideal for:**
- Image, text, or time series data
- Datasets where features lack semantic meaning
- When you need fully interpretable features
- Production systems requiring deterministic behavior

## 🔒 Safety & Data Protection

CAAFE includes comprehensive safety measures:
- **Data Leakage Prevention**: Target column never accessible to LLM
- **Code Sandboxing**: Blocks dangerous operations (import, exec, eval, file access)
- **Input Validation**: Checks for infinite values, extreme outliers, and NaN issues
- **Error Recovery**: Robust exception handling with automatic fallbacks
- **JSON Schema Validation**: Structured outputs prevent malformed code generation

## 📝 Requirements

- Python 3.9+
- pandas>=2.0, xgboost>=2.0, scikit-learn>=1.3, numpy>=1.24
- **For OpenAI**: openai>=1.0, internet connection, API key
- **For Ollama**: ollama package, local models (llama3:8b recommended)
- **Optional**: python-dotenv for environment variable management

## 🤝 Contributing

This implementation prioritizes simplicity and performance. When contributing:

1. Maintain clean, readable code with comprehensive error handling
2. Preserve the simple API while adding new LLM providers
3. Add tests for new functionality, especially safety validations
4. Follow the original CAAFE methodology with XGBoost optimization
5. Ensure data leakage protection for all new features

## 📄 License

MIT License

## 🙏 Credits

- Original [CAAFE research](https://github.com/noahho/CAAFE) by Noah Hollmann et al.
- Inspired by the elegance and simplicity of the original implementation
- Built for the XGBoost ecosystem