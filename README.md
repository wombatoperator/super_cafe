# CAAFE: Context-Aware Automated Feature Engineering

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

CAAFE (Context-Aware Automated Feature Engineering) is a powerful Python library that uses Large Language Models (LLMs) to automatically generate high-quality features for machine learning models. It intelligently determines whether feature engineering will improve your model's performance before investing computational resources.

## 🎯 Key Features

- **🧠 LLM-Powered Generation**: Uses OpenAI, Google Gemini, Anthropic Claude, or local models to generate contextually relevant features
- **⚡ XGBoost Optimized**: Specifically designed to improve gradient boosting model performance
- **🔒 Production Ready**: Comprehensive security validation and sandboxed code execution
- **📊 Smart Validation**: Determines when your model has reached its performance ceiling
- **💾 Intelligent Caching**: Learns from successful features across datasets
- **🔄 Adaptive Thresholds**: Harder to improve strong baselines, easier for weak ones

## 🚀 Quick Start

### Installation

```bash
pip install caafe[openai]  # For OpenAI models
# or
pip install caafe[gemini]  # For Google Gemini
# or  
pip install caafe[all]     # For all LLM providers
```

### Basic Usage

```python
import pandas as pd
from caafe import validate_feature_engineering

# Load your data
X, y = load_your_data()  # pandas DataFrame and Series

# Validate if feature engineering is worthwhile
results = validate_feature_engineering(
    X, y,
    description="Customer churn prediction with transaction history",
    provider="openai",  # or "gemini", "anthropic"
    model="gpt-4o"
)

# Check results
print(f"Recommendation: {results['recommendation']}")
print(f"Performance improvement: {results['total_improvement']:.3f}")

if results['worth_further_investment']:
    print("✅ Feature engineering improved your model!")
    # Apply the generated features
    X_enhanced = apply_features(X, results['final_code'])
else:
    print("🎯 Your model is already at its performance ceiling")
```

## 📚 Two Implementations

CAAFE provides two complementary approaches:

### 1. SUPER CAAFE (Recommended)
**Modern validation-focused approach** that determines if feature engineering is worthwhile:

```python
from caafe import SuperCAAFE

caafe = SuperCAAFE(
    provider="openai",
    model="gpt-4o", 
    max_iterations=10
)

results = caafe.probe_performance_ceiling(
    X, y, 
    dataset_context="E-commerce customer behavior analysis"
)
```

### 2. Original CAAFE  
**Classic iterative approach** for direct feature generation:

```python
from caafe import CAAFE

caafe = CAAFE(provider="openai", model="gpt-4o-mini")
X_enhanced = caafe.generate_features(
    X, y,
    description="Loan default prediction dataset"
)
```

## 🛡️ Security & Safety

CAAFE implements multiple security layers:

- **AST Validation**: Prevents execution of dangerous operations
- **Sandboxed Execution**: Limited namespace with safe built-ins only
- **Target Leakage Detection**: Automatically prevents data leakage
- **Code Review**: All generated code is validated before execution

## 📖 Core Concepts

### The Critic
- **Deterministic Evaluation**: Uses XGBoost with fixed parameters for reproducible results
- **Adaptive Thresholds**: Automatically adjusts acceptance criteria based on baseline strength
- **Statistical Validation**: Proper cross-validation and holdout testing

### The Generator
- **Multi-Provider Support**: Works with OpenAI, Gemini, Claude, and local models
- **Context-Aware Prompting**: Uses dataset description and column information
- **Iterative Learning**: Learns from previous attempts within the same session

### The Cache
- **Cross-Dataset Learning**: Successful features are cached for similar problems
- **Semantic Matching**: Finds relevant features based on dataset similarity
- **Impact Scoring**: Prioritizes features with proven performance improvements

## 🎛️ Configuration

### Environment Variables

```bash
export OPENAI_API_KEY="sk-..."
export GEMINI_API_KEY="..."
export ANTHROPIC_API_KEY="..."
```

### Advanced Configuration

```python
from caafe import SuperCAAFE

caafe = SuperCAAFE(
    provider="openai",
    model="gpt-4o",
    max_iterations=20,
    patience=5,  # Stop after 5 consecutive failures
    cache_dir="./my_cache",
    random_state=42
)
```

## 📊 Performance Guidelines

CAAFE's effectiveness depends on your baseline model strength:

| Baseline ROC-AUC | Expected Improvement | Recommendation |
|------------------|---------------------|----------------|
| < 0.6 | 2-10% | High potential |
| 0.6-0.7 | 1-5% | Moderate potential |
| 0.7-0.85 | 0.5-2% | Diminishing returns |
| 0.85-0.95 | 0.1-1% | Near ceiling |
| > 0.95 | < 0.5% | At performance ceiling |

## 🔗 Examples

Check out our [examples directory](examples/) for:

- **Simple Example**: Basic usage with real datasets
- **Advanced Usage**: Custom configurations and multi-provider setups
- **Benchmark Studies**: Performance comparisons across datasets

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
git clone https://github.com/caafe-ai/caafe.git
cd caafe
pip install -e .[dev]

# Run tests
pytest

# Run linting
black src/ tests/
ruff check src/ tests/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use CAAFE in your research, please cite:

```bibtex
@article{hollmann2023caafe,
  title={Context-Aware Automated Feature Engineering},
  author={Hollmann, Noah and Müller, Samuel and Hutter, Frank},
  journal={arXiv preprint arXiv:2305.03403},
  year={2023}
}
```

## 🙋 Support

- 📖 [Documentation](https://github.com/caafe-ai/caafe#readme)
- 🐛 [Issue Tracker](https://github.com/caafe-ai/caafe/issues)
- 💬 [Discussions](https://github.com/caafe-ai/caafe/discussions)

---

**CAAFE**: Making feature engineering intelligent, safe, and effective. 🎯