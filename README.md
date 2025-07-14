# SUPER CAAFE: A Self-Improving Validation Framework for Automated Feature Engineering

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

SUPER CAAFE is a novel validation framework for automated semantic feature engineering that addresses the "strong baseline problem" - the challenge of generating features that provide statistically significant improvements for already powerful models like XGBoost. 

This work is inspired by the foundational research of [Hollmann et al. (2023)](https://arxiv.org/pdf/2305.03403) but represents a complete, independent implementation with distinct innovations in intelligent caching, stateless critics, and rigorous validation protocols.

## Key Features

- **Strong Baseline Problem Solution**: Designed specifically to enhance already powerful models (ROC-AUC > 0.85)
- **Self-Improving Intelligence**: Intelligent caching with in-context few-shot learning from past successes
- **High-Throughput Critic**: XGBoost 'hist' accelerated evaluation for rapid feature validation
- **Provider-Agnostic Architecture**: Seamless integration with OpenAI, Google, Anthropic, or local models via Ollama
- **Rigorous Validation Framework**: Multi-layered protocol with cross-validation and holdout confirmation
- **Enhanced Security**: AST validation with mathematical flexibility for complex transformations
- **Performance Ceiling Detection**: Automatically determines when further feature engineering is worthwhile

## Quick Start

### Installation

```bash
pip install -e .
# Ensure you have your API keys set:
export OPENAI_API_KEY="sk-..."
export GEMINI_API_KEY="..."
export ANTHROPIC_API_KEY="..."
```

### Basic Usage

```python
import pandas as pd
from caafe import SuperCAAFE

# Load your data
X, y = load_your_data()  # pandas DataFrame and Series

# Initialize SUPER CAAFE
caafe = SuperCAAFE(
    provider="openai",  # or "gemini", "anthropic", "ollama"
    model="gpt-4o"
)

# Validate if feature engineering is worthwhile
results = caafe.probe_performance_ceiling(
    X, y,
    dataset_context="Customer churn prediction with transaction history"
)

# Check results
print(f"Recommendation: {results['recommendation']}")
print(f"Performance improvement: {results['total_improvement']:.3f}")

if results['worth_further_investment']:
    print("Feature engineering improved your model!")
    X_enhanced = results['enhanced_features']
else:
    print("Your model is already at its performance ceiling")
```

## Framework Architecture

### Advanced Configuration

```python
from caafe import SuperCAAFE

# Initialize with full configuration options
caafe = SuperCAAFE(
    provider="openai",        # or "gemini", "anthropic", "ollama"
    model="gpt-4o", 
    max_iterations=10,
    intelligent_cache=True,   # Enable few-shot learning
    patience=5,               # Stop after 5 consecutive failures
    cache_dir="./feature_cache",
    random_state=42
)

# Probe performance ceiling with domain context
results = caafe.probe_performance_ceiling(
    X, y, 
    dataset_context="Industrial manufacturing sensor data with 512 features",
    domain="manufacturing"
)

# Framework determines if further engineering is worthwhile
if results['worth_further_investment']:
    print(f"Significant improvement: +{results['improvement']:.3f} ROC-AUC")
else:
    print("Model has reached performance ceiling")
```

## Security & Safety

SUPER CAAFE implements multiple security layers:

- **AST Validation**: Prevents execution of dangerous operations
- **Sandboxed Execution**: Limited namespace with safe built-ins only
- **Target Leakage Detection**: Automatically prevents data leakage
- **Code Review**: All generated code is validated before execution

## Core Technical Innovations

### The High-Throughput Critic
- **XGBoost 'hist' Acceleration**: High-speed evaluation optimized for rapid iteration
- **Stateless Design**: Fixed parameters ensure reproducible, deterministic results
- **Multi-Layered Validation**: Cross-validation + holdout confirmation protocol
- **Strong Baseline Awareness**: Adaptive thresholds based on baseline model performance

### Self-Improving Generator
- **Provider-Agnostic Interface**: Seamless switching between OpenAI, Gemini, Anthropic, Ollama
- **Chain-of-Thought Prompting**: Structured reasoning with hypothesis → method → implementation
- **Domain-Specific Expertise**: Transitions from pattern matching to expert-level reasoning
- **Tool-Augmented LLM**: Access to mathematical functions and statistical operations

### Intelligent Cache System
- **In-Context Few-Shot Learning**: LLM learns from past successes via cached examples
- **Cross-Dataset Knowledge Transfer**: Successful patterns applied to similar problems
- **Semantic Feature Matching**: Intelligent retrieval based on dataset similarity
- **Performance-Weighted Prioritization**: Most impactful features surface first

## Configuration

### Environment Variables

```bash
export OPENAI_API_KEY="sk-..."
export GEMINI_API_KEY="..."
export ANTHROPIC_API_KEY="..."
```

## Benchmark Results & Performance Guidelines

### SUPER CAAFE Validation Results
Our comprehensive 9-dataset benchmark demonstrates three key patterns:

| Dataset Type | Baseline ROC | Final ROC | Improvement | Key Finding |
|--------------|-------------|-----------|-------------|-------------|
| **Industrial Complex** | 0.500-0.682 | 0.512-0.695 | **+1.2-1.9%** | **High-impact domain features** |
| **Standard ML** | 0.824-0.985 | 0.827-0.990 | **+0.2-0.5%** | **Consistent marginal gains** |
| **Optimized Strong** | 0.933-0.846 | 0.933-0.846 | **+0.0%** | **Correctly identified ceiling** |

### Performance Expectations by Baseline Strength

| Baseline ROC-AUC | SUPER CAAFE Behavior | Expected Outcome | Business Value |
|------------------|---------------------|------------------|----------------|
| **< 0.6** | **High-impact generation** | **2-10% gains** | **Excellent ROI** |
| **0.6-0.7** | **Moderate enhancement** | **1-5% gains** | **Good ROI** |
| **0.7-0.85** | **Careful optimization** | **0.5-2% gains** | **Marginal ROI** |
| **0.85-0.95** | **Rigorous validation** | **0.1-1% gains** | **Questionable ROI** |
| **> 0.95** | **Ceiling detection** | **No improvement** | **Prevents degradation** |

## Examples

Check out our [examples directory](examples/) for:

- **Simple Example**: Basic usage with real datasets
- **Advanced Usage**: Custom configurations and multi-provider setups
- **Benchmark Studies**: Performance comparisons across datasets

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
git clone https://github.com/your-org/super-caafe.git
cd super-caafe
pip install -e .

# Run tests
pytest

# Run linting
black src/ tests/
ruff check src/ tests/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use SUPER CAAFE in your research, please cite both the original foundational work and our framework:

### Original CAAFE Foundation
```bibtex
@article{hollmann2023caafe,
  title={Context-Aware Automated Feature Engineering},
  author={Hollmann, Noah and Müller, Samuel and Hutter, Frank},
  journal={arXiv preprint arXiv:2305.03403},
  year={2023}
}
```

### SUPER CAAFE Framework
```bibtex
@article{super_caafe2024,
  title={SUPER CAAFE: A Self-Improving Validation Framework for Automated Semantic Feature Engineering},
  author={[Your Name]},
  year={2024},
  note={Available at: papers/Super_CAAFE.pdf}
}
```

## Research Paper

The complete SUPER CAAFE research paper is available in this repository: [`papers/Super_CAAFE.pdf`](papers/Super_CAAFE.pdf)

**Abstract**: SUPER CAAFE addresses the "strong baseline problem" in automated feature engineering - the challenge of generating features that provide statistically significant improvements for already powerful models. Through intelligent caching, chain-of-thought prompting, and rigorous validation protocols, SUPER CAAFE successfully generates high-impact features for complex industrial datasets while correctly identifying performance ceilings for optimized models.

## Support

- Documentation: This README
- Issue Tracker: GitHub Issues
- Discussions: GitHub Discussions