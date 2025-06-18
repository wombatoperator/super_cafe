# Modern CAAFE: Streamlined LLM-Powered Feature Engineering for XGBoost

## Project Overview

Build a clean, efficient implementation of CAAFE (Context-Aware Automated Feature Engineering) that uses modern LLMs to generate high-quality features specifically optimized for XGBoost performance. Focus on simplicity, elegance, and practical results.

## Background & Original Research

### Original CAAFE (2023)
- **Core Innovation**: LLMs generate semantically meaningful features based on dataset descriptions
- **Key Results**: 11/14 datasets improved, ROC AUC from 0.798 → 0.822
- **Method**: Iterative generation → validation → feedback loop
- **Limitation**: Focused on Random Forest/TabPFN, used outdated OpenAI API v0.28

### Why XGBoost Focus?
- **Industry Standard**: Most widely used gradient boosting framework
- **Feature Sensitivity**: XGBoost benefits significantly from well-engineered features
- **Interpretability**: Built-in feature importance for validation
- **Performance**: Consistently strong across diverse tabular datasets

## Project Goals

### Primary Objectives
1. **Modernize for XGBoost**: Optimize feature generation specifically for gradient boosting
2. **Multi-LLM Support**: OpenAI v1.0+, Claude, and local models
3. **Streamlined Architecture**: Clean, minimal codebase focused on core functionality
4. **Practical Performance**: Measurable XGBoost performance improvements

### Success Metrics
- Consistent XGBoost performance improvements across test datasets
- Sub-5-minute feature generation for typical datasets
- Clean API that data scientists can easily adopt

## Streamlined Project Structure

```
modern-caafe/
├── README.md
├── pyproject.toml
├── .gitignore
├── 
├── src/
│   └── caafe/
│       ├── __init__.py
│       ├── core.py                   # Main CAAFE class
│       ├── llm_providers.py          # All LLM integrations
│       ├── prompts.py                # Prompt engineering
│       ├── safety.py                 # Code validation & execution
│       ├── evaluation.py             # XGBoost-focused evaluation
│       └── utils.py                  # Data handling utilities
│
├── experiments/
│   ├── benchmark_datasets.py         # Dataset collection
│   ├── run_comparison.py             # vs baselines
│   └── analyze_results.py            # Results analysis
│
├── tests/
│   ├── test_core.py
│   ├── test_llm_providers.py
│   ├── test_safety.py
│   └── test_data/
│
└── examples/
    ├── quickstart.py                 # Simple usage example
    └── advanced_usage.py             # Full feature demo
```

## Core Architecture

### Main CAAFE Class
```python
class CAAFE:
    def __init__(
        self, 
        llm_provider: str = "openai",
        model: str = "gpt-4o",
        max_iterations: int = 5,
        xgb_params: Optional[Dict] = None
    ):
        self.llm = self._get_provider(llm_provider, model)
        self.evaluator = XGBoostEvaluator(xgb_params)
        self.safety = CodeValidator()
    
    def generate_features(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        dataset_description: str
    ) -> pd.DataFrame:
        """Generate and validate features optimized for XGBoost"""
        # Core iterative loop
```

### LLM Provider Interface
```python
class LLMProvider(Protocol):
    def generate_features(self, prompt: str) -> str: ...

class OpenAIProvider(LLMProvider): ...
class ClaudeProvider(LLMProvider): ...
class LocalProvider(LLMProvider): ...  # Ollama/vLLM
```

### XGBoost-Optimized Evaluation
```python
class XGBoostEvaluator:
    def __init__(self, params: Optional[Dict] = None):
        self.params = params or {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'random_state': 42
        }
    
    def evaluate_feature(
        self, 
        X_original: pd.DataFrame,
        X_with_feature: pd.DataFrame, 
        y: pd.Series
    ) -> Tuple[float, float]:  # (improvement, feature_importance)
        """Cross-validate XGBoost performance with new feature"""
```

## Implementation Plan

### Phase 1: Core Framework (Week 1)
**Focus**: Build the essential components

**Tasks**:
1. **Project Setup**
   - Modern Python project structure
   - Core dependencies (xgboost, pandas, scikit-learn)
   - Basic testing framework

2. **Core CAAFE Class**
   - Main feature generation loop
   - XGBoost integration
   - Simple prompt construction

3. **OpenAI Integration**
   - API v1.0+ implementation
   - Error handling and retries
   - Basic prompt templates

**Deliverable**: Working CAAFE with OpenAI + XGBoost

### Phase 2: Multi-LLM & Safety (Week 2)
**Focus**: Expand LLM support and ensure safety

**Tasks**:
1. **Additional LLM Providers**
   - Claude integration
   - Local model support (Ollama)
   - Unified provider interface

2. **Code Safety**
   - AST validation
   - Whitelist enforcement
   - Sandboxed execution

3. **XGBoost Optimization**
   - Feature importance analysis
   - Cross-validation framework
   - Performance metrics

**Deliverable**: Multi-LLM CAAFE with robust safety

### Phase 3: Evaluation & Polish (Week 3)
**Focus**: Benchmarking and refinement

**Tasks**:
1. **Benchmark Framework**
   - Dataset collection (OpenML + Kaggle)
   - Baseline comparisons
   - Statistical testing

2. **Prompt Engineering**
   - XGBoost-specific prompts
   - Model-specific optimization
   - Few-shot examples

3. **Performance Analysis**
   - Feature quality metrics
   - Cost-performance analysis
   - Results visualization

**Deliverable**: Complete system with benchmarks

## XGBoost-Optimized Prompting

### Dataset-Aware Feature Generation
```python
XGBOOST_PROMPT_TEMPLATE = """
You are an expert data scientist specializing in XGBoost feature engineering.

Dataset: {dataset_description}
Target: {target_description}
Features: {feature_info}

Generate ONE high-quality feature that will improve XGBoost performance:

Key principles for XGBoost features:
1. Create non-linear interactions that trees can exploit
2. Handle missing values appropriately
3. Generate features with meaningful variance
4. Consider feature interactions XGBoost might miss

Provide:
- Python code (pandas operations only)
- Brief explanation of why this helps XGBoost
- Expected impact on model performance

Code format:
```python
# Feature: descriptive_name
# Why: Brief explanation of XGBoost benefit
df['new_feature'] = # single line of pandas code
```
"""
```

### XGBoost-Specific Feature Patterns
- **Interaction Features**: Multiplicative/ratio features for tree ensembles
- **Binning**: Categorical encodings that create clean splits
- **Aggregations**: Group statistics that capture non-linear patterns
- **Missing Value Indicators**: Explicit missing value handling
- **Target Encoding**: Statistical features (with proper validation)

## Streamlined Dependencies

### Core Requirements
```toml
[tool.poetry.dependencies]
python = "^3.9"
pandas = "^2.0"
xgboost = "^2.0"
scikit-learn = "^1.3"
openai = "^1.0"
anthropic = "^0.25"
numpy = "^1.24"

[tool.poetry.group.dev.dependencies]
pytest = "^7.0"
black = "^23.0"
ruff = "^0.1"
```

### Minimal Feature Set
- **Core Generation**: Feature creation and validation
- **Multi-LLM**: OpenAI, Claude, local models
- **XGBoost Integration**: Performance evaluation and feature importance
- **Safety**: Code validation and sandboxed execution
- **Benchmarking**: Comparison framework

## Key Implementation Details

### Simplified API
```python
# Quick usage
caafe = CAAFE(llm_provider="openai", model="gpt-4o")
X_enhanced = caafe.generate_features(
    X=train_df, 
    y=target, 
    dataset_description="Customer churn prediction dataset..."
)

# Train XGBoost with enhanced features
model = xgb.XGBClassifier()
model.fit(X_enhanced, y)
```

### XGBoost-Focused Evaluation
```python
def evaluate_feature_xgboost(X_orig, X_new, y, cv_folds=5):
    """Compare XGBoost performance with/without new feature"""
    
    # Original performance
    scores_orig = cross_val_score(
        XGBClassifier(**DEFAULT_PARAMS), X_orig, y, 
        cv=cv_folds, scoring='roc_auc'
    )
    
    # Enhanced performance  
    scores_new = cross_val_score(
        XGBClassifier(**DEFAULT_PARAMS), X_new, y,
        cv=cv_folds, scoring='roc_auc'
    )
    
    improvement = scores_new.mean() - scores_orig.mean()
    significance = ttest_rel(scores_new, scores_orig).pvalue
    
    return improvement, significance
```

### Streamlined Safety
```python
class SimpleCodeValidator:
    ALLOWED_OPERATIONS = {
        'pd.', 'np.', '.str.', '.dt.', '.fillna', '.groupby',
        '.transform', '.apply', '.map', '.replace', '.cut', '.qcut'
    }
    
    FORBIDDEN_PATTERNS = [
        'import', 'exec', 'eval', '__', 'subprocess', 'os.'
    ]
    
    def validate_code(self, code: str) -> bool:
        """Simple but effective validation"""
        # Check forbidden patterns
        # Validate basic syntax
        # Ensure pandas operations only
```

## Research Questions (Simplified)

1. **XGBoost Performance**: How much can LLM-generated features improve XGBoost vs baselines?
2. **Feature Quality**: What types of features do different LLMs generate for XGBoost?
3. **Cost-Effectiveness**: Which LLM provides best performance per dollar?
4. **Generalization**: Do generated features transfer across similar datasets?

## Success Criteria

### Technical Success
- [ ] Multi-LLM integration (OpenAI, Claude, local)
- [ ] Consistent XGBoost performance improvements
- [ ] Sub-5-minute feature generation
- [ ] Robust safety validation

### Research Success
- [ ] Statistically significant improvements over baselines
- [ ] Clear insights about LLM feature generation patterns
- [ ] Reproducible results across datasets

### Usability Success
- [ ] Simple, intuitive API
- [ ] Easy installation and setup
- [ ] Clear documentation and examples

## Getting Started

### Immediate Tasks
1. Set up minimal project structure
2. Implement core CAAFE class with XGBoost integration
3. Add OpenAI provider with XGBoost-optimized prompts
4. Create basic safety validation
5. Build simple evaluation framework

### Development Principles
- **Simplicity First**: Choose the simplest solution that works
- **XGBoost Optimized**: Every decision should benefit XGBoost performance
- **Clean Code**: Readable, maintainable, well-tested
- **Fast Iteration**: Quick feedback loops for feature generation

This streamlined approach focuses on building a production-ready tool that data scientists can immediately use to improve their XGBoost models with LLM-generated features. The emphasis is on practical results rather than academic completeness.