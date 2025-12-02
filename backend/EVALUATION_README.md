# Comprehensive Evaluation Framework

This directory contains a complete evaluation suite designed for PhD-level research on the multi-stage text classification system.

## Overview

The evaluation framework provides:
- **Comprehensive Metrics**: All standard classification metrics plus advanced metrics
- **Ablation Studies**: Measure contribution of each system component
- **Baseline Comparisons**: Compare against standard ML methods
- **Cross-Validation**: Robust k-fold validation with confidence intervals
- **Error Analysis**: Detailed analysis of failure cases
- **Statistical Testing**: Significance tests between models

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Evaluation Dataset

Create `evaluation_dataset.json` with format:
```json
[
  {
    "text": "Example text here",
    "expected": "neutral"
  },
  ...
]
```

### 3. Run Complete Evaluation

```bash
python run_comprehensive_evaluation.py
```

This will run all evaluation components and save results to `results/` directory.

## Individual Components

### Comprehensive Evaluation

```python
from comprehensive_evaluation import ComprehensiveEvaluator, load_evaluation_dataset
from multistage_classifier import MultiStageClassifier

test_data = load_evaluation_dataset()
classifier = MultiStageClassifier()
label_names = ['neutral', 'stress', 'unsafe_environment', 
               'emotional_distress', 'self_harm_low', 'self_harm_high', 'positive']

evaluator = ComprehensiveEvaluator(classifier, label_names)
metrics = evaluator.evaluate_dataset(test_data)
evaluator.print_metrics(metrics)
evaluator.save_results(metrics)
```

### Ablation Study

```python
from comprehensive_evaluation import AblationStudy

ablation = AblationStudy(classifier, test_data, label_names)
results = ablation.run_ablation()
ablation.print_ablation_results()
ablation.save_ablation_results()
```

### Baseline Comparison

```python
from baseline_comparison import BaselineComparison

comparison = BaselineComparison(test_data, label_names)
results = comparison.run_all_baselines()
comparison.print_comparison_table()
comparison.save_comparison_results()
```

### Cross-Validation

```python
from cross_validation import CrossValidator

cv = CrossValidator(test_data, label_names, k=5)
results = cv.run_cross_validation()
cv.print_cv_results(results)
cv.save_cv_results(results)
```

### Error Analysis

```python
from error_analysis import ErrorAnalyzer

analyzer = ErrorAnalyzer(evaluator)
analysis = analyzer.analyze_errors()
analyzer.print_error_analysis(analysis)
analyzer.save_error_analysis(analysis)
```

### Statistical Significance Testing

```python
from statistical_significance_test import StatisticalSignificanceTest

results = StatisticalSignificanceTest.compare_models(
    y_true, pred1, pred2, scores1, scores2
)
StatisticalSignificanceTest.print_test_results(results)
```

## Output Files

All results are saved to the `results/` directory:

- `comprehensive_evaluation.json`: Full evaluation metrics
- `ablation_study_results.json`: Ablation study results
- `baseline_comparison_results.json`: Baseline comparison
- `cross_validation_results.json`: Cross-validation results
- `error_analysis_results.json`: Error analysis
- `all_evaluation_results.json`: Combined results
- `summary_report.txt`: Human-readable summary

## Metrics Explained

### Primary Metrics
- **Accuracy**: Overall correctness
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall

### Aggregation Methods
- **Macro**: Average across classes (unweighted)
- **Micro**: Global calculation across all samples
- **Weighted**: Average weighted by class support

### Advanced Metrics
- **Cohen's Kappa**: Agreement beyond chance
- **Matthews Correlation Coefficient**: Balanced measure for binary/multi-class
- **Confusion Matrix**: Per-class error patterns

## For PhD Research

This framework is designed to meet PhD-level research requirements:

1. **Rigorous Evaluation**: Multiple metrics and validation methods
2. **Reproducibility**: Fixed seeds, documented procedures
3. **Statistical Validity**: Significance testing, confidence intervals
4. **Comprehensive Analysis**: Ablation studies, error analysis
5. **Baseline Comparison**: Standard methods for context
6. **Documentation**: Methodology and report templates included

## Next Steps

1. **Expand Dataset**: Add more evaluation examples
2. **Run Evaluations**: Execute all components
3. **Analyze Results**: Review error patterns and metrics
4. **Write Report**: Use `RESEARCH_REPORT_TEMPLATE.md`
5. **Iterate**: Improve based on findings

## Troubleshooting

### Import Errors
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check Python path includes backend directory

### Missing Dataset
- Create `evaluation_dataset.json` with test examples
- Or use the sample dataset created automatically

### Memory Issues
- Reduce dataset size for testing
- Use smaller k for cross-validation

## Contact

For questions or issues, refer to the main project documentation.

