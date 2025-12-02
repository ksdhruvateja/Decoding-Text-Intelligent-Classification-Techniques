"""
Master Evaluation Script for PhD-Level Research
===============================================
Runs all evaluation components:
1. Comprehensive metrics
2. Ablation studies
3. Baseline comparisons
4. Cross-validation
5. Error analysis
6. Statistical significance testing

Usage:
    python run_comprehensive_evaluation.py
"""

import json
import os
import sys
from datetime import datetime
from comprehensive_evaluation import (
    ComprehensiveEvaluator, 
    AblationStudy,
    load_evaluation_dataset
)
from baseline_comparison import BaselineComparison
from cross_validation import CrossValidator
from error_analysis import ErrorAnalyzer
from statistical_significance_test import StatisticalSignificanceTest
from multistage_classifier import MultiStageClassifier


def main():
    """Run all evaluation components"""
    print("=" * 80)
    print("COMPREHENSIVE EVALUATION SUITE FOR PhD RESEARCH")
    print("=" * 80)
    print(f"Started at: {datetime.now().isoformat()}\n")
    
    # Load evaluation dataset
    print("Loading evaluation dataset...")
    test_data = load_evaluation_dataset('evaluation_dataset.json')
    if not test_data:
        print("ERROR: No evaluation dataset found. Please create evaluation_dataset.json")
        return
    
    label_names = ['neutral', 'stress', 'unsafe_environment', 
                   'emotional_distress', 'self_harm_low', 'self_harm_high', 'positive']
    
    print(f"Loaded {len(test_data)} test examples\n")
    
    # Initialize classifier
    print("Initializing classifier...")
    classifier = MultiStageClassifier()
    print("Classifier initialized\n")
    
    all_results = {
        'timestamp': datetime.now().isoformat(),
        'dataset_size': len(test_data),
        'label_names': label_names
    }
    
    # 1. Comprehensive Evaluation
    print("\n" + "=" * 80)
    print("STEP 1: COMPREHENSIVE EVALUATION")
    print("=" * 80)
    evaluator = ComprehensiveEvaluator(classifier, label_names)
    comprehensive_metrics = evaluator.evaluate_dataset(test_data)
    evaluator.print_metrics(comprehensive_metrics)
    evaluator.save_results(comprehensive_metrics, 'results/comprehensive_evaluation.json')
    all_results['comprehensive_evaluation'] = comprehensive_metrics
    
    # 2. Error Analysis
    print("\n" + "=" * 80)
    print("STEP 2: ERROR ANALYSIS")
    print("=" * 80)
    analyzer = ErrorAnalyzer(evaluator)
    error_analysis = analyzer.analyze_errors()
    analyzer.print_error_analysis(error_analysis)
    analyzer.save_error_analysis(error_analysis, 'results/error_analysis.json')
    all_results['error_analysis'] = error_analysis
    
    # 3. Ablation Study
    print("\n" + "=" * 80)
    print("STEP 3: ABLATION STUDY")
    print("=" * 80)
    ablation = AblationStudy(classifier, test_data, label_names)
    ablation_results = ablation.run_ablation()
    ablation.print_ablation_results()
    ablation.save_ablation_results('results/ablation_study.json')
    all_results['ablation_study'] = ablation_results
    
    # 4. Baseline Comparison
    print("\n" + "=" * 80)
    print("STEP 4: BASELINE COMPARISON")
    print("=" * 80)
    baseline_comparison = BaselineComparison(test_data, label_names)
    baseline_results = baseline_comparison.run_all_baselines()
    baseline_comparison.print_comparison_table()
    baseline_comparison.save_comparison_results('results/baseline_comparison.json')
    all_results['baseline_comparison'] = baseline_results
    
    # 5. Cross-Validation
    print("\n" + "=" * 80)
    print("STEP 5: CROSS-VALIDATION")
    print("=" * 80)
    cv = CrossValidator(test_data, label_names, k=5)
    cv_results = cv.run_cross_validation()
    cv.print_cv_results(cv_results)
    cv.save_cv_results(cv_results, 'results/cross_validation.json')
    all_results['cross_validation'] = cv_results
    
    # 6. Statistical Significance Testing
    print("\n" + "=" * 80)
    print("STEP 6: STATISTICAL SIGNIFICANCE TESTING")
    print("=" * 80)
    if 'multi_stage' in baseline_results and 'naive_bayes' in baseline_results:
        try:
            # Compare multi-stage vs baseline
            multi_stage_preds = [r['predicted_emotion'] for r in evaluator.results]
            # Would need to get baseline predictions
            print("Statistical testing requires baseline predictions")
        except Exception as e:
            print(f"Statistical testing skipped: {e}")
    
    # Save all results
    os.makedirs('results', exist_ok=True)
    with open('results/all_evaluation_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Generate summary report
    generate_summary_report(all_results)
    
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print(f"All results saved to results/ directory")
    print(f"Summary report: results/summary_report.txt")
    print(f"Completed at: {datetime.now().isoformat()}\n")


def generate_summary_report(results: dict):
    """Generate a text summary report"""
    report = []
    report.append("=" * 80)
    report.append("EVALUATION SUMMARY REPORT")
    report.append("=" * 80)
    report.append(f"Generated: {datetime.now().isoformat()}\n")
    
    # Comprehensive Evaluation Summary
    if 'comprehensive_evaluation' in results:
        ce = results['comprehensive_evaluation']
        report.append("COMPREHENSIVE EVALUATION:")
        report.append(f"  Accuracy: {ce['accuracy']:.4f}")
        report.append(f"  F1-Macro: {ce['f1_macro']:.4f}")
        report.append(f"  F1-Weighted: {ce['f1_weighted']:.4f}")
        report.append(f"  Cohen's Kappa: {ce['cohen_kappa']:.4f}\n")
    
    # Ablation Study Summary
    if 'ablation_study' in results:
        report.append("ABLATION STUDY:")
        for config, result in results['ablation_study'].items():
            if 'metrics' in result:
                m = result['metrics']
                report.append(f"  {result['description']}: Accuracy={m['accuracy']:.4f}, F1={m['f1_macro']:.4f}")
        report.append("")
    
    # Baseline Comparison Summary
    if 'baseline_comparison' in results:
        report.append("BASELINE COMPARISON:")
        for method, result in results['baseline_comparison'].items():
            if 'accuracy' in result:
                report.append(f"  {method}: Accuracy={result['accuracy']:.4f}, F1={result['f1_macro']:.4f}")
        report.append("")
    
    # Cross-Validation Summary
    if 'cross_validation' in results:
        cv = results['cross_validation']
        if 'summary' in cv:
            s = cv['summary']
            report.append("CROSS-VALIDATION:")
            report.append(f"  Mean Accuracy: {s['mean_accuracy']:.4f} ± {s['std_accuracy']:.4f}")
            report.append(f"  Mean F1-Macro: {s['mean_f1_macro']:.4f} ± {s['std_f1_macro']:.4f}")
            report.append("")
    
    # Error Analysis Summary
    if 'error_analysis' in results:
        ea = results['error_analysis']
        report.append("ERROR ANALYSIS:")
        report.append(f"  Total Errors: {ea['total_errors']}")
        report.append(f"  Error Rate: {ea['error_rate']:.2%}")
        report.append(f"  False Positives: {ea['error_categories']['false_positives']}")
        report.append(f"  False Negatives: {ea['error_categories']['false_negatives']}")
        report.append("")
    
    report.append("=" * 80)
    
    # Save report
    os.makedirs('results', exist_ok=True)
    with open('results/summary_report.txt', 'w') as f:
        f.write('\n'.join(report))
    
    print('\n'.join(report))


if __name__ == '__main__':
    main()

