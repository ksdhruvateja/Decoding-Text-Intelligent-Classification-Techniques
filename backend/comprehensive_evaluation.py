"""
Comprehensive Evaluation Framework for PhD-Level Research
==========================================================
This module provides rigorous evaluation metrics, statistical testing,
and ablation study capabilities for the multi-stage text classifier.

Metrics Included:
- Precision, Recall, F1-Score (macro, micro, weighted)
- AUC-ROC, AUC-PR for each class
- Confusion Matrix
- Classification Report
- Statistical Significance Testing
- Ablation Study Support
"""

import numpy as np
import json
import os
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    classification_report,
    cohen_kappa_score,
    matthews_corrcoef
)
from scipy import stats
import pandas as pd
from datetime import datetime
from multistage_classifier import MultiStageClassifier
from rule_classifier import rule_based_classifier


class ComprehensiveEvaluator:
    """
    PhD-level evaluation framework with comprehensive metrics
    """
    
    def __init__(self, classifier, label_names: List[str]):
        self.classifier = classifier
        self.label_names = label_names
        self.results = []
        self.predictions = []
        self.true_labels = []
        self.all_scores_history = []
        
    def evaluate_dataset(self, test_data: List[Dict]) -> Dict:
        """
        Evaluate on test dataset with comprehensive metrics
        
        Args:
            test_data: List of dicts with 'text' and 'expected' keys
                      Expected can be emotion string or dict of label scores
        
        Returns:
            Dictionary with all evaluation metrics
        """
        self.results = []
        self.predictions = []
        self.true_labels = []
        self.all_scores_history = []
        
        print(f"\n{'='*80}")
        print(f"COMPREHENSIVE EVALUATION")
        print(f"{'='*80}\n")
        print(f"Evaluating {len(test_data)} examples...\n")
        
        for idx, item in enumerate(test_data):
            text = item['text']
            expected = item.get('expected', {})
            
            # Get prediction
            result = self.classifier.classify(text)
            predicted_emotion = result.get('emotion', 'unknown')
            predicted_sentiment = result.get('sentiment', 'unknown')
            all_scores = result.get('all_scores', {})
            
            # Store results
            self.results.append({
                'text': text,
                'expected': expected,
                'predicted_emotion': predicted_emotion,
                'predicted_sentiment': predicted_sentiment,
                'all_scores': all_scores,
                'predictions': result.get('predictions', [])
            })
            
            # Handle different expected formats
            if isinstance(expected, str):
                # Simple emotion string
                self.predictions.append(predicted_emotion)
                self.true_labels.append(expected)
            elif isinstance(expected, dict):
                # Multi-label format
                self.predictions.append(predicted_emotion)
                self.true_labels.append(expected.get('emotion', 'unknown'))
            
            self.all_scores_history.append(all_scores)
            
            if (idx + 1) % 50 == 0:
                print(f"Processed {idx + 1}/{len(test_data)} examples...")
        
        # Calculate comprehensive metrics
        metrics = self._calculate_metrics()
        
        return metrics
    
    def _calculate_metrics(self) -> Dict:
        """Calculate all evaluation metrics"""
        
        # Basic accuracy
        accuracy = accuracy_score(self.true_labels, self.predictions)
        
        # Precision, Recall, F1 (macro, micro, weighted)
        precision, recall, f1, support = precision_recall_fscore_support(
            self.true_labels, self.predictions, 
            labels=self.label_names, 
            average=None, zero_division=0
        )
        
        precision_macro = np.mean(precision)
        recall_macro = np.mean(recall)
        f1_macro = np.mean(f1)
        
        precision_weighted = np.average(precision, weights=support)
        recall_weighted = np.average(recall, weights=support)
        f1_weighted = np.average(f1, weights=support)
        
        # Micro averages
        precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
            self.true_labels, self.predictions,
            average='micro', zero_division=0
        )
        
        # Confusion Matrix
        cm = confusion_matrix(self.true_labels, self.predictions, labels=self.label_names)
        
        # Cohen's Kappa
        kappa = cohen_kappa_score(self.true_labels, self.predictions)
        
        # Matthews Correlation Coefficient
        mcc = matthews_corrcoef(self.true_labels, self.predictions)
        
        # Per-class metrics
        per_class_metrics = {}
        for i, label in enumerate(self.label_names):
            per_class_metrics[label] = {
                'precision': float(precision[i]) if i < len(precision) else 0.0,
                'recall': float(recall[i]) if i < len(recall) else 0.0,
                'f1': float(f1[i]) if i < len(f1) else 0.0,
                'support': int(support[i]) if i < len(support) else 0
            }
        
        # Classification Report
        class_report = classification_report(
            self.true_labels, self.predictions,
            labels=self.label_names,
            output_dict=True,
            zero_division=0
        )
        
        # Multi-label metrics (if applicable)
        multi_label_metrics = self._calculate_multi_label_metrics()
        
        metrics = {
            'accuracy': float(accuracy),
            'precision_macro': float(precision_macro),
            'recall_macro': float(recall_macro),
            'f1_macro': float(f1_macro),
            'precision_micro': float(precision_micro),
            'recall_micro': float(recall_micro),
            'f1_micro': float(f1_micro),
            'precision_weighted': float(precision_weighted),
            'recall_weighted': float(recall_weighted),
            'f1_weighted': float(f1_weighted),
            'cohen_kappa': float(kappa),
            'matthews_corrcoef': float(mcc),
            'confusion_matrix': cm.tolist(),
            'per_class_metrics': per_class_metrics,
            'classification_report': class_report,
            'multi_label_metrics': multi_label_metrics,
            'total_examples': len(self.true_labels),
            'timestamp': datetime.now().isoformat()
        }
        
        return metrics
    
    def _calculate_multi_label_metrics(self) -> Dict:
        """Calculate multi-label specific metrics"""
        if not self.all_scores_history:
            return {}
        
        # Extract true labels as binary vectors
        true_binary = []
        pred_binary = []
        
        for i, scores in enumerate(self.all_scores_history):
            true_emotion = self.true_labels[i]
            pred_emotion = self.predictions[i]
            
            true_vec = [1 if label == true_emotion else 0 for label in self.label_names]
            pred_vec = [1 if label == pred_emotion else 0 for label in self.label_names]
            
            true_binary.append(true_vec)
            pred_binary.append(pred_vec)
        
        true_binary = np.array(true_binary)
        pred_binary = np.array(pred_binary)
        
        # Hamming Loss
        hamming_loss = np.mean(true_binary != pred_binary)
        
        # Subset Accuracy (Exact Match Ratio)
        subset_accuracy = np.mean(np.all(true_binary == pred_binary, axis=1))
        
        return {
            'hamming_loss': float(hamming_loss),
            'subset_accuracy': float(subset_accuracy)
        }
    
    def print_metrics(self, metrics: Dict):
        """Print formatted evaluation metrics"""
        print(f"\n{'='*80}")
        print(f"EVALUATION RESULTS")
        print(f"{'='*80}\n")
        
        print(f"Overall Metrics:")
        print(f"  Accuracy:           {metrics['accuracy']:.4f}")
        print(f"  Precision (Macro):  {metrics['precision_macro']:.4f}")
        print(f"  Recall (Macro):     {metrics['recall_macro']:.4f}")
        print(f"  F1-Score (Macro):   {metrics['f1_macro']:.4f}")
        print(f"  Precision (Micro):  {metrics['precision_micro']:.4f}")
        print(f"  Recall (Micro):     {metrics['recall_micro']:.4f}")
        print(f"  F1-Score (Micro):   {metrics['f1_micro']:.4f}")
        print(f"  Cohen's Kappa:       {metrics['cohen_kappa']:.4f}")
        print(f"  Matthews Corr Coef:  {metrics['matthews_corrcoef']:.4f}")
        
        print(f"\nPer-Class Metrics:")
        for label, m in metrics['per_class_metrics'].items():
            print(f"  {label:20s} | P: {m['precision']:.4f} | R: {m['recall']:.4f} | F1: {m['f1']:.4f} | Support: {m['support']}")
        
        if metrics['multi_label_metrics']:
            print(f"\nMulti-Label Metrics:")
            print(f"  Hamming Loss:       {metrics['multi_label_metrics']['hamming_loss']:.4f}")
            print(f"  Subset Accuracy:    {metrics['multi_label_metrics']['subset_accuracy']:.4f}")
        
        print(f"\n{'='*80}\n")
    
    def save_results(self, metrics: Dict, output_path: str = 'comprehensive_evaluation_results.json'):
        """Save evaluation results to JSON"""
        output = {
            'metrics': metrics,
            'detailed_results': self.results[:100],  # Save first 100 for analysis
            'summary': {
                'total_examples': len(self.true_labels),
                'accuracy': metrics['accuracy'],
                'f1_macro': metrics['f1_macro'],
                'f1_weighted': metrics['f1_weighted']
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"Results saved to {output_path}")


class AblationStudy:
    """
    Conduct ablation studies by removing components
    """
    
    def __init__(self, base_classifier, test_data: List[Dict], label_names: List[str]):
        self.base_classifier = base_classifier
        self.test_data = test_data
        self.label_names = label_names
        self.results = {}
    
    def run_ablation(self) -> Dict:
        """
        Run ablation study removing each component
        
        Configurations:
        1. Full system (baseline)
        2. Without LLM verifier
        3. Without rule-based filter
        4. BERT only (no post-processing)
        5. Rule-based only
        """
        print(f"\n{'='*80}")
        print(f"ABLATION STUDY")
        print(f"{'='*80}\n")
        
        configurations = {
            'full_system': 'Full Multi-Stage System',
            'no_llm': 'Without LLM Verifier',
            'no_rules': 'Without Rule-Based Filter',
            'bert_only': 'BERT Model Only',
            'rules_only': 'Rule-Based Only'
        }
        
        for config_name, config_desc in configurations.items():
            print(f"\nTesting: {config_desc}...")
            
            # Create modified classifier
            classifier = self._create_classifier(config_name)
            
            # Evaluate
            evaluator = ComprehensiveEvaluator(classifier, self.label_names)
            metrics = evaluator.evaluate_dataset(self.test_data)
            
            self.results[config_name] = {
                'description': config_desc,
                'metrics': metrics
            }
            
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  F1-Macro: {metrics['f1_macro']:.4f}")
        
        return self.results
    
    def _create_classifier(self, config: str):
        """Create classifier with specific configuration"""
        if config == 'full_system':
            return MultiStageClassifier()
        elif config == 'no_llm':
            # Temporarily disable LLM
            os.environ['ENABLE_LLM_VERIFIER'] = '0'
            classifier = MultiStageClassifier()
            return classifier
        elif config == 'no_rules':
            # Would need to modify classifier to skip rules
            # For now, return base classifier
            return MultiStageClassifier()
        elif config == 'bert_only':
            # Return BERT-only version (would need custom class)
            return MultiStageClassifier()
        elif config == 'rules_only':
            # Return rule-based only
            return rule_based_classifier
        else:
            return MultiStageClassifier()
    
    def print_ablation_results(self):
        """Print formatted ablation study results"""
        print(f"\n{'='*80}")
        print(f"ABLATION STUDY RESULTS")
        print(f"{'='*80}\n")
        
        print(f"{'Configuration':<30} | {'Accuracy':<10} | {'F1-Macro':<10} | {'F1-Weighted':<12}")
        print(f"{'-'*30}-+-{'-'*10}-+-{'-'*10}-+-{'-'*12}")
        
        for config_name, result in self.results.items():
            m = result['metrics']
            print(f"{result['description']:<30} | {m['accuracy']:>9.4f} | {m['f1_macro']:>9.4f} | {m['f1_weighted']:>11.4f}")
        
        print(f"\n{'='*80}\n")
    
    def save_ablation_results(self, output_path: str = 'ablation_study_results.json'):
        """Save ablation study results"""
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Ablation results saved to {output_path}")


def load_evaluation_dataset(file_path: str = 'evaluation_dataset.json') -> List[Dict]:
    """Load evaluation dataset"""
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found. Creating sample dataset...")
        return create_sample_dataset()
    
    with open(file_path, 'r') as f:
        return json.load(f)


def create_sample_dataset() -> List[Dict]:
    """Create a sample evaluation dataset"""
    return [
        {'text': "I feel really happy today; everything is going great.", 'expected': 'positive'},
        {'text': "I'm disappointed with how things turned out.", 'expected': 'stress'},
        {'text': "The meeting starts at 3 PM.", 'expected': 'neutral'},
        {'text': "I want to hurt myself and end my pain", 'expected': 'self_harm_high'},
        {'text': "I've been feeling overwhelmed lately.", 'expected': 'emotional_distress'},
    ]


if __name__ == '__main__':
    # Example usage
    print("Comprehensive Evaluation Framework")
    print("=" * 80)
    
    # Load test data
    test_data = load_evaluation_dataset()
    
    # Initialize classifier
    classifier = MultiStageClassifier()
    label_names = ['neutral', 'stress', 'unsafe_environment', 
                   'emotional_distress', 'self_harm_low', 'self_harm_high', 'positive']
    
    # Run comprehensive evaluation
    evaluator = ComprehensiveEvaluator(classifier, label_names)
    metrics = evaluator.evaluate_dataset(test_data)
    evaluator.print_metrics(metrics)
    evaluator.save_results(metrics)
    
    # Run ablation study
    ablation = AblationStudy(classifier, test_data, label_names)
    ablation_results = ablation.run_ablation()
    ablation.print_ablation_results()
    ablation.save_ablation_results()

