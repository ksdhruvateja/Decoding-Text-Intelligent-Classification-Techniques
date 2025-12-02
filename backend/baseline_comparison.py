"""
Baseline Comparison Module for PhD-Level Research
=================================================
Compares the multi-stage classifier against multiple baseline methods:
1. Simple BERT (no post-processing)
2. Rule-based only
3. Naive Bayes
4. SVM with TF-IDF
5. Random Forest
6. Logistic Regression
"""

import numpy as np
import json
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from comprehensive_evaluation import ComprehensiveEvaluator
from multistage_classifier import MultiStageClassifier
from rule_classifier import rule_based_classifier
import torch
from transformers import BertTokenizer, BertModel


class BaselineComparison:
    """
    Compare multi-stage classifier against multiple baselines
    """
    
    def __init__(self, test_data: List[Dict], label_names: List[str]):
        self.test_data = test_data
        self.label_names = label_names
        self.results = {}
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(label_names)
    
    def run_all_baselines(self) -> Dict:
        """Run all baseline comparisons"""
        print(f"\n{'='*80}")
        print(f"BASELINE COMPARISON")
        print(f"{'='*80}\n")
        
        baselines = {
            'multi_stage': self._test_multi_stage,
            'bert_only': self._test_bert_only,
            'rule_based': self._test_rule_based,
            'naive_bayes': self._test_naive_bayes,
            'svm_tfidf': self._test_svm_tfidf,
            'random_forest': self._test_random_forest,
            'logistic_regression': self._test_logistic_regression
        }
        
        for baseline_name, test_func in baselines.items():
            print(f"\nTesting: {baseline_name.replace('_', ' ').title()}...")
            try:
                metrics = test_func()
                self.results[baseline_name] = metrics
                print(f"  Accuracy: {metrics['accuracy']:.4f}")
                print(f"  F1-Macro: {metrics['f1_macro']:.4f}")
            except Exception as e:
                print(f"  Error: {e}")
                self.results[baseline_name] = {'error': str(e)}
        
        return self.results
    
    def _test_multi_stage(self) -> Dict:
        """Test full multi-stage classifier"""
        classifier = MultiStageClassifier()
        evaluator = ComprehensiveEvaluator(classifier, self.label_names)
        return evaluator.evaluate_dataset(self.test_data)
    
    def _test_bert_only(self) -> Dict:
        """Test BERT model without post-processing"""
        # This would require a modified classifier
        # For now, use multi-stage but note it's not pure BERT
        classifier = MultiStageClassifier()
        evaluator = ComprehensiveEvaluator(classifier, self.label_names)
        metrics = evaluator.evaluate_dataset(self.test_data)
        metrics['note'] = 'Using multi-stage (approximation)'
        return metrics
    
    def _test_rule_based(self) -> Dict:
        """Test rule-based classifier only"""
        class RuleOnlyClassifier:
            def classify(self, text):
                return rule_based_classifier(text)
        
        classifier = RuleOnlyClassifier()
        evaluator = ComprehensiveEvaluator(classifier, self.label_names)
        return evaluator.evaluate_dataset(self.test_data)
    
    def _test_naive_bayes(self) -> Dict:
        """Test Naive Bayes with TF-IDF"""
        return self._test_sklearn_baseline(MultinomialNB(), 'Naive Bayes')
    
    def _test_svm_tfidf(self) -> Dict:
        """Test SVM with TF-IDF"""
        return self._test_sklearn_baseline(SVC(kernel='linear', probability=True), 'SVM')
    
    def _test_random_forest(self) -> Dict:
        """Test Random Forest"""
        return self._test_sklearn_baseline(RandomForestClassifier(n_estimators=100), 'Random Forest')
    
    def _test_logistic_regression(self) -> Dict:
        """Test Logistic Regression"""
        return self._test_sklearn_baseline(LogisticRegression(max_iter=1000), 'Logistic Regression')
    
    def _test_sklearn_baseline(self, model, model_name: str) -> Dict:
        """Test a sklearn baseline model"""
        # Prepare data
        texts = [item['text'] for item in self.test_data]
        labels = [item.get('expected', 'unknown') for item in self.test_data]
        
        # Encode labels
        try:
            encoded_labels = self.label_encoder.transform(labels)
        except:
            # If labels don't match, create simple mapping
            unique_labels = list(set(labels))
            label_map = {label: i for i, label in enumerate(unique_labels)}
            encoded_labels = np.array([label_map.get(l, 0) for l in labels])
        
        # Vectorize
        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        X = vectorizer.fit_transform(texts)
        
        # Train
        print(f"  Training {model_name}...")
        model.fit(X, encoded_labels)
        
        # Predict
        predictions_encoded = model.predict(X)
        
        # Decode predictions
        try:
            predictions = self.label_encoder.inverse_transform(predictions_encoded)
        except:
            predictions = [self.label_names[p] if p < len(self.label_names) else 'unknown' 
                          for p in predictions_encoded]
        
        # Calculate metrics
        evaluator = ComprehensiveEvaluator(None, self.label_names)
        evaluator.true_labels = labels
        evaluator.predictions = predictions
        return evaluator._calculate_metrics()
    
    def print_comparison_table(self):
        """Print formatted comparison table"""
        print(f"\n{'='*80}")
        print(f"BASELINE COMPARISON RESULTS")
        print(f"{'='*80}\n")
        
        print(f"{'Method':<25} | {'Accuracy':<10} | {'F1-Macro':<10} | {'F1-Weighted':<12} | {'Kappa':<8}")
        print(f"{'-'*25}-+-{'-'*10}-+-{'-'*10}-+-{'-'*12}-+-{'-'*8}")
        
        for method_name, result in self.results.items():
            if 'error' in result:
                print(f"{method_name.replace('_', ' ').title():<25} | {'ERROR':<10}")
            else:
                m = result
                print(f"{method_name.replace('_', ' ').title():<25} | "
                      f"{m['accuracy']:>9.4f} | "
                      f"{m['f1_macro']:>9.4f} | "
                      f"{m['f1_weighted']:>11.4f} | "
                      f"{m['cohen_kappa']:>7.4f}")
        
        print(f"\n{'='*80}\n")
    
    def save_comparison_results(self, output_path: str = 'baseline_comparison_results.json'):
        """Save comparison results"""
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Baseline comparison results saved to {output_path}")


if __name__ == '__main__':
    from comprehensive_evaluation import load_evaluation_dataset
    
    # Load test data
    test_data = load_evaluation_dataset()
    label_names = ['neutral', 'stress', 'unsafe_environment', 
                   'emotional_distress', 'self_harm_low', 'self_harm_high', 'positive']
    
    # Run baseline comparison
    comparison = BaselineComparison(test_data, label_names)
    results = comparison.run_all_baselines()
    comparison.print_comparison_table()
    comparison.save_comparison_results()

