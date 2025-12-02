"""
Error Analysis Module for PhD-Level Research
============================================
Performs detailed error analysis to identify:
- Common failure patterns
- Confusion between classes
- Text characteristics of errors
- Model limitations
"""

import json
import re
from typing import List, Dict, Tuple
from collections import defaultdict, Counter
import pandas as pd
from comprehensive_evaluation import ComprehensiveEvaluator


class ErrorAnalyzer:
    """
    Analyze classification errors in detail
    """
    
    def __init__(self, evaluator: ComprehensiveEvaluator):
        self.evaluator = evaluator
        self.errors = []
        self.error_patterns = defaultdict(list)
    
    def analyze_errors(self) -> Dict:
        """Perform comprehensive error analysis"""
        print(f"\n{'='*80}")
        print(f"ERROR ANALYSIS")
        print(f"{'='*80}\n")
        
        # Identify errors
        self._identify_errors()
        
        # Analyze error patterns
        confusion_analysis = self._analyze_confusion()
        text_characteristics = self._analyze_text_characteristics()
        error_categories = self._categorize_errors()
        
        analysis = {
            'total_errors': len(self.errors),
            'error_rate': len(self.errors) / len(self.evaluator.true_labels),
            'confusion_analysis': confusion_analysis,
            'text_characteristics': text_characteristics,
            'error_categories': error_categories,
            'common_errors': self._get_common_errors(),
            'recommendations': self._generate_recommendations()
        }
        
        return analysis
    
    def _identify_errors(self):
        """Identify all classification errors"""
        self.errors = []
        for i, result in enumerate(self.evaluator.results):
            true_label = self.evaluator.true_labels[i]
            pred_label = self.evaluator.predictions[i]
            
            if true_label != pred_label:
                self.errors.append({
                    'index': i,
                    'text': result['text'],
                    'true_label': true_label,
                    'predicted_label': pred_label,
                    'all_scores': result.get('all_scores', {}),
                    'predictions': result.get('predictions', [])
                })
    
    def _analyze_confusion(self) -> Dict:
        """Analyze confusion between classes"""
        confusion_pairs = defaultdict(int)
        
        for error in self.errors:
            pair = (error['true_label'], error['predicted_label'])
            confusion_pairs[pair] += 1
        
        # Sort by frequency
        sorted_confusion = sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'confusion_matrix_pairs': dict(confusion_pairs),
            'most_common_confusions': sorted_confusion[:10],
            'total_confusion_types': len(confusion_pairs)
        }
    
    def _analyze_text_characteristics(self) -> Dict:
        """Analyze characteristics of texts that cause errors"""
        error_texts = [e['text'] for e in self.errors]
        correct_texts = [r['text'] for i, r in enumerate(self.evaluator.results) 
                        if i not in [e['index'] for e in self.errors]]
        
        error_stats = self._calculate_text_stats(error_texts)
        correct_stats = self._calculate_text_stats(correct_texts)
        
        return {
            'error_texts': {
                'avg_length': error_stats['avg_length'],
                'avg_words': error_stats['avg_words'],
                'common_words': error_stats['common_words'][:10]
            },
            'correct_texts': {
                'avg_length': correct_stats['avg_length'],
                'avg_words': correct_stats['avg_words'],
                'common_words': correct_stats['common_words'][:10]
            },
            'differences': {
                'length_diff': error_stats['avg_length'] - correct_stats['avg_length'],
                'word_diff': error_stats['avg_words'] - correct_stats['avg_words']
            }
        }
    
    def _calculate_text_stats(self, texts: List[str]) -> Dict:
        """Calculate statistics for a set of texts"""
        lengths = [len(text) for text in texts]
        word_counts = [len(text.split()) for text in texts]
        
        # Common words
        all_words = []
        for text in texts:
            words = re.findall(r'\b\w+\b', text.lower())
            all_words.extend(words)
        
        word_freq = Counter(all_words)
        
        return {
            'avg_length': sum(lengths) / len(lengths) if lengths else 0,
            'avg_words': sum(word_counts) / len(word_counts) if word_counts else 0,
            'common_words': word_freq.most_common(20)
        }
    
    def _categorize_errors(self) -> Dict:
        """Categorize errors by type"""
        categories = {
            'false_positives': [],  # Predicted risk when safe
            'false_negatives': [],  # Predicted safe when risk
            'class_confusion': []   # Wrong class but same risk level
        }
        
        risk_labels = ['self_harm_high', 'self_harm_low', 'unsafe_environment', 'emotional_distress']
        safe_labels = ['neutral', 'positive', 'conversational']
        
        for error in self.errors:
            true_label = error['true_label']
            pred_label = error['predicted_label']
            
            true_is_risk = true_label in risk_labels
            pred_is_risk = pred_label in risk_labels
            
            if not true_is_risk and pred_is_risk:
                categories['false_positives'].append(error)
            elif true_is_risk and not pred_is_risk:
                categories['false_negatives'].append(error)
            else:
                categories['class_confusion'].append(error)
        
        return {
            'false_positives': len(categories['false_positives']),
            'false_negatives': len(categories['false_negatives']),
            'class_confusion': len(categories['class_confusion']),
            'examples': {
                'false_positives': categories['false_positives'][:5],
                'false_negatives': categories['false_negatives'][:5],
                'class_confusion': categories['class_confusion'][:5]
            }
        }
    
    def _get_common_errors(self, n: int = 10) -> List[Dict]:
        """Get most common error patterns"""
        return self.errors[:n]
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on error analysis"""
        recommendations = []
        
        if len(self.errors) > 0:
            error_rate = len(self.errors) / len(self.evaluator.true_labels)
            
            if error_rate > 0.3:
                recommendations.append("High error rate detected. Consider retraining with more data.")
            
            confusion_analysis = self._analyze_confusion()
            if confusion_analysis['total_confusion_types'] > 5:
                recommendations.append("Multiple confusion patterns detected. Consider class-specific thresholds.")
            
            error_categories = self._categorize_errors()
            if error_categories['false_negatives'] > error_categories['false_positives']:
                recommendations.append("More false negatives than false positives. Lower thresholds for risk detection.")
            elif error_categories['false_positives'] > error_categories['false_negatives']:
                recommendations.append("More false positives than false negatives. Increase thresholds for risk detection.")
        
        return recommendations
    
    def print_error_analysis(self, analysis: Dict):
        """Print formatted error analysis"""
        print(f"\nError Summary:")
        print(f"  Total Errors: {analysis['total_errors']}")
        print(f"  Error Rate: {analysis['error_rate']:.2%}")
        
        print(f"\nConfusion Analysis:")
        print(f"  Total Confusion Types: {analysis['confusion_analysis']['total_confusion_types']}")
        print(f"  Most Common Confusions:")
        for (true, pred), count in analysis['confusion_analysis']['most_common_confusions'][:5]:
            print(f"    {true} -> {pred}: {count} times")
        
        print(f"\nError Categories:")
        print(f"  False Positives: {analysis['error_categories']['false_positives']}")
        print(f"  False Negatives: {analysis['error_categories']['false_negatives']}")
        print(f"  Class Confusion: {analysis['error_categories']['class_confusion']}")
        
        print(f"\nRecommendations:")
        for rec in analysis['recommendations']:
            print(f"  - {rec}")
        
        print(f"\n{'='*80}\n")
    
    def save_error_analysis(self, analysis: Dict, output_path: str = 'error_analysis_results.json'):
        """Save error analysis results"""
        with open(output_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        print(f"Error analysis saved to {output_path}")


if __name__ == '__main__':
    from comprehensive_evaluation import load_evaluation_dataset, ComprehensiveEvaluator
    from multistage_classifier import MultiStageClassifier
    
    # Load and evaluate
    test_data = load_evaluation_dataset()
    classifier = MultiStageClassifier()
    label_names = ['neutral', 'stress', 'unsafe_environment', 
                   'emotional_distress', 'self_harm_low', 'self_harm_high', 'positive']
    
    evaluator = ComprehensiveEvaluator(classifier, label_names)
    evaluator.evaluate_dataset(test_data)
    
    # Analyze errors
    analyzer = ErrorAnalyzer(evaluator)
    analysis = analyzer.analyze_errors()
    analyzer.print_error_analysis(analysis)
    analyzer.save_error_analysis(analysis)

