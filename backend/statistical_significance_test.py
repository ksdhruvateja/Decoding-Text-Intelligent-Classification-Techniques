"""
Statistical Significance Testing Module
=======================================
Implements statistical tests for comparing models:
- McNemar's test
- Paired t-test
- Wilcoxon signed-rank test
"""

import numpy as np
from scipy import stats
from typing import List, Tuple, Dict


class StatisticalSignificanceTest:
    """
    Perform statistical significance tests between models
    """
    
    @staticmethod
    def mcnemar_test(y_true: List, y_pred1: List, y_pred2: List) -> Dict:
        """
        McNemar's test for paired nominal data
        Tests if two models have significantly different error rates
        
        Args:
            y_true: True labels
            y_pred1: Predictions from model 1
            y_pred2: Predictions from model 2
        
        Returns:
            Dictionary with test results
        """
        # Create contingency table
        both_correct = sum(1 for i in range(len(y_true)) 
                          if y_true[i] == y_pred1[i] and y_true[i] == y_pred2[i])
        both_wrong = sum(1 for i in range(len(y_true)) 
                        if y_true[i] != y_pred1[i] and y_true[i] != y_pred2[i])
        model1_correct = sum(1 for i in range(len(y_true)) 
                            if y_true[i] == y_pred1[i] and y_true[i] != y_pred2[i])
        model2_correct = sum(1 for i in range(len(y_true)) 
                            if y_true[i] != y_pred1[i] and y_true[i] == y_pred2[i])
        
        contingency = [[both_correct, model1_correct],
                      [model2_correct, both_wrong]]
        
        # McNemar's test
        try:
            result = stats.mcnemar(contingency, exact=False, correction=True)
            significant = result.pvalue < 0.05
        except:
            # Use exact test if needed
            result = stats.mcnemar(contingency, exact=True)
            significant = result.pvalue < 0.05
        
        return {
            'test_name': "McNemar's Test",
            'statistic': float(result.statistic),
            'p_value': float(result.pvalue),
            'significant': significant,
            'contingency_table': contingency,
            'interpretation': f"Models are {'significantly' if significant else 'not significantly'} different (p={result.pvalue:.4f})"
        }
    
    @staticmethod
    def paired_t_test(scores1: List[float], scores2: List[float]) -> Dict:
        """
        Paired t-test for comparing two models on continuous scores
        
        Args:
            scores1: Scores from model 1
            scores2: Scores from model 2
        
        Returns:
            Dictionary with test results
        """
        if len(scores1) != len(scores2):
            raise ValueError("Score lists must have same length")
        
        differences = [s1 - s2 for s1, s2 in zip(scores1, scores2)]
        
        t_stat, p_value = stats.ttest_rel(scores1, scores2)
        
        return {
            'test_name': 'Paired t-test',
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significant': p_value < 0.05,
            'mean_difference': float(np.mean(differences)),
            'std_difference': float(np.std(differences)),
            'interpretation': f"Model 1 is {'significantly' if p_value < 0.05 else 'not significantly'} {'better' if np.mean(differences) > 0 else 'worse'} than Model 2 (p={p_value:.4f})"
        }
    
    @staticmethod
    def wilcoxon_test(scores1: List[float], scores2: List[float]) -> Dict:
        """
        Wilcoxon signed-rank test (non-parametric alternative to paired t-test)
        
        Args:
            scores1: Scores from model 1
            scores2: Scores from model 2
        
        Returns:
            Dictionary with test results
        """
        if len(scores1) != len(scores2):
            raise ValueError("Score lists must have same length")
        
        statistic, p_value = stats.wilcoxon(scores1, scores2, alternative='two-sided')
        
        return {
            'test_name': 'Wilcoxon Signed-Rank Test',
            'statistic': float(statistic),
            'p_value': float(p_value),
            'significant': p_value < 0.05,
            'interpretation': f"Models are {'significantly' if p_value < 0.05 else 'not significantly'} different (p={p_value:.4f})"
        }
    
    @staticmethod
    def compare_models(y_true: List, pred1: List, pred2: List, 
                     scores1: List[float] = None, scores2: List[float] = None) -> Dict:
        """
        Comprehensive comparison between two models
        
        Args:
            y_true: True labels
            pred1: Predictions from model 1
            pred2: Predictions from model 2
            scores1: Optional confidence scores from model 1
            scores2: Optional confidence scores from model 2
        
        Returns:
            Dictionary with all test results
        """
        results = {}
        
        # McNemar's test
        results['mcnemar'] = StatisticalSignificanceTest.mcnemar_test(y_true, pred1, pred2)
        
        # If scores provided, do additional tests
        if scores1 and scores2:
            results['paired_t'] = StatisticalSignificanceTest.paired_t_test(scores1, scores2)
            results['wilcoxon'] = StatisticalSignificanceTest.wilcoxon_test(scores1, scores2)
        
        return results
    
    @staticmethod
    def print_test_results(results: Dict):
        """Print formatted test results"""
        print("\n" + "=" * 80)
        print("STATISTICAL SIGNIFICANCE TEST RESULTS")
        print("=" * 80 + "\n")
        
        for test_name, result in results.items():
            print(f"{result['test_name']}:")
            print(f"  Statistic: {result.get('statistic', result.get('t_statistic', 'N/A')):.4f}")
            print(f"  P-value: {result['p_value']:.4f}")
            print(f"  Significant: {'Yes' if result['significant'] else 'No'} (α=0.05)")
            print(f"  Interpretation: {result['interpretation']}")
            print()


if __name__ == '__main__':
    # Example usage
    y_true = ['positive', 'negative', 'neutral', 'positive', 'negative']
    pred1 = ['positive', 'negative', 'neutral', 'positive', 'neutral']
    pred2 = ['positive', 'neutral', 'neutral', 'positive', 'negative']
    
    results = StatisticalSignificanceTest.compare_models(y_true, pred1, pred2)
    StatisticalSignificanceTest.print_test_results(results)

