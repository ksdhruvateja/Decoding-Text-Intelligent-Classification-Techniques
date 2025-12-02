# Methodology Documentation

## Multi-Stage Text Classification System for Mental Health Detection

### 1. System Architecture

#### 1.1 Overview
The system employs a three-stage classification pipeline designed to maximize accuracy while minimizing false positives and false negatives in mental health text classification.

#### 1.2 Stage 1: Rule-Based Pre-Filtering
- **Purpose**: Initial sentiment analysis and critical pattern detection
- **Components**:
  - Positive keyword detection
  - Negative keyword detection
  - Critical self-harm pattern matching (regex)
  - Neutral/informational pattern detection
- **Output**: Base sentiment (positive/neutral/negative) with confidence score

#### 1.3 Stage 2: BERT-Based Classification
- **Model**: BERT-base-uncased with custom classification head
- **Architecture**:
  - Input: Text sequences (max 128 tokens)
  - Encoder: BERT transformer (12 layers, 768 hidden size)
  - Classification head: Linear layer with dropout (0.3)
  - Output: Multi-label probabilities for 6 classes
- **Classes**: 
  - neutral
  - stress
  - unsafe_environment
  - emotional_distress
  - self_harm_low
  - self_harm_high
- **Calibration**: Temperature scaling applied for probability calibration

#### 1.4 Stage 3: LLM Verification (Optional)
- **Model**: facebook/bart-large-mnli (zero-shot classification)
- **Purpose**: Final verification and override capability
- **Function**: Validates BERT predictions and can force safe/risk classifications

#### 1.5 Post-Processing
- **Threshold Application**: Class-specific thresholds
- **Rule Overrides**: Can override model predictions based on rule analysis
- **Final Classification**: Determines emotion and sentiment labels

### 2. Training Methodology

#### 2.1 Data Preparation
- **Dataset**: Mental health text classification dataset
- **Preprocessing**:
  - Text normalization
  - Tokenization using BERT tokenizer
  - Sequence padding/truncation to 128 tokens
- **Split**: Train (80%) / Validation (20%)

#### 2.2 Model Training
- **Optimizer**: AdamW
- **Learning Rate**: 2e-5
- **Batch Size**: 16
- **Epochs**: 5 (with early stopping)
- **Loss Function**: Binary Cross-Entropy with Logits
- **Regularization**: Dropout (0.3)

#### 2.3 Calibration
- **Method**: Temperature Scaling
- **Purpose**: Calibrate probability outputs for better threshold application
- **Validation**: Calibrated on validation set

### 3. Evaluation Methodology

#### 3.1 Metrics
- **Primary Metrics**:
  - Accuracy
  - Precision (macro, micro, weighted)
  - Recall (macro, micro, weighted)
  - F1-Score (macro, micro, weighted)
- **Secondary Metrics**:
  - Cohen's Kappa
  - Matthews Correlation Coefficient
  - Per-class metrics
  - Multi-label metrics (Hamming Loss, Subset Accuracy)

#### 3.2 Evaluation Procedures
1. **Comprehensive Evaluation**: Full test set evaluation
2. **Cross-Validation**: 5-fold stratified cross-validation
3. **Ablation Studies**: Remove components to measure contribution
4. **Baseline Comparison**: Compare against standard ML methods
5. **Error Analysis**: Detailed analysis of failure cases
6. **Statistical Testing**: Significance tests between models

#### 3.3 Threshold Optimization
- **Method**: Class-specific thresholds based on validation performance
- **Thresholds**:
  - self_harm_high: 0.85 (very high bar)
  - self_harm_low: 0.65 (high bar)
  - unsafe_environment: 0.70 (high bar)
  - emotional_distress: 0.40 (lower bar)
  - stress: 0.50 (medium bar)
  - neutral: 0.45 (medium-low bar)

### 4. Ablation Study Design

#### 4.1 Configurations Tested
1. **Full System**: All components active
2. **No LLM Verifier**: Disable LLM verification stage
3. **No Rule Filter**: Disable rule-based pre-filtering
4. **BERT Only**: Pure BERT model without post-processing
5. **Rule-Based Only**: Only rule-based classification

#### 4.2 Analysis
- Measure performance degradation when removing each component
- Identify critical components
- Understand contribution of each stage

### 5. Baseline Methods

#### 5.1 Baselines Compared
1. **Naive Bayes**: MultinomialNB with TF-IDF
2. **SVM**: Support Vector Machine with linear kernel and TF-IDF
3. **Random Forest**: 100 trees with TF-IDF features
4. **Logistic Regression**: Linear classifier with TF-IDF
5. **Rule-Based Only**: Pure rule-based classifier
6. **BERT Only**: BERT without multi-stage processing

#### 5.2 Comparison Metrics
- Same metrics as main evaluation
- Statistical significance testing
- Performance ranking

### 6. Statistical Analysis

#### 6.1 Significance Tests
- **McNemar's Test**: For comparing error rates between models
- **Paired t-test**: For comparing continuous scores
- **Wilcoxon Signed-Rank Test**: Non-parametric alternative

#### 6.2 Confidence Intervals
- 95% confidence intervals for cross-validation results
- Standard deviation reporting
- Mean ± std format

### 7. Error Analysis

#### 7.1 Error Categories
- **False Positives**: Safe text classified as risk
- **False Negatives**: Risk text classified as safe
- **Class Confusion**: Wrong class but correct risk level

#### 7.2 Analysis Dimensions
- Confusion matrix analysis
- Text characteristics of errors
- Common error patterns
- Recommendations for improvement

### 8. Reproducibility

#### 8.1 Random Seeds
- All random operations use seed=42
- Ensures reproducibility

#### 8.2 Data Splits
- Fixed train/validation/test splits
- Stratified sampling where applicable

#### 8.3 Hyperparameters
- All hyperparameters documented
- No hyperparameter tuning on test set

### 9. Limitations

#### 9.1 Known Limitations
- Model trained on specific dataset (may not generalize)
- LLM verifier requires significant computational resources
- Thresholds may need adjustment for different domains

#### 9.2 Future Work
- Domain adaptation
- Active learning for threshold optimization
- Multi-lingual support
- Real-time deployment optimization

### 10. Ethical Considerations

#### 10.1 Mental Health Classification
- High-stakes application requiring careful validation
- False negatives more critical than false positives
- Need for human oversight in production

#### 10.2 Bias Considerations
- Potential bias in training data
- Need for diverse dataset
- Regular bias audits recommended

---

**Document Version**: 1.0  
**Last Updated**: 2024  
**Author**: Research Team

