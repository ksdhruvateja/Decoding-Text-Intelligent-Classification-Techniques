# Training Improvements for Better Scores and Metrics

## Issues Identified and Fixed

### 1. ✅ **Class Imbalance Problem**
**Issue:** Your dataset has imbalanced classes:
- neutral: 21.8%
- stress: 17.8%
- unsafe_environment: 14.8%
- emotional_distress: 18.4%
- self_harm_low: 13.2%
- self_harm_high: 14.1%

**Why it hurts scores:** The model learns to predict majority classes more often, ignoring minority classes.

**Fix Applied:**
```python
# Calculate class weights using inverse frequency
weights = total_samples / (n_classes * positive_samples_per_class)
criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
```
This makes the model pay more attention to underrepresented classes.

---

### 2. ✅ **Metrics Calculation Issues**
**Issue:** The old metrics calculation had:
- Division by zero errors when no true positives exist
- NaN values propagating through calculations
- No handling of edge cases

**Why it hurts scores:** Incorrect metrics don't reflect true model performance.

**Fix Applied:**
```python
# Add epsilon to prevent division by zero
epsilon = 1e-10
precision = tp / (tp + fp + epsilon)
recall = tp / (tp + fn + epsilon)
f1 = 2 * precision * recall / (precision + recall + epsilon)

# Handle NaN values explicitly
precision = torch.nan_to_num(precision, nan=0.0)
recall = torch.nan_to_num(recall, nan=0.0)
f1 = torch.nan_to_num(f1, nan=0.0)
```

---

### 3. ✅ **No Data Augmentation**
**Issue:** Small dataset (583 training samples) leads to overfitting.

**Why it hurts scores:** Model memorizes training data instead of learning patterns.

**Fix Applied:**
```python
def augment_text(self, text):
    # Random word dropout (10% of words)
    # Helps model generalize better
    if random.random() > 0.3:
        return text
    words = text.split()
    num_to_drop = max(1, int(len(words) * 0.1))
    # Drop random words
```

---

### 4. ✅ **Suboptimal Model Architecture**
**Issue:** 
- Single dropout layer
- No layer normalization
- Poor weight initialization

**Why it hurts scores:** Model can't learn complex patterns effectively.

**Fix Applied:**
```python
class ImprovedBERTClassifier(nn.Module):
    def __init__(self):
        # Layer normalization for stable training
        self.layer_norm = nn.LayerNorm(768)
        
        # Multi-layer classifier (768 -> 384 -> 192 -> 6)
        self.fc1 = nn.Linear(768, 384)
        self.fc2 = nn.Linear(384, 192)
        self.fc3 = nn.Linear(192, n_classes)
        
        # Multiple dropout layers
        self.dropout = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.15)
        
        # Xavier initialization for better convergence
        nn.init.xavier_uniform_(self.fc1.weight)
```

---

### 5. ✅ **Fixed Threshold Problem**
**Issue:** Using 0.5 threshold for all classes doesn't work for imbalanced data.

**Why it hurts scores:** Some classes need different thresholds to balance precision/recall.

**Fix Applied:**
```python
def find_optimal_thresholds(predictions, labels, label_names):
    # Try thresholds from 0.3 to 0.8
    # Find the one that maximizes F1 for each class
    for threshold in np.arange(0.3, 0.8, 0.05):
        # Calculate F1 and keep best threshold
```

---

### 6. ✅ **Poor Learning Rate Schedule**
**Issue:** 
- No warmup period
- Fixed learning rate throughout training
- No weight decay

**Why it hurts scores:** Model trains unstably and overfits.

**Fix Applied:**
```python
# Learning rate warmup (10% of total steps)
warmup_steps = int(0.1 * total_steps)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

# Weight decay for regularization
optimizer = AdamW(
    model.parameters(),
    lr=2e-5,
    weight_decay=0.01  # L2 regularization
)
```

---

### 7. ✅ **No Gradient Clipping**
**Issue:** Gradients can explode during training.

**Why it hurts scores:** Training becomes unstable, loss spikes.

**Fix Applied:**
```python
loss.backward()
# Clip gradients to prevent explosion
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

---

### 8. ✅ **No Early Stopping**
**Issue:** Model continues training even when overfitting.

**Why it hurts scores:** Validation performance degrades while training loss decreases.

**Fix Applied:**
```python
patience = 5
patience_counter = 0

if avg_f1 > best_f1:
    best_f1 = avg_f1
    patience_counter = 0
    # Save model
else:
    patience_counter += 1
    
if patience_counter >= patience:
    print("Early stopping triggered")
    break
```

---

### 9. ✅ **Missing Reproducibility**
**Issue:** Different results each run due to random seeds.

**Why it hurts scores:** Can't reproduce best results.

**Fix Applied:**
```python
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
```

---

### 10. ✅ **Insufficient Training Monitoring**
**Issue:** No visibility into what's happening during training.

**Why it hurts scores:** Can't identify issues early.

**Fix Applied:**
- Per-batch loss tracking
- Training metrics (not just validation)
- Detailed per-class metrics
- Training history storage
- Progress bars with live metrics

---

## Expected Improvements

With these fixes, you should see:

1. **Higher F1 Scores** (0.70-0.85 range expected)
2. **Better Precision-Recall Balance**
3. **More Stable Training** (smooth loss curves)
4. **Better Generalization** (smaller gap between train/val metrics)
5. **Improved Minority Class Performance**

---

## How to Use the Improved Training

1. **Train the model:**
```bash
cd backend
.\venv\Scripts\Activate.ps1
python train_bert_improved.py
```

2. **Monitor the output:**
- Watch for decreasing loss
- Check per-class F1 scores
- Note optimal thresholds for each class

3. **Use optimal thresholds in production:**
The model saves optimal thresholds that maximize F1 for each class.

---

## Key Metrics to Watch

### During Training:
- **Loss should decrease steadily** (no wild fluctuations)
- **Training F1 should increase** (but not to 1.0 - that's overfitting)
- **Validation F1 should track training F1** (small gap is good)

### After Training:
- **F1 Score > 0.70** for all classes is good
- **Precision and Recall should be balanced** (both > 0.65)
- **Accuracy > 0.80** per class is excellent

### Red Flags:
- ❌ Training F1 = 0.95, Validation F1 = 0.60 → Overfitting
- ❌ Loss oscillating wildly → Learning rate too high
- ❌ One class with F1 = 0.0 → Class imbalance not handled
- ❌ Precision = 0.9, Recall = 0.3 → Threshold too high

---

## Additional Tips

1. **More Data**: If possible, collect more training samples (1000+ ideal)
2. **Class Balance**: Try to balance classes better in training data
3. **Hyperparameter Tuning**: Experiment with:
   - Learning rate (1e-5 to 5e-5)
   - Batch size (8, 16, 32)
   - Dropout (0.1 to 0.5)
   - Number of epochs (10-20)

4. **Model Architecture**: Try different classifier heads:
   - Different hidden layer sizes
   - More/fewer layers
   - Different activation functions

---

## Troubleshooting

### If F1 scores are still low (<0.60):

1. **Check your data quality:**
   - Are labels correct?
   - Is text clean and meaningful?
   - Are there too many similar texts?

2. **Verify training is working:**
   - Loss should decrease
   - Training accuracy should increase
   - No NaN or Inf values

3. **Try adjusting hyperparameters:**
   - Lower learning rate (1e-5)
   - Increase epochs (20)
   - Increase batch size (32)

4. **Check for data leakage:**
   - Train and validation sets should be separate
   - No duplicate texts between sets

---

## Comparison: Old vs New

| Aspect | Old Training | New Training |
|--------|-------------|--------------|
| Class Weights | ❌ None | ✅ Inverse frequency |
| Data Augmentation | ❌ None | ✅ Word dropout |
| Learning Rate Schedule | ❌ Fixed | ✅ Warmup + decay |
| Model Architecture | ❌ Simple (768→6) | ✅ Deep (768→384→192→6) |
| Metrics | ❌ Basic | ✅ Comprehensive |
| Threshold | ❌ Fixed 0.5 | ✅ Optimized per class |
| Early Stopping | ❌ None | ✅ Patience = 5 |
| Gradient Clipping | ❌ None | ✅ Max norm = 1.0 |
| Weight Decay | ❌ None | ✅ 0.01 |
| Reproducibility | ❌ Random | ✅ Seeded |

---

## Next Steps

1. ✅ Run `train_bert_improved.py`
2. ⏳ Wait for training to complete (~15-30 mins)
3. ⏳ Check final metrics
4. ⏳ Update `model.py` to use optimal thresholds
5. ⏳ Test with sample predictions

Good luck with your training! 🚀
