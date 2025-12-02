"""
Advanced BERT Classifier with State-of-the-Art Techniques
==========================================================
Implements:
- Multiple BERT variants (BERT, RoBERTa, DeBERTa)
- Advanced pooling strategies (attention, mean-max)
- Focal loss for imbalanced classes
- Label smoothing
- Advanced regularization
- Ensemble support
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    BertModel, BertTokenizer,
    RobertaModel, RobertaTokenizer,
    DebertaModel, DebertaTokenizer,
    AutoModel, AutoTokenizer
)
import numpy as np
from typing import Optional, Dict, List


class AttentionPooling(nn.Module):
    """Attention-based pooling instead of simple CLS token"""
    def __init__(self, hidden_size=768):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, hidden_states, attention_mask):
        # hidden_states: [batch_size, seq_len, hidden_size]
        # attention_mask: [batch_size, seq_len]
        
        # Compute attention weights
        attn_weights = self.attention(hidden_states)  # [batch_size, seq_len, 1]
        attn_weights = attn_weights.squeeze(-1)  # [batch_size, seq_len]
        
        # Mask out padding tokens
        attn_weights = attn_weights.masked_fill(attention_mask == 0, float('-inf'))
        attn_weights = F.softmax(attn_weights, dim=1)  # [batch_size, seq_len]
        
        # Weighted sum
        pooled = torch.sum(attn_weights.unsqueeze(-1) * hidden_states, dim=1)
        return pooled


class MeanMaxPooling(nn.Module):
    """Mean-Max pooling strategy"""
    def __init__(self):
        super(MeanMaxPooling, self).__init__()
    
    def forward(self, hidden_states, attention_mask):
        # Mean pooling
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_hidden = torch.sum(hidden_states * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        mean_pooled = sum_hidden / sum_mask
        
        # Max pooling
        hidden_states[attention_mask == 0] = float('-inf')
        max_pooled = torch.max(hidden_states, dim=1)[0]
        
        # Concatenate
        return torch.cat([mean_pooled, max_pooled], dim=1)


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        # Compute BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Compute p_t
        pt = torch.exp(-bce_loss)
        
        # Compute focal loss
        focal_loss = ((1 - pt) ** self.gamma) * bce_loss
        
        # Apply alpha weighting if provided
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = self.alpha
            else:
                alpha_t = self.alpha.unsqueeze(0).expand_as(targets)
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingBCE(nn.Module):
    """Binary Cross-Entropy with Label Smoothing"""
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingBCE, self).__init__()
        self.smoothing = smoothing
    
    def forward(self, inputs, targets):
        # Smooth labels
        targets = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        return F.binary_cross_entropy_with_logits(inputs, targets)


class AdvancedBERTClassifier(nn.Module):
    """
    Advanced BERT classifier with multiple improvements:
    - Multiple model variants
    - Advanced pooling (attention, mean-max)
    - Residual connections
    - Layer normalization
    - Dropout scheduling
    """
    
    def __init__(
        self,
        model_name='bert-base-uncased',
        n_classes=6,
        dropout=0.3,
        pooling_strategy='attention',  # 'cls', 'attention', 'mean_max', 'mean'
        use_residual=True,
        hidden_size=768
    ):
        super(AdvancedBERTClassifier, self).__init__()
        
        self.model_name = model_name
        self.pooling_strategy = pooling_strategy
        self.use_residual = use_residual
        
        # Load base model
        if 'roberta' in model_name.lower():
            self.backbone = RobertaModel.from_pretrained(model_name)
            hidden_size = self.backbone.config.hidden_size
        elif 'deberta' in model_name.lower():
            self.backbone = DebertaModel.from_pretrained(model_name)
            hidden_size = self.backbone.config.hidden_size
        else:
            self.backbone = BertModel.from_pretrained(model_name)
            hidden_size = self.backbone.config.hidden_size
        
        self.hidden_size = hidden_size
        
        # Pooling layer
        if pooling_strategy == 'attention':
            self.pooling = AttentionPooling(hidden_size)
            pool_output_size = hidden_size
        elif pooling_strategy == 'mean_max':
            self.pooling = MeanMaxPooling()
            pool_output_size = hidden_size * 2
        elif pooling_strategy == 'mean':
            self.pooling = None
            pool_output_size = hidden_size
        else:  # 'cls'
            self.pooling = None
            pool_output_size = hidden_size
        
        # Classification head with residual connections
        self.dropout1 = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(pool_output_size)
        
        self.fc1 = nn.Linear(pool_output_size, 512)
        self.dropout2 = nn.Dropout(dropout * 0.7)
        self.layer_norm2 = nn.LayerNorm(512)
        
        self.fc2 = nn.Linear(512, 256)
        self.dropout3 = nn.Dropout(dropout * 0.5)
        self.layer_norm3 = nn.LayerNorm(256)
        
        self.fc3 = nn.Linear(256, n_classes)
        
        # Residual connection (if dimensions match)
        if use_residual and pool_output_size == 256:
            self.residual_proj = nn.Linear(pool_output_size, 256)
        else:
            self.residual_proj = None
        
        self.activation = nn.GELU()  # GELU instead of ReLU
    
    def forward(self, input_ids, attention_mask, return_hidden=False):
        # Get BERT outputs
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        
        # Get hidden states
        hidden_states = outputs.last_hidden_state  # [batch, seq_len, hidden_size]
        
        # Pooling
        if self.pooling_strategy == 'cls':
            pooled = hidden_states[:, 0, :]  # CLS token
        elif self.pooling_strategy == 'mean':
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_hidden = torch.sum(hidden_states * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            pooled = sum_hidden / sum_mask
        else:
            pooled = self.pooling(hidden_states, attention_mask)
        
        # Classification head
        x = self.layer_norm1(pooled)
        x = self.dropout1(x)
        
        # First layer
        x1 = self.fc1(x)
        x1 = self.layer_norm2(x1)
        x1 = self.activation(x1)
        x1 = self.dropout2(x1)
        
        # Second layer
        x2 = self.fc2(x1)
        x2 = self.layer_norm3(x2)
        x2 = self.activation(x2)
        x2 = self.dropout3(x2)
        
        # Residual connection (if applicable)
        if self.use_residual and self.residual_proj is not None:
            x2 = x2 + self.residual_proj(x)
        
        # Final layer
        logits = self.fc3(x2)
        
        if return_hidden:
            return logits, x2
        return logits


class EnsembleClassifier(nn.Module):
    """Ensemble of multiple models"""
    def __init__(self, models: List[nn.Module], weights: Optional[List[float]] = None):
        super(EnsembleClassifier, self).__init__()
        self.models = nn.ModuleList(models)
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        self.weights = weights
    
    def forward(self, input_ids, attention_mask):
        all_logits = []
        for model in self.models:
            logits = model(input_ids, attention_mask)
            all_logits.append(logits)
        
        # Weighted average
        weighted_logits = sum(w * logits for w, logits in zip(self.weights, all_logits))
        return weighted_logits


# Temperature Scaling with better initialization
class AdvancedTemperatureScaling(nn.Module):
    """Improved temperature scaling with per-class temperatures"""
    def __init__(self, num_classes=6, init_temp=1.5):
        super(AdvancedTemperatureScaling, self).__init__()
        self.temperature = nn.Parameter(torch.ones(num_classes) * init_temp)
    
    def forward(self, logits):
        # Per-class temperature scaling
        temperature = self.temperature.unsqueeze(0).expand(logits.size(0), -1)
        return logits / temperature


if __name__ == '__main__':
    # Test the advanced classifier
    print("Testing Advanced BERT Classifier...")
    
    model = AdvancedBERTClassifier(
        model_name='bert-base-uncased',
        n_classes=6,
        pooling_strategy='attention',
        use_residual=True
    )
    
    # Dummy input
    input_ids = torch.randint(0, 1000, (2, 128))
    attention_mask = torch.ones(2, 128)
    
    output = model(input_ids, attention_mask)
    print(f"Output shape: {output.shape}")
    print("✓ Advanced classifier working!")

