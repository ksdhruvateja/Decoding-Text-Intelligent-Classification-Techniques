"""
Reinforcement Learning from Human Feedback (RLHF) Trainer
==========================================================
Fine-tunes model using human feedback signals for improved accuracy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from typing import List, Dict, Tuple
import numpy as np
from tqdm import tqdm
import json
import os
from datetime import datetime


class FeedbackDataset(Dataset):
    """Dataset for RLHF training with feedback signals"""
    
    def __init__(self, feedback_data: List[Dict], tokenizer, max_length=128):
        self.data = feedback_data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_names = ['neutral', 'stress', 'unsafe_environment', 
                           'emotional_distress', 'self_harm_low', 'self_harm_high']
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        true_labels = item['true_labels']
        predicted_labels = item.get('predicted_labels', true_labels)
        feedback = item.get('feedback', 1.0)  # 1.0 = correct, 0.0 = incorrect
        
        # Convert labels to tensor
        true_label_vec = torch.FloatTensor([true_labels.get(label, 0) for label in self.label_names])
        pred_label_vec = torch.FloatTensor([predicted_labels.get(label, 0) for label in self.label_names])
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'true_labels': true_label_vec,
            'predicted_labels': pred_label_vec,
            'feedback': torch.FloatTensor([feedback])
        }


class RLHFTrainer:
    """Reinforcement Learning from Human Feedback trainer"""
    
    def __init__(self, model_path: str = None, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        # Load or create model
        if model_path and os.path.exists(model_path):
            print(f"[RLHF] Loading model from {model_path}")
            # Load your custom model architecture here
            # For now, using base BERT
            self.model = BertForSequenceClassification.from_pretrained(
                'bert-base-uncased',
                num_labels=6
            )
            # Load custom weights if available
            try:
                state_dict = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state_dict, strict=False)
            except:
                print("[RLHF] Could not load custom weights, using base model")
        else:
            self.model = BertForSequenceClassification.from_pretrained(
                'bert-base-uncased',
                num_labels=6
            )
        
        self.model.to(self.device)
        self.model.train()
    
    def compute_reward(self, true_labels: torch.Tensor, pred_labels: torch.Tensor, 
                      feedback: torch.Tensor) -> torch.Tensor:
        """Compute reward signal based on feedback and accuracy"""
        # Binary accuracy reward
        accuracy = (true_labels == pred_labels).float().mean()
        
        # Combine with explicit feedback
        reward = feedback * 0.7 + accuracy * 0.3
        
        return reward
    
    def compute_rlhf_loss(self, logits: torch.Tensor, true_labels: torch.Tensor,
                         predicted_labels: torch.Tensor, feedback: torch.Tensor) -> torch.Tensor:
        """Compute RLHF loss combining classification loss and reward signal"""
        # Standard classification loss (BCE)
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, true_labels, reduction='none'
        ).mean(dim=1)
        
        # Reward signal (higher reward = lower loss)
        rewards = self.compute_reward(true_labels, predicted_labels, feedback)
        reward_loss = -torch.log(torch.sigmoid(logits).mean(dim=1) + 1e-8) * rewards
        
        # Combine losses
        total_loss = bce_loss.mean() + 0.3 * reward_loss.mean()
        
        return total_loss
    
    def train_step(self, batch, optimizer):
        """Single training step"""
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        true_labels = batch['true_labels'].to(self.device)
        predicted_labels = batch['predicted_labels'].to(self.device)
        feedback = batch['feedback'].to(self.device)
        
        # Forward pass
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
        # Compute RLHF loss
        loss = self.compute_rlhf_loss(logits, true_labels, predicted_labels, feedback)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        optimizer.step()
        
        return loss.item()
    
    def train(self, feedback_data: List[Dict], epochs: int = 5, 
              batch_size: int = 16, learning_rate: float = 2e-5):
        """Train model with RLHF"""
        print(f"[RLHF] Training with {len(feedback_data)} feedback examples")
        
        # Create dataset and dataloader
        dataset = FeedbackDataset(feedback_data, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        
        # Training loop
        for epoch in range(epochs):
            print(f"\n[RLHF] Epoch {epoch + 1}/{epochs}")
            total_loss = 0
            
            for batch in tqdm(dataloader, desc="Training"):
                loss = self.train_step(batch, optimizer)
                total_loss += loss
            
            avg_loss = total_loss / len(dataloader)
            print(f"[RLHF] Average loss: {avg_loss:.4f}")
        
        print("[RLHF] Training complete!")
    
    def save_model(self, path: str):
        """Save trained model"""
        torch.save(self.model.state_dict(), path)
        print(f"[RLHF] Model saved to {path}")


def create_feedback_data_from_predictions(
    texts: List[str],
    true_labels: List[Dict],
    predicted_labels: List[Dict]
) -> List[Dict]:
    """Create feedback data from predictions (simulated human feedback)"""
    feedback_data = []
    
    for text, true, pred in zip(texts, true_labels, predicted_labels):
        # Determine feedback (1.0 = correct, 0.0 = incorrect)
        # Check if predictions match true labels
        matches = all(true.get(label, 0) == (1 if pred.get(label, 0) > 0.5 else 0) 
                     for label in true.keys())
        feedback = 1.0 if matches else 0.0
        
        feedback_data.append({
            'text': text,
            'true_labels': true,
            'predicted_labels': pred,
            'feedback': feedback
        })
    
    return feedback_data


def main():
    """Example RLHF training"""
    print("="*80)
    print("REINFORCEMENT LEARNING FROM HUMAN FEEDBACK (RLHF)")
    print("="*80)
    
    # Example: Load feedback data
    # In practice, this would come from human annotators or validation set
    feedback_data_path = 'feedback_data.json'
    
    if os.path.exists(feedback_data_path):
        with open(feedback_data_path, 'r') as f:
            feedback_data = json.load(f)
    else:
        print(f"[RLHF] No feedback data found at {feedback_data_path}")
        print("[RLHF] Create feedback_data.json with format:")
        print("""
        [
          {
            "text": "example text",
            "true_labels": {"neutral": 1, "stress": 0, ...},
            "predicted_labels": {"neutral": 0.9, "stress": 0.1, ...},
            "feedback": 1.0
          }
        ]
        """)
        return
    
    # Initialize trainer
    trainer = RLHFTrainer(device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Train
    trainer.train(feedback_data, epochs=5, batch_size=16)
    
    # Save model
    output_path = 'checkpoints/rlhf_finetuned_model.pt'
    os.makedirs('checkpoints', exist_ok=True)
    trainer.save_model(output_path)
    
    print(f"\n✓ RLHF training complete!")
    print(f"✓ Model saved to {output_path}")


if __name__ == '__main__':
    main()

