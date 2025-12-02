import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from tqdm import tqdm
from model import EmotionClassifier, TextClassificationService
import os

class TextEmotionDataset(Dataset):
    """Custom Dataset for text classification"""
    
    def __init__(self, text_data, label_data, tokenizer_instance, max_seq_len=128):
        self.text_samples = text_data
        self.labels_array = label_data
        self.tokenizer = tokenizer_instance
        self.maximum_length = max_seq_len
        
    def __len__(self):
        return len(self.text_samples)
    
    def __getitem__(self, index):
        text_content = str(self.text_samples[index])
        label_values = self.labels_array[index]
        
        encoded_text = self.tokenizer.encode_plus(
            text_content,
            add_special_tokens=True,
            max_length=self.maximum_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoded_text['input_ids'].flatten(),
            'attention_mask': encoded_text['attention_mask'].flatten(),
            'labels': torch.FloatTensor(label_values)
        }


class ModelTrainer:
    """Handles model training and evaluation"""
    
    def __init__(self, model_instance, device='cpu'):
        self.model = model_instance
        self.device_type = device
        self.model.to(self.device_type)
        
    def train_single_epoch(self, data_loader, optimizer_instance, loss_function):
        """Train for one epoch"""
        self.model.train()
        cumulative_loss = 0
        progress_bar = tqdm(data_loader, desc='Training')
        
        for batch_idx, batch_data in enumerate(progress_bar):
            input_token_ids = batch_data['input_ids'].to(self.device_type)
            attention_masks = batch_data['attention_mask'].to(self.device_type)
            target_labels = batch_data['labels'].to(self.device_type)
            
            optimizer_instance.zero_grad()
            
            predicted_logits = self.model(input_token_ids, attention_masks)
            batch_loss = loss_function(predicted_logits, target_labels)
            
            batch_loss.backward()
            optimizer_instance.step()
            
            cumulative_loss += batch_loss.item()
            progress_bar.set_postfix({'loss': cumulative_loss / (batch_idx + 1)})
            
        return cumulative_loss / len(data_loader)
    
    def evaluate_model(self, data_loader, loss_function):
        """Evaluate model on validation set"""
        self.model.eval()
        cumulative_loss = 0
        
        with torch.no_grad():
            for batch_data in tqdm(data_loader, desc='Evaluating'):
                input_token_ids = batch_data['input_ids'].to(self.device_type)
                attention_masks = batch_data['attention_mask'].to(self.device_type)
                target_labels = batch_data['labels'].to(self.device_type)
                
                predicted_logits = self.model(input_token_ids, attention_masks)
                batch_loss = loss_function(predicted_logits, target_labels)
                
                cumulative_loss += batch_loss.item()
        
        return cumulative_loss / len(data_loader)
    
    def save_checkpoint(self, save_path, epoch_num, optimizer_state, train_loss, val_loss):
        """Save model checkpoint"""
        checkpoint_dict = {
            'epoch': epoch_num,
            'model_state': self.model.state_dict(),
            'optimizer_state': optimizer_state,
            'train_loss': train_loss,
            'val_loss': val_loss
        }
        torch.save(checkpoint_dict, save_path)
        print(f'Checkpoint saved to {save_path}')


def prepare_training_data(csv_file_path):
    """Load and prepare data from CSV"""
    dataframe = pd.read_csv(csv_file_path)
    
    # Assuming columns: 'comment_text', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'
    text_column = dataframe['comment_text'].values
    label_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    label_matrix = dataframe[label_columns].values
    
    return text_column, label_matrix


def train_classification_model(
    data_path,
    epochs=5,
    batch_size=16,
    learning_rate=2e-5,
    validation_split=0.2,
    checkpoint_dir='checkpoints'
):
    """Main training function"""
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Load data
    print('Loading data...')
    texts, labels = prepare_training_data(data_path)
    
    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=validation_split, random_state=42
    )
    
    # Initialize service and get tokenizer
    service = TextClassificationService(device=device)
    
    # Create datasets
    train_dataset = TextEmotionDataset(train_texts, train_labels, service.bert_tokenizer)
    val_dataset = TextEmotionDataset(val_texts, val_labels, service.bert_tokenizer)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model and trainer
    model = EmotionClassifier(num_categories=len(service.emotion_categories))
    trainer = ModelTrainer(model, device)
    
    # Setup optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = nn.BCEWithLogitsLoss()
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        print(f'\nEpoch {epoch + 1}/{epochs}')
        print('-' * 50)
        
        train_loss = trainer.train_single_epoch(train_loader, optimizer, loss_fn)
        val_loss = trainer.evaluate_model(val_loader, loss_fn)
        
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Val Loss: {val_loss:.4f}')
        
        # Save checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch + 1}.pt')
        trainer.save_checkpoint(
            checkpoint_path,
            epoch + 1,
            optimizer.state_dict(),
            train_loss,
            val_loss
        )
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(checkpoint_dir, 'best_model.pt')
            trainer.save_checkpoint(
                best_model_path,
                epoch + 1,
                optimizer.state_dict(),
                train_loss,
                val_loss
            )
            print(f'Best model saved with val_loss: {val_loss:.4f}')


if __name__ == '__main__':
    # Example usage
    train_classification_model(
        data_path='data/train.csv',
        epochs=5,
        batch_size=16,
        learning_rate=2e-5
    )