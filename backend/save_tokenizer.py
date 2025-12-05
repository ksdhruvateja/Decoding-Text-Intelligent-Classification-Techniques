"""Quick training script to generate tokenizer.json"""
import torch
import json
from lightweight_train import *

# Load data
with open('training_data.json', 'r', encoding='utf-8') as f:
    all_data = json.load(f)

# Build tokenizer
tokenizer = Tokenizer(vocab_size=10000)
tokenizer.build_vocab([item['text'] for item in all_data])

# Save tokenizer
os.makedirs('checkpoints', exist_ok=True)
tokenizer_data = {
    'word2idx': tokenizer.word2idx,
    'idx2word': tokenizer.idx2word
}

with open('checkpoints/tokenizer.json', 'w') as f:
    json.dump(tokenizer_data, f)

print("Tokenizer saved to checkpoints/tokenizer.json")
