"""
LLM Fine-Tuning for Mental Health Classification
================================================
Fine-tunes a large language model (LLM) for multi-label classification
using HuggingFace Transformers and TRL library.
"""

import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import json
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer
import os

def load_jsonl(file_path):
    """Load JSONL file"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def format_prompt_completion(example):
    """Format prompt and completion for training"""
    prompt = example['prompt']
    completion = example['completion']
    # Combine prompt and completion
    text = f"{prompt} {completion}<|endoftext|>"
    return {"text": text}

def main():
    print("="*80)
    print("LLM FINE-TUNING FOR MENTAL HEALTH CLASSIFICATION")
    print("="*80)
    
    # Configuration
    MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"  # Or "meta-llama/Llama-3.1-8B-Instruct"
    # For smaller models, use: "microsoft/Phi-3-mini-4k-instruct" or "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    # Check if data exists
    train_file = 'llm_finetuning_data_train.jsonl'
    val_file = 'llm_finetuning_data_val.jsonl'
    
    if not os.path.exists(train_file):
        print(f"\n❌ {train_file} not found!")
        print("Please run: python prepare_llm_finetuning_data.py")
        return
    
    print(f"\n✓ Using model: {MODEL_NAME}")
    print(f"✓ Training data: {train_file}")
    print(f"✓ Validation data: {val_file}")
    
    # Load data
    print("\nLoading data...")
    train_data = load_jsonl(train_file)
    val_data = load_jsonl(val_file)
    
    print(f"✓ Loaded {len(train_data)} training examples")
    print(f"✓ Loaded {len(val_data)} validation examples")
    
    # Format data
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    
    train_dataset = train_dataset.map(format_prompt_completion)
    val_dataset = val_dataset.map(format_prompt_completion)
    
    # Load tokenizer and model
    print("\nLoading tokenizer and model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✓ Device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Use LoRA for efficient fine-tuning
    print("\nSetting up LoRA for efficient fine-tuning...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32,
        device_map="auto" if device.type == 'cuda' else None,
    )
    
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    
    model = get_peft_model(model, lora_config)
    print("✓ LoRA configured")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./llm_checkpoints",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        learning_rate=2e-4,
        fp16=device.type == 'cuda',
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none",
    )
    
    # Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        max_seq_length=512,
    )
    
    # Train
    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80)
    trainer.train()
    
    # Save model
    print("\nSaving fine-tuned model...")
    model.save_pretrained("./llm_checkpoints/final_model")
    tokenizer.save_pretrained("./llm_checkpoints/final_model")
    print("✓ Model saved to ./llm_checkpoints/final_model")
    
    print("\n" + "="*80)
    print("FINE-TUNING COMPLETE!")
    print("="*80)
    print("\nNext step: Run RLHF tuning")
    print("  python rlhf_tune_classifier.py")


if __name__ == '__main__':
    main()

