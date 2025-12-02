"""
RLHF (Reinforcement Learning from Human Feedback) Tuning
========================================================
Refines the fine-tuned LLM using human feedback to reduce false positives.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler
from peft import PeftModel
import json
import os

def load_feedback_data():
    """
    Load human feedback data
    Format: {"text": "...", "correct_labels": "...", "incorrect_labels": "..."}
    """
    feedback_file = 'human_feedback_data.json'
    
    if os.path.exists(feedback_file):
        with open(feedback_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    # Create example feedback data
    feedback_data = [
        {
            "text": "I will marry her",
            "correct_labels": "emotion: positive, sentiment: positive, stress: low, risk: none",
            "incorrect_labels": "emotion: self_harm_high, sentiment: negative, stress: high, risk: high",
            "feedback": "This is a positive relationship statement, not self-harm"
        },
        {
            "text": "I am unstoppable",
            "correct_labels": "emotion: positive, sentiment: positive, stress: low, risk: none",
            "incorrect_labels": "emotion: self_harm_high, sentiment: negative, stress: high, risk: high",
            "feedback": "This is a confident/empowered statement, not self-harm"
        },
        {
            "text": "I'm sick of people wasting my time",
            "correct_labels": "emotion: stress, sentiment: negative, stress: high, risk: low",
            "incorrect_labels": "emotion: self_harm_high, sentiment: negative, stress: high, risk: high",
            "feedback": "This is frustration/annoyance, not self-harm"
        },
    ]
    
    with open(feedback_file, 'w', encoding='utf-8') as f:
        json.dump(feedback_data, f, indent=2)
    
    return feedback_data

def reward_function(predicted_labels, correct_labels):
    """
    Calculate reward based on label accuracy
    Higher reward for correct predictions, penalty for false positives
    """
    reward = 0.0
    
    # Parse labels
    predicted = {k.strip(): v.strip() for k, v in [item.split(':') for item in predicted_labels.split(',')]}
    correct = {k.strip(): v.strip() for k, v in [item.split(':') for item in correct_labels.split(',')]}
    
    # Reward for correct predictions
    for key in ['emotion', 'sentiment', 'stress', 'risk']:
        if key in predicted and key in correct:
            if predicted[key] == correct[key]:
                reward += 1.0
            else:
                reward -= 0.5  # Penalty for incorrect
    
    # Extra penalty for false positives on safe statements
    if correct.get('risk', '') == 'none' and predicted.get('risk', '') != 'none':
        reward -= 2.0  # Strong penalty for false positive risk
    
    if correct.get('emotion', '') == 'positive' and predicted.get('emotion', '') in ['self_harm_high', 'self_harm_low']:
        reward -= 3.0  # Very strong penalty for misclassifying positive as self-harm
    
    return reward

def main():
    print("="*80)
    print("RLHF TUNING FOR MENTAL HEALTH CLASSIFICATION")
    print("="*80)
    
    # Check if fine-tuned model exists
    model_path = "./llm_checkpoints/final_model"
    if not os.path.exists(model_path):
        print(f"\n❌ Fine-tuned model not found at {model_path}")
        print("Please run: python llm_finetune_classifier.py")
        return
    
    # Load model
    print("\nLoading fine-tuned model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✓ Device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    base_model = AutoModelForCausalLM.from_pretrained(model_path)
    
    # Load with value head for PPO
    model = AutoModelForCausalLMWithValueHead.from_pretrained(model_path)
    model.to(device)
    
    # Load feedback data
    print("\nLoading human feedback data...")
    feedback_data = load_feedback_data()
    print(f"✓ Loaded {len(feedback_data)} feedback examples")
    
    # PPO Configuration
    ppo_config = PPOConfig(
        model_name=model_path,
        learning_rate=1e-5,
        batch_size=4,
        mini_batch_size=2,
        gradient_accumulation_steps=4,
        optimize_cuda_cache=True,
    )
    
    # PPO Trainer
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        ref_model=base_model,
        tokenizer=tokenizer,
    )
    
    # Training loop
    print("\n" + "="*80)
    print("STARTING RLHF TRAINING")
    print("="*80)
    
    generation_kwargs = {
        "max_new_tokens": 50,
        "temperature": 0.7,
        "do_sample": True,
    }
    
    for epoch in range(3):
        print(f"\nEpoch {epoch + 1}/3")
        
        for feedback in feedback_data:
            text = feedback['text']
            correct_labels = feedback['correct_labels']
            
            # Generate prediction
            query_tensor = tokenizer.encode(text, return_tensors="pt").to(device)
            response_tensor = ppo_trainer.generate(
                query_tensor,
                return_prompt_length_greater_than=0,
                **generation_kwargs
            )
            
            # Decode response
            response_text = tokenizer.decode(response_tensor[0])
            predicted_labels = response_text.split("Label:")[-1].strip()
            
            # Calculate reward
            reward = reward_function(predicted_labels, correct_labels)
            
            # PPO step
            stats = ppo_trainer.step([query_tensor[0]], [response_tensor[0]], [reward])
            ppo_trainer.log_stats(stats, {}, [reward])
            
            print(f"  Text: {text[:50]}...")
            print(f"  Reward: {reward:.2f}")
    
    # Save RLHF-tuned model
    print("\nSaving RLHF-tuned model...")
    model.save_pretrained("./llm_checkpoints/rlhf_model")
    tokenizer.save_pretrained("./llm_checkpoints/rlhf_model")
    print("✓ Model saved to ./llm_checkpoints/rlhf_model")
    
    print("\n" + "="*80)
    print("RLHF TUNING COMPLETE!")
    print("="*80)
    print("\nNext step: Use hybrid classifier")
    print("  python hybrid_classifier.py")


if __name__ == '__main__':
    main()

