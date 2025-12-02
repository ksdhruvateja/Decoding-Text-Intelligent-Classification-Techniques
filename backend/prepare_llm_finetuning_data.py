"""
Prepare Data for LLM Fine-Tuning
=================================
Converts training data into JSONL format for LLM fine-tuning
with multi-label classification format.
"""

import json
import random

def format_for_llm_finetuning(input_file, output_file):
    """
    Convert training data to LLM fine-tuning format (JSONL)
    
    Format:
    {
        "prompt": "Text: {text}\nLabel:",
        "completion": "emotion: {emotion}, sentiment: {sentiment}, stress: {stress}, risk: {risk}"
    }
    """
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    output_data = []
    
    label_names = ['neutral', 'stress', 'unsafe_environment', 
                   'emotional_distress', 'self_harm_low', 'self_harm_high']
    
    for item in data:
        text = item['text']
        labels = item['labels']
        
        # Determine emotion
        if labels.get('self_harm_high', 0) == 1:
            emotion = 'self_harm_high'
        elif labels.get('self_harm_low', 0) == 1:
            emotion = 'self_harm_low'
        elif labels.get('unsafe_environment', 0) == 1:
            emotion = 'unsafe_environment'
        elif labels.get('emotional_distress', 0) == 1:
            emotion = 'emotional_distress'
        elif labels.get('stress', 0) == 1:
            emotion = 'stress'
        elif labels.get('neutral', 0) == 1:
            # Check if it's positive or neutral
            positive_keywords = ['love', 'loved', 'amazing', 'wonderful', 'great', 'excellent',
                               'grateful', 'happy', 'excited', 'proud', 'confident', 'empowered',
                               'motivated', 'unstoppable', 'marry', 'marriage', 'cherish', 'adore']
            text_lower = text.lower()
            if any(kw in text_lower for kw in positive_keywords):
                emotion = 'positive'
            else:
                emotion = 'neutral'
        else:
            emotion = 'neutral'
        
        # Determine sentiment
        if emotion in ['positive', 'neutral']:
            sentiment = 'positive' if emotion == 'positive' else 'neutral'
        else:
            sentiment = 'negative'
        
        # Determine stress level
        if labels.get('stress', 0) == 1:
            stress = 'high'
        elif labels.get('emotional_distress', 0) == 1:
            stress = 'medium'
        elif emotion in ['positive', 'neutral']:
            stress = 'low'
        else:
            stress = 'medium'
        
        # Determine risk level
        if labels.get('self_harm_high', 0) == 1:
            risk = 'high'
        elif labels.get('self_harm_low', 0) == 1:
            risk = 'medium'
        elif labels.get('unsafe_environment', 0) == 1:
            risk = 'medium'
        elif emotion in ['positive', 'neutral']:
            risk = 'none'
        else:
            risk = 'low'
        
        # Create prompt and completion
        prompt = f"Text: {text}\nLabel:"
        completion = f"emotion: {emotion}, sentiment: {sentiment}, stress: {stress}, risk: {risk}"
        
        output_data.append({
            "prompt": prompt,
            "completion": completion
        })
    
    # Shuffle
    random.shuffle(output_data)
    
    # Split into train/val (80/20)
    split_idx = int(len(output_data) * 0.8)
    train_data = output_data[:split_idx]
    val_data = output_data[split_idx:]
    
    # Save as JSONL
    with open(output_file.replace('.jsonl', '_train.jsonl'), 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    with open(output_file.replace('.jsonl', '_val.jsonl'), 'w', encoding='utf-8') as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"✓ Converted {len(train_data)} training examples")
    print(f"✓ Converted {len(val_data)} validation examples")
    print(f"✓ Saved to {output_file.replace('.jsonl', '_train.jsonl')} and {output_file.replace('.jsonl', '_val.jsonl')}")


def main():
    print("="*80)
    print("PREPARING DATA FOR LLM FINE-TUNING")
    print("="*80)
    
    # Check if training data exists
    import os
    if not os.path.exists('train_data.json'):
        print("\n❌ train_data.json not found!")
        print("Please run: python generate_comprehensive_fixed_training_data.py")
        return
    
    # Convert to LLM format
    print("\nConverting training data to LLM fine-tuning format...")
    format_for_llm_finetuning('train_data.json', 'llm_finetuning_data.jsonl')
    
    print("\n✓ Data preparation complete!")
    print("\nNext steps:")
    print("1. Run: python llm_finetune_classifier.py")
    print("2. Then: python rlhf_tune_classifier.py")
    print("3. Finally: python hybrid_classifier.py")


if __name__ == '__main__':
    main()

