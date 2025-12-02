"""
LLM-Based Data Generation and Augmentation
==========================================
Uses GPT/LLM models to generate diverse, high-quality training data
"""

import os
import json
import random
from typing import List, Dict, Optional
from datetime import datetime

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("[LLM Data Generator] OpenAI not available. Install with: pip install openai")

try:
    from transformers import pipeline
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


class LLMDataGenerator:
    """Generate training data using LLMs"""
    
    def __init__(self, use_openai: bool = True, use_hf: bool = True):
        self.use_openai = use_openai and OPENAI_AVAILABLE
        self.use_hf = use_hf and HF_AVAILABLE
        self.openai_client = None
        self.hf_pipeline = None
        
        if self.use_openai:
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key:
                self.openai_client = openai.OpenAI(api_key=api_key)
                print("[LLM Data Generator] OpenAI GPT initialized")
            else:
                print("[LLM Data Generator] OpenAI API key not found. Using Hugging Face only.")
                self.use_openai = False
        
        if self.use_hf:
            try:
                self.hf_pipeline = pipeline(
                    "text-generation",
                    model="gpt2",  # Lightweight model for generation
                    device=0 if os.getenv('CUDA_VISIBLE_DEVICES') else -1
                )
                print("[LLM Data Generator] Hugging Face pipeline initialized")
            except Exception as e:
                print(f"[LLM Data Generator] HF pipeline failed: {e}")
                self.use_hf = False
    
    def generate_with_gpt(self, category: str, emotion: str, count: int = 10) -> List[Dict]:
        """Generate training examples using GPT"""
        if not self.openai_client:
            return []
        
        examples = []
        prompt = f"""Generate {count} diverse, realistic text examples for mental health classification.

Category: {category}
Emotion: {emotion}

Requirements:
- Each example should be 5-30 words
- Natural, conversational language
- Diverse phrasing and contexts
- Realistic scenarios
- No repetition

Return ONLY a JSON array of strings, one example per string.
Example format: ["example 1", "example 2", ...]

Examples for {emotion}:"""
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",  # or "gpt-4" for better quality
                messages=[
                    {"role": "system", "content": "You are an expert at generating realistic text examples for mental health classification."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.9,  # Higher temperature for diversity
                max_tokens=1000
            )
            
            content = response.choices[0].message.content.strip()
            # Try to extract JSON array
            if content.startswith('['):
                texts = json.loads(content)
                for text in texts:
                    examples.append({
                        'text': text,
                        'category': category,
                        'emotion': emotion,
                        'generated_by': 'gpt',
                        'timestamp': datetime.now().isoformat()
                    })
        except Exception as e:
            print(f"[LLM Data Generator] GPT generation error: {e}")
        
        return examples
    
    def generate_with_hf(self, category: str, emotion: str, seed_text: str, count: int = 5) -> List[Dict]:
        """Generate training examples using Hugging Face models"""
        if not self.hf_pipeline:
            return []
        
        examples = []
        prompt = f"{seed_text} I"
        
        try:
            for _ in range(count):
                generated = self.hf_pipeline(
                    prompt,
                    max_length=len(prompt.split()) + 15,
                    num_return_sequences=1,
                    temperature=0.8,
                    do_sample=True,
                    pad_token_id=50256
                )
                
                text = generated[0]['generated_text'].strip()
                # Clean up the text
                if len(text) > len(prompt):
                    text = text[len(prompt):].strip()
                    if text and 5 <= len(text.split()) <= 30:
                        examples.append({
                            'text': text,
                            'category': category,
                            'emotion': emotion,
                            'generated_by': 'hf',
                            'timestamp': datetime.now().isoformat()
                        })
        except Exception as e:
            print(f"[LLM Data Generator] HF generation error: {e}")
        
        return examples
    
    def augment_existing(self, text: str, label: str, method: str = 'paraphrase') -> List[str]:
        """Augment existing training examples using LLMs"""
        augmented = []
        
        if method == 'paraphrase' and self.openai_client:
            try:
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a text paraphrasing expert. Generate 3 different paraphrases that maintain the same meaning and emotional tone."},
                        {"role": "user", "content": f"Paraphrase this text 3 different ways, keeping the same meaning:\n\n{text}\n\nReturn as JSON array."}
                    ],
                    temperature=0.7,
                    max_tokens=200
                )
                
                content = response.choices[0].message.content.strip()
                if content.startswith('['):
                    import json
                    paraphrases = json.loads(content)
                    augmented.extend(paraphrases)
            except Exception as e:
                print(f"[LLM Data Generator] Paraphrase error: {e}")
        
        return augmented
    
    def generate_comprehensive_dataset(self, target_size: int = 2000) -> List[Dict]:
        """Generate comprehensive training dataset using LLMs"""
        print(f"[LLM Data Generator] Generating {target_size} examples...")
        
        all_examples = []
        
        # Define categories and emotions
        categories = {
            'neutral': ['neutral', 'conversational', 'informational'],
            'stress': ['stress', 'frustration', 'complaint'],
            'emotional_distress': ['emotional_distress', 'sadness', 'anxiety'],
            'self_harm_low': ['self_harm_low', 'thoughts', 'ideation'],
            'self_harm_high': ['self_harm_high', 'intent', 'plan'],
            'unsafe_environment': ['unsafe_environment', 'threat', 'danger']
        }
        
        examples_per_category = target_size // len(categories)
        
        for category, emotions in categories.items():
            print(f"  Generating {examples_per_category} examples for {category}...")
            
            # Use GPT for high-quality generation
            if self.use_openai:
                gpt_examples = self.generate_with_gpt(
                    category, 
                    emotions[0], 
                    count=min(examples_per_category, 50)  # GPT API limits
                )
                all_examples.extend(gpt_examples)
            
            # Use HF for additional examples
            if self.use_hf and len(all_examples) < target_size:
                seed_texts = {
                    'neutral': "I went to the store",
                    'stress': "I'm frustrated with",
                    'emotional_distress': "I feel really sad",
                    'self_harm_low': "Sometimes I think about",
                    'self_harm_high': "I want to",
                    'unsafe_environment': "I'm going to hurt"
                }
                
                hf_examples = self.generate_with_hf(
                    category,
                    emotions[0],
                    seed_texts.get(category, "I feel"),
                    count=min(10, examples_per_category // 2)
                )
                all_examples.extend(hf_examples)
        
        print(f"[LLM Data Generator] Generated {len(all_examples)} examples")
        return all_examples


def convert_to_training_format(llm_examples: List[Dict]) -> List[Dict]:
    """Convert LLM-generated examples to training format"""
    label_names = ['neutral', 'stress', 'unsafe_environment', 
                   'emotional_distress', 'self_harm_low', 'self_harm_high']
    
    training_data = []
    
    for example in llm_examples:
        emotion = example.get('emotion', 'neutral')
        labels = {label: 0 for label in label_names}
        
        # Map emotion to label
        if emotion in label_names:
            labels[emotion] = 1
        elif emotion == 'conversational' or emotion == 'informational':
            labels['neutral'] = 1
        elif emotion == 'frustration' or emotion == 'complaint':
            labels['stress'] = 1
        elif emotion == 'sadness' or emotion == 'anxiety':
            labels['emotional_distress'] = 1
        elif emotion == 'thoughts' or emotion == 'ideation':
            labels['self_harm_low'] = 1
        elif emotion == 'intent' or emotion == 'plan':
            labels['self_harm_high'] = 1
        elif emotion == 'threat' or emotion == 'danger':
            labels['unsafe_environment'] = 1
        
        training_data.append({
            'text': example['text'],
            'labels': labels
        })
    
    return training_data


def main():
    """Generate LLM-based training data"""
    print("="*80)
    print("LLM-BASED TRAINING DATA GENERATION")
    print("="*80)
    
    generator = LLMDataGenerator(use_openai=True, use_hf=True)
    
    # Generate comprehensive dataset
    llm_examples = generator.generate_comprehensive_dataset(target_size=1500)
    
    # Convert to training format
    training_data = convert_to_training_format(llm_examples)
    
    # Save to file
    output_path = 'llm_generated_training_data.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Generated {len(training_data)} training examples")
    print(f"✓ Saved to {output_path}")
    print("\nNext steps:")
    print("1. Review the generated data")
    print("2. Merge with existing training data if needed")
    print("3. Use in training pipeline")


if __name__ == '__main__':
    main()

