"""
Massive Data Collection for Comprehensive Training
==================================================
Collects training data from multiple sources:
1. Web scraping (Reddit, Twitter, forums)
2. LLM generation (GPT-4, Claude, etc.)
3. Public datasets
4. Data augmentation
5. Synthetic data generation
"""

import json
import random
import re
import requests
from typing import List, Dict
import time
from datetime import datetime
import os

class MassiveDataCollector:
    """Collects training data from multiple sources"""
    
    def __init__(self):
        self.collected_data = []
        self.label_names = ['neutral', 'stress', 'unsafe_environment', 
                           'emotional_distress', 'self_harm_low', 'self_harm_high']
    
    def generate_with_llm(self, num_examples=1000):
        """Generate training data using LLM (GPT-4, Claude, etc.)"""
        print(f"\n{'='*80}")
        print("GENERATING DATA WITH LLM")
        print(f"{'='*80}")
        
        # Try OpenAI first
        try:
            import openai
            client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            
            categories = [
                ("positive confident", "Generate 200 positive, confident, empowered statements"),
                ("positive relationship", "Generate 200 positive relationship/love statements"),
                ("frustration", "Generate 200 frustration/annoyance statements (NOT self-harm)"),
                ("neutral", "Generate 200 neutral everyday statements"),
                ("stress", "Generate 200 stress statements"),
                ("emotional_distress", "Generate 200 emotional distress statements"),
                ("self_harm_low", "Generate 100 low-risk self-harm ideation statements"),
                ("self_harm_high", "Generate 100 high-risk self-harm statements"),
                ("unsafe_environment", "Generate 100 unsafe environment/threat statements"),
            ]
            
            for category, prompt_template in categories:
                print(f"\nGenerating {category} examples...")
                full_prompt = f"""{prompt_template}
                
                Generate diverse, realistic examples with variations in:
                - Sentence structure
                - Vocabulary
                - Tone
                - Context
                - Length (short to long)
                
                Return as JSON array of strings. Each string should be a complete, natural statement.
                """
                
                try:
                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": "You are a data generation expert. Generate diverse, realistic text examples."},
                            {"role": "user", "content": full_prompt}
                        ],
                        temperature=0.9,
                        max_tokens=4000
                    )
                    
                    # Parse response
                    content = response.choices[0].message.content
                    # Extract JSON if wrapped in markdown
                    if "```json" in content:
                        content = content.split("```json")[1].split("```")[0]
                    elif "```" in content:
                        content = content.split("```")[1].split("```")[0]
                    
                    examples = json.loads(content)
                    
                    # Create labeled data
                    for example in examples:
                        labels = self._get_labels_for_category(category)
                        self.collected_data.append({
                            "text": example,
                            "labels": labels,
                            "source": "llm_openai",
                            "category": category
                        })
                    
                    print(f"  ✓ Generated {len(examples)} {category} examples")
                    time.sleep(1)  # Rate limiting
                    
                except Exception as e:
                    print(f"  ⚠ Error generating {category}: {e}")
                    continue
                    
        except ImportError:
            print("⚠ OpenAI not available, skipping LLM generation")
        except Exception as e:
            print(f"⚠ LLM generation error: {e}")
    
    def _get_labels_for_category(self, category: str) -> Dict:
        """Get label vector for category"""
        labels = {label: 0 for label in self.label_names}
        
        if category == "positive confident" or category == "positive relationship":
            labels['neutral'] = 1
        elif category == "frustration":
            labels['stress'] = 1
            labels['emotional_distress'] = 1
        elif category == "neutral":
            labels['neutral'] = 1
        elif category == "stress":
            labels['stress'] = 1
        elif category == "emotional_distress":
            labels['emotional_distress'] = 1
        elif category == "self_harm_low":
            labels['self_harm_low'] = 1
        elif category == "self_harm_high":
            labels['self_harm_high'] = 1
        elif category == "unsafe_environment":
            labels['unsafe_environment'] = 1
        
        return labels
    
    def scrape_reddit_data(self, subreddits=None, limit=1000):
        """Scrape data from Reddit (requires praw library)"""
        print(f"\n{'='*80}")
        print("SCRAPING REDDIT DATA")
        print(f"{'='*80}")
        
        if subreddits is None:
            subreddits = [
                'depression', 'anxiety', 'SuicideWatch', 'mentalhealth',
                'happy', 'gratitude', 'GetMotivated', 'selfimprovement',
                'relationships', 'relationship_advice', 'offmychest',
                'confession', 'TrueOffMyChest', 'CasualConversation'
            ]
        
        try:
            import praw
            reddit = praw.Reddit(
                client_id=os.getenv('REDDIT_CLIENT_ID'),
                client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
                user_agent='MentalHealthClassifier/1.0'
            )
            
            for subreddit_name in subreddits:
                try:
                    print(f"\nScraping r/{subreddit_name}...")
                    subreddit = reddit.subreddit(subreddit_name)
                    
                    count = 0
                    for submission in subreddit.hot(limit=limit):
                        text = submission.title + " " + (submission.selftext or "")
                        if len(text) > 20 and len(text) < 500:
                            labels = self._classify_reddit_text(subreddit_name, text)
                            self.collected_data.append({
                                "text": text,
                                "labels": labels,
                                "source": f"reddit_{subreddit_name}",
                                "category": subreddit_name
                            })
                            count += 1
                            if count >= limit // len(subreddits):
                                break
                    
                    print(f"  ✓ Collected {count} examples from r/{subreddit_name}")
                    time.sleep(2)  # Rate limiting
                    
                except Exception as e:
                    print(f"  ⚠ Error scraping r/{subreddit_name}: {e}")
                    continue
                    
        except ImportError:
            print("⚠ PRAW not installed. Install with: pip install praw")
        except Exception as e:
            print(f"⚠ Reddit scraping error: {e}")
    
    def _classify_reddit_text(self, subreddit: str, text: str) -> Dict:
        """Classify Reddit text based on subreddit context"""
        labels = {label: 0 for label in self.label_names}
        text_lower = text.lower()
        
        if subreddit in ['depression', 'SuicideWatch', 'anxiety']:
            if any(word in text_lower for word in ['kill myself', 'suicide', 'end my life', 'hurt myself']):
                labels['self_harm_high'] = 1
            elif any(word in text_lower for word in ['think about', 'thoughts about', 'considering']):
                labels['self_harm_low'] = 1
            else:
                labels['emotional_distress'] = 1
        elif subreddit in ['happy', 'gratitude', 'GetMotivated', 'selfimprovement']:
            labels['neutral'] = 1
        elif subreddit in ['relationships', 'relationship_advice']:
            if any(word in text_lower for word in ['love', 'marry', 'happy', 'grateful']):
                labels['neutral'] = 1
            else:
                labels['stress'] = 1
        elif subreddit in ['offmychest', 'confession', 'TrueOffMyChest']:
            labels['stress'] = 1
            labels['emotional_distress'] = 1
        else:
            labels['neutral'] = 1
        
        return labels
    
    def load_public_datasets(self):
        """Load data from public datasets"""
        print(f"\n{'='*80}")
        print("LOADING PUBLIC DATASETS")
        print(f"{'='*80}")
        
        # Common Crawl, Wikipedia, etc. would go here
        # For now, we'll use generated comprehensive data
        
        datasets = [
            'train_data.json',  # Our existing comprehensive data
            'generate_comprehensive_fixed_training_data.py'  # Generate more
        ]
        
        for dataset in datasets:
            if os.path.exists(dataset):
                try:
                    with open(dataset, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        for item in data:
                            item['source'] = 'public_dataset'
                            self.collected_data.append(item)
                    print(f"  ✓ Loaded {len(data)} examples from {dataset}")
                except Exception as e:
                    print(f"  ⚠ Error loading {dataset}: {e}")
    
    def augment_data(self, data: List[Dict], augmentation_factor=3):
        """Augment data with variations"""
        print(f"\n{'='*80}")
        print("AUGMENTING DATA")
        print(f"{'='*80}")
        
        augmented = []
        
        for item in data:
            text = item['text']
            labels = item['labels']
            
            # Original
            augmented.append(item)
            
            # Variations
            variations = self._generate_variations(text)
            for variation in variations[:augmentation_factor]:
                augmented.append({
                    "text": variation,
                    "labels": labels.copy(),
                    "source": item.get('source', 'unknown') + '_augmented',
                    "category": item.get('category', 'unknown')
                })
        
        print(f"  ✓ Augmented {len(data)} examples to {len(augmented)} examples")
        return augmented
    
    def _generate_variations(self, text: str) -> List[str]:
        """Generate text variations"""
        variations = []
        
        # Capitalization variations
        variations.append(text.lower())
        variations.append(text.upper())
        variations.append(text.capitalize())
        
        # Punctuation variations
        if text[-1] not in '.!?':
            variations.append(text + '.')
            variations.append(text + '!')
            variations.append(text + '?')
        
        # Synonym replacement (simple)
        synonyms = {
            'happy': ['glad', 'pleased', 'joyful', 'delighted'],
            'sad': ['unhappy', 'down', 'depressed', 'upset'],
            'angry': ['mad', 'furious', 'irritated', 'annoyed'],
            'worried': ['concerned', 'anxious', 'nervous', 'stressed'],
        }
        
        text_lower = text.lower()
        for word, syns in synonyms.items():
            if word in text_lower:
                for syn in syns:
                    variation = text_lower.replace(word, syn)
                    variations.append(variation.capitalize())
        
        return variations[:5]  # Limit variations
    
    def save_data(self, filename='massive_training_data.json'):
        """Save collected data"""
        print(f"\n{'='*80}")
        print("SAVING DATA")
        print(f"{'='*80}")
        
        # Shuffle
        random.shuffle(self.collected_data)
        
        # Split train/val
        split_idx = int(len(self.collected_data) * 0.8)
        train_data = self.collected_data[:split_idx]
        val_data = self.collected_data[split_idx:]
        
        # Save
        with open(filename.replace('.json', '_train.json'), 'w', encoding='utf-8') as f:
            json.dump(train_data, f, indent=2, ensure_ascii=False)
        
        with open(filename.replace('.json', '_val.json'), 'w', encoding='utf-8') as f:
            json.dump(val_data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Saved {len(train_data)} training examples")
        print(f"✓ Saved {len(val_data)} validation examples")
        print(f"✓ Total: {len(self.collected_data)} examples")
        
        # Statistics
        self._print_statistics()
    
    def _print_statistics(self):
        """Print data statistics"""
        print(f"\n{'='*80}")
        print("DATA STATISTICS")
        print(f"{'='*80}")
        
        category_counts = {}
        source_counts = {}
        
        for item in self.collected_data:
            category = item.get('category', 'unknown')
            source = item.get('source', 'unknown')
            
            category_counts[category] = category_counts.get(category, 0) + 1
            source_counts[source] = source_counts.get(source, 0) + 1
        
        print("\nBy Category:")
        for category, count in sorted(category_counts.items(), key=lambda x: -x[1]):
            print(f"  {category}: {count}")
        
        print("\nBy Source:")
        for source, count in sorted(source_counts.items(), key=lambda x: -x[1]):
            print(f"  {source}: {count}")


def main():
    """Main data collection pipeline"""
    print("="*80)
    print("MASSIVE DATA COLLECTION FOR COMPREHENSIVE TRAINING")
    print("="*80)
    print("\nThis will collect data from multiple sources:")
    print("  1. LLM generation (GPT-4, Claude)")
    print("  2. Reddit scraping")
    print("  3. Public datasets")
    print("  4. Data augmentation")
    print("\nThis may take 1-2 hours depending on sources...")
    print("="*80)
    
    collector = MassiveDataCollector()
    
    # Step 1: Generate with LLM
    collector.generate_with_llm(num_examples=2000)
    
    # Step 2: Scrape Reddit
    collector.scrape_reddit_data(limit=5000)
    
    # Step 3: Load public datasets
    collector.load_public_datasets()
    
    # Step 4: Generate comprehensive fixed data
    print(f"\n{'='*80}")
    print("GENERATING COMPREHENSIVE FIXED DATA")
    print(f"{'='*80}")
    try:
        from generate_comprehensive_fixed_training_data import generate_comprehensive_fixed_data
        generate_comprehensive_fixed_data()
        
        # Load the generated data
        with open('train_data.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                item['source'] = 'comprehensive_fixed'
                collector.collected_data.append(item)
        print(f"✓ Added {len(data)} examples from comprehensive fixed data")
    except Exception as e:
        print(f"⚠ Error generating comprehensive data: {e}")
    
    # Step 5: Augment data
    if collector.collected_data:
        collector.collected_data = collector.augment_data(collector.collected_data, augmentation_factor=2)
    
    # Step 6: Save
    collector.save_data('massive_training_data.json')
    
    print("\n" + "="*80)
    print("DATA COLLECTION COMPLETE!")
    print("="*80)
    print("\nNext steps:")
    print("  1. Review the data: massive_training_data_train.json")
    print("  2. Train model: python train_massive.py")
    print("  3. Or use existing: python train_comprehensive_fixed.py")


if __name__ == '__main__':
    main()

