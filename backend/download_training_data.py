"""
Download training data from Kaggle's Toxic Comment Classification Challenge
This dataset contains real examples of toxic comments from Wikipedia's talk page edits
"""

import pandas as pd
import numpy as np
import os
import urllib.request
import zipfile

def create_sample_training_data():
    """
    Create a high-quality sample dataset with realistic toxic and non-toxic comments
    for training the model with better accuracy
    """
    
    # Create data directory
    os.makedirs('data', exist_ok=True)
    
    # High-quality training samples with proper labels
    training_data = {
        'comment_text': [
            # Clean examples (all labels = 0)
            "Thank you for your contribution to this article. Your edits are very helpful.",
            "I appreciate your perspective on this topic. Let's continue the discussion respectfully.",
            "Great work on improving this article! The sources you added are excellent.",
            "I disagree with your point, but I respect your opinion. Let's find common ground.",
            "This is an interesting topic. I'd like to learn more about your viewpoint.",
            "The weather is beautiful today and I hope everyone is having a great day.",
            "I love learning new things and expanding my knowledge on various subjects.",
            "Could you please provide more information about this topic? I'm curious to know more.",
            "Your explanation was very clear and helped me understand the concept better.",
            "I found your article very informative and well-researched. Thank you for sharing.",
            
            # Mildly toxic examples
            "Your argument is completely wrong and makes no sense at all.",
            "This is the dumbest thing I've ever read on this website.",
            "Stop posting nonsense and do some research before commenting.",
            "You clearly have no idea what you're talking about here.",
            "This article is trash and needs to be completely rewritten.",
            
            # Toxic examples
            "You're an idiot if you actually believe this garbage.",
            "Shut up already, nobody wants to hear your stupid opinions.",
            "This is complete BS and you're a moron for writing it.",
            "You're such a fool. Your posts are always worthless garbage.",
            "What an idiotic comment. You should be embarrassed.",
            
            # Severe toxic examples
            "You're a pathetic waste of space and everyone hates you.",
            "I hope something terrible happens to you, you worthless piece of garbage.",
            "You deserve to suffer for being such a horrible person.",
            "People like you are the worst. The world would be better without you.",
            
            # Obscene examples
            "This is f***ing ridiculous and complete bulls***.",
            "What the hell is wrong with you? This is s***ty work.",
            "This is absolute cr*p and you should feel ashamed.",
            "Get the f*** out of here with that nonsense.",
            
            # Threat examples
            "I know where you live and I'm coming for you.",
            "You better watch your back because I will find you.",
            "I'm going to make sure you regret posting this.",
            "Someone should teach you a lesson you won't forget.",
            
            # Insult examples
            "You're a complete loser with no friends or life.",
            "Everyone thinks you're annoying and nobody likes you.",
            "You're ugly, stupid, and have no redeeming qualities.",
            "What a pathetic excuse for a human being you are.",
            
            # Identity hate examples
            "People from your country are all criminals and terrorists.",
            "Your religion is evil and should be banned everywhere.",
            "Your race is inferior and that's a scientific fact.",
            "People like you don't belong in our society.",
            
            # Multi-label toxic examples
            "You're a f***ing idiot and I hope you get what's coming to you.",
            "Shut your filthy mouth you worthless piece of trash.",
            "You disgusting pig, someone should put you in your place.",
            "I hate people like you, you're all the same garbage.",
        ],
        'toxic': [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # Clean
            1, 1, 1, 1, 1,  # Mildly toxic
            1, 1, 1, 1, 1,  # Toxic
            1, 1, 1, 1,  # Severe toxic
            1, 1, 1, 1,  # Obscene
            1, 1, 1, 1,  # Threat
            1, 1, 1, 1,  # Insult
            1, 1, 1, 1,  # Identity hate
            1, 1, 1, 1   # Multi-label
        ],
        'severe_toxic': [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            1, 1, 1, 1,
            0, 0, 0, 0,
            1, 1, 1, 1,
            0, 0, 0, 0,
            1, 1, 1, 1,
            1, 1, 1, 1
        ],
        'obscene': [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0,
            1, 1, 1, 1,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            1, 1, 1, 0
        ],
        'threat': [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            1, 1, 1, 0,
            0, 0, 0, 0,
            1, 1, 1, 1,
            0, 0, 0, 0,
            0, 0, 0, 0,
            1, 0, 1, 0
        ],
        'insult': [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            1, 1, 1, 1, 1,
            1, 1, 1, 1,
            0, 0, 0, 0,
            0, 0, 0, 0,
            1, 1, 1, 1,
            0, 0, 0, 0,
            1, 1, 1, 1
        ],
        'identity_hate': [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
            1, 1, 1, 1,
            0, 0, 0, 1
        ]
    }
    
    # Create DataFrame
    df = pd.DataFrame(training_data)
    
    # Add more balanced samples by duplicating clean examples
    # This helps prevent overfitting to toxic samples
    clean_samples = df[df['toxic'] == 0].copy()
    for _ in range(2):  # Duplicate clean samples 2 more times
        df = pd.concat([df, clean_samples], ignore_index=True)
    
    # Shuffle the dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save to CSV
    output_path = 'data/train.csv'
    df.to_csv(output_path, index=False)
    
    print(f"✓ Created training dataset with {len(df)} samples")
    print(f"✓ Saved to: {output_path}")
    print(f"\nDataset statistics:")
    print(f"  - Total samples: {len(df)}")
    print(f"  - Toxic samples: {df['toxic'].sum()}")
    print(f"  - Clean samples: {len(df) - df['toxic'].sum()}")
    print(f"\nLabel distribution:")
    for col in ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']:
        print(f"  - {col}: {df[col].sum()} ({df[col].sum()/len(df)*100:.1f}%)")
    
    return df


def download_kaggle_dataset():
    """
    Download the full Kaggle Toxic Comment dataset
    Note: Requires Kaggle API credentials
    """
    try:
        import kaggle
        
        print("Downloading Kaggle Toxic Comment Classification dataset...")
        print("This requires Kaggle API credentials in ~/.kaggle/kaggle.json")
        
        # Download the dataset
        kaggle.api.dataset_download_files(
            'julian3833/jigsaw-toxic-comment-classification-challenge',
            path='data/',
            unzip=True
        )
        
        print("✓ Successfully downloaded Kaggle dataset")
        return True
    except ImportError:
        print("Kaggle package not installed. Install with: pip install kaggle")
        return False
    except Exception as e:
        print(f"Error downloading from Kaggle: {e}")
        print("Creating sample dataset instead...")
        return False


if __name__ == '__main__':
    print("=" * 60)
    print("TOXIC COMMENT TRAINING DATA SETUP")
    print("=" * 60)
    print()
    
    # Try to download from Kaggle first
    if not download_kaggle_dataset():
        # If that fails, create high-quality sample data
        print("\nCreating high-quality sample training dataset...")
        create_sample_training_data()
    
    print("\n" + "=" * 60)
    print("SETUP COMPLETE")
    print("=" * 60)
    print("\nYou can now train the model by running:")
    print("  python train.py")
    print()
