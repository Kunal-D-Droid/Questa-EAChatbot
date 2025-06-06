import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
import os
from datasets import load_dataset

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Define the 6 target emotions
TARGET_EMOTIONS = ['joy', 'sadness', 'anger', 'fear', 'surprise', 'neutral']

# Mapping from GoEmotions labels (28) to our 6 target emotions
# GoEmotions labels: admiration, amusement, anger, annoyance, approval, caring, confusion, curiosity, desire, disappointment, disapproval, disgust, embarrassment, excitement, fear, gratitude, grief, joy, love, nervousness, optimism, pride, realization, relief, remorse, sadness, surprise, neutral.
GOEMOTIONS_TO_TARGET = {
    'admiration': 'joy',
    'amusement': 'joy',
    'anger': 'anger',
    'annoyance': 'anger', # Mapping annoyance to anger
    'approval': 'joy', # Mapping approval to joy
    'caring': 'joy', # Mapping caring to joy
    'confusion': 'neutral', # Mapping confusion to neutral
    'curiosity': 'neutral', # Mapping curiosity to neutral
    'desire': 'joy', # Mapping desire to joy
    'disappointment': 'sadness', # Mapping disappointment to sadness
    'disapproval': 'anger', # Mapping disapproval to anger
    'disgust': 'anger', # Mapping disgust to anger
    'embarrassment': 'sadness', # Mapping embarrassment to sadness
    'excitement': 'joy', # Mapping excitement to joy
    'fear': 'fear',
    'gratitude': 'joy', # Mapping gratitude to joy
    'grief': 'sadness',
    'joy': 'joy',
    'love': 'joy',
    'nervousness': 'fear', # Mapping nervousness to fear
    'optimism': 'joy',
    'pride': 'joy',
    'realization': 'neutral', # Mapping realization to neutral
    'relief': 'joy', # Mapping relief to joy
    'remorse': 'sadness', # Mapping remorse to sadness
    'sadness': 'sadness',
    'surprise': 'surprise',
    'neutral': 'neutral'
}

def clean_text(text):
    # Ensure input is a string
    if not isinstance(text, str):
        return ""
        
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

def map_goemotions_to_target(goemotion_labels, goemotion_label_names):
    """Maps GoEmotions labels (indices) to our target 6 emotions."""
    mapped_emotions = []
    # GoEmotions can have multiple labels, we'll prioritize non-neutral emotions
    for label_id in goemotion_labels:
        goemotion_name = goemotion_label_names[label_id]
        target_emotion = GOEMOTIONS_TO_TARGET.get(goemotion_name)
        if target_emotion and target_emotion != 'neutral':
            mapped_emotions.append(target_emotion)
    
    # If no non-neutral emotions were mapped, check for neutral
    if not mapped_emotions:
         for label_id in goemotion_labels:
            goemotion_name = goemotion_label_names[label_id]
            target_emotion = GOEMOTIONS_TO_TARGET.get(goemotion_name)
            if target_emotion == 'neutral':
                mapped_emotions.append('neutral')
                break # Only need one neutral label
    
    # If still no mapped emotions (e.g., empty original labels), default to neutral
    if not mapped_emotions:
        return 'neutral'
        
    # For simplicity, return the first mapped emotion (prioritizing non-neutral)
    return mapped_emotions[0]

def prepare_dataset():
    print("Loading GoEmotions dataset...")
    # Load the simplified version of the GoEmotions dataset
    dataset = load_dataset('go_emotions', 'simplified')
    
    # Combine train, validation, and test splits for training
    df = pd.concat([
        dataset['train'].to_pandas(),
        dataset['validation'].to_pandas(),
        dataset['test'].to_pandas()
    ], ignore_index=True)
    
    print("Mapping emotions to target emotions...")
    # Get the list of GoEmotions label names
    goemotion_label_names = dataset['train'].features['labels'].feature.names
    
    # Apply the mapping function
    df['emotion'] = df['labels'].apply(lambda x: map_goemotions_to_target(x, goemotion_label_names))
    
    # Filter out any rows that somehow didn't get an emotion mapped (shouldn't happen with current logic but good safeguard)
    df = df[df['emotion'].isin(TARGET_EMOTIONS)]
    
    print("Cleaning text...")
    # Clean the text
    df['text'] = df['text'].apply(clean_text)
    
    print("Saving processed dataset...")
    # Save the processed dataset
    df = df[['text', 'emotion']] # Keep only necessary columns
    df.to_csv('emotion_dataset.csv', index=False)
    
    # Create emotion mapping for the 6 target emotions
    emotion_mapping = {emotion: idx for idx, emotion in enumerate(TARGET_EMOTIONS)}
    
    # Save emotion mapping
    import json
    with open('emotion_mapping.json', 'w') as f:
        json.dump(emotion_mapping, f)
    
    print("Dataset preparation completed!")
    print(f"Total samples: {len(df)}")
    print("Emotion distribution:")
    print(df['emotion'].value_counts())
    return df

if __name__ == "__main__":
    prepare_dataset() 