import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import json
import os
from data_preparation import prepare_dataset

# Constants
MAX_WORDS = 10000
MAX_LEN = 100
EMBEDDING_DIM = 100

def load_and_preprocess_data():
    # Load the processed dataset and emotion mapping
    df = pd.read_csv('emotion_dataset.csv')
    
    # Ensure text column is string type
    df['text'] = df['text'].astype(str)
    
    # Load emotion mapping
    with open('emotion_mapping.json', 'r') as f:
        emotion_mapping = json.load(f)
    
    # Convert emotions to numerical labels
    df['label'] = df['emotion'].map(emotion_mapping)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['label'], test_size=0.2, random_state=42
    )
    
    # Tokenize the text
    tokenizer = Tokenizer(num_words=MAX_WORDS)
    tokenizer.fit_on_texts(X_train)
    
    # Convert text to sequences
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    
    # Pad sequences
    X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_LEN)
    X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_LEN)
    
    # Save tokenizer in the emotion_model directory
    with open('tokenizer.json', 'w') as f:
        json.dump(tokenizer.word_index, f)
    
    return X_train_pad, X_test_pad, y_train, y_test

def create_model():
    model = Sequential([
        Embedding(MAX_WORDS, EMBEDDING_DIM, input_length=MAX_LEN),
        LSTM(128, return_sequences=True),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(6, activation='softmax')  # 6 emotions
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model():
    # Create model directory if it doesn't exist
    os.makedirs('emotion_model', exist_ok=True)
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    
    # Create and compile model
    model = create_model()
    
    # Define callbacks
    checkpoint = ModelCheckpoint(
        'emotion_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        batch_size=64,
        epochs=20,
        validation_data=(X_test, y_test),
        callbacks=[checkpoint, early_stopping]
    )
    
    # Save model architecture in the emotion_model directory
    model.save('emotion_model.h5')
    
    # Load emotion mapping again to save emotions list
    with open('emotion_mapping.json', 'r') as f:
        emotion_mapping = json.load(f)
        
    # Save emotions list in the emotion_model directory
    with open('emotions.json', 'w') as f:
        json.dump(list(emotion_mapping.keys()), f)
    
    return model

if __name__ == "__main__":
    train_model() 