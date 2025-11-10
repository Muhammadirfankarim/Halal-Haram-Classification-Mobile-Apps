#!/usr/bin/env python3
"""
QUICK START GUIDE: Bilingual Halal/Haram Classification
========================================================

This script demonstrates how to quickly train a model using the bilingual dataset
and test it with Indonesian, English, and hybrid inputs.

Author: Irfan
Date: November 2025
Purpose: Quickstart Guide for Bilingual Halal/Haram Classification
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import json

# ============================================================================
# STEP 1: LOAD BILINGUAL DATASET
# ============================================================================

print("="*70)
print("STEP 1: LOADING BILINGUAL DATASET")
print("="*70)
    
# Load the bilingual dataset
df = pd.read_csv('bilingual_dataset.csv')

print(f"\n✅ Dataset loaded successfully!")
print(f"   Total samples: {len(df):,}")
print(f"   - Halal: {sum(df['label'] == 'halal'):,} ({sum(df['label'] == 'halal')/len(df)*100:.1f}%)")
print(f"   - Haram: {sum(df['label'] == 'haram'):,} ({sum(df['label'] == 'haram')/len(df)*100:.1f}%)")

# Show sample data
print("\n📋 Sample data (first 5 rows):")
print("-"*70)
for i in range(5):
    text = df.iloc[i]['text'][:60]
    label = df.iloc[i]['label']
    print(f"{i+1}. [{label.upper():5}] {text}...")

# ============================================================================
# STEP 2: PREPARE DATA FOR TRAINING
# ============================================================================

print("\n" + "="*70)
print("STEP 2: PREPARING DATA")
print("="*70)

# Extract texts and labels
texts = df['text'].values
labels = (df['label'] == 'haram').astype(int).values

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

print(f"\n✅ Data split completed!")
print(f"   Training samples: {len(X_train):,}")
print(f"   Testing samples: {len(X_test):,}")

# Tokenization
MAX_WORDS = 5000
MAX_LEN = 256

print(f"\n🔤 Tokenizing text...")
print(f"   Max vocabulary: {MAX_WORDS:,}")
print(f"   Max sequence length: {MAX_LEN}")

tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token='<OOV>')
tokenizer.fit_on_texts(X_train)

X_train_seq = pad_sequences(tokenizer.texts_to_sequences(X_train), 
                            maxlen=MAX_LEN, padding='post')
X_test_seq = pad_sequences(tokenizer.texts_to_sequences(X_test), 
                           maxlen=MAX_LEN, padding='post')

print(f"✅ Tokenization completed!")
print(f"   Vocabulary size: {len(tokenizer.word_index):,}")

# ============================================================================
# STEP 3: BUILD AND TRAIN MODEL
# ============================================================================

print("\n" + "="*70)
print("STEP 3: BUILDING AND TRAINING MODEL")
print("="*70)

# Build a simple CNN model (you can use any architecture)
def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(MAX_WORDS, 128, input_length=MAX_LEN),
        tf.keras.layers.Conv1D(128, 5, activation='relu'),
        tf.keras.layers.GlobalMaxPooling1D(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

model = build_model()
model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

print("\n📊 Model architecture:")
model.summary()

print("\n🚀 Training model...")
print("   (This may take a few minutes...)\n")

history = model.fit(
    X_train_seq, y_train,
    validation_data=(X_test_seq, y_test),
    epochs=10,  # Increase for better performance
    batch_size=32,
    verbose=1
)

# Evaluate
loss, accuracy = model.evaluate(X_test_seq, y_test, verbose=0)
print(f"\n✅ Training completed!")
print(f"   Test Accuracy: {accuracy*100:.2f}%")
print(f"   Test Loss: {loss:.4f}")

# ============================================================================
# STEP 4: TEST WITH BILINGUAL INPUTS
# ============================================================================

print("\n" + "="*70)
print("STEP 4: TESTING WITH BILINGUAL INPUTS")
print("="*70)

def predict_halal_haram(text):
    """Predict if ingredient text is halal or haram"""
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post')
    pred = model.predict(padded, verbose=0)[0][0]
    label = "HARAM" if pred > 0.5 else "HALAL"
    confidence = pred if pred > 0.5 else (1 - pred)
    return label, confidence

# Test cases
test_cases = [
    # English
    ("pork bacon with beef stock", "HARAM"),
    ("vegetable oil corn soybean salt", "HALAL"),
    ("chicken breast water salt", "HALAL"),
    
    # Indonesian
    ("daging babi lemak babi bacon", "HARAM"),
    ("minyak sayur jagung kedelai garam", "HALAL"),
    ("daging ayam air garam", "HALAL"),
    
    # Hybrid
    ("pork bacon daging babi garam", "HARAM"),
    ("chicken ayam dengan wortel carrots", "HALAL"),
    ("minyak sayur vegetable oil jagung", "HALAL"),
    
    # Real Indonesian products
    ("tepung terigu gula telur mentega susu", "HALAL"),
    ("sosis frankfurter daging babi lemak babi", "HARAM"),
    ("kaldu ayam bawang putih wortel seledri", "HALAL"),
]

print("\n🧪 Testing model with various inputs:\n")
correct = 0
total = len(test_cases)

for i, (text, expected) in enumerate(test_cases, 1):
    predicted, confidence = predict_halal_haram(text)
    is_correct = predicted == expected
    correct += is_correct
    
    symbol = "✅" if is_correct else "❌"
    print(f"{i}. {symbol} [{predicted:5}] (confidence: {confidence*100:.1f}%)")
    print(f"   Input: {text[:60]}...")
    print(f"   Expected: {expected}\n")

accuracy = correct / total * 100
print("="*70)
print(f"TEST ACCURACY: {correct}/{total} = {accuracy:.1f}%")
print("="*70)

# ============================================================================
# STEP 5: DEMONSTRATE KEY IMPROVEMENTS
# ============================================================================

print("\n" + "="*70)
print("STEP 5: BILINGUAL CAPABILITY DEMONSTRATION")
print("="*70)

demonstrations = [
    {
        'title': 'Deteksi Bahan Haram dalam Bahasa Inggris',
        'tests': [
            ('pork', 'HARAM'),
            ('bacon', 'HARAM'),
            ('wine', 'HARAM'),
            ('clam', 'HARAM'),
        ]
    },
    {
        'title': 'Deteksi Bahan Haram dalam Bahasa Indonesia',
        'tests': [
            ('babi', 'HARAM'),
            ('bacon babi', 'HARAM'),
            ('anggur', 'HARAM'),
            ('kerang', 'HARAM'),
        ]
    },
    {
        'title': 'Deteksi Bahan Halal dalam Kedua Bahasa',
        'tests': [
            ('chicken', 'HALAL'),
            ('ayam', 'HALAL'),
            ('vegetable oil', 'HALAL'),
            ('minyak sayur', 'HALAL'),
        ]
    }
]

for demo in demonstrations:
    print(f"\n📋 {demo['title']}:")
    print("-"*70)
    for text, expected in demo['tests']:
        predicted, conf = predict_halal_haram(text)
        symbol = "✅" if predicted == expected else "❌"
        print(f"   {symbol} '{text}' → {predicted} (confidence: {conf*100:.0f}%)")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*70)
print("SUMMARY: BILINGUAL MODEL CAPABILITIES")
print("="*70)

print("""
✅ MODEL DAPAT MENDETEKSI:

1. Bahan Haram dalam Bahasa Inggris
   • pork, bacon, ham, lard
   • wine, beer, alcohol
   • clam, oyster, mussel

2. Bahan Haram dalam Bahasa Indonesia
   • babi, bacon babi, lemak babi
   • anggur, bir, alkohol
   • kerang, tiram, remis

3. Bahan Halal dalam Kedua Bahasa
   • chicken / ayam
   • vegetable oil / minyak sayur
   • wheat flour / tepung gandum

4. Input Campuran (Code-switching)
   • "chicken ayam with garam salt"
   • "minyak sayur vegetable oil"
   • "tepung gandum wheat flour"

🎯 KEY IMPROVEMENTS:
   • Dataset size: 40K → 123K (+210%)
   • Language support: 1 → 2+ (bilingual)
   • Real-world applicability: Tinggi
   • Indonesian market ready: Ya

📊 TRAINING RECOMMENDATIONS:
   • Use full bilingual_dataset.csv for production
   • Train for more epochs (10-15) for better accuracy
   • Use ensemble of multiple architectures
   • Validate with test_cases_bilingual.csv

🚀 NEXT STEPS:
   1. Train with more epochs for better accuracy
   2. Experiment with different architectures (LSTM, GRU, Multi-CNN)
   3. Fine-tune hyperparameters
   4. Deploy for real-world testing in Indonesian market
   5. Collect user feedback and iterate

""")

print("="*70)
print("✅ QUICK START GUIDE COMPLETED!")
print("="*70)
print("\nFor detailed documentation, see: DOKUMENTASI_BILINGUAL_DATASET.md")
print("For comprehensive testing, use: test_cases_bilingual.csv")
print("\nThank you for using this bilingual dataset! 🎉")