# import necessary libraries
import argparse
import pandas as pd
import numpy as np
import pickle
import re
import os
import sys
from scipy.sparse import hstack

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class Config:
    MODEL_DIR = './models/'
    MAX_SEQUENCE_LENGTH = 150
    VOCAB_SIZE = 10000

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_features(df):
    features = pd.DataFrame()
    
    features['exclamation_count'] = df['text'].str.count('!')
    features['question_count'] = df['text'].str.count(r'\?')  # Fixed: added 'r' prefix
    features['ellipsis_count'] = df['text'].str.count(r'\.\.\.')
    
    features['capital_ratio'] = df['text'].apply(
        lambda x: sum(1 for c in x if c.isupper()) / (len(x) + 1)
    )
    features['has_all_caps_word'] = df['text'].str.contains(r'\b[A-Z]{2,}\b').astype(int)
    
    features['text_length'] = df['text'].str.len()
    features['word_count'] = df['text'].str.split().str.len()
    
    return features

class SarcasmDetector:
    def __init__(self, model_dir='./models/'):
        self.model_dir = model_dir
        
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Models directory '{model_dir}' not found.")
        
        try:
            # Load Tokenizers [cite: 148, 216]
            with open(os.path.join(model_dir, 'tokenizer.pkl'), 'rb') as f:
                self.tokenizer = pickle.load(f)
            with open(os.path.join(model_dir, 'tokenizer_clean.pkl'), 'rb') as f:
                self.tokenizer_clean = pickle.load(f)
            
            # Load Deep Learning Models (.h5 files as specified)
            # 1. Standard LSTM
            self.model_lstm = load_model(os.path.join(model_dir, 'lstm_model.h5'))
            # 2. Bidirectional LSTM
            self.model_bilstm = load_model(os.path.join(model_dir, 'bilstm_model.h5'))
            # 3. CNN Model
            self.model_cnn = load_model(os.path.join(model_dir, 'cnn_model.h5'))
            
            print(" - All models loaded successfully.")
            
        except Exception as e:
            print(f"Error loading models: {e}")
            sys.exit(1)
    
    def predict(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        
        df = pd.DataFrame({'text': texts})
        clean_texts = df['text'].apply(preprocess_text)
        seq_standard = pad_sequences(self.tokenizer.texts_to_sequences(df['text']), 
                                    maxlen=Config.MAX_SEQUENCE_LENGTH)
        seq_clean = pad_sequences(self.tokenizer_clean.texts_to_sequences(clean_texts), 
                                 maxlen=Config.MAX_SEQUENCE_LENGTH)
        
        print("Running Ensemble Inference...")
        pred_lstm = (self.model_lstm.predict(seq_standard, verbose=0) > 0.5).astype(int).flatten()
        pred_bilstm = (self.model_bilstm.predict(seq_clean, verbose=0) > 0.5).astype(int).flatten()
        pred_cnn = (self.model_cnn.predict(seq_clean, verbose=0) > 0.5).astype(int).flatten()
        
        votes = pred_lstm + pred_bilstm + pred_cnn
        ensemble_predictions = (votes >= 2).astype(int)
        
        return ensemble_predictions
    
    def predict_with_confidence(self, texts):
        predictions = self.predict(texts)
        
        votes = 0
        for model_key, model in self.models.items():
            votes += self.predict(texts)
        
        confidence = votes / len(self.models)
        
        return predictions, confidence

def main():
    parser = argparse.ArgumentParser(
        description='Sarcasm Detection Inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Example usage: python predict_sarcasm.py --input test_data.csv --output predictions.csv
        """
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input CSV file'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to output CSV file'
    )
    
    parser.add_argument(
        '--model-dir',
        type=str,
        default='./models/',
        help='Directory containing trained models (default: ./models/)'
    )
    
    args = parser.parse_args()
    
    print("-" * 20)
    print(f" - Input file: {args.input}")
    print(f" - Output file: {args.output}")
    print()
    
    # Load input data
    try:
        df = pd.read_csv(args.input)
        print(f" - Loaded {len(df)} examples from {args.input}")
    except FileNotFoundError:
        print(f"Error: Input file '{args.input}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading input file: {e}")
        sys.exit(1)
    
    if 'text' not in df.columns:
        print("Error: Input CSV must have a column named 'text'")
        print(f"Found columns: {df.columns.tolist()}")
        sys.exit(1)
    
    # Handle missing text values
    if df['text'].isnull().any():
        print(f"Warning: Found {df['text'].isnull().sum()} missing text values")
        print("Filling missing values with empty string")
        df['text'] = df['text'].fillna('')
    
    detector = SarcasmDetector(model_dir=args.model_dir)
    
    print(f"\nMaking predictions...")
    predictions = detector.predict(df['text'])
    
    output_df = pd.DataFrame({
        'text': df['text'],
        'prediction': predictions
    })
    
    # Save predictions
    try:
        output_df.to_csv(args.output, index=False)
        print(f" - Saved predictions to {args.output}")
    except Exception as e:
        print(f" ! Error saving predictions: {e}")
        sys.exit(1)
    
    print("\n" + "-" * 20)
    print("SUMMARY")
    print("-" * 20)
    print(f"Total examples: {len(predictions)}")
    print(f"Predicted NOT sarcastic (0): {(predictions == 0).sum()} ({100*(predictions == 0).sum()/len(predictions):.1f}%)")
    print(f"Predicted sarcastic (1): {(predictions == 1).sum()} ({100*(predictions == 1).sum()/len(predictions):.1f}%)")
    
    print("-----")
    print(f"Recall: {recall_score(df['label'], predictions)*100:.4f}%")
    print(f"Precision: {precision_score(df['label'], predictions)*100:.4f}%")
    print(f"F1: {f1_score(df['label'], predictions)*100:.4f}%")
    print(f"Accuracy: {accuracy_score(df['label'], predictions)*100:.4f}%")
    
    print("-----")

if __name__ == "__main__":
    main()
