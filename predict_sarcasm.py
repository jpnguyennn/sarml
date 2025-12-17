"""
Sarcasm Detection Inference Script
Required interface for CS 461 Final Project

Usage:
    python predict_sarcasm.py --input test_data.csv --output predictions.csv

Input format: CSV with at least one column titled 'text'
Output format: CSV with two columns: text and prediction (0=not sarcasm, 1=sarcasm)
"""

import argparse
import pandas as pd
import numpy as np
import pickle
import re
import os
import sys
from scipy.sparse import hstack

# TensorFlow/Keras imports
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Statistic imports
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# ==================== CONFIGURATION ====================
class Config:
    """Configuration for inference"""
    MODEL_DIR = './models/'
    MAX_SEQUENCE_LENGTH = 150
    VOCAB_SIZE = 10000


# ==================== PREPROCESSING ====================
def preprocess_text(text):
    """Clean and preprocess text - MUST match training preprocessing"""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
    text = re.sub(r'\s+', ' ', text).strip()    # Clean whitespace
    return text


def extract_features(df):
    """Extract manual features - MUST match training feature extraction"""
    features = pd.DataFrame()
    
    # Punctuation counts
    features['exclamation_count'] = df['text'].str.count('!')
    features['question_count'] = df['text'].str.count(r'\?')  # Fixed: added 'r' prefix
    features['ellipsis_count'] = df['text'].str.count(r'\.\.\.')
    
    # Capitalization
    features['capital_ratio'] = df['text'].apply(
        lambda x: sum(1 for c in x if c.isupper()) / (len(x) + 1)
    )
    features['has_all_caps_word'] = df['text'].str.contains(r'\b[A-Z]{2,}\b').astype(int)
    
    # Length metrics
    features['text_length'] = df['text'].str.len()
    features['word_count'] = df['text'].str.split().str.len()
    
    return features


# ==================== SARCASM DETECTOR ====================
class SarcasmDetector:
    """
    Loads trained models and performs inference on new text
    """
    
    def __init__(self, model_dir='./models/'):
        """Load trained models and preprocessors"""
        self.model_dir = model_dir
        
        # Check if models directory exists
        if not os.path.exists(model_dir):
            raise FileNotFoundError(
                f"Models directory '{model_dir}' not found. "
                "Please ensure you have trained models in the models/ directory."
            )
        
        print("Loading models...")
        
        try:
            # Load ensemble configuration
            with open(f'{model_dir}model_weights.pkl', 'rb') as f:
                self.ensemble_config = pickle.load(f)
            
            print(f"Ensemble: {self.ensemble_config['best_ensemble']}")
            
            # Load preprocessors
            with open(f'{model_dir}tfidf_vectorizer.pkl', 'rb') as f:
                self.tfidf_vectorizer = pickle.load(f)
            
            with open(f'{model_dir}tokenizer.pkl', 'rb') as f:
                self.tokenizer = pickle.load(f)
            
            # Load models based on ensemble configuration
            self.models = {}
            
            for model_key in self.ensemble_config['model_keys']:
                if 'svm' in model_key:
                    with open(f'{model_dir}svm_model.pkl', 'rb') as f:
                        self.models[model_key] = pickle.load(f)
                elif 'stacked_bilstm' in model_key:
                    self.models[model_key] = load_model(
                        f'{model_dir}stacked_bilstm_model.h5',
                        compile=False
                    )
                elif 'cnn' in model_key:
                    self.models[model_key] = load_model(
                        f'{model_dir}cnn_model.h5',
                        compile=False
                    )
            
            print(f"✓ Successfully loaded {len(self.models)} models")
            
        except FileNotFoundError as e:
            print(f"Error: Required model file not found: {e}")
            print("\nPlease run the training script first:")
            print("  python src/train_model.py")
            sys.exit(1)
        except Exception as e:
            print(f"Error loading models: {e}")
            sys.exit(1)
    
    def predict(self, texts):
        """
        Predict sarcasm for a list of texts
        
        Args:
            texts: List of strings or pandas Series
            
        Returns:
            numpy array of predictions (0 or 1)
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Convert to DataFrame for feature extraction
        df = pd.DataFrame({'text': texts})
        
        # Store original text for features
        original_texts = df['text'].copy()
        
        # Preprocess text
        clean_texts = df['text'].apply(preprocess_text)
        
        # Get predictions from each model in ensemble
        predictions = {}
        
        for model_key, model in self.models.items():
            if 'svm' in model_key:
                # SVM uses TF-IDF features
                tfidf_features = self.tfidf_vectorizer.transform(clean_texts)
                
                # Check if this SVM uses manual features
                if 'with_features' in model_key:
                    manual_features = extract_features(pd.DataFrame({'text': original_texts}))
                    combined_features = hstack([tfidf_features, manual_features])
                    predictions[model_key] = model.predict(combined_features)
                else:
                    predictions[model_key] = model.predict(tfidf_features)
            
            else:
                # Neural networks use sequences
                sequences = self.tokenizer.texts_to_sequences(clean_texts)
                padded_sequences = pad_sequences(
                    sequences,
                    maxlen=Config.MAX_SEQUENCE_LENGTH
                )
                
                probs = model.predict(padded_sequences, verbose=0)
                predictions[model_key] = (probs > 0.5).astype(int).flatten()
        
        # Ensemble: Majority voting
        votes = sum(predictions.values())
        threshold = len(predictions) / 2
        ensemble_predictions = (votes > threshold).astype(int)
        
        return ensemble_predictions
    
    def predict_with_confidence(self, texts):
        """
        Predict with confidence scores
        
        Returns:
            predictions, confidence_scores
        """
        predictions = self.predict(texts)
        
        # Confidence is the proportion of models that agree
        votes = 0
        for model_key, model in self.models.items():
            # Get individual predictions (simplified for confidence)
            votes += self.predict(texts)
        
        confidence = votes / len(self.models)
        
        return predictions, confidence


# ==================== COMMAND LINE INTERFACE ====================
def main():
    """Main inference function matching required interface"""
    parser = argparse.ArgumentParser(
        description='Sarcasm Detection Inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    python predict_sarcasm.py --input test_data.csv --output predictions.csv

Input format:
    CSV file with at least one column titled 'text'

Output format:
    CSV file with two columns: 'text' and 'prediction'
    Predictions: 0 (not sarcasm) or 1 (sarcasm)
        """
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input CSV file with text column'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to output CSV file for predictions'
    )
    
    parser.add_argument(
        '--model-dir',
        type=str,
        default='./models/',
        help='Directory containing trained models (default: ./models/)'
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("SARCASM DETECTION - INFERENCE")
    print("="*60)
    print(f"Input file: {args.input}")
    print(f"Output file: {args.output}")
    print()
    
    # Load input data
    try:
        df = pd.read_csv(args.input)
        print(f"✓ Loaded {len(df)} examples from {args.input}")
    except FileNotFoundError:
        print(f"Error: Input file '{args.input}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading input file: {e}")
        sys.exit(1)
    
    # Check for 'text' column
    if 'text' not in df.columns:
        print("Error: Input CSV must have a column named 'text'")
        print(f"Found columns: {df.columns.tolist()}")
        sys.exit(1)
    
    # Handle missing text values
    if df['text'].isnull().any():
        print(f"Warning: Found {df['text'].isnull().sum()} missing text values")
        print("Filling missing values with empty string")
        df['text'] = df['text'].fillna('')
    
    # Initialize detector
    detector = SarcasmDetector(model_dir=args.model_dir)
    
    # Make predictions
    print(f"\nMaking predictions...")
    predictions = detector.predict(df['text'])
    
    # Create output DataFrame
    output_df = pd.DataFrame({
        'text': df['text'],
        'prediction': predictions
    })
    
    # Save predictions
    try:
        output_df.to_csv(args.output, index=False)
        print(f"✓ Saved predictions to {args.output}")
    except Exception as e:
        print(f"Error saving predictions: {e}")
        sys.exit(1)
    
    # Print summary statistics
    print("\n" + "="*60)
    print("PREDICTION SUMMARY")
    print("="*60)
    print(f"Total examples: {len(predictions)}")
    print(f"Predicted NOT sarcastic (0): {(predictions == 0).sum()} ({100*(predictions == 0).sum()/len(predictions):.1f}%)")
    print(f"Predicted sarcastic (1): {(predictions == 1).sum()} ({100*(predictions == 1).sum()/len(predictions):.1f}%)")
    
    # outputting recall, precision, f1, accuracy
    print("-----")
    print(f"Recall: {recall_score(df['label'], predictions)}")
    print(f"Precision: {precision_score(df['label'], predictions)}")
    print(f"F1: {f1_score(df['label'], predictions)}")
    print(f"Accuracy: {accuracy_score(df['label'], predictions)}")
    
    print("-----")
    
    print("Examples:")
    for idx in range(min(5, len(output_df))):
        text = output_df.iloc[idx]['text'][:80]
        pred = "SARCASTIC" if output_df.iloc[idx]['prediction'] == 1 else "NOT SARCASTIC"
        print(f"  [{pred:15}] {text}...")
    
    print("\n✓ Inference complete!")


if __name__ == "__main__":
    main()
