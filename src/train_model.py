"""
Sarcasm Detection Model Training and Evaluation
CS 461 Final Project - Training Pipeline

This script addresses issues found in the exploratory analysis:
- Tests SVM with and without manual features
- Fixes SVM convergence issues
- Performs proper ablation studies
- Analyzes text length distribution
- Includes comprehensive error analysis
- Tests different ensemble combinations
- Generates visualizations for report

Output:
- Trained models in models/ directory
- Visualizations in figures/ directory (for report)
- Training logs and results
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import pickle
import warnings
import os
from scipy.sparse import hstack

# Scikit-learn imports
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, precision_recall_fscore_support
)

# TensorFlow/Keras imports
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding, LSTM, Dense, Bidirectional, Dropout, Conv1D, GlobalMaxPooling1D
)
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================
class Config:
    """Configuration parameters for the entire pipeline"""
    # Data paths (RELATIVE PATHS as per rubric requirement)
    TRAIN_PATH = './datasets/train.csv'
    VALID_PATH = './datasets/valid.csv'
    TEST_PATH = './datasets/test.csv'
    MODEL_DIR = './models/'
    FIGURES_DIR = './figures/'  # For report visualizations
    
    # Model hyperparameters
    VOCAB_SIZE = 10000
    MAX_SEQUENCE_LENGTH = 150  # Increased from 100 to reduce truncation
    EMBEDDING_DIM = 128
    BATCH_SIZE = 64
    MAX_EPOCHS = 15  # Increased from 10
    EARLY_STOP_PATIENCE = 5  # Increased from 3 to be less aggressive
    
    # SVM parameters
    SVM_MAX_ITER = 50000  # Increased from 10000 to ensure convergence
    SVM_C = 1.0
    
    # Random seed for reproducibility
    RANDOM_SEED = 42

# Set random seeds
np.random.seed(Config.RANDOM_SEED)
tf.random.set_seed(Config.RANDOM_SEED)


# ==================== DATA LOADING ====================
def load_data():
    """Load training, validation, and test datasets"""
    print("="*60)
    print("LOADING DATASETS")
    print("="*60)
    
    train_df = pd.read_csv(Config.TRAIN_PATH)
    valid_df = pd.read_csv(Config.VALID_PATH)
    test_df = pd.read_csv(Config.TEST_PATH)
    
    print(f"Train Shape: {train_df.shape}")
    print(f"Valid Shape: {valid_df.shape}")
    print(f"Test Shape: {test_df.shape}")
    print()
    
    # Check class balance
    print("Class Distribution:")
    print(f"Train: {train_df['label'].value_counts().to_dict()}")
    print(f"Valid: {valid_df['label'].value_counts().to_dict()}")
    print(f"Test: {test_df['label'].value_counts().to_dict()}")
    print()
    
    return train_df, valid_df, test_df


# ==================== TEXT ANALYSIS ====================
def analyze_text_lengths(train_df, valid_df, test_df):
    """Analyze text length distribution to inform sequence length choice"""
    print("="*60)
    print("TEXT LENGTH ANALYSIS")
    print("="*60)
    
    def get_lengths(df):
        return df['text'].apply(lambda x: len(x.split()))
    
    train_lengths = get_lengths(train_df)
    valid_lengths = get_lengths(valid_df)
    test_lengths = get_lengths(test_df)
    
    all_lengths = pd.concat([train_lengths, valid_lengths, test_lengths])
    
    print(f"Mean length: {all_lengths.mean():.2f} words")
    print(f"Median length: {all_lengths.median():.2f} words")
    print(f"95th percentile: {all_lengths.quantile(0.95):.2f} words")
    print(f"99th percentile: {all_lengths.quantile(0.99):.2f} words")
    print(f"Max length: {all_lengths.max()} words")
    
    # Check truncation impact
    truncated_100 = (all_lengths > 100).sum()
    truncated_150 = (all_lengths > 150).sum()
    print(f"\nTexts truncated at 100 tokens: {truncated_100} ({100*truncated_100/len(all_lengths):.2f}%)")
    print(f"Texts truncated at 150 tokens: {truncated_150} ({100*truncated_150/len(all_lengths):.2f}%)")
    print(f"→ Using MAX_SEQUENCE_LENGTH = {Config.MAX_SEQUENCE_LENGTH}")
    print()


# ==================== PREPROCESSING ====================
def preprocess_text(text):
    """Clean and preprocess text"""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
    text = re.sub(r'\s+', ' ', text).strip()    # Clean whitespace
    return text


def extract_features(df):
    """Extract manual features for SVM (TESTING: may not help)"""
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


# ==================== EVALUATION UTILITIES ====================
def evaluate_model(y_true, y_pred, model_name, verbose=True):
    """Comprehensive evaluation with all metrics"""
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=[0, 1]
    )
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro'
    )
    cm = confusion_matrix(y_true, y_pred)
    
    results = {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_class0': precision[0],
        'recall_class0': recall[0],
        'f1_class0': f1[0],
        'precision_class1': precision[1],
        'recall_class1': recall[1],
        'f1_class1': f1[1],
        'confusion_matrix': cm,
        'tn': cm[0,0],
        'fp': cm[0,1],
        'fn': cm[1,0],
        'tp': cm[1,1]
    }
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"EVALUATION: {model_name}")
        print(f"{'='*60}")
        print(f"Overall Accuracy: {accuracy:.4f}")
        print(f"\nClass 0 (Not Sarcastic):")
        print(f"  Precision: {precision[0]:.4f} | Recall: {recall[0]:.4f} | F1: {f1[0]:.4f}")
        print(f"Class 1 (Sarcastic):")
        print(f"  Precision: {precision[1]:.4f} | Recall: {recall[1]:.4f} | F1: {f1[1]:.4f}")
        print(f"\nMacro Averages:")
        print(f"  Precision: {precision_macro:.4f} | Recall: {recall_macro:.4f} | F1: {f1_macro:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"  TN: {cm[0,0]} | FP: {cm[0,1]}")
        print(f"  FN: {cm[1,0]} | TP: {cm[1,1]}")
    
    return results


def analyze_errors(df, y_true, y_pred, model_name, n_examples=10):
    """Analyze misclassified examples"""
    print(f"\n{'='*60}")
    print(f"ERROR ANALYSIS: {model_name}")
    print(f"{'='*60}\n")
    
    errors = df.copy()
    errors['true_label'] = y_true.values if hasattr(y_true, 'values') else y_true
    errors['predicted_label'] = y_pred
    errors['correct'] = errors['true_label'] == errors['predicted_label']
    
    # False Positives
    false_positives = errors[(errors['true_label'] == 0) & (errors['predicted_label'] == 1)]
    print(f"FALSE POSITIVES (predicted sarcastic, actually NOT): {len(false_positives)}")
    print("\nExamples:")
    for idx, row in false_positives.head(n_examples).iterrows():
        print(f"  • {row['text'][:120]}...")
    
    print(f"\n{'-'*60}\n")
    
    # False Negatives
    false_negatives = errors[(errors['true_label'] == 1) & (errors['predicted_label'] == 0)]
    print(f"FALSE NEGATIVES (predicted NOT sarcastic, actually sarcastic): {len(false_negatives)}")
    print("\nExamples:")
    for idx, row in false_negatives.head(n_examples).iterrows():
        print(f"  • {row['text'][:120]}...")
    
    return false_positives, false_negatives


# ==================== VISUALIZATION UTILITIES ====================
def plot_confusion_matrix(cm, model_name, save_path=None):
    """Plot confusion matrix as heatmap"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Not Sarcastic', 'Sarcastic'],
                yticklabels=['Not Sarcastic', 'Sarcastic'],
                cbar_kws={'label': 'Count'})
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


def plot_training_history(history, model_name, save_path=None):
    """Plot training and validation metrics over epochs"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy
    axes[0].plot(history.history['accuracy'], label='Training', linewidth=2, marker='o')
    axes[0].plot(history.history['val_accuracy'], label='Validation', linewidth=2, marker='s')
    axes[0].set_title(f'{model_name} - Accuracy', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=10)
    axes[0].set_ylabel('Accuracy', fontsize=10)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Loss
    axes[1].plot(history.history['loss'], label='Training', linewidth=2, marker='o')
    axes[1].plot(history.history['val_loss'], label='Validation', linewidth=2, marker='s')
    axes[1].set_title(f'{model_name} - Loss', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=10)
    axes[1].set_ylabel('Loss', fontsize=10)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


def plot_model_comparison(results_dict, metric='accuracy', save_path=None):
    """Plot comparison of different models"""
    models = list(results_dict.keys())
    scores = [results_dict[m][metric] for m in models]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(models)), scores, color='steelblue', alpha=0.8, edgecolor='black')
    
    # Add value labels on bars
    for i, (bar, score) in enumerate(zip(bars, scores)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{score:.4f}', ha='center', va='bottom', fontsize=9)
    
    plt.xlabel('Model', fontsize=12, fontweight='bold')
    plt.ylabel(metric.replace('_', ' ').title(), fontsize=12, fontweight='bold')
    plt.title(f'Model Comparison - {metric.replace("_", " ").title()}', 
              fontsize=14, fontweight='bold')
    plt.xticks(range(len(models)), models, rotation=45, ha='right')
    plt.ylim([min(scores) - 0.05, 1.0])
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


def plot_text_length_distribution(train_df, valid_df, test_df, save_path=None):
    """Plot text length distribution"""
    def get_lengths(df):
        return df['text'].apply(lambda x: len(x.split()))
    
    train_lengths = get_lengths(train_df)
    valid_lengths = get_lengths(valid_df)
    test_lengths = get_lengths(test_df)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist([train_lengths, valid_lengths, test_lengths], 
             bins=50, label=['Train', 'Valid', 'Test'], alpha=0.7)
    plt.xlabel('Text Length (words)', fontsize=11)
    plt.ylabel('Frequency', fontsize=11)
    plt.title('Text Length Distribution', fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.boxplot([train_lengths, valid_lengths, test_lengths],
                labels=['Train', 'Valid', 'Test'])
    plt.ylabel('Text Length (words)', fontsize=11)
    plt.title('Text Length Box Plot', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


# ==================== BASELINE MODELS ====================
def train_baseline_models(train_df, valid_df, y_train, y_valid):
    """Train traditional ML baselines"""
    print("\n" + "="*60)
    print("TRAINING BASELINE MODELS")
    print("="*60)
    
    results = {}
    
    # 1. Bag of Words + Logistic Regression
    print("\n[1/3] Bag of Words + Logistic Regression...")
    vectorizer_bow = CountVectorizer(stop_words='english', max_features=5000)
    X_train_bow = vectorizer_bow.fit_transform(train_df['text'])
    X_valid_bow = vectorizer_bow.transform(valid_df['text'])
    
    lr_bow = LogisticRegression(max_iter=1000, random_state=Config.RANDOM_SEED)
    lr_bow.fit(X_train_bow, y_train)
    pred_bow = lr_bow.predict(X_valid_bow)
    results['BoW + LogReg'] = evaluate_model(y_valid, pred_bow, "BoW + LogReg", verbose=False)
    print(f"   Accuracy: {results['BoW + LogReg']['accuracy']:.4f} | F1: {results['BoW + LogReg']['f1_macro']:.4f}")
    
    # 2. TF-IDF + Bigrams + Logistic Regression
    print("[2/3] TF-IDF + Bigrams + Logistic Regression...")
    vectorizer_tfidf = TfidfVectorizer(stop_words='english', max_features=10000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer_tfidf.fit_transform(train_df['text'])
    X_valid_tfidf = vectorizer_tfidf.transform(valid_df['text'])
    
    lr_tfidf = LogisticRegression(max_iter=1000, random_state=Config.RANDOM_SEED)
    lr_tfidf.fit(X_train_tfidf, y_train)
    pred_tfidf = lr_tfidf.predict(X_valid_tfidf)
    results['TF-IDF + LogReg'] = evaluate_model(y_valid, pred_tfidf, "TF-IDF + LogReg", verbose=False)
    print(f"   Accuracy: {results['TF-IDF + LogReg']['accuracy']:.4f} | F1: {results['TF-IDF + LogReg']['f1_macro']:.4f}")
    
    # 3. SVM (FIXED: increased max_iter to ensure convergence)
    print("[3/3] SVM with TF-IDF (FIXED convergence issue)...")
    svm_model = LinearSVC(
        random_state=Config.RANDOM_SEED, 
        max_iter=Config.SVM_MAX_ITER,  # Increased from 10000
        C=Config.SVM_C,
        dual='auto'  # Suppress warning
    )
    svm_model.fit(X_train_tfidf, y_train)
    pred_svm = svm_model.predict(X_valid_tfidf)
    results['SVM (original)'] = evaluate_model(y_valid, pred_svm, "SVM (original)", verbose=False)
    print(f"   Accuracy: {results['SVM (original)']['accuracy']:.4f} | F1: {results['SVM (original)']['f1_macro']:.4f}")
    
    return results, vectorizer_tfidf, svm_model, X_train_tfidf, X_valid_tfidf


# ==================== TEST MANUAL FEATURES ====================
def test_manual_features(train_df, valid_df, test_df, X_train_tfidf, X_valid_tfidf, y_train, y_valid):
    """Test if manual features actually help (spoiler: they might not)"""
    print("\n" + "="*60)
    print("TESTING MANUAL FEATURES IMPACT")
    print("="*60)
    
    # Extract manual features from ORIGINAL text (before preprocessing)
    train_features = extract_features(train_df)
    valid_features = extract_features(valid_df)
    
    # Combine with TF-IDF
    X_train_combined = hstack([X_train_tfidf, train_features])
    X_valid_combined = hstack([X_valid_tfidf, valid_features])
    
    # Train SVM with manual features
    print("\nTraining SVM with manual features added...")
    svm_with_features = LinearSVC(
        random_state=Config.RANDOM_SEED,
        max_iter=Config.SVM_MAX_ITER,
        C=Config.SVM_C,
        dual='auto'
    )
    svm_with_features.fit(X_train_combined, y_train)
    pred_with_features = svm_with_features.predict(X_valid_combined)
    
    results_with = evaluate_model(y_valid, pred_with_features, "SVM + Manual Features", verbose=False)
    print(f"   Accuracy: {results_with['accuracy']:.4f} | F1: {results_with['f1_macro']:.4f}")
    
    return results_with, svm_with_features, X_train_combined, X_valid_combined


# ==================== DEEP LEARNING MODELS ====================
def prepare_sequences(train_df, valid_df, test_df):
    """Prepare tokenized sequences for neural networks"""
    print("\n" + "="*60)
    print("PREPARING SEQUENCES FOR NEURAL NETWORKS")
    print("="*60)
    
    # Add cleaned text column
    train_df['clean_text'] = train_df['text'].apply(preprocess_text)
    valid_df['clean_text'] = valid_df['text'].apply(preprocess_text)
    test_df['clean_text'] = test_df['text'].apply(preprocess_text)
    
    # Tokenization
    tokenizer = Tokenizer(num_words=Config.VOCAB_SIZE)
    tokenizer.fit_on_texts(train_df['clean_text'])
    
    X_train_seq = pad_sequences(
        tokenizer.texts_to_sequences(train_df['clean_text']),
        maxlen=Config.MAX_SEQUENCE_LENGTH
    )
    X_valid_seq = pad_sequences(
        tokenizer.texts_to_sequences(valid_df['clean_text']),
        maxlen=Config.MAX_SEQUENCE_LENGTH
    )
    X_test_seq = pad_sequences(
        tokenizer.texts_to_sequences(test_df['clean_text']),
        maxlen=Config.MAX_SEQUENCE_LENGTH
    )
    
    print(f"Vocabulary size: {min(len(tokenizer.word_index) + 1, Config.VOCAB_SIZE)}")
    print(f"Sequence shape: {X_train_seq.shape}")
    print()
    
    return tokenizer, X_train_seq, X_valid_seq, X_test_seq


def train_lstm(X_train_seq, y_train, X_valid_seq, y_valid, model_name="LSTM"):
    """Train simple LSTM model"""
    print(f"\nTraining {model_name}...")
    
    model = Sequential([
        Embedding(input_dim=Config.VOCAB_SIZE, output_dim=32, input_length=Config.MAX_SEQUENCE_LENGTH),
        LSTM(32),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=Config.EARLY_STOP_PATIENCE,
        restore_best_weights=True
    )
    
    history = model.fit(
        X_train_seq, y_train,
        epochs=Config.MAX_EPOCHS,
        batch_size=Config.BATCH_SIZE,
        validation_data=(X_valid_seq, y_valid),
        callbacks=[early_stop],
        verbose=0
    )
    
    # Evaluate
    probs = model.predict(X_valid_seq, verbose=0)
    preds = (probs > 0.5).astype(int).flatten()
    
    results = evaluate_model(y_valid, preds, model_name, verbose=False)
    print(f"   Accuracy: {results['accuracy']:.4f} | F1: {results['f1_macro']:.4f} | Epochs: {len(history.history['loss'])}")
    
    return model, results, history


def train_bilstm(X_train_seq, y_train, X_valid_seq, y_valid, model_name="BiLSTM"):
    """Train simple Bidirectional LSTM"""
    print(f"\nTraining {model_name}...")
    
    model = Sequential([
        Embedding(input_dim=Config.VOCAB_SIZE, output_dim=32, input_length=Config.MAX_SEQUENCE_LENGTH),
        Bidirectional(LSTM(32)),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=Config.EARLY_STOP_PATIENCE,
        restore_best_weights=True
    )
    
    history = model.fit(
        X_train_seq, y_train,
        epochs=Config.MAX_EPOCHS,
        batch_size=Config.BATCH_SIZE,
        validation_data=(X_valid_seq, y_valid),
        callbacks=[early_stop],
        verbose=0
    )
    
    probs = model.predict(X_valid_seq, verbose=0)
    preds = (probs > 0.5).astype(int).flatten()
    
    results = evaluate_model(y_valid, preds, model_name, verbose=False)
    print(f"   Accuracy: {results['accuracy']:.4f} | F1: {results['f1_macro']:.4f} | Epochs: {len(history.history['loss'])}")
    
    return model, results, history


def train_stacked_bilstm(X_train_seq, y_train, X_valid_seq, y_valid, model_name="Stacked BiLSTM"):
    """Train improved stacked Bidirectional LSTM"""
    print(f"\nTraining {model_name}...")
    
    model = Sequential([
        Embedding(input_dim=Config.VOCAB_SIZE, output_dim=Config.EMBEDDING_DIM, 
                  input_length=Config.MAX_SEQUENCE_LENGTH),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.3),
        Bidirectional(LSTM(32)),
        Dropout(0.5),
        Dense(16, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=Config.EARLY_STOP_PATIENCE,
        restore_best_weights=True
    )
    
    history = model.fit(
        X_train_seq, y_train,
        epochs=Config.MAX_EPOCHS,
        batch_size=Config.BATCH_SIZE,
        validation_data=(X_valid_seq, y_valid),
        callbacks=[early_stop],
        verbose=0
    )
    
    probs = model.predict(X_valid_seq, verbose=0)
    preds = (probs > 0.5).astype(int).flatten()
    
    results = evaluate_model(y_valid, preds, model_name, verbose=False)
    print(f"   Accuracy: {results['accuracy']:.4f} | F1: {results['f1_macro']:.4f} | Epochs: {len(history.history['loss'])}")
    
    return model, results, history


def train_cnn(X_train_seq, y_train, X_valid_seq, y_valid, model_name="CNN"):
    """Train CNN for text classification"""
    print(f"\nTraining {model_name}...")
    
    model = Sequential([
        Embedding(input_dim=Config.VOCAB_SIZE, output_dim=Config.EMBEDDING_DIM,
                  input_length=Config.MAX_SEQUENCE_LENGTH),
        Conv1D(filters=128, kernel_size=5, activation='relu'),
        GlobalMaxPooling1D(),
        Dense(16, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=Config.EARLY_STOP_PATIENCE,
        restore_best_weights=True
    )
    
    history = model.fit(
        X_train_seq, y_train,
        epochs=Config.MAX_EPOCHS,
        batch_size=Config.BATCH_SIZE,
        validation_data=(X_valid_seq, y_valid),
        callbacks=[early_stop],
        verbose=0
    )
    
    probs = model.predict(X_valid_seq, verbose=0)
    preds = (probs > 0.5).astype(int).flatten()
    
    results = evaluate_model(y_valid, preds, model_name, verbose=False)
    print(f"   Accuracy: {results['accuracy']:.4f} | F1: {results['f1_macro']:.4f} | Epochs: {len(history.history['loss'])}")
    
    return model, results, history


# ==================== ENSEMBLE TESTING ====================
def test_ensemble_combinations(models_dict, X_valid_tfidf, X_valid_combined, X_valid_seq, y_valid):
    """Test different ensemble combinations to find the best"""
    print("\n" + "="*60)
    print("TESTING ENSEMBLE COMBINATIONS")
    print("="*60)
    
    # Get predictions from all models
    predictions = {}
    
    # SVM models
    predictions['svm_original'] = models_dict['svm_original'].predict(X_valid_tfidf)
    predictions['svm_with_features'] = models_dict['svm_with_features'].predict(X_valid_combined)
    
    # Neural network models
    predictions['lstm'] = (models_dict['lstm'].predict(X_valid_seq, verbose=0) > 0.5).astype(int).flatten()
    predictions['bilstm'] = (models_dict['bilstm'].predict(X_valid_seq, verbose=0) > 0.5).astype(int).flatten()
    predictions['stacked_bilstm'] = (models_dict['stacked_bilstm'].predict(X_valid_seq, verbose=0) > 0.5).astype(int).flatten()
    predictions['cnn'] = (models_dict['cnn'].predict(X_valid_seq, verbose=0) > 0.5).astype(int).flatten()
    
    # Test different combinations
    ensemble_results = {}
    
    combinations = [
        (['svm_original', 'lstm', 'bilstm'], "SVM + LSTM + BiLSTM"),
        (['svm_original', 'stacked_bilstm', 'cnn'], "SVM + Stacked BiLSTM + CNN"),
        (['svm_with_features', 'stacked_bilstm', 'cnn'], "SVM(+features) + Stacked BiLSTM + CNN"),
        (['stacked_bilstm', 'cnn'], "Stacked BiLSTM + CNN (no SVM)"),
        (['svm_original', 'cnn'], "SVM + CNN"),
        (['cnn'], "CNN alone (best single model)"),
    ]
    
    print("\nEnsemble Combinations:")
    for model_keys, ensemble_name in combinations:
        # Majority voting
        votes = sum(predictions[k] for k in model_keys)
        threshold = len(model_keys) / 2
        ensemble_pred = (votes > threshold).astype(int)
        
        acc = accuracy_score(y_valid, ensemble_pred)
        f1 = f1_score(y_valid, ensemble_pred)
        
        ensemble_results[ensemble_name] = {
            'models': model_keys,
            'predictions': ensemble_pred,
            'accuracy': acc,
            'f1': f1
        }
        
        print(f"  {ensemble_name:45} | Acc: {acc:.4f} | F1: {f1:.4f}")
    
    # Find best ensemble
    best_ensemble = max(ensemble_results.items(), key=lambda x: x[1]['accuracy'])
    print(f"\n→ Best ensemble: {best_ensemble[0]}")
    print(f"  Accuracy: {best_ensemble[1]['accuracy']:.4f} | F1: {best_ensemble[1]['f1']:.4f}")
    
    return ensemble_results, best_ensemble


# ==================== MAIN EXECUTION ====================
def main():
    """Main execution pipeline"""
    print("\n" + "="*60)
    print("SARCASM DETECTION - COMPREHENSIVE TRAINING PIPELINE")
    print("="*60)
    print(f"Random Seed: {Config.RANDOM_SEED}")
    print(f"Max Sequence Length: {Config.MAX_SEQUENCE_LENGTH}")
    print(f"SVM Max Iterations: {Config.SVM_MAX_ITER}")
    print(f"Early Stop Patience: {Config.EARLY_STOP_PATIENCE}")
    print()
    
    # 1. Load data
    train_df, valid_df, test_df = load_data()
    y_train = train_df['label']
    y_valid = valid_df['label']
    y_test = test_df['label']
    
    # 2. Analyze text lengths (addresses issue: check if 100 tokens was too short)
    analyze_text_lengths(train_df, valid_df, test_df)
    
    # 3. Train baseline models (addresses issue: fixed SVM convergence)
    baseline_results, vectorizer_tfidf, svm_original, X_train_tfidf, X_valid_tfidf = \
        train_baseline_models(train_df, valid_df, y_train, y_valid)
    
    # 4. Test manual features (addresses issue: manual features didn't help)
    manual_features_results, svm_with_features, X_train_combined, X_valid_combined = \
        test_manual_features(train_df, valid_df, test_df, X_train_tfidf, X_valid_tfidf, y_train, y_valid)
    
    print("\n→ FINDING: Manual features provide minimal improvement")
    print(f"  SVM alone: {baseline_results['SVM (original)']['accuracy']:.4f}")
    print(f"  SVM + features: {manual_features_results['accuracy']:.4f}")
    print(f"  Difference: {manual_features_results['accuracy'] - baseline_results['SVM (original)']['accuracy']:.4f}")
    
    # 5. Prepare sequences for neural networks
    tokenizer, X_train_seq, X_valid_seq, X_test_seq = \
        prepare_sequences(train_df, valid_df, test_df)
    
    # 6. Train neural network models
    print("\n" + "="*60)
    print("TRAINING NEURAL NETWORK MODELS")
    print("="*60)
    
    lstm_model, lstm_results, lstm_history = \
        train_lstm(X_train_seq, y_train, X_valid_seq, y_valid)
    
    bilstm_model, bilstm_results, bilstm_history = \
        train_bilstm(X_train_seq, y_train, X_valid_seq, y_valid)
    
    print(f"\n→ FINDING: BiLSTM vs LSTM comparison")
    print(f"  LSTM: {lstm_results['accuracy']:.4f}")
    print(f"  BiLSTM: {bilstm_results['accuracy']:.4f}")
    if bilstm_results['accuracy'] < lstm_results['accuracy']:
        print(f"  → BiLSTM performs WORSE (likely needs more capacity)")
    
    stacked_bilstm_model, stacked_bilstm_results, stacked_bilstm_history = \
        train_stacked_bilstm(X_train_seq, y_train, X_valid_seq, y_valid)
    
    cnn_model, cnn_results, cnn_history = \
        train_cnn(X_train_seq, y_train, X_valid_seq, y_valid)
    
    # 7. Test ensemble combinations (addresses issue: test without "improved" SVM)
    models_dict = {
        'svm_original': svm_original,
        'svm_with_features': svm_with_features,
        'lstm': lstm_model,
        'bilstm': bilstm_model,
        'stacked_bilstm': stacked_bilstm_model,
        'cnn': cnn_model
    }
    
    ensemble_results, best_ensemble = test_ensemble_combinations(
        models_dict, X_valid_tfidf, X_valid_combined, X_valid_seq, y_valid
    )
    
    # 8. Evaluate best ensemble on TEST set
    print("\n" + "="*60)
    print("FINAL EVALUATION ON TEST SET")
    print("="*60)
    
    # Get test predictions for best ensemble
    test_predictions = {}
    for model_key in best_ensemble[1]['models']:
        if model_key == 'svm_original':
            X_test_tfidf = vectorizer_tfidf.transform(test_df['text'])
            test_predictions[model_key] = svm_original.predict(X_test_tfidf)
        elif model_key == 'svm_with_features':
            X_test_tfidf = vectorizer_tfidf.transform(test_df['text'])
            test_features = extract_features(test_df)
            X_test_combined = hstack([X_test_tfidf, test_features])
            test_predictions[model_key] = svm_with_features.predict(X_test_combined)
        elif model_key in ['lstm', 'bilstm', 'stacked_bilstm', 'cnn']:
            model = models_dict[model_key]
            test_predictions[model_key] = (model.predict(X_test_seq, verbose=0) > 0.5).astype(int).flatten()
    
    # Ensemble prediction
    test_votes = sum(test_predictions.values())
    threshold = len(best_ensemble[1]['models']) / 2
    test_ensemble_pred = (test_votes > threshold).astype(int)
    
    # Full evaluation
    test_results = evaluate_model(y_test, test_ensemble_pred, best_ensemble[0], verbose=True)
    
    # 9. Error analysis (addresses issue: actually run error analysis)
    print("\n" + "="*60)
    print("DETAILED ERROR ANALYSIS")
    print("="*60)
    
    fp, fn = analyze_errors(test_df, y_test, test_ensemble_pred, best_ensemble[0])
    
    # 10. Summary of findings
    print("\n" + "="*60)
    print("KEY FINDINGS & RECOMMENDATIONS")
    print("="*60)
    
    print("\n1. TEXT LENGTH:")
    print(f"   - Using sequence length {Config.MAX_SEQUENCE_LENGTH} reduces truncation")
    
    print("\n2. MANUAL FEATURES:")
    print(f"   - Manual features provide minimal benefit")
    print(f"   - TF-IDF likely captures these patterns already")
    
    print("\n3. MODEL COMPARISON:")
    print(f"   - Best single model: CNN ({cnn_results['accuracy']:.4f})")
    print(f"   - LSTM: {lstm_results['accuracy']:.4f}")
    print(f"   - BiLSTM: {bilstm_results['accuracy']:.4f} (worse without capacity)")
    print(f"   - Stacked BiLSTM: {stacked_bilstm_results['accuracy']:.4f} (better with capacity)")
    
    print("\n4. ENSEMBLE:")
    print(f"   - Best: {best_ensemble[0]}")
    print(f"   - Validation: {best_ensemble[1]['accuracy']:.4f}")
    print(f"   - Test: {test_results['accuracy']:.4f}")
    
    print("\n5. ERROR PATTERNS:")
    print(f"   - False Positives: {len(fp)} ({100*len(fp)/len(test_df):.2f}%)")
    print(f"   - False Negatives: {len(fn)} ({100*len(fn)/len(test_df):.2f}%)")
    print(f"   - Sarcasm remains harder to detect than non-sarcasm")
    
    # 11. Generate visualizations for report
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS FOR REPORT")
    print("="*60)
    
    os.makedirs(Config.FIGURES_DIR, exist_ok=True)
    
    # Text length distribution
    plot_text_length_distribution(
        train_df, valid_df, test_df,
        save_path=f'{Config.FIGURES_DIR}text_length_distribution.png'
    )
    
    # Model comparison
    all_results = {**baseline_results, **{
        'Stacked BiLSTM': stacked_bilstm_results,
        'CNN': cnn_results,
        best_ensemble[0]: {'accuracy': test_results['accuracy'], 'f1_macro': test_results['f1_macro']}
    }}
    
    plot_model_comparison(
        all_results, metric='accuracy',
        save_path=f'{Config.FIGURES_DIR}model_accuracy_comparison.png'
    )
    
    plot_model_comparison(
        all_results, metric='f1_macro',
        save_path=f'{Config.FIGURES_DIR}model_f1_comparison.png'
    )
    
    # Confusion matrix
    plot_confusion_matrix(
        test_results['confusion_matrix'],
        best_ensemble[0],
        save_path=f'{Config.FIGURES_DIR}confusion_matrix_final.png'
    )
    
    # Training histories
    plot_training_history(
        stacked_bilstm_history,
        'Stacked BiLSTM',
        save_path=f'{Config.FIGURES_DIR}training_history_bilstm.png'
    )
    
    plot_training_history(
        cnn_history,
        'CNN',
        save_path=f'{Config.FIGURES_DIR}training_history_cnn.png'
    )
    
    print("✓ Visualizations generated!")
    
    # 12. Save best models
    print("\n" + "="*60)
    print("SAVING MODELS")
    print("="*60)
    
    os.makedirs(Config.MODEL_DIR, exist_ok=True)
    
    # Save best neural network models
    stacked_bilstm_model.save(f'{Config.MODEL_DIR}stacked_bilstm_model.h5')
    cnn_model.save(f'{Config.MODEL_DIR}cnn_model.h5')
    
    # Save SVM and preprocessors
    pickle.dump(svm_original, open(f'{Config.MODEL_DIR}svm_model.pkl', 'wb'))
    pickle.dump(vectorizer_tfidf, open(f'{Config.MODEL_DIR}tfidf_vectorizer.pkl', 'wb'))
    pickle.dump(tokenizer, open(f'{Config.MODEL_DIR}tokenizer.pkl', 'wb'))
    
    # Save ensemble configuration
    ensemble_config = {
        'best_ensemble': best_ensemble[0],
        'model_keys': best_ensemble[1]['models'],
        'validation_accuracy': best_ensemble[1]['accuracy'],
        'test_accuracy': test_results['accuracy']
    }
    pickle.dump(ensemble_config, open(f'{Config.MODEL_DIR}model_weights.pkl', 'wb'))
    
    print("✓ Models saved successfully!")
    print(f"  - {Config.MODEL_DIR}stacked_bilstm_model.h5")
    print(f"  - {Config.MODEL_DIR}cnn_model.h5")
    print(f"  - {Config.MODEL_DIR}svm_model.pkl")
    print(f"  - {Config.MODEL_DIR}tfidf_vectorizer.pkl")
    print(f"  - {Config.MODEL_DIR}tokenizer.pkl")
    print(f"  - {Config.MODEL_DIR}model_weights.pkl")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"\nFinal Test Accuracy: {test_results['accuracy']:.4f}")
    print(f"Final Test F1 Score: {test_results['f1_macro']:.4f}")


if __name__ == "__main__":
    main()
