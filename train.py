import re
from typing import Dict, List, Optional, Tuple, Union
import time
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

# Fixer les seeds pour la reproductibilité
np.random.seed(42)
tf.random.set_seed(42)

from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    LearningRateScheduler
)
from tensorflow.keras.layers import (
    BatchNormalization,
    Concatenate,
    Conv1D,
    Dense,
    Dropout,
    Embedding,
    GlobalMaxPooling1D,
    Input,
    SpatialDropout1D,
)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.regularizers import l1_l2
import re
import warnings
warnings.filterwarnings('ignore')

# ==================== MULTI-SCALE CNN ====================

def build_multiscale_cnn(max_len: int, max_words: int, num_classes: int) -> Model:
    """
    Build a multi-scale CNN with parallel filter branches.
    
    Captures different levels of context using various kernel sizes.
    
    Args:
        max_len: Maximum sequence length
        max_words: Maximum vocabulary size
        num_classes: Number of output classes
        
    Returns:
        Compiled Keras Functional model
    """
    print("\n🏗️  Multi-Scale CNN...")
    
    input_layer = Input(shape=(max_len,))
    
    # Embedding partagé
    embedding = Embedding(
        max_words, 100, 
        input_length=max_len,
        embeddings_regularizer=l1_l2(1e-5, 1e-4)
    )(input_layer)
    embedding = SpatialDropout1D(0.2)(embedding)
    
    # Branches parallèles avec différentes tailles de filtres
    conv_blocks = []
    for kernel_size in [2, 3, 4, 5]:
        conv = Conv1D(
            32, kernel_size, 
            activation='relu', 
            padding='valid',
            kernel_regularizer=l1_l2(1e-5, 1e-4)
        )(embedding)
        conv = GlobalMaxPooling1D()(conv)
        conv_blocks.append(conv)
    
    # Concatener toutes les branches
    merged = Concatenate()(conv_blocks)
    
    # Dense layers
    x = Dense(64, activation='relu', kernel_regularizer=l1_l2(1e-5, 1e-4))(merged)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    x = Dense(32, activation='relu', kernel_regularizer=l1_l2(1e-5, 1e-4))(x)
    x = Dropout(0.4)(x)
    
    output = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=input_layer, outputs=output)
    
    return model


# ==================== CLASSE PRINCIPALE ====================

class BBCMultiScaleCNN:
    """Classification BBC avec Multi-Scale CNN"""
    
    def __init__(self):
        self.max_words = 5000
        self.max_len = 200
        
        self.tokenizer = None
        self.label_encoder = None
        self.model = None
        
    def preprocess_text(self, text):
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r'[^a-z\s]', ' ', text)
        text = ' '.join([word for word in text.split() if len(word) >= 2])
        return text.strip()
    
    def load_and_prepare_data(self, train_path, test_path):
        """Charge et prépare les données"""
        print("\n📂 Chargement des données BBC...")
        
        self.train_df = pd.read_csv(train_path)
        self.test_df = pd.read_csv(test_path)
        
        self.train_df.columns = self.train_df.columns.str.strip()
        self.test_df.columns = self.test_df.columns.str.strip()
        
        #Prétraitement
        print("🧹 Prétraitement...")
        self.train_df['text_clean'] = self.train_df['Text'].apply(self.preprocess_text)
        self.test_df['text_clean'] = self.test_df['Text'].apply(self.preprocess_text)
        
        #Labels
        self.label_encoder = LabelEncoder()
        self.train_df['label'] = self.label_encoder.fit_transform(self.train_df['Category'])
        self.num_classes = len(self.label_encoder.classes_)
        
        #Tokenization
        print("🔤 Tokenization...")
        self.tokenizer = Tokenizer(num_words=self.max_words, oov_token='<OOV>')
        self.tokenizer.fit_on_texts(self.train_df['text_clean'])
        
        X_train_seq = self.tokenizer.texts_to_sequences(self.train_df['text_clean'])
        X_test_seq = self.tokenizer.texts_to_sequences(self.test_df['text_clean'])
        
        self.X_train_full = pad_sequences(X_train_seq, maxlen=self.max_len, padding='post')
        self.X_test_final = pad_sequences(X_test_seq, maxlen=self.max_len, padding='post')
        
        self.y_train_full = keras.utils.to_categorical(self.train_df['label'], self.num_classes)
        
        #Split
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.X_train_full, 
            self.y_train_full,
            test_size=0.20,
            random_state=42,
            stratify=self.train_df['label']
        )
        
        #Class weights
        class_weights_array = compute_class_weight(
            'balanced',
            classes=np.unique(self.train_df['label']),
            y=self.train_df['label']
        )
        self.class_weights = dict(enumerate(class_weights_array))
        
        print(f"\n✅ Préparé:")
        print(f"   Train: {self.X_train.shape}")
        print(f"   Val: {self.X_val.shape}")
        print(f"   Test: {self.X_test_final.shape}")
    
    def build_and_train(self, epochs=100):
        """Construit et entraîne le modèle Multi-Scale CNN"""
        print("\n🎯 Stratégie: MULTI-SCALE CNN")
        self.model = build_multiscale_cnn(self.max_len, self.max_words, self.num_classes)
        self._train_single_model(epochs)
    
    def _train_single_model(self, epochs):
        """Entraîne un modèle unique"""
        start_time = time.time()
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Afficher le résumé du modèle
        print("\n📋 Architecture du modèle:")
        self.model.summary()
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, verbose=1),
            ModelCheckpoint('best_multiscale.keras', monitor='val_accuracy', save_best_only=True)
        ]
        
        print(f"\n🚀 Entraînement...")
        
        training_start = time.time()
        history = self.model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_val, self.y_val),
            epochs=epochs,
            batch_size=32,
            callbacks=callbacks,
            class_weight=self.class_weights,
            verbose=1
        )
        training_time = time.time() - training_start
        
        #Résultats
        final_train = history.history['accuracy'][-1]
        final_val = history.history['val_accuracy'][-1]
        gap = final_train - final_val
        
        total_time = time.time() - start_time
        
        print(f"\n📊 Résultats:")
        print(f"   Train: {final_train*100:.2f}%")
        print(f"   Val: {final_val*100:.2f}%")
        print(f"   Gap: {gap*100:.2f}%")
        print(f"\n⏱️  Temps:")
        print(f"   Entraînement: {training_time:.2f}s")
        print(f"   Total: {total_time:.2f}s")
    
    def evaluate(self):
        """Evaluation"""
        print("\n📊 Évaluation...")
        
        y_pred_proba = self.model.predict(self.X_val, verbose=0)
        
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_true = np.argmax(self.y_val, axis=1)
        
        accuracy = accuracy_score(y_true, y_pred)
        print(f"\n🎯 Val Accuracy: {accuracy*100:.2f}%")
        
        print("\n📈 Rapport:")
        print(classification_report(
            y_true, y_pred,
            target_names=self.label_encoder.classes_,
            digits=4
        ))
        
        return accuracy
    
    def predict_test(self):
        """Prédictions sur test"""
        print("\n🔮 Prédiction TEST...")
        
        y_pred_proba = self.model.predict(self.X_test_final, verbose=0)
        
        y_pred = np.argmax(y_pred_proba, axis=1)
        predicted = self.label_encoder.inverse_transform(y_pred)
        
        print("\n📊 Distribution:")
        for cat in self.label_encoder.classes_:
            count = np.sum(predicted == cat)
            pct = count / len(predicted) * 100
            print(f"   {cat:15s}: {count:3d} ({pct:5.1f}%)")
        
        return predicted
    
    def create_submission(self, predictions):
        path = 'data/BBC_Predictions_multiscale.csv'
        
        submission = pd.DataFrame({
            'ArticleId': self.test_df['ArticleId'],
            'Category': predictions
        })
        
        submission.to_csv(path, index=False)
        print(f"\n✅ Fichier: {path}")
        
        return submission


# ==================== MAIN ====================

def main():
    """Entraînement du modèle Multi-Scale CNN"""
    
    print("="*80)
    print("🚀 BBC NEWS - MULTI-SCALE CNN")
    print("="*80)
    
    # Informations système
    print(f"\n🖥️  Environnement:")
    print(f"   TensorFlow: {tf.__version__}")
    print(f"   GPU disponible: {len(tf.config.list_physical_devices('GPU')) > 0}")
    print(f"   CPU cores: {os.cpu_count() if 'os' in globals() else 'Non détecté'}")
    
    start_total = time.time()
    
    classifier = BBCMultiScaleCNN()
    
    classifier.load_and_prepare_data(
        train_path='data/BBC News Train.csv',
        test_path='data/BBC News Test.csv'
    )
    
    classifier.build_and_train(epochs=100)
    
    accuracy = classifier.evaluate()
    predictions = classifier.predict_test()
    classifier.create_submission(predictions)
    
    total_time = time.time() - start_total
    
    print(f"\n✅ TERMINÉ! Val Accuracy: {accuracy*100:.2f}%")
    print(f"\n⏱️  Temps total d'exécution: {total_time:.2f}s")
    
    return classifier


if __name__ == "__main__":
    classifier = main()