from flask import Flask, request, jsonify
import numpy as np
import re
from datetime import datetime
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os
import pandas as pd
from tensorflow.keras.models import load_model

app = Flask(__name__)

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BBC News Classifier</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%);
            min-height: 100vh;
            padding: 20px;
            color: #333;
            position: relative;
            overflow-x: hidden;
        }
        
        body::before {
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-image: 
                radial-gradient(circle at 20% 50%, rgba(99, 102, 241, 0.1) 0%, transparent 50%),
                radial-gradient(circle at 80% 20%, rgba(139, 92, 246, 0.1) 0%, transparent 50%),
                radial-gradient(circle at 40% 80%, rgba(59, 130, 246, 0.1) 0%, transparent 50%);
            z-index: -1;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(20px);
            border-radius: 24px;
            box-shadow: 
                0 32px 64px rgba(0, 0, 0, 0.2),
                0 0 0 1px rgba(255, 255, 255, 0.1);
            padding: 50px;
            position: relative;
            overflow: hidden;
        }
        
        .container::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 6px;
            background: linear-gradient(90deg, 
                #3b82f6 0%, 
                #8b5cf6 25%, 
                #6366f1 50%, 
                #8b5cf6 75%, 
                #3b82f6 100%);
            background-size: 200% 100%;
            animation: gradient-shift 3s ease-in-out infinite;
        }
        
        @keyframes gradient-shift {
            0%, 100% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
        }
        
        h1 {
            color: #1e293b;
            text-align: center;
            margin-bottom: 15px;
            font-size: 3.2em;
            font-weight: 800;
            letter-spacing: -1px;
            background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 40px;
            font-size: 1.1em;
            font-weight: 300;
        }
        
        .input-section {
            margin-bottom: 35px;
        }
        
        label {
            display: block;
            color: #1a1a1a;
            font-weight: 600;
            margin-bottom: 12px;
            font-size: 1.1em;
            letter-spacing: 0.5px;
        }
        
        textarea {
            width: 100%;
            padding: 18px;
            border: 2px solid #e0e0e0;
            border-radius: 12px;
            font-size: 16px;
            font-family: inherit;
            resize: vertical;
            min-height: 160px;
            transition: all 0.3s;
            background: #fafafa;
            line-height: 1.6;
        }
        
        textarea:focus {
            outline: none;
            border-color: #6366f1;
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
        }
        
        .button-group {
            display: flex;
            gap: 12px;
            margin-top: 20px;
        }
        
        button {
            flex: 1;
            padding: 16px 25px;
            border: none;
            border-radius: 10px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
            letter-spacing: 0.5px;
        }
        
        .btn-predict {
            background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
            color: white;
            box-shadow: 0 6px 20px rgba(59, 130, 246, 0.3);
        }
        
        .btn-predict:hover {
            transform: translateY(-3px);
            box-shadow: 0 6px 20px rgba(99, 102, 241, 0.4);
        }
        
        .btn-predict:active {
            transform: translateY(0);
        }
        
        .btn-clear {
            background: #f5f5f5;
            color: #555;
            border: 1px solid #ddd;
        }
        
        .btn-clear:hover {
            background: #e8e8e8;
            transform: translateY(-2px);
        }
        
        .btn-example {
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            color: white;
            box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3);
        }
        
        .btn-example:hover {
            transform: translateY(-3px);
            box-shadow: 0 6px 20px rgba(16, 185, 129, 0.4);
        }
        
        .loading {
            display: none;
            text-align: center;
            margin: 30px 0;
        }
        
        .spinner {
            border: 4px solid rgba(99, 102, 241, 0.1);
            border-top: 4px solid #6366f1;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .results {
            display: none;
            margin-top: 40px;
            padding: 30px;
            background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
            border-radius: 18px;
            border: 1px solid #e8e8e8;
            box-shadow: 0 5px 25px rgba(0,0,0,0.05);
        }
        
        .result-header {
            text-align: center;
            margin-bottom: 30px;
            padding-bottom: 25px;
            border-bottom: 2px solid #f0f0f0;
        }
        
        .predicted-category {
            display: inline-block;
            padding: 18px 45px;
            background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
            color: white;
            border-radius: 60px;
            font-size: 1.8em;
            font-weight: 700;
            margin-bottom: 15px;
            letter-spacing: 1px;
            box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);
        }
        
        .confidence {
            color: #555;
            font-size: 1.3em;
            font-weight: 500;
            margin-top: 10px;
        }
        
        .confidence-bar {
            width: 100%;
            height: 35px;
            background: #f0f0f0;
            border-radius: 18px;
            overflow: hidden;
            margin: 20px 0;
            border: 2px solid #e8e8e8;
        }
        
        .confidence-fill {
            height: 100%;
            background: linear-gradient(90deg, #3b82f6 0%, #8b5cf6 100%);
            transition: width 0.6s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: 700;
            font-size: 1.1em;
            letter-spacing: 0.5px;
        }
        
        .predictions-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 20px;
            margin-top: 30px;
        }
        
        .prediction-card {
            background: white;
            padding: 25px 20px;
            border-radius: 15px;
            box-shadow: 0 3px 15px rgba(0,0,0,0.08);
            transition: all 0.3s;
            text-align: center;
            border: 2px solid transparent;
        }
        
        .prediction-card:hover {
            transform: translateY(-8px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.15);
            border-color: #f0f0f0;
        }
        
        .category-name {
            font-weight: 700;
            color: #1a1a1a;
            margin-bottom: 12px;
            font-size: 1.2em;
            text-transform: capitalize;
            letter-spacing: 0.5px;
        }
        
        .category-prob {
            font-size: 2.2em;
            font-weight: 800;
            color: #3b82f6;
            margin-bottom: 12px;
            line-height: 1;
        }
        
        .keywords-found {
            font-size: 0.85em;
            color: #888;
            margin-top: 10px;
            min-height: 30px;
        }
        
        .examples {
            margin-top: 35px;
            padding: 25px;
            background: linear-gradient(135deg, #f0fdf4 0%, #f7fee7 100%);
            border-radius: 15px;
            border: 1px solid #bbf7d0;
        }
        
        .examples h3 {
            color: #16a34a;
            margin-bottom: 20px;
            font-size: 1.4em;
            font-weight: 700;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .example-item {
            padding: 15px;
            background: white;
            border-radius: 10px;
            margin-bottom: 12px;
            cursor: pointer;
            transition: all 0.3s;
            border: 1px solid #f0f0f0;
        }
        
        .example-item:hover {
            background: #f0fdf4;
            border-color: #bbf7d0;
            transform: translateX(5px);
        }
        
        .example-label {
            font-weight: 700;
            color: #1a1a1a;
            margin-bottom: 8px;
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 1.05em;
        }
        
        .example-text {
            color: #666;
            font-size: 0.95em;
            line-height: 1.6;
        }
        
        /* Responsive adjustments */
        @media (max-width: 768px) {
            .container {
                padding: 30px 20px;
            }
            
            h1 {
                font-size: 2.2em;
            }
            
            .button-group {
                flex-direction: column;
            }
            
            button {
                width: 100%;
            }
            
            .predictions-grid {
                grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🧠 BBC News Classifier</h1>
        <p class="subtitle">Classification avec Multi-Scale CNN Deep Learning</p>
        
        <div class="input-section">
            <label for="text-input">Entrez votre article (en anglais) :</label>
            <textarea id="text-input" placeholder="Tapez ou collez votre texte ici...
Exemple : The football team won their final match with a brilliant goal in the last minute..."></textarea>
            
            <div class="button-group">
                <button class="btn-predict" onclick="predict()">
                    <i class="fas fa-network-wired"></i> Analyser 
                </button>
                <button class="btn-clear" onclick="clearAll()">
                    <i class="fas fa-trash"></i> Effacer
                </button>
                <button class="btn-example" onclick="showExamples()">
                    <i class="fas fa-lightbulb"></i> Exemples
                </button>
            </div>
        </div>
        
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p style="margin-top: 15px; color: #666; font-size: 1.1em;">Analyse en cours...</p>
        </div>
        
        <div class="results" id="results">
            <div class="result-header">
                <div class="predicted-category" id="predicted-category"></div>
                <div class="confidence" id="confidence-text"></div>
                <div class="confidence-bar">
                    <div class="confidence-fill" id="confidence-bar"></div>
                </div>
            </div>
            
            <div class="predictions-grid" id="predictions-grid"></div>
        </div>
        
        <div class="examples" id="examples" style="display: none;">
            <h3><i class="fas fa-file-alt"></i> Exemples par catégorie</h3>
            
            <div class="example-item" onclick="useExample('sport')">
                <div class="example-label"><i class="fas fa-running"></i> 🏃 Sport</div>
                <div class="example-text">The football team won their final match with a brilliant goal in the last minute. The striker scored the winning goal after a perfect pass from the midfielder.</div>
            </div>
            
            <div class="example-item" onclick="useExample('business')">
                <div class="example-label"><i class="fas fa-briefcase"></i> 💼 Business</div>
                <div class="example-text">The company announced record profits in the fourth quarter with shares rising 15% on the stock market. The CEO said the growth was driven by strong sales in emerging markets.</div>
            </div>
            
            <div class="example-item" onclick="useExample('politics')">
                <div class="example-label"><i class="fas fa-landmark"></i> 🏛️ Politics</div>
                <div class="example-text">The Prime Minister announced new government policies on education. The Labour party criticized the bill saying it doesn't go far enough to help students.</div>
            </div>
            
            <div class="example-item" onclick="useExample('tech')">
                <div class="example-label"><i class="fas fa-laptop-code"></i> 💻 Tech</div>
                <div class="example-text">The new smartphone features advanced AI technology and faster processing power. The device includes improved camera software and longer battery life for mobile users.</div>
            </div>
            
            <div class="example-item" onclick="useExample('entertainment')">
                <div class="example-label"><i class="fas fa-film"></i> 🎬 Entertainment</div>
                <div class="example-text">The actor won the best performance award at the Hollywood film festival. The movie has been a box office hit and received critical acclaim from audiences worldwide.</div>
            </div>
        </div>
    </div>
    
    <script>
        const examples = {
            'sport': 'The football team won their final match with a brilliant goal in the last minute. The striker scored the winning goal after a perfect pass from the midfielder.',
            'business': 'The company announced record profits in the fourth quarter with shares rising 15% on the stock market. The CEO said the growth was driven by strong sales in emerging markets.',
            'politics': 'The Prime Minister announced new government policies on education. The Labour party criticized the bill saying it doesn\\'t go far enough to help students.',
            'tech': 'The new smartphone features advanced AI technology and faster processing power. The device includes improved camera software and longer battery life for mobile users.',
            'entertainment': 'The actor won the best performance award at the Hollywood film festival. The movie has been a box office hit and received critical acclaim from audiences worldwide.'
        };
        
        function showExamples() {
            const examplesDiv = document.getElementById('examples');
            examplesDiv.style.display = examplesDiv.style.display === 'none' ? 'block' : 'none';
        }
        
        function useExample(category) {
            document.getElementById('text-input').value = examples[category];
        }
        
        function clearAll() {
            document.getElementById('text-input').value = '';
            document.getElementById('results').style.display = 'none';
            document.getElementById('examples').style.display = 'none';
        }
        
        async function predict() {
            const text = document.getElementById('text-input').value.trim();
            
            if (!text) {
                alert('⚠️ Veuillez entrer un texte à analyser');
                return;
            }
            
            document.getElementById('loading').style.display = 'block';
            document.getElementById('results').style.display = 'none';
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ text: text })
                });
                
                const data = await response.json();
                
                if (data.success) {
                    displayResults(data);
                } else {
                    alert('❌ Erreur: ' + data.error);
                }
            } catch (error) {
                alert('❌ Erreur de connexion: ' + error.message);
            } finally {
                document.getElementById('loading').style.display = 'none';
            }
        }
        
        function displayResults(data) {
            document.getElementById('predicted-category').textContent = data.predicted_category.toUpperCase();
            
            document.getElementById('confidence-text').textContent = `Confiance: ${data.confidence}%`;
            document.getElementById('confidence-bar').style.width = data.confidence + '%';
            document.getElementById('confidence-bar').textContent = data.confidence + '%';
            
            const bar = document.getElementById('confidence-bar');
            if (data.confidence >= 80) {
                bar.style.background = 'linear-gradient(90deg, #3b82f6 0%, #8b5cf6 100%)';
            } else if (data.confidence >= 60) {
                bar.style.background = 'linear-gradient(90deg, #10b981 0%, #059669 100%)';
            } else {
                bar.style.background = 'linear-gradient(90deg, #f59e0b 0%, #d97706 100%)';
            }
            
            const grid = document.getElementById('predictions-grid');
            grid.innerHTML = '';
            
            for (const [category, info] of Object.entries(data.all_predictions)) {
                const card = document.createElement('div');
                card.className = 'prediction-card';
                
                card.innerHTML = `
                    <div class="category-name">${category}</div>
                    <div class="category-prob">${info.probability}%</div>
                    <div class="keywords-found">
                        <span style="color: #3b82f6;">Multi-Scale CNN</span>
                    </div>
                `;
                
                grid.appendChild(card);
            }
            
            document.getElementById('results').style.display = 'block';
            document.getElementById('results').scrollIntoView({ behavior: 'smooth' });
        }
        
        document.getElementById('text-input').addEventListener('keydown', function(e) {
            if (e.ctrlKey && e.key === 'Enter') {
                predict();
            }
        });
    </script>
</body>
</html>"""

# Configuration du modèle
MODEL_PATH = 'best_multiscale.keras'
TOKENIZER_PATH = 'tokenizer_random.pkl'
LABEL_ENCODER_PATH = 'label_encoder_random.pkl'

# Variables globales pour le modèle
model = None
tokenizer = None
label_encoder = None
max_words = 5000
max_len = 200

def load_model():
    """Charge le modèle multiscale"""
    global model, tokenizer, label_encoder
    
    try:
        print("🔄 Chargement du modèle multiscale...") 
        
        # Importer la classe depuis train.py
        from train import BBCMultiScaleCNN
        
        # Créer le classifier multiscale
        classifier = BBCMultiScaleCNN()
        
        # Charger les données pour préparer le modèle
        classifier.load_and_prepare_data(
            train_path='data/BBC News Train.csv',
            test_path='data/BBC News Test.csv'
        )
        
        # Charger le modèle entraîné avec compatibilité améliorée
        if os.path.exists(MODEL_PATH):
            try:
                # Charger avec compile=False pour éviter les problèmes de compilation
                model = keras.models.load_model(MODEL_PATH, compile=False)
                classifier.model = model
                print("✅ Modèle multiscale chargé avec succès")
                print(f"📊 Input shape attendu: {model.input_shape}")
            except Exception as e:
                print(f"❌ Erreur lors du chargement du modèle: {str(e)}")
                print("💡 Tentative de reconstruction du modèle...")
                
                # Reconstruire le modèle avec la même architecture
                try:
                    from train import build_multiscale_cnn
                    model = build_multiscale_cnn(classifier.max_len, classifier.max_words, classifier.num_classes)
                    classifier.model = model
                    print("✅ Modèle multiscale reconstruit (non entraîné)")
                    print("⚠️ Le modèle fonctionnera mais avec des prédictions aléatoires")
                except Exception as rebuild_error:
                    print(f"❌ Erreur de reconstruction: {rebuild_error}")
                    return False
        else:
            print(f"❌ Fichier modèle non trouvé: {MODEL_PATH}")
            print("💡 Assurez-vous d'avoir entraîné le modèle avec: python train.py")
            return False
        
        # Utiliser le tokenizer et label encoder du classifier
        tokenizer = classifier.tokenizer
        label_encoder = classifier.label_encoder
        print("✅ Tokenizer et label encoder multiscale chargés avec succès")
        
        print(f"📊 Modèle multiscale prêt pour {len(label_encoder.classes_)} catégories")
        print(f"📊 Classes: {list(label_encoder.classes_)}")
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors du chargement du modèle: {str(e)}")
        return False

def preprocess_text(text):
    """Nettoie le texte (même prétraitement que train.py)"""
    if not text:
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = ' '.join([word for word in text.split() if len(word) >= 2])
    return text.strip()

def predict_category(text):
    """Prédit la catégorie d'un texte avec le modèle multiscale"""
    if not text or not text.strip():
        return {
            'error': 'Le texte est vide',
            'success': False
        }
    
    try:
        # Prétraitement
        text_clean = preprocess_text(text)
        
        # Tokenization et padding
        seq = tokenizer.texts_to_sequences([text_clean])
        padded = pad_sequences(seq, maxlen=max_len, padding='post')
        
        # Prédiction avec le modèle multiscale
        proba = model.predict(padded, verbose=0)[0]
        predicted_idx = np.argmax(proba)
        predicted_category = label_encoder.inverse_transform([predicted_idx])[0]
        confidence = float(proba[predicted_idx])
        
        # Toutes les probabilités
        all_probabilities = {}
        for idx, cat in enumerate(label_encoder.classes_):
            all_probabilities[cat] = {
                'probability': round(float(proba[idx]) * 100, 2)
            }
        
        # Tri par probabilité
        sorted_predictions = sorted(
            all_probabilities.items(),
            key=lambda x: x[1]['probability'],
            reverse=True
        )
        
        return {
            'success': True,
            'predicted_category': predicted_category,
            'confidence': round(confidence * 100, 2),
            'text_length': len(text.split()),
            'cleaned_text_length': len(text_clean.split()),
            'all_predictions': dict(sorted_predictions),
            'top_3': [
                {
                    'category': cat,
                    'probability': data['probability']
                }
                for cat, data in sorted_predictions[:3]
            ],
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'mode': 'multiscale',
            'model_info': {
                'type': 'Multi-Scale CNN',
                'architecture': 'Embedding -> Multi-Scale CNN (2,3,4,5) -> Dense',
                'features': 'CNN multi-échelles pour différents niveaux de contexte',
                'reliability': 'Haute performance sur données BBC',
                'note': 'Modèle multiscale identique à test_real_samples.py'
            }
        }
    
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

# ==================== ROUTES ====================

@app.route('/')
def home():
    """Page d'accueil"""
    return HTML_TEMPLATE

@app.route('/predict', methods=['POST'])
def predict():
    """Route de prédiction"""
    text = request.json['text']
    result = predict_category(text)
    return jsonify(result)

@app.route('/model_info')
def model_info():
    """Informations détaillées sur le modèle"""
    if model and tokenizer and label_encoder:
        return jsonify({
            'model_loaded': True,
            'model_type': 'Multi-Scale CNN (standalone)',
            'model_path': MODEL_PATH,
            'categories': label_encoder.classes_.tolist(),
            'max_words': max_words,
            'max_len': max_len,
            'architecture': {
                'input': f'Séquence de {max_len} tokens',
                'embedding': f'Couche Embedding ({max_words} vocabulaire -> 100 dimensions)',
                'cnn_multiscale': 'CNN parallèle avec filtres de taille 2,3,4,5',
                'merge': 'Concaténation des branches CNN',
                'classifier': 'Dense layers avec régularisation',
                'output': f'Softmax sur {len(label_encoder.classes_)} catégories'
            },
            'features': [
                'Multi-Scale CNN pour différents niveaux de contexte',
                'Filtres parallèles (2,3,4,5) pour capture locale et globale',
                'Régularisation L1/L2 pour éviter overfitting',
                'Batch Normalization et Dropout',
                'Early Stopping et ReduceLROnPlateau',
                'Application Flask',
                'Modèle multiscale éprouvé'
            ],
            'performance': {
                'val_accuracy': '~95% (estimé)',
                'train_accuracy': '~99% (estimé)',
                'reliability': 'Haute performance - modèle multiscale éprouvé',
                'status': 'Multi-Scale CNN - Application Standalone'
            },
        })
    else:
        return jsonify({
            'model_loaded': False,
            'error': 'Modèle multiscale non chargé'
        })

#Initialisation au démarrage
if __name__ == '__main__':
    if load_model():
        print("🚀 Application démarrée - Multi-Scale CNN")
        print("🌐 Accédez à http://localhost:5000 pour utiliser l'interface")
        print("📊 Pour voir les infos du modèle: http://localhost:5000/model_info")
        app.run(debug=True, port=5000)
    else:
        print("❌ Impossible de démarrer l'application sans modèle")
        print("💡 Assurez-vous d'avoir entraîné le modèle avec: python train.py")
