# 🧠 BBC News Classifier - Multi-Scale CNN

Un projet complet de classification d'articles de news BBC utilisant un réseau de neurones convolutif multi-échelles (Multi-Scale CNN) avec une architecture Dockerisée.

## 📋 Description

Ce projet implémente un classificateur de textes avancé capable de catégoriser automatiquement des articles de news BBC en 5 catégories :
- **Sport** 🏃‍♂️
- **Business** 💼
- **Politics** 🏛️
- **Technology** 💻
- **Entertainment** 🎬

L'architecture utilise un modèle **Multi-Scale CNN** qui capture des caractéristiques à différentes échelles de contexte grâce à des filtres parallèles de tailles 2, 3, 4 et 5.

## 🏗️ Architecture du Modèle

### Multi-Scale CNN
```
Input (sequence) → Embedding → Multi-Scale CNN → Dense → Softmax
                                   ↓
                            ┌─────────────────┐
                            │ Filter Size 2  │
                            ├─────────────────┤
                            │ Filter Size 3  │
                            ├─────────────────┤
                            │ Filter Size 4  │
                            ├─────────────────┤
                            │ Filter Size 5  │
                            └─────────────────┘
```

### Caractéristiques techniques
- **Embedding**: 100 dimensions, vocabulaire de 5000 mots
- **CNN parallèle**: 4 branches avec filtres de tailles 2, 3, 4, 5
- **Régularisation**: L1/L2, Dropout (0.4-0.5), Batch Normalization
- **Optimisation**: Adam optimizer avec Early Stopping et ReduceLROnPlateau
- **Performance**: ~95% accuracy sur validation

## 📁 Structure du Projet

```
projet-bbc-docker/
├── app.py                    # Application Flask avec interface web
├── train.py                  # Script d'entraînement du modèle
├── requirements.txt          # Dépendances Python
├── docker-compose.yml       # Configuration Docker Compose
├── Dockerfile.train         # Dockerfile pour l'entraînement
├── Dockerfile.deploy        # Dockerfile pour le déploiement
├── data/                     # Données BBC
│   ├── BBC News Train.csv   # Jeu d'entraînement
│   ├── BBC News Test.csv    # Jeu de test
│   └── BBC_Predictions_multiscale.csv  # Prédictions
├── best_multiscale.keras     # Modèle entraîné
└── README.md                 # Ce fichier
```

## 🚀 Démarrage Rapide

### Prérequis
- Docker et Docker Compose installés
- Git

### 1. Cloner le projet
```bash
git clone <repository-url>
cd projet-bbc-docker
```

### 2. Entraîner le modèle
```bash
docker-compose --profile train up --build
```

### 3. Déployer l'application
```bash
docker-compose --profile deploy up --build
```

### 4. Accéder à l'interface
Ouvrez votre navigateur sur `http://localhost:5000`

## 🐳 Dockerisation

Le projet utilise une architecture Docker à deux services :

### Service d'entraînement (`train`)
- **Image**: Python 3.9-slim
- **Fichier**: `Dockerfile.train`
- **Commande**: `python train.py`
- **Volume**: Montage du dossier `data/`

### Service de déploiement (`deploy`)
- **Image**: Python 3.9-slim
- **Fichier**: `Dockerfile.deploy`
- **Port**: 5000 (Flask)
- **Volume**: Montage du modèle et des données

## 💻 Utilisation Locale

### Installation
```bash
# Créer un environnement virtuel
python -m venv venv

# Activer l'environnement
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Installer les dépendances
pip install -r requirements.txt
```

### Entraînement
```bash
python train.py
```

### Lancement de l'application
```bash
python app.py
```

## 🎯 Performance

### Métriques du modèle
- **Accuracy**: ~93% (validation)
- **Architecture**: Multi-Scale CNN
- **Taille du modèle**: ~15MB

## 🌐 Interface Web

L'application Flask offre une interface moderne avec :
- **Saisie de texte**: Zone de texte pour entrer les articles
- **Analyse en temps réel**: Prédiction avec scores de confiance
- **Visualisation**: Barres de progression et cartes de probabilités
- **Exemples**: Textes prédéfinis pour chaque catégorie
- **Design responsive**: Compatible mobile/desktop

### Fonctionnalités
- Prédiction en une fraction de seconde
- Affichage des probabilités pour toutes les catégories
- Score de confiance avec indicateur visuel
- Exemples intégrés pour tester rapidement

## 📊 Dataset

### BBC News Dataset
- **Articles**: 2225 exemples d'entraînement
- **Catégories**: 5 classes équilibrées
- **Langue**: Anglais
- **Source**: BBC News

### Prétraitement
- Nettoyage du texte (minuscules, ponctuation)
- Tokenization avec vocabulaire de 5000 mots
- Séquences padées à 200 tokens
- Encodage des labels avec LabelEncoder

## 🔧 Configuration

### Hyperparamètres
```python
max_words = 5000      # Taille du vocabulaire
max_len = 200         # Longueur maximale des séquences
embedding_dim = 100   # Dimension des embeddings
epochs = 100          # Nombre d'époques maximum
batch_size = 32       # Taille des batchs
```

### Callbacks
- **EarlyStopping**: Patience de 15 époques
- **ReduceLROnPlateau**: Réduction du learning rate
- **ModelCheckpoint**: Sauvegarde du meilleur modèle

## 🛠️ Technologies Utilisées

### Backend
- **Python 3.9**: Langage principal
- **TensorFlow 2.15**: Framework Deep Learning
- **Keras**: API de haut niveau
- **Scikit-learn**: Prétraitement et métriques
- **Flask**: Application web

### Frontend
- **HTML5/CSS3**: Structure et style
- **JavaScript**: Interactivité
- **Font Awesome**: Icônes
- **Design responsive**: Mobile-first

### DevOps
- **Docker**: Conteneurisation
- **Docker Compose**: Orchestration
- **Git**: Version control




## 🐛 Dépannage

### Problèmes courants
1. **Modèle non trouvé**: Exécutez d'abord `python train.py`
2. **Port 5000 occupé**: Changez le port dans `docker-compose.yml`
3. **Memory error**: Réduisez `batch_size` dans `train.py`
4. **Docker build failed**: Vérifiez votre connexion internet

### Solutions
```bash
# Vérifier les logs Docker
docker-compose logs train
docker-compose logs deploy

# Reconstruire les images
docker-compose build --no-cache

# Nettoyer les volumes
docker-compose down -v
```


## 👥 Contributeurs
 - **DERBANI Salwa**  - 🔗 [GitHub](https://github.com/sader04)
 - **KOUDIA Selma**  - 🔗 [GitHub](https://github.com/selmakoudia03)

---

