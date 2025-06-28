# 🎓 Prévention de l'Abandon Scolaire via le Data Mining

Ce projet vise à développer un système intelligent, interactif et opérationnel permettant de prédire le risque d’abandon scolaire en exploitant des techniques avancées de **Data Mining**, incluant la classification, le clustering, l’exploration de données, et les règles d’association. L’application est déployée via **Streamlit** pour offrir une interface accessible et interactive.

---

## 📌 Objectifs

- Identifier les profils d’étudiants à risque d’abandon
- Offrir une analyse exploratoire visuelle des comportements
- Segmenter les étudiants en groupes homogènes via **K-Means**
- Prédire le risque d’abandon grâce à **Random Forest**
- Découvrir des patterns comportementaux avec les **règles d’association**
- Proposer des **simulations individuelles** et recommandations
- Générer des **rapports personnalisés en PDF**

---

## 📊 Données utilisées

### 🧪 Données réelles — **Open University Learning Analytics Dataset (OULAD)**

| Fichier CSV              | Description                                    |
|--------------------------|------------------------------------------------|
| `studentInfo.csv`        | Données socio-démographiques                  |
| `assessments.csv`        | Informations sur les évaluations              |
| `studentAssessment.csv`  | Résultats des étudiants                       |

Variables dérivées après fusion :
- Score moyen, nombre d’évaluations, crédits étudiés
- Âge, sexe, région, niveau parental, handicap
- Données synthétiques : temps sur Moodle, participation aux forums, satisfaction
- Variable cible `abandon` : 1 (abandon) / 0 (réussite)

### 🧪 Données synthétiques
- 1000 observations simulées avec contrôle statistique
- Ratio d’abandon : 20%
- Variables simulées réalistes : âge, sexe, note moyenne, temps Moodle, etc.

---

## 🛠️ Architecture du Code

### 📂 Fichiers principaux

- `abandon_scolaire.py` : Code principal de l’application Streamlit
- `assessments.csv`, `studentAssessment.csv`, `studentInfo.csv` : Données réelles OULAD

### ⚙️ Technologies utilisées

- `Streamlit` : Interface utilisateur web
- `Pandas`, `Numpy` : Traitement des données
- `Matplotlib`, `Seaborn`, `Plotly` : Visualisations
- `scikit-learn` : Machine learning (KMeans, RandomForest)
- `mlxtend` : Règles d’association (Apriori)
- `ReportLab` : Génération des rapports PDF

---

## 🔬 Fonctionnalités de l'application

### 1. Analyse exploratoire
- Histogrammes dynamiques par statut d’abandon
- Matrice de corrélation
- Boxplots interactifs

### 2. Clustering K-Means
- Segmentation des étudiants en 3 groupes homogènes
- Visualisation interactive des clusters

### 3. Classification avec Random Forest
- Modèle supervisé entraîné pour prédire le risque d’abandon
- Affichage de la **probabilité d’abandon** en temps réel

### 4. Règles d'association (Apriori)
- Discrétisation des variables continues
- Extraction de patterns comportementaux significatifs
- Exemples :
  - Faible score → risque élevé
  - Homme + crédits faibles → vulnérabilité accrue

### 5. Simulation individuelle
- Interface avec sliders et menus déroulants
- Estimation du risque d’abandon pour un profil donné
- Génération automatique de rapport PDF avec résultats

---

## 💻 Interface utilisateur

- **Deux onglets principaux** :
  - `Données réelles (OULAD)` : Analyse sur données authentiques
  - `Données synthétiques` : Validation croisée sur données simulées
- Visualisations interactives (zoom, filtre)
- Simulation individuelle personnalisée
- Téléchargement du rapport PDF avec résultats

---

## 🧠 Techniques de Data Mining

| Technique           | Objectif                                      |
|---------------------|-----------------------------------------------|
| K-Means Clustering  | Segmentation des profils d’étudiants          |
| Random Forest       | Prédiction supervisée du risque d’abandon     |
| Apriori             | Détection de patterns comportementaux         |
| Analyse exploratoire| Compréhension des corrélations et distributions|

---

## 🚀 Installation et exécution

### ✅ Prérequis

Installer les dépendances :

```bash
pip install streamlit pandas numpy scikit-learn plotly seaborn matplotlib mlxtend reportlab

▶️ Lancer l’application :

```bash
streamlit run abandon_scolaire.py

📘 Licence
Ce projet est libre d’utilisation
