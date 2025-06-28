# Importation des bibliothèques nécessaires pour le développement de l'application Streamlit et l'analyse de données
import streamlit as st  # Interface web interactive
import pandas as pd  # Manipulation de données avec DataFrames
import numpy as np  # Calculs numériques et génération de données aléatoires
import seaborn as sns  # Visualisation statistique (ex: heatmap)
import matplotlib.pyplot as plt  # Graphiques classiques
import plotly.express as px  # Visualisations interactives
from sklearn.cluster import KMeans  # Algorithme de clustering non supervisé
from sklearn.ensemble import RandomForestClassifier  # Classifieur supervisé (arbres aléatoires)
from sklearn.preprocessing import StandardScaler  # Normalisation des données
from mlxtend.frequent_patterns import apriori, association_rules  # Algorithmes de règles d'association
import io  # Manipulation des fichiers en mémoire (ex: PDF)
from reportlab.pdfgen import canvas  # Génération de documents PDF
from reportlab.lib.pagesizes import letter  # Format de page pour PDF

# Configuration de la page Streamlit : titre de l'onglet du navigateur et mise en page large
st.set_page_config(page_title="Prévention de l'abandon scolaire", layout="wide")

# Fonction pour charger et prétraiter les données OULAD
@st.cache_data  # Mise en cache pour éviter de recharger inutilement les données
def load_oulad_data():
    assessments = pd.read_csv("assessments.csv")  # Chargement du fichier des évaluations
    student_assessment = pd.read_csv("studentAssessment.csv")  # Évaluations faites par les étudiants
    student_info = pd.read_csv("studentInfo.csv")  # Informations générales sur les étudiants

    # Fusion des fichiers CSV sur les clés communes
    merged = student_assessment.merge(assessments, on='id_assessment', how='left')  # Fusion sur les ID d'évaluation
    merged = merged.merge(student_info, on='id_student', how='left')  # Fusion avec les infos étudiants

    # Agrégation des données par étudiant
    df_grouped = merged.groupby('id_student').agg({
        'score': 'mean',  # Moyenne des scores
        'date_submitted': 'count',  # Nombre d'évaluations soumises
        'studied_credits': 'mean',  # Moyenne des crédits étudiés
        'age_band': 'first',  # Première tranche d'âge rencontrée
        'gender': 'first',  # Sexe
        'region': 'first',  # Région
        'highest_education': 'first',  # Niveau d'étude parental
        'disability': 'first',  # Handicap ou non
        'final_result': 'first'  # Résultat final
    }).reset_index()

    # Renommage des colonnes pour plus de clarté
    df_grouped.rename(columns={
        'score': 'score_moyen',
        'date_submitted': 'nb_evaluations',
        'studied_credits': 'credits_etudies',
        'age_band': 'age',
        'gender': 'sexe',
        'highest_education': 'niveau_parental',
        'disability': 'handicap',
        'final_result': 'abandon'
    }, inplace=True)

    # Ajout de colonnes synthétiques simulées
    df_grouped['temps_moodle'] = np.random.uniform(0, 20, size=len(df_grouped))  # Temps passé sur la plateforme Moodle
    df_grouped['participation_forum'] = np.random.randint(0, 50, size=len(df_grouped))  # Messages sur le forum
    df_grouped['satisfaction'] = np.random.uniform(1, 5, size=len(df_grouped))  # Satisfaction étudiante simulée

    # Transformation de la variable 'abandon' en binaire : 1 = échec/retrait, 0 = réussite
    df_grouped['abandon'] = df_grouped['abandon'].map(lambda x: 1 if x in ['Withdrawn', 'Fail'] else 0)

    return df_grouped  # Retourne le DataFrame préparé

# Génère des données synthétiques pour simuler d'autres scénarios
@st.cache_data
def generate_synthetic_data(n=1000):
    np.random.seed(42)  # Reproductibilité des données aléatoires
    return pd.DataFrame({
        'age': np.random.randint(18, 50, n),  # Tranche d'âge
        'sexe': np.random.choice(['M', 'F'], n),  # Sexe
        'region': np.random.choice(['Urbain', 'Rural'], n),  # Zone géographique
        'niveau_parental': np.random.choice(['Secondaire', 'Supérieur', 'Aucun'], n),  # Éducation parentale
        'note_moyenne': np.random.uniform(0, 100, n),  # Note moyenne simulée
        'taux_absentéisme': np.random.uniform(0, 50, n),  # Absences simulées
        'temps_moodle': np.random.uniform(0, 20, n),  # Temps sur Moodle
        'participation_forum': np.random.randint(0, 50, n),  # Participation forum
        'satisfaction': np.random.uniform(1, 5, n),  # Satisfaction simulée
        'abandon': np.random.choice([0, 1], n, p=[0.8, 0.2])  # 80% non-abandon, 20% abandon
    })

# Prétraitement : gestion des valeurs manquantes et encodage des variables catégorielles
def preprocess(df):
    df = df.fillna(df.mean(numeric_only=True))  # Remplir les valeurs numériques manquantes par la moyenne
    cat_cols = df.select_dtypes(include='object').columns  # Colonnes catégorielles
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)  # Encodage one-hot des variables catégorielles
    return df

# Pipeline complet d’analyse exploratoire et prédictive
def analysis_pipeline(df, label, pdf_key):
    df = preprocess(df)  # Prétraitement des données

    # 🔍 Analyse exploratoire
    st.subheader("Analyse Exploratoire")
    st.plotly_chart(px.histogram(df, x=label, color='abandon'))  # Histogramme interactif

    # 🔥 Carte de corrélation
    fig, ax = plt.subplots()
    sns.heatmap(df.corr(), annot=False, cmap='coolwarm', ax=ax)  # Heatmap sans valeurs affichées
    st.pyplot(fig)

    # 📦 Boxplots de comparaison par variable
    st.subheader("Boxplots")
    num_cols = df.select_dtypes(include=np.number).columns  # Colonnes numériques
    for col in num_cols:
        if col != 'abandon':  # Ne pas tracer pour la cible
            fig = px.box(df, y=col, color='abandon')  # Boxplot par classe d’abandon
            st.plotly_chart(fig)

    # 📊 Clustering K-Means
    st.subheader("Clustering K-Means")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df.drop(['abandon'], axis=1, errors='ignore'))  # Normalisation des données
    df['cluster'] = KMeans(n_clusters=3, random_state=42).fit_predict(X_scaled)  # Application du clustering
    if label in df.columns:
        st.plotly_chart(px.scatter(df, x=label, y='temps_moodle', color='cluster'))  # Visualisation 2D

    # 🌲 Classification Random Forest
    st.subheader("Classification Random Forest")
    X = df.drop(['abandon', 'cluster'], axis=1, errors='ignore')  # Variables explicatives
    y = df['abandon']  # Cible
    clf = RandomForestClassifier().fit(X, y)  # Entraînement du modèle
    st.success("Modèle entraîné avec succès")

    # 📐 Règles d'association (Apriori)
    st.subheader("Règles d'association")
    df_assoc = df.copy()
    for col in df_assoc.select_dtypes(include=['float64', 'int64']).columns:
        if df_assoc[col].nunique() >= 3:
            # Discrétisation des colonnes continues en 3 classes : bas, moyen, haut
            df_assoc[col] = pd.qcut(df_assoc[col], q=3, labels=['Low', 'Medium', 'High'], duplicates='drop')
    df_assoc = pd.get_dummies(df_assoc)  # Encodage binaire
    df_assoc = df_assoc.loc[:, df_assoc.apply(lambda x: set(x.unique()).issubset({0, 1}))]  # Ne garder que les colonnes binaires

    # Génération des règles
    if not df_assoc.empty:
        freq_items = apriori(df_assoc, min_support=0.1, use_colnames=True)  # Objets fréquents
        rules = association_rules(freq_items, metric='confidence', min_threshold=0.6)  # Règles filtrées
        st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence']])  # Affichage
    else:
        st.info("Aucune règle d'association trouvée.")

    # 🧪 Simulation personnalisée pour un étudiant
    st.subheader("Simulation individuelle")
    input_data = {}
    for col in X.columns:
        if X[col].dtype in ['float64', 'int64']:
            input_data[col] = st.slider(f"{col}", float(X[col].min()), float(X[col].max()), float(X[col].mean()))
        else:
            input_data[col] = st.selectbox(f"{col}", X[col].unique())  # Sélection pour variables catégorielles
    input_df = pd.DataFrame([input_data])
    input_df = pd.get_dummies(input_df).reindex(columns=X.columns, fill_value=0)  # Aligner les colonnes

    risk = clf.predict_proba(input_df)[0][1]  # Prédiction de probabilité d’abandon
    st.metric(label="Risque d'abandon estimé", value=f"{risk:.2%}")  # Affichage en pourcentage

    # 📄 Génération d’un rapport PDF téléchargeable
    st.subheader("Téléchargement du rapport")
    if st.button("Générer PDF", key=pdf_key):
        buffer = io.BytesIO()  # Buffer en mémoire pour le PDF
        c = canvas.Canvas(buffer, pagesize=letter)  # Début du PDF
        c.drawString(100, 750, f"Risque d'abandon: {risk:.2%}")  # Titre
        y = 720
        for k, v in input_data.items():
            c.drawString(100, y, f"{k}: {v}")  # Affichage des paramètres
            y -= 20
        c.save()  # Enregistrement du PDF
        buffer.seek(0)
        st.download_button("Télécharger le rapport PDF", buffer, "rapport_abandon.pdf", "application/pdf")  # Bouton de téléchargement

# Interface principale de l’application
st.title("🎓 Prévention de l'abandon scolaire")

# Deux onglets : un pour les données réelles, un pour les données synthétiques
tabs = st.tabs(["Données réelles (OULAD)", "Données synthétiques"])

# Onglet 1 : données réelles
with tabs[0]:
    df_oulad = load_oulad_data()  # Chargement des données OULAD
    analysis_pipeline(df_oulad, label='score_moyen', pdf_key="pdf_oulad")  # Pipeline d’analyse

# Onglet 2 : données simulées
with tabs[1]:
    df_synth = generate_synthetic_data()  # Création des données synthétiques
    analysis_pipeline(df_synth, label='note_moyenne', pdf_key="pdf_synth")  # Pipeline d’analyse
