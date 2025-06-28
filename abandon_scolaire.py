import streamlit as st  # Pour créer l'application web interactive
import pandas as pd  # Pour la manipulation de données sous forme de DataFrame
import numpy as np  # Pour les calculs numériques
import seaborn as sns  # Pour les visualisations statistiques
import matplotlib.pyplot as plt  # Pour les graphiques
import plotly.express as px  # Pour les graphiques interactifs
from sklearn.cluster import KMeans  # Pour le clustering non supervisé
from sklearn.ensemble import RandomForestClassifier  # Pour la classification supervisée
from sklearn.preprocessing import StandardScaler  # Pour normaliser les données
from mlxtend.frequent_patterns import apriori, association_rules  # Pour les règles d'association (Apriori)
import io  # Pour manipuler les flux de données (PDF)
from reportlab.pdfgen import canvas  # Pour générer des PDF
from reportlab.lib.pagesizes import letter  # Pour définir le format de page du PDF

# Configuration initiale de la page Streamlit
st.set_page_config(page_title="Prévention de l'abandon scolaire", layout="wide")

# Chargement et traitement des données OULAD (données réelles)
@st.cache_data  # Mise en cache des résultats pour éviter de recharger à chaque exécution
def load_oulad_data():
    assessments = pd.read_csv("assessments.csv")  # Chargement des évaluations
    student_assessment = pd.read_csv("studentAssessment.csv")  # Chargement des résultats étudiants
    student_info = pd.read_csv("studentInfo.csv")  # Chargement des infos sur les étudiants

    # Fusion des datasets sur les clés correspondantes
    merged = student_assessment.merge(assessments, on='id_assessment', how='left')
    merged = merged.merge(student_info, on='id_student', how='left')

    # Agrégation des données au niveau de chaque étudiant
    df_grouped = merged.groupby('id_student').agg({
        'score': 'mean',  # Moyenne des scores
        'date_submitted': 'count',  # Nombre d'évaluations soumises
        'studied_credits': 'mean',  # Moyenne des crédits
        'age_band': 'first',  # Première valeur d'âge (invariant pour un étudiant)
        'gender': 'first',
        'region': 'first',
        'highest_education': 'first',
        'disability': 'first',
        'final_result': 'first'
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

    # Ajout de colonnes synthétiques simulant d'autres comportements
    df_grouped['temps_moodle'] = np.random.uniform(0, 20, size=len(df_grouped))  # Temps passé en ligne
    df_grouped['participation_forum'] = np.random.randint(0, 50, size=len(df_grouped))  # Messages sur le forum
    df_grouped['satisfaction'] = np.random.uniform(1, 5, size=len(df_grouped))  # Satisfaction (note sur 5)

    # Transformation de la variable cible : 1 = abandon (échec ou retrait), 0 = réussite
    df_grouped['abandon'] = df_grouped['abandon'].map(lambda x: 1 if x in ['Withdrawn', 'Fail'] else 0)

    return df_grouped

# Génération de données synthétiques pour simulation
@st.cache_data
def generate_synthetic_data(n=1000):
    np.random.seed(42)  # Pour reproductibilité
    return pd.DataFrame({
        'age': np.random.randint(18, 50, n),
        'sexe': np.random.choice(['M', 'F'], n),
        'region': np.random.choice(['Urbain', 'Rural'], n),
        'niveau_parental': np.random.choice(['Secondaire', 'Supérieur', 'Aucun'], n),
        'note_moyenne': np.random.uniform(0, 100, n),
        'taux_absentéisme': np.random.uniform(0, 50, n),
        'temps_moodle': np.random.uniform(0, 20, n),
        'participation_forum': np.random.randint(0, 50, n),
        'satisfaction': np.random.uniform(1, 5, n),
        'abandon': np.random.choice([0, 1], n, p=[0.8, 0.2])  # 80% de réussite, 20% d'abandon
    })

# Prétraitement des données (nettoyage + encodage)
def preprocess(df):
    df = df.fillna(df.mean(numeric_only=True))  # Remplacement des valeurs manquantes numériques par la moyenne
    cat_cols = df.select_dtypes(include='object').columns  # Sélection des colonnes catégorielles
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)  # Encodage one-hot
    return df

# Pipeline d'analyse globale
def analysis_pipeline(df, label, pdf_key):
    df = preprocess(df)  # Nettoyage des données

    st.subheader("Analyse Exploratoire")
    st.plotly_chart(px.histogram(df, x=label, color='abandon'))  # Histogramme interactif

    fig, ax = plt.subplots()  # Création d'une figure matplotlib
    sns.heatmap(df.corr(), annot=False, cmap='coolwarm', ax=ax)  # Carte de corrélation
    st.pyplot(fig)

    st.subheader("Boxplots")
    num_cols = df.select_dtypes(include=np.number).columns  # Colonnes numériques
    for col in num_cols:
        if col != 'abandon':
            fig = px.box(df, y=col, color='abandon')  # Boxplot par classe
            st.plotly_chart(fig)

    st.subheader("Clustering K-Means")
    scaler = StandardScaler()  # Standardisation des données
    X_scaled = scaler.fit_transform(df.drop(['abandon'], axis=1, errors='ignore'))
    df['cluster'] = KMeans(n_clusters=3, random_state=42).fit_predict(X_scaled)  # Clustering en 3 groupes
    if label in df.columns:
        st.plotly_chart(px.scatter(df, x=label, y='temps_moodle', color='cluster'))  # Visualisation 2D

    st.subheader("Classification Random Forest")
    X = df.drop(['abandon', 'cluster'], axis=1, errors='ignore')  # Variables explicatives
    y = df['abandon']  # Variable cible
    clf = RandomForestClassifier().fit(X, y)  # Entraînement du modèle
    st.success("Modèle entraîné avec succès")

    st.subheader("Règles d'association")
    df_assoc = df.copy()  # Copie du dataset pour transformation
    for col in df_assoc.select_dtypes(include=['float64', 'int64']).columns:
        if df_assoc[col].nunique() >= 3:
            df_assoc[col] = pd.qcut(df_assoc[col], q=3, labels=['Low', 'Medium', 'High'], duplicates='drop')  # Discrétisation
    df_assoc = pd.get_dummies(df_assoc)  # Encodage binaire
    df_assoc = df_assoc.loc[:, df_assoc.apply(lambda x: set(x.unique()).issubset({0, 1}))]  # Garde les colonnes binaires
    if not df_assoc.empty:
        freq_items = apriori(df_assoc, min_support=0.1, use_colnames=True)  # Items fréquents
        rules = association_rules(freq_items, metric='confidence', min_threshold=0.6)  # Règles avec confiance >= 60%
        st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence']])
    else:
        st.info("Aucune règle d'association trouvée.")

    st.subheader("Simulation individuelle")
    input_data = {}
    for col in X.columns:
        if X[col].dtype in ['float64', 'int64']:
            input_data[col] = st.slider(f"{col}", float(X[col].min()), float(X[col].max()), float(X[col].mean()))  # Entrée via curseur
        else:
            input_data[col] = st.selectbox(f"{col}", X[col].unique())  # Choix via menu déroulant

    input_df = pd.DataFrame([input_data])
    input_df = pd.get_dummies(input_df).reindex(columns=X.columns, fill_value=0)  # Encodage + réindexation
    risk = clf.predict_proba(input_df)[0][1]  # Probabilité d'abandon
    st.metric(label="Risque d'abandon estimé", value=f"{risk:.2%}")  # Affichage du risque

    st.subheader("Téléchargement du rapport")
    if st.button("Générer PDF", key=pdf_key):
        buffer = io.BytesIO()  # Mémoire tampon pour le PDF
        c = canvas.Canvas(buffer, pagesize=letter)
        c.drawString(100, 750, f"Risque d'abandon: {risk:.2%}")  # Texte principal
        y = 720
        for k, v in input_data.items():
            c.drawString(100, y, f"{k}: {v}")
            y -= 20  # Décalage vertical
        c.save()
        buffer.seek(0)  # Retour au début du flux
        st.download_button("Télécharger le rapport PDF", buffer, "rapport_abandon.pdf", "application/pdf")  # Téléchargement

# Interface principale
st.title("\ud83c\udf93 Prévention de l'abandon scolaire")
tabs = st.tabs(["Données réelles (OULAD)", "Données synthétiques"])  # Onglets d'analyse

with tabs[0]:  # Première tab : données réelles
    df_oulad = load_oulad_data()
    analysis_pipeline(df_oulad, label='score_moyen', pdf_key="pdf_oulad")

with tabs[1]:  # Deuxième tab : données générées
    df_synth = generate_synthetic_data()
    analysis_pipeline(df_synth, label='note_moyenne', pdf_key="pdf_synth")
