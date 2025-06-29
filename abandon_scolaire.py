# ============================================
# Projet : Prévention de l'abandon scolaire avec Streamlit
# Auteur : KOMOSSI Sosso
# Description : Application interactive prédictive avec exploration, clustering, classification, simulation et rapport PDF
# ============================================

# === Importation des bibliothèques ===
import streamlit as st  # Bibliothèque pour créer une interface web interactive
import pandas as pd  # Manipulation efficace des données sous forme de DataFrames
import numpy as np  # Fonctions numériques avancées et génération aléatoire
import seaborn as sns  # Visualisation statistique (heatmaps, boxplots)
import matplotlib.pyplot as plt  # Création de graphiques simples
import plotly.express as px  # Graphiques interactifs avancés
from sklearn.cluster import KMeans  # Algorithme de clustering non supervisé
from sklearn.ensemble import RandomForestClassifier  # Classifieur supervisé robuste
from sklearn.model_selection import train_test_split  # Séparation en ensemble d'entraînement/test
from sklearn.preprocessing import StandardScaler  # Normalisation des données
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay  # Matrice de confusion
from mlxtend.frequent_patterns import apriori, association_rules  # Extraction de règles d'association
import io  # Gestion de fichiers en mémoire
from reportlab.pdfgen import canvas  # Génération de documents PDF
from reportlab.lib.pagesizes import letter  # Format de page standard pour PDF
from reportlab.platypus import Table, TableStyle, Paragraph, SimpleDocTemplate, Spacer  # Tableaux et paragraphes
from reportlab.lib import colors  # Couleurs pour PDF
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle  # Styles pour les paragraphes PDF

# === Configuration de la page Streamlit ===
st.set_page_config(page_title="Prévention de l'abandon scolaire", layout="wide")  # Configuration du titre et layout de la page

# === Fonction de chargement des données OULAD (avec cache) ===
@st.cache_data  # Décorateur pour mettre en cache les données et éviter le rechargement

def load_oulad_data():  # Fonction pour charger et traiter les données OULAD
    assessments = pd.read_csv("assessments.csv")  # Lecture du fichier CSV des évaluations
    student_assessment = pd.read_csv("studentAssessment.csv")  # Lecture du fichier CSV des notes des étudiants
    student_info = pd.read_csv("studentInfo.csv")  # Lecture du fichier CSV des informations des étudiants

    merged = student_assessment.merge(assessments, on='id_assessment')  # Fusion des évaluations avec les notes par ID d'évaluation
    merged = merged.merge(student_info, on='id_student')  # Fusion avec les informations étudiants par ID étudiant

    df = merged.groupby('id_student').agg({  # Regroupement par étudiant et agrégation des données
        'score': 'mean',  # Score moyen par étudiant
        'date_submitted': 'count',  # Nombre d'évaluations soumises
        'studied_credits': 'mean',  # Moyenne des crédits étudiés
        'age_band': 'first',  # Première valeur de la tranche d'âge
        'gender': 'first',  # Première valeur du genre
        'region': 'first',  # Première valeur de la région
        'highest_education': 'first',  # Première valeur du niveau d'éducation
        'disability': 'first',  # Première valeur du handicap
        'final_result': 'first'  # Première valeur du résultat final
    }).reset_index()  # Réinitialisation de l'index

    df.rename(columns={  # Renommage des colonnes en français
        'score': 'score_moyen',  # Renommage de score en score_moyen
        'date_submitted': 'nb_evaluations',  # Renommage en nombre d'évaluations
        'studied_credits': 'credits_etudies',  # Renommage en crédits étudiés
        'age_band': 'age',  # Renommage en âge
        'gender': 'sexe',  # Renommage en sexe
        'highest_education': 'niveau_parental',  # Renommage en niveau parental
        'disability': 'handicap',  # Renommage en handicap
        'final_result': 'abandon'  # Renommage en abandon
    }, inplace=True)  # Modification directe du DataFrame

    df['temps_moodle'] = np.random.uniform(0, 20, size=len(df))  # Génération aléatoire du temps passé sur Moodle
    df['participation_forum'] = np.random.randint(0, 50, size=len(df))  # Génération aléatoire de la participation aux forums
    df['satisfaction'] = np.random.uniform(1, 5, size=len(df))  # Génération aléatoire du niveau de satisfaction
    df['abandon'] = df['abandon'].map(lambda x: 1 if x in ['Withdrawn', 'Fail'] else 0)  # Conversion en variable binaire (1=abandon, 0=succès)

    return df  # Retour du DataFrame traité

@st.cache_data  # Décorateur pour mettre en cache les données synthétiques
def generate_synthetic_data(n=1000):  # Fonction pour générer des données synthétiques
    np.random.seed(42)  # Fixation de la graine aléatoire pour la reproductibilité
    return pd.DataFrame({  # Création d'un DataFrame avec des données aléatoires
        'age': np.random.randint(18, 50, n),  # Âge aléatoire entre 18 et 50 ans
        'sexe': np.random.choice(['M', 'F'], n),  # Sexe aléatoire (M ou F)
        'region': np.random.choice(['Urbain', 'Rural'], n),  # Région aléatoire (Urbain ou Rural)
        'niveau_parental': np.random.choice(['Secondaire', 'Supérieur', 'Aucun'], n),  # Niveau d'éducation parental aléatoire
        'note_moyenne': np.random.uniform(0, 100, n),  # Note moyenne aléatoire entre 0 et 100
        'taux_absenteisme': np.random.uniform(0, 50, n),  # Taux d'absentéisme aléatoire entre 0 et 50%
        'temps_moodle': np.random.uniform(0, 20, n),  # Temps passé sur Moodle aléatoire
        'participation_forum': np.random.randint(0, 50, n),  # Participation aux forums aléatoire
        'satisfaction': np.random.uniform(1, 5, n),  # Satisfaction aléatoire entre 1 et 5
        'abandon': np.random.choice([0, 1], n, p=[0.8, 0.2])  # Variable d'abandon avec 20% de probabilité d'abandon
    })

def preprocess(df):  # Fonction de préprocessing des données
    df = df.copy()  # Création d'une copie du DataFrame pour éviter les modifications
    df = df.fillna(df.mean(numeric_only=True))  # Remplissage des valeurs manquantes par la moyenne
    cat_cols = df.select_dtypes(include='object').columns  # Sélection des colonnes catégorielles
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)  # Encodage one-hot des variables catégorielles
    return df  # Retour du DataFrame préprocessé

def analysis_pipeline(df, label, pdf_key):  # Pipeline principal d'analyse des données
    df = preprocess(df)  # Préprocessing des données

    st.subheader("1. Statistiques Descriptives")  # Affichage du titre de la section
    st.dataframe(df.describe())  # Affichage des statistiques descriptives

    st.subheader("2. Histogrammes pertinents")  # Affichage du titre de la section histogrammes
    for col in ['score_moyen', 'note_moyenne', 'satisfaction', 'temps_moodle', 'participation_forum']:  # Boucle sur les colonnes importantes
        if col in df.columns:  # Vérification de l'existence de la colonne
            st.plotly_chart(px.histogram(df, x=col, color=df['abandon'].astype(str)))  # Création d'un histogramme coloré par abandon

    st.subheader("3. Matrice de corrélation")  # Affichage du titre de la section corrélation
    fig, ax = plt.subplots()  # Création d'une figure matplotlib
    sns.heatmap(df.corr(), annot=False, cmap="coolwarm", ax=ax)  # Création d'une heatmap de corrélation
    st.pyplot(fig)  # Affichage de la figure dans Streamlit

    st.subheader("4. Boxplots par variable")  # Affichage du titre de la section boxplots
    for col in df.select_dtypes(include=np.number).columns:  # Boucle sur les colonnes numériques
        if col != 'abandon':  # Exclusion de la variable cible
            st.plotly_chart(px.box(df, y=col, color=df['abandon'].astype(str)))  # Création d'un boxplot coloré par abandon

    st.subheader("5. Clustering KMeans")  # Affichage du titre de la section clustering
    X_clust = StandardScaler().fit_transform(df.drop(['abandon'], axis=1, errors='ignore'))  # Standardisation des données pour le clustering
    df['cluster'] = KMeans(n_clusters=3, random_state=42).fit_predict(X_clust)  # Application de K-means avec 3 clusters
    st.plotly_chart(px.scatter(df, x=label, y='temps_moodle', color='cluster'))  # Visualisation des clusters

    st.subheader("6. Classification Random Forest + matrice de confusion")  # Affichage du titre de la section classification
    X = df.drop(['abandon', 'cluster'], axis=1, errors='ignore')  # Définition des variables explicatives
    y = df['abandon']  # Définition de la variable cible
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)  # Division train/test
    clf = RandomForestClassifier().fit(X_train, y_train)  # Entraînement du modèle Random Forest
    y_pred = clf.predict(X_test)  # Prédiction sur l'ensemble de test
    cm = confusion_matrix(y_test, y_pred)  # Calcul de la matrice de confusion
    fig_cm, ax_cm = plt.subplots()  # Création d'une figure pour la matrice de confusion
    ConfusionMatrixDisplay(cm).plot(ax=ax_cm)  # Affichage de la matrice de confusion
    st.pyplot(fig_cm)  # Affichage dans Streamlit

    st.subheader("7. Règles d'association (Apriori)")  # Affichage du titre de la section règles d'association
    df_assoc = df.copy()  # Copie du DataFrame pour les règles d'association
    for col in df_assoc.select_dtypes(include=['float64', 'int64']).columns:  # Boucle sur les colonnes numériques
        if df_assoc[col].nunique() > 2:  # Si la colonne a plus de 2 valeurs uniques
            df_assoc[col] = pd.qcut(df_assoc[col], 3, labels=["Low", "Medium", "High"], duplicates='drop')  # Discrétisation en quartiles
    df_assoc = pd.get_dummies(df_assoc)  # Encodage one-hot
    df_assoc = df_assoc.loc[:, df_assoc.apply(lambda x: set(x.unique()).issubset({0, 1}))]  # Sélection des colonnes binaires
    freq_items = apriori(df_assoc, min_support=0.1, use_colnames=True)  # Extraction des itemsets fréquents
    if not freq_items.empty:  # Si des itemsets fréquents sont trouvés
        rules = association_rules(freq_items, metric="confidence", min_threshold=0.6)  # Génération des règles d'association
        st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence']])  # Affichage des règles
    else:  # Si aucun itemset fréquent n'est trouvé
        st.info("Aucune règle d'association trouvée.")  # Message d'information

    st.subheader("8. Simulation d'étudiant")  # Affichage du titre de la section simulation
    input_data = {}  # Dictionnaire pour stocker les données d'entrée
    for col in X.columns:  # Boucle sur toutes les colonnes de features
        if X[col].dtype in [np.float64, np.int64]:  # Si la colonne est numérique
            input_data[col] = st.slider(f"{col}", float(X[col].min()), float(X[col].max()), float(X[col].mean()))  # Slider pour saisie numérique
        else:  # Si la colonne est catégorielle
            input_data[col] = st.selectbox(f"{col}", X[col].unique())  # Selectbox pour saisie catégorielle

    input_df = pd.DataFrame([input_data])  # Création d'un DataFrame avec les données d'entrée
    input_df = pd.get_dummies(input_df).reindex(columns=X.columns, fill_value=0)  # Encodage et alignement avec les colonnes du modèle
    risk = clf.predict_proba(input_df)[0][1]  # Prédiction de la probabilité d'abandon
    st.metric(label="Risque d'abandon estimé", value=f"{risk:.2%}")  # Affichage du risque sous forme de métrique

    st.subheader("9. Rapport PDF")  # Affichage du titre de la section rapport PDF
    if st.button("Générer PDF", key=pdf_key):  # Bouton pour générer le PDF
        buffer = io.BytesIO()  # Création d'un buffer en mémoire
        doc = SimpleDocTemplate(buffer, pagesize=letter)  # Création du document PDF
        elements = []  # Liste pour stocker les éléments du PDF

        styles = getSampleStyleSheet()  # Récupération des styles par défaut
        title_style = ParagraphStyle(  # Création d'un style pour le titre
            name='TitleStyle', parent=styles['Heading1'], alignment=1, fontSize=18, spaceAfter=20, textColor=colors.darkblue  # Définition des propriétés du style
        )
        elements.append(Paragraph("Rapport de Risque d'Abandon Scolaire", title_style))  # Ajout du titre au PDF

        table_data = [["Variable", "Valeur"]] + [[k, str(v)] for k, v in input_data.items()]  # Création des données du tableau
        table = Table(table_data, hAlign='LEFT')  # Création du tableau
        table.setStyle(TableStyle([  # Application du style au tableau
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#003366')),  # Couleur de fond de l'en-tête
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),  # Couleur du texte de l'en-tête
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),  # Grille du tableau
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),  # Couleur de fond des cellules
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold')  # Police de l'en-tête
        ]))
        elements.append(table)  # Ajout du tableau au PDF
        elements.append(Spacer(1, 20))  # Ajout d'un espace

        elements.append(Paragraph(f"<b>Risque estimé d'abandon :</b> {risk:.2%}", styles['Normal']))  # Ajout du risque estimé
        reco = "<b>Recommandation :</b> Mettre en place un suivi pédagogique renforcé" if risk >= 0.5 else "<b>Recommandation :</b> Maintenir la progression actuelle"  # Définition de la recommandation
        elements.append(Paragraph(reco, styles['Normal']))  # Ajout de la recommandation

        doc.build(elements)  # Construction du PDF
        buffer.seek(0)  # Retour au début du buffer
        st.download_button("Télécharger le rapport PDF", buffer, "rapport_abandon.pdf", "application/pdf")  # Bouton de téléchargement

# === Interface utilisateur principale ===
st.title("🎓 Prévention de l'abandon scolaire grâce au Data Mining")  # Titre principal de l'application
tabs = st.tabs(["Données réelles (OULAD)", "Données synthétiques"])  # Création des onglets

with tabs[0]:  # Premier onglet pour les données OULAD
    df_oulad = load_oulad_data()  # Chargement des données OULAD
    analysis_pipeline(df_oulad, label='score_moyen', pdf_key="pdf_oulad")  # Exécution du pipeline d'analyse

with tabs[1]:  # Deuxième onglet pour les données synthétiques
    df_synth = generate_synthetic_data()  # Génération des données synthétiques
    analysis_pipeline(df_synth, label='note_moyenne', pdf_key="pdf_synth")  # Exécution du pipeline d'analyse
