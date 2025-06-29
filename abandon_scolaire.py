# ============================================
# Projet : Prévention de l'abandon scolaire avec Streamlit
# Auteur : KOMOSSI Sosso
# ============================================

# === Importation des bibliothèques ===
import streamlit as st  # Bibliothèque pour créer des applications web interactives
import pandas as pd  # Manipulation et analyse de données sous forme de DataFrames
import numpy as np  # Calculs numériques et génération de nombres aléatoires
import seaborn as sns  # Visualisation statistique avancée (heatmaps, distributions)
import matplotlib.pyplot as plt  # Création de graphiques et visualisations
import plotly.express as px  # Graphiques interactifs modernes
from sklearn.cluster import KMeans  # Algorithme de clustering non supervisé
from sklearn.ensemble import RandomForestClassifier  # Algorithme de classification supervisée
from sklearn.model_selection import train_test_split  # Division des données en ensembles d'entraînement et de test
from sklearn.preprocessing import StandardScaler  # Normalisation et standardisation des données
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay  # Évaluation des performances de classification
from mlxtend.frequent_patterns import apriori, association_rules  # Extraction de règles d'association
import io  # Gestion des flux de données en mémoire
from reportlab.lib.pagesizes import letter  # Format de page pour les documents PDF
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle  # Éléments de mise en page PDF
from reportlab.lib import colors  # Palette de couleurs pour les documents PDF
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle  # Styles de formatage pour les paragraphes PDF

# === Configuration de la page Streamlit ===
st.set_page_config(page_title="Prévention de l'abandon scolaire", layout="wide")  # Configuration du titre et de la largeur de la page

# === Chargement des données réelles OULAD ===
@st.cache_data  # Mise en cache des données pour éviter le rechargement à chaque exécution
def load_oulad_data():  # Fonction pour charger et prétraiter les données OULAD
    assessments = pd.read_csv("assessments.csv")  # Lecture du fichier CSV contenant les informations sur les évaluations
    student_assessment = pd.read_csv("studentAssessment.csv")  # Lecture du fichier CSV contenant les notes des étudiants
    student_info = pd.read_csv("studentInfo.csv")  # Lecture du fichier CSV contenant les informations personnelles des étudiants

    merged = student_assessment.merge(assessments, on='id_assessment')  # Fusion des données d'évaluation avec les informations d'évaluation
    merged = merged.merge(student_info, on='id_student')  # Fusion avec les informations personnelles des étudiants

    df = merged.groupby('id_student').agg({  # Regroupement des données par étudiant et agrégation
        'score': 'mean',  # Calcul du score moyen par étudiant
        'date_submitted': 'count',  # Comptage du nombre d'évaluations soumises
        'studied_credits': 'mean',  # Calcul de la moyenne des crédits étudiés
        'age_band': 'first',  # Récupération de la première valeur de la tranche d'âge
        'gender': 'first',  # Récupération de la première valeur du genre
        'region': 'first',  # Récupération de la première valeur de la région
        'highest_education': 'first',  # Récupération de la première valeur du niveau d'éducation
        'disability': 'first',  # Récupération de la première valeur du statut de handicap
        'final_result': 'first'  # Récupération de la première valeur du résultat final
    }).reset_index()  # Réinitialisation de l'index après le regroupement

    df.rename(columns={  # Renommage des colonnes pour une meilleure lisibilité en français
        'score': 'score_moyen',  # Renommage de 'score' en 'score_moyen'
        'date_submitted': 'nb_evaluations',  # Renommage de 'date_submitted' en 'nb_evaluations'
        'studied_credits': 'credits_etudies',  # Renommage de 'studied_credits' en 'credits_etudies'
        'age_band': 'age',  # Renommage de 'age_band' en 'age'
        'gender': 'sexe',  # Renommage de 'gender' en 'sexe'
        'highest_education': 'niveau_parental',  # Renommage de 'highest_education' en 'niveau_parental'
        'disability': 'handicap',  # Renommage de 'disability' en 'handicap'
        'final_result': 'abandon'  # Renommage de 'final_result' en 'abandon'
    }, inplace=True)  # Modification directe du DataFrame original

    df['temps_moodle'] = np.random.uniform(0, 20, size=len(df))  # Génération aléatoire du temps passé sur la plateforme Moodle
    df['participation_forum'] = np.random.randint(0, 50, size=len(df))  # Génération aléatoire du nombre de participations aux forums
    df['satisfaction'] = np.random.uniform(1, 5, size=len(df))  # Génération aléatoire du niveau de satisfaction (échelle 1-5)
    df['abandon'] = df['abandon'].map(lambda x: 1 if x in ['Withdrawn', 'Fail'] else 0)  # Conversion de la variable d'abandon en binaire (1=abandon, 0=réussite)

    return df  # Retour du DataFrame prétraité

# === Génération de données synthétiques ===
@st.cache_data  # Mise en cache des données synthétiques pour éviter la régénération
def generate_synthetic_data(n=1000):  # Fonction pour générer des données synthétiques
    np.random.seed(42)  # Fixation de la graine aléatoire pour assurer la reproductibilité
    return pd.DataFrame({  # Création d'un DataFrame avec des données synthétiques
        'age': np.random.randint(18, 50, n),  # Génération d'âges aléatoires entre 18 et 50 ans
        'sexe': np.random.choice(['M', 'F'], n),  # Génération aléatoire du sexe (Masculin ou Féminin)
        'region': np.random.choice(['Urbain', 'Rural'], n),  # Génération aléatoire de la région (Urbain ou Rural)
        'niveau_parental': np.random.choice(['Secondaire', 'Supérieur', 'Aucun'], n),  # Génération aléatoire du niveau d'éducation parental
        'note_moyenne': np.random.uniform(0, 100, n),  # Génération de notes moyennes aléatoires entre 0 et 100
        'taux_absenteisme': np.random.uniform(0, 50, n),  # Génération de taux d'absentéisme aléatoires entre 0 et 50%
        'temps_moodle': np.random.uniform(0, 20, n),  # Génération du temps passé sur Moodle (0-20 heures)
        'participation_forum': np.random.randint(0, 50, n),  # Génération du nombre de participations aux forums (0-50)
        'satisfaction': np.random.uniform(1, 5, n),  # Génération du niveau de satisfaction (1-5)
        'abandon': np.random.choice([0, 1], n, p=[0.8, 0.2])  # Génération de la variable d'abandon avec 20% de probabilité d'abandon
    })

# === Prétraitement des données ===
def preprocess(df):  # Fonction de prétraitement des données
    df = df.copy()  # Création d'une copie du DataFrame pour éviter les modifications sur l'original
    df = df.fillna(df.mean(numeric_only=True))  # Remplacement des valeurs manquantes par la moyenne pour les colonnes numériques
    cat_cols = df.select_dtypes(include='object').columns  # Identification des colonnes catégorielles (type object)
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)  # Encodage one-hot des variables catégorielles avec suppression de la première modalité
    return df  # Retour du DataFrame prétraité

# === Pipeline d'analyse ===
def analysis_pipeline(df, label, pdf_key):  # Pipeline principal d'analyse des données
    df = preprocess(df)  # Prétraitement des données avant analyse

    st.subheader("1. Statistiques Descriptives")  # Affichage du titre de la section des statistiques descriptives
    st.dataframe(df.describe())  # Affichage des statistiques descriptives (moyenne, médiane, écart-type, etc.)

    st.subheader("2. Histogrammes")  # Affichage du titre de la section des histogrammes
    for col in ['score_moyen', 'note_moyenne', 'satisfaction', 'temps_moodle', 'participation_forum']:  # Boucle sur les colonnes importantes à visualiser
        if col in df.columns:  # Vérification de l'existence de la colonne dans le DataFrame
            st.plotly_chart(px.histogram(df, x=col, color=df['abandon'].astype(str)))  # Création d'un histogramme coloré selon le statut d'abandon

    st.subheader("3. Matrice de Corrélation")  # Affichage du titre de la section de corrélation
    fig, ax = plt.subplots()  # Création d'une nouvelle figure matplotlib
    sns.heatmap(df.corr(), annot=False, cmap="coolwarm", ax=ax)  # Création d'une heatmap des corrélations entre variables
    st.pyplot(fig)  # Affichage de la figure dans Streamlit

    st.subheader("4. Boxplots")  # Affichage du titre de la section des boxplots
    for col in df.select_dtypes(include=np.number).columns:  # Boucle sur toutes les colonnes numériques
        if col != 'abandon':  # Exclusion de la variable cible pour éviter la redondance
            st.plotly_chart(px.box(df, y=col, color=df['abandon'].astype(str)))  # Création d'un boxplot coloré selon le statut d'abandon

    st.subheader("5. Clustering K-Means")  # Affichage du titre de la section de clustering
    X_clust = StandardScaler().fit_transform(df.drop(['abandon'], axis=1, errors='ignore'))  # Standardisation des données pour le clustering
    df['cluster'] = KMeans(n_clusters=3, random_state=42).fit_predict(X_clust)  # Application de l'algorithme K-Means avec 3 clusters
    st.plotly_chart(px.scatter(df, x=label, y='temps_moodle', color='cluster'))  # Visualisation des clusters dans un graphique en nuage de points

    st.subheader("6. Entraînement du modèle Random Forest")  # Affichage du titre de la section d'entraînement du modèle
    X = df.drop(['abandon', 'cluster'], axis=1, errors='ignore')  # Définition des variables explicatives (features)
    y = df['abandon']  # Définition de la variable cible (target)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)  # Division des données en ensembles d'entraînement et de test
    clf = RandomForestClassifier()  # Création d'une instance du classificateur Random Forest
    clf.fit(X_train, y_train)  # Entraînement du modèle sur les données d'entraînement
    st.success("Le modèle Random Forest a été entraîné avec succès !")  # Message de confirmation de l'entraînement

    st.subheader("7. Matrice de confusion")  # Affichage du titre de la section de la matrice de confusion
    y_pred = clf.predict(X_test)  # Prédiction sur l'ensemble de test
    cm = confusion_matrix(y_test, y_pred)  # Calcul de la matrice de confusion
    fig_cm, ax_cm = plt.subplots()  # Création d'une nouvelle figure pour la matrice de confusion
    ConfusionMatrixDisplay(cm).plot(ax=ax_cm)  # Affichage de la matrice de confusion
    st.pyplot(fig_cm)  # Affichage de la figure dans Streamlit

    st.subheader("8. Règles d'association")  # Affichage du titre de la section des règles d'association
    df_assoc = df.copy()  # Création d'une copie du DataFrame pour l'analyse des règles d'association
    for col in df_assoc.select_dtypes(include=['float64', 'int64']).columns:  # Boucle sur les colonnes numériques
        if df_assoc[col].nunique() > 2:  # Si la colonne a plus de 2 valeurs uniques
            df_assoc[col] = pd.qcut(df_assoc[col], 3, labels=["Low", "Medium", "High"], duplicates='drop')  # Discrétisation en 3 catégories
    df_assoc = pd.get_dummies(df_assoc)  # Encodage one-hot de toutes les variables
    df_assoc = df_assoc.loc[:, df_assoc.apply(lambda x: set(x.unique()).issubset({0, 1}))]  # Sélection des colonnes binaires uniquement
    freq_items = apriori(df_assoc, min_support=0.1, use_colnames=True)  # Extraction des itemsets fréquents avec support minimum de 0.1
    if not freq_items.empty:  # Si des itemsets fréquents ont été trouvés
        rules = association_rules(freq_items, metric="confidence", min_threshold=0.6)  # Génération des règles d'association avec confiance minimum de 0.6
        st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence']])  # Affichage des règles trouvées
    else:  # Si aucun itemset fréquent n'a été trouvé
        st.info("Aucune règle d'association trouvée.")  # Message d'information

    st.subheader("9. Simulation individuelle")  # Affichage du titre de la section de simulation
    input_data = {}  # Dictionnaire pour stocker les données d'entrée de l'utilisateur
    for col in X.columns:  # Boucle sur toutes les colonnes de features
        if X[col].dtype in [np.float64, np.int64]:  # Si la colonne est numérique
            input_data[col] = st.slider(f"{col}", float(X[col].min()), float(X[col].max()), float(X[col].mean()))  # Création d'un slider pour la saisie numérique
        else:  # Si la colonne est catégorielle
            input_data[col] = st.selectbox(f"{col}", X[col].unique())  # Création d'une boîte de sélection pour la saisie catégorielle

    input_df = pd.DataFrame([input_data])  # Création d'un DataFrame avec les données saisies
    input_df = pd.get_dummies(input_df).reindex(columns=X.columns, fill_value=0)  # Encodage et alignement avec les colonnes du modèle
    risk = clf.predict_proba(input_df)[0][1]  # Calcul de la probabilité d'abandon pour l'étudiant simulé
    st.metric(label="Risque d'abandon estimé", value=f"{risk:.2%}")  # Affichage du risque estimé sous forme de métrique

    st.subheader("10. Rapport PDF")  # Affichage du titre de la section de génération de rapport
    if st.button("Générer PDF", key=pdf_key):  # Bouton pour déclencher la génération du PDF
        buffer = io.BytesIO()  # Création d'un buffer en mémoire pour le PDF
        doc = SimpleDocTemplate(buffer, pagesize=letter)  # Création du document PDF avec format lettre
        styles = getSampleStyleSheet()  # Récupération des styles par défaut
        elements = []  # Liste pour stocker les éléments du PDF

        title_style = ParagraphStyle(  # Création d'un style personnalisé pour le titre
            name='TitleStyle',  # Nom du style
            parent=styles['Heading1'],  # Style parent
            alignment=1,  # Alignement centré
            fontSize=18,  # Taille de police
            spaceAfter=20,  # Espace après le titre
            textColor=colors.darkblue  # Couleur du texte
        )
        elements.append(Paragraph("Rapport de Risque d'Abandon Scolaire", title_style))  # Ajout du titre au PDF

        table_data = [["Variable", "Valeur"]] + [[k, str(v)] for k, v in input_data.items()]  # Création des données du tableau
        table = Table(table_data)  # Création du tableau
        table.setStyle(TableStyle([  # Application du style au tableau
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#003366')),  # Couleur de fond de l'en-tête
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),  # Couleur du texte de l'en-tête
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),  # Bordures du tableau
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),  # Couleur de fond des cellules
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold')  # Police en gras pour l'en-tête
        ]))
        elements.append(table)  # Ajout du tableau au PDF
        elements.append(Spacer(1, 20))  # Ajout d'un espace vertical
        elements.append(Paragraph(f"<b>Risque estimé d'abandon :</b> {risk:.2%}", styles['Normal']))  # Ajout du risque estimé
        reco = "<b>Recommandation :</b> Suivi rapproché par le conseiller pédagogique" if risk >= 0.5 else "<b>Recommandation :</b> Maintenir les efforts actuels"  # Détermination de la recommandation
        elements.append(Paragraph(reco, styles['Normal']))  # Ajout de la recommandation

        doc.build(elements)  # Construction du document PDF
        buffer.seek(0)  # Retour au début du buffer
        st.download_button("Télécharger le rapport PDF", buffer, "rapport_abandon.pdf", "application/pdf")  # Bouton de téléchargement du PDF

# === Interface principale ===
st.title("🎓 Prévention de l'abandon scolaire grâce au Data Mining")  # Titre principal de l'application
tabs = st.tabs(["Données réelles (OULAD)", "Données synthétiques"])  # Création de deux onglets

with tabs[0]:  # Premier onglet pour les données réelles OULAD
    df_oulad = load_oulad_data()  # Chargement des données OULAD
    analysis_pipeline(df_oulad, label='score_moyen', pdf_key="pdf_oulad")  # Exécution du pipeline d'analyse

with tabs[1]:  # Deuxième onglet pour les données synthétiques
    df_synth = generate_synthetic_data()  # Génération des données synthétiques
    analysis_pipeline(df_synth, label='note_moyenne', pdf_key="pdf_synth")  # Exécution du pipeline d'analyse
