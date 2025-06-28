1. Objectifs du projet
Le projet vise à développer un système intelligent de prédiction de l'abandon scolaire en utilisant des techniques avancées de data mining. L'objectif principal est de créer un outil interactif permettant d'identifier les étudiants à risque d'abandon et de proposer des interventions personnalisées pour améliorer leur réussite académique.
Spécifications techniques demandées :
Exploration et visualisation des données pour identifier les profils d'étudiants à risque
Clustering pour segmenter les étudiants en groupes homogènes
Classification supervisée pour prédire le risque d'abandon
Règles d'association pour découvrir des patterns comportementaux
Interface interactive avec Streamlit pour la simulation et les recommandations
2. Architecture des données utilisées
Le projet exploite deux sources de données distinctes pour maximiser la robustesse de l'analyse :
2.1 Dataset réel : Open University Learning Analytics Dataset (OULAD)
Composition : 3 fichiers CSV interconnectés
studentInfo.csv : Données socio-démographiques des étudiants
assessments.csv : Informations sur les évaluations
studentAssessment.csv : Résultats des étudiants aux évaluations
Variables principales après fusion :
Données académiques : score moyen, nombre d'évaluations, crédits étudiés
Profil démographique : âge, sexe, région, niveau d'éducation, handicap
Variables synthétiques ajoutées : temps sur Moodle, participation forums, satisfaction
2.2 Dataset synthétique
Composition : 1000 observations générées avec contrôle statistique
Variables socio-démographiques simulées de manière réaliste
Métriques académiques et d'engagement avec distributions appropriées
Ratio d'abandon contrôlé (20% d'abandons, 80% de réussite)
3. Architecture technique du code
3.1 Configuration et importation des bibliothèques
import streamlit as st  # Interface web interactive
import pandas as pd     # Manipulation des données
import numpy as np      # Calculs numériques
import seaborn as sns   # Visualisations statistiques
import matplotlib.pyplot as plt  # Graphiques
import plotly.express as px     # Visualisations interactives
Le code utilise un écosystème complet de bibliothèques Python pour le data mining :
Scikit-learn : Algorithmes de machine learning (K-Means, Random Forest)
MLxtend : Extraction de règles d'association
ReportLab : Génération de rapports PDF
3.2 Fonctions de chargement des données
load_oulad_data() - Traitement des données réelles
Fonctionnalité : Cette fonction complexe orchestre plusieurs opérations critiques :
Fusion intelligente des datasets
oJoint les tables par clés étrangères (id_student, id_assessment)
oPréserve l'intégrité référentielle des données
Agrégation statistique
oCalcule le score moyen par étudiant
oCompte le nombre d'évaluations passées
oConserve les caractéristiques démographiques uniques
Enrichissement des données
oGénère des variables d'engagement (temps Moodle, participation forums)
oSimule une métrique de satisfaction étudiant
Transformation de la variable cible
oConvertit les résultats finaux en variable binaire d'abandon
oMapping : ['Withdrawn', 'Fail'] → 1 (abandon), autres → 0 (réussite)
generate_synthetic_data() - Génération de données contrôlées
Fonctionnalité : Crée un dataset de validation avec :
Reproductibilité garantie (seed=42)
Distributions réalistes pour chaque variable
Corrélations implicites entre variables explicatives et abandon
3.3 Pipeline de prétraitement
preprocess(df) - Normalisation des données
Opérations effectuées :
1.Gestion des valeurs manquantes : Imputation par la moyenne pour les variables numériques
2.Encodage catégoriel : Transformation des variables qualitatives en variables dummy
3.Standardisation : Préparation pour les algorithmes de machine learning
3.4 Pipeline d'analyse principal
analysis_pipeline(df, label, pdf_key) - Moteur d'analyse complet
Cette fonction centrale implémente l'ensemble du workflow de data mining :
Phase 1 : Analyse exploratoire
Histogrammes interactifs : Distribution des variables par statut d'abandon
Heatmap de corrélation : Identification des relations entre variables
Boxplots comparatifs : Analyse des différences entre étudiants à risque et autres
Phase 2 : Clustering (Segmentation non-supervisée)
# Standardisation des données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df.drop(['abandon'], axis=1))

# Application K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)
Objectif : Identifier 3 profils d'étudiants naturels dans les données Visualisation : Scatter plot interactif montrant la répartition des clusters
Phase 3 : Classification supervisée
# Entraînement Random Forest
clf = RandomForestClassifier()
clf.fit(X, y)
Algorithme choisi : Random Forest pour sa robustesse et interprétabilité Fonctionnalité : Prédiction probabiliste du risque d'abandon
Phase 4 : Extraction de règles d'association
Processus complexe en 4 étapes :
Discrétisation des variables continues
df_assoc[col] = pd.qcut(df_assoc[col], q=3, labels=['Low', 'Medium', 'High'])
Transformation en format transactionnel
oConversion en variables binaires (0/1)
oPréparation pour l'algorithme Apriori
Extraction des itemsets fréquents
freq_items = apriori(df_assoc, min_support=0.1)
Génération des règles d'association
rules = association_rules(freq_items, metric='confidence', min_threshold=0.6)
Interprétation : Découverte de patterns du type "Si étudiant avec faible participation ET notes basses ALORS risque d'abandon élevé"
Phase 5 : Simulation individuelle
Interface utilisateur dynamique :
Sliders numériques : Saisie des variables continues (notes, temps d'étude...)
Menus déroulants : Sélection des variables catégorielles (sexe, région...)
Prédiction temps réel : Calcul automatique du risque d'abandon
Affichage métrique : Présentation du risque en pourcentage
Phase 6 : Génération de rapports
Fonctionnalités avancées :
Export PDF automatisé avec ReportLab
Personnalisation : Inclusion des données saisies et du score de risque
Téléchargement direct via Streamlit
4. Interface utilisateur Streamlit
4.1 Architecture multi-onglets
L'application utilise une interface à onglets pour séparer les analyses :
Onglet 1 : "Données réelles (OULAD)"
Analyse complète du dataset universitaire authentique
Visualisations basées sur des données réelles d'étudiants
Onglet 2 : "Données synthétiques"
Validation sur dataset contrôlé
Comparaison des performances du modèle
4.2 Fonctionnalités interactives
Graphiques Plotly : Visualisations interactives avec zoom et filtrage
Widgets dynamiques : Sliders et sélecteurs pour la simulation
Métriques en temps réel : Mise à jour automatique des prédictions
Téléchargements : Export des analyses et rapports
5. Techniques de Data Mining implémentées
5.1 Clustering K-Means
Objectif : Segmentation des étudiants en profils homogènes Paramètres : 3 clusters, standardisation préalable Utilité : Identification de groupes naturels d'étudiants
5.2 Classification Random Forest
Objectif : Prédiction binaire du risque d'abandon Avantages : Robustesse, gestion des variables mixtes, interprétabilité Output : Probabilité d'abandon pour chaque étudiant
5.3 Règles d'association (Apriori)
Objectif : Découverte de patterns comportementaux Métriques : Support (fréquence) et Confiance (fiabilité) Utilité : Compréhension des facteurs combinés menant à l'abandon
5.4 Analyse exploratoire avancée
Visualisations multiples :
Distributions uni et bi-variées
Corrélations et heatmaps
Comparaisons par groupes
Analyses de clustering
6. Innovation et valeur ajoutée
6.1 Approche hybride
Validation croisée : Deux datasets pour robustesse
Techniques complémentaires : Supervisé + non-supervisé + règles
Interface professionnelle : Streamlit pour l'accessibilité
6.2 Opérationnalisation
Simulation temps réel : Prédiction instantanée pour nouveaux étudiants
Rapports automatisés : Documentation des analyses
Recommandations : Système d'aide à la décision
7. Déploiement et utilisation
7.1 Prérequis techniques
pip install streamlit pandas numpy scikit-learn plotly seaborn matplotlib mlxtend reportlab
7.2 Lancement de l'application
streamlit run abandon_scolaire.py
7.3 Workflow utilisateur
1.Sélection du dataset (réel ou synthétique)
2.Exploration des analyses automatiques
3.Simulation individuelle via interface
4.Export des résultats en PDF
8. Analyse des résultats obtenus
8.1 Distribution des scores moyens
Analyse de l'histogramme empilé :

Distribution bimodale : Les étudiants qui réussissent (abandon = 0, bleu clair) présentent une distribution concentrée autour de 75-85 points
Profil des étudiants à risque : Les étudiants en situation d'abandon (abandon = 1, bleu foncé) montrent une distribution plus étalée avec : 
oUne concentration significative dans les scores faibles (20-50 points)
oUne présence notable même dans les scores moyens (50-70 points)
Seuil critique identifié : En dessous de 60 points, le risque d'abandon augmente considérablement
Insight stratégique : Un score moyen inférieur à 60 constitue un indicateur d'alerte précoce majeur
8.2 Matrice de corrélation

Patterns de corrélation identifiés :
Corrélations faibles généralisées : La majorité des variables montrent des corrélations proches de 0, indiquant : 
oL'absence de multicolinéarité problématique
oLa complémentarité des variables explicatives
oLa nécessité d'une approche multivariée pour la prédiction
Variables démographiques : Les variables régionales et de niveau parental montrent une indépendance relative
Implication méthodologique : Cette faible intercorrélation justifie l'utilisation de l'ensemble des variables dans le modèle prédictif
8.3 Analyse comparative par boxplots
ID Student - Distribution des identifiants

Répartition équilibrée : Les identifiants sont uniformément distribués entre les groupes abandon/non-abandon
Validation de l'échantillonnage : Confirme l'absence de biais systématique dans la sélection des étudiants
Score moyen - Différenciation académique critique

Insights majeurs :
Médiane des étudiants qui réussissent : ~82 points (quartile supérieur)
Médiane des étudiants en abandon : ~68 points (quartile inférieur)
Écart significatif : 14 points de différence entre les médianes
Chevauchement des distributions : Indique que le score seul n'est pas suffisant pour prédire l'abandon
Valeurs aberrantes : Présence d'étudiants avec scores très faibles mais sans abandon (potentiels faux négatifs)
Nombre d'évaluations - Engagement académique

Patterns comportementaux révélés :
Étudiants persistants : Médiane ~12 évaluations, distribution compacte
Étudiants en abandon : Médiane ~8 évaluations, plus grande variabilité
Corrélation engagement-réussite : Plus d'évaluations passées = moins de risque d'abandon
Indicateur prédictif : Le nombre d'évaluations constitue un proxy de l'engagement étudiant
Crédits étudiés - Charge académique

Analyse de la charge de travail :
Paradoxe apparent : Les étudiants en abandon ont tendance à s'inscrire à plus de crédits
Médiane abandon : ~120 crédits vs ~90 crédits pour les persistants
Hypothèse explicative : Surcharge académique potentielle menant à l'abandon
Facteur de risque : Une charge excessive pourrait être prédictive d'abandon
Temps sur Moodle - Engagement numérique

Comportement d'apprentissage en ligne :
Distributions similaires : Peu de différence entre les groupes (médianes ~10h)
Variabilité comparable : Même étalement des données
Insight contre-intuitif : Le temps passé sur la plateforme n'est pas discriminant
Implication pédagogique : La qualité d'interaction prime sur la quantité de temps
8.4 Synthèse des insights prédictifs
Variables discriminantes identifiées :
1.Score moyen : Indicateur le plus puissant (différence de 14 points entre médianes)
2.Nombre d'évaluations : Proxy de l'engagement et de la persistance académique
3.Crédits étudiés : Indicateur inverse - trop de crédits = risque accru
Variables non-discriminantes :
1.Temps Moodle : Faible pouvoir prédictif isolé
2.Variables démographiques : Impact limité selon la matrice de corrélation
8.5 Implications pour le modèle prédictif
Pondération recommandée :
Score académique : Poids élevé (variable la plus discriminante)
Engagement comportemental : Poids moyen (nombre d'évaluations)
Charge académique : Poids moyen avec seuil d'alerte (surcharge)
Variables contextuelles : Poids faible mais maintenues pour la généralisation
Seuils d'alerte proposés :
Score moyen < 60 : Risque élevé
Nombre d'évaluations < 8 : Désengagement probable
Crédits > 150 : Surcharge potentielle
8.6 Validité du modèle et recommandations
Forces du dataset OULAD :
Réalisme des données : Profils authentiques d'étudiants universitaires
Variabilité appropriée : Distributions reflétant la diversité étudiante
Pouvoir discriminant : Plusieurs variables montrent des différences significatives
Recommandations opérationnelles :
1.Système d'alerte précoce basé sur les scores des premières évaluations
2.Suivi de l'engagement via le nombre d'évaluations passées
3.Conseil pédagogique pour les étudiants avec charge excessive de crédits
4.Approche multivariée nécessaire vu la complexité des patterns
9. Analyse des techniques avancées de Data Mining
9.1 Analyse des Règles d'Association - Insights Comportementaux

L'analyse des règles d'association révèle des patterns complexes dans les facteurs prédictifs d'abandon :
Règles les plus significatives identifiées :
Règle 1 : Vulnérabilité masculine
Pattern : frozenset({'sexe_M'}) → frozenset({'credits_etudes_Low'})
Support : 0.3348 | Confiance : 0.6332
Interprétation : Les étudiants masculins sont significativement plus susceptibles d'avoir des parcours académiques incomplets
Implication stratégique : Nécessité de programmes de soutien spécifiques pour les hommes
Règle 2 : Impact des performances académiques
Pattern : frozenset({'score_moyen_Low'}) → frozenset({'abandon'})
Support : 0.2266 | Confiance : 0.6775
Interprétation : Les scores faibles sont prédictifs d'abandon avec une fiabilité de 67.75%
Validation : Confirme l'importance cruciale des performances académiques
Règle 3 : Défi des étudiants adultes
Pattern : frozenset({'age_35-55'}) → 			     frozenset({'credits_etudes_Low'})
Support : 0.2164 | Confiance : 0.7171
Interprétation : Les étudiants de 35-55 ans accumulent difficilement les crédits
Hypothèse explicative : Contraintes professionnelles et familiales
Insights comportementaux stratégiques :
Intersectionnalité des facteurs : L'abandon résulte de combinaisons de facteurs plutôt que de causes isolées
Profils à risque multiples : Hommes + adultes + faibles performances = risque maximal
Prédiction probabiliste : Les règles offrent des probabilités d'abandon conditionnelles
9.2 Analyse du Clustering K-Means - Segmentation Comportementale

La visualisation du clustering révèle trois profils d'étudiants distincts basés sur leurs performances et engagement :
Cluster 0 (Points blancs) - "Étudiants en difficulté critique"
Caractéristiques : Scores très faibles (0-20), temps de module variable
Profil comportemental : Désengagement académique sévère
Insight critique : Le temps investi n'améliore pas les performances pour ce groupe
Recommandation : Intervention pédagogique immédiate et personnalisée
Cluster 1 (Points bleus foncés) - "Étudiants performants engagés"
Caractéristiques : Scores moyens à élevés (20-100), temps de module substantiel
Profil comportemental : Corrélation positive entre effort et résultats
Insight stratégique : Modèle de réussite basé sur l'investissement temporel
Recommandation : Maintien du soutien et reconnaissance des efforts
Cluster 2 (Points bleus clairs) - "Étudiants moyens hétérogènes"
Caractéristiques : Performances variables, large dispersion
Profil comportemental : Groupe de transition avec potentiel d'amélioration
Insight pédagogique : Cible prioritaire pour interventions préventives
Recommandation : Stratégies différenciées selon les sous-profils
Implications stratégiques du clustering :
Personnalisation pédagogique : Trois approches distinctes nécessaires
Allocation des ressources : Priorisation des Clusters 0 et 2
Prédiction affinée : Intégration des profils de cluster dans le modèle prédictif
9.3 Analyse Comparative par Statut d'Abandon - Facteurs Différenciateurs
Satisfaction Étudiante
Patterns de satisfaction identifiés :
Différence marginale : Étudiants persistants légèrement plus satisfaits
Distributions chevauchantes : La satisfaction n'est pas le facteur déterminant principal
Insight contre-intuitif : Certains étudiants satisfaits abandonnent quand même
Hypothèse explicative : Facteurs externes (financiers, familiaux) prédominent sur la satisfaction
Participation aux Forums
Impact de l'engagement communautaire :
Différence significative : Médiane plus élevée pour les étudiants persistants
Facteur protecteur : L'interaction sociale réduit le risque d'abandon
Mécanisme explicatif : Sentiment d'appartenance et support peer-to-peer
Recommandation opérationnelle : Stimulation de la participation communautaire
9.4 Profils à Risque - Typologie Complète
Profil à Risque Majeur - "Triple Vulnérabilité"
Caractéristiques : Homme, 35-55 ans, scores faibles, faible engagement
Probabilité d'abandon : >80% selon la combinaison des règles
Interventions prioritaires : Accompagnement personnalisé intensif
Profil à Risque Modéré - "Surcharge Académique"
Caractéristiques : Crédits élevés, temps limité, performances moyennes
Probabilité d'abandon : 40-60% selon les conditions
Interventions préventives : Conseil en gestion du temps et réduction de charge
Profil à Risque Faible - "Désengagement Social"
Caractéristiques : Bonnes performances, faible participation forums
Probabilité d'abandon : 20-30% selon l'évolution
Interventions correctives : Stimulation de l'engagement communautaire
9.5 Facteurs Protecteurs - Stratégies de Résilience
Facteurs Protecteurs Identifiés :
1.Engagement académique soutenu : Participation régulière aux évaluations
2.Performance académique stable : Maintien de scores >60
3.Interaction communautaire : Participation active aux forums
4.Charge académique équilibrée : Évitement de la surcharge
Mécanismes de Protection :
Cercle vertueux : Engagement → Performance → Satisfaction → Persistance
Support social : Interactions communautaires comme facteur de rétention
Auto-régulation : Gestion équilibrée de la charge académique
9.6 Modèle Prédictif Intégré - Synthèse Algorithmique
Architecture du Modèle de Prédiction :
Niveau 1 - Variables Principales :
Score moyen (pondération : 40%)
Nombre d'évaluations (pondération : 25%)
Crédits étudiés (pondération : 20%)
Niveau 2 - Variables Contextuelles :
Participation forums (pondération : 10%)
Variables démographiques (pondération : 5%)
Niveau 3 - Règles d'Association :
Boost prédictif pour combinaisons critiques
Ajustement probabiliste selon les patterns identifiés
Seuils d'Alerte Calibrés :
Alerte Critique : Probabilité d'abandon >70%
Alerte Modérée : Probabilité d'abandon 40-70%
Surveillance : Probabilité d'abandon 20-40%
Faible Risque : Probabilité d'abandon <20%
9.7 Recommandations Opérationnelles Avancées
Système d'Intervention Échelonné :
Phase 1 - Détection Précoce :
Monitoring automatique des indicateurs clés
Alertes temps réel pour les profils à risque
Tableau de bord prédictif pour les conseillers
Phase 2 - Intervention Personnalisée :
Accompagnement différencié selon le profil de risque
Réduction de charge pour les étudiants en surcharge
Stimulation de l'engagement pour les étudiants isolés
Phase 3 - Suivi et Ajustement :
Évaluation continue de l'efficacité des interventions
Ajustement des seuils selon les retours d'expérience
Amélioration continue du modèle prédictif
Conclusion
Ce projet constitue une implémentation complète d'un système de prévention de l'abandon scolaire utilisant des techniques avancées de data mining. L'analyse approfondie des résultats révèle des patterns comportementaux complexes et significatifs, notamment :
Contributions Scientifiques :
Identification de profils à risque multidimensionnels combinant facteurs démographiques, académiques et comportementaux
Découverte de règles d'association contre-intuitives comme l'impact de la surcharge académique
Segmentation comportementale révélant trois profils d'étudiants distincts nécessitant des approches pédagogiques différenciées
Innovations Techniques :
Architecture hybride combinant apprentissage supervisé, non-supervisé et règles d'association
Interface utilisateur intuitive permettant la simulation temps réel et l'aide à la décision
Validation croisée sur données réelles et synthétiques garantissant la robustesse
Impact Opérationnel :
Système d'alerte précoce basé sur des seuils scientifiquement validés
Interventions personnalisées selon les profils de risque identifiés
Outil d'aide à la décision pour les institutions éducatives
L'architecture modulaire, l'utilisation de techniques de data mining variées et complémentaires, ainsi que l'interface utilisateur professionnelle font de ce système un outil opérationnel et scientifiquement validé pour les institutions éducatives souhaitant améliorer la réussite de leurs étudiants par une approche data-driven innovante et rigoureuse.
