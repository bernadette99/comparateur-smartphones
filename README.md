# Comparateur de Smartphones - Analyse de Sentiment par IA

Dashboard interactif pour l'analyse de sentiment des avis de smartphones. Ce projet utilise des modèles de Machine Learning (FastText et SVM) pour classifier et visualiser les points forts et faibles de différents modèles à partir de données utilisateurs.

**➡️ [Voir le site en direct](https://bernadette99.github.io/comparateur-smartphones/)** ## ✨ Fonctionnalités

* **Analyse de Sentiment par IA** : Classification des avis en Positif, Négatif ou Neutre avec un modèle FastText + SVM.
* **Dashboard Interactif** : Visualisation des données par téléphone avec des jauges, graphiques et nuages de mots.
* **Analyses Globales** : Comparaison des marques et modèles sur la base des notes, du prix et du sentiment général.
* **Génération d'Assets** : Le projet inclut les scripts Python pour effectuer l'analyse et générer tous les graphiques.

## 🛠️ Technologies utilisées

* **Analyse de Données & IA** : Python, Pandas, Scikit-learn, Gensim (FastText), NLTK.
* **Front-End (Site Web)** : HTML5, CSS3, JavaScript.

## 🚀 Comment lancer ce projet localement

1.  Clonez ce dépôt.
2.  Créez un environnement virtuel (`conda create --name .venv python=3.12`).
3.  Activez l'environement virtuel (`conda activate ia-projet`)
3.  Installez les dépendances (`pip install -r requirements.txt`).
4.  Exécutez le script d'analyse (`python fastText_SVM.py`).
5.  Ouvrez le fichier `index.html` dans votre navigateur.