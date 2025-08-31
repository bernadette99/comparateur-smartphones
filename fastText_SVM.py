import pandas as pd
import numpy as np
import re
import os
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from gensim.models import FastText
from sklearn.svm import SVC
from wordcloud import WordCloud
from tqdm import tqdm


# --- ÉTAPE 1: PRÉPARATION DU TEXTE ---

nltk.download('punkt_tab')
nltk.download('stopwords')
# On s'assure que les outils NLTK sont disponibles
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    print("🧠 Téléchargement des outils NLTK (punkt, stopwords)...")
    

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """Nettoie et segmente le texte en une liste de mots (tokens)."""
    text = str(text).lower()
    text = re.sub(r'\W|\d', ' ', text)
    tokens = word_tokenize(text)
    return [word for word in tokens if word not in stop_words and len(word) > 1]

# --- ÉTAPE 2: ANALYSE DE SENTIMENT AVEC FASTTEXT + SVM ---

def analyze_sentiments_fasttext_svm(df):
    """Analyse les sentiments en utilisant FastText pour la vectorisation et SVM pour la classification."""
    print("🧠 Préparation et entraînement du modèle IA (FastText + SVM)...")

    # 2a. Prétraitement du texte
    df['tokens'] = df['review_text'].apply(preprocess_text)

    # 2b. Entraînement du modèle FastText
    print("  -> Entraînement du modèle FastText...")
    ft_model = FastText(sentences=df['tokens'], vector_size=100, window=5, min_count=1, workers=4)

    # 2c. Vectorisation des avis avec le modèle FastText entraîné
    print("  -> Vectorisation des avis...")
    vectors = []
    for review_tokens in tqdm(df['tokens'], desc="Vectorisation"):
        word_vectors = [ft_model.wv[word] for word in review_tokens if word in ft_model.wv]
        if not word_vectors:
            vectors.append(np.zeros(ft_model.vector_size))
        else:
            vectors.append(np.mean(word_vectors, axis=0))
    X = np.array(vectors)

    # 2d. Préparation des étiquettes (labels)
    def map_rating_to_sentiment(rating):
        if rating <= 2: return 'Négatif'
        elif rating == 3: return 'Neutre'
        else: return 'Positif'
    y = df['star_rating'].apply(map_rating_to_sentiment)

    # 2e. Entraînement du modèle SVM
    print("  -> Entraînement du modèle SVM...")
    svm_model = SVC(kernel='rbf', random_state=42) # Modèle SVM avec noyau linear donne de mauvais résultats(j'ai testé)
    svm_model.fit(X, y)
    
    # 2f. Prédire le sentiment pour l'ensemble des données
    df['ai_sentiment'] = svm_model.predict(X)
    print("✅ Analyse de sentiment par l'IA terminée.")
    return df



# --- ÉTAPE 3: GÉNÉRATION DES WORD CLOUDS ---
def generate_wordclouds(df_analyzed, output_dir):
    print("\n🎨 Génération des Word Clouds...")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir); print(f"Dossier '{output_dir}' créé.")
    
    stopwords = set(WordCloud().stopwords)
    custom_stopwords = {'gb', 'mp', 'mah', 'inch', 'phone', 'smartphone'}
    stopwords.update(custom_stopwords)

    for model_name in df_analyzed['model_name'].unique():
        for sentiment in ['Positif', 'Négatif']:
            subset = df_analyzed[(df_analyzed['model_name'] == model_name) & (df_analyzed['ai_sentiment'] == sentiment)]
            if not subset.empty:
                text = " ".join(review for review in subset.review_text.astype(str))
                if text.strip():
                    wc = WordCloud(stopwords=stopwords, background_color="white", max_words=50, colormap='viridis' if sentiment == 'Positif' else 'plasma_r').generate(text)
                    filename = f"wordcloud_{model_name.replace(' ', '_')}_{sentiment}_ai.png"
                    filepath = os.path.join(output_dir, filename)
                    wc.to_file(filepath)
                    print(f"  -> Word Cloud sauvegardé : {filepath}")
    print("✅ Génération des Word Clouds terminée.")



# --- ÉTAPE 4: GÉNÉRATION DES GRAPHIQUES D'ANALYSE ---
def generate_plots(df_analyzed, output_dir):
    """Génère et sauvegarde les 4 graphiques d'analyse globale """
    print("\n📊 Génération des graphiques d'analyse globale...")
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'sans-serif'] 



    # --- 1. Comparaison par Marque ---
    avg_rating_brand = df_analyzed.groupby('brand')['star_rating'].mean().sort_values(ascending=False).reset_index()
    plt.figure(figsize=(10, 5))
    ax = sns.barplot(x='star_rating', y='brand', data=avg_rating_brand, palette='plasma', orient='h')
    
    
    ax.set_title("Quelle marque a la meilleure note moyenne ?", fontsize=16, weight='bold', pad=20)
    plt.suptitle("Ce graphique compare la satisfaction globale des utilisateurs pour chaque marque.", fontsize=12, y=0.92)
    ax.set_xlabel("Note Moyenne Donnée par les Utilisateurs (sur 5)", fontsize=12)
    ax.set_ylabel("Marque", fontsize=12)
    
    ax.set_xlim(0, 5)
    for index, value in enumerate(avg_rating_brand['star_rating']):
        ax.text(value + 0.05, index, f'{value:.2f}', va='center', weight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.9]) # Ajuster pour le sous-titre
    plt.savefig(os.path.join(output_dir, 'average_rating_by_brand.png'))
    plt.close()
    print("  -> Graphique 'Note par Marque' sauvegardé.")



    # --- 2. Comparaison par Modèle ---
    avg_rating_model = df_analyzed.groupby('model_name')['star_rating'].mean().sort_values(ascending=False).reset_index()
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x='star_rating', y='model_name', data=avg_rating_model, palette='viridis', orient='h')
    
    
    ax.set_title("Quel est le modèle de téléphone le mieux noté ?", fontsize=16, weight='bold', pad=20)
    plt.suptitle("Ce graphique compare la satisfaction des utilisateurs pour chaque modèle spécifique.", fontsize=12, y=0.92)
    ax.set_xlabel("Note Moyenne Donnée par les Utilisateurs (sur 5)", fontsize=12)
    ax.set_ylabel("Modèle", fontsize=12)
    
    ax.set_xlim(0, 5)
    for index, value in enumerate(avg_rating_model['star_rating']):
        ax.text(value + 0.05, index, f'{value:.2f}', va='center', weight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.savefig(os.path.join(output_dir, 'average_rating_by_model.png'))
    plt.close()
    print("  -> Graphique 'Note par Modèle' sauvegardé.")



    # --- 3. Distribution des Sentiments par Note ---
    plt.figure(figsize=(12, 7))
    ax = sns.countplot(data=df_analyzed, x='star_rating', hue='ai_sentiment', order=[1, 2, 3, 4, 5], 
                       palette={'Positif': '#28a745', 'Négatif': '#dc3545', 'Neutre': '#6c757d'})
    
   
    ax.set_title("Une note élevée garantit-elle un commentaire positif ?", fontsize=16, weight='bold', pad=20)
    plt.suptitle("Ce graphique montre si les commentaires avec 1, 2, 3, 4 ou 5 étoiles sont jugés Positifs, Négatifs ou Neutres par l'IA.", fontsize=12, y=0.92)
    ax.set_xlabel("Note en Étoiles Donnée par l'Utilisateur", fontsize=12)
    ax.set_ylabel("Nombre de Commentaires", fontsize=12)
    ax.legend(title='Sentiment Prédit par l\'IA')
    
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', xytext=(0, 5), textcoords='offset points', weight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.savefig(os.path.join(output_dir, 'sentiment_distribution_by_rating.png'))
    plt.close()
    print("  -> Graphique 'Distribution des Sentiments' sauvegardé.")



    # --- 4. Prix vs. Note---
    plt.figure(figsize=(13, 8))
    ax = sns.scatterplot(data=df_analyzed, x='price', y='star_rating', hue='brand', 
                         size='units_sold', sizes=(200, 2000), 
                         palette='bright', alpha=0.8, edgecolor='black', linewidth=1)

    # Création d'un dataframe avec des modèles uniques pour les annotations
    unique_models_df = df_analyzed.drop_duplicates(subset='model_name')

    # Itération sur ce dataframe propre pour placer le texte
    for i, row in unique_models_df.iterrows():
        plt.text(x=row['price'] + 5, 
                 y=row['star_rating'], 
                 s=row['model_name'], 
                 fontdict=dict(color='black', size=10, weight='bold'))

    ax.set_title("Chaque bulle est un téléphone : Prix, Satisfaction et Volume de Ventes", fontsize=16, weight='bold', pad=20)
    plt.suptitle("Ce graphique montre si les téléphones plus chers sont mieux notés. La taille de la bulle représente son succès commercial.", fontsize=12, y=0.92)
    ax.set_xlabel("Prix du Téléphone (€)", fontsize=12)
    ax.set_ylabel("Note Moyenne Donnée par les Utilisateurs", fontsize=12)
    
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[:4], labels=labels[:4], title='Marque')

    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.savefig(os.path.join(output_dir, 'price_vs_rating_scatter.png'))
    plt.close()
    print("  -> Graphique 'Prix vs. Note' sauvegardé.")
    print("✅ Génération des graphiques .")


# --- ÉTAPE 5: GÉNÉRATION DU CODE JAVASCRIPT ---
def generate_javascript_code(df_analyzed):
    print("\n" + "="*50); print("📋 COPIEZ LE CODE CI-DESSOUS DANS VOTRE FICHIER script.js 📋"); print("="*50 + "\n")
    print("const phoneData = {")
    for model_name in sorted(df_analyzed['model_name'].unique()):
        model_subset = df_analyzed[df_analyzed['model_name'] == model_name]
        specs = model_subset.iloc[0]
        sentiment_counts = model_subset['ai_sentiment'].value_counts(normalize=True) * 100
        js_object = f"""
    "{model_name}": {{
        brand: "{specs['brand']}", score: {model_subset['star_rating'].mean():.2f},
        sentiments: {{ positive: {sentiment_counts.get("Positif", 0):.0f}, negative: {sentiment_counts.get("Négatif", 0):.0f}, neutral: {sentiment_counts.get("Neutre", 0):.0f} }},
        specs: {{
            screen: "{specs['screen_size']:.1f}\\"", battery: "{specs['battery']} mAh", camera: "{specs['camera_main']}",
            price: "{specs['price']}€", ram: "{specs['ram']} GB RAM", storage: "{specs['storage']} GB",
            has_5g: "{'Oui' if specs['has_5g'] else 'Non'}",
            water_resistant: "{specs['water_resistant'] if pd.notna(specs['water_resistant']) else 'Non spécifié'}"
        }},
        positiveCloud: "./images/wordcloud_{model_name.replace(' ', '_')}_Positif_ai.png",
        negativeCloud: "./images/wordcloud_{model_name.replace(' ', '_')}_Négatif_ai.png"
    }},"""
        print(js_object)
    print("};"); print("\n" + "="*50)

# --- SCRIPT PRINCIPAL ---
if __name__ == "__main__":
    data_path = 'Donnees/smartphone_reviews_sales_data.csv'
    output_dir = './images'

    try:
        df = pd.read_csv(data_path)
        print(f"✅ Fichier '{data_path}' chargé.")

        # 1. Analyse des données avec FastText + SVM
        df_analyzed = analyze_sentiments_fasttext_svm(df)

        # Enregistrement des résultats de l'analyse pour vérification
        df_analyzed.to_csv(os.path.join("Donnees", 'smartphone_analysis_results.csv'), index=False)
        print(f"✅ Résultats de l'analyse enregistrés dans 'Donnees/smartphone_analysis_results.csv'.")

        # 2. Création des Word Clouds
        generate_wordclouds(df_analyzed, output_dir=output_dir)

        # 3. Création des graphiques d'analyse
        generate_plots(df_analyzed, output_dir=output_dir)

        # 4. Génération du code pour le site web
        generate_javascript_code(df_analyzed)

    except FileNotFoundError:
        print(f"❌ ERREUR : Le fichier '{data_path}' est introuvable.")
    except Exception as e:
        print(f"❌ Une erreur inattendue est survenue : {e}")