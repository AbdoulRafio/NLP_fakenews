# ============================================================================================== #
# Auteur : BANGAGNE Abdoul Rafio
# Utiliser directement l'environnement installé sur Anaconda (nlp_env)
# cd nom_dossier pour se déplacer dans un dossier 
# Pour lancer l'application : streamlit run app.py
# ============================================================================================== #

# Importation des packages
import streamlit as st
import pickle
import re
import unicodedata
import string
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Chargement des objets sauvegardés
with open("models/sentiment_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Fonction de nettoyage
def clean_tweet(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'http\S+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation + "’‘“”"))
    text = re.sub(r'\d+', '', text)

    mots_inutiles = [
        'zzzzaaaacccchhh', 'aaa', 'aa', 'chelseahandlerhumanurinalthis', 
        'pictwittercombklebyz', 'httpsctcoafhpshflm', 'herehttpsyoutubeainwckby', 
        'pictwittercomgxoytep', 'whgovjointpressconf', 'trump', 'transparency',
        'httpstcoqyrmcw', 'émigré', 'zuniga', 'zune', 'zumwaltclass', 'zurich', 
        'zverse', 'zvia', 'zweiman', 'zulloinsider', 'zukunfteuropaa', 'zuckerberg'
    ]
    for mot in mots_inutiles:
        text = re.sub(rf'\b{re.escape(mot)}\b', '', text)
    text = re.sub(r'\b[a-zA-Z]\b', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Interface utilisateur
st.title("Fake News Detector – Analyse de la Fiabilité des Articles")

user_input = st.text_area("Entrez un texte à analyser :", height=200)

if st.button("Prédire la fiabilité"):
    if user_input.strip() == "":
        st.warning("Veuillez entrer un texte.")
    else:
        clean = clean_tweet(user_input)
        vect = vectorizer.transform([clean])
        prediction = model.predict(vect)[0]

        label_map = {0: "🔴😞 Fake News", 1: "🟢😊 Real News"}
        st.success(f"Résultat : **{label_map.get(prediction, 'Inconnu')}**")

        # Génération du WordCloud
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(clean)
        st.subheader("🔍 Nuage de mots du texte analysé :")
        fig, ax = plt.subplots()
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)

# ✅ Extra UI enhancements
st.markdown("---")
st.markdown("### ✨ Why use this app?")
st.markdown("- 🔥 **Instant Fake News Detection** for your tweets!")
st.markdown("- 🎨 **Beautiful & Minimal UI** for easy interaction.")
st.markdown("- 🚀 **Fast & Efficient** model built with TF-IDF & Logistic Regression.")
st.markdown("---")
st.markdown("💡 *Built with ❤️ using Streamlit and Machine Learning!* ✨")
