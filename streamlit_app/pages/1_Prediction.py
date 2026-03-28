import os
import pickle
import streamlit as st

base_path = os.path.dirname(os.path.dirname(__file__))
models_path = os.path.join(base_path, "models")

st.title("🔮 Prédiction de Note et Détection de Thèmes")

st.markdown("""
### Comment ça marche ?
Entrez un avis client en français. Le modèle va :
1. **Prédire la note** (1 à 5 étoiles) avec un modèle de Régression Logistique entraîné sur TF-IDF
2. **Détecter les thèmes** abordés (prix, service client, sinistre, etc.) par recherche de mots-clés
""")

tfidf = pickle.load(open(os.path.join(models_path, "tfidf_vectorizer.pkl"), "rb"))
model = pickle.load(open(os.path.join(models_path, "best_model.pkl"), "rb"))
themes = pickle.load(open(os.path.join(models_path, "themes.pkl"), "rb"))

text = st.text_area(
    "✏️ Entrez un avis client :",
    height=150,
    placeholder="Ex: Le service client est très réactif, j'ai été remboursé rapidement..."
)

if text and st.button("🚀 Analyser", type="primary"):
    X = tfidf.transform([text.lower()])
    pred = model.predict(X)[0]

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"### Note prédite : {'⭐' * int(pred)} ({int(pred)}/5)")
        if hasattr(model, "predict_proba"):
            st.markdown("**Probabilités par classe :**")
            for i, p in enumerate(model.predict_proba(X)[0]):
                st.progress(float(p), text=f"{i+1}★ : {p*100:.1f}%")

    with col2:
        st.markdown("### 🏷️ Thèmes détectés :")
        found_any = False
        for n, info in themes.items():
            found = [k for k in info["mots_cles"] if k in text.lower()]
            if found:
                found_any = True
                st.success(f"**{n}** : {', '.join(found)}")
        if not found_any:
            st.info("Aucun thème spécifique détecté.")
