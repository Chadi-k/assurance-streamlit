import os
import pickle
import numpy as np
import streamlit as st

base_path = os.path.dirname(os.path.dirname(__file__))
models_path = os.path.join(base_path, "models")

st.title("🔍 Explication des Prédictions")

st.markdown("""
### À quoi sert cette page ?
Comprenez **pourquoi** le modèle a attribué une note à un avis.
Les mots sont classés par leur **contribution** à la prédiction :
- **Mots positifs** : poussent la note vers le haut
- **Mots négatifs** : poussent la note vers le bas
""")

tfidf = pickle.load(open(os.path.join(models_path, "tfidf_vectorizer.pkl"), "rb"))
model = pickle.load(open(os.path.join(models_path, "best_model.pkl"), "rb"))

text = st.text_area("✏️ Entrez un avis à expliquer :", height=150)

if text and st.button("🔬 Expliquer", type="primary"):
    X = tfidf.transform([text.lower()])
    pred = model.predict(X)[0]
    st.markdown(f"### Prédiction : {'⭐' * int(pred)} ({int(pred)}/5)")

    st.markdown("---")
    if hasattr(model, "coef_"):
        feats = tfidf.get_feature_names_out()
        ci = int(pred) - 1
        coefs = model.coef_[ci] if model.coef_.shape[0] > 1 else model.coef_[0]
        contribs = X.toarray()[0] * coefs

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### ✅ Mots qui favorisent cette note")
            for i in contribs.argsort()[-10:][::-1]:
                if contribs[i] > 0:
                    st.markdown(f"**+{contribs[i]:.3f}** `{feats[i]}`")
        with c2:
            st.markdown("#### ❌ Mots qui défavorisent cette note")
            for i in contribs.argsort()[:10]:
                if contribs[i] < 0:
                    st.markdown(f"**{contribs[i]:.3f}** `{feats[i]}`")
