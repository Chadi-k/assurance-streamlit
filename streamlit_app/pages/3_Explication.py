import streamlit as st
import pickle
import numpy as np

st.title("🔍 Explication")

tfidf = pickle.load(open("models/tfidf_vectorizer.pkl", "rb"))
model = pickle.load(open("models/best_model.pkl", "rb"))

text = st.text_area("Avis :", height=150)

if text and st.button("Analyser"):
    X = tfidf.transform([text.lower()])
    pred = model.predict(X)[0]
    st.markdown(f"### Prédiction : {int(pred)}/5")

    if hasattr(model, "coef_"):
        feats = tfidf.get_feature_names_out()
        ci = int(pred) - 1
        coefs = model.coef_[ci] if model.coef_.shape[0] > 1 else model.coef_[0]
        contribs = X.toarray()[0] * coefs

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**✅ Mots positifs :**")
            for i in contribs.argsort()[-10:][::-1]:
                if contribs[i] > 0:
                    st.write(f"+ {feats[i]} ({contribs[i]:.3f})")
        with c2:
            st.markdown("**❌ Mots négatifs :**")
            for i in contribs.argsort()[:10]:
                if contribs[i] < 0:
                    st.write(f"- {feats[i]} ({contribs[i]:.3f})")
