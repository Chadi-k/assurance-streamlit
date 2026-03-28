import os
import pickle
import streamlit as st

base_path = os.path.dirname(os.path.dirname(__file__))
models_path = os.path.join(base_path, "models")

st.title("🔮 Prédiction")

tfidf = pickle.load(open(os.path.join(models_path, "tfidf_vectorizer.pkl"), "rb"))
model = pickle.load(open(os.path.join(models_path, "best_model.pkl"), "rb"))
themes = pickle.load(open(os.path.join(models_path, "themes.pkl"), "rb"))

text = st.text_area("Avis :", height=150)

if text and st.button("Prédire"):
    X = tfidf.transform([text.lower()])
    pred = model.predict(X)[0]
    st.markdown(f"### Note prédite : {int(pred)}/5")

    if hasattr(model, "predict_proba"):
        for i, p in enumerate(model.predict_proba(X)[0]):
            st.progress(float(p), text=f"{i+1}★ : {p*100:.1f}%")

    for n, info in themes.items():
        found = [k for k in info["mots_cles"] if k in text.lower()]
        if found:
            st.success(f"🏷️ {n}: {', '.join(found)}")
