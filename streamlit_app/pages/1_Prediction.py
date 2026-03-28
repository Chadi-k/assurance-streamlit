import streamlit as st
import pickle

st.title("🔮 Prédiction")

tfidf = pickle.load(open("models/tfidf_vectorizer.pkl", "rb"))
model = pickle.load(open("models/best_model.pkl", "rb"))
themes = pickle.load(open("models/themes.pkl", "rb"))

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
