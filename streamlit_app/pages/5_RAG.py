import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("🤖 RAG")

df = pd.read_csv("data/train_clean.csv")
tfidf = TfidfVectorizer(max_features=10000)
matrix = tfidf.fit_transform(df["avis"].fillna(""))

q = st.text_input("Question :")

if q and st.button("Générer"):
    sims = cosine_similarity(tfidf.transform([q]), matrix).flatten()
    top = sims.argsort()[-5:][::-1]
    ctx = []

    for i, idx in enumerate(top):
        r = df.iloc[idx]
        ctx.append(f"[Note:{int(r['note'])}/5] {r['avis']}")
        with st.expander(f"Doc {i+1} - {r['assureur']}"):
            st.write(r["avis"])

    prompt = "Contexte:
" + "
".join(ctx) + f"

Question: {q}
Réponse:"
    st.code(prompt[:1000])
    notes = [df.iloc[idx]["note"] for idx in top]
    st.info(f"Note moyenne : {np.mean(notes):.1f}/5")
