import os
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

base_path = os.path.dirname(os.path.dirname(__file__))
data_path = os.path.join(base_path, "data")

st.title("🔎 Recherche d'Avis")

df = pd.read_csv(os.path.join(data_path, "train_clean.csv"))
tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
matrix = tfidf.fit_transform(df["avis"].fillna(""))

query = st.text_input("Rechercher :")

if query:
    sims = cosine_similarity(tfidf.transform([query]), matrix).flatten()
    for rank, idx in enumerate(sims.argsort()[-10:][::-1]):
        if sims[idx] > 0:
            r = df.iloc[idx]
            with st.expander(
                f"#{rank+1} | {int(r['note'])}/5 | "
                f"{r['assureur']} ({sims[idx]:.3f})"
            ):
                st.write(r["avis"])
