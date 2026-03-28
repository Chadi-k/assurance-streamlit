import os
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

base_path = os.path.dirname(os.path.dirname(__file__))
data_path = os.path.join(base_path, "data")

st.title("🔎 Recherche Sémantique d'Avis")

st.markdown("""
### À quoi sert cette page ?
Recherchez des avis par **mots-clés**. Le système utilise la similarité TF-IDF/cosinus
pour retrouver les avis les plus pertinents, avec des filtres par assureur et par note.
""")

df = pd.read_csv(os.path.join(data_path, "train_clean.csv"))
tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
matrix = tfidf.fit_transform(df["avis"].fillna(""))

query = st.text_input("🔍 Rechercher :", placeholder="Ex: remboursement dentaire, accident voiture...")

col1, col2 = st.columns(2)
with col1:
    f_assureur = st.multiselect("Filtrer par assureur :", sorted(df["assureur"].dropna().unique()))
with col2:
    f_note = st.slider("Filtrer par note :", 1, 5, (1, 5))

if query:
    sims = cosine_similarity(tfidf.transform([query]), matrix).flatten()
    mask = (df["note"] >= f_note[0]) & (df["note"] <= f_note[1])
    if f_assureur:
        mask &= df["assureur"].isin(f_assureur)
    sims[~mask.values] = 0

    st.markdown(f"### 📋 Top 10 résultats pour *\"{query}\"*")
    for rank, idx in enumerate(sims.argsort()[-10:][::-1]):
        if sims[idx] > 0:
            r = df.iloc[idx]
            with st.expander(
                f"#{rank+1} | {'⭐'*int(r['note'])} | "
                f"{r['assureur']} — {r['produit']} (score: {sims[idx]:.3f})"
            ):
                st.write(r["avis"])

    st.markdown("---")
    st.markdown("### 📊 Métriques globales")
    metrics = df.groupby("assureur")["note"].agg(["count", "mean"]).sort_values("count", ascending=False).head(15)
    metrics.columns = ["Nb avis", "Note moyenne"]
    st.dataframe(metrics.style.format({"Note moyenne": "{:.2f}"}), use_container_width=True)
