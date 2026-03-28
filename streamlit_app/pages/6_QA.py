import os
import streamlit as st
import pandas as pd
from collections import Counter

base_path = os.path.dirname(os.path.dirname(__file__))
data_path = os.path.join(base_path, "data")

st.title("❓ Questions-Réponses sur les Avis")

st.markdown("""
### À quoi sert cette page ?
Posez des questions sur la base de données d'avis clients. Le système répond automatiquement
en analysant les données : classements, statistiques, recherche d'avis pertinents.
""")

df = pd.read_csv(os.path.join(data_path, "train_clean.csv"))

questions = [
    "Quel est le meilleur assureur ?",
    "Quel est le pire assureur ?",
    "Quels sont les principaux problèmes ?",
    "Quel produit est le mieux noté ?",
    "Combien d'avis mentionnent le remboursement ?",
    "Autre question..."
]

sel = st.selectbox("📋 Choisissez une question :", questions)
if sel == "Autre question...":
    q = st.text_input("Votre question :")
else:
    q = sel

if q and st.button("💡 Répondre", type="primary"):
    ql = q.lower()
    st.markdown("---")

    if "meilleur" in ql and "assureur" in ql:
        st.markdown("### 🏆 Assureurs les mieux notés (min 20 avis)")
        top = df.groupby("assureur")["note"].agg(["mean", "count"])
        st.dataframe(
            top[top["count"] >= 20].sort_values("mean", ascending=False).head(10),
            use_container_width=True
        )
    elif "pire" in ql:
        st.markdown("### ⚠️ Assureurs les moins bien notés (min 20 avis)")
        bot = df.groupby("assureur")["note"].agg(["mean", "count"])
        st.dataframe(
            bot[bot["count"] >= 20].sort_values("mean").head(10),
            use_container_width=True
        )
    elif "probleme" in ql or "problème" in ql:
        st.markdown("### 🔴 Mots les plus fréquents dans les avis négatifs (1-2★)")
        neg = df[df["note"] <= 2]
        words = Counter(" ".join(neg["avis"].dropna().astype(str).str.lower()).split()).most_common(20)
        st.dataframe(pd.DataFrame(words, columns=["Mot", "Fréquence"]), use_container_width=True)
    elif "produit" in ql:
        st.markdown("### 📦 Notes moyennes par produit")
        st.dataframe(
            df.groupby("produit")["note"].agg(["mean", "count"])
            .sort_values("mean", ascending=False),
            use_container_width=True
        )
    elif "remboursement" in ql:
        rembours = df[df["avis"].astype(str).str.contains("rembours", case=False, na=False)]
        st.markdown(f"### 💰 {len(rembours)} avis mentionnent le remboursement")
        st.metric("Note moyenne", f"{rembours['note'].mean():.2f}/5")
    else:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        st.markdown("### 🔎 Avis les plus pertinents")
        tv = TfidfVectorizer(max_features=5000)
        mx = tv.fit_transform(df["avis"].fillna(""))
        s = cosine_similarity(tv.transform([q]), mx).flatten()
        for idx in s.argsort()[-5:][::-1]:
            r = df.iloc[idx]
            st.markdown(f"- {'⭐'*int(r['note'])} **{r['assureur']}**: {str(r['avis'])[:200]}")
