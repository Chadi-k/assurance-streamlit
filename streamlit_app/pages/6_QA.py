import streamlit as st
import pandas as pd
from collections import Counter

st.title("❓ QA")

df = pd.read_csv("data/train_clean.csv")

q = st.text_input("Question :")

if q and st.button("Répondre"):
    ql = q.lower()

    if "meilleur" in ql:
        top = df.groupby("assureur")["note"].agg(["mean", "count"])
        st.dataframe(
            top[top["count"] >= 20].sort_values("mean", ascending=False).head(10)
        )
    elif "pire" in ql:
        bot = df.groupby("assureur")["note"].agg(["mean", "count"])
        st.dataframe(
            bot[bot["count"] >= 20].sort_values("mean").head(10)
        )
    elif "produit" in ql:
        st.dataframe(
            df.groupby("produit")["note"]
            .agg(["mean", "count"])
            .sort_values("mean", ascending=False)
        )
    else:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        tv = TfidfVectorizer(max_features=5000)
        mx = tv.fit_transform(df["avis"].fillna(""))
        s = cosine_similarity(tv.transform([q]), mx).flatten()
        for idx in s.argsort()[-5:][::-1]:
            r = df.iloc[idx]
            st.markdown(
                f"- {int(r['note'])}/5 **{r['assureur']}**: "
                f"{str(r['avis'])[:200]}"
            )
