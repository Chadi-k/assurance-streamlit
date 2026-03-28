import streamlit as st
import pandas as pd
from collections import Counter

st.title("📝 Résumé par Assureur")

df = pd.read_csv("data/train_clean.csv")
assureur = st.selectbox("Assureur :", sorted(df["assureur"].unique()))
s = df[df["assureur"] == assureur]

c1, c2, c3 = st.columns(3)
c1.metric("Nb avis", len(s))
c2.metric("Note moy.", f"{s['note'].mean():.1f}/5")
c3.metric("% Positifs", f"{(s['note']>=4).mean()*100:.0f}%")

st.bar_chart(s["note"].value_counts().sort_index())

if "avis_clean2" in s.columns:
    w = Counter(" ".join(s["avis_clean2"].dropna()).split()).most_common(20)
    st.dataframe(pd.DataFrame(w, columns=["Mot", "Fréquence"]))
