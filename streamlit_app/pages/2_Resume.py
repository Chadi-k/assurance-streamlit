import os
import streamlit as st
import pandas as pd
from collections import Counter

base_path = os.path.dirname(os.path.dirname(__file__))
data_path = os.path.join(base_path, "data")

st.title("📝 Analyse par Assureur")

st.markdown("""
### À quoi sert cette page ?
Sélectionnez un assureur pour voir :
- Ses **métriques clés** (nombre d'avis, note moyenne, % de clients satisfaits)
- La **distribution des notes**
- Les **mots les plus fréquents** dans les avis
""")

df = pd.read_csv(os.path.join(data_path, "train_clean.csv"))
assureur = st.selectbox("🏢 Choisissez un assureur :", sorted(df["assureur"].dropna().unique()))
s = df[df["assureur"] == assureur]

st.markdown("---")
c1, c2, c3, c4 = st.columns(4)
c1.metric("📊 Nb avis", len(s))
c2.metric("⭐ Note moy.", f"{s['note'].mean():.1f}/5")
c3.metric("😊 % Positifs", f"{(s['note']>=4).mean()*100:.0f}%")
c4.metric("😠 % Négatifs", f"{(s['note']<=2).mean()*100:.0f}%")

st.markdown("### 📊 Distribution des notes")
st.bar_chart(s["note"].value_counts().sort_index())

if "avis_clean2" in s.columns:
    st.markdown("### 🔤 Mots les plus fréquents")
    w = Counter(" ".join(s["avis_clean2"].dropna().astype(str)).split()).most_common(20)
    st.dataframe(pd.DataFrame(w, columns=["Mot", "Fréquence"]), use_container_width=True)

st.markdown("### 📋 Exemples d'avis")
for note in [5, 1]:
    sample = s[s["note"] == note]["avis"].head(2)
    for a in sample:
        st.markdown(f"{'⭐'*note} _{str(a)[:200]}_")
