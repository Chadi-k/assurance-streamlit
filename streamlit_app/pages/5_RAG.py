import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("🤖 RAG — Retrieval-Augmented Generation")

st.markdown("""
### À quoi sert cette page ?
Le **RAG** combine recherche d'information et génération de texte :
1. **Retrieval** : on retrouve les 5 avis les plus pertinents pour votre question
2. **Augmented** : ces avis servent de contexte
3. **Generation** : un prompt est généré pour un LLM (GPT, Claude, etc.)

Copiez le prompt généré dans un LLM pour obtenir une réponse naturelle.
""")

df = pd.read_csv("data/train_clean.csv")
tfidf = TfidfVectorizer(max_features=10000)
matrix = tfidf.fit_transform(df["avis"].fillna(""))

q = st.text_input("❓ Posez votre question :",
    placeholder="Ex: Que pensent les clients du remboursement chez Direct Assurance ?")

if q and st.button("🔍 Rechercher et générer", type="primary"):
    sims = cosine_similarity(tfidf.transform([q]), matrix).flatten()
    top = sims.argsort()[-5:][::-1]

    st.markdown("### 📚 Avis retrouvés")
    ctx = []
    for i, idx in enumerate(top):
        r = df.iloc[idx]
        ctx.append(f"[Note:{int(r['note'])}/5] {r['avis']}")
        with st.expander(f"Doc {i+1} — {'⭐'*int(r['note'])} {r['assureur']} ({r['produit']})"):
            st.write(r["avis"])

    notes = [df.iloc[idx]["note"] for idx in top]
    avg = np.mean(notes)
    st.info(f"📊 Note moyenne des avis pertinents : **{avg:.1f}/5** — "
            f"Sentiment : {'positif ✅' if avg >= 3.5 else 'mitigé ⚠️' if avg >= 2.5 else 'négatif ❌'}")

    prompt = "Contexte (avis clients):
" + "
".join(ctx) + f"

Question: {q}
Réponse:"
    st.markdown("### 💬 Prompt à copier dans un LLM")
    st.code(prompt[:2000], language="text")
