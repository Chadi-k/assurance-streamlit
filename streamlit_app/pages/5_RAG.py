import os
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

base_path = os.path.dirname(os.path.dirname(__file__))
data_path = os.path.join(base_path, "data")
models_path = os.path.join(base_path, "models")

st.title("🤖 RAG — Retrieval-Augmented Generation")

st.markdown("""
### À quoi sert cette page ?
Le **RAG** combine recherche d'information et génération de texte :
1. **Détection** : identification de l'assureur et du thème dans votre question
2. **Retrieval** : recherche des avis pertinents filtrés par assureur et thème
3. **Generation** : un LLM génère une réponse basée sur les avis retrouvés
""")

# --- Charger données et thèmes ---
df = pd.read_csv(os.path.join(data_path, "train_clean.csv"))
try:
    themes = pickle.load(open(os.path.join(models_path, "themes.pkl"), "rb"))
except:
    themes = {}

# --- Fonctions utilitaires ---
def detect_assureur(question, assureurs_list):
    q_lower = question.lower()
    for assureur in assureurs_list:
        if assureur.lower() in q_lower:
            return assureur
        variants = assureur.lower().replace("'", " ").replace("-", " ").split()
        for v in variants:
            if len(v) > 3 and v in q_lower:
                return assureur
    return None

def detect_theme(question):
    q_lower = question.lower()
    detected = []
    theme_keywords_map = {
        "prix": ["prix", "tarif", "cher", "coût", "cotisation", "augmentation"],
        "service_client": ["service client", "téléphone", "conseiller", "contact", "joignable"],
        "remboursement": ["remboursement", "rembourser", "prise en charge", "soins", "dentaire"],
        "sinistre": ["sinistre", "accident", "dégât", "expertise", "réparation", "panne"],
        "contrat": ["contrat", "résiliation", "souscription", "garantie", "couverture"],
        "satisfaction": ["satisfait", "recommande", "qualité", "avis", "pensent"],
        "digital": ["site", "application", "espace client", "en ligne"]
    }
    for theme, keywords in theme_keywords_map.items():
        if any(kw in q_lower for kw in keywords):
            detected.append(theme)
    return detected if detected else ["general"]

def generate_heuristic_response(question, context_docs, notes, detected_assureur, detected_themes):
    avg_note = np.mean(notes)
    nb_pos = sum(1 for n in notes if n >= 4)
    nb_neg = sum(1 for n in notes if n <= 2)
    nb_neutre = len(notes) - nb_pos - nb_neg
    assureur_str = detected_assureur if detected_assureur else "les assureurs analysés"
    themes_str = ", ".join(detected_themes) if detected_themes[0] != "general" else "général"
    
    # Extraire les mots les plus fréquents dans les avis retrouvés
    from collections import Counter
    all_words = " ".join([d.split("] ")[-1] if "] " in d else d for d in context_docs]).lower().split()
    stop = {"le","la","les","de","du","des","un","une","et","en","au","à","ce","se","est","pas","plus","qui","que","je","ne","mon","ma","il","nous","très","bien","pour","dans","sur","avec"}
    filtered = [w for w in all_words if w not in stop and len(w) > 2]
    top_words = [w for w, c in Counter(filtered).most_common(8)]
    
    # Construire une réponse structurée
    if avg_note >= 4:
        sentiment_intro = f"Les clients sont **globalement très satisfaits** de **{assureur_str}** concernant le thème **{themes_str}**."
    elif avg_note >= 3:
        sentiment_intro = f"Les avis sont **mitigés** concernant **{assureur_str}** sur le thème **{themes_str}**."
    else:
        sentiment_intro = f"Les clients expriment une **insatisfaction notable** envers **{assureur_str}** concernant **{themes_str}**."
    
    # Extraire des citations courtes des avis
    points_pos = []
    points_neg = []
    for doc, note in zip(context_docs, notes):
        text = doc.split("] ")[-1] if "] " in doc else doc
        short = text[:150].strip()
        if note >= 4:
            points_pos.append(short)
        elif note <= 2:
            points_neg.append(short)
    
    response = f"""{sentiment_intro}

📊 **Statistiques** : Sur {len(notes)} avis analysés, la note moyenne est de **{avg_note:.1f}/5** ({nb_pos} positifs, {nb_neutre} neutres, {nb_neg} négatifs).

🔑 **Mots-clés récurrents** : {", ".join(top_words)}
"""
    
    if points_pos:
        response += f"""
✅ **Points positifs relevés** :
- _{points_pos[0][:120]}..._
"""
        if len(points_pos) > 1:
            response += f"- _{points_pos[1][:120]}..._\n"
    
    if points_neg:
        response += f"""
❌ **Points négatifs relevés** :
- _{points_neg[0][:120]}..._
"""
        if len(points_neg) > 1:
            response += f"- _{points_neg[1][:120]}..._\n"
    
    return response

# --- Interface ---
q = st.text_input("❓ Posez votre question :",
    placeholder="Ex: Que pensent les clients du service client chez Direct Assurance ?")

if q and st.button("🔍 Analyser et répondre", type="primary"):

    # Étape 1 : Détection assureur et thème
    st.markdown("---")
    st.markdown("### 🔎 Étape 1 : Analyse de la question")

    assureurs_list = df["assureur"].dropna().unique().tolist()
    detected_assureur = detect_assureur(q, assureurs_list)
    detected_themes = detect_theme(q)

    col1, col2 = st.columns(2)
    with col1:
        if detected_assureur:
            st.success(f"🏢 Assureur détecté : **{detected_assureur}**")
        else:
            st.info("🏢 Aucun assureur spécifique → recherche globale")
    with col2:
        st.info(f"🏷️ Thèmes : **{', '.join(detected_themes)}**")

    # Étape 2 : Filtrage et Retrieval
    st.markdown("### 📚 Étape 2 : Recherche des avis pertinents")

    df_filtered = df.copy()
    if detected_assureur:
        df_filtered = df_filtered[df_filtered["assureur"] == detected_assureur]
    if len(df_filtered) < 5:
        df_filtered = df.copy()
        st.warning("Pas assez d'avis, recherche élargie.")

    tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    matrix = tfidf.fit_transform(df_filtered["avis"].fillna(""))
    sims = cosine_similarity(tfidf.transform([q]), matrix).flatten()

    # Boost thématique
    theme_kw = {
        "prix": ["prix", "tarif", "cher"],
        "service_client": ["service", "téléphone", "conseiller"],
        "remboursement": ["remboursement", "soins", "dentaire"],
        "sinistre": ["sinistre", "accident", "expertise"],
        "contrat": ["contrat", "résiliation", "garantie"],
        "satisfaction": ["satisfait", "recommande"],
        "digital": ["site", "application"],
        "general": []
    }
    for theme in detected_themes:
        for kw in theme_kw.get(theme, []):
            mask = df_filtered["avis"].fillna("").str.lower().str.contains(kw)
            sims[mask.values] *= 1.3

    top_k = 7
    top_idx = sims.argsort()[-top_k:][::-1]

    context_docs = []
    for i, idx in enumerate(top_idx):
        r = df_filtered.iloc[idx]
        doc_text = f"[Note:{int(r['note'])}/5 | {r['assureur']} | {r['produit']}] {r['avis']}"
        context_docs.append(doc_text)
        with st.expander(f"Doc {i+1} — {'⭐'*int(r['note'])} {r['assureur']} ({r['produit']}) — score: {sims[idx]:.3f}"):
            st.write(r["avis"])

    notes = [df_filtered.iloc[idx]["note"] for idx in top_idx]
    avg_note = np.mean(notes)

    col1, col2, col3 = st.columns(3)
    col1.metric("Note moyenne", f"{avg_note:.1f}/5")
    col2.metric("Avis positifs", f"{sum(1 for n in notes if n >= 4)}/{len(notes)}")
    col3.metric("Avis négatifs", f"{sum(1 for n in notes if n <= 2)}/{len(notes)}")

    # Étape 3 : Génération de la réponse
    st.markdown("### 💬 Étape 3 : Réponse générée")

    response = generate_heuristic_response(q, context_docs, notes, detected_assureur, detected_themes)
    st.markdown(response)
