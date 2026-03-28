import os
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

base_path = os.path.dirname(os.path.dirname(__file__))
data_path = os.path.join(base_path, "data")
models_path = os.path.join(base_path, "models")

st.title("🤖 RAG — Retrieval-Augmented Generation")

st.markdown("""
### À quoi sert cette page ?
Le **RAG** combine recherche d'information et génération de texte :
1. **Détection** : identification de l'assureur et du thème dans votre question
2. **Retrieval** : recherche des avis pertinents filtrés par assureur et thème
3. **Generation** : un LLM génère une réponse naturelle basée sur les avis retrouvés
""")

# --- Charger données et thèmes ---
import pickle

df = pd.read_csv(os.path.join(data_path, "train_clean.csv"))
try:
    themes = pickle.load(open(os.path.join(models_path, "themes.pkl"), "rb"))
except:
    themes = {}

# --- Charger le LLM (flan-t5-small, léger et gratuit) ---
@st.cache_resource
def load_llm():
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        model_name = "google/flan-t5-small"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        return tokenizer, model, True
    except Exception as e:
        return None, None, False

llm_tokenizer, llm_model, llm_available = load_llm()
if llm_available:
    st.success("✅ LLM chargé (google/flan-t5-small)")
else:
    st.warning("⚠️ LLM non disponible. pip install transformers")

# --- Fonctions utilitaires ---
def detect_assureur(question, assureurs_list):
    q_lower = question.lower()
    for assureur in assureurs_list:
        if assureur.lower() in q_lower:
            return assureur
        # Gérer les variantes courantes
        variants = assureur.lower().replace("'", " ").replace("-", " ").split()
        for v in variants:
            if len(v) > 3 and v in q_lower:
                return assureur
    return None

def detect_theme(question, themes_dict):
    q_lower = question.lower()
    detected = []
    theme_keywords_map = {
        "prix": ["prix", "tarif", "cher", "coût", "cotisation", "augmentation"],
        "service_client": ["service client", "téléphone", "conseiller", "contact", "joignable", "accueil"],
        "remboursement": ["remboursement", "rembourser", "prise en charge", "soins", "dentaire", "optique"],
        "sinistre": ["sinistre", "accident", "dégât", "expertise", "réparation", "panne", "vol"],
        "contrat": ["contrat", "résiliation", "souscription", "garantie", "couverture"],
        "satisfaction": ["satisfait", "recommande", "qualité", "avis", "opinion", "pensent"],
        "digital": ["site", "application", "espace client", "en ligne"]
    }
    for theme, keywords in theme_keywords_map.items():
        if any(kw in q_lower for kw in keywords):
            detected.append(theme)
    return detected if detected else ["general"]

def generate_response(question, context_docs, tokenizer, model):
    context = "\n".join(context_docs[:5])
    prompt = f"""Based on the following customer reviews about an insurance company, answer the question in French.

Reviews:
{context[:1500]}

Question: {question}

Answer in French with a detailed analysis:"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(
        **inputs,
        max_length=300,
        num_beams=4,
        early_stopping=True,
        no_repeat_ngram_size=3
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# --- Interface ---
q = st.text_input("❓ Posez votre question :",
    placeholder="Ex: Que pensent les clients du service client chez Direct Assurance ?")

if q and st.button("🔍 Analyser et répondre", type="primary"):

    # Étape 1 : Détection assureur et thème
    st.markdown("---")
    st.markdown("### 🔎 Étape 1 : Analyse de la question")

    assureurs_list = df["assureur"].dropna().unique().tolist()
    detected_assureur = detect_assureur(q, assureurs_list)
    detected_themes = detect_theme(q, themes)

    col1, col2 = st.columns(2)
    with col1:
        if detected_assureur:
            st.success(f"🏢 Assureur détecté : **{detected_assureur}**")
        else:
            st.info("🏢 Aucun assureur spécifique détecté → recherche globale")
    with col2:
        st.info(f"🏷️ Thèmes détectés : **{', '.join(detected_themes)}**")

    # Étape 2 : Filtrage et Retrieval
    st.markdown("### 📚 Étape 2 : Recherche des avis pertinents")

    # Filtrer par assureur si détecté
    df_filtered = df.copy()
    if detected_assureur:
        df_filtered = df_filtered[df_filtered["assureur"] == detected_assureur]

    if len(df_filtered) < 5:
        df_filtered = df.copy()
        st.warning(f"Pas assez d'avis pour {detected_assureur}, recherche élargie.")

    # TF-IDF + cosine sur le sous-ensemble filtré
    tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    matrix = tfidf.fit_transform(df_filtered["avis"].fillna(""))
    sims = cosine_similarity(tfidf.transform([q]), matrix).flatten()

    # Boost les avis qui mentionnent les thèmes détectés
    theme_keywords_flat = {
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
        keywords = theme_keywords_flat.get(theme, [])
        for kw in keywords:
            mask = df_filtered["avis"].fillna("").str.lower().str.contains(kw)
            sims[mask.values] *= 1.3  # Boost de 30%

    top_k = 7
    top_idx = sims.argsort()[-top_k:][::-1]

    context_docs = []
    for i, idx in enumerate(top_idx):
        r = df_filtered.iloc[idx]
        doc_text = f"[Note:{int(r['note'])}/5 | {r['assureur']} | {r['produit']}] {r['avis']}"
        context_docs.append(doc_text)
        with st.expander(f"Doc {i+1} — {'⭐'*int(r['note'])} {r['assureur']} ({r['produit']}) — score: {sims[idx]:.3f}"):
            st.write(r["avis"])

    # Stats sur les avis retrouvés
    notes = [df_filtered.iloc[idx]["note"] for idx in top_idx]
    avg_note = np.mean(notes)
    nb_pos = sum(1 for n in notes if n >= 4)
    nb_neg = sum(1 for n in notes if n <= 2)

    col1, col2, col3 = st.columns(3)
    col1.metric("Note moyenne", f"{avg_note:.1f}/5")
    col2.metric("Avis positifs", f"{nb_pos}/{len(notes)}")
    col3.metric("Avis négatifs", f"{nb_neg}/{len(notes)}")

    # Étape 3 : Génération LLM
    st.markdown("### 💬 Étape 3 : Réponse générée")

    if llm_available:
        with st.spinner("🤖 Génération de la réponse en cours..."):
            response = generate_response(q, context_docs, llm_tokenizer, llm_model)

        if response and len(response.strip()) > 10:
            st.markdown(f"**🤖 Réponse du LLM :**")
            st.markdown(f"> {response}")
        else:
            # Fallback : réponse heuristique
            st.markdown("**🤖 Réponse générée (analyse automatique) :**")
            assureur_str = detected_assureur if detected_assureur else "les assureurs"
            sentiment = "positif ✅" if avg_note >= 3.5 else "mitigé ⚠️" if avg_note >= 2.5 else "négatif ❌"
            st.markdown(f"> Sur la base de {len(notes)} avis analysés, le sentiment global concernant "
                        f"**{assureur_str}** sur le thème **{', '.join(detected_themes)}** est "
                        f"**{sentiment}** avec une note moyenne de **{avg_note:.1f}/5**. "
                        f"{nb_pos} avis sont positifs et {nb_neg} sont négatifs.")
    else:
        # Sans LLM : réponse heuristique + prompt copiable
        assureur_str = detected_assureur if detected_assureur else "les assureurs"
        sentiment = "positif ✅" if avg_note >= 3.5 else "mitigé ⚠️" if avg_note >= 2.5 else "négatif ❌"
        st.markdown(f"""**📊 Analyse automatique :**

Sur la base de **{len(notes)} avis** analysés, le sentiment global concernant
**{assureur_str}** sur le thème **{', '.join(detected_themes)}** est **{sentiment}**
avec une note moyenne de **{avg_note:.1f}/5**.
- {nb_pos} avis positifs (4-5★)
- {nb_neg} avis négatifs (1-2★)
""")

    # Toujours afficher le prompt copiable
    st.markdown("### 📋 Prompt pour LLM externe (ChatGPT, Claude...)")
    prompt_text = f"""Tu es un analyste spécialisé en assurance. Voici des avis clients :

{chr(10).join(context_docs[:5])}

Question : {q}

Réponds en français avec :
1. Un résumé du sentiment global
2. Les points positifs mentionnés
3. Les points négatifs mentionnés
4. Une recommandation"""
    st.code(prompt_text[:2000], language="text")
    st.caption("💡 Copiez ce prompt dans ChatGPT ou Claude pour une réponse plus détaillée")
