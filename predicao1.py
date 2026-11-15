# -*- coding: utf-8 -*-
"""App Streamlit — usa o MESMO preprocess_text do treino (zero skew)."""
import streamlit as st
import pickle, os, json
from pre_proces import preprocess_text

@st.cache_resource
def carregar_modelos():
    base = os.path.dirname(os.path.abspath(__file__))
    clf_p, tf_p, md_p = (os.path.join(base, n) for n in ("clf.pkl", "tfidf.pkl", "metadata.json"))
    if not (os.path.exists(clf_p) and os.path.exists(tf_p)):
        st.error("clf.pkl / tfidf.pkl não encontrados na pasta do app.")
        return None, None, {}
    clf = pickle.load(open(clf_p, "rb"))
    tfidf = pickle.load(open(tf_p, "rb"))
    meta = json.load(open(md_p)) if os.path.exists(md_p) else {}
    return clf, tfidf, meta

clf, tfidf, meta = carregar_modelos()

st.title("PREDIÇÃO DE TEXTOS")
st.markdown("### Detector de texto gerado por IA")
if meta:
    st.caption(f"Modelo: {meta.get('model')} | acc teste: {meta.get('test_accuracy'):.3f} | preprocess: {meta.get('preprocess')}")

input_ = st.text_area("Insira o seu texto", placeholder="Cole aqui o texto...", height=200)

if st.button("🔍 Analisar Texto", type="primary"):
    if not input_:
        st.warning("Insira um texto.")
    elif clf is None:
        st.error("Modelos não carregados.")
    else:
        n_words = len(input_.split())
        if n_words < 20:
            st.warning("Textos muito curtos (<20 palavras) dão previsões pouco fiáveis.")
        with st.spinner("Analisando..."):
            texto_limpo = preprocess_text(input_)          # <- mesma função do treino
            X = tfidf.transform([texto_limpo])
            pred = clf.predict(X)[0]
            if hasattr(clf, "predict_proba"):
                conf_txt = f"probabilidade {clf.predict_proba(X)[0].max():.3f}"
            else:
                conf_txt = f"margem de decisão {abs(clf.decision_function(X)[0]):.3f}"
            st.markdown("---")
            if int(pred) == 1:
                st.error("🤖 TEXTO GERADO POR IA")
            else:
                st.success("✍️ TEXTO HUMANO")
            st.info(f"Confiança: {conf_txt}")
            with st.expander("📝 Ver texto processado"):
                st.text(texto_limpo)