
'''import streamlit as st
import string
from spacy.lang.en.stop_words import STOP_WORDS as stopwords

st.title(" PREDICÇÃO DE TEXTOS ")
input_ = st.text_area(label = ' Insira o seu texto ')
contractions= {
    # Pronomes + be
    "i'm": "i am",
    "you're": "you are",
    "he's": "he is",
    "she's": "she is",
    "it's": "it is",
    "we're": "we are",
    "they're": "they are",
    "i've": "i have",
    "you've": "you have",
    "we've": "we have",
    "they've": "they have",
    "i'd": "i would",
    "you'd": "you would",
    "he'd": "he would",
    "she'd": "she would",
    "we'd": "we would",
    "they'd": "they would",
    "i'll": "i will",
    "you'll": "you will",
    "he'll": "he will",
    "she'll": "she will",
    "it'll": "it will",
    "we'll": "we will",
    "they'll": "they will",
    
    # Negativos (não + verbos auxiliares)
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "hasn't": "has not",
    "haven't": "have not",
    "hadn't": "had not",
    "doesn't": "does not",
    "don't": "do not",
    "didn't": "did not",
    "won't": "will not",
    "wouldn't": "would not",
    "shan't": "shall not",
    "shouldn't": "should not",
    "can't": "cannot",
    "couldn't": "could not",
    "mustn't": "must not",
    "mightn't": "might not",
    "needn't": "need not",
    
    # Verbos auxiliares
    "let's": "let us",
    "that's": "that is",
    "what's": "what is",
    "who's": "who is",
    "where's": "where is",
    "when's": "when is",
    "why's": "why is",
    "how's": "how is",
    "there's": "there is",
    "here's": "here is",
    
    # Outras contrações comuns
    "could've": "could have",
    "should've": "should have",
    "would've": "would have",
    "might've": "might have",
    "must've": "must have",
    "ought've": "ought to have",
    "ma'am": "madam",
    "o'clock": "of the clock",
    "y'all": "you all",
    "ain't": "is not",  # ou "am not" / "are not" / "has not" / "have not"
    
    # Variações com maiúsculas (primeira letra)
    "I'm": "I am",
    "You're": "You are",
    "He's": "He is",
    "She's": "She is",
    "It's": "It is",
    "We're": "We are",
    "They're": "They are",
    "I've": "I have",
    "You've": "You have",
    "We've": "We have",
    "They've": "They have",
    "I'd": "I would",
    "You'd": "You would",
    "He'd": "He would",
    "She'd": "She would",
    "We'd": "We would",
    "They'd": "They would",
    "I'll": "I will",
    "You'll": "You will",
    "He'll": "He will",
    "She'll": "She will",
    "It'll": "It will",
    "We'll": "We will",
    "They'll": "They will",
    "Isn't": "Is not",
    "Aren't": "Are not",
    "Wasn't": "Was not",
    "Weren't": "Were not",
    "Hasn't": "Has not",
    "Haven't": "Have not",
    "Hadn't": "Had not",
    "Doesn't": "Does not",
    "Don't": "Do not",
    "Didn't": "Did not",
    "Won't": "Will not",
    "Wouldn't": "Would not",
    "Can't": "Cannot",
    "Couldn't": "Could not",
    "Shouldn't": "Should not",
    "Mustn't": "Must not",
    "Let's": "Let us",
    "That's": "That is",
    "What's": "What is",
    "Who's": "Who is",
    "Where's": "Where is",
    "When's": "When is",
    "Why's": "Why is",
    "How's": "How is",
    "There's": "There is",
    "Here's": "Here is",
    "Could've": "Could have",
    "Should've": "Should have",
    "Would've": "Would have",
    "Might've": "Might have",
    "Must've": "Must have",
}

def clean_text(text):
    # remove tags
    tags = ['\n','\'']
    for tag in tags:
        text = text.replace(tag,'')
    
    # remove punctuation
    text = [x for x in text if x not in string.punctuation]
    text = ''.join(text)
    
    # remove stopwords
    text = ' '.join([t for t in text.split() if t not in stopwords])

    if type(text) is str:
            for key in contractions:
                value = contractions[key]
                text = text.replace(key, value)
           
    return text'''



# ============================================
# Predicao.py - App Streamlit para detecção de IA
# ============================================

# Instalar antes: pip install streamlit scikit-learn
import streamlit as st
import string
import pickle
import os

# ============================================
# 1. DICIONÁRIO DE CONTRAÇÕES
# ============================================
contractions = {
    "i'm": "i am", "you're": "you are", "he's": "he is",
    "she's": "she is", "it's": "it is", "we're": "we are",
    "they're": "they are", "i've": "i have", "you've": "you have",
    "we've": "we have", "they've": "they have", "i'd": "i would",
    "you'd": "you would", "he'd": "he would", "she'd": "she would",
    "we'd": "we would", "they'd": "they would", "i'll": "i will",
    "you'll": "you will", "he'll": "he will", "she'll": "she will",
    "it'll": "it will", "we'll": "we will", "they'll": "they will",
    "isn't": "is not", "aren't": "are not", "wasn't": "was not",
    "weren't": "were not", "hasn't": "has not", "haven't": "have not",
    "hadn't": "had not", "doesn't": "does not", "don't": "do not",
    "didn't": "did not", "won't": "will not", "wouldn't": "would not",
    "shan't": "shall not", "shouldn't": "should not", "can't": "cannot",
    "couldn't": "could not", "mustn't": "must not", "mightn't": "might not",
    "needn't": "need not", "let's": "let us", "that's": "that is",
    "what's": "what is", "who's": "who is", "where's": "where is",
    "when's": "when is", "why's": "why is", "how's": "how is",
    "there's": "there is", "here's": "here is", "could've": "could have",
    "should've": "should have", "would've": "would have",
    "might've": "might have", "must've": "must have",
    "I'm": "I am", "You're": "You are", "He's": "He is",
    "She's": "She is", "It's": "It is", "We're": "We are",
    "They're": "They are", "I've": "I have", "You've": "You have",
    "We've": "We have", "They've": "They have", "I'd": "I would",
    "You'd": "You would", "He'd": "He would", "She'd": "She would",
    "We'd": "We would", "They'd": "They would", "I'll": "I will",
    "You'll": "You will", "He'll": "He will", "She'll": "She will",
    "It'll": "It will", "We'll": "We will", "They'll": "They will",
    "Isn't": "Is not", "Aren't": "Are not", "Wasn't": "Was not",
    "Weren't": "Were not", "Hasn't": "Has not", "Haven't": "Have not",
    "Hadn't": "Had not", "Doesn't": "Does not", "Don't": "Do not",
    "Didn't": "Did not", "Won't": "Will not", "Wouldn't": "Would not",
    "Can't": "Cannot", "Couldn't": "Could not", "Shouldn't": "Should not",
    "Mustn't": "Must not", "Let's": "Let us", "That's": "That is",
    "What's": "What is", "Who's": "Who is", "Where's": "Where is",
    "When's": "When is", "Why's": "Why is", "How's": "How is",
    "There's": "There is", "Here's": "Here is", "Could've": "Could have",
    "Should've": "Should have", "Would've": "Would have",
    "Might've": "Might have", "Must've": "Must have",
}

# ============================================
# 2. LISTA DE STOPWORDS (sem precisar do spacy)
# ============================================
stopwords = set([
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
    "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
    "this", "but", "his", "by", "from", "they", "we", "say", "her",
    "she", "or", "an", "will", "my", "one", "all", "would", "there",
    "their", "what", "so", "up", "out", "if", "about", "who", "get",
    "which", "go", "me", "when", "make", "can", "like", "time", "no",
    "just", "him", "know", "take", "people", "into", "year", "your",
    "good", "some", "could", "them", "see", "other", "than", "then",
    "now", "look", "only", "come", "its", "over", "think", "also",
    "back", "after", "use", "two", "how", "our", "work", "first",
    "well", "way", "even", "new", "want", "because", "any", "these",
    "give", "day", "most", "us", "is", "are", "was", "were", "been",
    "has", "had", "did", "does", "being", "having", "doing", "am"
])

# ============================================
# 3. FUNÇÃO DE LIMPEZA DE TEXTO
# ============================================
def clean_text(text):
    # remove tags (\n e ')
    tags = ['\n', "'"]
    for tag in tags:
        text = text.replace(tag, '')
    
    # remove pontuação
    text = ''.join([x for x in text if x not in string.punctuation])
    
    # expandir contrações
    for key, value in contractions.items():
        text = text.replace(key, value)
    
    # remover stopwords
    text = ' '.join([t for t in text.split() if t.lower() not in stopwords])
    
    return text

# ============================================
# 4. CARREGAR MODELO E VETORIZADOR
# ============================================
@st.cache_resource
def carregar_modelos():
    """Carrega os arquivos .pkl com cache para não recarregar toda vez"""
    # Ajuste o caminho se os arquivos estiverem em outra pasta
    caminho_base = os.path.dirname(os.path.abspath(__file__))
    
    clf_path = os.path.join(caminho_base, 'clf.pkl')
    tfidf_path = os.path.join(caminho_base, 'tfidf.pkl')
    
    if not os.path.exists(clf_path):
        st.error(f"Arquivo 'clf.pkl' não encontrado em: {caminho_base}")
        st.info("Certifique-se de que os arquivos clf.pkl e tfidf.pkl estão na mesma pasta do Predicao.py")
        return None, None
    
    if not os.path.exists(tfidf_path):
        st.error(f"Arquivo 'tfidf.pkl' não encontrado em: {caminho_base}")
        return None, None
    
    with open(clf_path, 'rb') as f:
        clf = pickle.load(f)
    
    with open(tfidf_path, 'rb') as f:
        tfidf = pickle.load(f)
    
    return clf, tfidf

clf_svm, tfidf_svm = carregar_modelos()

# ============================================
# 5. INTERFACE STREAMLIT
# ============================================
st.title(" PREDIÇÃO DE TEXTOS")
st.markdown("### Detector de Texto Gerado por IA")
st.markdown("---")

input_ = st.text_area(
    label="Insira o seu texto",
    placeholder="Cole aqui o texto que deseja analisar...",
    height=200
)

if st.button("🔍 Analisar Texto", type="primary"):
    if not input_:
        st.warning("Por favor, insira um texto para analisar.")
    elif clf_svm is None or tfidf_svm is None:
        st.error("Modelos não carregados. Verifique se os arquivos .pkl existem.")
    else:
        with st.spinner("Analisando..."):
            # Limpar o texto
            texto_limpo = clean_text(input_)
            
            # Transformar com TF-IDF
            texto_tfidf = tfidf_svm.transform([texto_limpo])
            
            # Fazer predição
            predicao = clf_svm.predict(texto_tfidf)[0]
            
            # Obter confiança (decision_function)
            confianca = clf_svm.decision_function(texto_tfidf)[0]
        
        st.markdown("---")
        st.subheader("📊 Resultado da Análise")
        
        if predicao == 1.0:
            st.error("🤖 **TEXTO GERADO POR IA**")
            st.info(f"Confiança do modelo: {abs(confianca):.4f}")
        else:
            st.success("✍️ **TEXTO HUMANO**")
            st.info(f"Confiança do modelo: {abs(confianca):.4f}")
        
        with st.expander("📝 Ver texto processado"):
            st.text(texto_limpo)