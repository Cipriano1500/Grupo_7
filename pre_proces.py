# -*- coding: utf-8 -*-
"""Pre-processamento PARTILHADO entre o notebook de treino e o Predicao.py.
Garante zero skew treino/servir: é a MESMA função nos dois lados."""
import re
import string

STOPWORDS = set("""the be to of and a in that have i it for not on with he as you do at
this but his by from they we say her she or an will my one all would there their what so
up out if about who get which go me when make can like time no just him know take people
into year your good some could them see other than then now look only come its over think
also back after use two how our work first well way even new want because any these give
day most us is are was were been has had did does being having doing am""".split())

CONTRACTIONS = {
    "i'm ": "i am ", "you're ": "you are ", "he's ": "he is ", "she's ": "she is ",
    "it's ": "it is ", "we're ": "we are ", "they're ": "they are ", "i've ": "i have ",
    "you've ": "you have ", "we've ": "we have ", "they've ": "they have ",
    "i'd ": "i would ", "you'd ": "you would ", "he'd ": "he would ", "she'd ": "she would ",
    "we'd ": "we would ", "they'd ": "they would ", "i'll ": "i will ", "you'll ": "you will ",
    "he'll ": "he will ", "she'll ": "she will ", "it'll ": "it will ", "we'll ": "we will ",
    "they'll ": "they will ", "isn't ": "is not ", "aren't ": "are not ", "wasn't ": "was not ",
    "weren't ": "were not ", "hasn't ": "has not ", "haven't ": "have not ",
    "hadn't ": "had not ", "doesn't ": "does not ", "don't ": "do not ", "didn't ": "did not ",
    "won't ": "will not ", "wouldn't ": "would not ", "shan't ": "shall not ",
    "shouldn't ": "should not ", "can't ": "cannot ", "couldn't ": "could not ",
    "mustn't ": "must not ", "mightn't ": "might not ", "needn't ": "need not ",
    "let's ": "let us ", "that's ": "that is ", "what's ": "what is ", "who's ": "who is ",
    "where's ": "where is ", "when's ": "when is ", "why's ": "why is ", "how's ": "how is ",
    "there's ": "there is ", "here's ": "here is ", "could've ": "could have ",
    "should've ": "should have ", "would've ": "would have ", "might've ": "might have ",
    "must've ": "must have ", "ought've ": "ought to have ", "ma'am ": "madam ",
    "y'all ": "you all ", "ain't ": "is not ",
}

def preprocess_text(text):
    """Ordem correta: lowercase -> contrações (c/ apóstrofos) -> pontuação -> stopwords."""
    text = str(text).lower()
    text = text + " "                      # garante casamento das chaves com espaço final
    for k, v in CONTRACTIONS.items():
        text = text.replace(k, v)
    text = text.replace("\n", " ")
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = " ".join(t for t in text.split() if t not in STOPWORDS)
    return re.sub(r"\s+", " ", text).strip()