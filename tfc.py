# -*- coding: utf-8 -*-
"""Corre este script uma vez para gerar o notebook tarefa2_ai_human.ipynb (versão final)."""
import json

cells = []

def md(src):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": src})

def code(src):
    cells.append({"cell_type": "code", "metadata": {},
                  "execution_count": None, "outputs": [], "source": src})

md('''# Tarefa 2 — Deteção de texto AI vs Human )

**Treino:** `AI_Human.csv` (colunas `text,generated`)
`dataset1_inputs.csv` (`ID,Text`) + `dataset1_outputs.csv` (`ID,Label`)

''')

code('''import numpy as np
import re
import math
import json
import csv
import os
from collections import Counter

SEED = 42
DATASET_PATH    = "AI_Human.csv"            # dataset de treino (Tarefa 1)
TEACHER_INPUTS  = "dataset1_inputs.csv"     # validação externa do docente
TEACHER_OUTPUTS = "dataset1_outputs.csv"    # labels p/ métricas locais (opcional)
MAX_FEATURES = 5000     # 8 GB RAM ok | se 4 GB: 3000
MAX_TRAIN    = 40000    # se 4 GB: 20000
MAX_VAL      = 10000
MAX_TEST     = 10000''')

md('''## 1. Leitura do dataset de treino + verificação dos labels''')

code('''def parse_label(value):
    """human -> 0; TUDO o resto (1/true/ai/gpt*/chatgpt/machine...) -> 1."""
    v = str(value).strip().lower()
    if v in {"0", "false", "no", "human", "humano", "real"}:
        return 0
    return 1

def inspect_label_column(path, max_rows=2000):
    """Mostra os valores CRUS da coluna 'generated' para confirmares o parsing."""
    vals = Counter()
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            vals[repr(row.get("generated"))] += 1
            if i >= max_rows:
                break
    return vals

def load_ai_human_csv(path):
    texts, labels = [], []
    with open(path, "r", encoding="utf-8-sig") as f:   # utf-8-sig remove BOM
        reader = csv.DictReader(f)
        for row in reader:
            text = (row.get("text") or "").strip()
            if not text:
                continue
            texts.append(text)
            labels.append(parse_label(row.get("generated")))
    return texts, np.array(labels, dtype=np.int64)

def split_data(texts, labels, val_frac=0.15, test_frac=0.15, seed=SEED):
    n = len(texts)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    n_test, n_val = int(n * test_frac), int(n * val_frac)
    test_idx = idx[:n_test]
    val_idx  = idx[n_test:n_test + n_val]
    train_idx = idx[n_test + n_val:]
    texts = np.asarray(texts, dtype=object)
    labels = np.asarray(labels)
    return (texts[train_idx].tolist(), texts[val_idx].tolist(), texts[test_idx].tolist(),
            labels[train_idx], labels[val_idx], labels[test_idx])

def sample_split(texts, labels, n, seed=SEED):
    """Amostra um subconjunto p/ caber na RAM (enunciado permite usar parte dos dados)."""
    if len(texts) <= n:
        return texts, labels
    rng = np.random.RandomState(seed)
    idx = np.sort(rng.choice(len(texts), n, replace=False))
    return [texts[i] for i in idx], labels[idx]''')

md('''## 2. Leitura no formato do docente (2 ficheiros, join por ID) + escrita de previsões

⚠️ O dataset do docente é **só para validação/submissão** — nunca treinar com ele.''')

code('''def load_teacher_dataset(inputs_path, outputs_path=None):
    """Lê datasetX_inputs.csv (ID,Text) e, se existir, datasetX_outputs.csv (ID,Label)."""
    ids, texts = [], []
    with open(inputs_path, "r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            ids.append((row.get("ID") or row.get("id") or "").strip())
            texts.append((row.get("Text") or row.get("text") or "").strip())

    labels = None
    if outputs_path:
        lab = {}
        with open(outputs_path, "r", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                rid = (row.get("ID") or row.get("id") or "").strip()
                lab[rid] = parse_label(row.get("Label"))   # "Human"->0, "AI"->1
        labels = np.array([lab[i] for i in ids], dtype=np.int64)

    return ids, texts, labels

def write_predictions_csv(path, ids, preds):
    """Escreve previsões no formato do docente (ID,Label com 'AI'/'Human')."""
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ID", "Label"])
        for i, p in zip(ids, preds):
            w.writerow([i, "AI" if p == 1 else "Human"])''')

md('''## 3. Features (TF-IDF em 1 só passagem — 1 única matriz alocada)

Vocabulário + IDF aprendidos **apenas no treino** (sem data leakage).''')

code('''STOPWORDS = set("""a an and are as at be by for from has have he in is it its of on or
that the to was were will with you your this these those i we they them his her their
not no yes can could should would may might must shall about into over after before
between under more most very such only own same so than too also just because but if
then else when where why how all any both each few other some through during above
below again further once here there""".split())

def tokenize(text):
    text = str(text).lower()
    tokens = re.findall(r"[a-z']+", text)
    return [t.strip("'") for t in tokens if len(t) > 1 and t.strip("'") not in STOPWORDS]

def fit_text_vectorizer(texts, max_features=MAX_FEATURES):
    counts, df = Counter(), Counter()
    for text in texts:
        tokens = tokenize(text)
        counts.update(tokens)
        df.update(set(tokens))
    vocab = {w: i for i, (w, _) in enumerate(counts.most_common(max_features))}
    N = len(texts)
    idf = np.zeros(len(vocab), dtype=np.float32)
    for w, i in vocab.items():
        idf[i] = math.log((1 + N) / (1 + df.get(w, 0))) + 1.0
    return vocab, idf

def texts_to_tfidf_direct(texts, vocab, idf):
    """TF-IDF (TF sublinear + IDF + L2) alocando APENAS a matriz final."""
    X = np.zeros((len(texts), len(vocab)), dtype=np.float32)
    for i, text in enumerate(texts):
        counts = Counter(t for t in tokenize(text) if t in vocab)
        if counts:
            idx  = np.array([vocab[t] for t in counts], dtype=np.int64)
            vals = np.array([(1.0 + math.log(c)) * idf[vocab[t]]
                             for t, c in counts.items()], dtype=np.float32)
            X[i, idx] = vals
        n = np.linalg.norm(X[i])
        if n > 0:
            X[i] /= n
    return X''')

md('''## 4. Modelo (NumPy puro)

- `hidden_dims=()` → regressão logística (baseline obrigatória)
- `hidden_dims=(256,128)` → DNN com melhorias: LeakyReLU, He/Xavier, Adam, L2, Dropout invertido, early stopping.''')

code('''def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

class BinaryDNNNumpy:
    def __init__(self, hidden_dims=(256, 128), activation="leaky_relu",
                 learning_rate=1e-3, lambda_l2=1e-5, dropout_rate=0.3, seed=SEED):
        self.hidden_dims = list(hidden_dims)
        self.activation = activation
        self.learning_rate = learning_rate
        self.lambda_l2 = lambda_l2
        self.dropout_rate = dropout_rate
        self.seed = seed
        self.params = {}
        self.layer_dims = None

    def _init_params(self, input_dim):
        dims = [input_dim] + self.hidden_dims + [1]
        self.layer_dims = dims
        np.random.seed(self.seed)
        self.params = {}
        for l in range(1, len(dims)):
            fan_in, fan_out = dims[l - 1], dims[l]
            scale = np.sqrt(2.0 / fan_in) if (l < len(dims) - 1 and
                    self.activation in ("relu", "leaky_relu")) else np.sqrt(1.0 / fan_in)
            self.params[f"W{l}"] = (np.random.randn(fan_out, fan_in) * scale).astype(np.float32)
            self.params[f"b{l}"] = np.zeros((fan_out, 1), dtype=np.float32)

    def _activate(self, Z, kind):
        if kind == "relu":       return np.maximum(0, Z)
        if kind == "leaky_relu": return np.where(Z > 0, Z, 0.01 * Z)
        if kind == "tanh":       return np.tanh(Z)
        raise ValueError(kind)

    def _activation_deriv(self, Z, kind):
        if kind == "relu":       return (Z > 0).astype(np.float32)
        if kind == "leaky_relu": return np.where(Z > 0, 1.0, 0.01).astype(np.float32)
        if kind == "tanh":       return 1.0 - np.tanh(Z) ** 2
        raise ValueError(kind)

    def _forward(self, X, training=False):
        A = X.astype(np.float32)
        caches = []
        L = len(self.layer_dims) - 1
        for l in range(1, L + 1):
            W, b = self.params[f"W{l}"], self.params[f"b{l}"]
            A_prev = A
            Z = W.dot(A_prev) + b
            if l < L:
                A = self._activate(Z, self.activation)
                mask = None
                if training and self.dropout_rate > 0:
                    mask = (np.random.rand(*A.shape) >= self.dropout_rate).astype(np.float32) / (1.0 - self.dropout_rate)
                    A = A * mask
                caches.append((A_prev, Z, mask))
            else:
                A = sigmoid(Z)
                caches.append((A_prev, Z, None))
        return A, caches

    def _backward(self, Y, A, caches, m):
        L = len(self.layer_dims) - 1
        grads = {}
        dA = None
        for l in range(L, 0, -1):
            A_prev, Z, mask = caches[l - 1]
            W = self.params[f"W{l}"]
            if l == L:
                dZ = A - Y
            else:
                if mask is not None:
                    dA = dA * mask
                dZ = dA * self._activation_deriv(Z, self.activation)
            dW = dZ.dot(A_prev.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m
            if self.lambda_l2 > 0:
                dW += (self.lambda_l2 / m) * W
            grads[f"dW{l}"], grads[f"db{l}"] = dW, db
            if l > 1:
                dA = W.T.dot(dZ)
        return grads

    def _init_adam(self):
        self.m = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.t = 0

    def _update_adam(self, grads):
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for k in self.params:
            g = grads["d" + k]
            self.m[k] = b1 * self.m[k] + (1 - b1) * g
            self.v[k] = b2 * self.v[k] + (1 - b2) * g * g
            m_hat = self.m[k] / (1 - b1 ** self.t)
            v_hat = self.v[k] / (1 - b2 ** self.t)
            self.params[k] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + eps)

    def loss(self, X, Y):
        A, _ = self._forward(X, training=False)
        eps = 1e-12
        m = Y.shape[1]
        ce = -np.mean(Y * np.log(A + eps) + (1 - Y) * np.log(1 - A + eps))
        if self.lambda_l2 > 0:
            ce += (self.lambda_l2 / (2 * m)) * sum(np.sum(self.params[f"W{l}"] ** 2)
                                                   for l in range(1, len(self.layer_dims)))
        return float(ce)

    def fit(self, X, Y, X_val=None, Y_val=None, epochs=60, batch_size=64,
            patience=6, verbose=True):
        np.random.seed(self.seed)
        self._init_params(X.shape[0])
        self._init_adam()
        m = X.shape[1]
        Y = Y.reshape(1, -1).astype(np.float32)
        if Y_val is not None:
            Y_val = Y_val.reshape(1, -1).astype(np.float32)
        best_loss, best_params, bad = np.inf, None, 0
        for epoch in range(epochs):
            idx = np.random.permutation(m)
            for s in range(0, m, batch_size):
                b = idx[s:s + batch_size]
                A, caches = self._forward(X[:, b], training=True)
                grads = self._backward(Y[:, b], A, caches, len(b))
                self._update_adam(grads)
            tr = self.loss(X, Y)
            val = self.loss(X_val, Y_val) if X_val is not None else tr
            if verbose:
                print(f"Epoch {epoch+1:03d} | train {tr:.5f} | val {val:.5f}")
            if val < best_loss - 1e-5:
                best_loss, best_params, bad = val, {k: v.copy() for k, v in self.params.items()}, 0
            else:
                bad += 1
                if bad >= patience:
                    if verbose: print("Early stopping.")
                    break
        if best_params is not None:
            self.params = best_params
        return self

    def predict_proba(self, X):
        A, _ = self._forward(X, training=False)
        return A.flatten()

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(np.int64)''')

md('''## 5. Métricas por classe (evita o "accuracy 1.0" enganador)''')

code('''def classification_report_numpy(y_true, y_pred):
    y_true = np.asarray(y_true).flatten().astype(int)
    y_pred = np.asarray(y_pred).flatten().astype(int)
    rep = {}
    for cls in (0, 1):
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        fn = np.sum((y_true == cls) & (y_pred != cls))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec  = tp / (tp + fn) if tp + fn else 0.0
        f1   = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        rep[cls] = {"precision": float(prec), "recall": float(rec), "f1": float(f1)}
    rep["accuracy"] = float(np.mean(y_true == y_pred))
    rep["macro_f1"] = (rep[0]["f1"] + rep[1]["f1"]) / 2
    return rep

def print_report(name, y_true, y_pred):
    rep = classification_report_numpy(y_true, y_pred)
    print(f"=== {name} ===  accuracy={rep['accuracy']:.4f} | macro_f1={rep['macro_f1']:.4f}")
    for cls, tag in ((0, "human"), (1, "ai")):
        r = rep[cls]
        print(f"  {tag:>5}: precision={r['precision']:.4f}  recall={r['recall']:.4f}  f1={r['f1']:.4f}")''')

md('''## 6. Guardar / carregar (modelo + vectorizador)''')

code('''def save_artifacts(model_path, vocab_path, idf_path, model, vocab, idf):
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    np.savez(model_path, **model.params)
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)
    np.save(idf_path, idf)

def load_artifacts(model_path, vocab_path, idf_path, model):
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    idf = np.load(idf_path)
    model._init_params(len(vocab))
    data = np.load(model_path)
    model.params = {k: data[k] for k in data.files}
    return model, vocab, idf''')

md('''## 7. Pipeline de treino (AI_Human.csv) — sanity checks, split, amostragem, features''')

code('''print("Valores crus da coluna 'generated':", inspect_label_column(DATASET_PATH))

texts, labels = load_ai_human_csv(DATASET_PATH)
print(f"Exemplos: {len(texts)} | [human, ai] = {np.bincount(labels)}")
assert len(np.bincount(labels)) == 2, "ERRO: só uma classe! Verifica o parse_label."

texts_tr, texts_va, texts_te, y_tr, y_va, y_te = split_data(texts, labels)

texts_tr, y_tr = sample_split(texts_tr, y_tr, MAX_TRAIN)
texts_va, y_va = sample_split(texts_va, y_va, MAX_VAL)
texts_te, y_te = sample_split(texts_te, y_te, MAX_TEST)
print("Após amostragem -> train:", np.bincount(y_tr),
      "| val:", np.bincount(y_va), "| test:", np.bincount(y_te))''')

code('''vocab, idf = fit_text_vectorizer(texts_tr, max_features=MAX_FEATURES)

X_tr = texts_to_tfidf_direct(texts_tr, vocab, idf).T
X_va = texts_to_tfidf_direct(texts_va, vocab, idf).T
X_te = texts_to_tfidf_direct(texts_te, vocab, idf).T
print("X_train:", X_tr.shape, "| X_val:", X_va.shape, "| X_test:", X_te.shape)''')

md('''### 7.1 Baseline: regressão logística''')

code('''logreg = BinaryDNNNumpy(hidden_dims=(), learning_rate=0.05, lambda_l2=1e-4, dropout_rate=0.0)
logreg.fit(X_tr, y_tr, X_va, y_va, epochs=40, batch_size=256, patience=4)
print_report("LogReg TEST", y_te, logreg.predict(X_te))''')

md('''### 7.2 DNN com melhorias''')

code('''dnn = BinaryDNNNumpy(hidden_dims=(256, 128), activation="leaky_relu",
                     learning_rate=1e-3, lambda_l2=1e-5, dropout_rate=0.3)
dnn.fit(X_tr, y_tr, X_va, y_va, epochs=80, batch_size=64, patience=7)
print_report("DNN TEST", y_te, dnn.predict(X_te))''')

code('''save_artifacts("models/dnn_tfidf.npz", "models/vocab.json", "models/idf.npy", dnn, vocab, idf)
print("Modelo e vectorizador guardados em models/")''')

md('''## 8. Validação externa — dataset1 do docente (NUNCA treinar com isto)

Textos de 100–120 palavras sobre ciências naturais: é o teste de generalização real.''')

code('''ids1, texts1, y1 = load_teacher_dataset(TEACHER_INPUTS, TEACHER_OUTPUTS)
print("Dataset1:", len(texts1), "exemplos | [human, ai] =", np.bincount(y1))

X1 = texts_to_tfidf_direct(texts1, vocab, idf).T   # mesmo vocab/idf do treino

print_report("LogReg @ Dataset1", y1, logreg.predict(X1))
print_report("DNN    @ Dataset1", y1, dnn.predict(X1))''')

md('''## 9. Notebook de submissão (formato do docente: ID,Text → ID,Label)

Lê os inputs, aplica o modelo treinado guardado e escreve as previsões. Se os outputs existirem localmente, calcula métricas.''')

code('''model, vocab, idf = load_artifacts(
    "models/dnn_tfidf.npz", "models/vocab.json", "models/idf.npy",
    BinaryDNNNumpy(hidden_dims=(256, 128), activation="leaky_relu"))

has_outputs = os.path.exists(TEACHER_OUTPUTS)
ids, texts, y_true = load_teacher_dataset(
    TEACHER_INPUTS, TEACHER_OUTPUTS if has_outputs else None)

X = texts_to_tfidf_direct(texts, vocab, idf).T
preds = model.predict(X)

write_predictions_csv("predictions_dataset1.csv", ids, preds)
print("Previsões escritas em predictions_dataset1.csv")

if y_true is not None:
    print_report("Submissão @ Dataset1", y_true, preds)''')

md('''## 10. Notas / hiperparâmetros a otimizar (só com a validação)

- `MAX_FEATURES`: 3000 / 5000 / 10000 (conforme RAM e cobertura de vocabulário científico)
- `hidden_dims`: () / (64,) / (128,64) / (256,128)
- `activation`: relu / leaky_relu / tanh
- `dropout_rate`: 0.0–0.5 ; `lambda_l2`: 0–1e-3 ; `learning_rate`: 1e-2–3e-4
- Sinais de saúde: loss 0.05–0.5 ; accuracy 0.8–0.95 ; precision/recall > 0.7 nas duas classes
- Se o dataset1 ficar fraco: juntar textos científicos ao treino (HC3, abstracts) e/ou char n-grams''')

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"}
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

with open("trf2_ai_human.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("OK: ficheiro tarefa2_ai_human.ipynb criado (versão final).")