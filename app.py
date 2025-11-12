# app.py
# Tarefa 3 – Dashboard interativo (Streamlit)
# Autor: Guilherme Uchida – Matrícula 241008317
# Requisitos: streamlit, scikit-learn, imbalanced-learn, pandas, numpy, matplotlib, seaborn

import os
import time
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, log_loss, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from imblearn.over_sampling import SMOTE

st.set_page_config(
    page_title="Tarefa 3 • Cancelamento de Reservas (Hotel Booking Demand)",
    page_icon="🏨",
    layout="wide"
)

# -------------------------
# Sidebar – Dataset
# -------------------------
st.sidebar.header("📦 Dados")
st.sidebar.write(
    "Envie o arquivo **hotel_bookings.csv** ou deixe o arquivo com esse nome na raiz do app."
)

uploaded = st.sidebar.file_uploader("Enviar hotel_bookings.csv", type=["csv"])
default_path = "hotel_bookings.csv"
if uploaded:
    df = pd.read_csv(uploaded)
elif os.path.exists(default_path):
    df = pd.read_csv(default_path)
else:
    st.error("⚠️ Não encontrei o arquivo hotel_bookings.csv. Faça o upload na barra lateral.")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.caption("Autor: **Guilherme Uchida – Matrícula 241008317**")

# -------------------------
# Funções utilitárias
# -------------------------
NUM_SEED = 42
np.random.seed(NUM_SEED)

def clean_and_engineer(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()

    # Tipos
    if "reservation_status_date" in df.columns:
        df["reservation_status_date"] = pd.to_datetime(df["reservation_status_date"])

    # Ausentes
    df["children"] = df["children"].fillna(0)
    df["country"] = df["country"].fillna("Unknown")
    df["agent"] = df["agent"].fillna(0)
    df["company"] = df["company"].fillna(0)

    # Outliers em ADR (winsorize simples)
    df = df[df["adr"].notna()]
    df = df[df["adr"] >= 0]
    df.loc[df["adr"] > 510, "adr"] = 510  # limite conservador

    # Remove registros sem hóspedes
    if {"adults", "children", "babies"}.issubset(df.columns):
        df = df[(df["adults"] + df["children"] + df["babies"]) > 0]

    # Engenharia de variáveis
    df["total_guests"] = df["adults"] + df["children"] + df["babies"]
    df["stays_total_nights"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
    df["has_special_request"] = (df["total_of_special_requests"] > 0).astype(int)
    df["has_booking_changes"] = (df["booking_changes"] > 0).astype(int)
    df["is_repeated_guest"] = df["is_repeated_guest"].astype(int)

    # Seleção de colunas de interesse
    keep = [
        "hotel","lead_time","arrival_date_year","arrival_date_week_number","arrival_date_day_of_month",
        "stays_in_weekend_nights","stays_in_week_nights","adults","children","babies","meal",
        "country","market_segment","distribution_channel","is_repeated_guest","previous_cancellations",
        "previous_bookings_not_canceled","reserved_room_type","assigned_room_type","booking_changes",
        "deposit_type","agent","company","days_in_waiting_list","customer_type","adr",
        "required_car_parking_spaces","total_of_special_requests","reservation_status",
        "total_guests","stays_total_nights","has_special_request","has_booking_changes"
    ]
    keep = [c for c in keep if c in df.columns]
    df = df[keep + ["is_canceled"]]

    # Dummies
    cat_cols = ["hotel","meal","market_segment","distribution_channel","deposit_type","customer_type","reserved_room_type","assigned_room_type","country","reservation_status"]
    cat_cols = [c for c in cat_cols if c in df.columns]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    return df

@st.cache_data(show_spinner=False)
def prepare_data(df_raw: pd.DataFrame, test_size=0.2, random_state=NUM_SEED):
    df_prep = clean_and_engineer(df_raw)

    X = df_prep.drop(columns=["is_canceled"])
    y = df_prep["is_canceled"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Escalonamento
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # SMOTE no treino
    sm = SMOTE(random_state=random_state, n_jobs=1)
    X_train_bal, y_train_bal = sm.fit_resample(X_train_scaled, y_train)

    meta = {
        "feature_names": X.columns.tolist(),
        "train_shape": X_train_bal.shape,
        "test_shape": X_test_scaled.shape,
        "pos_rate_train_before": y_train.mean(),
        "pos_rate_train_after": y_train_bal.mean(),
    }
    return X_train_bal, X_test_scaled, y_train_bal, y_test, meta

def eval_metrics(y_true, y_prob, y_pred, model_name: str, elapsed: float):
    # Algumas libs devolvem probas 1D; garantir shape
    if y_prob.ndim > 1:
        y_prob = y_prob[:, 1]
    metrics = {
        "Modelo": model_name,
        "Acurácia": accuracy_score(y_true, y_pred),
        "Precisão": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1-Score": f1_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_prob),
        "Log-Loss": log_loss(y_true, np.clip(y_prob, 1e-8, 1-1e-8)),
        "Tempo (s)": elapsed
    }
    return metrics

def plot_confusion(cm, title):
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Valor Predito"); ax.set_ylabel("Valor Real")
    ax.set_title(title)
    return fig

def plot_roc_curves(curves, title="Curvas ROC"):
    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    for name, (fpr, tpr, auc) in curves.items():
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {auc:.4f})")
    ax.plot([0,1],[0,1],"k--",lw=1, label="Classificador Aleatório")
    ax.set_xlabel("Taxa de Falsos Positivos")
    ax.set_ylabel("Taxa de Verdadeiros Positivos")
    ax.set_title(title)
    ax.legend()
    return fig

def rank_df(metrics_list):
    dfm = pd.DataFrame(metrics_list)
    # Ranking pela AUC (desc), depois F1 (desc), depois Log-Loss (asc)
    dfm["_rank_key"] = (-dfm["AUC"], -dfm["F1-Score"], dfm["Log-Loss"])
    dfm = dfm.sort_values(by=["AUC","F1-Score","Log-Loss"], ascending=[False, False, True]).drop(columns=["_rank_key"])
    dfm.index = np.arange(1, len(dfm)+1)
    return dfm

# -------------------------
# Prepara dados
# -------------------------
with st.spinner("Preparando dados (limpeza, dummies, scaler e SMOTE)..."):
    X_train, X_test, y_train, y_test, meta = prepare_data(df)

st.markdown("## 🏨 Tarefa 3 — Previsão de Cancelamentos (Hotel Booking Demand)")
st.caption("Autor: **Guilherme Uchida – Matrícula 241008317** • Universidade de Brasília – FT/EPR")

with st.expander("Ver resumo da preparação dos dados", expanded=False):
    st.write({
        "Amostras treino (após SMOTE)": meta["train_shape"],
        "Amostras teste": meta["test_shape"],
    })
    st.write("Observação: o **SMOTE** foi aplicado somente no conjunto de treino, após **StandardScaler**.")

# -------------------------
# Sidebar – Algoritmos e Hiperparâmetros
# -------------------------
st.sidebar.header("⚙️ Modelagem")

alg_choice = st.sidebar.selectbox(
    "Escolha o algoritmo (ou 'Comparar todos')",
    ["Regressão Logística (RL)", "KNN", "SVM", "Comparar todos"]
)

st.sidebar.subheader("Hiperparâmetros")

# Hiperparâmetros default
hp = {}

if alg_choice in ["Regressão Logística (RL)", "Comparar todos"]:
    st.sidebar.markdown("**Regressão Logística (RL)**")
    hp["rl_C"] = st.sidebar.slider("C (RL)", 0.01, 10.0, 1.0, 0.01)
    hp["rl_penalty"] = st.sidebar.selectbox("Penalidade", ["l2"])
    st.sidebar.markdown("---")

if alg_choice in ["KNN", "Comparar todos"]:
    st.sidebar.markdown("**KNN**")
    hp["knn_k"] = st.sidebar.slider("k (vizinhos)", 3, 31, 5, 2)
    hp["knn_metric"] = st.sidebar.selectbox("Métrica de distância", ["manhattan","euclidean","minkowski"])
    st.sidebar.markdown("---")

if alg_choice in ["SVM", "Comparar todos"]:
    st.sidebar.markdown("**SVM**")
    hp["svm_kernel"] = st.sidebar.selectbox("Kernel", ["rbf","linear"])
    hp["svm_C"] = st.sidebar.slider("C (SVM)", 0.1, 20.0, 10.0, 0.1)
    if hp["svm_kernel"] == "rbf":
        hp["svm_gamma"] = st.sidebar.selectbox("γ (gamma)", ["scale","auto"])
    else:
        hp["svm_gamma"] = "scale"
    st.sidebar.info("⚠️ SVM pode ser mais demorado em amostras grandes.")
    st.sidebar.markdown("---")

# -------------------------
# Treinar/Evaluar
# -------------------------
def train_rl():
    model = LogisticRegression(
        C=hp["rl_C"], penalty="l2", solver="liblinear", random_state=NUM_SEED
    )
    t0 = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - t0
    y_prob = model.predict_proba(X_test)[:,1]
    y_pred = (y_prob>=0.5).astype(int)
    metrics = eval_metrics(y_test, y_prob, y_pred, "Regressão Logística", elapsed)
    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    return model, metrics, cm, (fpr, tpr, metrics["AUC"])

def train_knn():
    model = KNeighborsClassifier(
        n_neighbors=hp["knn_k"], metric=hp["knn_metric"]
    )
    t0 = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - t0
    # predict_proba disponível no KNN
    y_prob = model.predict_proba(X_test)[:,1]
    y_pred = model.predict(X_test)
    metrics = eval_metrics(y_test, y_prob, y_pred, "KNN", elapsed)
    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    return model, metrics, cm, (fpr, tpr, metrics["AUC"])

def train_svm():
    model = SVC(
        kernel=hp["svm_kernel"], C=hp["svm_C"], gamma=hp["svm_gamma"],
        probability=True, random_state=NUM_SEED
    )
    t0 = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - t0
    y_prob = model.predict_proba(X_test)[:,1]
    y_pred = model.predict(X_test)
    metrics = eval_metrics(y_test, y_prob, y_pred, "SVM", elapsed)
    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    return model, metrics, cm, (fpr, tpr, metrics["AUC"])

# -------------------------
# Execução
# -------------------------
if alg_choice == "Regressão Logística (RL)":
    model, m, cm, roc_data = train_rl()
    col1, col2 = st.columns([1,1])
    with col1:
        st.subheader("Métricas")
        st.write(pd.DataFrame([m]).set_index("Modelo"))
        st.caption("Interpretação: RL tende a maior **Precisão**; boa **AUC** com interpretabilidade alta.")
    with col2:
        st.subheader("Matriz de Confusão")
        st.pyplot(plot_confusion(cm, "Regressão Logística"))
    st.subheader("Curva ROC")
    st.pyplot(plot_roc_curves({"Regressão Logística": ((roc_data[0]), (roc_data[1]), m["AUC"])}))

elif alg_choice == "KNN":
    model, m, cm, roc_data = train_knn()
    col1, col2 = st.columns([1,1])
    with col1:
        st.subheader("Métricas")
        st.write(pd.DataFrame([m]).set_index("Modelo"))
        st.caption("Interpretação: KNN costuma entregar maior **Recall** (sensibilidade para cancelados).")
    with col2:
        st.subheader("Matriz de Confusão")
        st.pyplot(plot_confusion(cm, "KNN"))
    st.subheader("Curva ROC")
    st.pyplot(plot_roc_curves({"KNN": ((roc_data[0]), (roc_data[1]), m["AUC"])}))

elif alg_choice == "SVM":
    model, m, cm, roc_data = train_svm()
    col1, col2 = st.columns([1,1])
    with col1:
        st.subheader("Métricas")
        st.write(pd.DataFrame([m]).set_index("Modelo"))
        st.caption("Interpretação: SVM (RBF) frequentemente atinge maior **AUC/F1**, porém com maior custo computacional.")
    with col2:
        st.subheader("Matriz de Confusão")
        st.pyplot(plot_confusion(cm, "SVM"))
    st.subheader("Curva ROC")
    st.pyplot(plot_roc_curves({"SVM": ((roc_data[0]), (roc_data[1]), m["AUC"])}))

else:
    # Comparar todos
    st.subheader("Treinando e comparando os três modelos…")
    rl_model, rl_m, rl_cm, rl_roc = train_rl()
    knn_model, knn_m, knn_cm, knn_roc = train_knn()
    svm_model, svm_m, svm_cm, svm_roc = train_svm()

    metrics_list = [rl_m, knn_m, svm_m]
    df_rank = rank_df(metrics_list)

    c1, c2 = st.columns([1.3, 1])
    with c1:
        st.subheader("📊 Tabela comparativa e ranking (AUC → F1 → Log-Loss)")
        st.dataframe(df_rank.style.format({
            "Acurácia":"{:.4f}","Precisão":"{:.4f}","Recall":"{:.4f}",
            "F1-Score":"{:.4f}","AUC":"{:.4f}","Log-Loss":"{:.4f}","Tempo (s)":"{:.3f}"
        }))
    with c2:
        best = df_rank.iloc[0]
        st.success(f"🏆 **Melhor modelo:** {best['Modelo']}  \nAUC: {best['AUC']:.4f} • F1: {best['F1-Score']:.4f} • Log-Loss: {best['Log-Loss']:.4f}")

    st.subheader("Curvas ROC (comparação)")
    curves = {
        "Regressão Logística": (rl_roc[0], rl_roc[1], rl_m["AUC"]),
        "KNN": (knn_roc[0], knn_roc[1], knn_m["AUC"]),
        "SVM": (svm_roc[0], svm_roc[1], svm_m["AUC"]),
    }
    st.pyplot(plot_roc_curves(curves, "Comparação de Curvas ROC"))

    with st.expander("Interpretação automática (resumo)"):
        st.markdown("""
- **SVM (RBF)** costuma liderar em **AUC** e **F1**, equilibrando bem precisão e recall, ideal quando há capacidade computacional.
- **KNN** frequentemente apresenta recall alto (sensibilidade), útil quando **capturar cancelamentos** é prioridade (ex.: políticas de retenção).
- **Regressão Logística** tende a maior **precisão** e **interpretabilidade** (coeficientes), sendo ótima para comunicação com áreas de negócio. 
- Recomenda-se calibrar o **threshold** (0.5 → custo-ótimo) conforme o custo de falsos positivos/negativos.
""")
