import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, log_loss, RocCurveDisplay, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import io, time

st.set_page_config(page_title="SIEP – Tarefa 3 | Cancelamento de Reservas", layout="wide")
st.title("SIEP – Tarefa 3 · Hotel Booking Demand")
st.caption("Autor: Guilherme Uchida – Matrícula 241008317")

st.sidebar.header("Dados")
uploaded = st.sidebar.file_uploader("Envie o arquivo hotel_bookings.csv", type=["csv"])

target = "is_canceled"

def preprocess(df: pd.DataFrame):
    df = df.copy()
    y = df[target].astype(int)
    X = df.drop(columns=[target])
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    num_imputer = SimpleImputer(strategy="median")
    cat_imputer = SimpleImputer(strategy="most_frequent")

    X_num = num_imputer.fit_transform(X[num_cols]) if len(num_cols) else np.empty((len(X),0))
    # IQR cap
    if X_num.size:
        q1 = np.nanpercentile(X_num,25,axis=0); q3 = np.nanpercentile(X_num,75,axis=0)
        iqr = q3 - q1; low = q1 - 1.5*iqr; high = q3 + 1.5*iqr
        X_num = np.clip(X_num, low, high)
        scaler = StandardScaler().fit(X_num)
        X_num = scaler.transform(X_num)
    X_num_df = pd.DataFrame(X_num, columns=num_cols, index=X.index) if len(num_cols) else pd.DataFrame(index=X.index)

    if len(cat_cols):
        X_cat = cat_imputer.fit_transform(X[cat_cols])
        X_cat_df = pd.DataFrame(X_cat, columns=cat_cols, index=X.index)
        X_cat_df = pd.get_dummies(X_cat_df, columns=cat_cols, drop_first=True)
    else:
        X_cat_df = pd.DataFrame(index=X.index)

    X_pp = X_num_df.join(X_cat_df)
    return X_pp, y

def train_and_eval(model, X_train, y_train, X_test, y_test, name):
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)[:,1]
    else:
        proba = model.decision_function(X_test)
        # map to 0..1 via min-max if needed
        proba = (proba - proba.min())/(proba.max()-proba.min()+1e-9)

    preds = (proba >= 0.5).astype(int)

    auc = roc_auc_score(y_test, proba)
    f1 = f1_score(y_test, preds)
    pr = precision_score(y_test, preds)
    rc = recall_score(y_test, preds)
    try:
        ll = log_loss(y_test, np.clip(proba, 1e-6, 1-1e-6))
    except:
        ll = np.nan
    cm = confusion_matrix(y_test, preds)

    return {"model": name, "AUC": auc, "F1": f1, "Precision": pr, "Recall": rc, "LogLoss": ll, "TrainTime": train_time, "proba": proba, "cm": cm, "estimator": model}

if uploaded is not None:
    df = pd.read_csv(uploaded)
    if target not in df.columns:
        st.error(f"O CSV precisa conter a coluna alvo `{target}`.")
        st.stop()

    st.subheader("Pré-visualização")
    st.dataframe(df.head())

    X_pp, y = preprocess(df)

    X_train, X_test, y_train, y_test = train_test_split(X_pp, y, test_size=0.2, random_state=42, stratify=y)

    pos_ratio = y_train.mean()
    if pos_ratio < 0.4 or pos_ratio > 0.6:
        sm = SMOTE(random_state=42)
        X_train, y_train = sm.fit_resample(X_train, y_train)
        st.info(f"SMOTE aplicado. Proporção positiva pós-SMOTE: {y_train.mean():.2f}")

    st.sidebar.header("Modelo")
    algo = st.sidebar.selectbox("Algoritmo", ["Regressão Logística", "KNN", "SVM"])

    res_all = []

    if algo == "Regressão Logística":
        C = st.sidebar.select_slider("C (RL)", options=[0.01,0.1,0.5,1.0,2.0,5.0,10.0], value=1.0)
        penalty = st.sidebar.selectbox("Penalidade", ["l2","l1"])
        solver = "liblinear" if penalty in ["l1","l2"] else "lbfgs"
        model = LogisticRegression(max_iter=400, C=C, penalty=penalty, solver=solver)
        res = train_and_eval(model, X_train, y_train, X_test, y_test, f"LogisticRegression(C={C},pen={penalty})")
        res_all.append(res)

    elif algo == "KNN":
        k = st.sidebar.select_slider("k", options=[3,5,7,9,11,15], value=5)
        metric = st.sidebar.selectbox("Métrica", ["euclidean","manhattan"])
        weights = st.sidebar.selectbox("Pesos", ["uniform","distance"])
        model = KNeighborsClassifier(n_neighbors=k, metric=metric, weights=weights)
        res = train_and_eval(model, X_train, y_train, X_test, y_test, f"KNN(k={k},{metric},{weights})")
        res_all.append(res)

    else:  # SVM
        kernel = st.sidebar.selectbox("Kernel", ["linear","rbf"])
        C = st.sidebar.select_slider("C (SVM)", options=[0.1,0.5,1.0,2.0,5.0,10.0], value=1.0)
        if kernel == "rbf":
            gamma = st.sidebar.selectbox("gamma", ["scale","auto",0.1,0.01])
            model = SVC(kernel=kernel, C=C, gamma=gamma, probability=True)
            name = f"SVM({kernel},C={C},gamma={gamma})"
        else:
            model = SVC(kernel=kernel, C=C, probability=True)
            name = f"SVM({kernel},C={C})"
        res = train_and_eval(model, X_train, y_train, X_test, y_test, name)
        res_all.append(res)

    # Para ranking, também treinamos rapidamente os outros modelos com defaults leves
    with st.expander("Comparar com outros modelos (padrões rápidos)"):
        do_compare = st.checkbox("Rodar comparação rápida (pode levar alguns segundos)", value=True)
        if do_compare:
            # RL default
            res_all.append(train_and_eval(LogisticRegression(max_iter=400), X_train, y_train, X_test, y_test, "LogisticRegression(default)"))
            # KNN default
            res_all.append(train_and_eval(KNeighborsClassifier(n_neighbors=5, metric="euclidean"), X_train, y_train, X_test, y_test, "KNN(default)"))
            # SVM default
            res_all.append(train_and_eval(SVC(kernel="rbf", probability=True), X_train, y_train, X_test, y_test, "SVM RBF(default)"))

    # Tabela de métricas
    df_res = pd.DataFrame([{k:v for k,v in r.items() if k not in ["proba","cm","estimator"]} for r in res_all])
    st.subheader("Métricas")
    st.dataframe(df_res.sort_values(["AUC","F1"], ascending=False))

    # Ranking
    rank_df = df_res.copy().sort_values(["AUC","F1"], ascending=False).reset_index(drop=True)
    best_row = rank_df.iloc[0]
    st.success(f"🏆 Melhor (rank por AUC, desempate por F1): **{best_row['model']}** — AUC={best_row['AUC']:.4f}, F1={best_row['F1']:.4f}")

    # Curva ROC do primeiro resultado (usuário) + Confusion Matrix
    res_primary = res_all[0]
    st.subheader("Curva ROC")
    fig, ax = plt.subplots()
    RocCurveDisplay.from_predictions(y_true=y_test, y_pred=res_primary["proba"], ax=ax)
    st.pyplot(fig)

    st.subheader("Matriz de Confusão")
    cm = res_primary["cm"]
    fig2, ax2 = plt.subplots()
    im = ax2.imshow(cm, cmap="Blues")
    ax2.set_xlabel("Predito")
    ax2.set_ylabel("Real")
    ax2.set_xticks([0,1]); ax2.set_yticks([0,1])
    for i in range(2):
        for j in range(2):
            ax2.text(j, i, cm[i,j], ha="center", va="center")
    st.pyplot(fig2)

    # Interpretação automática simples
    st.subheader("Interpretação Automática")
    tips = []
    if "Logistic" in res_primary["model"]:
        tips.append("A Regressão Logística permite interpretação direta via coeficientes (odds ratios): valores >1 indicam maior chance de cancelamento.")
    if "KNN" in res_primary["model"]:
        tips.append("KNN depende fortemente do **escalonamento** e da **métrica de distância**; teste k e métricas para robustez.")
    if "SVM" in res_primary["model"]:
        tips.append("SVM com kernel RBF captura relações **não-lineares**; ajuste de C e γ é crucial para equilíbrio viés-variância.")
    tips.append("Operacionalmente, considere **overbooking controlado (3–5%)** em períodos/segmentos de maior risco e ofertas de retenção para reservas de alto risco.")
    for t in tips:
        st.write("• " + t)

else:
    st.info("Faça upload do `hotel_bookings.csv` para iniciar a análise.")