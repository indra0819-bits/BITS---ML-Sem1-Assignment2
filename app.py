import io
import zipfile
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)

# XGBoost (required by assignment)
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False


# -------------------------
# Page Config
# -------------------------
st.set_page_config(page_title="ML Classification Benchmark (Advanced)", layout="wide")
st.title("ML Classification Benchmark — Advanced Upload Only")

# -------------------------
# Top Banner: Upload + About
# -------------------------
top_left, top_right = st.columns([1, 1])

with top_left:
    st.subheader("Upload Dataset")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])

with top_right:
    st.subheader("About This App")
    st.markdown(
        """
Upload any classification CSV (Kaggle/UCI). The app trains **6 models** on the same dataset: Logistic Regression  / Decision Tree  / KNN  / Naive Bayes (Gaussian)  / Random Forest  / XGBoost  

**Metrics:** Accuracy, AUC (OvR macro), Precision/Recall/F1 (macro), MCC
"""
    )

# -------------------------
# Session State Init
# -------------------------
if "adv_state" not in st.session_state:
    st.session_state.adv_state = {
        "metrics_df": None,
        "results": None,
        "class_labels": None,
        "reports": None,
        "last_signature": None,  # to detect dataset/target changes
    }


# -------------------------
# Helpers
# -------------------------
def apply_missing_value_strategy(df: pd.DataFrame, strategy: str) -> pd.DataFrame:
    """
    - Drop rows: drop any row with any missing value
    - Impute: Numeric mean/median, Categorical mode
    """
    if strategy == "Drop rows with any missing values":
        return df.dropna(axis=0).copy()

    df2 = df.copy()
    use_mean = strategy == "Impute: Numeric=mean, Categorical=mode"

    for col in df2.columns:
        if df2[col].isna().any():
            if pd.api.types.is_numeric_dtype(df2[col]):
                fill_val = df2[col].mean() if use_mean else df2[col].median()
                df2[col] = df2[col].fillna(fill_val)
            else:
                mode = df2[col].mode(dropna=True)
                df2[col] = df2[col].fillna(mode.iloc[0] if len(mode) else "missing")
    return df2


def coerce_features_and_target(df: pd.DataFrame, target_col: str):
    """
    - Separates X, y
    - One-hot encodes non-numeric features
    - Encodes target to integers if needed
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")

    y_raw = df[target_col]
    X_raw = df.drop(columns=[target_col])

    # One-hot encode categorical features
    X = pd.get_dummies(X_raw, drop_first=False)

    # Encode y
    if pd.api.types.is_numeric_dtype(y_raw):
        y = y_raw.copy()
        # If floats are integer-like, cast
        if pd.api.types.is_float_dtype(y) and np.all(np.isclose(y, np.round(y))):
            y = y.round().astype(int)
        target_names = sorted([str(v) for v in pd.unique(y)])
    else:
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y_raw.astype(str)), name=target_col)
        target_names = list(le.classes_)

    if y.isna().any():
        raise ValueError("Target contains missing values after preprocessing.")

    return X, y, target_names


def compute_auc_general(y_true, y_proba, n_classes: int):
    """
    AUC:
    - binary: ROC AUC using positive class probability
    - multi-class: OvR macro AUC
    """
    if y_proba is None:
        return np.nan
    try:
        if n_classes == 2:
            return roc_auc_score(y_true, y_proba[:, 1])
        return roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
    except Exception:
        return np.nan


def get_proba_or_scores(model, X_test):
    """
    Prefer predict_proba; fallback to decision_function -> pseudo-proba.
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_test)

    if hasattr(model, "decision_function"):
        scores = np.array(model.decision_function(X_test))
        if scores.ndim == 1:
            s = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
            return np.vstack([1 - s, s]).T
        exp_scores = np.exp(scores - scores.max(axis=1, keepdims=True))
        return exp_scores / (exp_scores.sum(axis=1, keepdims=True) + 1e-9)

    return None


def make_lr_pipeline(enable_scaling: bool, C: float):
    steps = []
    if enable_scaling:
        steps.append(("scaler", StandardScaler(with_mean=False)))
    # NOTE: no multi_class arg (compat across sklearn versions)
    steps.append(("clf", LogisticRegression(C=C, max_iter=5000, solver="lbfgs")))
    return Pipeline(steps)


def make_knn_pipeline(enable_scaling: bool, k: int):
    steps = []
    if enable_scaling:
        steps.append(("scaler", StandardScaler(with_mean=False)))
    steps.append(("clf", KNeighborsClassifier(n_neighbors=k)))
    return Pipeline(steps)


def build_models(
    n_classes: int,
    random_state: int,
    use_scaling: bool,
    lr_c: float,
    dt_max_depth: int,
    dt_min_samples_split: int,
    knn_k: int,
    rf_n_estimators: int,
    rf_max_depth: int,
    xgb_n_estimators: int,
    xgb_max_depth: int,
    xgb_learning_rate: float,
    xgb_subsample: float,
    xgb_colsample_bytree: float,
):
    models = {
        "Logistic Regression": make_lr_pipeline(enable_scaling=use_scaling, C=lr_c),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=dt_max_depth, min_samples_split=dt_min_samples_split, random_state=random_state
        ),
        "KNN": make_knn_pipeline(enable_scaling=use_scaling, k=knn_k),
        "Naive Bayes (Gaussian)": GaussianNB(),
        "Random Forest": RandomForestClassifier(
            n_estimators=rf_n_estimators, max_depth=rf_max_depth, random_state=random_state, n_jobs=-1
        ),
    }

    if XGBOOST_AVAILABLE:
        if n_classes == 2:
            models["XGBoost"] = XGBClassifier(
                n_estimators=xgb_n_estimators,
                max_depth=xgb_max_depth,
                learning_rate=xgb_learning_rate,
                subsample=xgb_subsample,
                colsample_bytree=xgb_colsample_bytree,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=random_state,
                n_jobs=-1,
            )
        else:
            models["XGBoost"] = XGBClassifier(
                n_estimators=xgb_n_estimators,
                max_depth=xgb_max_depth,
                learning_rate=xgb_learning_rate,
                subsample=xgb_subsample,
                colsample_bytree=xgb_colsample_bytree,
                objective="multi:softprob",
                num_class=n_classes,
                eval_metric="mlogloss",
                random_state=random_state,
                n_jobs=-1,
            )

    return models


def evaluate_all_models(X_train, X_test, y_train, y_test, models, class_labels):
    n_classes = len(class_labels)
    rows = []
    results = {}
    reports = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = get_proba_or_scores(model, X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1m = f1_score(y_test, y_pred, average="macro", zero_division=0)
        mcc = matthews_corrcoef(y_test, y_pred)
        auc = compute_auc_general(y_test, y_proba, n_classes=n_classes)

        cm = confusion_matrix(y_test, y_pred)

        rep_dict = classification_report(
            y_test,
            y_pred,
            labels=list(range(n_classes)),
            target_names=class_labels,
            output_dict=True,
            zero_division=0,
        )
        rep_df = pd.DataFrame(rep_dict).T

        row = {
            "Model": name,
            "Accuracy": acc,
            "AUC (OvR macro)": auc,
            "Precision (macro)": prec,
            "Recall (macro)": rec,
            "F1 (macro)": f1m,
            "MCC": mcc,
        }
        rows.append(row)

        results[name] = {
            "metrics": row,
            "cm": cm,
            "report_df": rep_df,
        }
        reports[name] = rep_df

    metrics_df = pd.DataFrame(rows).sort_values(by="MCC", ascending=False)
    return metrics_df, results, reports


def render_metrics_one_row(m: dict):
    """
    Streamlit-native metrics in a single row (6 columns).
    Avoids HTML rendering issues.
    """
    cols = st.columns(6)
    cols[0].metric("Accuracy", f"{m['Accuracy']:.4f}")
    cols[1].metric(
        "AUC",
        f"{m['AUC (OvR macro)']:.4f}" if not np.isnan(m["AUC (OvR macro)"]) else "NA",
    )
    cols[2].metric("Precision (macro)", f"{m['Precision (macro)']:.4f}")
    cols[3].metric("Recall (macro)", f"{m['Recall (macro)']:.4f}")
    cols[4].metric("F1 (macro)", f"{m['F1 (macro)']:.4f}")
    cols[5].metric("MCC", f"{m['MCC']:.4f}")


# -------------------------
# Sidebar Controls (Preprocess + Hyperparams)
# -------------------------
st.sidebar.header("1) Split & Preprocessing")
test_size = st.sidebar.slider("Test size", 0.1, 0.4, 0.2, 0.05)
random_state = st.sidebar.number_input("Random state", min_value=0, max_value=9999, value=42, step=1)
use_stratify = st.sidebar.checkbox("Use stratified split (recommended)", value=True)
use_scaling = st.sidebar.checkbox("Enable feature scaling (LR & KNN)", value=True)

missing_strategy = st.sidebar.selectbox(
    "Missing value strategy",
    [
        "Impute: Numeric=median, Categorical=mode",
        "Impute: Numeric=mean, Categorical=mode",
        "Drop rows with any missing values",
    ],
)

st.sidebar.header("2) Hyperparameters")
lr_c = st.sidebar.slider("Logistic Regression: C", 0.01, 10.0, 1.0, 0.01)

dt_max_depth = st.sidebar.slider("Decision Tree: max_depth", 1, 30, 7, 1)
dt_min_samples_split = st.sidebar.slider("Decision Tree: min_samples_split", 2, 30, 2, 1)

knn_k = st.sidebar.slider("KNN: n_neighbors (k)", 1, 25, 5, 1)

rf_n_estimators = st.sidebar.slider("Random Forest: n_estimators", 50, 800, 250, 50)
rf_max_depth = st.sidebar.slider("Random Forest: max_depth", 1, 40, 12, 1)

xgb_n_estimators = st.sidebar.slider("XGBoost: n_estimators", 50, 800, 250, 50)
xgb_max_depth = st.sidebar.slider("XGBoost: max_depth", 1, 15, 4, 1)
xgb_learning_rate = st.sidebar.slider("XGBoost: learning_rate", 0.01, 0.5, 0.1, 0.01)
xgb_subsample = st.sidebar.slider("XGBoost: subsample", 0.5, 1.0, 0.9, 0.05)
xgb_colsample_bytree = st.sidebar.slider("XGBoost: colsample_bytree", 0.5, 1.0, 0.9, 0.05)

if not XGBOOST_AVAILABLE:
    st.sidebar.warning("XGBoost not installed. Install with `pip install xgboost`.")


# -------------------------
# Main: Load & Prepare Data
# -------------------------
if uploaded is None:
    st.info("Upload a CSV to begin.")
    st.stop()

df = pd.read_csv(uploaded)

with st.expander("Preview data"):
        n_prev = st.slider("Rows to preview", 10, min(569, len(df)), 50, 10, key="prev_default")
        st.dataframe(df.head(n_prev), width="stretch")

st.divider()

# Under preview: Target (left) + Train (right)
col_target, col_train = st.columns([1, 1])

with col_target:
    st.subheader("Target Variable")
    auto_target = df.columns[-1]
    default_index = list(df.columns).index(auto_target)
    target_col = st.selectbox(
        f"Auto-detected = last column ({auto_target})",
        options=list(df.columns),
        index=default_index,
        key="target_column_select",
    )

with col_train:
    st.subheader("Train Models")
    st.write("")
    st.write("")
    run_btn = st.button("Train & Evaluate Models")

# Missing values
df_clean = apply_missing_value_strategy(df, missing_strategy)
if df_clean.shape[0] != df.shape[0]:
    st.warning(f"Rows changed after missing value handling: {df.shape[0]} → {df_clean.shape[0]}")

# Coerce X/y
try:
    X, y, target_names = coerce_features_and_target(df_clean, target_col=target_col)
except Exception as e:
    st.error(f"CSV processing error: {e}")
    st.stop()

n_classes = len(np.unique(y))
class_labels = [str(x) for x in target_names] if len(target_names) == n_classes else [f"Class {i}" for i in range(n_classes)]

st.subheader("Dataset Summary")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", f"{X.shape[0]}")
c2.metric("Features (after encoding)", f"{X.shape[1]}")
c3.metric("Classes", f"{n_classes}")
c4.metric("Target", target_col)

if X.shape[1] < 12:
    st.warning(f"Features after encoding = {X.shape[1]} (< 12 assignment minimum).")
if X.shape[0] < 500:
    st.warning(f"Rows = {X.shape[0]} (< 500 assignment minimum).")

# If user changes dataset/target, previous results may not match; clear on signature change
signature = (X.shape[0], X.shape[1], target_col, missing_strategy, use_scaling, test_size, random_state)
if st.session_state.adv_state["last_signature"] is not None and st.session_state.adv_state["last_signature"] != signature:
    st.session_state.adv_state["metrics_df"] = None
    st.session_state.adv_state["results"] = None
    st.session_state.adv_state["class_labels"] = None
    st.session_state.adv_state["reports"] = None
st.session_state.adv_state["last_signature"] = signature


# -------------------------
# Train & Evaluate
# -------------------------
if run_btn:
    stratify_val = y if use_stratify and n_classes > 1 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_val
    )

    models = build_models(
        n_classes=n_classes,
        random_state=random_state,
        use_scaling=use_scaling,
        lr_c=lr_c,
        dt_max_depth=dt_max_depth,
        dt_min_samples_split=dt_min_samples_split,
        knn_k=knn_k,
        rf_n_estimators=rf_n_estimators,
        rf_max_depth=rf_max_depth,
        xgb_n_estimators=xgb_n_estimators,
        xgb_max_depth=xgb_max_depth,
        xgb_learning_rate=xgb_learning_rate,
        xgb_subsample=xgb_subsample,
        xgb_colsample_bytree=xgb_colsample_bytree,
    )

    with st.spinner("Training models..."):
        metrics_df, results, reports = evaluate_all_models(X_train, X_test, y_train, y_test, models, class_labels)

    st.session_state.adv_state["metrics_df"] = metrics_df
    st.session_state.adv_state["results"] = results
    st.session_state.adv_state["class_labels"] = class_labels
    st.session_state.adv_state["reports"] = reports

    st.success("Training complete!")


# -------------------------
# Display Results (Persisted)
# -------------------------
state = st.session_state.adv_state
if state["metrics_df"] is None:
    st.info("Click **Train & Evaluate Models** to generate results.")
    st.stop()

metrics_df = state["metrics_df"]
results = state["results"]
reports = state["reports"]
class_labels = state["class_labels"]

st.subheader("Metrics Table")
st.dataframe(
    metrics_df.style.format(
        {
            "Accuracy": "{:.4f}",
            "AUC (OvR macro)": "{:.4f}",
            "Precision (macro)": "{:.4f}",
            "Recall (macro)": "{:.4f}",
            "F1 (macro)": "{:.4f}",
            "MCC": "{:.4f}",
        }
    ),
    width="stretch",
)

st.download_button(
    "⬇️ Download Metrics Table (CSV)",
    data=metrics_df.to_csv(index=False).encode("utf-8"),
    file_name="model_metrics.csv",
    mime="text/csv",
)

st.subheader("Model Details")
model_name = st.selectbox("Select a model", options=list(results.keys()), key="model_select")

out = results[model_name]
m = out["metrics"]

render_metrics_one_row(m)

st.write("**Confusion Matrix**")
cm = out["cm"]
cm_df = pd.DataFrame(
    cm,
    index=[f"True {c}" for c in class_labels],
    columns=[f"Pred {c}" for c in class_labels],
)
st.dataframe(cm_df, width="stretch")

st.write("**Classification Report**")
rep_df = out["report_df"]
st.dataframe(rep_df, width="stretch")

st.download_button(
    f"⬇️ Download {model_name} Report (CSV)",
    data=rep_df.to_csv(index=True).encode("utf-8"),
    file_name=f"classification_report_{model_name.replace(' ', '_').lower()}.csv",
    mime="text/csv",
)

# ZIP all reports
zip_buffer = io.BytesIO()
with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
    for mn, df_report in reports.items():
        fname = f"classification_report_{mn.replace(' ', '_').lower()}.csv"
        zf.writestr(fname, df_report.to_csv(index=True))
zip_buffer.seek(0)

st.download_button(
    "⬇️ Download ALL Reports (ZIP)",
    data=zip_buffer.getvalue(),
    file_name="classification_reports.zip",
    mime="application/zip",
)
