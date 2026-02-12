import io
import zipfile
import streamlit as st
import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
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
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    roc_curve,
    classification_report,
)

# XGBoost (required)
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False


st.set_page_config(page_title="ML Classification Benchmark", layout="wide")
st.title("ML Classification Benchmark — Simple + Advanced (2 Tabs)")

st.write(
    """
This app includes **two modes**:
- **Default Data (Simple)**: Breast Cancer dataset + ROC curve (binary)
- **Advanced Upload**: CSV upload (binary/multi-class) + preprocessing options + exports
"""
)

# -------------------------
# Shared helpers
# -------------------------
@st.cache_data
def load_default_breast_cancer():
    data = load_breast_cancer(as_frame=True)
    df = data.frame.copy()
    X = df[data.feature_names]
    y = df["target"]
    target_names = list(data.target_names)  # ['malignant', 'benign']
    return df, X, y, target_names


def compute_auc_general(y_true, y_proba, n_classes: int):
    """
    AUC:
    - binary: ROC AUC on positive class probability
    - multi-class: OvR macro
    Returns NaN if cannot compute.
    """
    if y_proba is None:
        return np.nan
    try:
        if n_classes == 2:
            if y_proba.ndim == 2:
                return roc_auc_score(y_true, y_proba[:, 1])
            return roc_auc_score(y_true, y_proba)
        else:
            return roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
    except Exception:
        return np.nan


def get_proba_or_scores(model, X_test, n_classes):
    """
    Try to get probabilities; fallback to decision_function -> pseudo probabilities.
    """
    y_proba = None
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)
    elif hasattr(model, "decision_function"):
        scores = np.array(model.decision_function(X_test))
        if scores.ndim == 1:
            s = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
            y_proba = np.vstack([1 - s, s]).T
        else:
            exp_scores = np.exp(scores - scores.max(axis=1, keepdims=True))
            y_proba = exp_scores / (exp_scores.sum(axis=1, keepdims=True) + 1e-9)
    return y_proba


def make_lr_pipeline(enable_scaling: bool, C: float):
    steps = []
    if enable_scaling:
        steps.append(("scaler", StandardScaler(with_mean=False)))
    steps.append(("clf", LogisticRegression(C=C, max_iter=5000, solver="lbfgs", multi_class="auto")))
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
    models = {}

    models["Logistic Regression"] = make_lr_pipeline(enable_scaling=use_scaling, C=lr_c)

    models["Decision Tree"] = DecisionTreeClassifier(
        max_depth=dt_max_depth,
        min_samples_split=dt_min_samples_split,
        random_state=random_state,
    )

    models["KNN"] = make_knn_pipeline(enable_scaling=use_scaling, k=knn_k)

    models["Naive Bayes (Gaussian)"] = GaussianNB()

    models["Random Forest"] = RandomForestClassifier(
        n_estimators=rf_n_estimators,
        max_depth=rf_max_depth,
        random_state=random_state,
        n_jobs=-1,
    )

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


# -------------------------
# Advanced upload helpers
# -------------------------
def apply_missing_value_strategy(df: pd.DataFrame, strategy: str) -> pd.DataFrame:
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
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")

    y_raw = df[target_col]
    X_raw = df.drop(columns=[target_col])

    # One-hot encode categorical features
    X = pd.get_dummies(X_raw, drop_first=False)

    # Encode y
    if pd.api.types.is_numeric_dtype(y_raw):
        y = y_raw.copy()
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


def evaluate_metrics_and_report(model, X_train, X_test, y_train, y_test, class_labels):
    n_classes = len(class_labels)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = get_proba_or_scores(model, X_test, n_classes=n_classes)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)
    auc = compute_auc_general(y_test, y_proba, n_classes=n_classes)

    cm = confusion_matrix(y_test, y_pred)

    report_dict = classification_report(
        y_test,
        y_pred,
        labels=list(range(n_classes)),
        target_names=class_labels,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report_dict).T

    return {
        "Accuracy": acc,
        "AUC (OvR macro)": auc,
        "Precision (macro)": prec,
        "F1 (macro)": f1,
        "MCC": mcc,
        "ConfusionMatrix": cm,
        "ReportDF": report_df,
        "y_proba": y_proba,
        "y_pred": y_pred,
    }


# -------------------------
# Sidebar: shared model hyperparameters
# -------------------------
st.sidebar.header("Model Hyperparameters (Both Tabs)")

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

st.sidebar.divider()

if not XGBOOST_AVAILABLE:
    st.sidebar.warning("XGBoost not installed. Install with: pip install xgboost")


# -------------------------
# Tabs
# -------------------------
tab1, tab2 = st.tabs(["🧪 Default Data (Simple)", "📤 Advanced Upload (CSV)"])


# =====================================================================
# TAB 1: Default dataset (simple) + ROC curve (binary)
# =====================================================================
with tab1:
    df, X, y, target_names = load_default_breast_cancer()
    n_classes = len(np.unique(y))
    class_labels = [str(t) for t in target_names]

    st.subheader("Dataset Summary (Default)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{X.shape[0]}")
    c2.metric("Features", f"{X.shape[1]}")
    c3.metric("Classes", f"{n_classes}")
    c4.metric("Target names", ", ".join(class_labels))

    with st.expander("Preview data"):
        n = st.slider("Rows to preview", 10, min(569, len(df)), 50, 10, key="preview_default")
        st.dataframe(df.head(n), width="stretch")

    st.markdown("### Train & Evaluate (Simple)")
    colA, colB, colC = st.columns([1, 1, 1])
    test_size_simple = colA.slider("Test size", 0.1, 0.4, 0.2, 0.05, key="ts_simple")
    random_state_simple = colB.number_input("Random state", min_value=0, max_value=9999, value=42, step=1, key="rs_simple")
    use_scaling_simple = colC.checkbox("Enable scaling (LR & KNN)", value=True, key="sc_simple")

    run_simple = st.button("Train & Evaluate (Default Dataset)", key="run_simple")

    if run_simple:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size_simple, random_state=random_state_simple, stratify=y
        )

        models = build_models(
            n_classes=2,
            random_state=random_state_simple,
            use_scaling=use_scaling_simple,
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

        results = []
        detailed = {}

        for name, model in models.items():
            out = evaluate_metrics_and_report(model, X_train, X_test, y_train, y_test, class_labels=class_labels)
            detailed[name] = out
            results.append(
                {
                    "Model": name,
                    "Accuracy": out["Accuracy"],
                    "AUC": out["AUC (OvR macro)"],  # binary AUC
                    "Precision": out["Precision (macro)"],
                    "F1": out["F1 (macro)"],
                    "MCC": out["MCC"],
                }
            )

        results_df = pd.DataFrame(results).sort_values(by="MCC", ascending=False)

        st.subheader("Metrics Comparison (Default Dataset)")
        st.dataframe(
            results_df.style.format(
                {"Accuracy": "{:.4f}", "AUC": "{:.4f}", "Precision": "{:.4f}", "F1": "{:.4f}", "MCC": "{:.4f}"}
            ),
            width="stretch",
        )

        st.download_button(
            label="⬇️ Download Metrics Table (CSV)",
            data=results_df.to_csv(index=False).encode("utf-8"),
            file_name="default_dataset_metrics.csv",
            mime="text/csv",
        )

        st.subheader("Detailed View")
        selected = st.selectbox("Select a model", list(detailed.keys()), key="sel_default")
        sel = detailed[selected]

        left, right = st.columns([1, 1])

        with left:
            st.write("**Confusion Matrix**")
            cm = sel["ConfusionMatrix"]
            cm_df = pd.DataFrame(
                cm,
                index=[f"True {class_labels[0]}", f"True {class_labels[1]}"],
                columns=[f"Pred {class_labels[0]}", f"Pred {class_labels[1]}"],
            )
            st.dataframe(cm_df, width="stretch")

        with right:
            if sel["y_proba"] is not None and not np.isnan(sel["AUC (OvR macro)"]):
                st.write("**ROC Curve** (Binary)")
                fpr, tpr, _ = roc_curve(y_test, sel["y_proba"][:, 1])
                roc_df = pd.DataFrame({"False Positive Rate": fpr, "True Positive Rate": tpr})
                st.line_chart(roc_df, x="False Positive Rate", y="True Positive Rate", width="stretch")
                st.caption(f"AUC = {sel['AUC (OvR macro)']:.4f}")
            else:
                st.info("ROC/AUC not available for this model in the current setup.")


# =====================================================================
# TAB 2: Advanced upload (CSV) — multi-class + preprocessing + exports
# =====================================================================
with tab2:
    st.subheader("Upload CSV (Binary or Multi-class)")
    st.write(
        """
Upload a CSV from Kaggle/UCI.
- **Auto target detection** defaults to the **last column**.
- You can override and select the correct target column.
- Handles categorical features via one-hot encoding.
"""
    )

    # Advanced controls
    adv_col1, adv_col2, adv_col3 = st.columns([1, 1, 1])
    test_size_adv = adv_col1.slider("Test size", 0.1, 0.4, 0.2, 0.05, key="ts_adv")
    random_state_adv = adv_col2.number_input("Random state", min_value=0, max_value=9999, value=42, step=1, key="rs_adv")
    use_scaling_adv = adv_col3.checkbox("Enable scaling (LR & KNN)", value=True, key="sc_adv")

    missing_strategy = st.selectbox(
        "Missing value strategy",
        [
            "Impute: Numeric=median, Categorical=mode",
            "Impute: Numeric=mean, Categorical=mode",
            "Drop rows with any missing values",
        ],
        key="miss_adv",
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"], key="uploader_adv")

    if uploaded is None:
        st.info("Upload a CSV to enable Advanced mode.")
    else:
        df_up = pd.read_csv(uploaded)

        st.write("**Uploaded CSV Preview**")
        st.dataframe(df_up.head(20), width="stretch")

        auto_target = df_up.columns[-1]
        use_auto = st.checkbox("Auto-detect target (last column)", value=True, key="auto_target_adv")

        if use_auto:
            target_col = auto_target
            st.info(f"Auto target selected: **{target_col}**")
        else:
            target_col = st.selectbox("Select target column", options=list(df_up.columns), index=len(df_up.columns) - 1)

        # Apply missing strategy on entire dataframe
        df_clean = apply_missing_value_strategy(df_up, missing_strategy)

        if df_clean.shape[0] != df_up.shape[0]:
            st.warning(f"Rows changed after missing-value handling: {df_up.shape[0]} → {df_clean.shape[0]}")

        try:
            X_up, y_up, target_names_up = coerce_features_and_target(df_clean, target_col=target_col)
        except Exception as e:
            st.error(f"CSV processing error: {e}")
            st.stop()

        n_classes_up = len(np.unique(y_up))
        if len(target_names_up) == n_classes_up:
            class_labels_up = [str(x) for x in target_names_up]
        else:
            class_labels_up = [f"Class {i}" for i in range(n_classes_up)]

        st.subheader("Dataset Summary (Uploaded)")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", f"{X_up.shape[0]}")
        c2.metric("Features (after encoding)", f"{X_up.shape[1]}")
        c3.metric("Classes", f"{n_classes_up}")
        c4.metric("Target", target_col)

        if X_up.shape[1] < 12:
            st.warning(f"Features after encoding = {X_up.shape[1]} (< 12 assignment minimum).")
        if X_up.shape[0] < 500:
            st.warning(f"Rows = {X_up.shape[0]} (< 500 assignment minimum).")

        use_stratify_adv = st.checkbox("Use stratified split", value=True, key="strat_adv")
        stratify_val = y_up if use_stratify_adv and n_classes_up > 1 else None

        run_adv = st.button("Train & Evaluate (Uploaded CSV)", key="run_adv")

        if run_adv:
            X_train, X_test, y_train, y_test = train_test_split(
                X_up, y_up, test_size=test_size_adv, random_state=random_state_adv, stratify=stratify_val
            )

            models = build_models(
                n_classes=n_classes_up,
                random_state=random_state_adv,
                use_scaling=use_scaling_adv,
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

            results = []
            confusion_matrices = {}
            reports = {}

            for name, model in models.items():
                try:
                    out = evaluate_metrics_and_report(
                        model, X_train, X_test, y_train, y_test, class_labels=class_labels_up
                    )
                    confusion_matrices[name] = out["ConfusionMatrix"]
                    reports[name] = out["ReportDF"]

                    results.append(
                        {
                            "Model": name,
                            "Accuracy": out["Accuracy"],
                            "AUC (OvR macro)": out["AUC (OvR macro)"],
                            "Precision (macro)": out["Precision (macro)"],
                            "F1 (macro)": out["F1 (macro)"],
                            "MCC": out["MCC"],
                        }
                    )
                except Exception as e:
                    st.warning(f"Model '{name}' failed: {e}")
                    results.append(
                        {
                            "Model": name,
                            "Accuracy": np.nan,
                            "AUC (OvR macro)": np.nan,
                            "Precision (macro)": np.nan,
                            "F1 (macro)": np.nan,
                            "MCC": np.nan,
                        }
                    )

            results_df = pd.DataFrame(results).sort_values(by="MCC", ascending=False)

            st.subheader("Metrics Comparison (Uploaded CSV)")
            st.dataframe(
                results_df.style.format(
                    {
                        "Accuracy": "{:.4f}",
                        "AUC (OvR macro)": "{:.4f}",
                        "Precision (macro)": "{:.4f}",
                        "F1 (macro)": "{:.4f}",
                        "MCC": "{:.4f}",
                    }
                ),
                width="stretch",
            )

            st.download_button(
                label="⬇️ Download Metrics Table (CSV)",
                data=results_df.to_csv(index=False).encode("utf-8"),
                file_name="uploaded_dataset_metrics.csv",
                mime="text/csv",
            )

            st.subheader("Confusion Matrix Viewer")
            selected_cm = st.selectbox("Choose a model", options=list(confusion_matrices.keys()), key="sel_cm_adv")
            cm = confusion_matrices[selected_cm]
            cm_df = pd.DataFrame(
                cm,
                index=[f"True {c}" for c in class_labels_up],
                columns=[f"Pred {c}" for c in class_labels_up],
            )
            st.dataframe(cm_df, width="stretch")

            st.subheader("Classification Report (per model) + Export")
            selected_rep = st.selectbox("Choose a model (report)", options=list(reports.keys()), key="sel_rep_adv")
            rep_df = reports[selected_rep]
            st.dataframe(rep_df, width="stretch")

            st.download_button(
                label=f"⬇️ Download {selected_rep} report (CSV)",
                data=rep_df.to_csv(index=True).encode("utf-8"),
                file_name=f"classification_report_{selected_rep.replace(' ', '_').lower()}.csv",
                mime="text/csv",
            )

            # ZIP all reports
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                for model_name, df_report in reports.items():
                    fname = f"classification_report_{model_name.replace(' ', '_').lower()}.csv"
                    zf.writestr(fname, df_report.to_csv(index=True))
            zip_buffer.seek(0)

            st.download_button(
                label="⬇️ Download ALL classification reports (ZIP)",
                data=zip_buffer.getvalue(),
                file_name="classification_reports.zip",
                mime="application/zip",
            )
