import io
import zipfile
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from collections import Counter

from model.models import (
    apply_missing_value_strategy,
    prepare_X_y,
    build_models,
    evaluate_all_models,
    ensure_model_dir,
)

# -------------------------
# Page Config
# -------------------------
st.set_page_config(page_title="ML Classification Benchmark (Advanced)", layout="wide")
st.title("ML Classification Benchmark")

# -------------------------
# Session State
# -------------------------
if "adv_state" not in st.session_state:
    st.session_state.adv_state = {
        "metrics_df": None,
        "results": None,
        "class_labels": None,
        "reports": None,
        "last_signature": None,
        "X_test": None,
        "y_test": None,
    }

# Reset cached results (fixes ROC/proba issues due to old state)
if st.sidebar.button("Reset cached results"):
    st.session_state.adv_state = {
        "metrics_df": None,
        "results": None,
        "class_labels": None,
        "reports": None,
        "last_signature": None,
        "X_test": None,
        "y_test": None,
    }
    st.rerun()

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
Upload any classification CSV (Kaggle/UCI). The app trains **6 models** on the same dataset:

**Logistic Regression  / Decision Tree  / KNN  / Naive Bayes (Gaussian)  / Random Forest  / XGBoost**

**Metrics:** Accuracy, AUC (OvR macro), Precision/Recall/F1 (macro), MCC  
**Extras:** ROC Curve (binary + multiclass OvR), confusion matrix, report exports
"""
    )

# -------------------------
# Sidebar Controls
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


# -------------------------
# Safe stratify helper
# -------------------------
def safe_stratify(y, test_size_val: float):
    counts = Counter(y)
    min_count = min(counts.values())
    n_classes = len(counts)

    if min_count < 2:
        return None, f"Stratify disabled: at least one class has only {min_count} sample(s)."

    n = len(y)
    n_test = int(np.ceil(test_size_val * n))
    if n_test < n_classes:
        return None, (
            f"Stratify disabled: test set size ({n_test}) < number of classes ({n_classes}). "
            f"Increase test_size."
        )

    return y, None


# -------------------------
# ROC plotter (binary + multiclass OvR)
# -------------------------
def plot_roc_from_proba(model_name: str, y_test, y_proba, class_labels):
    if y_test is None:
        st.warning("ROC curve not available: missing y_test. Please re-train.")
        return
    if y_proba is None:
        st.warning("ROC curve not available: model did not return probabilities/scores. Please re-train.")
        return

    n_classes = len(class_labels)
    fig, ax = plt.subplots()

    if n_classes == 2:
        fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        ax.plot([0, 1], [0, 1], linestyle="--")
        ax.set_title(f"ROC Curve — {model_name}")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right")
    else:
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve((y_test == i).astype(int), y_proba[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f"{class_labels[i]} (AUC={roc_auc:.3f})")

        ax.plot([0, 1], [0, 1], linestyle="--")
        ax.set_title(f"ROC Curve (OvR) — {model_name}")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right")

    st.pyplot(fig)


# -------------------------
# Main: Load Data
# -------------------------
if uploaded is None:
    st.info("Upload a CSV to begin.")
    st.stop()

df = pd.read_csv(uploaded)

with st.expander("Preview data", expanded=True):
    max_prev = max(10, min(200, len(df)))
    default_prev = min(50, max_prev)
    step_prev = 10 if max_prev >= 20 else 1
    n_prev = st.slider("Rows to preview", 10, max_prev, default_prev, step_prev, key="prev_rows")
    st.dataframe(df.head(n_prev), width="stretch")

st.divider()

# Under preview: Target selection + Train button
col_target, col_train = st.columns([1, 1])

with col_target:
    st.subheader("Target Variable")
    auto_target = df.columns[-1]
    default_index = list(df.columns).index(auto_target)
    target_col = st.selectbox(
        f"Auto-detected = last column ({auto_target})",
        options=list(df.columns),
        index=default_index,
        key="target_col",
    )

with col_train:
    st.subheader("Train & Evaluate")
    st.write("")
    st.write("")
    run_btn = st.button("Train & Evaluate Models")

# Missing values
df_clean = apply_missing_value_strategy(df, missing_strategy)
if df_clean.shape[0] != df.shape[0]:
    st.warning(f"Rows changed after missing value handling: {df.shape[0]} → {df_clean.shape[0]}")

# Prepare X/y (robust: y always label-encoded to 0..K-1)
try:
    X, y, class_labels = prepare_X_y(df_clean, target_col=target_col)
except Exception as e:
    st.error(f"CSV processing error: {e}")
    st.stop()

n_classes = len(class_labels)

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

# Clear cached results if dataset/settings changed
signature = (X.shape[0], X.shape[1], target_col, missing_strategy, use_scaling, test_size, random_state)
if st.session_state.adv_state["last_signature"] is not None and st.session_state.adv_state["last_signature"] != signature:
    st.session_state.adv_state.update(
        {"metrics_df": None, "results": None, "class_labels": None, "reports": None, "X_test": None, "y_test": None}
    )
st.session_state.adv_state["last_signature"] = signature


# -------------------------
# Train & Evaluate
# -------------------------
if run_btn:
    ensure_model_dir()

    stratify_val = None
    warn = None
    if use_stratify and n_classes > 1:
        stratify_val, warn = safe_stratify(y, test_size)
    if warn:
        st.warning(warn)

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

    # Persist
    st.session_state.adv_state["metrics_df"] = metrics_df
    st.session_state.adv_state["results"] = results
    st.session_state.adv_state["reports"] = reports
    st.session_state.adv_state["class_labels"] = class_labels
    st.session_state.adv_state["X_test"] = X_test
    st.session_state.adv_state["y_test"] = y_test

    st.success("Training complete! ✅")


# -------------------------
# Display Results
# -------------------------
state = st.session_state.adv_state
if state["metrics_df"] is None:
    st.info("Click **Train & Evaluate Models** to generate results.")
    st.stop()

metrics_df = state["metrics_df"]
results = state["results"]
reports = state["reports"]
class_labels = state["class_labels"]
X_test = state.get("X_test")
y_test = state.get("y_test")

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
m = metrics_df.loc[metrics_df["Model"] == model_name].iloc[0].to_dict()	

left_md, right_md = st.columns([1, 1])

with left_md:
	st.write("### Key Metrics")
	cols = st.columns(2)
	cols[0].metric("Accuracy", f"{m['Accuracy']:.4f}")
	cols[0].metric("Precision", f"{m['Precision (macro)']:.4f}")
	cols[0].metric("Recall", f"{m['Recall (macro)']:.4f}")
	cols[1].metric("AUC", f"{m['AUC (OvR macro)']:.4f}" if not np.isnan(m["AUC (OvR macro)"]) else "NA")
	cols[1].metric("F1", f"{m['F1 (macro)']:.4f}")
	cols[1].metric("MCC", f"{m['MCC']:.4f}")


with right_md:
    st.write("### ROC Curve")
    plot_roc_from_proba(model_name, y_test, out.get("y_proba"), class_labels)


st.write("**Confusion Matrix**")
cm_df = pd.DataFrame(
    out["cm"],
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
