# 📄 README.md

## Machine Learning Classification Assignment

**Streamlit ML Model Comparison Dashboard**

---

# a) Problem Statement

The objective of this assignment is to design and implement a complete Machine Learning classification pipeline and deploy it using Streamlit.

The application must:

• Train **6 classification models** on the **same dataset**
• Evaluate using standard ML metrics
• Support **binary and multi-class datasets**
• Allow users to **upload any dataset and evaluate models interactively**

---

# b) Dataset Description

## Primary Dataset — Breast Cancer Wisconsin (Default Tab)

Source: Scikit-learn Built-in Dataset

| Attribute | Value               |
| --------- | ------------------- |
| Samples   | 569                 |
| Features  | 30                  |
| Classes   | 2                   |
| Target    | Malignant vs Benign |

This dataset is widely used in medical diagnosis classification problems.

The **Default tab of the Streamlit app automatically trains and evaluates all models on this dataset.**

---

## Advanced Tab — Upload Any Dataset

The application also provides an **Advanced Upload tab** where users can:

• Upload any CSV dataset (Kaggle / UCI)
• Automatically detect target column
• Handle missing values
• Apply feature encoding & scaling
• Compare all 6 models interactively

This makes the app reusable for any classification dataset.

---

## Validation Dataset — Titanic Survival Dataset

To validate the flexibility of the app, the **Titanic dataset** from Kaggle was uploaded and tested using the Advanced tab.

| Attribute | Value                   |
| --------- | ----------------------- |
| Samples   | 891                     |
| Classes   | 2                       |
| Target    | Survived / Not Survived |

---

# c) Models Used

The following **6 Machine Learning models** were implemented:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Naive Bayes (Gaussian)
5. Random Forest (Ensemble)
6. XGBoost (Ensemble)

---

# Evaluation Metrics

All models are evaluated using:

| Metric            | Description                    |
| ----------------- | ------------------------------ |
| Accuracy          | Overall prediction correctness |
| AUC               | Ability to distinguish classes |
| Precision (Macro) | Correct positive predictions   |
| Recall (Macro)    | Ability to capture positives   |
| F1 Score (Macro)  | Balance of Precision & Recall  |
| MCC               | Balanced correlation metric    |

---

# d) Model Comparison Table

## Results on Breast Cancer Dataset (Default Tab)

| ML Model Name       | Accuracy   | AUC        | Precision  | Recall     | F1         | MCC        |
| ------------------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| Logistic Regression | **0.9825** | **0.9954** | **0.9812** | **0.9812** | **0.9812** | **0.9623** |
| XGBoost             | 0.9649     | 0.9934     | 0.9737     | 0.9524     | 0.9615     | 0.9258     |
| KNN                 | 0.9561     | 0.9788     | 0.9551     | 0.9504     | 0.9526     | 0.9054     |
| Random Forest       | 0.9474     | 0.9934     | 0.9435     | 0.9435     | 0.9435     | 0.8869     |
| Naive Bayes         | 0.9386     | 0.9878     | 0.9360     | 0.9315     | 0.9337     | 0.8676     |
| Decision Tree       | 0.9123     | 0.9157     | 0.9019     | 0.9157     | 0.9075     | 0.8174     |

---

# Validation Results — Titanic Dataset (Advanced Tab)

| ML Model Name       | Accuracy | AUC        | Precision | Recall | F1     | MCC    |
| ------------------- | -------- | ---------- | --------- | ------ | ------ | ------ |
| Logistic Regression | 0.7933   | 0.7952     | 0.9016    | 0.5034 | 0.5034 | 0.4391 |
| XGBoost             | 0.7709   | **0.8377** | 0.7057    | 0.6050 | 0.6050 | 0.4002 |
| Decision Tree       | 0.7318   | 0.6960     | 0.6447    | 0.5392 | 0.5392 | 0.2687 |
| KNN                 | 0.7263   | 0.6866     | 0.2421    | 0.2805 | 0.2805 | 0.0000 |
| Random Forest       | 0.7263   | 0.7887     | 0.2421    | 0.2805 | 0.2805 | 0.0000 |
| Naive Bayes         | 0.4302   | 0.7486     | 0.6889    | 0.4534 | 0.4534 | 0.3618 |

---

# Observations About Model Performance

| ML Model            | Observation                                                                                                       |
| ------------------- | ----------------------------------------------------------------------------------------------------------------- |
| Logistic Regression | Best overall performer on Breast Cancer dataset. Excellent accuracy, AUC, and MCC indicate strong generalization. |
| XGBoost             | Highest AUC on Titanic dataset and strong performance overall. Excellent probability ranking ability.             |
| KNN                 | Performs well on low-dimensional data (Breast Cancer) but struggles with high-dimensional Titanic dataset.        |
| Naive Bayes         | Performs consistently but assumes feature independence, limiting performance.                                     |
| Random Forest       | Strong ensemble model but slightly lower than Logistic Regression on this dataset.                                |
| Decision Tree       | Lowest performance due to overfitting tendency.                                                                   |

---

# Streamlit Application Features

## Default Tab

• Runs all models on Breast Cancer dataset
• Displays model results sequentially
• Metrics dashboard + confusion matrices

## Advanced Upload Tab

• Upload any dataset of choice
• Automatic preprocessing
• Model comparison dashboard
• Export metrics & classification reports

---

# How to Run

Install dependencies:

```
pip install -r requirements.txt
```

Run Streamlit:

```
streamlit run app.py
```

Open:

```
http://localhost:8501
```

---

# Conclusion

This project demonstrates a full end-to-end ML pipeline with deployment.
The application is reusable for **any classification dataset** and provides an **interactive comparison dashboard** for multiple models.
