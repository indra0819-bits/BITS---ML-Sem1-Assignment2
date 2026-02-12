# 📄 README.md

## Machine Learning Classification Assignment

---

# a) Problem Statement

The objective of this assignment is to build and evaluate multiple Machine Learning classification models on a public dataset.
The application must:

• Support both **binary and multi-class classification**
• Train **6 different ML models** on the **same dataset**
• Evaluate models using standard classification metrics
• Deploy the solution using **Streamlit**

For this implementation, two datasets were used:

1. **Default Dataset** – Breast Cancer Wisconsin dataset (Scikit-learn)
2. **Custom Dataset** – Titanic Survival Dataset (Kaggle)

The Streamlit app allows:

* Running models on default dataset
* Uploading any CSV dataset and running the full pipeline

---

# b) Dataset Description

## 1) Breast Cancer Dataset (Default)

Source: Scikit-learn built-in dataset

| Attribute | Value              |
| --------- | ------------------ |
| Samples   | 569                |
| Features  | 30                 |
| Classes   | 2                  |
| Target    | Malignant / Benign |

This dataset is commonly used for binary classification problems in healthcare.

---

## 2) Titanic Dataset (Advanced Upload)

Source: Kaggle Titanic Survival Dataset

| Attribute               | Value                   |
| ----------------------- | ----------------------- |
| Samples                 | 891                     |
| Features after encoding | 1730                    |
| Classes                 | 2                       |
| Target                  | Survived / Not Survived |

The dataset predicts whether a passenger survived the Titanic disaster based on demographic and travel features.

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

Each model was evaluated using:

| Metric            | Description                        |
| ----------------- | ---------------------------------- |
| Accuracy          | Overall correctness of model       |
| AUC               | Ability to distinguish classes     |
| Precision (Macro) | Correct positive predictions       |
| Recall (Macro)    | Ability to capture positives       |
| F1 Score (Macro)  | Balance between Precision & Recall |
| MCC               | Balanced correlation metric        |

---

# d) Model Comparison Table (Titanic Dataset)

| ML Model Name       | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
| ------------------- | -------- | ------ | --------- | ------ | ------ | ------ |
| Logistic Regression | 0.7933   | 0.7952 | 0.9016    | 0.5034 | 0.5034 | 0.4391 |
| Decision Tree       | 0.7318   | 0.6960 | 0.6447    | 0.5392 | 0.5392 | 0.2687 |
| KNN                 | 0.7263   | 0.6866 | 0.2421    | 0.2805 | 0.2805 | 0.0000 |
| Naive Bayes         | 0.4302   | 0.7486 | 0.6889    | 0.4534 | 0.4534 | 0.3618 |
| Random Forest       | 0.7263   | 0.7887 | 0.2421    | 0.2805 | 0.2805 | 0.0000 |
| XGBoost             | 0.7709   | 0.8377 | 0.7057    | 0.6050 | 0.6050 | 0.4002 |

---

# Observations About Model Performance

| ML Model            | Observation                                                                                                               |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| Logistic Regression | Best overall balance of performance. High precision and strong MCC indicate good generalization and balanced predictions. |
| Decision Tree       | Moderate performance. Captures nonlinear patterns but prone to overfitting.                                               |
| KNN                 | Poor performance due to high dimensionality (1730 features). Suffers from curse of dimensionality.                        |
| Naive Bayes         | Very low accuracy but reasonable AUC, indicating probability ranking ability despite poor classification.                 |
| Random Forest       | Accuracy decent but poor macro metrics suggest class imbalance and weak minority class prediction.                        |
| XGBoost             | Best AUC score. Strong probability ranking and good overall performance, second best model after Logistic Regression.     |

---

# Streamlit Application Features

The deployed application supports:

### Default Tab

• Runs all models on Breast Cancer dataset
• Displays metrics sequentially for each model

### Advanced Upload Tab

• Upload any CSV dataset
• Automatic target detection
• Missing value handling
• Feature scaling toggle
• Model comparison dashboard
• Export metrics & classification reports

---

# How to Run the App

Install dependencies:

```
pip install -r requirements.txt
```

Run Streamlit:

```
streamlit run app.py
```

Open browser:

```
http://localhost:8501
```

---

# Conclusion

This project demonstrates a complete end-to-end Machine Learning pipeline including:

• Data preprocessing
• Model training and evaluation
• Performance comparison
• Interactive deployment using Streamlit

The comparison shows that **Logistic Regression and XGBoost** perform best for the Titanic dataset, while **KNN struggles in high-dimensional feature space**.
