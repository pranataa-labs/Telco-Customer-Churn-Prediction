# 📞 Telco Customer Churn Prediction System

---

## 📌 Project Overview

This project develops an end-to-end Machine Learning classification framework designed to predict customer churn risk for a telecommunications provider. By identifying high-risk churn patterns early, customer success teams can deploy proactive retention interventions (such as loyalty incentives, targeted promotions, or service tier upgrades) to safeguard recurring revenues.

The modeling framework utilizes **Supervised Learning** applied to **Binary Classification** with class-imbalance remediation.

---

## 📊 Dataset Specifications

The dataset utilized is the **Telco Customer Churn** benchmark dataset from Kaggle, capturing customer demographics, subscribed service tiers, contract structures, billing mechanisms, and churn history.

* **Dataset Source:** Kaggle Telco Customer Churn
* **Initial Volume:** 7,043 Customer Records
* **Initial Dimensions:** 21 Feature Attributes

---

## 🔬 End-To-End Data Science Pipeline

### 1. Data Ingestion & Understanding
* Ingests raw CSV records into Pandas DataFrames.
* Inspects data types, summary statistics, and null spaces.
* Identifies whitespace entries in `TotalCharges` requiring type casting to numeric floats.

### 2. Data Cleaning & Feature Preprocessing
* **TotalCharges Reconciliation:** Casts `TotalCharges` to `float64` and resolves 11 missing records.
* **Identifier Pruning:** Drops non-informative `customerID` attributes.
* **Binary Encoding:** Maps binary variables (`gender`, `Partner`, `Dependents`, `PhoneService`, `PaperlessBilling`, `Churn`) to boolean numeric integers (`0` and `1`).
* **One-Hot Encoding:** Encodes nominal categorical features (`MultipleLines`, `InternetService`, `Contract`, `PaymentMethod`).

### 3. Exploratory Data Analysis (EDA) & Target Imbalance
* **Target Distribution:** Uncovers a moderate class imbalance (~26.58% Churn vs. 73.42% Retained).
* **Correlation Modeling:** Identifies principal drivers:
  * **Positive Churn Drivers:** `MonthlyCharges`, `PaperlessBilling_Yes`, `InternetService_Fiber optic`.
  * **Negative Churn Drivers (Retention Anchors):** `tenure`, `Contract_Two year`, `OnlineSecurity_Yes`, `TechSupport_Yes`.

### 4. Stratified Splitting & Class Imbalance Handling
* Partitions data into 80% Training and 20% Out-of-Sample Test splits with stratified target sampling.
* Applies **SMOTE (Synthetic Minority Over-sampling Technique)** on the **Training Set** to synthesize minority samples while preserving virgin distributions on the test set.

### 5. Model Architecture & Training
* Evaluates three competitive algorithms:
  * **Logistic Regression** (L2-Regularized Linear Classifier)
  * **Random Forest Classifier** (Bagging Ensemble of Decision Trees)
  * **XGBoost Classifier** (Gradient-Boosted Tree Ensemble)

### 6. Quantitative Evaluation & Comparative Performance

| Classifier Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 76.12% | 54.09% | **67.11%** | **59.90%** | **0.8230** |
| **Random Forest** | **77.75%** | **57.84%** | 60.16% | 58.98% | 0.8120 |
| **XGBoost** | 76.33% | 55.14% | 58.82% | 56.92% | 0.8099 |

* **Managerial Selection:** **Logistic Regression** was selected as the optimal production model due to superior **Recall (67.11%)** and **ROC-AUC (0.8230)**. In churn prevention, maximizing recall captures the largest fraction of departing customers for proactive retention outreach.

---

## 🚀 Execution & Setup Instructions

1. **Repository Cloning:**
   ```bash
   git clone https://github.com/pranataa-labs/Telco-Customer-Churn-Prediction.git
   cd Telco-Customer-Churn-Prediction
   ```
2. **Environment & Dependency Installation:**
   ```bash
   pip install pandas numpy scikit-learn imbalanced-learn xgboost matplotlib seaborn jupyter
   ```
3. **Run Notebook:**
   Launch `Telco_Customer_Churn_Prediction.ipynb` in JupyterLab or Google Colab and execute all cells.

---

## 🛠️ Technology Stack

* **Python 3.10+**
* **Pandas & NumPy** (Data processing)
* **Scikit-Learn** (Preprocessing, logistic regression, random forest & metrics)
* **Imbalanced-Learn** (SMOTE oversampling)
* **XGBoost** (Gradient boosting)
* **Matplotlib & Seaborn** (ROC curves & confusion matrices)
