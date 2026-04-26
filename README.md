This is a professional, clean version of your **Customer Churn Prediction** README file. I have removed all personal contact information and ownership details, streamlined the formatting for better readability, and ensured the technical insights are front and center.

---

# Customer Churn Prediction Project

This project focuses on analyzing customer behavior within a telecommunications company to predict churn. By identifying at-risk customers, businesses can implement proactive retention strategies.

## Project Overview

- **Dataset:** [Telco Customer Churn Dataset (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Size:** 7,043 customers
- **Features:** 21 variables including demographic, account, and service information.
- **Objective:** Build a machine learning pipeline to identify high-risk churn segments and uncover the primary drivers of customer attrition.

---

## Phase 1: Exploratory Data Analysis (EDA)

The initial analysis focused on understanding the distribution of features and their correlation with the target variable (`Churn`).

### Key Insights:
* **Churn Rate:** The dataset shows a **26.54%** churn rate vs. a **73.46%** retention rate.
* **Tenure:** Strong inverse correlation; customers with longer tenure are significantly less likely to leave.
* **Financial Drivers:** Higher monthly charges correlate with increased churn risk.
* **Demographics:** Senior citizens exhibit a higher likelihood of churn compared to younger demographics.
* **Data Cleaning:** Addressed 11 missing values in `TotalCharges` and corrected data types for numerical analysis.

---

## Phase 2: Modeling & Evaluation

Multiple algorithms were tested to establish a baseline and identify the most effective predictive architecture.

### Model Performance Comparison:

| Model | Accuracy |
| :--- | :--- |
| **Random Forest** | **0.8069** |
| Logistic Regression | 0.8041 |
| Decision Tree | 0.7942 |

### Challenges Addressed:
* **Data Integrity:** Resolved `KeyError` and naming conflicts (`df_new` vs `df_encoded`) during the preprocessing stage.
* **Feature Engineering:** Tested additional features; however, results indicated the original feature set provided the most robust predictive power.
* **Encoding:** Successfully implemented one-hot encoding for categorical variables and binary encoding for the target.

---

## Phase 3: Model Optimization

The final stage involved moving beyond basic models to optimized ensemble methods and refined evaluation techniques.

### Best Model Selection:
* **Model:** Optimized **XGBoost**
* **Final Accuracy:** **0.7984** (Note: Accuracy was slightly lower but more generalized through cross-validation).
* **Deployment File:** `best_churn_model.pkl`

### Top 5 Predictive Features:
1.  **Contract_Two year** (High Retention Indicator)
2.  **InternetService_Fiber optic** (High Churn Risk Indicator)
3.  **Contract_One year**
4.  **PaymentMethod_Electronic check**
5.  **InternetService_No**

---

## Key Learnings & Future Scope

* **Validation Matters:** Cross-validation provided a more realistic estimate of performance than a single train/test split.
* **Algorithm Selection:** XGBoost demonstrated superior handling of tabular data and complex feature interactions.
* **Feature Importance:** Understanding *why* a model predicts churn is as valuable as the prediction itself.

### Upcoming Milestones:
* **Deployment:** Building an interactive web application for real-time predictions.
* **Explainability:** Integrating SHAP or LIME to provide "reasoning" for individual customer risk scores.
* **Class Imbalance:** Further exploring SMOTE or cost-sensitive learning to improve recall for churning customers.

---

## Setup & Installation

To replicate this analysis, install the following dependencies:

```bash
pip install pandas numpy matplotlib seaborn jupyter scikit-learn xgboost
```

### Usage
1. Clone the repository.
2. Download the `WA_Fn-UseC_-Telco-Customer-Churn.csv` from Kaggle.
3. Run the Jupyter notebooks in sequence: `week1_eda.ipynb` followed by the modeling scripts.
