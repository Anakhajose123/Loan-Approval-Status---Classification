# 🏦 Loan Prediction Classification Project

## 📌 Project Overview

Loan approval is a crucial decision-making process for financial institutions.
This project aims to **predict whether a loan will be approved or not** based on applicant details using **Machine Learning classification algorithms**.

The project covers **end-to-end steps**:

* Data Understanding
* Exploratory Data Analysis (EDA)
* Data Preprocessing & Feature Engineering
* Model Building
* Model Evaluation
* Best Model Selection

---

## 🎯 Problem Statement

Given customer information such as income, education, employment status, credit history, and property area, predict:

> **Will the loan be approved? (Yes / No)**

This is a **Binary Classification Problem**.

---

## 📂 Dataset Description

The dataset contains customer and loan-related features.

### 🔹 Features

| Column Name       | Description                        |
| ----------------- | ---------------------------------- |
| Loan_ID           | Unique loan identifier             |
| Gender            | Male / Female                      |
| Married           | Applicant marital status           |
| Dependents        | Number of dependents               |
| Education         | Graduate / Not Graduate            |
| Self_Employed     | Self-employed status               |
| ApplicantIncome   | Applicant income                   |
| CoapplicantIncome | Co-applicant income                |
| LoanAmount        | Loan amount requested              |
| Loan_Amount_Term  | Loan repayment term                |
| Credit_History    | Credit history (1 = Good, 0 = Bad) |
| Property_Area     | Urban / Semiurban / Rural          |
| Loan_Status       | **Target variable (Y / N)**        |

---

## 🧪 Exploratory Data Analysis (EDA)

### ✔ Steps Performed:

* Dataset shape, data types, and summary statistics
* Missing value detection and treatment
* Univariate analysis (categorical & numerical)
* Bivariate analysis with target variable
* Correlation analysis
* Outlier detection
* Data distribution and skewness analysis

### 📊 Key Insights:

* **Credit_History** has the strongest influence on loan approval
* Applicants with higher income and good credit history have higher approval chances
* Certain categorical features show clear approval patterns

---

## 🛠 Data Preprocessing

* Handled missing values using **median (numerical)** and **mode (categorical)**
* Encoded categorical variables using:

  * Label Encoding
    
* Feature scaling using **StandardScaler**
* Feature engineering:

  * Total Income

---

## 🤖 Machine Learning Models Used

The following classification models were trained and evaluated:

1. **Logistic Regression**
2. **KNN**
3. **Naive Bayes**
4. **SVC**
5. **Decision Tree Classifier**
6. **Random Forest Classifier**
7. **Gradient Boosting**

---

## 📈 Model Evaluation Metrics

Models were evaluated using:

* Accuracy Score
* Confusion Matrix
* Precision, Recall, F1-score
* ROC Curve

### 🏆 Best Model

✅ **Gradient Boosting Classifer** performed best with higher accuracy and balanced performance.

---

## 🔍 Feature Importance

Feature importance analysis showed that:

* Credit_History
* Applicant Income
* Loan Amount
  are key contributors in predicting loan approval.

---

## 📁 Project Structure

```
Loan_Prediction_Classification/
│
├── data/
│   └── loan_approval_dataset.csv
│
├── notebook/
│   └── CL - Assignment.ipynb
│
├── README.md
│
└── requirements.txt
```

---

## 🧰 Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Jupyter Notebook

---

## 🚀 Conclusion

This project demonstrates a **complete machine learning pipeline**, from raw data analysis to model evaluation.
Proper EDA and feature engineering significantly improved the prediction performance.

---

## 🙌 Author

**Anakha Jose**
Aspiring Data Scientist | Machine Learning Enthusiast

---
