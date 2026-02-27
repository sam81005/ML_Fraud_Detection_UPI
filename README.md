# 💳 UPI Fraud Detection System (Machine Learning)

A Machine Learning-based Fraud Detection System designed to identify fraudulent UPI (Unified Payments Interface) transactions using advanced classification algorithms.

This project focuses on building a production-ready fraud detection pipeline including data preprocessing, feature engineering, model training, evaluation, and deployment-ready artifacts.

---

## 📌 Project Overview

Digital payment systems like UPI are highly vulnerable to fraud. This project aims to:

* Detect fraudulent UPI transactions
* Reduce false positives
* Improve fraud detection accuracy using ensemble learning
* Deploy a ready-to-use prediction system

The final solution includes trained models, preprocessing pipelines, and a prediction interface.

---

## 🛠️ Tech Stack

* Python 3
* NumPy
* Pandas
* Scikit-learn
* LightGBM
* Joblib (Model Serialization)

---

## 🧠 Model Used

### 🔹 LightGBM Classifier

LightGBM was selected due to:

* High performance on tabular data
* Faster training speed
* Better handling of imbalanced datasets
* Efficient gradient boosting framework

---

## 📂 Project Structure

```
ML_Fraud_Detection_UPI/
│
├── app.py                          # Prediction / Deployment script
├── train_model.py                  # Model training pipeline
├── lgbm_upi_model.joblib           # Trained LightGBM model
├── lgbm_model_columns.joblib       # Feature column reference
├── real_fraud_scaler.joblib        # Data scaler for preprocessing
├── ultimate_upi_model.joblib       # Optimized final model
├── ultimate_scaler.joblib          # Final preprocessing scaler
├── ultimate_model_columns.joblib   # Final feature reference
└── README.md
```

---

## ⚙️ ML Pipeline

### 1️⃣ Data Preprocessing

* Handling missing values
* Feature scaling using StandardScaler
* Feature alignment using saved column structure

### 2️⃣ Model Training

* Train-test split
* LightGBM classifier training
* Hyperparameter tuning
* Model evaluation using classification metrics

### 3️⃣ Model Evaluation

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

### 4️⃣ Model Saving

* Model saved using Joblib
* Scaler and feature columns saved separately
* Deployment-ready artifacts generated

---

## 🚀 How to Run

### 🔹 Train the Model

```bash
python train_model.py
```

### 🔹 Run Prediction App

```bash
python app.py
```

---

## 📊 Key Features

* Handles imbalanced fraud datasets
* Production-ready model serialization
* Separate scaler & feature alignment system
* Clean modular training pipeline
* Easy deployment integration

---

## 📈 Future Improvements

* Add real-time API deployment using Flask/FastAPI
* Implement SMOTE for imbalance handling
* Add model explainability using SHAP
* Deploy on cloud (AWS / Azure)
* Add Streamlit dashboard

---

## 🎯 Learning Outcomes

Through this project, I gained hands-on experience with:

* End-to-end ML pipeline development
* Feature engineering for financial fraud detection
* Gradient boosting models (LightGBM)
* Handling imbalanced classification problems
* Model deployment preparation

---

## 👨‍💻 Author

**Samarth Karmakar**
B.Tech Computer Science
Machine Learning & AI Enthusiast

GitHub: [https://github.com/sam81005](https://github.com/sam81005)
