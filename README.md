# ❤️ Heart Disease Prediction System

This project uses Machine Learning techniques to predict the likelihood of heart disease in patients based on clinical data. It leverages the UCI Heart Disease dataset and applies Logistic Regression and Random Forest models to classify patients as either having heart disease (1) or not (0).

## 📌 Features

Data preprocessing (missing values, categorical encoding, scaling).

Exploratory Data Analysis (EDA) with visualizations (correlation heatmap, histograms).

### ML Models:

Logistic Regression (baseline model)

Random Forest (best accuracy: ~88.6%)

Feature importance analysis for key risk factors.

Exported trained model & scaler (.pkl files).

User-friendly input template (heart_user_template.csv).

Supports predictions on new patient data via uploaded CSV.

## 📊 Dataset

### Source: UCI Heart Disease Dataset

### Key features:

Age, Sex, Chest Pain Type (cp)

Resting Blood Pressure (trestbps)

Serum Cholesterol (chol)

Fasting Blood Sugar (fbs)

Resting ECG Results (restecg)

Maximum Heart Rate Achieved (thalach)

Exercise-Induced Angina (exang)

ST Depression (oldpeak), Slope, Ca, Thal

### Target:

num > 0 → Heart Disease (1)

num = 0 → No Heart Disease (0)

## ⚙️ Installation & Setup

Clone the repo and install dependencies:

git clone https://github.com/yourusername/heart-disease-prediction.git
cd heart-disease-prediction
pip install -r requirements.txt

## 🚀 Usage

Run the Jupyter Notebook / Colab Notebook to train models.

Prediction with user data:

Fill in values in heart_user_template.csv.

Run the prediction script:

import joblib, pandas as pd

scaler = joblib.load("heart_scaler.pkl")
model = joblib.load("heart_rf_model.pkl")
user_df = pd.read_csv("heart_user_template.csv")

# preprocess as per pipeline, then:
preds = model.predict(scaler.transform(user_df))
print(preds)


0 = No Heart Disease

1 = Heart Disease

## 📈 Results

Logistic Regression → Moderate accuracy

Random Forest → 88.6% accuracy (best performer)

Feature Importance reveals key predictors:

Age, Cholesterol, Resting BP, Thal, ST depression, etc.

## 📷 Visualizations

Correlation Heatmap

Histograms of key features

Random Forest Feature Importance

## 🛠️ Technologies Used

Python (Pandas, NumPy, Matplotlib, Seaborn)

Scikit-learn (Logistic Regression, Random Forest, preprocessing)

Joblib (Model Saving/Loading)

Google Colab / Jupyter Notebook

## 📌 Future Improvements

Add Deep Learning models (Neural Networks).

Deploy with Streamlit / Flask API for real-time predictions.

Train on larger and more diverse datasets.

Integrate with hospital information systems for clinical use.
