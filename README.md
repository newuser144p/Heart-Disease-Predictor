# 🫀 Heart Disease Predictor

A Streamlit web application that predicts the likelihood of heart disease using multiple machine learning models trained on clinical patient data.

---

## 📋 Overview

This app allows users to input clinical parameters and receive heart disease predictions from four trained ML models simultaneously. It supports both single-patient prediction via a form and bulk prediction via CSV upload.

---

## ✨ Features

- **Single Patient Prediction** — Enter clinical details and get instant predictions from all 4 models
- **Bulk CSV Prediction** — Upload a CSV file to run Logistic Regression predictions on multiple patients at once
- **Model Performance Dashboard** — Visual bar chart comparing the accuracy of all models
- **Download Results** — Export bulk predictions as a CSV file

---

## 🤖 Models Included

| Model | Accuracy |
|---|---|
| Logistic Regression | 85.86% |
| Random Forest | 84.23% |
| Support Vector Machine | 84.22% |
| Decision Tree | 80.97% |

Pre-trained models are stored as `.pkl` files and loaded at runtime.

---

## 🩺 Input Features

| Feature | Description |
|---|---|
| `Age` | Age of the patient (years) |
| `Sex` | Sex of the patient (Male / Female) |
| `ChestPainType` | Typical Angina, Atypical Angina, Non-Angina Pain, or Asymptomatic |
| `RestingBP` | Resting blood pressure (mm/Hg) |
| `Cholesterol` | Serum cholesterol (mm/dl) |
| `FastingBS` | Fasting blood sugar (≤ 120 mg/dl or > 120 mg/dl) |
| `RestingECG` | Resting ECG results (Normal / ST-T Wave Abnormality / Left Ventricular Hypertrophy) |
| `MaxHR` | Maximum heart rate achieved (60–202) |
| `ExerciseAngina` | Exercise-induced angina (Yes / No) |
| `Oldpeak` | ST depression induced by exercise relative to rest (0.0–10.0) |
| `ST_Slope` | Slope of the peak exercise ST segment (Upsloping / Flat / Downsloping) |

---

## 🗂️ Project Structure

```
HeartDisease/
├── app.py                  # Main Streamlit application
├── tree.pkl                # Trained Decision Tree model
├── LogisticRegression.pkl  # Trained Logistic Regression model
├── RandomForest.pkl        # Trained Random Forest model
└── SVM.pkl                 # Trained Support Vector Machine model
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/heart-disease-predictor.git
   cd heart-disease-predictor
   ```

2. **Install dependencies**
   ```bash
   pip install streamlit pandas numpy scikit-learn plotly
   ```

3. **Run the app**
   ```bash
   streamlit run app.py
   ```

4. Open your browser and navigate to `http://localhost:8501`

---

## 📂 Bulk Prediction CSV Format

For the bulk prediction tab, upload a CSV file with the following columns (no NaN values):

```
Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope
```

**Encoding conventions:**
- `Sex`: 0 = Male, 1 = Female
- `ChestPainType`: 0 = Typical Angina, 1 = Atypical Angina, 2 = Non-Angina Pain, 3 = Asymptomatic
- `FastingBS`: 0 = ≤ 120 mg/dl, 1 = > 120 mg/dl
- `RestingECG`: 0 = Normal, 1 = ST-T Wave Abnormality, 2 = Left Ventricular Hypertrophy
- `ExerciseAngina`: 0 = No, 1 = Yes
- `ST_Slope`: 0 = Upsloping, 1 = Flat, 2 = Downsloping

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit
- **Data Processing:** Pandas, NumPy
- **ML Models:** Scikit-learn
- **Visualization:** Plotly Express

---

## ⚠️ Disclaimer

This application is intended for **educational and research purposes only**. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider for medical decisions.
