# 🫀 Cardiovascular Disease Prediction

A machine learning project that predicts the risk of cardiovascular disease based on patient health data including blood pressure, BMI, cholesterol, and lifestyle factors.

---

## 📁 Project Structure

```
├── HealthCare p.ipynb   # Main notebook (EDA + modeling)
├── health_data.csv          # Dataset
└── README.md
```

---

## 📊 Dataset

The dataset contains medical records for patients with the following features:

| Feature | Description |
|---|---|
| `age` | Age in years |
| `gender` | 1 = Female, 2 = Male |
| `height` | Height in cm |
| `weight` | Weight in kg |
| `ap_hi` | Systolic blood pressure |
| `ap_lo` | Diastolic blood pressure |
| `cholesterol` | 1 = Normal, 2 = Above normal, 3 = Well above normal |
| `gluc` | Glucose level (same scale as cholesterol) |
| `smoke` | Smoking status (binary) |
| `alco` | Alcohol intake (binary) |
| `active` | Physical activity (binary) |
| `cardio` | **Target** — Presence of cardiovascular disease (0 = No, 1 = Yes) |

---

## ⚙️ Feature Engineering

- Converted age from days → years
- Filtered out physiologically impossible blood pressure & height values
- Calculated **BMI** from height and weight
- Created **`bp_status`** column:
  - `0` = Low blood pressure
  - `1` = Normal
  - `2` = High blood pressure

---

## 🔍 Exploratory Data Analysis

- Correlation heatmap with target variable
- Age vs BP Status distribution by cardio risk
- BMI & Age histograms segmented by cardio outcome
- Cardio risk % by age group and activity level

---

## 🤖 Models

### 1. LightGBM Classifier
- Used `class_weight={0:1, 1:1.5}` to handle class imbalance
- Tuned decision threshold to `0.4` for better recall
- Evaluated with Accuracy, ROC-AUC, and Classification Report

### 2. Support Vector Machine (SVM)
- Kernel: RBF
- Applied `StandardScaler` for feature normalization
- One-hot encoded all categorical variables

---

## 📦 Requirements

```bash
pip install pandas numpy matplotlib seaborn scikit-learn lightgbm
```

---

## 🚀 How to Run

1. Clone the repo:
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Open the notebook:
```bash
jupyter notebook session22healthy.ipynb
```

---

## 📈 Results

| Model | Accuracy | ROC-AUC |
|---|---|---|
| LightGBM | ~73% | ~0.80 |
| SVM (RBF) | ~72% | — |

> Results may vary slightly depending on data version and random seed.
