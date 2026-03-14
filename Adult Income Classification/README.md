# 💼 Adult Income Classification

An end-to-end Machine Learning pipeline that predicts whether an individual earns **more than $50K/year** based on US census data.

## 📌 Project Overview

This app builds and serves a **Logistic Regression** binary classifier trained on the US Census dataset. It covers the full ML workflow: EDA → Preprocessing → Model Training → Evaluation → Prediction UI.

## 📂 Dataset

**File:** `census.csv` — US Census data (32,560 records, 15 features)

| Feature | Type | Description |
|---|---|---|
| Age | Numerical | Age of the individual |
| Workclass | Categorical | Employment type |
| Education | Categorical | Highest education level |
| Education_num | Numerical | Education level as number |
| Marital_status | Categorical | Marital status |
| Occupation | Categorical | Type of occupation |
| Relationship | Categorical | Family relationship |
| Race | Categorical | Race |
| Sex | Categorical | Gender |
| Capital_gain | Numerical | Capital gain |
| Capital_loss | Numerical | Capital loss |
| Hours_per_week | Numerical | Working hours per week |
| Native_country | Categorical | Country of origin |
| **Income** | **Target** | **<=50K or >50K** |

## ⚙️ ML Pipeline

| Step | Details |
|---|---|
| **EDA** | Distribution plots, income balance check |
| **Preprocessing** | Replace `?` → drop nulls, One-Hot Encoding, StandardScaler |
| **Model** | `LogisticRegression(max_iter=1000)` from scikit-learn |
| **Evaluation** | Accuracy, Precision, Recall, F1-Score, Confusion Matrix |
| **Prediction** | Interactive form to predict income for a new individual |

## 🚀 Features

- 📊 Interactive EDA charts (income distribution, age & hours distributions)
- ⚙️ Configurable train/test split via sidebar slider
- 📈 Classification report metrics displayed as KPI cards
- 🔥 Confusion Matrix heatmap
- 🎯 Real-time income prediction for new individuals

## 🛠️ Tech Stack

- **Python**
- **Streamlit** — Web app framework
- **scikit-learn** — Logistic Regression, preprocessing, metrics
- **Matplotlib / Seaborn** — Visualizations
- **Pandas / NumPy** — Data processing

## 📦 Installation

```bash
pip install -r requirements.txt
streamlit run main.py
```

## 🌐 Live Demo

👉 [https://5vdjzzxsbwingmdzksfqxy.streamlit.app/](https://5vdjzzxsbwingmdzksfqxy.streamlit.app/)
