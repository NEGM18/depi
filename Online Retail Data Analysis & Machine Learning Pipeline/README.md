# 🛒 Online Retail Data Analysis & Machine Learning Pipeline

A complete end-to-end ML workflow built with Streamlit, based on the **UCI Online Retail II Dataset** (transactional data from a UK online retailer, 2009–2011).

## 📌 Project Objective

Guide students through a full ML workflow starting from raw transactional data, covering:
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Unsupervised Learning (Clustering)
- Supervised Learning (Classification)
- Time Series Analysis & Forecasting

## 🗂️ Dataset

**UCI Online Retail II** — [Kaggle Link](https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci)

| Column | Description |
|---|---|
| InvoiceNo | Invoice number (prefix 'C' = cancellation) |
| StockCode | Product code |
| Description | Product name |
| Quantity | Number of items (negative = returns) |
| InvoiceDate | Transaction date/time |
| UnitPrice | Price per unit (£) |
| CustomerID | Customer identifier |
| Country | Country of the customer |

## 🧪 6-Part ML Pipeline

| Part | Content |
|---|---|
| **Part 1: EDA** | Shape, dtypes, missing values, 5 visualizations |
| **Part 2: Preprocessing** | Per-column cleaning with explanations |
| **Part 3: Feature Engineering** | Revenue, RFM, return features, time-based features |
| **Part 4: Clustering** | KMeans + Elbow Method + PCA visualization + business interpretation |
| **Part 5: Classification** | High-value customer prediction with Random Forest |
| **Part 6: Time Series** | Weekly trend, seasonality heatmap, linear forecast |

## 🛠️ Tech Stack

- **Python** · **Streamlit** · **Pandas** · **NumPy**
- **scikit-learn** · **Matplotlib** · **Seaborn**

## 📦 Installation

```bash
pip install -r requirements.txt
streamlit run main.py
```

## 🌐 Live Demo

👉 [Deployment Link]
