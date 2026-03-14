# 🛍️ Mall Customer Segmentation

A Machine Learning project that groups mall customers based on their **Annual Income** and **Spending Score** using **Unsupervised Learning (K-Means Clustering)**.

## 📌 Project Overview

This app applies **K-Means Clustering** to the Mall Customers dataset to identify distinct customer segments. Understanding these segments helps businesses tailor marketing strategies and improve customer experience.

## 📂 Dataset

The dataset `Mall_Customers.csv` contains **200 customers** with the following features:

| Feature | Description |
|---|---|
| CustomerID | Unique customer identifier |
| Gender | Male / Female |
| Age | Age of the customer |
| Annual Income (k$) | Annual income in thousands |
| Spending Score (1-100) | Score assigned based on spending behavior |

> **Features used for clustering:** Annual Income and Spending Score.

## ⚙️ How It Works

1. **Elbow Method** — Calculates WCSS for K=1 to 10 to help determine the optimal number of clusters.
2. **K-Means Clustering** — Groups customers into K clusters using the `k-means++` initialization.
3. **Visualization** — Scatter plot displays each cluster with color-coded data points and centroid markers.
4. **Prediction** — User can input Annual Income and Spending Score to find out which cluster a new customer belongs to.

## 🚀 Features

- 📊 **Interactive Elbow Curve** to find optimal K
- 🎨 **Color-coded Cluster Scatter Plot**
- 🔢 **Sidebar Slider** to choose number of clusters dynamically
- 🎯 **Customer Cluster Predictor** based on user input
- 📋 Optional raw data view

## 🛠️ Tech Stack

- **Python**
- **Streamlit** — Web app framework
- **scikit-learn** — K-Means algorithm
- **Matplotlib / Seaborn** — Visualizations
- **Pandas** — Data manipulation

## 📦 Installation

```bash
pip install -r requirements.txt
streamlit run main.py
```

## 🌐 Live Demo

👉 [https://mall-customer-segmentation-omar-negm.streamlit.app/](https://mall-customer-segmentation-omar-negm.streamlit.app/)
