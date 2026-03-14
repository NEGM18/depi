import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import os
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Mall Customer Segmentation", page_icon="🛍️", layout="wide")

st.title("🛍️ Mall Customer Segmentation")
st.markdown("Group mall customers based on their **Annual Income** and **Spending Score** using the **K-Means Clustering** algorithm.")

@st.cache_data
def load_data():
    file_path = "Mall_Customers.csv"
    if not os.path.exists(file_path):
        file_path = os.path.join(os.path.dirname(__file__), "Mall_Customers.csv")
    
    df = pd.read_csv(file_path)
    return df

df = load_data()

st.sidebar.header("Settings")

if st.sidebar.checkbox("Show raw data"):
    st.subheader("Raw Data Summary")
    st.dataframe(df)

X = df.iloc[:, [3, 4]].values 

st.subheader("Elbow Method for Optimal K")
st.markdown("The **Elbow Method** helps identify the optimal number of clusters by finding the 'elbow' point where the Within-Cluster-Sum-of-Squares (WCSS) starts to decrease at a slower rate.")

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', n_init=10, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

fig_elbow, ax_elbow = plt.subplots(figsize=(8, 4))
ax_elbow.plot(range(1, 11), wcss, marker='o', linestyle='-', color='b')
ax_elbow.set_title('The Elbow Method')
ax_elbow.set_xlabel('Number of clusters (K)')
ax_elbow.set_ylabel('WCSS')
ax_elbow.grid(True, linestyle='--', alpha=0.7)

st.pyplot(fig_elbow)

st.divider()
st.subheader("K-Means Clustering Results")
st.markdown("Based on the Elbow curve above, choose the optimal number of clusters.")

k_clusters = st.sidebar.slider("Select the number of clusters (K)", min_value=2, max_value=10, value=5, step=1)

kmeans = KMeans(n_clusters=k_clusters, init='k-means++', n_init=10, random_state=42)
y_kmeans = kmeans.fit_predict(X)

fig_clusters, ax_clusters = plt.subplots(figsize=(10, 6))

colors = sns.color_palette("husl", k_clusters)

for i in range(k_clusters):
    ax_clusters.scatter(X[y_kmeans == i, 0], X[y_kmeans == i, 1], s=100, c=[colors[i]], label=f'Cluster {i+1}')

ax_clusters.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='yellow', label='Centroids', marker='*', edgecolor='black', linewidth=1)
ax_clusters.set_title(f'Clusters of customers (K={k_clusters})')
ax_clusters.set_xlabel('Annual Income (k$)')
ax_clusters.set_ylabel('Spending Score (1-100)')
ax_clusters.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax_clusters.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
st.pyplot(fig_clusters)

st.divider()

st.subheader("Predict Customer Cluster")
st.markdown("Enter the Annual Income and Spending Score of a customer to predict which cluster they belong to.")

col1, col2 = st.columns(2)

with col1:
    new_income = st.number_input("Annual Income (k$)", min_value=0, max_value=500, value=50, step=1)

with col2:
    new_score = st.number_input("Spending Score (1-100)", min_value=1, max_value=100, value=50, step=1)

if st.button("Predict Cluster", type="primary"):
    predicted_cluster = kmeans.predict([[new_income, new_score]])[0]
    
    st.success(f"🎯 Based on the input, this customer belongs to **Cluster {predicted_cluster + 1}**!")
    st.info(f"You can interpret this cluster by looking at the scatter plot above (Cluster {predicted_cluster + 1}).")
