import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Adult Income Classification",
    page_icon="💼",
    layout="wide"
)

st.title("💼 Adult Income Classification")
st.markdown(
    "An end-to-end ML pipeline that predicts whether an individual earns **>$50K/year** "
    "based on US census data using **Logistic Regression**."
)

@st.cache_data
def load_data():
    kaggle_path = "/kaggle/input/us-adult-income-update/census.csv"
    local_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "census.csv")
    path = kaggle_path if os.path.exists(kaggle_path) else local_path
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    return df

df = load_data()

st.sidebar.header("⚙️ Settings")
show_raw = st.sidebar.checkbox("Show raw data")
test_size = st.sidebar.slider("Test set size (%)", min_value=10, max_value=40, value=20, step=5)

st.header("📊 Exploratory Data Analysis")

col1, col2, col3 = st.columns(3)
col1.metric("Total Records", f"{df.shape[0]:,}")
col2.metric("Features", df.shape[1] - 1)
col3.metric("Missing Values (after clean)", 0)

if show_raw:
    st.subheader("Sample Data")
    st.dataframe(df.head(20), use_container_width=True)

fig_dist, axes = plt.subplots(1, 2, figsize=(12, 4))

income_counts = df["Income"].value_counts()
axes[0].bar(income_counts.index, income_counts.values,
            color=["#4C72B0", "#DD8452"], edgecolor="white")
axes[0].set_title("Income Class Distribution")
axes[0].set_xlabel("Income")
axes[0].set_ylabel("Count")
for i, v in enumerate(income_counts.values):
    axes[0].text(i, v + 100, f"{v:,}\n({v/len(df)*100:.1f}%)", ha="center", fontsize=9)

df.groupby("Income")["Age"].plot(kind="kde", ax=axes[1], legend=True)
axes[1].set_title("Age Distribution by Income Class")
axes[1].set_xlabel("Age")
axes[1].set_ylabel("Density")

plt.tight_layout()
st.pyplot(fig_dist)

fig2, ax2 = plt.subplots(figsize=(12, 3))
df.groupby("Income")["Hours_per_week"].plot(kind="kde", ax=ax2, legend=True)
ax2.set_title("Hours per Week Distribution by Income")
ax2.set_xlabel("Hours per Week")
ax2.set_ylabel("Density")
st.pyplot(fig2)

st.divider()

st.header("🔧 Data Preprocessing")

@st.cache_data
def preprocess(df, test_sz):
    df = df.copy()
    df.replace("?", np.nan, inplace=True)
    rows_before = len(df)
    df.dropna(inplace=True)
    rows_after = len(df)

    df["Income"] = df["Income"].apply(lambda x: 1 if x.strip() == ">50K" else 0)

    if "Fnlwgt" in df.columns:
        df.drop("Fnlwgt", axis=1, inplace=True)

    cat_cols = df.select_dtypes(include="object").columns.tolist()
    num_cols = [c for c in df.select_dtypes(include=np.number).columns if c != "Income"]

    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    X = df_encoded.drop("Income", axis=1)
    y = df_encoded["Income"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_sz / 100, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    num_in_encoded = [c for c in X_train.columns if c in num_cols]
    X_train[num_in_encoded] = scaler.fit_transform(X_train[num_in_encoded])
    X_test[num_in_encoded] = scaler.transform(X_test[num_in_encoded])

    return X_train, X_test, y_train, y_test, scaler, X.columns.tolist(), rows_before, rows_after, cat_cols, num_cols

X_train, X_test, y_train, y_test, scaler, feature_cols, rows_before, rows_after, cat_cols, num_cols = preprocess(df, test_size)

col1, col2, col3 = st.columns(3)
col1.metric("Rows before cleaning", f"{rows_before:,}")
col2.metric("Rows after cleaning (dropped ?)", f"{rows_after:,}")
col3.metric("Features after One-Hot Encoding", len(feature_cols))

with st.expander("ℹ️ Preprocessing steps applied"):
    st.markdown(f"""
    1. **Missing values**: Replaced `?` with NaN → dropped **{rows_before - rows_after}** rows with missing values.
    2. **Target encoding**: `Income` → `1` (>50K), `0` (<=50K).
    3. **Dropped**: `Fnlwgt` (sampling weight – not a predictive feature).
    4. **One-Hot Encoding** on categorical features: `{', '.join(cat_cols)}`.
    5. **StandardScaler** applied to numerical features: `{', '.join(num_cols)}`.
    6. **Train/Test split**: {100 - test_size}% / {test_size}%
    """)

st.divider()

st.header("🤖 Model Training — Logistic Regression")

@st.cache_resource
def train_model(X_train, y_train):
    model = LogisticRegression(max_iter=1000, random_state=42, solver="lbfgs")
    model.fit(X_train, y_train)
    return model

model = train_model(X_train.values, y_train.values)
y_pred = model.predict(X_test.values)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=["<=50K", ">50K"], output_dict=True)

col1, col2, col3, col4 = st.columns(4)
col1.metric("✅ Accuracy", f"{accuracy*100:.2f}%")
col2.metric("Precision (>50K)", f"{report['>50K']['precision']*100:.1f}%")
col3.metric("Recall (>50K)", f"{report['>50K']['recall']*100:.1f}%")
col4.metric("F1-Score (>50K)", f"{report['>50K']['f1-score']*100:.1f}%")

st.subheader("Confusion Matrix")
fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm,
            xticklabels=["<=50K", ">50K"], yticklabels=["<=50K", ">50K"])
ax_cm.set_xlabel("Predicted")
ax_cm.set_ylabel("Actual")
ax_cm.set_title("Confusion Matrix")
st.pyplot(fig_cm)

st.divider()

st.header("🎯 Predict Income Class for a New Individual")
st.markdown("Fill in the details below and click **Predict** to find out whether this person likely earns **>$50K/year**.")

workclasses = sorted([x for x in df["Workclass"].unique() if x != "?"])
educations = sorted([x for x in df["Education"].unique()])
marital_statuses = sorted([x for x in df["Marital_status"].unique()])
occupations = sorted([x for x in df["Occupation"].unique() if x != "?"])
relationships = sorted([x for x in df["Relationship"].unique()])
races = sorted([x for x in df["Race"].unique()])
native_countries = sorted([x for x in df["Native_country"].unique() if x != "?"])

col1, col2, col3 = st.columns(3)
with col1:
    age = st.number_input("Age", min_value=17, max_value=90, value=35)
    workclass = st.selectbox("Workclass", workclasses)
    education = st.selectbox("Education", educations)
    education_num = st.number_input("Education Num (years)", min_value=1, max_value=16, value=10)

with col2:
    marital_status = st.selectbox("Marital Status", marital_statuses)
    occupation = st.selectbox("Occupation", occupations)
    relationship = st.selectbox("Relationship", relationships)
    race = st.selectbox("Race", races)

with col3:
    sex = st.selectbox("Sex", ["Male", "Female"])
    capital_gain = st.number_input("Capital Gain", min_value=0, max_value=100000, value=0, step=100)
    capital_loss = st.number_input("Capital Loss", min_value=0, max_value=5000, value=0, step=100)
    hours_per_week = st.number_input("Hours per Week", min_value=1, max_value=99, value=40)
    native_country = st.selectbox("Native Country", native_countries)

if st.button("🔍 Predict Income Class", type="primary"):
    input_dict = {
        "Age": [age],
        "Workclass": [workclass],
        "Education": [education],
        "Education_num": [education_num],
        "Marital_status": [marital_status],
        "Occupation": [occupation],
        "Relationship": [relationship],
        "Race": [race],
        "Sex": [sex],
        "Capital_gain": [capital_gain],
        "Capital_loss": [capital_loss],
        "Hours_per_week": [hours_per_week],
        "Native_country": [native_country],
    }
    input_df = pd.DataFrame(input_dict)

    df_train_raw = df.copy()
    df_train_raw.replace("?", np.nan, inplace=True)
    df_train_raw.dropna(inplace=True)
    df_train_raw["Income"] = df_train_raw["Income"].apply(lambda x: 1 if ">50K" in x else 0)
    if "Fnlwgt" in df_train_raw.columns:
        df_train_raw.drop(["Fnlwgt", "Income"], axis=1, inplace=True)
    else:
        df_train_raw.drop(["Income"], axis=1, inplace=True)

    combined = pd.concat([df_train_raw, input_df], ignore_index=True)
    combined_encoded = pd.get_dummies(combined, drop_first=True)
    input_encoded = combined_encoded.iloc[[-1]]

    for col in feature_cols:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    input_encoded = input_encoded[feature_cols]

    num_in_encoded = [c for c in input_encoded.columns if c in num_cols]
    input_encoded[num_in_encoded] = scaler.transform(input_encoded[num_in_encoded])

    prediction = model.predict(input_encoded.values)[0]
    probability = model.predict_proba(input_encoded.values)[0]

    if prediction == 1:
        st.success(f"💰 Prediction: This individual likely earns **>$50K/year**")
    else:
        st.info(f"📊 Prediction: This individual likely earns **≤$50K/year**")

    col_a, col_b = st.columns(2)
    col_a.metric("Probability of ≤$50K", f"{probability[0]*100:.1f}%")
    col_b.metric("Probability of >$50K", f"{probability[1]*100:.1f}%")
