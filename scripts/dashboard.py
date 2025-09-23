import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("🔥 Wildfire Surge Prediction Dashboard")

# ---------------------- File Selector ----------------------
st.sidebar.header("📁 Choose Prediction Source")
file_option = st.sidebar.selectbox("Select prediction file:", ["predictions.csv", "rolling_predictions.csv"])
df = pd.read_csv(f"data/{file_option}")

# ---------------------- Date Conversion ----------------------
df["date"] = pd.to_datetime(df["date"], errors="coerce")

# ---------------------- District Filter (if available) ----------------------
if "district" in df.columns:
    st.sidebar.header("🗺️ Filter by District")
    selected_district = st.sidebar.selectbox("Choose a district:", sorted(df["district"].unique()))
    df = df[df["district"] == selected_district]

# ---------------------- Surge Timeline ----------------------
st.subheader("📈 Surge Detection Over Time")
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(df["date"], df["actual"], label="Actual", color="firebrick", linewidth=2)
ax.plot(df["date"], df["predicted"], label="Predicted", linestyle="--", color="dodgerblue")
ax.set_xlabel("Date")
ax.set_ylabel("Surge Flag")
ax.legend()
st.pyplot(fig)

# ---------------------- Probability Distribution ----------------------
if "probability" in df.columns:
    st.subheader("📊 Surge Probability Distribution")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df["probability"], bins=30, color="orange", edgecolor="black")
    ax.set_xlabel("Surge Probability")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Surge Probabilities")
    st.pyplot(fig)
    st.caption("Higher probabilities indicate stronger surge signals.")

# ---------------------- Feature Explorer ----------------------
numeric_cols = df.select_dtypes(include="number").columns.tolist()
excluded = ["actual", "predicted", "probability"]
feature_cols = [col for col in numeric_cols if col not in excluded]

if feature_cols:
    st.subheader("🧪 Feature Explorer")
    selected_feature = st.selectbox("Choose a feature to visualize:", feature_cols)
    fig2, ax2 = plt.subplots(figsize=(10, 3))
    ax2.plot(df["date"], df[selected_feature], color="teal")
    ax2.set_title(f"{selected_feature} over time")
    st.pyplot(fig2)

# ---------------------- Feature Importance ----------------------
try:
    importance_df = pd.read_csv("data/feature_importance.csv")
    st.subheader("🔍 Top Features Driving Surge Predictions")
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    ax3.barh(importance_df["Feature"], importance_df["Importance"], color="forestgreen")
    ax3.set_xlabel("Importance Score")
    ax3.set_title("🔥 Feature Importance")
    ax3.invert_yaxis()
    st.pyplot(fig3)
except FileNotFoundError:
    st.warning("Feature importance file not found.")

# ---------------------- Summary Metrics ----------------------
st.subheader("📊 Surge Detection Summary")
total = len(df)
surges = df["actual"].sum()
predicted_surges = df["predicted"].sum()
correct = (df["actual"] == df["predicted"]).sum()
accuracy = correct / total if total > 0 else 0

st.markdown(f"""
- **Total Samples**: {total}  
- **Actual Surges**: {surges}  
- **Predicted Surges**: {predicted_surges}  
- **Correct Predictions**: {correct}  
- **Model Accuracy**: {accuracy:.2%}
""")
