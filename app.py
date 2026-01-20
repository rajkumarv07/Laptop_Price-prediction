import streamlit as st
import pandas as pd
import joblib
import json
import os

# ------------------ Load Model & Data ------------------
try:
    # Auto-train if model or data is missing
    if not os.path.exists("laptop_price_model.pkl") or not os.path.exists("laptop (3).csv"):
        import train_model  # must generate: laptop_price_model.pkl, model_analysis.json

    model = joblib.load("laptop_price_model.pkl")

    with open("model_analysis.json") as f:
        analysis = json.load(f)

    df = pd.read_csv("laptop (3).csv")
    df = df.drop(columns=[c for c in df.columns if "Unnamed" in c]).dropna()

except Exception as e:
    st.error(f"⚠️ Error loading model or data: {e}")
    st.stop()

# ------------------ Streamlit Config ------------------
st.set_page_config(page_title="Laptop Price Predictor", layout="wide")
st.title("💻 Laptop Price Predictor – SmartTech")

tab1, tab2 = st.tabs(["🔮 Price Predictor", "📊 Model Analysis"])

# ------------------ Predictor Tab ------------------
with tab1:
    st.subheader("Enter Laptop Specifications")

    company = st.selectbox("Company", sorted(df["Company"].unique()))
    type_name = st.selectbox("Type", sorted(df["TypeName"].unique()))
    inches = st.number_input("Screen Size (Inches)", 10.0, 20.0, 15.6)

    screen = st.selectbox("Screen Resolution", sorted(df["ScreenResolution"].unique()))
    cpu = st.selectbox("CPU", sorted(df["Cpu"].unique()))
    ram = st.number_input("RAM (GB)", 2, 64, 16)

    storage = st.number_input("Total Storage (GB)", 64, 4096, 512)
    gpu = st.selectbox("GPU", sorted(df["Gpu"].unique()))
    os_choice = st.selectbox("Operating System", sorted(df["OpSys"].unique()))
    weight = st.number_input("Weight (kg)", 0.5, 5.0, 1.9)

    if st.button("Predict Price"):
        data = pd.DataFrame([{
            "Company": company,
            "TypeName": type_name,
            "Inches": inches,
            "ScreenResolution": screen,
            "Cpu": cpu,
            "Ram": ram,
            "StorageGB": storage,
            "Gpu": gpu,
            "OpSys": os_choice,
            "Weight": weight
        }])

        try:
            price = model.predict(data)[0]
            st.success(f"💰 Predicted Price: ₹ {price:,.2f}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ------------------ Analysis Tab ------------------
with tab2:
    st.subheader("Model Performance Comparison")

    results_df = pd.DataFrame(analysis["results"]).T
    st.dataframe(results_df, use_container_width=True)

    st.markdown(f"### 🏆 Best Model: **{analysis['best_model']}**")

    st.markdown("""
    **Metrics Explanation:**
    - **R²**: How well the model explains price variance  
    - **MAE**: Average absolute error  
    - **RMSE**: Penalizes large errors  
    """)

    st.divider()
    st.subheader("📊 Model Comparison Charts")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("R² Score")
        st.line_chart(results_df["R2"])

    with col2:
        st.write("MAE (Lower is Better)")
        st.line_chart(results_df["MAE"])

    with col3:
        st.write("RMSE (Lower is Better)")
        st.line_chart(results_df["RMSE"])

    st.divider()
    st.subheader("💰 Price Distribution in Dataset")

    prices = df["Price"]
    st.area_chart(prices.value_counts().sort_index())

    st.divider()
    st.subheader("🧠 Feature Importance (Random Forest)")

    try:
        if hasattr(model, "named_steps"):
            rf_model = model.named_steps["model"]
            prep = model.named_steps["prep"]
            feature_names = prep.get_feature_names_out()
            importances = rf_model.feature_importances_

            fi_df = pd.DataFrame({
                "Feature": feature_names,
                "Importance": importances
            }).sort_values(by="Importance", ascending=False).head(15)

            st.write("Top 15 Influential Features")
            st.bar_chart(fi_df.set_index("Feature"))
        else:
            st.info("Feature importance not available for this model.")
    except Exception as e:
        st.error(f"Feature importance extraction failed: {e}")
