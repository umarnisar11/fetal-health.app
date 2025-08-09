import streamlit as st
import numpy as np 
import joblib
from tensorflow.keras.models import load_model

# Load model and scaler
@st.cache_resource
def load_model_and_scaler():
    model = load_model("fetal_helath.h5")
    scaler = joblib.load("scaler.pkl")  # Make sure this file exists in the same folder
    return model, scaler

model, scaler = load_model_and_scaler()

# Feature names
features = [
    'baseline value', 'accelerations', 'fetal_movement', 'uterine_contractions',
    'light_decelerations', 'severe_decelerations', 'prolongued_decelerations',
    'abnormal_short_term_variability', 'mean_value_of_short_term_variability',
    'percentage_of_time_with_abnormal_long_term_variability', 'mean_value_of_long_term_variability',
    'histogram_width', 'histogram_min', 'histogram_max', 'histogram_number_of_peaks',
    'histogram_number_of_zeroes', 'histogram_mode', 'histogram_mean', 'histogram_median',
    'histogram_variance', 'histogram_tendency'
]

class_labels = {
    0: "🟢 Normal",
    1: "🟡 Suspicious",
    2: "🔴 Pathological"
}

# UI Title
st.set_page_config(page_title="Fetal Health Predictor", layout="centered")
st.title("🤰 Fetal Health Classification")
st.markdown("Enter the required CTG parameters below to predict fetal health status:")

# Form for input
with st.form("prediction_form"):
    cols = st.columns(3)
    user_inputs = []
    for i, feature in enumerate(features):
        col = cols[i % 3]
        with col:
            val = st.number_input(
                label=feature.replace("_", " ").capitalize(),
                min_value=0.00,
                max_value=1000.0 if "percentage" not in feature else 100.0,
                value=0.000,  # ← changed from 0.00
                step=0.01,
                format="%.3f"  # ← changed from %.2f to show 3 decimal digits
            )
            user_inputs.append(val)
    submit = st.form_submit_button("🔍 Predict")

# Prediction
if submit:
    try:
        X = np.array(user_inputs).reshape(1, -1)
        X_scaled = scaler.transform(X)
        prediction = model.predict(X_scaled)
        class_idx = np.argmax(prediction, axis=1)[0]
        result = class_labels.get(class_idx, "Unknown")

        st.subheader("Prediction Result")
        if class_idx == 0:
            st.success(f"**Fetal Health Status: {result}**")
        elif class_idx == 1:
            st.warning(f"**Fetal Health Status: {result}**")
        else:
            st.error(f"**Fetal Health Status: {result}**")

        st.markdown("""
        **Legend:**
        - 🟢 **Normal**: Healthy fetal state  
        - 🟡 **Suspicious**: Monitor with care  
        - 🔴 **Pathological**: Medical attention needed
        """)

    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")

# Footer
st.markdown("---")
st.caption("Developed for educational use. Not a substitute for clinical judgment.")
