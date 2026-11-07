import streamlit as st
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "model.pkl"

# ── Load model (cached so it only loads once) ──────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Used Car Price Predictor", page_icon="🚗")

st.title("🚗 Used Car Price Predictor")
st.write("Enter the car details below to get an estimated selling price.")

# ── Input widgets ──────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    model_year = st.number_input(
        "Model Year",
        min_value=1990,
        max_value=2025,
        value=2015,
        step=1,
        help="The year the car was manufactured.",
    )

with col2:
    milage = st.number_input(
        "Mileage (miles)",
        min_value=0,
        max_value=1_000_000,
        value=50_000,
        step=1_000,
        help="Total miles driven.",
    )

# ── Predict ────────────────────────────────────────────────────────────────────
if st.button("Predict Price", type="primary"):
    # Model was fitted on a raw numpy array — pass values in the same order:
    # [milage, model_year]  (as used during training)
    input_array = np.array([[milage, model_year]])
    prediction  = model.predict(input_array)[0]

    if prediction < 0:
        st.warning("The model returned a negative estimate — the input values may be outside the training range.")
    else:
        st.success(f"### Estimated Price: **${prediction:,.2f}**")
        st.caption("Note: This is an ML estimate based on mileage and model year only.")
