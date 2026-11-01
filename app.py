from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from src.car_price_pipeline import load_saved_feature_columns, load_saved_model


def build_prediction_payload() -> dict:
    return {
        "make": st.selectbox("Car brand", ["Toyota", "Honda", "Ford", "BMW", "Mercedes", "Audi", "Volkswagen", "Nissan"]),
        "model": st.selectbox("Car model", ["Corolla", "Camry", "Prius", "RAV4", "Civic", "Accord", "CR-V", "Fit", "Focus", "Fiesta", "Mustang", "Explorer", "3 Series", "5 Series", "X5", "X3", "C Class", "E Class", "GLC", "GLE", "A3", "A4", "Q5", "Q7", "Golf", "Passat", "Tiguan", "Polo", "Altima", "Sentra", "Juke", "Qashqai"]),
        "year": st.slider("Manufacturing year", min_value=2003, max_value=2025, value=2018),
        "mileage": st.slider("Mileage (km)", min_value=1000, max_value=220000, value=60000, step=1000),
        "fuel_type": st.selectbox("Fuel type", ["Petrol", "Diesel", "Hybrid", "Electric"]),
        "transmission": st.selectbox("Transmission", ["Manual", "Automatic"]),
        "engine_size": st.selectbox("Engine size (L)", [1.4, 1.6, 1.8, 2.0, 2.4, 2.8, 3.0]),
        "body_type": st.selectbox("Body type", ["Sedan", "SUV", "Hatchback", "Coupe"]),
        "seats": st.selectbox("Seats", [2, 4, 5, 7]),
    }


def main() -> None:
    st.set_page_config(page_title="Car Price Predictor", page_icon="🚗", layout="centered")
    st.title("Used Car Price Predictor")
    st.write("Estimate a used vehicle's market value using a trained machine learning model.")

    try:
        model = load_saved_model()
        feature_columns = load_saved_feature_columns()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.info("Run the following command first: python scripts/train_model.py")
        return

    payload = build_prediction_payload()
    input_df = pd.DataFrame([payload])
    prediction = model.predict(input_df[feature_columns])[0]

    st.subheader("Estimated Price")
    st.metric(label="Predicted Market Value", value=f"${prediction:,.2f}")
    st.caption("This estimate is based on the trained model and should be used as a market reference, not a guaranteed sale price.")


if __name__ == "__main__":
    main()
