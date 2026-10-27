from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT / 'src'
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import streamlit as st

try:
    from src.predict import predict_price
except ImportError:
    from predict import predict_price

st.set_page_config(page_title='Used Car Price Predictor', page_icon='🚗', layout='centered')

st.title('Used Car Price Predictor')
st.caption('Estimate the market value of a used car based on common listing features.')

with st.form('car_price_form'):
    brand = st.text_input('Brand', 'Toyota')
    model = st.text_input('Model', 'Corolla')
    model_year = st.number_input('Model year', min_value=1990, max_value=2030, value=2020)
    milage = st.number_input('Mileage', min_value=0, max_value=500000, value=45000)
    fuel_type = st.selectbox('Fuel type', ['Gasoline', 'Hybrid', 'Diesel', 'Electric', 'E85 Flex Fuel'])
    transmission = st.selectbox('Transmission', ['Automatic', 'A/T', 'Manual', 'CVT'])
    accident = st.selectbox('Accident status', ['None reported', 'At least 1 accident or damage reported'])
    clean_title = st.selectbox('Clean title', ['Yes', 'No'])

    submitted = st.form_submit_button('Predict price')

if submitted:
    sample = {
        'brand': brand,
        'model': model,
        'model_year': model_year,
        'milage': milage,
        'fuel_type': fuel_type,
        'transmission': transmission,
        'accident': accident,
        'clean_title': clean_title,
    }

    try:
        prediction = predict_price(sample)
        st.success(f'Estimated car price: ${prediction:,.2f}')
    except FileNotFoundError:
        st.error('The model has not been trained yet. Please run: python src/train_model.py')
