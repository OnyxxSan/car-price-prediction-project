# Used Car Price Prediction

A portfolio-ready machine learning project that predicts used car prices using attributes such as brand, model, year, mileage, fuel type, transmission, accident history, and title status.

## Overview

This project follows a complete data science workflow for a real-world pricing problem: estimating the fair market value of a used car. The dataset includes listings with vehicle features and a target price, and the model learns from those patterns to estimate price for new inputs.

The app is built with Python and Streamlit so users can enter car details and receive an estimated price in seconds.

## Problem and why it matters

Used car prices vary substantially based on age, mileage, condition, fuel type, transmission, and accident history. Buyers and sellers often need a consistent way to estimate fair value. This project provides a simple, explainable prediction model that helps with that task.

## Dataset

The project uses `used_cars.csv`, which contains used-car listing data and the target variable `price`.

### Features used in the model

- brand
- model
- model_year
- milage
- fuel_type
- transmission
- accident
- clean_title
- car_age

### Target variable

- price: used-car selling price in USD

## Workflow

1. Load the raw dataset
2. Clean and standardize fields
3. Generate derived features like `car_age`
4. Split the data into train/test sets
5. Compare multiple regression models
6. Select the final model and save it
7. Use the model in a Streamlit app for interactive prediction

## Final model

The project demonstrates several regression models, and the production model used for prediction is a `LinearRegression` pipeline with preprocessing for numeric and categorical columns.

## Metrics

The project reports MAE, RMSE, and R² on a held-out test set. It also includes a baseline comparison using the median price as a simple benchmark.

## Project structure

```text
car-price-prediction-project/
├── app/
│   └── streamlit_app.py
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── model_comparison.py
│   ├── predict.py
│   └── train_model.py
├── models/
├── notebooks/
├── DS_Car_Price_Project.ipynb
├── README.md
├── requirements.txt
├── used_cars.csv
└── .gitignore
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train the model

```bash
python src/train_model.py
```

This script loads the dataset, cleans it, trains the final pipeline, evaluates it, and saves the model to `models/used_car_model.joblib`.

## Run the app

```bash
streamlit run app/streamlit_app.py
```

## Notes

This project is designed as a clean, interview-friendly example of an end-to-end data science workflow: ingestion, preprocessing, modeling, evaluation, and deployment through a simple app.
