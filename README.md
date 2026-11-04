# Used Car Price Prediction

A portfolio-ready machine learning project that predicts used car prices using vehicle attributes such as brand, model, year, mileage, fuel type, transmission, accident history, and title status.

## Overview

This project follows a complete data science workflow for a real-world pricing problem: estimating the fair market value of a used car. Buyers and sellers often struggle with inconsistent pricing across listings, so this project builds a practical pricing estimator based on historical data.

The app is built with Python and Streamlit, allowing users to enter car details and receive an estimated market price in seconds.

## Why this project matters

Used car prices are influenced by many factors, including age, mileage, condition, fuel efficiency, transmission type, and accident history. Those details vary across listings, making price estimation difficult without a consistent model.

This project addresses that problem by training a predictive model from historical listing data and exposing it through a simple interactive interface.

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

The data is cleaned before training so invalid values are removed, numeric columns are standardized, and a `car_age` feature is generated.

## Machine learning workflow

The project includes the full end-to-end pipeline:

1. Load the raw dataset
2. Clean and standardize fields
3. Engineer useful features like `car_age`
4. Split the data into training and test sets
5. Train and evaluate multiple regression models
6. Select the final model for production
7. Save the model and use it for predictions via Streamlit

## Model details

The production model is a `LinearRegression` pipeline that preprocesses numeric and categorical features using `ColumnTransformer`, `SimpleImputer`, and `OneHotEncoder`.

The project also includes:

- cross-validation on the full dataset
- baseline comparison against median-price predictions
- RMSE compatibility handling for newer scikit-learn versions
- validation tests for preprocessing and prediction input format

## Current project files

- `src/data_preprocessing.py` — dataset loading and cleaning logic
- `src/train_model.py` — training, validation, metric reporting, and model save logic
- `src/predict.py` — input feature building and price prediction helper
- `tests/test_pipeline.py` — pytest checks for dataset and feature behavior
- `requirements.txt` — required dependencies, including `pytest`

## Project structure

```text
car-price-prediction-project/
├── app/
│   └── streamlit_app.py
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── predict.py
│   └── train_model.py
├── tests/
│   └── test_pipeline.py
├── models/
├── notebooks/
├── DS_Car_Price_Project.ipynb
├── README.md
├── requirements.txt
├── used_cars.csv
└── .gitignore
```

## Setup

### 1. Install Python

Download and install Python 3.10+ from the official Python website.

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Train the model

```bash
python src/train_model.py
```

This script loads the dataset, cleans it, trains the selected pipeline, evaluates it, and saves the model to `models/used_car_model.joblib`.

## Run the app

```bash
streamlit run app/streamlit_app.py
```

The app opens a simple interface where a user can enter vehicle details and get an estimated price.

## Testing

Run the project checks with:

```bash
pytest
```

The tests cover:

- dataset loading
- cleaning of invalid values and age feature generation
- prediction input formatting

## Notes

This project is designed to be a clean, interview-friendly example of an end-to-end data science workflow: ingestion, preprocessing, modeling, evaluation, and deployment through a simple app.
