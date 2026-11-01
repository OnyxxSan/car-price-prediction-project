# Car Price Prediction Project

A portfolio-ready machine learning project that predicts used car prices from vehicle attributes such as brand, model, year, mileage, fuel type, engine size, transmission, and body type.

This project was built to demonstrate an end-to-end data science workflow in a simple, understandable way. It is designed to be easy to run in VS Code, clear enough for interviews, and professional enough for a GitHub portfolio.

## Project Overview

The main goal of this project is to estimate the fair market value of a used car. Instead of relying on guesswork, the model learns patterns from historical vehicle data and uses those patterns to predict a likely price for a new input.

This project is useful for a portfolio because it shows practical machine learning skills beyond a classroom example:
- data generation and cleaning
- feature preparation
- model comparison
- metric evaluation
- model saving and reuse
- building a simple user-facing prediction app

## Why This Project Is a Good Portfolio Example

This project demonstrates a realistic data science workflow:
1. Create or load a dataset
2. Explore the data and understand key features
3. Split the data into training and testing sets
4. Train multiple regression models
5. Compare performance using evaluation metrics
6. Select the best model
7. Save the trained model
8. Use the model inside a Streamlit app
9. Present the work clearly in a notebook and README

This is a strong internship portfolio example because it shows that the student can move from raw data to a working solution and explain the process clearly.

## Dataset

The project uses a synthetic used-car dataset stored in `data/car_prices.csv`.

The dataset includes fields such as:
- `make`: brand of the car
- `model`: model name
- `year`: manufacturing year
- `mileage`: distance driven
- `fuel_type`: petrol, diesel, hybrid, electric
- `transmission`: manual or automatic
- `engine_size`: size of the engine in liters
- `body_type`: sedan, SUV, hatchback, coupe
- `seats`: number of seats
- `price`: target value to predict

This dataset is useful for portfolio work because it is simple, reproducible, and realistic enough to show the full predictive workflow without requiring external API access or complex setup.

## Workflow

The project follows a standard machine learning pipeline:

1. Generate the dataset
2. Inspect the data and verify its structure
3. Split the data into train and test sets
4. Prepare categorical and numeric features
5. Train several regression models
6. Compare model performance
7. Choose the best-performing model
8. Save the trained model
9. Launch a Streamlit application for prediction
10. Document the analysis in a notebook

## Machine Learning Models Used

This project compares three regression models:

- Linear Regression: easy to interpret and useful as a baseline
- Random Forest Regressor: captures nonlinear relationships well
- Gradient Boosting Regressor: strong performer on structured tabular data

The model comparison helps answer the practical question:
"Which model gives the most accurate price predictions?"

## Evaluation Metrics

The project evaluates models using:
- Mean Absolute Error (MAE): average absolute difference between predicted and actual prices
- Root Mean Squared Error (RMSE): penalizes larger errors more strongly
- R-squared (R²): measures how well the model explains the variation in the target

A strong model should have:
- lower MAE and RMSE
- higher R², ideally close to 1.0

## Tools and Libraries

- Python
- pandas
- NumPy
- scikit-learn
- matplotlib
- seaborn
- Streamlit
- joblib
- Jupyter Notebook

## Key Results

The project produced the following verified results during model comparison:

| Model | MAE | RMSE | R² |
| --- | ---: | ---: | ---: |
| Linear Regression | 2479.42 | 3138.45 | 0.9435 |
| Random Forest | 3360.20 | 4134.82 | 0.9020 |
| Gradient Boosting | 2823.97 | 3459.01 | 0.9314 |

The best-performing model on this dataset was Linear Regression, with an R² of 0.9435.

## Project Structure

```text
car-price-prediction-project/
├── README.md
├── requirements.txt
├── app.py
├── .gitignore
├── data/
│   └── car_prices.csv
├── notebooks/
│   └── car_price_analysis.ipynb
├── scripts/
│   ├── generate_data.py
│   ├── model_comparison.py
│   └── train_model.py
├── src/
│   └── car_price_pipeline.py
├── models/
│   └── car_price_model.joblib
├── .venv/
└── .vscode/
    └── extensions.json
```

## Setup Instructions

### 1. Open the project in VS Code

Open the folder containing this project in VS Code.

### 2. Create a virtual environment

```bash
cd "/Users/onyxxm/Downloads/Car-Price Project"
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install project dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run the Project

### Generate the dataset

```bash
python scripts/generate_data.py
```

### Compare models

```bash
python scripts/model_comparison.py
```

### Train the final model

```bash
python scripts/train_model.py
```

### Launch the Streamlit app

```bash
streamlit run app.py
```

Then open the local URL in the browser:

```text
http://localhost:8501
```

## App Behavior

The Streamlit app allows a user to input details such as:
- car brand
- car model
- manufacturing year
- mileage
- fuel type
- transmission
- engine size
- body type
- number of seats

The app then loads the trained model and displays an estimated price.

## Notebook Purpose

The notebook in `notebooks/car_price_analysis.ipynb` is used for:
- exploratory analysis
- visual exploration of data patterns
- checking how different features relate to price
- explaining the model-building process clearly

This matters for interviews because it demonstrates that you can do more than just train a model; you can also understand what the model is doing and explain your choices.

## Resume-Friendly Description

Machine learning project for predicting used car prices using vehicle attributes and regression models. Built an end-to-end workflow including data preparation, model comparison, evaluation, and a Streamlit application for interactive price predictions.

## Interview-Friendly Summary

I built a car price prediction project in Python using scikit-learn and Streamlit. The project compares several regression models, evaluates their performance using MAE, RMSE, and R², and includes a simple web app that predicts a used car's estimated price from user inputs.

## Common Setup Notes

- If the project cannot import the source files, make sure you are running the script from the repository root.
- If the dataset is missing, run `python scripts/generate_data.py`.
- If the trained model is missing, run `python scripts/train_model.py`.
- If Streamlit is not installed, run `python -m pip install -r requirements.txt`.

## Recommended Next Improvements

- add more realistic vehicle data and extra features
- compare additional models such as XGBoost and CatBoost
- add feature importance charts
- improve the app UI and add more input validation
- deploy the app to Streamlit Cloud
- create a short project presentation or slide deck

## License

This project is intended for educational, learning, and portfolio use.
