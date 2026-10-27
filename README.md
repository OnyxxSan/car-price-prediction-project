# Used Car Price Prediction Project

A data science project for predicting used car prices from listing features such as mileage, age, transmission, fuel type, accident history, and title condition.

## Project goal

This project estimates the price of a used car based on a set of vehicle attributes. It is designed as a portfolio-ready data science workflow in a real-world domain.

## Why this project matters

Used car pricing is often inconsistent across listings, making it difficult for buyers and sellers to judge fair market value. This project demonstrates how a machine learning model can support pricing decisions, improve transparency, and reduce uncertainty.

## Project workflow

The project follows a standard end-to-end machine learning workflow:

1. Data acquisition
   - Load the raw dataset from `used_cars.csv`
   - Validate the schema and key feature columns

2. Data cleaning
   - Standardize column names
   - Convert mileage and price into numeric values
   - Remove invalid or empty records
   - Fix obvious formatting inconsistencies in raw listing data

3. Feature engineering
   - Create a vehicle age feature from `model_year`
   - Preserve the most informative attributes for pricing
   - Prepare categorical metadata for encoding

4. Train/test split
   - Split data into training and testing sets with a fixed random seed
   - Use the test set to measure real-world performance

5. Model comparison
   - Train multiple regression models
   - Compare performance using MAE, RMSE, and R²
   - Identify the strongest model for deployment

6. Model selection and saving
   - Save the best-performing model artifact to the `models/` directory
   - Make predictions reproducible for future use

7. User-facing prediction app
   - Use Streamlit to allow users to enter vehicle attributes
   - Return an estimated market price in a simple interface

8. Documentation and portfolio presentation
   - Summarize model assumptions, limitations, and next improvements
   - Present the project as a clear data science portfolio item

## Repository structure

```text
car-price-prediction-project/
├── app/
│   └── streamlit_app.py
├── data/
│   ├── raw/
│   │   └── .gitkeep
│   └── processed/
│       └── .gitkeep
├── models/
│   └── .gitkeep
├── notebooks/
│   └── .gitkeep
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── model_comparison.py
│   ├── train_model.py
│   └── predict.py
├── .gitignore
├── README.md
├── requirements.txt
├── used_cars.csv
├── DS_Car_Price_Project.ipynb
└── app/streamlit_app.py
```

## Dataset

The project uses the provided used car dataset (`used_cars.csv`), which includes fields such as:
- brand
- model
- model year
- mileage
- fuel type
- transmission
- accident status
- clean title
- price

## Model comparison

The project includes a benchmarking step to compare multiple regression models:

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | TBD | TBD | TBD |
| Ridge Regression | TBD | TBD | TBD |
| Random Forest Regressor | TBD | TBD | TBD |
| Gradient Boosting Regressor | TBD | TBD | TBD |

Run the benchmark script to populate the exact values:

```bash
python src/model_comparison.py
```

## Quick start

### 1. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Benchmark models

```bash
python src/model_comparison.py
```

### 4. Train the best model

```bash
python src/train_model.py
```

### 5. Run the app

```bash
streamlit run app/streamlit_app.py
```

## App usage

The Streamlit app allows a user to enter key vehicle details and receive an estimated price. This makes the project easy to demo in interviews, class presentations, or portfolio reviews.

## Project status

This repo is now organized as a portfolio-style data science project while preserving the original exploratory notebook and raw dataset for continuity.

## Future improvements

- Add richer feature engineering, such as depreciation rate and vehicle age effects
- Compare additional models, including XGBoost if needed
- Add SHAP or similar explainability tools for feature importance
- Expand the app with price ranges and user-friendly visual output
- Deploy the app on a public cloud platform
- Add a stronger business narrative for customer or seller use cases

## Notes

The original notebook (`DS_Car_Price_Project.ipynb`) remains in the repository as the exploratory reference, while the structured codebase makes the project easier to showcase and maintain.
