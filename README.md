# Used Car Price Prediction Project

A data science project for predicting used car prices from listing features such as mileage, age, transmission, fuel type, accident history, and title condition.

## Project goal

This project estimates the price of a used car based on a set of vehicle attributes. It is designed as a portfolio-ready example of a machine learning workflow in a real-world domain.

## Why this project matters

Used car pricing is often inconsistent across online listings. Buyers and sellers need better valuation tools to understand fair market value, make informed decisions, and reduce uncertainty.

This project demonstrates:
- data cleaning and preprocessing
- exploratory data analysis
- feature engineering
- model comparison
- evaluation with appropriate metrics
- a simple prediction app

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
│   ├── train_model.py
│   └── predict.py
├── .gitignore
├── README.md
├── requirements.txt
├── used_cars.csv
└── DS_Car_Price_Project.ipynb
```

## Dataset

The project currently uses the provided used car dataset (`used_cars.csv`), which includes features such as:
- brand
- model
- model year
- mileage
- fuel type
- engine
- transmission
- exterior color
- interior color
- accident status
- clean title
- price

## Workflow

The full project workflow is designed to mirror a practical data science pipeline:

1. Data acquisition
   - Load the raw dataset from `used_cars.csv`
   - Confirm schema and inspect key columns

2. Data cleaning
   - Standardize column names
   - Convert price and mileage into numeric values
   - Remove invalid or empty values
   - Handle obvious formatting issues in raw listing text

3. Feature engineering
   - Create vehicle age from `model_year`
   - Keep a minimal set of highly informative variables
   - Prepare categorical variables for modeling

4. Train/test split
   - Separate the data into training and test sets
   - Use the test set for unbiased evaluation

5. Model building
   - Train a baseline regressor
   - Compare multiple regression models
   - Save the best-performing model artifact

6. Evaluation
   - Measure MAE, RMSE, and R²
   - Check if predictions are reliable and interpretable
   - Identify where the model may underperform

7. Deployment / usage
   - Build a simple Streamlit app to make predictions from user input
   - Make the project easy to demo in interviews or portfolio reviews

8. Documentation and presentation
   - Explain the business problem clearly
   - Summarize findings and model performance
   - Show how the project could be extended in future work

## Models to compare

The project is structured to compare a few common regression models, including:
- Linear Regression
- Ridge / Lasso regression
- Random Forest Regressor
- XGBoost or Gradient Boosting models

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

### 3. Train the model

```bash
python src/train_model.py
```

### 4. Run the app

```bash
streamlit run app/streamlit_app.py
```

## Example of the prediction flow

After training, the app allows a user to input key car attributes and receives a predicted market price.

## Project status

This repo is now organized as a more professional portfolio project structure, while keeping the original exploratory notebook and dataset for continuity.

## Future improvements

- Add more robust feature engineering
- Compare more advanced regression models
- Add SHAP or feature importance analysis
- Expand the app UI with more advanced filters
- Add deployment to a cloud hosting platform
- Include model validation and cross-validation results

## Notes

The original notebook (`DS_Car_Price_Project.ipynb`) remains in the repository for reference, while the project now has a cleaner structure for reproducibility and portfolio use.
