# -Click-Through-Rate-Prediction

# Click-Through Rate (CTR) Prediction

This project implements and compares several machine learning models to predict whether a user will click on an advertisement (CTR prediction) using a provided dataset. The models include XGBoost (with hyperparameter tuning and bagging), Random Forest, Decision Tree, and Logistic Regression.

---

## Features

- **Data Preprocessing:** Handles categorical variables using label encoding and standardizes features where appropriate.
- **Model Training:** Trains and evaluates multiple models:
  - XGBoost (with GridSearchCV for hyperparameter tuning)
  - XGBoost with Bagging
  - Random Forest (with cross-validation)
  - Decision Tree
  - Logistic Regression
- **Evaluation Metrics:** Reports accuracy and classification metrics for each model.
- **Comparison:** Enables easy comparison of model performance for CTR prediction.

---

## Requirements

- Python 3.7+
- pandas
- numpy
- scikit-learn
- xgboost

Install dependencies with:

```sh
pip install pandas numpy scikit-learn xgboost
```

## Model Details

- **XGBoost:** Uses grid search for hyperparameter optimization and 5-fold cross-validation.
- **Bagging with XGBoost:** Uses BaggingClassifier to ensemble XGBoost models.
- **Random Forest:** Evaluated with cross-validation and test set.
- **Decision Tree & Logistic Regression:** Standard training and evaluation.

---

## Results

After running the notebook, you will see:
- Accuracy for each model (on test set and via cross-validation where applicable)
- Classification reports (precision, recall, f1-score)
- Best hyperparameters for XGBoost (via GridSearchCV)

---
