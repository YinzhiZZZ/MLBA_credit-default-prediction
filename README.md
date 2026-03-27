# MLBA_credit-default-prediction

## Project Overview
Binary classification model to predict credit card default risk
using the UCI Default of Credit Card Clients dataset (30,000 records, 23 features).

## Setup
1. Clone this repo
2. Install dependencies:  pip install -r requirements.txt
3. Place UCI_Credit_Card.csv into the /data folder

## How to Run
Run notebooks in order:

| Notebook | Description | Status |
|----------|-------------|--------|
| notebooks/01_EDA.ipynb | Exploratory data analysis | ✅ Done |
| notebooks/02_logistic.ipynb | Logistic Regression baseline | ✅ Done |
| notebooks/03_nonlinear.ipynb | Random Forest & XGBoost | 🔄 In progress |

## Models
- Logistic Regression (baseline)
- Random Forest
- XGBoost

## Evaluation Metrics
ROC-AUC, Precision, Recall, F1, Accuracy

## Team
- Yinzhi Chen
- Ethan Wong
- Kevin Wang
- Ahmed Imtanan