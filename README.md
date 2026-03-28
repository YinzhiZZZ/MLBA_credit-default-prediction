# MLBA_credit-default-prediction

## Project Overview
Binary classification model to predict credit card default risk
using the UCI Default of Credit Card Clients dataset (30,000 records, 23 features).

## Team
- Yinzhi Chen
- Ethan Wong
- Kevin Wang
- Ahmed Imtanan

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

## Expected Results

Running the notebooks with `random_state=42` on the original CSV will reproduce these exact numbers:

**Test Set:**

| Metric | Value |
|---|---|
| AUC-ROC | 0.7081 |
| Recall (Default) | 0.6202 |
| F1 Score (Default) | 0.4613 |
| Precision (Default) | 0.3672 |
| Accuracy | 0.6797 |

**5-Fold Cross-Validation (Training Set):**

| Metric | Mean | Std Dev |
|---|---|---|
| AUC-ROC | 0.7265 | ±0.0108 |
| Recall | 0.6459 | ±0.0159 |

If your numbers differ, check that (1) the CSV has not been modified, (2) `random_state=42` is set throughout, and (3) the same scikit-learn version is used.




