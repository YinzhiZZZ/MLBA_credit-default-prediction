# Credit Card Default Prediction

A proof-of-concept ML system that predicts whether a credit card customer will default on their next payment. 

## Team

- Yinzhi Chen
- Kevin Wang
- Ethan Wong


## Business Problem

Credit default prediction is a core problem in consumer finance. Banks must identify high-risk borrowers early to reduce losses and improve portfolio performance. This system allows a risk officer to input a customer's profile and repayment history and receive an instant default probability score.

## Dataset

**UCI Default of Credit Card Clients**
- 30,000 records, 23 features
- Target variable: default payment next month (1 = default, 0 = no default)
- Class imbalance: ~22% defaulters, ~78% non-defaulters
- No missing values
- Source: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients

Features cover four dimensions:
- Customer demographics (age, sex, education, marital status)
- Credit profile (credit limit)
- Six months of billing history (BILL_AMT1–6)
- Six months of repayment behavior (PAY_0–6, PAY_AMT1–6)

## Models

Three models were trained and evaluated on the same 80/20 stratified train/test split:

| Model | AUC-ROC | Recall | F1 | Precision | Accuracy |
|-------|---------|--------|----|-----------|----------|
| Logistic Regression | 0.7081 | 0.6202 | 0.4613 | 0.3672 | 0.6797 |
| Random Forest | 0.7522 | 0.3429 | 0.4483 | 0.6472 | 0.8133 |
| **XGBoost (final)** | **0.7723** | **0.6081** | **0.5300** | **0.4697** | **0.7615** |

**XGBoost was selected as the final model** based on highest AUC-ROC and F1 score. Random Forest was rejected despite high accuracy because its Recall of 0.34 means it misses 66% of actual defaulters — unacceptable for a credit risk use case where false negatives are costly.

Primary evaluation metrics are AUC-ROC and Recall, as the dataset is imbalanced and missing a defaulter is substantially more costly than a false alarm.

## AWS Architecture

```
GitHub (code) ──────────────────────────────────────► SageMaker Notebook
                                                            ▲        │
Amazon S3 ──── models/xgboost_model.pkl ────────────────────┘        │ save predictions
               data/test_batch.csv ─── read batch data ──────────────┘
                                                                       ▼
Risk Officer ──── manual input (ipywidgets UI) ──────► Amazon DynamoDB
                                                       credit-predictions table
```

**Services used:**
- **Amazon S3**: Stores trained model artifact (`xgboost_model.pkl`) and batch input data (`test_batch.csv`)
- **Amazon SageMaker Notebook**: Cloud runtime environment that loads the model from S3 and serves an interactive prediction UI
- **Amazon DynamoDB**: Persists all prediction results (single and batch) for audit and review
- **GitHub**: Version control for all code and notebooks

## Repository Structure

```
MLBA_credit-default-prediction/
├── data/
│   └── (place UCI_Credit_Card.csv here)
├── models/
│   ├── logistic_regression.pkl
│   ├── scaler.pkl
│   └── xgboost_model.pkl
├── notebooks/
│   ├── 01_EDA.ipynb                # Exploratory data analysis
│   ├── 02_logistic.ipynb           # Logistic Regression baseline
│   ├── 03_nonlinear.ipynb          # Random Forest and XGBoost comparison
│   └── 04_predict_demo.ipynb       # Demo UI (SageMaker deployment)
├── .gitignore
└── README.md
```

## Environment Setup

### Local (for running notebooks 01–03)

```bash
git clone https://github.com/YinzhiZZZ/MLBA_credit-default-prediction
cd MLBA_credit-default-prediction
pip install -r requirements.txt
```

Place `UCI_Credit_Card.csv` in the `data/` folder, then run notebooks in order.

### AWS SageMaker (for running the demo)

1. Log in to AWS Academy Learner Lab → Start Lab → open AWS Console
2. Go to **SageMaker → Notebook instances → credit-default-notebook → Open Jupyter**
3. Create a new notebook with `conda_python3` kernel
4. Run the following cells in order:

**Cell 1 — Setup:**
```python
import os
os.chdir('/home/ec2-user/SageMaker')
os.system('git clone https://github.com/YinzhiZZZ/MLBA_credit-default-prediction')
os.system('aws s3 cp s3://credit-default-model-mlba/models/xgboost_model.pkl MLBA_credit-default-prediction/models/xgboost_model.pkl')
print("Setup complete!")
```

**Cell 2 — Install xgboost:**
```python
import subprocess
subprocess.run(['conda', 'install', '-y', '-c', 'conda-forge', 'xgboost'], capture_output=True, text=True)
print("xgboost installed!")
```

**Cell 3 — Run demo UI:**
Open `notebooks/04_predict_demo.ipynb` and run the prediction cell.

## How to Reproduce Results

Run notebooks 01–03 in order with `random_state=42` throughout. Expected test set results:

| Metric | Logistic Regression | Random Forest | XGBoost |
|--------|--------------------|--------------| --------|
| AUC-ROC | 0.7081 | 0.7522 | 0.7723 |
| Recall | 0.6202 | 0.3429 | 0.6081 |
| F1 | 0.4613 | 0.4483 | 0.5300 |

## Demo UI

The demo notebook (`04_predict_demo.ipynb`) provides two modes:

**Tab 1 — Single Customer Prediction:**
Risk officer manually inputs customer profile (credit limit, age, payment status, bill amounts) and receives an instant default probability and risk classification (HIGH / LOW RISK).

**Tab 2 — Batch Prediction:**
Reads `test_batch.csv` directly from S3, runs predictions on all records, and saves results to DynamoDB. Displays a summary table sorted by default probability.

All predictions are automatically saved to the `credit-predictions` DynamoDB table with timestamp, source, and key customer attributes.

## Dependencies

See `requirements.txt` for full list. Key packages:

```
pandas
numpy
scikit-learn
xgboost
joblib
matplotlib
seaborn
ipywidgets
boto3
```

## References

- Yeh, I.-C., & Lien, C.-h. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. *Expert Systems with Applications*.
- Chang, V., et al. (2024). Credit risk prediction using machine learning and deep learning. *Risks, 12*(11), 174.
- UCI Machine Learning Repository: Default of Credit Card Clients. https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
