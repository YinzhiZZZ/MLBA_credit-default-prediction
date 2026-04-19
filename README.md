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
GitHub (code) ──────────────────────────────► EC2 (Streamlit app)
                                                    ▲        │
Amazon S3 ──── models/xgboost_model.pkl ────────────┘        │ save predictions
               data/test_batch.csv ─── read batch data ──────┘
                                                               ▼
Risk Officer ──── browser (52.23.161.87:8501) ──► Amazon DynamoDB
                                                  credit-predictions table
IAM Role (LabInstanceProfile) ──────────────────► auto auth for EC2
```

**Services used:**
- **Amazon S3**: Stores trained model artifact and batch input data
- **Amazon EC2** (t3.micro, Ubuntu): Hosts the Streamlit web application
- **Amazon DynamoDB**: Persists all prediction results for audit and review
- **IAM Role (LabInstanceProfile)**: Grants EC2 automatic credential-free access to S3 and DynamoDB
- **GitHub**: Version control for all code and notebooks

**Infrastructure decisions:**
- Lambda rejected: XGBoost runtime dependencies exceed the 262 MB unzipped package limit
- SageMaker Endpoint rejected: explicit IAM deny on CreateEndpointConfig in AWS Academy environment
- EC2 selected: no package size restrictions, operates within Academy IAM permissions

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
│   └── 04_predict_demo.ipynb       # SageMaker demo UI (backup)
├── app.py                          # Streamlit web application (primary UI)
├── requirements.txt
├── .gitignore
└── README.md
```

## Environment Setup

### Local (notebooks 01–03)

```bash
git clone https://github.com/YinzhiZZZ/MLBA_credit-default-prediction
cd MLBA_credit-default-prediction
pip install -r requirements.txt
```

Place `UCI_Credit_Card.csv` in the `data/` folder, then run notebooks in order.

### Streamlit Web App (primary UI)

Accessible at: **http://52.23.161.87:8501**

No login required. To redeploy after session restart:

```bash
source venv/bin/activate
cd MLBA_credit-default-prediction
git pull
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

### SageMaker Notebook (backup UI)

1. AWS Academy → Start Lab → AWS Console
2. SageMaker → Notebook instances → credit-default-notebook → Open Jupyter
3. Open `notebooks/04_predict_demo.ipynb` → Kernel → Restart and Run All

## Demo UI

**Primary: Streamlit Web App (EC2)**
- Tab 1: Single customer prediction via interactive form
- Tab 2: Batch prediction reading test_batch.csv directly from S3

**Backup: SageMaker Notebook (ipywidgets)**
- Interactive widgets inside Jupyter, accessible via AWS Console

All predictions saved to DynamoDB `credit-predictions` table.

## How to Reproduce Results

Run notebooks 01–03 with `random_state=42`. Expected results:

| Metric | Logistic Regression | Random Forest | XGBoost |
|--------|--------------------|--------------| --------|
| AUC-ROC | 0.7081 | 0.7522 | 0.7723 |
| Recall | 0.6202 | 0.3429 | 0.6081 |
| F1 | 0.4613 | 0.4483 | 0.5300 |

## References

- Yeh, I.-C., & Lien, C.-h. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. *Expert Systems with Applications*.
- Chang, V., et al. (2024). Credit risk prediction using machine learning and deep learning. *Risks, 12*(11), 174.
- UCI Machine Learning Repository. https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
