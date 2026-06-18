# Credit Card Fraud Detection System

A machine learning project that builds and evaluates fraud detection 
models on a credit card transaction dataset, with a focus on handling 
class imbalance and choosing the right evaluation metrics for 
high-stakes classification problems.

## The Problem

Standard accuracy is a misleading metric for fraud detection. A model 
that predicts "not fraud" for every transaction would score 98.5% 
accuracy on this dataset — and catch zero actual fraud cases. This 
project addresses that directly by prioritizing recall and F1-score 
over accuracy, and by handling severe class imbalance explicitly.

## Dataset

10,000 credit card transactions with the following features:
- `transaction_hour` — hour of day the transaction occurred
- `amount` — transaction amount
- `merchant_category` — type of merchant
- `foreign_transaction` — whether the transaction was international
- `location_mismatch` — whether billing and transaction locations differ
- `device_trust_score` — trust score of the device used
- `velocity_last_24h` — number of transactions in the last 24 hours
- `cardholder_age` — age of the cardholder
- `is_fraud` — target variable (1 = fraud, 0 = not fraud)

Class distribution: 98.49% not fraud, 1.51% fraud — severely imbalanced.

## Project Structure

```
fraud-detection-system/
    data/
        fraud_data.csv
    output/
        confusion_matrices.png
        feature_importance.png
    src/
        eda_fraud.ipynb       # EDA and class imbalance analysis
        model_training.ipynb  # Model training and evaluation
    README.md
```

## Approach

### Exploratory Data Analysis
Before building any model, I explored the data to understand 
what separates fraud from legitimate transactions:

| Feature | Not Fraud | Fraud |
|---------|-----------|-------|
| device_trust_score | 62.2 | 37.9 |
| foreign_transaction | 9.1% | 54.3% |
| location_mismatch | 8.0% | 47.7% |
| transaction_hour | 11.7 | 3.8 |

Key finding: fraud transactions cluster in the early morning hours, 
involve foreign transactions at 6x the normal rate, and come from 
devices with significantly lower trust scores.

### Handling Class Imbalance
Used `class_weight='balanced'` in both models — this tells the 
algorithm to pay proportionally more attention to the rare fraud 
class rather than optimizing purely for the majority class.

### Models Trained
- Logistic Regression with balanced class weights
- Random Forest with balanced class weights

## Results

| Model | Precision (Fraud) | Recall (Fraud) | F1 (Fraud) |
|-------|-------------------|----------------|------------|
| Logistic Regression | 0.26 | **1.00** | 0.42 |
| Random Forest | **1.00** | 0.67 | **0.80** |

**Logistic Regression** catches every single fraud case (perfect recall) 
but raises more false alarms — useful when missing fraud is the 
costliest mistake.

**Random Forest** is highly precise — when it flags fraud, it's always 
right — but misses about 1 in 3 fraud cases. Better F1-score overall.

The right model depends on the business context:
- High-value transactions where missing fraud is catastrophic → 
  Logistic Regression
- Consumer-facing applications where false alarms hurt user experience 
  → Random Forest

## Feature Importance

Random Forest identified these as the most predictive features:

1. `device_trust_score` — strongest predictor
2. `transaction_hour` — fraud concentrates in early morning hours
3. `velocity_last_24h` — rapid successive transactions signal fraud
4. `foreign_transaction` — international transactions 6x more likely fraudulent
5. `location_mismatch` — billing/location mismatch strongly correlates with fraud
6. `amount` — moderate predictor
7. `cardholder_age` — weakest predictor; fraud does not discriminate by age

These findings are consistent with the EDA analysis — the model 
independently confirmed the patterns identified during exploration.

## Key Learnings

- Accuracy is a misleading metric for imbalanced datasets
- The precision-recall tradeoff is a business decision, not just 
  a technical one — what you optimize for depends on the cost of 
  each type of error
- EDA findings and model feature importance should tell a 
  consistent story — when they do, it validates the analysis

## Tech Stack

- Python
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook