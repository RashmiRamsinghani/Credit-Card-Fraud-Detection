# Credit Card Fraud Detection

## Objective

To detect fraudulent credit card transactions using machine learning techniques while addressing the severe class imbalance present in fraud detection data.

## Dataset

The dataset contains credit card transactions with:
- Time
- Transaction Amount
- Anonymized features (V1–V28)
- Class, where 0 represents legitimate transactions and 1 represents fraud

The dataset is publicly available and is not included in this repository.

## Methodology

1. Exploratory Data Analysis
2. Analysis of class imbalance and transaction characteristics
3. Train-test split using stratification
4. SMOTE applied to the training data to address class imbalance
5. Mutual Information analysis for feature relevance
6. Logistic Regression as an interpretable baseline model
7. XGBoost as a nonlinear machine learning model
8. Model evaluation using Precision, Recall, F1-score and ROC-AUC
9. Threshold analysis to study the precision-recall trade-off

## Model Results

| Model | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|
| Logistic Regression | 14.73% | 90.54% | 25.33% | 98.72% |
| XGBoost | 85.43% | 87.16% | 86.29% | 98.66% |

XGBoost substantially improved precision and F1-score compared with Logistic Regression while maintaining high recall.

## Threshold Analysis

For XGBoost, different classification thresholds were evaluated to understand the trade-off between precision and recall.

A threshold of 0.7 produced:
- Precision: 87.6%
- Recall: 85.8%
- F1-score: 86.7%

Among the tested thresholds, 0.7 provided the highest F1-score.

## Tools & Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Imbalanced-learn (SMOTE)

## Project File

The complete analysis and implementation are available in the Jupyter Notebook:

`Fraud data analysis.ipynb`
