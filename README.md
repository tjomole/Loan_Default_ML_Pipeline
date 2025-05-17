# Loan_Default_ML_Pipeline
A step-by-step annotated walkthrough of an end-to-end machine learning pipeline for predicting loan default. It leverages structured preprocessing, multiple ML models, and evaluation metrics to determine the most effective model for predicting defaults.

# Overview
This is a self‐contained Python model/script that: Loads a loan‐default dataset, preprocesses it (missing values, encoding, scaling), splits into train/test, trains five models: (Logistic Regression, Decision Tree, Random Forest, k-Nearest Neighbors, and Feed-forward Neural Network (via Keras)). Each model is then evaluated with accuracy, precision, recall, F1, ROC AUC, and confusion matrix.

# DataSet
The dataset used can be found here: https://www.kaggle.com/code/hoale2908/mortgage-loan-default-data-preprocessing

# Detailed Wallkthrough

1. Library Imports: Essential libraries are imported for data processing (pandas, numpy), visualization (seaborn, matplotlib), machine learning (scikit-learn), and neural networks (tensorflow.keras).
2. Access and Load Data:
Purpose - Read a zipped CSV dataset and save it in a usable format.
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    ...
This section unzips the file and loads it into a DataFrame, preparing it for cleaning and preprocessing.
