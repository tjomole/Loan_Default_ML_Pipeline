# Loan_Default_ML_Pipeline
A step-by-step annotated walkthrough of an end-to-end machine learning pipeline for predicting loan default. It leverages structured preprocessing, multiple ML models, and evaluation metrics to determine the most effective model for predicting defaults.

# Overview
This is a self‐contained Python model/script that: Loads a loan‐default dataset, preprocesses it (missing values, encoding, scaling), splits into train/test, trains five models: (Logistic Regression, Decision Tree, Random Forest, k-Nearest Neighbors, and Feed-forward Neural Network (via Keras)). Each model is then evaluated with accuracy, precision, recall, F1, ROC AUC, and confusion matrix.

# DataSet
The dataset used can be found here: https://www.kaggle.com/datasets/yasserh/loan-default-dataset/discussion/522084

# Detailed Walkthrough

1. Library Imports: Essential libraries are imported for data processing (pandas, numpy), visualization (seaborn, matplotlib), machine learning (scikit-learn), and neural networks (tensorflow.keras).
2. Access and Load Data:
Purpose - Read a zipped CSV dataset and save it in a usable format.
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    ...
This section unzips the file and loads it into a DataFrame, preparing it for cleaning and preprocessing.
3. Initial Exploration and Cleaning

Purpose: Understand data structure, remove irrelevant columns, and inspect for duplicates/missing values.

loan.drop(columns=[...], inplace=True)
loan = loan.drop_duplicates()

Reduces dimensionality and cleanses the dataset by removing redundant or irrelevant features.

4. Feature Engineering and Imputation

Purpose: Improve data quality and completeness.

loan['property_value1'] = loan['property_value'] - 8000
loan['loan_amount1'] = loan['loan_amount'] - 6500

Adjust financial figures.

Categorical variables filled with "Missing", numerical variables imputed using KNNImputer.

LTV is recalculated for accuracy.

5. Export Cleaned Data

Purpose: Save the cleaned dataset for modeling.

loan_clean.to_csv('/content/Loan_Default_Clean.csv')

Ensures reproducibility and separation of data cleaning from modeling.

6. Configuration and Load for Modeling

Purpose: Define constants and reload the clean dataset.

DATA_PATH = "/content/Loan_Default_Clean.csv"
df = pd.read_csv(DATA_PATH)

7. Data Splitting and Column Type Detection

Purpose: Separate features and target; identify column types.

X_train, X_test, y_train, y_test = train_test_split(...)

Train-test split with stratification ensures balanced class distribution.

8. Preprocessing Pipeline

Purpose: Standardize and encode data for ML models.

ColumnTransformer([
    ("nums", num_pipe, num_cols),
    ("cats", cat_pipe, cat_cols),
])

Pipelines handle missing values and scale/encode features in one modular step.

9. Model Training & Evaluation Helper

Purpose: Reusable function to fit, predict, and evaluate.

def train_and_evaluate(model, name):
    ...

Ensures consistent application of preprocessing and evaluation across all models.

10. Classical Machine Learning Models

Purpose: Train Logistic Regression, Decision Tree, Random Forest, and k-Nearest Neighbors.

train_and_evaluate(LogisticRegression(...), "Logistic Regression")

Each model is fit within a pipeline and evaluated using accuracy, precision, recall, F1 score, ROC AUC, and confusion matrix.

11. Feed-forward Neural Network (FFNN)

Purpose: Train a deep learning model for comparison.

Sequential([
    Dense(64, activation="relu", ...),
    Dropout(0.5),
    ...
])

Neural net adds non-linear representation power; useful for complex decision boundaries.

12. Evaluation Metrics Used

Metrics include:

Accuracy: Overall correctness.

Precision: % of positive predictions that were correct.

Recall: % of actual positives correctly identified.

F1 Score: Harmonic mean of precision and recall.

ROC AUC: Model's ability to distinguish between classes.

Confusion Matrix: Breakdown of TP, FP, FN, TN.

# Overall Performance Summary

| Model                         | Accuracy | Precision (1) | Recall (1) | F1-Score (1) | ROC AUC |
| ----------------------------- | :------: | :-----------: | :--------: | :----------: | :-----: |
| **Logistic Regression**       |  0.8100  |     0.7558    |   0.3387   |    0.4678    |  0.7735 |
| **Decision Tree**             |  0.9326  |     0.8632    |   0.8631   |    0.8632    |  0.9092 |
| **Random Forest**             |  0.9207  |     0.9125    |   0.7500   |    0.8233    |  0.9720 |
| **k-Nearest Neighbors (k=5)** |  0.8526  |     0.7977    |   0.5382   |    0.6428    |  0.8534 |
| **Neural Network**            |  0.8940  |     0.8659    |   0.6743   |    0.7582    |  0.9374 |

Note: “Precision (1)” etc. refer to the metrics on the positive class (default = 1).

# Ranking

1. Decision Tree: Best balance of accuracy, precision, recall, F1, and strong AUC.
2. Random Forest: Top ROC AUC and very high precision; a close second.
3. Feed-forward Neural Network: Strong discriminative power (AUC) and decent recall after minimal tuning.
4. k-Nearest Neighbors: Mid-level performance across metrics; simple but outpaced by others.
5. Logistic Regression: Baseline model with weakest recall and AUC; valuable for interpretability.

# Conclusion

This pipeline provides a modular, scalable approach to supervised ML for classification. The integration of traditional ML and deep learning, combined with robust preprocessing and consistent evaluation, ensures a comprehensive assessment of model performance for predicting loan defaults.
