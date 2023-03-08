# Import libraries
import joblib
import json
import numpy as np
import os
import pandas as pd
from tabular_data import load_airbnb
from sklearn import ensemble, linear_model, metrics, model_selection, preprocessing, tree
import warnings

# Define the file path and load the data
df = pd.read_csv("airbnb-property-listings/tabular_data/clean_tabular_data.csv", index_col = "ID")
X, y = load_airbnb(df, "Category")

# Split the data into training, validation, and test sets
np.random.seed(1)
X = X.select_dtypes(include = ["int64", "float64"])
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.3)
X_validation, X_test, y_validation, y_test = model_selection.train_test_split(X_test, y_test, test_size = 0.5)

# Print the shape of training data
print(X_train.shape, y_train.shape, end = "\n\n")

# Fit and transform the data
scaler = preprocessing.StandardScaler()
X_train = scaler.fit_transform(X_train)
X_validation = scaler.transform(X_validation)
X_test = scaler.transform(X_test)

# Create a logistic regression model to predict the category from the tabular data
lr = linear_model.LogisticRegression()
lr.fit(X_train, y_train)

# Make predictions on the training and testing data
y_train_pred = lr.predict(X_train)
y_test_pred = lr.predict(X_test)

# Define a function to evaluate the predictions on the training and testing data
def evaluate_model_performance(y_train, y_train_pred, y_test, y_test_pred):

    """Create a dictionary to store the results"""
    results = {}

    """Evaluate the model performance on the training set"""
    train_accuracy = round(metrics.accuracy_score(y_train, y_train_pred), 3)
    train_precision = round(metrics.precision_score(y_train, y_train_pred, average = "macro"), 3)
    train_recall = round(metrics.recall_score(y_train, y_train_pred, average = "macro"), 3)
    train_f1_score = round(metrics.f1_score(y_train, y_train_pred, average = "macro"), 3)
    
    """Add the training results to the dictionary"""
    results["Train"] = {
        "Accuracy": train_accuracy,
        "Precision": train_precision,
        "Recall": train_recall,
        "F1 Score": train_f1_score
    }
    
    """Evaluate the model performance on the testing set"""
    test_accuracy = round(metrics.accuracy_score(y_test, y_test_pred), 3)
    test_precision = round(metrics.precision_score(y_test, y_test_pred, average = "macro"), 3)
    test_recall = round(metrics.recall_score(y_test, y_test_pred, average = "macro"), 3)
    test_f1_score = round(metrics.f1_score(y_test, y_test_pred, average = "macro"), 3)
    
    """Add the testing results to the dictionary"""
    results["Test"] = {
        "Accuracy": test_accuracy,
        "Precision": test_precision,
        "Recall": test_recall,
        "F1 Score": test_f1_score
    }
    
    return results