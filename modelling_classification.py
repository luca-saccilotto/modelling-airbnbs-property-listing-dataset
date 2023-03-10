# Import libraries
import json
import numpy as np
import os
import pandas as pd
from sklearn import ensemble, linear_model, metrics, model_selection, preprocessing, tree
import warnings

# Import methods defined previously
from modelling_regression import save_model
from tabular_data import load_airbnb

# Define the file path and load the data
df = pd.read_csv("airbnb-property-listings/tabular_data/clean_tabular_data.csv", index_col = "ID")
X, y = load_airbnb(df, "Category")

# Split the data into training, validation, and test sets
np.random.seed(1)
X = X.select_dtypes(include = ["int64", "float64"])
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.3)
X_validation, X_test, y_validation, y_test = model_selection.train_test_split(X_test, y_test, test_size = 0.5)

# Print the shape of training data
print(X_train.shape, y_train.shape)

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

# Create a dictionary containing the training, validation, and testing data
datasets = {
    "X_train": X_train,
    "y_train": y_train, 
    "X_test": X_test, 
    "y_test": y_test, 
    "X_validation": X_validation, 
    "y_validation": y_validation
}

# Use "GridSearchCV" to tune the hyperparameters of the model
def tune_classification_model_hyperparameters(model_class, datasets, hyperparameters):

    """Tuning hyperparameters"""
    grid = model_selection.GridSearchCV(model_class, hyperparameters) # verbose = 10
    grid.fit(datasets["X_train"], datasets["y_train"])

    """Model prediction on validation set using best hyperparameters"""
    y_val_pred = grid.predict(datasets["X_validation"])

    """Calculate metrics on validation set"""
    validation_accuracy = round(metrics.accuracy_score(datasets["y_validation"], y_val_pred), 3)
    validation_precision = round(metrics.precision_score(datasets["y_validation"], y_val_pred, average="macro"), 3)
    validation_recall = round(metrics.recall_score(datasets["y_validation"], y_val_pred, average="macro"), 3)
    validation_f1_score = round(metrics.f1_score(datasets["y_validation"], y_val_pred, average="macro"), 3)

    """Store best model, best hyperparameters, and best metrics in a dictionary"""
    best_model = grid.best_estimator_
    best_params = grid.best_params_
    best_metrics = {
        "Accuracy": validation_accuracy,
        "Precision": validation_precision,
        "Recall": validation_recall,
        "F1 Score": validation_f1_score
        }
    
    return best_model, best_params, best_metrics

# Create a dictionary containing the ranges of the hyperparameters to be tuned
logistic_regression_hyperparams = {
    "penalty": ["none", "l1", "l2"],
    "C": [0.1, 1, 10],
    "solver": ["lbfgs", "liblinear", "saga"],
}
decision_tree_hyperparams = {
    "max_depth": [3, 5, 10],
    "min_samples_split": [2, 5, 10],
    "criterion": ["gini", "entropy"],
}
random_forest_hyperparams = {
    "n_estimators": [50, 100, 200],
    "max_depth": [5, 10, 15],
    "min_samples_split": [2, 5, 10],
}
gradient_boosting_hyperparams = {
    "learning_rate": [0.001, 0.01, 0.1],
    "n_estimators": [50, 100, 150],
    "max_depth": [3, 5, 7],
}

# Define a dictionary of models with corresponding hyperparameters and folder names
models = {
    "logistic_regression": {
        "model_class": linear_model.LogisticRegression(),
        "hyperparameters": logistic_regression_hyperparams
    },
    "decision_tree": {
        "model_class": tree.DecisionTreeClassifier(),
        "hyperparameters": decision_tree_hyperparams
    },
    "random_forest": {
        "model_class": ensemble.RandomForestClassifier(),
        "hyperparameters": random_forest_hyperparams
    },
    "gradient_boosting": {
        "model_class": ensemble.GradientBoostingClassifier(),
        "hyperparameters": gradient_boosting_hyperparams
    }
}

# Define a function that tune, evaluate, and save all the models
def evaluate_all_models(models):

    for model_name, model_info in models.items():

        model_class = model_info["model_class"]
        hyperparameters = model_info["hyperparameters"]

        best_model, best_params, best_metrics = tune_classification_model_hyperparameters(model_class, datasets, hyperparameters)
        print(f"Model: {best_model}, Accuracy: {best_metrics['Accuracy']:.3f}")

        path = os.path.join("models", "classification", model_name)
        save_model(path, best_model, best_params, best_metrics)

# Define a function to find the best overall classification model
def find_best_model(models):

    best_model = None
    best_params = {}
    best_accuracy = float("-inf")
    best_metrics = {}

    for model_name, model_info in models.items():

        """Load model class from the dictionary"""
        model_class = model_info["model_class"]

        """Load hyperparameters and metrics from JSON files"""
        hyperparams_path = os.path.join("models", "classification", model_name, "hyperparameters.json")
        with open(hyperparams_path, "r") as f:
            hyperparameters = json.load(f)
        metrics_path = os.path.join("models", "classification", model_name, "metrics.json")
        with open(metrics_path, "r") as f:
            metrics = json.load(f)

        """Use the metric to find the best model"""
        if metrics["Accuracy"] > best_accuracy:
            best_accuracy = metrics["Accuracy"]

            best_model = model_class
            best_params = hyperparameters
            best_metrics = metrics

    return best_model, best_params, best_metrics

# Ensure that the code inside it is only executed if the script is being run directly
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    evaluate_all_models(models)
    best_model, best_params, best_metrics = find_best_model(models)
    print(f"- Model: {best_model}, Accuracy: {best_metrics['Accuracy']:.3f}")