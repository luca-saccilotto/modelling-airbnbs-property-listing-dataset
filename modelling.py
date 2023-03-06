# Import libraries
import joblib
import json
import numpy as np
import os
import pandas as pd
from tabular_data import load_airbnb
from sklearn import linear_model, metrics, model_selection, preprocessing

# Define the file path and load the data
df = pd.read_csv("airbnb-property-listings/tabular_data/clean_tabular_data.csv", index_col = "ID")
X, y = load_airbnb(df, "Price_Night")

# Fit and transform the data
scaler = preprocessing.StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training, validation, and test sets
np.random.seed(1)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.3)
X_validation, X_test, y_validation, y_test = model_selection.train_test_split(X_test, y_test, test_size = 0.5)

# Print the shape of training data
print(X_train.shape, y_train.shape, end = "\n\n")

# Create a simple regression model to predict the nightly cost of each listing
sgdr = linear_model.SGDRegressor()
sgdr.fit(X_train, y_train)

# Make predictions on the training and testing data
y_train_pred = sgdr.predict(X_train)
y_test_pred = sgdr.predict(X_test)

# Define a function to evaluate the predictions on the training and testing data
def evaluate_predictions(y_train, y_test, y_train_pred, y_test_pred):

    """Calculate MSE for training and testing data"""
    mse_train = metrics.mean_squared_error(y_train, y_train_pred)
    mse_test = metrics.mean_squared_error(y_test, y_test_pred)

    """Calculate RMSE for training and testing data"""
    rmse_train = np.sqrt(mse_train)
    rmse_test = np.sqrt(mse_test)
    
    """Calculate R-Squared for training and testing data"""
    r2_train = metrics.r2_score(y_train, y_train_pred)
    r2_test = metrics.r2_score(y_test, y_test_pred)
    
    """Return the results as a dictionary"""
    results = {"Train MSE": round(mse_train, 3),
               "Test MSE": round(mse_test, 3),
               "Train RMSE": round(rmse_train, 3),
               "Test RMSE": round(rmse_test, 3),
               "Train R-Squared": round(r2_train, 3),
               "Test R-Squared": round(r2_test, 3)}
    
    return results

performance_metrics = evaluate_predictions(y_train, y_test, y_train_pred, y_test_pred)
print(performance_metrics, end = "\n\n")

# Create a dictionary containing the training, validation, and testing data
datasets = {"X_train": X_train, "y_train": y_train, 
           "X_test": X_test, "y_test": y_test, 
           "X_validation": X_validation, "y_validation": y_validation}

# Create a dictionary containing the ranges of the hyperparameters to be tuned
hyperparameters = {"alpha": [0.0001, 0.001, 0.01, 0.1],
                   "learning_rate": ["constant", "optimal", "invscaling"],
                   "max_iter": [1000, 2000, 3000],
                   "tol": [1e-3, 1e-4, 1e-5],
                   "penalty": ["l1", "l2"]}

# Implement a custom function to tune the hyperparameters of the model
def custom_tune_regression_model_hyperparameters(model_class, datasets, hyperparameters):

    """Create a nested list comprehensions of hyperparameter combinations"""
    combinations = [{"alpha": alpha, "learning_rate": learning_rate, "max_iter": max_iter, "tol": tol, "penalty": penalty}
                    for alpha in hyperparameters["alpha"]
                    for learning_rate in hyperparameters["learning_rate"]
                    for max_iter in hyperparameters["max_iter"]
                    for tol in hyperparameters["tol"]
                    for penalty in hyperparameters["penalty"]]

    """Find the best model based on validation set"""
    best_model = None
    best_params = {}
    best_rmse = float("inf")
    best_metrics = {}

    for c in combinations:
        model = model_class(alpha = c["alpha"], learning_rate = c["learning_rate"], max_iter = c["max_iter"], tol = c["tol"], penalty = c["penalty"])
        model.fit(datasets["X_train"], datasets["y_train"])

        """Use the trained model to make predictions on the validation set"""
        y_pred_validation = model.predict(datasets["X_validation"])
        rmse_validation = metrics.mean_squared_error(datasets["y_validation"], y_pred_validation, squared = False)
        r2_validation = metrics.r2_score(datasets["y_validation"], y_pred_validation)

        """
        The combination of hyperparameters that results in the lowest 
        RMSE on the validation set is selected as the best hyperparameters

        """
        if rmse_validation < best_rmse:
            best_model = model
            best_params = c
            best_rmse = rmse_validation
            best_metrics["Validation RMSE"] = round(rmse_validation, 3)
            best_metrics["Validation R-Squared"] = round(r2_validation, 3)

    """Return the best model, its hyperparameters, and performance metrics"""
    return best_model, best_params, best_metrics

best_model, best_params, best_metrics = custom_tune_regression_model_hyperparameters(linear_model.SGDRegressor, datasets, hyperparameters)
print("Model:", best_model, "RMSE:", best_metrics["Validation RMSE"], "R-Squared:", best_metrics["Validation R-Squared"])

# Use "GridSearchCV" to tune the hyperparameters of the model
def tune_regression_model_hyperparameters(model_class, datasets, hyperparameters):

    """Tuning hyperparameters"""
    grid = model_selection.GridSearchCV(model_class, hyperparameters)
    grid.fit(datasets["X_train"], datasets["y_train"])

    """Model prediction on validation set using best hyperparameters"""
    y_pred = grid.predict(datasets["X_validation"])

    """Calculate metrics on validation set"""
    rmse = metrics.mean_squared_error(datasets["y_validation"], y_pred, squared = False)
    r2 = metrics.r2_score(datasets["y_validation"], y_pred)

    """Store best model, best hyperparameters, and best metrics in a dictionary"""
    best_model = grid.best_estimator_
    best_params = grid.best_params_
    best_metrics = {"RMSE": rmse, "R-Squared": r2}
    
    return best_model, best_params, best_metrics

best_model, best_params, best_metrics = tune_regression_model_hyperparameters(linear_model.SGDRegressor(), datasets, hyperparameters)
print(f"Model: {best_model}, RMSE: {best_metrics['RMSE']:.3f}, R-Squared: {best_metrics['R-Squared']:.3f}")

# Define a function to save the model
def save_model(folder, best_model, best_params, best_metrics):

    """Create the folder if it doesn't exist"""
    os.makedirs(folder, exist_ok = True)
    
    """Save the model, its hyperparameters, and its metrics"""
    joblib.dump(best_model, os.path.join(folder, "model.joblib"))
    with open(os.path.join(folder, "hyperparameters.json"), 'w') as f:
        json.dump(best_params, f)
    with open(os.path.join(folder, "metrics.json"), "w") as f:
        json.dump(best_metrics, f)

folder = os.path.join("models", "regression", "linear_regression")
save_model(folder, best_model, best_params, best_metrics)