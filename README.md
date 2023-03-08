# AiCore Scenario - Multimodal Property Intelligence System

This project aims to develop a framework to systematically train, tune, and evaluate a wide range of machine learning models, from simple regression to neural network models, that can be applied to various datasets.

## Milestone 1
In Milestone 1, the focus was on setting up the development environment. This involved installing the necessary tools and libraries required to run the program. A virtual environment was created to keep the dependencies for the project isolated from the other projects on the machine. This step helped to avoid potential conflicts with other projects.

## Milestone 2 & 3
In Milestone 3, the script `clean_tabular_data.py` was created, which cleans a tabular dataset of _Airbnb_ property listings. It removes or fills missing values and combines strings, returning the cleaned dataset. This script can be run as a standalone program or imported into other scripts, providing an efficient solution for data cleaning tasks. The output file `clean_tabular_data.csv` can be used for further analysis or machine learning applications.

## Milestone 4
In Milestone 4, the script `modelling.py` was created, which loads and splits data from a CSV file and performs standardization as a preprocessing step. Then, this data is utilized to train a simple linear regression model. To assess the performance of the model, the `evaluate_predictions` function is defined and executed on both the training and testing data.

To optimize the model's hyperparameters, two functions have been implemented. The first function, `custom_tune_regression_model_hyperparameters`, allows for manual tuning of the model's hyperparameters through a nested list comprehension. The second function, `tune_regression_model_hyperparameters`, employs `GridSearchCV` to perform automated hyperparameter tuning.

Finally, a function to tune, evaluate, and save multiple regression models called `evaluate_all_models` is defined. It takes in a dictionary of models, along with their corresponding hyperparameters. The models included are linear regression, decision tree, random forest, and gradient boosting. The best overall regression model is found by the `find_best_model` function, which compares their metrics.