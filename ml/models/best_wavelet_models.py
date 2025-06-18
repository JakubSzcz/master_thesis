import time

import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV


def random_forest_classifier(x_train: np.ndarray, x_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray,
                             print_report: bool = False):
    s_time = time.time()

    params = {"n_estimators": [700, 800, 900], "max_depth": [4, 5, 6]}
    grid = GridSearchCV(estimator=RandomForestClassifier(class_weight='balanced', random_state=42),
                        param_grid=params, cv=3, scoring='accuracy', n_jobs=-1)
    grid.fit(x_train, y_train)
    model = grid.best_estimator_
    y_pred = model.predict(x_test)

    if print_report:
        print("Random Forest report:\n", classification_report(y_test, y_pred))
    print(f"Random Forest accuracy: {accuracy_score(y_test, y_pred)}\t train time: {round(time.time() - s_time, 2)}s")
    print(grid.best_params_)

    return model


def knn_classifier(x_train: np.ndarray, x_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray,
                   print_report: bool = False):
    s_time = time.time()
    params = {"n_neighbors": [17, 19, 23, 25, 27, 28, 30]}
    grid = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=params, cv=3, scoring='accuracy', n_jobs=-1)
    grid.fit(x_train, y_train)

    model = grid.best_estimator_
    y_pred = model.predict(x_test)

    if print_report:
        print(classification_report(y_test, y_pred, zero_division=0))
    print(f"KNN Accuracy: {accuracy_score(y_test, y_pred)}\t train time: {round(time.time() - s_time, 2)}s")
    print(grid.best_params_)
    return model


def xgboost_classifier(x_train: np.ndarray, x_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray,
                       print_report: bool = False):
    s_time = time.time()
    model = xgb.XGBClassifier(eval_metric='mlogloss', random_state=42)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    if print_report:
        print(classification_report(y_test, y_pred, zero_division=0))
    print(f"XGBoost Accuracy:{accuracy_score(y_test, y_pred)}\t train time: {round(time.time() - s_time, 2)}s")
    return model


def mlp_classifier(x_train: np.ndarray, x_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray,
                   print_report: bool = False):
    s_time = time.time()
    params = {
        'hidden_layer_sizes': [(50, 50, 50), (50, 50, 50, 10), (100, 50, 50, 50, 10), (100, 50, 25, 50, 10)],
        'alpha': [0.0001, 0.001, 0.01],
    }
    grid = GridSearchCV(
        estimator=MLPClassifier(max_iter=400, random_state=42, early_stopping=True, solver='adam', activation='relu'),
        param_grid=params,
        cv=3, scoring='accuracy', n_jobs=-1)
    grid.fit(x_train, y_train)

    model = grid.best_estimator_
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    if print_report:
        print(classification_report(y_test, y_pred, zero_division=0))
    print(f"MLP Accuracy:{accuracy_score(y_test, y_pred)}\t train time: {round(time.time() - s_time, 2)}s")
    print(grid.best_params_)
    return model

"""
Finished data preparation with 109.07s. Total data after filtration: 315690
Preprocessing data...
Preprocessing finished.
Random Forest accuracy: 0.12729259716810795	 train time: 14279.42s
{'max_depth': 6, 'n_estimators': 700}
KNN Accuracy: 0.12619975292217048	 train time: 2072.59s
{'n_neighbors': 28}
MLP Accuracy:0.13375463270930343	 train time: 1476.08s
{'alpha': 0.001, 'hidden_layer_sizes': (50, 50, 50)}
XGBoost Accuracy:0.1343089739934746	 train time: 101.98s
Decision Tree Accuracy:0.12282619024992873	 train time: 1071.33s
{'max_depth': 5, 'min_samples_split': 2}
Total time: 19111.8s
"""

def decision_tree_classifier(x_train: np.ndarray, x_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray,
                             print_report: bool = False):
    s_time = time.time()
    params = {
        'max_depth': [None, 5, 10, 20, 50],
        'min_samples_split': [2, 5, 10]
    }
    grid = GridSearchCV(estimator=DecisionTreeClassifier(random_state=42), param_grid=params,
                        cv=3, scoring='accuracy', n_jobs=-1)
    grid.fit(x_train, y_train)
    model = grid.best_estimator_
    y_pred = model.predict(x_test)

    if print_report:
        print(classification_report(y_test, y_pred, zero_division=0))
    print(f"Decision Tree Accuracy:{accuracy_score(y_test, y_pred)}\t train time: {round(time.time() - s_time, 2)}s")
    print(grid.best_params_)
    return model
