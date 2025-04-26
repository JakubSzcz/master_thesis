from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier


def random_forest_classifier(x_train, x_test, y_train, y_test, print_report: bool = False):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(n_estimators=50, class_weight='balanced', random_state=42, max_depth=3))
    ])

    # evaluate model
    pipeline.fit(x_train, y_train)
    y_pred_all = pipeline.predict(x_test)
    if print_report:
        print("Random Forest report:\n", classification_report(y_test, y_pred_all))
    print("Random Forest accuracy:", accuracy_score(y_test, y_pred_all))

    return pipeline


def knn_classifier(x_train, x_test, y_train, y_test, n_neighbors=10, print_report: bool = False):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier(n_neighbors=n_neighbors))
    ])

    pipeline.fit(x_train, y_train)
    y_pred = pipeline.predict(x_test)
    if print_report:
        print(classification_report(y_test, y_pred, zero_division=0))
    print("KNN Accuracy:", accuracy_score(y_test, y_pred))

    return pipeline


def xgboost_classifier(x_train, x_test, y_train, y_test, print_report: bool = False):
    # Encode string labels to integers
    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(y_train)

    # Create and train the model
    model = xgb.XGBClassifier(eval_metric='mlogloss', random_state=42)
    model.fit(x_train, y_train_encoded)

    # Predict and decode labels
    y_pred_encoded = model.predict(x_test)
    y_pred = encoder.inverse_transform(y_pred_encoded)

    if print_report:
        print(classification_report(y_test, y_pred, zero_division=0))
    print("XGBoost Accuracy:", accuracy_score(y_test, y_pred))

    return model


def mlp_classifier(x_train, x_test, y_train, y_test, print_report: bool = False):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPClassifier(
            hidden_layer_sizes=(10, 2),
            max_iter=200,
            alpha=0.001,
            random_state=42,
            early_stopping=True,
            activation="relu"
        ))
    ])
    pipeline.fit(x_train, y_train)
    y_pred = pipeline.predict(x_test)

    if print_report:
        print(classification_report(y_test, y_pred, zero_division=0))
    print("MLP Accuracy:", accuracy_score(y_test, y_pred))

    return pipeline


def decision_tree_classifier(x_train, x_test, y_train, y_test, print_report: bool = False):

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('dtc', DecisionTreeClassifier(random_state=42))
    ])
    # Create and train the Decision Tree
    pipeline.fit(x_train, y_train)

    # Predict and decode results
    y_pred = pipeline.predict(x_test)
    # Output report
    if print_report:
        print(classification_report(y_test, y_pred, zero_division=0))
    print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred))

    return pipeline
