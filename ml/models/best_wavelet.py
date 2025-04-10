from scipy.stats import skew
import seaborn as sns
import os
import ast
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier


def print_features_v1_wavelet(df: pd.DataFrame, group_wavelet_family: bool = False, remove_outliers_flag: bool = False):

    if group_wavelet_family:
        df['wavelet_family'] = df['wavelet'].str.extract(r'^([a-zA-Z]+)')
    sns.set(style="whitegrid")
    #numerical_columns = ['mean', 'variance', 'std', 'skewness', 'energy', 'psnr', 'spectral_entropy']
    numerical_columns = ['mean', 'variance', 'std', 'skewness']

    # Create boxplots for each feature grouped by wavelet
    for col in numerical_columns:
        if remove_outliers_flag:
            df = remove_outliers(df, col)

        plt.figure(figsize=(14, 6))
        if group_wavelet_family:
            sns.boxplot(data=df, x='wavelet_family', y=col)
            plt.title(f'{col} distribution by Wavelet family')
        else:
            sns.boxplot(data=df, x='wavelet', y=col)
            plt.title(f'{col} distribution by Wavelet')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()


def remove_outliers(df: pd.DataFrame, column):
    df_col = df.copy()
    # Remove outliers using IQR
    Q1 = df_col[column].quantile(0.25)
    Q3 = df_col[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df_filtered = df_col[(df_col[column] >= lower_bound) & (df_col[column] <= upper_bound)]
    return df_filtered


def filter_dataset(df: pd.DataFrame, min_samples: int, top_number: int) -> pd.DataFrame | None:
    """
    Returns dataset of only mostly used wavelets
    :param df: original dataset
    :param min_samples: minimum number of wavelet occurrences
    :param top_number: how many wavelet of top occurrences should be taken into consideration
    :return: filtered dataset
    """
    class_counts = df["wavelet"].value_counts()
    valid_classes = class_counts[class_counts >= min_samples].index
    df_min_samples = df[df["wavelet"].isin(valid_classes)].copy()
    if df_min_samples.shape[0] == 0:
        return None

    # take top n wavelets
    top_wavelets = df_min_samples['wavelet'].value_counts().nlargest(top_number).index
    filtered_df = df_min_samples[df_min_samples['wavelet'].isin(top_wavelets)]
    return filtered_df

def get_top_100(arr):
    return np.sort(arr)[-100:][::-1]


def cast_string_array_to_ndarray(df: pd.DataFrame, return_top_100_freqs: bool = True) -> (np.array, np.ndarray):
    x_df = df['frequency_beans'].apply(ast.literal_eval)
    y_df = df['wavelet']
    df_sizes = x_df.apply(len)
    df_desired_size = df_sizes.mode()[0]
    inconsistent_indexes = df_sizes[df_sizes != df_desired_size].index

    x_df = x_df.drop(inconsistent_indexes).reset_index(drop=True)
    y_df = y_df.drop(inconsistent_indexes).reset_index(drop=True)
    if return_top_100_freqs:
        x_df["frequency_beans"] = x_df["frequency_beans"].apply(get_top_100)

    return np.stack(x_df), np.stack(y_df)


def prepare_sub_sets(csv_source_files_path, return_whole_df: bool = False):
    print("Data preparation...")
    csv_source_files = []
    files = os.listdir(csv_source_files_path)
    # files = ["best_wavelet_v2_badinerie.csv", "best_wavelet_v2_rondo-alla-turca.csv", "best_wavelet_v2_confutatis.csv",
    #          "best_wavelet_v2_sound.csv", "best_wavelet_v2_music.csv"]
    # read all csv
    for file in files:
        csv_source_files.append(pd.read_csv(csv_source_files_path + file))

    df_all = pd.concat(csv_source_files, ignore_index=True)
    if return_whole_df:
        return df_all

    # features and target
    # V1
    # df_all_filtered = filter_dataset(df_all, min_samples, top_number)
    # x = df_all_filtered.drop(columns=['wavelet', "n_samples"])  # features
    # y = df_all_filtered['wavelet']  # wavelet

    # V2
    x, y = cast_string_array_to_ndarray(df_all)

    # split for training and test sets for each df
    x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.2, random_state=42)
    print(f"Finished data preparation. Total data after filtration: {x.shape[0]}")
    return x_train, x_test, y_train, y_test


def random_forest_classifier(x_train, x_test, y_train, y_test, print_report: bool = False):
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42))
    ])

    # evaluate model
    pipe.fit(x_train, y_train)
    y_pred_all = pipe.predict(x_test)
    if print_report:
        print("Random Forest report:\n", classification_report(y_test, y_pred_all))
    print("Random Forest accuracy:", accuracy_score(y_test, y_pred_all))


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


def xgboost_classifier(x_train, x_test, y_train, y_test, print_report: bool = False):
    # Encode string labels to integers
    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(y_train)
    y_test_encoded = encoder.transform(y_test)

    # Create and train the model
    model = xgb.XGBClassifier(eval_metric='mlogloss', random_state=42)
    model.fit(x_train, y_train_encoded)

    # Predict and decode labels
    y_pred_encoded = model.predict(x_test)
    y_pred = encoder.inverse_transform(y_pred_encoded)

    if print_report:
        print(classification_report(y_test, y_pred, zero_division=0))
    print("XGBoost Accuracy:", accuracy_score(y_test, y_pred))


def mlp_classifier(x_train, x_test, y_train, y_test, print_report: bool = False):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=100)),
        ('mlp', MLPClassifier(
            hidden_layer_sizes=(1024, 64),
            max_iter=500,
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


def decision_tree_classifier(x_train, x_test, y_train, y_test, print_report: bool = False):
    # Encode string labels into integers
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


# PARAMETERS
# csv_source_files_path = "../datasets/best_wavelet/"
csv_source_files_path = "../datasets/best_wavelet_v2/"
min_samples = 5
top_number = 10  # max 16

x_train, x_test, y_train, y_test = prepare_sub_sets(csv_source_files_path)
#
# random_forest_classifier(x_train, x_test, y_train, y_test)
# knn_classifier(x_train, x_test, y_train, y_test, top_number)
# xgboost_classifier(x_train, x_test, y_train, y_test)
mlp_classifier(x_train, x_test, y_train, y_test)
# decision_tree_classifier(x_train, x_test, y_train, y_test)

df = prepare_sub_sets(csv_source_files_path, return_whole_df=True)
freqs, wavelets = cast_string_array_to_ndarray(df, True)

means = np.mean(freqs, axis=1)
variances = np.var(freqs, axis=1)
stds = np.std(freqs, axis=1)
skewnesses = skew(freqs, axis=1)
df_features = pd.DataFrame({
    'mean': means,
    'variance': variances,
    'std': stds,
    'skewness': skewnesses,
    'wavelet': wavelets
})
print_features_v1_wavelet(df_features, True, True)