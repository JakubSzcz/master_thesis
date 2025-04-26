import time
import os
import ast
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import skew
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

import util.math as mymath
import ml.models.best_wavelet_models as my_models


def print_features_v1_wavelet(frequency_bins: np.ndarray, wavelets: np.ndarray, group_wavelet_family: bool = False,
                              remove_outliers_flag: bool = False):
    """
    Visualize statistical parameters of each wavelet frequency distribution
    :param frequency_bins: N x M matrix of frequency distribution for different frames of signal
    :param wavelets: N x 1 matrix of wavelets used for compression for each frame
    :param group_wavelet_family: boolean flag to group wavelets by family
    :param remove_outliers_flag: boolean flag to remove outliers from frequency distribution
    """

    # calculate statistical metrics
    means = np.mean(frequency_bins, axis=1)
    variances = np.var(frequency_bins, axis=1)
    stds = np.std(frequency_bins, axis=1)
    skewnesses = skew(frequency_bins, axis=1)
    df = pd.DataFrame({
        'mean': means,
        'variance': variances,
        'std': stds,
        'skewness': skewnesses,
        'wavelet': wavelets
    })

    # extract wavelet family
    if group_wavelet_family:
        df['wavelet_family'] = df['wavelet'].str.extract(r'^([a-zA-Z]+)')

    # create boxplots for each feature grouped by wavelet
    sns.set(style="whitegrid")
    numerical_columns = ['mean', 'variance', 'std', 'skewness']
    for col in numerical_columns:
        # remove outliers
        if remove_outliers_flag:
            df = mymath.remove_outliers_df(df, col)

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


def cast_string_array_to_ndarray(df: pd.DataFrame, group_freq_method: str = None, log_transform_flag: bool = False,
                                 group_wavelet_family: bool = False) -> (np.array, np.ndarray):
    """
    Transform a string array retrieved from a.csv file to a numpy array of floats
    :param df: pandas dataframe N x 2 with string 'frequency_beans' and 'wavelet' columns
    :param group_freq_method: method for grouping frequencies into bins. Possible values: ["top_100", "mean_grouped_100]
    :param log_transform_flag: flag to apply log transformation to frequency bins
    :param group_wavelet_family: flag to group wavelets by family (e.g. coif12 -> coif)
    :return: tuple of two processed numpy arrays: (frequency bins, wavelets)
    """
    df = balance_data(df)
    #df = take_top_3_balanced(df)
    # cast string to array
    x_df = df['frequency_beans'].apply(ast.literal_eval)
    y_df = df['wavelet_family'] if group_wavelet_family else df['wavelet']

    # drop inconsistent sizes
    df_sizes = x_df.apply(len)
    df_desired_size = df_sizes.mode()[0]
    inconsistent_indexes = df_sizes[df_sizes != df_desired_size].index

    x_df = x_df.drop(inconsistent_indexes).reset_index(drop=True)
    y_df = y_df.drop(inconsistent_indexes).reset_index(drop=True)

    if log_transform_flag:
        x_df = x_df.apply(mymath.log_transform)

    if group_freq_method == "top_100":
        x_df = x_df.apply(mymath.get_top_100_bins)
    elif group_freq_method == "mean_grouped_100":
        x_df = x_df.apply(mymath.get_100_mean_bins)

    return np.stack(x_df), np.stack(y_df)

def take_top_3_balanced(df: pd.DataFrame) -> pd.DataFrame:
    top3_wavelets = df['wavelet'].value_counts().nlargest(8).index.tolist()
    df_top3 = df[df['wavelet'].isin(top3_wavelets)]
    min_count = df_top3['wavelet'].value_counts().min()

    # Step 4: Sample min_count rows from each wavelet
    balanced_top3 = df_top3.groupby('wavelet').sample(n=min_count, random_state=42).reset_index(drop=True)
    return balanced_top3


def balance_data(df: pd.DataFrame) -> pd.DataFrame:
    df["wavelet_family"] = df['wavelet'].str.extract(r'^([a-zA-Z]+)')[0]
    min_size = df['wavelet_family'].value_counts().min()
    df_balanced = df.groupby('wavelet_family').sample(n=min_size, random_state=42).reset_index(drop=True)
    return df_balanced


def prepare_sub_sets(take_all_files: bool = False, return_whole_df: bool = False,
                     group_freq_method: str = "mean_grouped_100") -> tuple:
    """
    Reads .csv files with data, process them and splits them into train and test sets
    :param group_freq_method: method for grouping frequencies into bins. Possible values: ["top_100", "mean_grouped_100]
    :param take_all_files: flag to take all files from the directory, otherwise only files with specific names are taken
    :param return_whole_df: flag to also return the whole dataframe instead of only train and test sets
    :return:
    """
    start_time = time.time()
    print("Start reading data...")
    csv_source_files_path = "../datasets/best_wavelet_v2/"
    csv_source_files = []
    if take_all_files:
        print("Reading all files from the directory...")
        files = os.listdir(csv_source_files_path)
    else:
        print("Reading preselected files from the directory...")
        # files = ["best_wavelet_v2_badinerie.csv", "best_wavelet_v2_rondo-alla-turca.csv",
        #          "best_wavelet_v2_confutatis.csv", "best_wavelet_v2_sound.csv", "best_wavelet_v2_music.csv"]
        files = ["best_wavelet_v2_badinerie.csv", "best_wavelet_v2_rondo-alla-turca.csv"]

    # read csv files
    for file in files:
        csv_source_files.append(pd.read_csv(csv_source_files_path + file))

    df_all = pd.concat(csv_source_files, ignore_index=True)
    df_all = df_all.sample(frac=1).reset_index(drop=True)

    print("Reading data finished.")

    print("Start processing data...")
    x, y = cast_string_array_to_ndarray(df_all, group_freq_method, True, True)

    # split for training and test sets for each df
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, stratify=y, test_size=0.2, random_state=42)
    print("Processing data finished.")
    print(
        f"Finished data preparation with {round(time.time() - start_time, 2)}s. Total data after filtration: {x.shape[0]}")

    if return_whole_df:
        return x_tr, x_te, y_tr, y_te, x, y
    return x_tr, x_te, y_tr, y_te


# EVALUATION
return_whole_df = False
if return_whole_df:
    x_train, x_test, y_train, y_test, x_all, y_all = prepare_sub_sets(take_all_files=False,
                                                                      return_whole_df=return_whole_df)
    print_features_v1_wavelet(x_all, y_all, group_wavelet_family=True, remove_outliers_flag=True)
else:
    x_train, x_test, y_train, y_test = prepare_sub_sets(take_all_files=False, return_whole_df=return_whole_df)

my_models.random_forest_classifier(x_train, x_test, y_train, y_test)
my_models.knn_classifier(x_train, x_test, y_train, y_test)
my_models.xgboost_classifier(x_train, x_test, y_train, y_test)
my_models.decision_tree_classifier(x_train, x_test, y_train, y_test)
model_mlp = my_models.mlp_classifier(x_train, x_test, y_train, y_test)

# joblib.dump(model_mlp, './../models/best_wavelet/mlp_100_4_200_01.joblib')
