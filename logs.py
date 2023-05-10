import os
import pandas as pd
import numpy as np


def create_parent_dir(parent_dir, parent_dir_info, main_dir):
    # Parent directory
    if parent_dir is None:
        parent_dir = f"{main_dir}logs/{parent_dir_info}"
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
    return parent_dir


def create_sub_dir(parent_dir, repetition_index, fold_index):

    # Subdirectory for the specific repetition and fold
    if repetition_index is not None and fold_index is not None:
        sub_dir = f"{parent_dir}/rep_{repetition_index}_fold_{fold_index}"
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)
    else:
        sub_dir = parent_dir

    return sub_dir


def save_test_results(log_dir, repetition_index, fold_index, test_acc, num_folds, num_reps):
    # Create an Excel file to log the test accuracies
    file_path = os.path.join(log_dir, "test_accuracies.xlsx")

    if not os.path.exists(file_path):
        columns = ["rep", "fold", "test_acc_fold", "avg_perf_rep",
                   "std_dev_rep", "avg_perf_overall", "std_dev_overall"]
        df = pd.DataFrame(columns=columns)
        # Pre-populate the repetition and fold fields
        for rep in range(num_reps):
            for fld in range(num_folds):
                new_row = pd.DataFrame({"rep": [rep], "fold": [fld],
                                        "test_acc_fold": [np.nan],
                                        "avg_perf_rep": [""],
                                        "std_dev_rep": [""],
                                        "avg_perf_overall": [""],
                                        "std_dev_overall": [""]})
                df = pd.concat([df, new_row], ignore_index=True)
    else:
        df = pd.read_excel(file_path, engine='openpyxl')

    df.loc[(df['rep'] == repetition_index) & (df['fold'] == fold_index), 'test_acc_fold'] = test_acc

    # Calculate the average performance and standard deviation for the last fold
    if fold_index == num_folds - 1:
        rep_df = df.loc[df['rep'] == repetition_index]
        avg_performance = rep_df['test_acc_fold'].mean()
        std_dev_performance = rep_df['test_acc_fold'].std()
        df.loc[(df['rep'] == repetition_index) & (df['fold'] == fold_index),
               ['avg_perf_rep', 'std_dev_rep']] = avg_performance, std_dev_performance
        print(f"Average performance for repetition {repetition_index}: {avg_performance}")

    # Calculate the overall average performance and standard deviation for the last repetition
    if repetition_index == num_reps - 1 and fold_index == num_folds - 1:
        overall_avg_performance = df['test_acc_fold'].mean()
        overall_std_dev_performance = df['test_acc_fold'].std()
        df.loc[(df['rep'] == repetition_index) & (df['fold'] == fold_index),
               ['avg_perf_overall', 'std_dev_overall']] = overall_avg_performance, overall_std_dev_performance
        print(f"Overall average performance: {overall_avg_performance}")

    df.to_excel(file_path, index=False, engine='openpyxl')


