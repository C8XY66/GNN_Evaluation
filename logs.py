import os
import pandas as pd
import numpy as np
from filelock import FileLock


def create_parent_dir(parent_dir, parent_dir_info, main_dir):
    # Parent directory
    if parent_dir is None:
        parent_dir = f"{main_dir}logs/{parent_dir_info}"
        os.makedirs(parent_dir, exist_ok=True)
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
    lock_path = os.path.join(log_dir, "test_accuracies.lock")

    with FileLock(lock_path):
        if not os.path.exists(file_path):
            columns = ["rep", "fold", "test_acc_fold", "avg_perf_rep",
                       "std_dev_rep", "avg_perf_overall", "std_dev_overall"]
            df = pd.DataFrame(columns=columns)
            # Pre-populate the repetition and fold fields
            for rep in range(num_reps):
                for fld in range(num_folds):
                    new_row = pd.DataFrame({"rep": [rep], "fold": [fld],
                                            "test_acc_fold": [np.nan],
                                            "avg_perf_rep": [np.nan],
                                            "std_dev_rep": [np.nan],
                                            "avg_perf_overall": [np.nan],
                                            "std_dev_overall": [np.nan]})
                    df = pd.concat([df, new_row], ignore_index=True)
        else:
            df = pd.read_excel(file_path, engine='openpyxl')

        df.loc[(df['rep'] == repetition_index) & (df['fold'] == fold_index), 'test_acc_fold'] = test_acc

        # Calculate the average performance and standard deviation for the repetition
        rep_df = df.loc[df['rep'] == repetition_index]
        avg_performance = rep_df['test_acc_fold'].dropna().mean()
        std_dev_performance = rep_df['test_acc_fold'].dropna().std(ddof=1)
        df.loc[(df['rep'] == repetition_index) & (df['fold'] == num_folds - 1),
        ['avg_perf_rep', 'std_dev_rep']] = avg_performance, std_dev_performance

        # Calculate the overall average performance and standard deviation
        overall_avg_performance = df['test_acc_fold'].dropna().mean()
        overall_std_dev_performance = df['test_acc_fold'].dropna().std(ddof=1)
        df.loc[(df['rep'] == num_reps - 1) & (df['fold'] == num_folds - 1),
        ['avg_perf_overall', 'std_dev_overall']] = overall_avg_performance, overall_std_dev_performance

        df.to_excel(file_path, index=False, engine='openpyxl')

