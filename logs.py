import config

import os
import csv
import datetime
import pytz


def create_log_dir(repetition_index, fold_index):

    # Current timestamp
    now = datetime.datetime.now(pytz.timezone('Europe/Zurich')).strftime("%Y-%m-%d_%H-%M")

    # Parent directory
    parent_dir_info = f"{config.DATASET_NAME}_reps_{config.REP}_folds_{config.N_SPLITS}_epochs_{config.EPOCHS}"

    if config.PARENT_DIR is None:
        config.PARENT_DIR = f"{config.MAIN_DIR}logs/{parent_dir_info}_{now}"
        if not os.path.exists(config.PARENT_DIR):
            os.makedirs(config.PARENT_DIR)

    # Subdirectory for the specific repetition and fold
    if repetition_index is not None and fold_index is not None:
        sub_dir = f"{config.PARENT_DIR}/rep_{repetition_index}_fold_{fold_index}"
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)
    else:
        sub_dir = config.PARENT_DIR

    return sub_dir


def save_test_results(log_dir, repetition_index, fold_index, test_acc, avg_performance=None,
                      overall_avg_performance=None):
    # Create a CSV file to log the test accuracies
    file_path = os.path.join(log_dir, 'test_accuracies.csv')

    if not os.path.exists(file_path):
        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(
                ['repetition', 'fold', 'test_accuracy_fold', 'Avg_Performance_Rep', 'Overall_Avg_Performance'])

    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)

        if repetition_index is not None and fold_index is not None:
            row = [repetition_index, fold_index, test_acc]
        else:
            row = ['', '', '']

        if avg_performance is not None:
            row.append(avg_performance)
        else:
            row.append('')

        if overall_avg_performance is not None:
            row.append(overall_avg_performance)
        else:
            row.append('')

        writer.writerow(row)

