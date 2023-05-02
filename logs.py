import os
import csv


MAIN_DIR = "/Users/johanna/PycharmProjects/"


def create_log_dir(parent_dir, parent_dir_info, repetition_index, fold_index):

    # Parent directory
    if parent_dir is None:
        parent_dir_created = f"{MAIN_DIR}logs/{parent_dir_info}"
        if not os.path.exists(parent_dir_created):
            os.makedirs(parent_dir_created)

    # Subdirectory for the specific repetition and fold
    if repetition_index is not None and fold_index is not None:
        sub_dir = f"{parent_dir_created}/rep_{repetition_index}_fold_{fold_index}"
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)
    else:
        sub_dir = parent_dir_created

    return sub_dir, parent_dir_created


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

