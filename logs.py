import os
import glob
from filelock import FileLock
import sqlite3


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


def delete_other_checkpoints(log_dir, best_trial_number):
    # Delete all trial checkpoints but the best
    checkpoint_files = glob.glob(os.path.join(log_dir, "checkpoints", "*.ckpt"))
    for file in checkpoint_files:
        if f"model_trial_{best_trial_number}" not in file:
            os.remove(file)


def save_test_results(log_dir, repetition_index, fold_index, test_acc, best_val_acc, model, dataset, experiment):
    # Create a SQLite database to log the test accuracies
    file_name = f"{model}_{dataset}_{experiment}_test_accuracies.db"
    db_path = os.path.join(log_dir, file_name)
    lock_path = os.path.join(log_dir, "db.lock")

    with FileLock(lock_path):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Create table if it doesn't exist
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS results
        (rep INTEGER, fold INTEGER, test_acc_fold REAL, best_val_acc REAL)
        ''')
        conn.commit()

        # Insert or update the test_acc_fold
        cursor.execute('''
        INSERT INTO results
        (rep, fold, test_acc_fold, best_val_acc)
        VALUES (?, ?, ?, ?)
        ''', (repetition_index, fold_index, test_acc, best_val_acc))
        conn.commit()

        conn.close()


