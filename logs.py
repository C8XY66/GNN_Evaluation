import os
import glob
from filelock import FileLock
import sqlite3


def create_parent_dir(parent_dir: str, parent_dir_info: str, main_dir: str):
    """
       Creates the parent directory if it doesn't exist.
       Args:
           parent_dir (str): Parent directory.
           parent_dir_info (str): Parent directory information for folder name.
           main_dir (str): Main directory where parent directory is created under 'log' folder.
       Returns:
           str: Path to the parent directory.
       """
    # Parent directory
    if parent_dir is None:
        parent_dir = f"{main_dir}logs/{parent_dir_info}"
        os.makedirs(parent_dir, exist_ok=True)
    return parent_dir


def create_sub_dir(parent_dir: str, repetition_index: int, fold_index: int):
    """
        Creates the subdirectory for a specific repetition and fold if it doesn't exist.
        Args:
            parent_dir (str): Parent directory where subdirectory is created.
            repetition_index (int): Index of the repetition.
            fold_index (int): Index of the fold.
        Returns:
            str: Path to the subdirectory.
        """
    if repetition_index is not None and fold_index is not None:
        sub_dir = f"{parent_dir}/rep_{repetition_index}_fold_{fold_index}"
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)
    else:
        sub_dir = parent_dir

    return sub_dir


def delete_other_checkpoints(log_dir: str, best_trial_number: int):
    """
        Deletes checkpoints for all trials but the best.
        Args:
            log_dir (str): Log directory where checkpoints are saved under 'checkpoints' folder.
            best_trial_number (int): Trial number of the best trial.
        """
    # Delete all trial checkpoints but the best
    checkpoint_files = glob.glob(os.path.join(log_dir, "checkpoints", "*.ckpt"))
    for file in checkpoint_files:
        file_trial_number = int(file.split('_')[-1].split('.')[0])
        if file_trial_number != best_trial_number:
            os.remove(file)


def save_test_results(log_dir: str, repetition_index: int, fold_index: int, test_acc: float, best_val_acc: float,
                      model: str, dataset: str, experiment: str):
    """
        Saves test results to a SQLite database.
        Args:
            log_dir (str): Log directory where results are saved
            repetition_index (int): Index of the repetition.
            fold_index (int): Index of the fold in the repetition.
            test_acc (float): Test accuracy for the fold.
            best_val_acc (float): Validation accuracy of the best trial of the fold.
            model (str): Model name for filename.
            dataset (str): Dataset name for filename.
            experiment (str): Experiment name for filename.
        """
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