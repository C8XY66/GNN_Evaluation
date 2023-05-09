from model import GNNModel
from datamodule import GraphDataModule
from logs import create_parent_dir, create_sub_dir, save_test_results
from training import create_trainer, objective

import datetime
import pytz
import os
import argparse
import numpy as np
import torch
import optuna
import logging
import warnings

logging.getLogger("pytorch_lightning.utilities.rank_zero").setLevel(logging.WARNING)
logging.getLogger("pytorch_lightning.accelerators.cuda").setLevel(logging.WARNING)
logging.getLogger("pytorch_lightning.callbacks.early_stopping").setLevel(logging.WARNING)
logging.getLogger("pytorch_lightning.callbacks.model_checkpoint").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning, module="pytorch_lightning.callbacks.model_checkpoint")
warnings.filterwarnings("ignore", category=UserWarning, module="pytorch_lightning.utilities")
warnings.filterwarnings("ignore", category=UserWarning, module="pytorch_lightning.trainer")
warnings.filterwarnings("ignore", category=UserWarning, module="optuna.trial")
warnings.filterwarnings("ignore", category=UserWarning)

parser = argparse.ArgumentParser()
parser.add_argument("--EXPERIMENT", type=str, default='with_node_features',
                    help="type of experiment: with_node_features, without_node_features (default: with_node_features)")
parser.add_argument("--MODEL", type=str, default="GIN", help="name of model: GIN, DGCNN, MLP (default: GIN)")
parser.add_argument("--DATASET", type=str, default="MUTAG", help="name of dataset (default: MUTAG)")
parser.add_argument("--N_SPLITS", type=int, default=10, help="number of folds dataset is split into")
parser.add_argument("--REPS", type=int, default=1, help="number of total repetitions")
parser.add_argument("--EPOCHS", type=int, default=1000, help="number of epochs to train each trial of fold")
parser.add_argument("--PATIENCE", type=int, default=100, help="patience for early stopping monitoring val_acc")
parser.add_argument("--TRIALS", type=int, default=8, help="number of trials for hyperparameter optimization")
parser.add_argument("--STARTING_REP", type=int, default=0, help="from which repetition to start (default: 0)")
parser.add_argument("--STARTING_FOLD", type=int, default=0, help="from which fold to start (default: 0)")
parser.add_argument("--MAIN_DIR", type=str, default="/Users/johanna/PycharmProjects/",
                    help="main directory of project, where logs are saved")
parser.add_argument("--PARENT_DIR", type=str, default=None,
                    help="name of parent directory for resuming interrupted run (default: None). Use format like "
                         "'/Users/johanna/PycharmProjects/logs/GIN_NCI1_reps_2_folds_5_epochs_100_2023-04-26_09-00'")
args = parser.parse_args()

# Check if inputs are valid
if args.EXPERIMENT not in ["with_node_features", "without_node_features"]:
    raise ValueError("Experiment must be 'with_node_features' or 'without_node_features'")
if args.MODEL not in ["GIN", "DGCNN", "MLP"]:
    raise ValueError("Model name must be 'GIN' or 'DGCNN'")
if args.DATASET not in ["MUTAG", "NCI1", "PROTEINS", "COLLAB", "IMDB-BINARY"]:
    raise ValueError("Dataset name must be one of the following: 'MUTAG', 'NCI1', 'PROTEINS', 'COLLAB', 'IMDB-BINARY'")
if args.PARENT_DIR is not None and not os.path.isdir(args.PARENT_DIR):
    raise ValueError("Invalid directory, should be of format: "
                     "'/Users/johanna/PycharmProjects/logs/GIN_NCI1_reps_2_folds_5_epochs_100_2023-04-26_09-00' ")


if __name__ == "__main__":
    # Check for CUDA system support and use GPU if available otherwise run on CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Experiment Setup
    experiment = args.EXPERIMENT
    model = args.MODEL
    dataset_type = "chemical" if args.DATASET in [ "MUTAG", "NCI1", "PROTEINS"] \
        else "social" if args.DATASET in ["COLLAB", "IMDB-BINARY"] else None

    # Log folder with current timestamp
    now = datetime.datetime.now(pytz.timezone("Europe/Zurich")).strftime("%Y-%m-%d_%H-%M")
    experiment_short = "WithoutNF" if args.EXPERIMENT == "without_node_features" else "WithNF"

    parent_dir_info = f"{model}_{args.DATASET}_{experiment_short}_reps_{args.REPS}_folds_{args.N_SPLITS}" \
                      f"_epochs_{args.EPOCHS}_pat_{args.PATIENCE}_{now}"
    parent_dir = create_parent_dir(parent_dir=args.PARENT_DIR, parent_dir_info=parent_dir_info, main_dir=args.MAIN_DIR)

    overall_performances = []

    # Experiment Loop
    for r in range(args.STARTING_REP, args.REPS):
        seed = r + 1  # Set a new seed for each repetition
        datamodule = GraphDataModule(dataset_name=args.DATASET, dataset_type=dataset_type,
                                     experiment=experiment, seed=seed)
        datamodule.prepare_data()
        fold_performances = []

        for fold in range(args.STARTING_FOLD if r == args.STARTING_REP else 0, args.N_SPLITS):
            # Create sub folder per fold of repetition
            log_dir = create_sub_dir(parent_dir=parent_dir, repetition_index=r, fold_index=fold)
            # Create a new study object for each fold
            study = optuna.create_study(direction="maximize",
                                        pruner=optuna.pruners.MedianPruner(n_startup_trials=50),
                                        sampler=optuna.samplers.TPESampler(seed=42),
                                        study_name=f"rep_{r}_fold_{fold}"
                                        )

            datamodule.setup(fold=fold)

            study.optimize(lambda trial: objective(trial=trial, datamodule=datamodule, log_dir=log_dir,
                                                   epochs=args.EPOCHS, patience=args.PATIENCE, model_name=model,
                                                   dataset_type=dataset_type), n_trials=args.TRIALS)

            print(f"Best trial for fold {fold}: Trial {study.best_trial.number} "
                  f"with Val Accuracy {study.best_trial.value:.4f}")

            # Load the model with the best hyperparameters
            checkpoint_name = f"model_trial_{study.best_trial.number}.ckpt"
            checkpoint_path = os.path.join(log_dir, "checkpoints", checkpoint_name)
            checkpoint = torch.load(checkpoint_path)  # Load the checkpoint dictionary from the file
            init_args = checkpoint["init_args"]  # Access the saved initialization parameters
            best_model = GNNModel(**init_args)  # Initialize the model using the saved parameters
            best_model.load_state_dict(checkpoint["state_dict"])

            # Test the best model
            trainer = create_trainer(log_dir=log_dir, epochs=args.EPOCHS, testing=True)
            test_result = trainer.test(model=best_model, datamodule=datamodule)
            test_acc = test_result[0]["test_acc"]
            print(f"Test accuracy for fold {fold}: {test_acc}")

            fold_performances.append(test_acc)

        avg_performance = np.mean(fold_performances)
        print(f"Average performance for repetition {r}: {avg_performance}")
        overall_performances.append(avg_performance)

        # Save test accuracies, average performance, and overall average performance after all folds are done
        for fold, test_acc in enumerate(fold_performances):
            if fold == args.N_SPLITS - 1:
                save_test_results(log_dir=parent_dir, repetition_index=r, fold_index=fold,
                                  test_acc=test_acc, avg_performance=avg_performance, overall_avg_performance=None)
            else:
                save_test_results(log_dir=parent_dir, repetition_index=r, fold_index=fold,
                                  test_acc=test_acc, avg_performance=None, overall_avg_performance=None)

    overall_avg_performance = np.mean(overall_performances)
    print(f"Overall average performance: {overall_avg_performance}")
    save_test_results(log_dir=parent_dir, repetition_index=None, fold_index=None,
                      test_acc=None, avg_performance=None, overall_avg_performance=overall_avg_performance)
