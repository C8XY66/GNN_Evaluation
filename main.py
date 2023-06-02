from model import GNNModel
from datamodule import GraphDataModule
from logs import create_parent_dir, create_sub_dir, save_test_results, delete_other_checkpoints
from training import create_trainer, objective

import datetime
import pytz
import os
import argparse
import torch
import optuna
import logging
import warnings

# Disable user warnings (Enable for troubleshooting)
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)

# USER INPUTS
parser = argparse.ArgumentParser()
# TODO: specify MAIN_DIR
parser.add_argument("--MAIN_DIR", type=str, default="/Users/",
                    help="main directory of project, where logs are saved")

parser.add_argument("--EXPERIMENT", type=str, default='WithNF',
                    help="type of experiment: WithNF, WithoutNF (default: WithNF). "
                         "For graphs with or without node features")
parser.add_argument("--MODEL", type=str, default="GIN", help="name of model: GIN, DGCNN, MLP (default: GIN)")
parser.add_argument("--DATASET", type=str, default="MUTAG", help="name of dataset: MUTAG, NCI1, PROTEINS, IMDB-BINARY, "
                                                                 "IMDB-MULTI (default: MUTAG)")
parser.add_argument("--FOLDS", type=int, default=10, help="number of folds dataset is split into")
parser.add_argument("--REPS", type=int, default=10, help="number of total repetitions")
parser.add_argument("--EPOCHS", type=int, default=1000, help="number of epochs to train each trial of fold")
parser.add_argument("--PATIENCE", type=int, default=100, help="patience for early stopping monitoring val_loss")
parser.add_argument("--TRIALS", type=int, default=50, help="number of trials for hyperparameter optimization "
                                                           "(default:50)")
parser.add_argument("--START_REP", type=int, default=0, help="from which repetition to start (default: 0)")
parser.add_argument("--STOP_REP", type=int, default=None, help="at which repetition to stop (default: None)")
parser.add_argument("--START_FOLD", type=int, default=0, help="from which fold to start (default: 0)")
parser.add_argument("--STOP_FOLD", type=int, default=None, help="at which fold to stop (default: None)")

parser.add_argument("--PARENT_DIR", type=str, default=None,
                    help="name of parent directory for resuming interrupted run (default: None).")
parser.add_argument("--RUNNING_MODE", type=str, default="single",
                    help="running mode: single, parallel (default: 'single'). Code executed as single run or folds run "
                         "separately in parallel")
args = parser.parse_args()

# Check if inputs are valid
if args.EXPERIMENT not in ["WithNF", "WithoutNF"]:
    raise ValueError("Experiment must be 'WithNF' or 'WithoutNF'")
if args.MODEL not in ["GIN", "DGCNN", "MLP"]:
    raise ValueError("Model name must be 'GIN' or 'DGCNN'")
if args.DATASET not in ["MUTAG", "NCI1", "PROTEINS", "IMDB-BINARY", "IMDB-MULTI"]:
    raise ValueError("Dataset name must be one of the following: 'MUTAG', 'NCI1', 'PROTEINS', 'IMDB-BINARY', "
                     "'IMDB-MULTI'")
if args.RUNNING_MODE not in ["single", "parallel"]:
    raise ValueError("Running mode must be one of the following: 'single', 'parallel'")
if not os.path.isdir(args.MAIN_DIR):
    raise ValueError("MAIN_DIR is an invalid directory")
if args.PARENT_DIR is not None and not os.path.isdir(args.PARENT_DIR):
    raise ValueError("PARENT_DIR is an invalid directory")


if __name__ == "__main__":
    # SET UP EXPERIMENT
    # Check for CUDA system support and use GPU if available otherwise run on CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset type passed to DataModule (data preprocessing) and objective -> model (type of MLP baseline)
    dataset_type = "chemical" if args.DATASET in ["MUTAG", "NCI1", "PROTEINS"] \
        else "social" if args.DATASET in ["IMDB-BINARY", "IMDB-MULTI"] else None

    # Create log folder
    parent_dir_info = f"{args.MODEL}_{args.DATASET}_{args.EXPERIMENT}_reps_{args.REPS}_folds_{args.FOLDS}" \
                      f"_epochs_{args.EPOCHS}_pat_{args.PATIENCE}"

    now = datetime.datetime.now(pytz.timezone("Europe/Zurich")).strftime("%Y-%m-%d_%H-%M")
    if args.RUNNING_MODE == "single": parent_dir_info += f"_{now}"  # Add timestamp for "single" running mode

    parent_dir = create_parent_dir(parent_dir=args.PARENT_DIR, parent_dir_info=parent_dir_info, main_dir=args.MAIN_DIR)

    # Datamodule
    datamodule = GraphDataModule(dataset_name=args.DATASET, dataset_type=dataset_type, experiment=args.EXPERIMENT)
    datamodule.prepare_data()

    # EXPERIMENT LOOP
    for rep in range(args.START_REP, args.REPS if args.STOP_REP is None else args.STOP_REP):
        datamodule.setup_rep(rep=rep, folds=args.FOLDS)  # Shuffle and split into folds

        for fold in range(args.START_FOLD if rep == args.START_REP else 0, args.FOLDS if args.STOP_FOLD is None
        else args.STOP_FOLD):
            datamodule.setup(fold=fold)  # Shuffle and split into train and validation sets
            # Create sub folder per fold of repetition
            log_dir = create_sub_dir(parent_dir=parent_dir, repetition_index=rep, fold_index=fold)

            # TRAINING
            # Create new optuna study for each fold
            study = optuna.create_study(direction="maximize",
                                        pruner=optuna.pruners.MedianPruner(n_warmup_steps=30),
                                        sampler=optuna.samplers.TPESampler(seed=42),
                                        study_name=f"rep_{rep}_fold_{fold}"
                                        )

            study.optimize(lambda trial: objective(trial=trial, datamodule=datamodule, log_dir=log_dir,
                                                   epochs=args.EPOCHS, patience=args.PATIENCE, model_name=args.MODEL,
                                                   dataset_type=dataset_type), n_trials=args.TRIALS)

            best_val_acc = study.best_trial.value
            print(f"Best trial for fold {fold}: Trial {study.best_trial.number} "
                  f"with Val Accuracy {best_val_acc:.4f}")

            # TESTING
            # Load the model with the best hyperparameters
            checkpoint_name = f"model_trial_{study.best_trial.number}.ckpt"
            checkpoint_path = os.path.join(log_dir, "checkpoints", checkpoint_name)
            checkpoint = torch.load(checkpoint_path)  # Load the checkpoint dictionary from the file
            init_args = checkpoint["init_args"]  # Access the saved initialization parameters
            best_model = GNNModel(**init_args)  # Initialize the model using the saved parameters
            best_model.load_state_dict(checkpoint["state_dict"])
            delete_other_checkpoints(log_dir=log_dir, best_trial_number=study.best_trial.number)

            # Test the best model
            trainer = create_trainer(log_dir=log_dir, epochs=args.EPOCHS, testing=True)
            test_result = trainer.test(model=best_model, datamodule=datamodule)
            test_acc = test_result[0]["test_acc"]
            print(f"Test accuracy for fold {fold}: {test_acc}")
            save_test_results(log_dir=parent_dir, repetition_index=rep, fold_index=fold, test_acc=test_acc,
                              best_val_acc=best_val_acc, model=args.MODEL, dataset=args.DATASET,
                              experiment=args.EXPERIMENT)
