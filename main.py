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

parser = argparse.ArgumentParser()
parser.add_argument('--MODEL', type=str, default='GIN')
parser.add_argument('--DATASET', type=str, default='NCI1')
parser.add_argument('--N_SPLITS', type=int, default=2)
parser.add_argument('--REP', type=int, default=1)
parser.add_argument('--EPOCHS', type=int, default=5)
parser.add_argument('--STARTING_REP', type=int, default=0)
parser.add_argument('--STARTING_FOLD', type=int, default=0)
parser.add_argument('--PARENT_DIR', type=str, default=None)
#/Users/johanna/PycharmProjects/logs/NCI1_reps_2_folds_5_epochs_100_2023-04-26_09-00
args = parser.parse_args()

SEED = 42
PARENT_DIR = args.PARENT_DIR


if __name__ == '__main__':
    # Log folder with current timestamp
    now = datetime.datetime.now(pytz.timezone('Europe/Zurich')).strftime("%Y-%m-%d_%H-%M")
    parent_dir_info = f"{args.DATASET}_reps_{args.REP}_folds_{args.N_SPLITS}_epochs_{args.EPOCHS}_{now}"
    parent_dir = create_parent_dir(PARENT_DIR, parent_dir_info)

    overall_performances = []

    for r in range(args.STARTING_REP, args.REP):
        seed = r + 1  # Set a new seed for each repetition
        datamodule = GraphDataModule(dataset_name=args.DATASET, seed=seed)
        datamodule.prepare_data()
        fold_performances = []

        for fold in range(args.STARTING_FOLD if r == args.STARTING_REP else 0, args.N_SPLITS):
            # Create sub folder per fold of repetition
            log_dir = create_sub_dir(parent_dir, r, fold)
            # Create a new study object for each fold
            study = optuna.create_study(direction="maximize",
                                        pruner=optuna.pruners.MedianPruner(),
                                        sampler=optuna.samplers.TPESampler(seed=SEED),
                                        study_name=f"rep_{r}_fold_{fold}",
                                        # storage=f"sqlite:///{log_dir}/rep_{r}_fold_{fold}_optuna.db",
                                        # load_if_exists=True
                                        )

            datamodule.setup("fit", fold)

            # Set number of trials according to number of hyperparameters to optimise per model
            n_trials = 2 if args.MODEL == "GIN" else 2 if args.MODEL == "DGCNN" else None
            study.optimize(lambda trial: objective(trial, datamodule, log_dir, args.EPOCHS,
                                                   model_name=args.MODEL), n_trials=n_trials)

            print(f"Best trial for fold {fold}: {study.best_trial.value}")

            # Load the model with the best hyperparameters
            checkpoint_name = f"model_trial_{study.best_trial.number}.ckpt"
            checkpoint_path = os.path.join(log_dir, "checkpoints", checkpoint_name)
            checkpoint = torch.load(checkpoint_path)  # Load the checkpoint dictionary from the file
            init_args = checkpoint["init_args"]  # Access the saved initialization parameters
            best_model = GNNModel(**init_args)  # Initialize the model using the saved parameters
            best_model.load_state_dict(checkpoint['state_dict'])

            # Test the best model
            datamodule.setup("test", fold)
            trainer = create_trainer(log_dir=log_dir, epochs=args.EPOCHS, testing=True)
            test_result = trainer.test(best_model, datamodule=datamodule)
            test_acc = test_result[0]["test_acc"]
            print(f"Test accuracy for fold {fold}: {test_acc}")

            fold_performances.append(test_acc)

        avg_performance = np.mean(fold_performances)
        print(f"Average performance for repetition {r}: {avg_performance}")
        overall_performances.append(avg_performance)

        # Save test accuracies, average performance, and overall average performance after all folds are done
        for fold, test_acc in enumerate(fold_performances):
            if fold == args.N_SPLITS - 1:
                save_test_results(parent_dir, r, fold, test_acc, avg_performance)
            else:
                save_test_results(parent_dir, r, fold, test_acc)

    overall_avg_performance = np.mean(overall_performances)
    print(f"Overall average performance: {overall_avg_performance}")
    save_test_results(parent_dir, None, None, None, None, overall_avg_performance)
