from model import GNNModel
from datamodule import GraphDataModule
import config
from logs import create_log_dir, save_test_results
from training import create_trainer, objective

import os
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

if __name__ == '__main__':

    overall_performances = []

    for r in range(config.STARTING_REP, config.REP):
        seed = r + 1  # Set a new seed for each repetition
        datamodule = GraphDataModule(dataset_name=config.DATASET_NAME, seed=seed)
        datamodule.prepare_data()
        fold_performances = []

        for fold in range(config.STARTING_FOLD if r == config.STARTING_REP else 0, config.N_SPLITS):
            log_dir = create_log_dir(r, fold)
            # Create a new study object for each fold
            study = optuna.create_study(direction="maximize",
                                        pruner=optuna.pruners.MedianPruner(),
                                        sampler=optuna.samplers.TPESampler(seed=config.SEED),
                                        study_name=f"rep_{r}_fold_{fold}",
                                        # storage=f"sqlite:///{log_dir}/rep_{r}_fold_{fold}_optuna.db",
                                        # load_if_exists=True
                                        )

            datamodule.setup("fit", fold)
            study.optimize(lambda trial: objective(trial, datamodule, config.EPOCHS, r, fold), n_trials=TRIALS)
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
            trainer = create_trainer(log_dir=log_dir, epochs=config.EPOCHS, testing=True)
            test_result = trainer.test(best_model, datamodule=datamodule)
            test_acc = test_result[0]["test_acc"]
            print(f"Test accuracy for fold {fold}: {test_acc}")

            fold_performances.append(test_acc)

        avg_performance = np.mean(fold_performances)
        print(f"Average performance for repetition {r}: {avg_performance}")
        overall_performances.append(avg_performance)

        # Save test accuracies, average performance, and overall average performance after all folds are done
        for fold, test_acc in enumerate(fold_performances):
            if fold == config.N_SPLITS - 1:
                save_test_results(config.PARENT_DIR, r, fold, test_acc, avg_performance)
            else:
                save_test_results(config.PARENT_DIR, r, fold, test_acc)

    overall_avg_performance = np.mean(overall_performances)
    print(f"Overall average performance: {overall_avg_performance}")
    save_test_results(config.PARENT_DIR, None, None, None, None, overall_avg_performance)
