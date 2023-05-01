import os
import csv
import datetime
import pytz
import numpy as np
from typing import Optional
from sklearn.model_selection import StratifiedKFold

import torch
import torch.nn.functional as F
from torchmetrics import Accuracy

import torch_geometric.transforms as T
# from torch_geometric.data.lightning import LightningDataset
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GIN, MLP, global_add_pool
from torch_geometric.data import InMemoryDataset

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
# Load the TensorBoard notebook extension
# %load_ext tensorboard

import optuna
from optuna.integration import PyTorchLightningPruningCallback
from optuna.visualization.matplotlib import plot_contour, plot_edf, plot_intermediate_values, plot_optimization_history, \
    plot_parallel_coordinate, plot_param_importances, plot_slice

MAIN_DIR = "/Users/johanna/PycharmProjects/"

import logging

logging.getLogger("pytorch_lightning.utilities.rank_zero").setLevel(logging.WARNING)
logging.getLogger("pytorch_lightning.accelerators.cuda").setLevel(logging.WARNING)
logging.getLogger("pytorch_lightning.callbacks.early_stopping").setLevel(logging.WARNING)
logging.getLogger("pytorch_lightning.callbacks.model_checkpoint").setLevel(logging.WARNING)

import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="pytorch_lightning.callbacks.model_checkpoint")
warnings.filterwarnings("ignore", category=UserWarning, module="pytorch_lightning.utilities")
warnings.filterwarnings("ignore", category=UserWarning, module="pytorch_lightning.trainer")
warnings.filterwarnings("ignore", category=UserWarning, module="optuna.trial")


# Hyperparameters

DATASET_NAME = 'NCI1'
lr = 0.01
EPOCHS = 10  # final = 1000
SEED = 42
TRIALS = 2
N_SPLITS = 2
REP = 1

# Reloading interrupted run
PARENT_DIR = None

# RUN = "NCI1_reps_2_folds_5_epochs_100_2023-04-26_09-00"
# PARENT_DIR = os.path.join(MAIN_DIR, "logs", RUN)

STARTING_REP = 0
STARTING_FOLD = 0


class GNNModel(pl.LightningModule):
    def __init__(self, in_channels: int, out_channels: int,
                 hidden_channels: int, dropout, num_layers=5):
        super().__init__()
        self.save_hyperparameters()
        self.gnn = GIN(in_channels, hidden_channels, num_layers,
                       dropout=dropout, jk='cat')

        self.classifier = MLP([hidden_channels, hidden_channels, out_channels],
                              norm="batch_norm", dropout=dropout)

        self.train_acc = Accuracy(task='multiclass', num_classes=out_channels)
        self.val_acc = Accuracy(task='multiclass', num_classes=out_channels)
        self.test_acc = Accuracy(task='multiclass', num_classes=out_channels)

    def forward(self, x, edge_index, batch):
        x = self.gnn(x, edge_index)
        x = global_add_pool(x, batch)
        x = self.classifier(x)
        return x

    def training_step(self, data, batch_idx):
        y_hat = self(data.x, data.edge_index, data.batch)
        loss = F.cross_entropy(y_hat, data.y)
        self.train_acc(y_hat.softmax(dim=-1), data.y)
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_acc', self.train_acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, data, batch_idx):
        y_hat = self(data.x, data.edge_index, data.batch)
        loss = F.cross_entropy(y_hat, data.y)
        self.val_acc(y_hat.softmax(dim=-1), data.y)
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_acc', self.val_acc, prog_bar=True, on_step=False, on_epoch=True)

    def test_step(self, data, batch_idx):
        y_hat = self(data.x, data.edge_index, data.batch)
        loss = F.cross_entropy(y_hat, data.y)
        self.test_acc(y_hat.softmax(dim=-1), data.y)
        self.log('test_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('test_acc', self.test_acc, prog_bar=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)

    def on_save_checkpoint(self, checkpoint):
        checkpoint["init_args"] = self.hparams


class GraphDataModule(pl.LightningDataModule):
    def __init__(self, dataset_name, n_splits=10, fold=0, seed=None, num_workers=0):
        super().__init__()
        self.dataset_name = dataset_name
        self.n_splits = n_splits
        self.fold = fold
        self.seed = seed
        self.num_workers = num_workers

    def prepare_data(self):
        if self.seed is not None:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)

        self.dataset = TUDataset(root='data/TUDataset', name=self.dataset_name)
        # self.dataset = self.dataset[:1000] #for quick experiments
        self.skf = StratifiedKFold(n_splits=self.n_splits)

    def setup(self, stage: Optional[str] = None, fold: int = 0, batch_size: int = 32):
        self.fold = fold
        self.batch_size = batch_size
        y = [data.y.item() for data in self.dataset]

        train_indices, test_indices = list(self.skf.split(torch.zeros(len(y)), y))[self.fold]
        train_dataset = self.dataset[train_indices]

        num_val = int(len(train_dataset) * 0.1)
        num_train = len(train_dataset) - num_val

        self.train_dataset, self.val_dataset = torch.utils.data.random_split(train_dataset, [num_train, num_val])
        self.test_dataset = self.dataset[test_indices]

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    @property
    def num_node_features(self):
        return self.dataset.num_node_features

    @property
    def num_classes(self):
        return self.dataset.num_classes


def create_log_dir(repetition_index, fold_index):
    global PARENT_DIR

    # Current timestamp
    now = datetime.datetime.now(pytz.timezone('Europe/Zurich')).strftime("%Y-%m-%d_%H-%M")

    # Parent directory
    parent_dir_info = f"{DATASET_NAME}_reps_{REP}_folds_{N_SPLITS}_epochs_{EPOCHS}"

    if PARENT_DIR is None:
        PARENT_DIR = f"{MAIN_DIR}logs/{parent_dir_info}_{now}"
        if not os.path.exists(PARENT_DIR):
            os.makedirs(PARENT_DIR)

    # Subdirectory for the specific repetition and fold
    if repetition_index is not None and fold_index is not None:
        sub_dir = f"{PARENT_DIR}/rep_{repetition_index}_fold_{fold_index}"
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)
    else:
        sub_dir = PARENT_DIR

    return sub_dir


def create_trainer(log_dir, epochs, pruning_callback=None, testing=False, trial=None):
    callbacks = []

    if not testing:

        # Training Callbacks
        early_stopping = EarlyStopping(monitor="val_acc", mode="max", patience=10, verbose=True)
        callbacks.append(early_stopping)

        model_checkpoint = ModelCheckpoint(dirpath=os.path.join(log_dir, "checkpoints"),
                                           filename=f"model_trial_{trial.number}",
                                           save_top_k=1,
                                           monitor="val_acc",
                                           mode="max",
                                           auto_insert_metric_name=True,
                                           verbose=True)
        callbacks.append(model_checkpoint)

        if pruning_callback is not None:
            callbacks.append(pruning_callback)

    # Create trainer
    trainer = pl.Trainer(
        callbacks=callbacks,
        max_epochs=epochs,
        log_every_n_steps=5,
        logger=TensorBoardLogger(save_dir=log_dir),
        enable_progress_bar=False,
        enable_model_summary=False,
    )

    return trainer


def objective(trial, datamodule, epochs, repetition_index, fold_index):
    # Optimise hyperparameters
    hidden_channels = trial.suggest_categorical('hidden_channels', [16, 32])
    batch_size = trial.suggest_categorical('batch_size', [32, 128])
    dropout = trial.suggest_categorical('dropout', [0.0, 0.5])

    # Model and DataModule
    datamodule.setup(fold=0, batch_size=batch_size)
    model = GNNModel(in_channels=datamodule.num_node_features,
                     out_channels=datamodule.num_classes,
                     hidden_channels=hidden_channels,
                     dropout=dropout)

    # Training
    pruning_callback = PyTorchLightningPruningCallback(trial, monitor="val_acc")  # from optuna-pl-integration

    log_dir = create_log_dir(repetition_index, fold_index)
    trainer = create_trainer(log_dir, epochs=epochs,
                             pruning_callback=pruning_callback,
                             trial=trial)

    hyperparameters = dict(hidden_channels=hidden_channels, batch_size=batch_size, epochs=epochs, dropout=dropout)
    trainer.logger.log_hyperparams(hyperparameters)

    trainer.fit(model, datamodule=datamodule)

    return trainer.callback_metrics['val_acc'].item()


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


if __name__ == '__main__':

    overall_performances = []

    for r in range(STARTING_REP, REP):
        seed = r + 1  # Set a new seed for each repetition
        datamodule = GraphDataModule(dataset_name=DATASET_NAME, seed=seed)
        datamodule.prepare_data()
        fold_performances = []

        for fold in range(STARTING_FOLD if r == STARTING_REP else 0, N_SPLITS):
            log_dir = create_log_dir(r, fold)
            # Create a new study object for each fold
            study = optuna.create_study(direction="maximize",
                                        pruner=optuna.pruners.MedianPruner(),
                                        sampler=optuna.samplers.TPESampler(seed=SEED),
                                        study_name=f"rep_{r}_fold_{fold}",
                                        # storage=f"sqlite:///{log_dir}/rep_{r}_fold_{fold}_optuna.db",
                                        # load_if_exists=True
                                        )

            datamodule.setup("fit", fold)
            study.optimize(lambda trial: objective(trial, datamodule, EPOCHS, r, fold), n_trials=TRIALS)
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
            trainer = create_trainer(log_dir=log_dir, epochs=EPOCHS, testing=True)
            test_result = trainer.test(best_model, datamodule=datamodule)
            test_acc = test_result[0]["test_acc"]
            print(f"Test accuracy for fold {fold}: {test_acc}")

            fold_performances.append(test_acc)

        avg_performance = np.mean(fold_performances)
        print(f"Average performance for repetition {r}: {avg_performance}")
        overall_performances.append(avg_performance)

        # Save test accuracies, average performance, and overall average performance after all folds are done
        for fold, test_acc in enumerate(fold_performances):
            if fold == N_SPLITS - 1:
                save_test_results(PARENT_DIR, r, fold, test_acc, avg_performance)
            else:
                save_test_results(PARENT_DIR, r, fold, test_acc)

    overall_avg_performance = np.mean(overall_performances)
    print(f"Overall average performance: {overall_avg_performance}")
    save_test_results(PARENT_DIR, None, None, None, None, overall_avg_performance)

# Commented out IPython magic to ensure Python compatibility.
# %tensorboard --logdir '/content/gdrive/My Drive/ColabNotebooks/logs/NCI1_reps_2_folds_5_epochs_20_2023-04-26_11-05' --port 6007
# %tensorboard --logdir '{MAIN_DIR}'
