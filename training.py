from model import GNNModel

import os
import yaml
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from optuna.integration import PyTorchLightningPruningCallback


def load_config_file(model_name):
    config_file = f"config_{model_name}.yaml"
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config


def create_trainer(log_dir, epochs, patience=None, testing=False, trial=None):
    callbacks = []

    if not testing:
        # Training Callbacks
        early_stopping = EarlyStopping(monitor="val_loss", mode="min", patience=patience, verbose=True)
        trial_pruning = PyTorchLightningPruningCallback(trial=trial, monitor="val_acc")
        model_checkpoint = ModelCheckpoint(dirpath=os.path.join(log_dir, "checkpoints"),
                                           filename=f"model_trial_{trial.number}",
                                           save_top_k=1,
                                           monitor="val_acc",
                                           mode="max",
                                           auto_insert_metric_name=True,
                                           verbose=True)
        callbacks.extend([early_stopping, model_checkpoint, trial_pruning])

    # Create trainer
    trainer = pl.Trainer(
        callbacks=callbacks,
        max_epochs=epochs,
        logger=TensorBoardLogger(save_dir=log_dir),
        enable_progress_bar=False,
        enable_model_summary=False,
    )

    return trainer


def objective(trial, datamodule, log_dir, epochs, patience, model_name, dataset_type):
    config = load_config_file(model_name=model_name)

    # OPTIMISE HYPERPARAMETERS
    # Set default values
    hyperparameters = {"num_layers": 0,
                       "hidden_channels": 0,
                       "batch_size": 0,
                       "learning_rate": 0.0,
                       "dropout": 0.0,
                       "weight_decay": 0.0,
                       "gin_train_eps": False}

    dgcnn_k = datamodule.calculate_k() if model_name == "DGCNN" else 0

    # Check config file for values, and choose hyperparameter combination
    for param, values in config.items():
        hyperparameters[param] = trial.suggest_categorical(param, values)

        # Pass hyperparameters to Model and DataModule
        datamodule.update_batch_size(hyperparameters["batch_size"])
        model = GNNModel(gnn_model_name=model_name,
                         dataset_type=dataset_type,
                         in_channels=datamodule.num_node_features,
                         out_channels=datamodule.num_classes,
                         hidden_channels=hyperparameters["hidden_channels"],
                         num_layers=hyperparameters["num_layers"],
                         learning_rate=hyperparameters["learning_rate"],
                         dropout=hyperparameters["dropout"],
                         weight_decay=hyperparameters["weight_decay"],
                         gin_train_eps=hyperparameters["gin_train_eps"],
                         dgcnn_k=dgcnn_k)
    # TRAINING
    trainer = create_trainer(log_dir=log_dir, epochs=epochs, patience=patience, trial=trial)
    trainer.logger.log_hyperparams(hyperparameters)
    trainer.fit(model=model, datamodule=datamodule)

    # Select best trial based on val_loss if val_acc is the same
    return trainer.callback_metrics["val_acc"].item() + 1e-8 * (1 - trainer.callback_metrics["val_loss"].item())




