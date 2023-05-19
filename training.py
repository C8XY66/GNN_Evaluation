from model import GNNModel

import os
import yaml
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from optuna.integration import PyTorchLightningPruningCallback


def load_config_file(model_name):
    """
    Load configuration file for a given model.
    Args:
        model_name (str): Name of the model for which to load the configuration.
    Returns:
        dict: Configuration parameters.
    """
    config_file = f"config_{model_name}.yaml"
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config


def create_trainer(log_dir, epochs, patience=None, trial=None, testing=False, ):
    """
    Creates a PyTorch Lightning trainer with callbacks for early stopping, trial pruning and model checkpointing.
    Args:
        log_dir (str): Path to the log directory.
        epochs (int): Maximum number of training epochs.
        patience (int, optional): Epochs to wait for improvement before training is stopped early. Default is None.
        trial (optuna.trial.Trial, optional): Optuna trial for trial pruning during training. Default is None.
        testing (bool, optional): Flag indicating if the trainer is used for testing (vs. training). Default is False.
    Returns:
        pl.Trainer: PyTorch Lightning trainer.
    """
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
    """
    Objective function for hyperparameter optimization.
    Args:
        trial (optuna.trial.Trial): Optuna trial object.
        datamodule (PyTorchLightning.LightningDataModule): DataModule for handling graph data.
        log_dir (str): Path to the log directory.
        epochs (int): Maximum number of training epochs.
        patience (int): Number of epochs with no improvement after which training will be stopped.
        model_name (str): Name of the model for which to optimize hyperparameters.
        dataset_type (str): Type of the dataset that is used (e.g. "chemical", "social").
    Returns:
        float: Validation accuracy, with a small amount added to break ties based on validation loss.
    """
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
                     dropout=hyperparameters["dropout"],
                     learning_rate=hyperparameters["learning_rate"],
                     weight_decay=hyperparameters["weight_decay"],
                     gin_train_eps=hyperparameters["gin_train_eps"],
                     dgcnn_k=dgcnn_k)
    # TRAINING
    trainer = create_trainer(log_dir=log_dir, epochs=epochs, patience=patience, trial=trial)
    trainer.logger.log_hyperparams(hyperparameters)
    trainer.fit(model=model, datamodule=datamodule)

    # Select best trial based on val_loss if val_acc is the same
    return trainer.callback_metrics["val_acc"].item() + 1e-8 * (1 - trainer.callback_metrics["val_loss"].item())




