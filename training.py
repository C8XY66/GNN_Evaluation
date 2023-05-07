from model import GNNModel

import os
import yaml
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from optuna.integration import PyTorchLightningPruningCallback


def load_config(config_file):
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config


def create_trainer(log_dir, epochs, pruning_callback=None, testing=False, trial=None):
    callbacks = []

    if not testing:

        # Training Callbacks
        early_stopping = EarlyStopping(monitor="val_acc", mode="max", patience=200, verbose=True)
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


def objective(trial, datamodule, log_dir, epochs, model_name, dataset_type):
    config_file = f"config_{model_name}.yaml"
    config = load_config(config_file=config_file)

    # Optimise hyperparameters
    hyperparameters = {}

    for param, values in config.items():
        if isinstance(values, list):
            hyperparameters[param] = trial.suggest_categorical(param, values)
        else:
            hyperparameters[param] = values

    # Model and DataModule
    datamodule.update_batch_size(hyperparameters["batch_size"])
    model = GNNModel(gnn_model_name=model_name,
                     in_channels=datamodule.num_node_features,
                     out_channels=datamodule.num_classes,
                     hidden_channels=hyperparameters["hidden_channels"],
                     num_layers=hyperparameters["num_layers"],
                     dropout=hyperparameters["dropout"],
                     learning_rate=hyperparameters["learning_rate"],
                     dataset_type=dataset_type)

    # Training
    pruning_callback = PyTorchLightningPruningCallback(trial=trial, monitor="val_acc")  # from optuna-pl-integration
    trainer = create_trainer(log_dir=log_dir, epochs=epochs,
                             pruning_callback=pruning_callback,
                             trial=trial)

    hyperparameters = dict(hidden_channels=hyperparameters["hidden_channels"], batch_size=hyperparameters["batch_size"],
                           epochs=epochs, dropout=hyperparameters["dropout"],
                           learning_rate=hyperparameters["learning_rate"])
    trainer.logger.log_hyperparams(hyperparameters)
    trainer.fit(model=model, datamodule=datamodule)

    # Print training and validation accuracies and losses
    train_acc = trainer.callback_metrics['train_acc']
    train_loss = trainer.callback_metrics['train_loss']
    val_acc = trainer.callback_metrics['val_acc']
    val_loss = trainer.callback_metrics['val_loss']
    print(
        f"Trial: {trial.number}, Train Accuracy: {train_acc:.4f}, Train Loss: {train_loss:.4f}, Val Accuracy: "
        f"{val_acc:.4f}, Val Loss: {val_loss:.4f}\n")

    return trainer.callback_metrics["val_acc"].item()

