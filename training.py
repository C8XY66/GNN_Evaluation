from model import GNNModel

import os
import yaml
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, Callback
from pytorch_lightning.loggers import TensorBoardLogger
from optuna.integration import PyTorchLightningPruningCallback


def load_config(config_file):
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config


class BestValAcc(Callback):
    def __init__(self, logger):
        super().__init__()
        self.logger = logger

    def on_validation_end(self, trainer, pl_module):
        if trainer.checkpoint_callback.best_model_path:
            best_val_acc = trainer.checkpoint_callback.best_model_score.item()
            self.logger.experiment.add_scalar("best_val_acc", best_val_acc, global_step=trainer.global_step)


def create_trainer(log_dir, epochs, patience=None, pruning_callback=None, testing=False, trial=None):
    callbacks = []
    logger = TensorBoardLogger(save_dir=log_dir)
    if not testing:

        # Training Callbacks
        early_stopping = EarlyStopping(monitor="val_loss", mode="min", patience=patience, verbose=True)
        callbacks.append(early_stopping)

        model_checkpoint = ModelCheckpoint(dirpath=os.path.join(log_dir, "checkpoints"),
                                           filename=f"model_trial_{trial.number}",
                                           save_top_k=1,
                                           monitor="val_acc",
                                           mode="max",
                                           auto_insert_metric_name=True,
                                           verbose=True)
        callbacks.append(model_checkpoint)

        best_val_acc_logger = BestValAcc(logger=logger)
        callbacks.append(best_val_acc_logger)

        if pruning_callback is not None:
            callbacks.append(pruning_callback)

    # Create trainer
    trainer = pl.Trainer(
        callbacks=callbacks,
        max_epochs=epochs,
        logger=logger,
        enable_progress_bar=False,
        enable_model_summary=False,
    )

    return trainer


def objective(trial, datamodule, log_dir, epochs, patience, model_name, dataset_type):
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
                     weight_decay=hyperparameters["weight_decay"],
                     dataset_type=dataset_type)

    # Training
    pruning_callback = PyTorchLightningPruningCallback(trial=trial, monitor="val_acc")
    trainer = create_trainer(log_dir=log_dir, epochs=epochs, patience=patience,
                             pruning_callback=pruning_callback, trial=trial)

    hyperparameters = dict(hidden_channels=hyperparameters["hidden_channels"], batch_size=hyperparameters["batch_size"],
                           epochs=epochs, dropout=hyperparameters["dropout"],
                           learning_rate=hyperparameters["learning_rate"])
    trainer.logger.log_hyperparams(hyperparameters)
    trainer.fit(model=model, datamodule=datamodule)

    # Select best trial based on val_loss if val_acc is the same
    return trainer.callback_metrics["val_acc"].item() + 1e-8 * (1 - trainer.callback_metrics["val_loss"].item())




