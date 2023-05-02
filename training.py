from model import GNNModel

import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from optuna.integration import PyTorchLightningPruningCallback


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


def objective(trial, datamodule, log_dir, epochs, model_name):
    # Optimise hyperparameters
    hidden_channels = trial.suggest_categorical('hidden_channels', [16, 32])
    batch_size = trial.suggest_categorical('batch_size', [32, 128])
    dropout = trial.suggest_categorical('dropout', [0.0, 0.5])

    # Model and DataModule
    datamodule.setup(fold=0, batch_size=batch_size)
    model = GNNModel(gnn_model_name=model_name,
                     in_channels=datamodule.num_node_features,
                     out_channels=datamodule.num_classes,
                     hidden_channels=hidden_channels,
                     dropout=dropout)

    # Training
    pruning_callback = PyTorchLightningPruningCallback(trial, monitor="val_acc")  # from optuna-pl-integration

    #log_dir = create_log_dir(repetition_index, fold_index)
    trainer = create_trainer(log_dir, epochs=epochs,
                             pruning_callback=pruning_callback,
                             trial=trial)

    hyperparameters = dict(hidden_channels=hidden_channels, batch_size=batch_size, epochs=epochs, dropout=dropout)
    trainer.logger.log_hyperparams(hyperparameters)

    trainer.fit(model, datamodule=datamodule)

    return trainer.callback_metrics['val_acc'].item()

