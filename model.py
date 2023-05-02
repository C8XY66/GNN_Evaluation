
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy
from torch_geometric.nn import GIN, MLP, global_add_pool
import pytorch_lightning as pl


class GINModel(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int, num_layers: int, dropout: float):

        super().__init__()

        self.gnn = GIN(in_channels=in_channels, hidden_channels=hidden_channels, num_layers=num_layers, dropout=dropout, jk='cat')

        self.classifier = MLP([hidden_channels, hidden_channels, out_channels],
                              norm="batch_norm", dropout=dropout)

    def forward(self, x, edge_index, batch):
        x = self.gnn(x, edge_index)
        x = global_add_pool(x, batch)
        x = self.classifier(x)

        return x


class GNNModel(pl.LightningModule):
    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int, dropout: float, learning_rate=0.01):

        super().__init__()
        self.learning_rate = learning_rate
        self.save_hyperparameters()

        self.gnn = GINModel(in_channels=in_channels, out_channels=out_channels, hidden_channels=hidden_channels, num_layers=5, dropout=dropout)

        self.train_acc = Accuracy(task='multiclass', num_classes=out_channels)
        self.val_acc = Accuracy(task='multiclass', num_classes=out_channels)
        self.test_acc = Accuracy(task='multiclass', num_classes=out_channels)

    def forward(self, x, edge_index, batch):
        x = self.gnn(x, edge_index, batch)
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
        return torch.optim.Adam(self.parameters(), self.learning_rate)

    def on_save_checkpoint(self, checkpoint):
        checkpoint["init_args"] = self.hparams

