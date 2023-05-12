import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy
from torch_geometric.nn import GIN, MLP, MessagePassing, global_add_pool, SortAggregation
from torch_geometric.utils import add_self_loops, degree
import pytorch_lightning as pl


# GIN Model
class GINModel(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int, num_layers: int, dropout: float):
        super().__init__()

        self.gnn = GIN(in_channels=in_channels, hidden_channels=hidden_channels, num_layers=num_layers,
                       dropout=dropout, jk="cat")

        self.classifier = MLP([hidden_channels, hidden_channels, out_channels],
                              norm="batch_norm", dropout=dropout)

    def forward(self, x, edge_index, batch):
        x = self.gnn(x, edge_index)
        x = global_add_pool(x, batch)
        x = self.classifier(x)

        return x


# DGCNN Model
class DGCNNConv(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(aggr="add")  # "Add" aggregation
        self.lin = nn.Linear(in_features=in_channels, out_features=out_channels)

    def forward(self, x, edge_index):
        # x has shape [N, in_channels], edge_index has shape [2, E]
        # Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Multiply node features by the linear layer.
        x = torch.tanh(self.lin(x))

        # Normalize node features.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Start propagating messages.
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        # x_j has shape [E, hidden_channels], norm has shape [E]
        # Normalize node features.
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        # no need to update node embeddings after message passing, just return the aggregated node embeddings
        return aggr_out


class DGCNNModel(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int, num_layers: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList()
        self.layers.append(DGCNNConv(in_channels=in_channels, out_channels=hidden_channels))
        for _ in range(num_layers - 2):
            self.layers.append(DGCNNConv(in_channels=hidden_channels, out_channels=hidden_channels))
        self.layers.append(DGCNNConv(in_channels=hidden_channels, out_channels=1))

        self.sort_aggr = SortAggregation(k=2910)  # self.k * self.total_latent_dim

        self.conv1D_1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=97, stride=97)
        self.max_pool = nn.MaxPool1d(2, 2)
        self.conv1D_2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=1)

        self.fc1 = nn.Linear(in_features=352, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=out_channels)

    def forward(self, x, edge_index, batch):

        for layer in self.layers:
            x = layer(x, edge_index)

        x = self.sort_aggr(x, batch)
        x = x.unsqueeze(1)  # Add channel dimension

        x = F.relu(self.conv1D_1(x))
        x = self.max_pool(x)
        x = F.relu(self.conv1D_2(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


# MLP Classifiers adapted from MolecularFingerprint and DeepMultisets
# https://github.com/diningphil/gnn-comparison/tree/master/models/graph_classifiers
class MLPModel(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, dataset_type):
        super().__init__()

        self.model_type = dataset_type

        if self.model_type == "chemical":
            self.mlp = torch.nn.Sequential(nn.Linear(in_channels, hidden_channels), nn.ReLU(),
                                           nn.Linear(hidden_channels, out_channels), nn.ReLU())
        elif self.model_type == "social":
            self.fc_vertex = nn.Linear(in_channels, hidden_channels)
            self.fc_global1 = nn.Linear(hidden_channels, hidden_channels)
            self.fc_global2 = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        if self.model_type == "chemical":
            return self.mlp(global_add_pool(x, batch))
        elif self.model_type == "social":
            x = F.relu(self.fc_vertex(x))
            x = global_add_pool(x, batch)  # sums all vertex embeddings belonging to the same graph!
            x = F.relu(self.fc_global1(x))
            x = self.fc_global2(x)
            return x


# General GNN
class GNNModel(pl.LightningModule):
    def __init__(self, gnn_model_name, in_channels: int, out_channels: int, hidden_channels: int, num_layers: int,
                 dropout: float, learning_rate: float, weight_decay: float, dataset_type: str):
        super().__init__()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.gnn_model_name = gnn_model_name
        self.save_hyperparameters()

        if self.gnn_model_name == "GIN":
            self.gnn = GINModel(in_channels=in_channels, out_channels=out_channels, hidden_channels=hidden_channels,
                                num_layers=num_layers, dropout=dropout)
        elif self.gnn_model_name == "DGCNN":
            self.gnn = DGCNNModel(in_channels=in_channels, out_channels=out_channels, hidden_channels=hidden_channels,
                                  num_layers=num_layers, dropout=dropout)
        elif self.gnn_model_name == "MLP":
            self.gnn = MLPModel(in_channels=in_channels, out_channels=out_channels, hidden_channels=hidden_channels,
                                dataset_type=dataset_type)

        self.train_acc = Accuracy(task="multiclass", num_classes=out_channels)
        self.val_acc = Accuracy(task="multiclass", num_classes=out_channels)
        self.test_acc = Accuracy(task="multiclass", num_classes=out_channels)

    def forward(self, x, edge_index, batch):
        x = self.gnn(x, edge_index, batch)
        return x

    def training_step(self, data, batch_idx):
        y_hat = self(data.x, data.edge_index, data.batch)
        train_loss = F.cross_entropy(y_hat, data.y)
        self.train_acc(y_hat.softmax(dim=-1), data.y)
        self.log("train_loss", train_loss, on_step=False, on_epoch=True)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True)

        return train_loss

    def validation_step(self, data, batch_idx):
        y_hat = self(data.x, data.edge_index, data.batch)
        val_loss = F.cross_entropy(y_hat, data.y)
        self.val_acc(y_hat.softmax(dim=-1), data.y)
        self.log("val_loss", val_loss, on_step=False, on_epoch=True)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True)

    def test_step(self, data, batch_idx):
        y_hat = self(data.x, data.edge_index, data.batch)
        test_loss = F.cross_entropy(y_hat, data.y)
        self.test_acc(y_hat.softmax(dim=-1), data.y)
        self.log("test_loss", test_loss)
        self.log("test_acc", self.test_acc)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), self.learning_rate, weight_decay=self.weight_decay)

        if self.gnn_model_name == "GIN":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
            return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
        else:
            return optimizer

    def on_save_checkpoint(self, checkpoint):
        checkpoint["init_args"] = self.hparams
