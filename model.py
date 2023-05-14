import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy
from torch_geometric.nn import GINConv, MessagePassing, global_add_pool, SortAggregation
from torch_geometric.utils import add_self_loops, degree
import pytorch_lightning as pl


# MLP used in GIN Model
class GINMLPModel(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features=in_channels, out_features=out_channels),
            nn.BatchNorm1d(num_features=out_channels), nn.ReLU(),
            nn.Linear(in_features=out_channels, out_features=out_channels),
            nn.BatchNorm1d(num_features=out_channels), nn.ReLU(),
            nn.Linear(in_features=out_channels, out_features=out_channels)
        )

    def forward(self, x):
        x = self.mlp(x)
        return x


# GIN Model
class GINModel(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int, num_layers: int, dropout: float,
                 train_eps: bool):
        super(GINModel, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList()

        # GRAPH CONVOLUTION
        # Input layer: in_channels as MLP input
        self.layers.append(GINConv(nn=GINMLPModel(in_channels=in_channels, out_channels=hidden_channels),
                                   train_eps=train_eps))
        self.layers.append(nn.BatchNorm1d(num_features=hidden_channels))

        # Hidden layers: hidden_channels as MLP input
        for _ in range(num_layers - 2):
            self.layers.append(GINConv(nn=GINMLPModel(in_channels=hidden_channels, out_channels=hidden_channels),
                                       train_eps=train_eps))
            self.layers.append(nn.BatchNorm1d(num_features=hidden_channels))

        # Last layer: no BatchNorm
        self.layers.append(GINConv(nn=GINMLPModel(in_channels=hidden_channels, out_channels=hidden_channels),
                                   train_eps=train_eps))

        # DENSE LAYER
        self.fc1 = nn.Linear(in_features=num_layers * hidden_channels, out_features=hidden_channels)
        self.fc2 = nn.Linear(in_features=hidden_channels, out_features=out_channels)


def forward(self, x, edge_index, batch):
    summed_layer_outputs = []
    for i in range(0, len(self.layers), 2):  # 2 because GINConv and BatchNorm come in pairs
        x = self.layers[i](x, edge_index)
        if i + 1 < len(self.layers):  # Check if last layer
            x = F.relu(self.layers[i + 1](x))  # BatchNorm layer
        summed_layer_outputs.append(global_add_pool(x, batch))

    x = torch.cat(summed_layer_outputs, dim=-1)  # concatenated sums

    x = F.relu(self.fc1(x))
    x = self.dropout(x)
    x = self.fc2(x)
    return x


# DGCNN Model
class DGCNNConv(MessagePassing):
    """
    Class Copied from https://github.com/diningphil/gnn-comparison/blob/master/models/graph_classifiers/DGCNN.py
    """
    def __init__(self, in_channels, out_channels):
        super(DGCNNConv, self).__init__(aggr='add')  # "Add" aggregation.
        self.lin = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # x has shape [N, in_channels], edge_index has shape [2, E]
        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)
        # Step 3-5: Start propagating messages.
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j, edge_index, size):
        # x_j has shape [E, out_channels]
        # Step 3: Normalize node features.
        src, dst = edge_index  # we assume source_to_target message passing
        deg = degree(src, size[0], dtype=x_j.dtype)
        deg = deg.pow(-1)
        norm = deg[dst]
        return norm.view(-1, 1) * x_j  # broadcasting the normalization term to all out_channels === hidden features

    def update(self, aggr_out):
        # aggr_out has shape [N, out_channels]
        # Step 5: Return new node embeddings.
        return aggr_out


class DGCNNModel(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int, num_layers: int, dropout: float,
                 k: int):
        super().__init__()
        self.k = k
        self.dropout = nn.Dropout(dropout)
        self.graph_conv_layers = nn.ModuleList()

        # GRAPH CONVOLUTION
        # Input layer: in_channels as input
        self.graph_conv_layers.append(DGCNNConv(in_channels=in_channels, out_channels=hidden_channels))

        # Hidden layers: hidden_channels as input
        for _ in range(num_layers - 2):
            self.graph_conv_layers.append(DGCNNConv(in_channels=hidden_channels, out_channels=hidden_channels))

        # Last layer: output = 1
        self.graph_conv_layers.append(DGCNNConv(in_channels=hidden_channels, out_channels=1))

        # SORT POOL
        self.sort_pool = SortAggregation(k=self.k)  # Calculated in datamodule.calculate_k()

        # 1-D CONVOLUTION
        self.total_latent_dim = (num_layers-1)*hidden_channels + 1  # Example 4 layers [32,32,32,1] -> 97

        self.conv1D_1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=self.total_latent_dim,
                                  stride=self.total_latent_dim)
        self.max_pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv1D_2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=1)

        dense_input_dim = (int(self.k / 2) - 4) * 32  # See details of calculation in thesis

        # DENSE LAYER
        self.fc1 = nn.Linear(in_features=dense_input_dim, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=out_channels)

    def forward(self, x, edge_index, batch):
        # Graph convolution, in x:[N, in_channels], edge_index:[2, E]
        for layer in self.graph_conv_layers:
            x = torch.tanh(layer(x, edge_index))

        # SortPool, in x:[N_total, c_last=1]
        x = self.sort_pool(x, batch)  # out x:[batch_size, self.k]

        # Padding, out x:[batch_size, self.k * self.total_latent_dim]
        padding_size = self.k * self.total_latent_dim - x.size(1)
        if padding_size > 0:
            padding = torch.zeros((x.size(0), padding_size), device=x.device)
            x = torch.cat((x, padding), dim=1)

        x = torch.unsqueeze(x, dim=1)  # Add a channels dimension

        # 1-D convolution, in x:[batch_size, channels=1, padded_k]
        x = F.relu(self.conv1D_1(x))
        x = self.max_pool(x)
        x = F.relu(self.conv1D_2(x))  # out x:[batch_size, channels_c_last=32, length_c_last=5]

        x = x.view(x.size(0), -1)  # Flatten

        # Dense layer, x:[batch_size, (channels_c_last * length_c_last)=160]
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)  # x:[batch_size, num_classes]
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
    def __init__(self, gnn_model_name, dataset_type: str, in_channels: int, out_channels: int, hidden_channels: int,
                 num_layers: int, learning_rate: float, dropout: float,  weight_decay: float,  gin_train_eps: bool,
                 dgcnn_k: int):
        super().__init__()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.gnn_model_name = gnn_model_name
        self.save_hyperparameters()

        if self.gnn_model_name == "GIN":
            self.gnn = GINModel(in_channels=in_channels, out_channels=out_channels, hidden_channels=hidden_channels,
                                num_layers=num_layers, dropout=dropout, train_eps=gin_train_eps)
        elif self.gnn_model_name == "DGCNN":
            self.gnn = DGCNNModel(in_channels=in_channels, out_channels=out_channels, hidden_channels=hidden_channels,
                                  num_layers=num_layers, dropout=dropout, k=dgcnn_k)
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
