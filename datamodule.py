
import numpy as np
from typing import Optional
from sklearn.model_selection import StratifiedKFold, train_test_split

import torch
import torch_geometric.transforms as T
from torch_geometric.data import InMemoryDataset
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
import pytorch_lightning as pl


class CustomInMemoryDataset(InMemoryDataset):
    def __init__(self, data_list, transform=None, pre_transform=None):
        super().__init__(transform=transform, pre_transform=pre_transform)
        self.data, self.slices = self.collate(data_list)

    def _download(self):
        pass

    def _process(self):
        pass


class GraphDataModule(pl.LightningDataModule):
    def __init__(self, dataset_name: str, dataset_type: str, experiment: str):
        super().__init__()

        self.dataset_name = dataset_name
        self.dataset_type = dataset_type
        self.experiment = experiment

    def prepare_data(self):
        self.dataset = TUDataset(root="data/TUDataset", name=self.dataset_name,
                                 pre_transform=T.OneHotDegree(500) if self.dataset_type == "social" else None)
        # Node neutralisation
        if self.experiment == "WithoutNF":
            neutralized_data_list = [self.neutralize_node_features(data) for data in self.dataset]
            self.dataset = CustomInMemoryDataset(neutralized_data_list)

    def setup_rep(self, rep: int, folds: int):
        # Set seed
        self.seed_rep = rep + 1
        torch.manual_seed(self.seed_rep)
        np.random.seed(self.seed_rep)

        # Shuffle dataset based on seed
        indices = torch.randperm(len(self.dataset), generator=torch.Generator().manual_seed(self.seed_rep))
        shuffled_data_list = [self.dataset[i] for i in indices]
        self.dataset = CustomInMemoryDataset(shuffled_data_list)

        # Create stratified folds using shuffled dataset
        y = [data.y.item() for data in shuffled_data_list]
        self.skf = StratifiedKFold(n_splits=folds, shuffle=False)
        self.splits = list(self.skf.split(torch.zeros(len(y)), y))

    def setup(self, fold=None, stage: Optional[str] = None):
        if stage is not None:
            return
        # Set seed
        seed_fold = fold + 1
        # Get train and test set for fold
        train_indices, test_indices = self.splits[fold]
        self.test_dataset = [self.dataset[i] for i in test_indices]
        train_dataset = [self.dataset[i] for i in train_indices]

        # Create stratified train/val split from train set
        train_labels = [data.y.item() for data in train_dataset]
        train_indices, val_indices = train_test_split(
            np.arange(len(train_dataset)),
            test_size=0.1,
            random_state=seed_fold,
            stratify=train_labels
        )
        self.train_dataset = [train_dataset[i] for i in train_indices]
        self.val_dataset = [train_dataset[i] for i in val_indices]

    def update_batch_size(self, batch_size):
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

    def calculate_k(self):
        # Compute number of nodes for graphs in dataset, sort them and find index that splits list at 60% percentile
        node_counts = [data.num_nodes for data in self.dataset]
        node_counts.sort()
        index = int(len(node_counts) * 0.6)
        k = node_counts[index]
        return k

    @staticmethod
    def neutralize_node_features(data):
        num_nodes = data.x.size(0)
        data.x = torch.zeros(num_nodes, 1)
        data.x[:, 0] = 1
        return data

    @property
    def num_node_features(self):
        return self.dataset.num_node_features

    @property
    def num_classes(self):
        return self.dataset.num_classes

