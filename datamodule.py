
import numpy as np
from typing import Optional
from sklearn.model_selection import StratifiedKFold

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
    def __init__(self, dataset_name: str, dataset_type: str, experiment: str,
                 n_splits=10, fold=0, seed=None):
        super().__init__()

        self.dataset_name = dataset_name
        self.dataset_type = dataset_type
        self.experiment = experiment
        self.n_splits = n_splits
        self.fold = fold
        self.seed = seed

    def prepare_data(self):
        if self.seed is not None:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)

        self.dataset = TUDataset(root="data/TUDataset", name=self.dataset_name,
                                 pre_transform=T.OneHotDegree(500) if self.dataset_type == "social" else None)

        # Node neutralisation
        if self.experiment == "WithoutNF":
            neutralized_data_list = [self.neutralize_node_features(data) for data in self.dataset]
            self.dataset = CustomInMemoryDataset(neutralized_data_list)

        # Shuffle dataset based on seed
        indices = torch.randperm(len(self.dataset), generator=torch.Generator().manual_seed(self.seed))
        shuffled_data_list = [self.dataset[i] for i in indices]
        self.dataset = CustomInMemoryDataset(shuffled_data_list)

        # Create stratified folds using shuffled dataset
        y = [data.y.item() for data in shuffled_data_list]
        self.skf = StratifiedKFold(n_splits=self.n_splits, shuffle=False)
        self.splits = list(self.skf.split(torch.zeros(len(y)), y))

    def setup(self, fold=None, stage: Optional[str] = None):
        if stage is not None:
            return

        self.fold = fold

        train_indices, test_indices = self.splits[self.fold]
        train_dataset = [self.dataset[i] for i in train_indices]

        num_val = int(len(train_dataset) * 0.1)
        num_train = len(train_dataset) - num_val

        generator = torch.Generator().manual_seed(self.seed)
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(train_dataset, [num_train, num_val],
                                                                             generator=generator)
        self.test_dataset = [self.dataset[i] for i in test_indices]

    def update_batch_size(self, batch_size):
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

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

