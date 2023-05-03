
import numpy as np
from typing import Optional
from sklearn.model_selection import StratifiedKFold

import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
import pytorch_lightning as pl


class GraphDataModule(pl.LightningDataModule):
    def __init__(self, dataset_name: str, dataset_type: str, experiment: str,
                 n_splits=10, fold=0, seed=None, num_workers=0):
        super().__init__()
        assert dataset_type in ['social', 'chemical'], "Invalid model_type. Must be either 'social' or 'chemical'."
        assert experiment in ['with_node_features', 'without_node_features'], \
            "Invalid experiment. Must be either 'with_node_features' or 'without_node_features'."

        self.dataset, self.batch_size, self.skf, self.splits = None
        self.train_dataset, self.val_dataset, self.test_dataset = None
        self.dataset_name = dataset_name
        self.dataset_type = dataset_type
        self.experiment = experiment
        self.n_splits = n_splits
        self.fold = fold
        self.seed = seed
        self.num_workers = num_workers

    def prepare_data(self):
        if self.seed is not None:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)

        self.dataset = TUDataset(root='data/TUDataset', name=self.dataset_name,
                                 pre_transform=T.OneHotDegree(135) if self.dataset_type == "social" else None)

        self.skf = StratifiedKFold(n_splits=self.n_splits)

    def setup(self, stage: Optional[str] = None, fold: int = 0, batch_size: int = 32):
        self.fold = fold
        self.batch_size = batch_size
        y = [data.y.item() for data in self.dataset]

        train_indices, test_indices = list(self.skf.split(torch.zeros(len(y)), y))[self.fold]
        train_dataset = self.dataset[train_indices]

        num_val = int(len(train_dataset) * 0.1)
        num_train = len(train_dataset) - num_val

        self.train_dataset, self.val_dataset = torch.utils.data.random_split(train_dataset, [num_train, num_val])
        self.test_dataset = self.dataset[test_indices]

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    @property
    def num_node_features(self):
        return self.dataset.num_node_features

    @property
    def num_classes(self):
        return self.dataset.num_classes

