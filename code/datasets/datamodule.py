import logging

import torch
from torch.utils.data import random_split
from torchvision.datasets import MNIST
import torch.utils.data.distributed
from torchvision.transforms import v2
import lightning as L

logger = logging.getLogger()


class DataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
    ):
        super().__init__()
        self.data_dir = data_dir

        self.batch_size = batch_size
        self.transform_train = v2.Compose(
            [
                # include any training transforms here
                v2.ToTensor(),
                v2.Normalize((0.1307,), (0.3081,)),
            ]
        )
        self.transform_predict = v2.Compose(
            [
                # include any prediction transforms here
                v2.ToTensor(),
                v2.Normalize((0.1307,), (0.3081,)),
            ]
        )

    def prepare_data(self) -> None:
        # download, create chips, etc.
        pass

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            mnist_full = MNIST(
                self.data_dir, train=True, download=True, transform=self.transform_train
            )

            self.dataset_train, self.dataset_val = random_split(
                mnist_full, [55000, 5000], generator=torch.Generator().manual_seed(42)
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.dataset_test = MNIST(
                self.data_dir,
                train=False,
                download=True,
                transform=self.transform_train,
            )

        if stage == "predict":
            self.dataset_predict = MNIST(
                self.data_dir,
                train=False,
                download=True,
                transform=self.transform_predict,
            )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_train, batch_size=self.batch_size
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_test, batch_size=self.batch_size
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_test, batch_size=self.batch_size
        )

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_predict, batch_size=self.batch_size
        )
