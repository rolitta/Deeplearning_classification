import os
import cv2
from PIL import Image
import numpy as np
from patchify import patchify
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import random_split
import lightning as L

class LandCoverDataset(Dataset):
    """
    Custom Dataset class to handle land cover image patches and their corresponding
    labels.
    """

    def __init__(
        self, data_dir: str, dataset_name: str, patch_size: int, transform=None
    ):
        self.transform = transform
        self.patch_size = patch_size

        self.minmaxscaler = MinMaxScaler()
        self.class_rgb = [
            np.array([60, 16, 152]),  # building
            np.array([132, 41, 246]),  # land
            np.array([110, 193, 228]),  # road
            np.array([254, 221, 58]),  # vegetation
            np.array([226, 169, 41]),  # water
            np.array([155, 155, 155]),  # unlabeled
        ]

        # read all image and mask files, and store them in lists
        self.masks = []
        self.images = []
        for tile_id in range(1, 8):
            for image_id in range(1, 20):
                image_path = os.path.join(
                    data_dir,
                    dataset_name,
                    f"Tile {tile_id}/images/image_part_00{image_id}.jpg",
                )
                mask_path = os.path.join(
                    data_dir,
                    dataset_name,
                    f"Tile {tile_id}/masks/image_part_00{image_id}.png",
                )
                images = self.read_images(image_path, is_mask=False)
                masks = self.read_images(mask_path, is_mask=True)
                if images:
                    if not masks:
                        raise ValueError(f"Mask not found for image {image_path}")
                    if len(masks) != len(images):
                        raise ValueError(
                            f"Number of masks and images do not match for image {image_path}"
                        )
                    self.images.extend(images)
                    self.masks.extend(masks)

    def read_images(self, image_path: str, is_mask: bool = False) -> list | None:
        """
        Method for reading a single image or mask file from disk and patchifying it.
        """
        image = cv2.imread(image_path, 1)
        if image is not None:
            if is_mask:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            size_x = (image.shape[1] // self.patch_size) * self.patch_size
            size_y = (image.shape[0] // self.patch_size) * self.patch_size
            image = Image.fromarray(image).crop((0, 0, size_x, size_y))
            image = np.array(image)
            patched_images = patchify(
                image,
                (self.patch_size, self.patch_size, 3),
                step=self.patch_size,
            )
            return np.reshape(patched_images, newshape=(-1)).tolist()

    def __len__(self):
        return len(self.images)

    def rgb_to_label(self, mask: np.ndarray) -> torch.Tensor:
        """Convert RGB mask to label indices."""
        label_segment = np.zeros(mask.shape[:2], dtype=np.uint8)
        for idx, color in enumerate(self.class_rgb):
            label_segment[np.all(mask == color, axis=-1)] = idx
        labels = np.array(label_segment)
        labels = np.expand_dims(labels, axis=3)

        # Convert labels to categorical format using PyTorch
        labels_categorical_dataset = torch.nn.functional.one_hot(
            torch.tensor(labels, dtype=torch.long), num_classes=len(self.class_rgb)
        )
        labels_categorical_dataset = labels_categorical_dataset.squeeze(
            3
        )  # Remove the singleton dimension added by one_hot

        return labels_categorical_dataset

    def __getitem__(self, idx):
        """
        Return a single image and mask
        """
        image = self.images[idx]
        mask = self.masks[idx]

        image = self.minmaxscaler.fit_transform(
            image.reshape(-1, image.shape[-1])
        ).reshape(image.shape)

        mask = self.rgb_to_label(mask)

        if self.transform:
            image = self.transform(image)
        return image, mask


class DataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset_root_folder,
        dataset_name,
        batch_size=32,
        patch_size=256,
        test_size=0.15,
    ):
        super(DataModule, self).__init__()
        self.dataset_root_folder = dataset_root_folder
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.test_size = test_size

    def setup(self, stage=None):
        """Prepare the datasets for training, validation, and testing."""
        dataset = LandCoverDataset(
            self.dataset_root_folder, self.dataset_name, self.patch_size
        )

        # Split the data into training and test sets
        self.train_dataset, self.val_dataset = random_split(
            dataset, [self.test_size, 1 - self.test_size]
        )

    def train_dataloader(self):
        """Return the training data loader."""
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        """Return the validation data loader."""
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        """Return the test data loader."""
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def predict_dataloader(self):
        """Return the prediction data loader."""
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
