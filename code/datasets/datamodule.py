import os
import cv2
from PIL import Image
import numpy as np
from patchify import patchify
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Dataset
import lightning as L
import random

class LandCoverDataset(Dataset):
    """Custom Dataset class to handle land cover image patches and their corresponding labels."""

    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


class DataModule(L.LightningDataModule):
    def __init__(self, dataset_root_folder, dataset_name, batch_size=32, patch_size=256, test_size=0.15, random_state=100):
        super(DataModule, self).__init__()
        self.dataset_root_folder = dataset_root_folder
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.test_size = test_size
        self.random_state = random_state
        self.minmaxscaler = MinMaxScaler()
        self.image_dataset = []
        self.mask_dataset = []
        self.colors = [
            np.array((0, 0, 255)),        # blue for water
            np.array((153, 77, 0)),       # brown for land
            np.array((128, 128, 128)),    # grey for road
            np.array((255, 128, 0)),      # red for building
            np.array((0, 255, 0)),        # green for vegetation
            np.array((0, 0, 0))           # black for unlabeled
        ]
        self.class_rgb = [
            np.array([60, 16, 152]),      # building
            np.array([132, 41, 246]),     # land
            np.array([110, 193, 228]),    # road
            np.array([254, 221, 58]),     # vegetation
            np.array([226, 169, 41]),     # water
            np.array([155, 155, 155])     # unlabeled
        ]

    def prepare_data(self):
        """Load the data from disk and apply the necessary preprocessing."""
        for image_type in ['images', 'masks']:
            image_extension = 'jpg' if image_type == 'images' else 'png'
            for tile_id in range(1, 8):
                for image_id in range(1, 20):
                    image_path = os.path.join(self.dataset_root_folder, self.dataset_name, f'Tile {tile_id}/{image_type}/image_part_00{image_id}.{image_extension}')
                    image = cv2.imread(image_path, 1)
                    if image is not None:
                        if image_type == 'masks':
                            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        size_x = (image.shape[1] // self.patch_size) * self.patch_size
                        size_y = (image.shape[0] // self.patch_size) * self.patch_size
                        image = Image.fromarray(image).crop((0, 0, size_x, size_y))
                        image = np.array(image)
                        patched_images = patchify(image, (self.patch_size, self.patch_size, 3), step=self.patch_size)
                        for i in range(patched_images.shape[0]):
                            for j in range(patched_images.shape[1]):
                                patch = patched_images[i, j, 0]
                                if image_type == 'images':
                                    patch = self.minmaxscaler.fit_transform(patch.reshape(-1, patch.shape[-1])).reshape(patch.shape)
                                    self.image_dataset.append(patch)
                                elif image_type == 'masks':
                                    self.mask_dataset.append(patch)

    def setup(self, stage=None):
        """Prepare the datasets for training, validation, and testing."""
        # Convert images and masks to numpy arrays
        self.image_dataset = np.array(self.image_dataset)
        self.mask_dataset = np.array(self.mask_dataset)

        # Convert RGB masks to label indices
        labels = [self.rgb_to_label(mask) for mask in self.mask_dataset]
        labels = np.array(labels)
        labels = np.expand_dims(labels, axis=3)

        # Convert labels to categorical format using PyTorch
        labels_categorical_dataset = torch.nn.functional.one_hot(torch.tensor(labels, dtype=torch.long), num_classes=len(self.class_rgb))
        labels_categorical_dataset = labels_categorical_dataset.squeeze(3)  # Remove the singleton dimension added by one_hot

        # Split the data into training and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.image_dataset, labels_categorical_dataset.numpy(), test_size=self.test_size, random_state=self.random_state
        )

        # Convert to PyTorch datasets
        self.train_dataset = LandCoverDataset(self.X_train, self.y_train)
        self.val_dataset = LandCoverDataset(self.X_test, self.y_test)

    def rgb_to_label(self, mask):
        """Convert RGB mask to label indices."""
        label_segment = np.zeros(mask.shape[:2], dtype=np.uint8)
        for idx, color in enumerate(self.class_rgb):
            label_segment[np.all(mask == color, axis=-1)] = idx
        return label_segment

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
