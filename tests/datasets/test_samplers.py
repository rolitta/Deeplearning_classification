import geopandas as gpd
from torch.utils.data import DataLoader

from solarnet.datasets.samplers import (
    ComboBatchSampler,
    NegativeRandomGeoSampler,
    PositiveChipSampler,
)
from solarnet.datasets import ImageDataset, collate_fn


class TestComboSampler:
    def test_iter_exits_properly(self):
        # Create positive and negative samplers
        image_dataset = ImageDataset(paths=["tests/test_data"])
        positive_sampler = PositiveChipSampler(
            dataset=image_dataset, label_file="tests/test_data/labels.gpkg", size=128
        )
        negative_sampler = NegativeRandomGeoSampler(
            dataset=image_dataset,
            label_file="tests/test_data/labels.gpkg",
            size=128,
            length=2,
        )

        # double check that the positive and negative sampler lengths match expectations
        assert len(positive_sampler) == 3
        assert len(negative_sampler) == 2

        for batch_size in [1, 2, 3]:
            # Create an instance of ComboSampler
            combo_sampler = ComboBatchSampler(
                samplers=[positive_sampler, negative_sampler], batch_size=batch_size
            )

            # check sampler lengths
            assert combo_sampler.length == (
                len(positive_sampler) + len(negative_sampler)
            )
            assert (
                len(combo_sampler)
                == (len(positive_sampler) + len(negative_sampler)) // batch_size
            )

            batches = []
            for batch in combo_sampler:

                # check that the batch size is correct
                assert len(batch) == batch_size

                batches.append(batch)

            # check that the number of batches is correct
            assert len(batches) == len(combo_sampler)

    def test_multiple_epochs(self):
        """
        Ensure that the dataloader can be iterated over multiple times
        """
        num_epochs = 3

        # Create positive and negative samplers
        image_dataset = ImageDataset(paths=["tests/test_data"])
        positive_sampler = PositiveChipSampler(
            dataset=image_dataset, label_file="tests/test_data/labels.gpkg", size=64
        )
        negative_sampler = NegativeRandomGeoSampler(
            dataset=image_dataset,
            label_file="tests/test_data/labels.gpkg",
            size=64,
            length=2,
        )

        # double check that the positive and negative sampler lengths match expectations
        assert len(positive_sampler) == 3
        assert len(negative_sampler) == 2

        for batch_size in [1, 2, 3]:
            # Create an instance of ComboSampler
            combo_sampler = ComboBatchSampler(
                samplers=[positive_sampler, negative_sampler], batch_size=batch_size
            )

            # check sampler lengths
            assert combo_sampler.length == (
                len(positive_sampler) + len(negative_sampler)
            )
            assert (
                len(combo_sampler)
                == (len(positive_sampler) + len(negative_sampler)) // batch_size
            )

            # try to iterate over the same sampler multiple times
            for _ in range(num_epochs):

                batches = []
                for batch in combo_sampler:

                    # check that the batch size is correct
                    assert len(batch) == batch_size

                    batches.append(batch)

                # check that the number of batches is correct
                assert len(batches) == len(combo_sampler)


class TestPositiveSampler:
    def test_sampler_length(self):
        """
        Check that the sampler returns the correct number of samples
        """
        image_dataset = ImageDataset(paths=["tests/test_data"])
        positive_sampler = PositiveChipSampler(
            dataset=image_dataset, label_file="tests/test_data/labels.gpkg", size=64
        )

        labels = gpd.read_file("tests/test_data/labels.gpkg")

        assert len(positive_sampler) == len(labels)

    def test_only_positive(self):
        """
        Check that the sampler only returns positive samples
        """
        image_dataset = ImageDataset(paths=["tests/test_data"])
        positive_sampler = PositiveChipSampler(
            dataset=image_dataset, label_file="tests/test_data/labels.gpkg", size=64
        )

        labels = gpd.read_file("tests/test_data/labels.gpkg")

        data_loader = DataLoader(
            image_dataset, sampler=positive_sampler, collate_fn=collate_fn
        )

        assert len(positive_sampler) == 3
        assert len(data_loader) == len(positive_sampler)

        for sample in data_loader:
            assert sample["image"].shape == (1, 3, 64, 64)

            for target in sample["target"]:
                # check that the target bounding box contains at least one label
                assert (
                    len(
                        labels.cx[
                            target["bbox"].minx : target["bbox"].maxx,
                            target["bbox"].miny : target["bbox"].maxy,
                        ]
                    )
                    > 0
                )


class TestNegativeSampler:
    def test_sampler_length(self):
        """
        Check that the sampler returns the correct number of samples
        """
        image_dataset = ImageDataset(paths=["tests/test_data"])
        negative_sampler = NegativeRandomGeoSampler(
            dataset=image_dataset,
            label_file="tests/test_data/labels.gpkg",
            size=64,
            length=2,
        )

        assert len(negative_sampler) == 2

    def test_only_negative(self):
        """
        Check that the sampler only returns negative samples
        """
        image_dataset = ImageDataset(paths=["tests/test_data"])
        negative_sampler = NegativeRandomGeoSampler(
            dataset=image_dataset,
            label_file="tests/test_data/labels.gpkg",
            size=64,
            length=5,
        )

        labels = gpd.read_file("tests/test_data/labels.gpkg")

        data_loader = DataLoader(
            image_dataset, sampler=negative_sampler, collate_fn=collate_fn
        )

        assert len(negative_sampler) == 5
        assert len(data_loader) == len(negative_sampler)

        for sample in data_loader:
            assert sample["image"].shape == (1, 3, 64, 64)

            for target in sample["target"]:
                # check that the target bounding box contains no labels
                assert (
                    len(
                        labels.cx[
                            target["bbox"].minx : target["bbox"].maxx,
                            target["bbox"].miny : target["bbox"].maxy,
                        ]
                    )
                    == 0
                )
