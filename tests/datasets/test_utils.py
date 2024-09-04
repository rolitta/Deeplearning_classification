import torch
import numpy as np

from solarnet.datasets.utils import collate_fn


class TestCollateFn:

    def test_collate_fn(self):
        """
        Ensure that the collate function correctly stacks samples
        """
        samples = [
            {
                "image": torch.ones(3, 256, 256),
                "masks": torch.ones(1, 1, 256, 256),
                "labels": [1],
            },
            {
                "image": torch.ones(3, 256, 256) * 2,
                "masks": torch.zeros(2, 1, 256, 256),
                "labels": [0, 0],
            },
            {
                "image": torch.ones(3, 256, 256) * 3,
                "masks": torch.ones(2, 1, 256, 256),
                "labels": [1, 1],
            },
        ]

        collated = collate_fn(samples)

        assert collated["image"].shape == torch.Size([3, 3, 256, 256])
        assert collated["image"].sum() == torch.Tensor([1179648.0])

        assert len(collated["target"]) == 3

        for ind in range(len(samples)):
            assert np.all(
                (collated["target"][ind]["masks"] == samples[ind]["masks"]).numpy()
            )

            assert collated["target"][ind]["labels"] == samples[ind]["labels"]
