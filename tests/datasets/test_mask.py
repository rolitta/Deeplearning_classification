from torchgeo.datasets.utils import BoundingBox
from rasterio.crs import CRS
import numpy as np

from solarnet.datasets.datasets import LabelDataset


class TestLabelDataset:

    def test_init(self):
        """
        Ensure that the LabelDataset class is initialized with
        the expected index
        """
        label_dataset = LabelDataset(
            paths=["tests/test_data"],
        )

        for hit in label_dataset.index.intersection(
            (-78.703, 41.071, -78.693, 41.085, 0, 9.2e18), objects=True
        ):
            assert hit.object == "tests/test_data/labels.gpkg"

    def test_getitem_with_label(self):
        """
        Ensure the correct mask and metadata are generated from different queries
        """

        res = 0.3
        label_dataset = LabelDataset(
            paths=["tests/test_data"],
            res=res,
            crs=CRS.from_epsg("26911"),
        )

        center = (251547, 4071559)

        query = BoundingBox(
            minx=center[0] - (64 * res),
            maxx=center[0] + (64 * res),
            miny=center[1] - (64 * res),
            maxy=center[1] + (64 * res),
            mint=0,
            maxt=9.2e18,
        )

        target = label_dataset[query]

        assert target["masks"].shape == (1, 128, 128)

        # the number of labelled pixels in the mask should be 7214
        assert np.sum(target["masks"].cpu().numpy()) == 7214

        # the number of boxes should be 1
        assert len(target["boxes"]) == 1

        assert np.all(target["boxes"].cpu().numpy()[0] == [26.0, 13.0, 91.0, 96.0])

        # the number of labels should be 1, and it should equal 1
        assert len(target["labels"]) == 1
        for label in target["labels"].cpu().numpy():
            assert label == 1

        # the area should be 8736.0
        assert target["area"].cpu().numpy()[0] == 8736.0

    def test_getitem_no_labels(self):
        """ "
        check a query that should return no labels
        """

        res = 0.3
        label_dataset = LabelDataset(
            paths=["tests/test_data"],
            res=res,
            crs=CRS.from_epsg("26911"),
        )

        center = (
            label_dataset.index.bounds[0],
            label_dataset.index.bounds[2],
        )

        query = BoundingBox(
            minx=center[0] - (32 * res),
            maxx=center[0] + (32 * res),
            miny=center[1] - (32 * res),
            maxy=center[1] + (32 * res),
            mint=0,
            maxt=9.2e18,
        )

        target = label_dataset[query]

        assert target["masks"].shape == (1, 64, 64)
        assert len(target["labels"]) == 1
        assert target["labels"].cpu().numpy()[0] == 0

        # the number of labelled pixels in the mask should be 0
        assert np.sum(target["masks"].cpu().numpy()) == 0
