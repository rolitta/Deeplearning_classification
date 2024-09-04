import torch
import numpy as np
from PIL import Image

from solarnet.datasets.transforms import SquarePad, MakeDataWorse


class TestSquarePad:
    def test_padding_small_images(self):
        """
        Check that padding works regardless of input image size
        """
        # Create an instance of SquarePad
        square_pad = SquarePad(im_size=224)

        # Create test tensors
        input_tensors = [
            torch.randn(1, 3, 180, 224),
            torch.randn(1, 3, 224, 180),
            torch.randn(1, 3, 180, 180),
            torch.randn(1, 3, 181, 224),
            torch.randn(1, 3, 224, 181),
            torch.randn(1, 3, 181, 180),
            torch.randn(1, 3, 224, 224),
            torch.randn(1, 3, 223, 224),
            torch.randn(1, 3, 223, 223),
            torch.randn(1, 3, 224, 223),
            torch.randn(1, 3, 1, 224),
            torch.randn(1, 3, 1, 1),
            torch.randn(1, 3, 224, 1),
        ]

        for input_tensor in input_tensors:
            # Apply the transform
            output_tensor = square_pad.apply_transform(input_tensor, {}, {})

            # Check if the output tensor has the expected size
            assert output_tensor.size() == (1, 3, 224, 224)

    def test_large_input(self):
        """
        check that images larger than the desired size raise a valueError
        """
        # Create an instance of SquarePad
        square_pad = SquarePad(im_size=224)

        # Create a test tensor
        input_tensors = [
            torch.randn(1, 3, 225, 225),
            torch.randn(1, 3, 225, 224),
            torch.randn(1, 3, 224, 225),
        ]

        for input_tensor in input_tensors:
            # Apply the transform
            try:
                _ = square_pad.apply_transform(input_tensor, {}, {})
            except ValueError:
                pass
            else:
                assert False, "Expected ValueError, but no exception was raised"


class TestMakeDataWorse:

    def test_same_size_no_change(self):
        """
        Check that input and output tensors remain the same if size and output_size are the same

        Note that we are using PIL images as input in this case, and expecting PIL images as output
        """

        for size in [224, 180, 200]:
            # Create an instance of MakeDataWorse
            make_data_worse = MakeDataWorse(size=size, output_size=size)
            # Create test tensors
            input_array = np.random.rand(size, size, 3) * 255
            # Apply the transform
            output_image = make_data_worse._transform(
                Image.fromarray(input_array.astype("uint8")).convert("RGB"), {}
            )
            # Check if the input and output arrays are the same
            assert np.allclose(
                input_array.astype("uint8"), np.array(output_image).astype("uint8")
            )

    def test_downsampling(self):
        """
        check that downsampling and then upsampling work as expected
        """
        # Create an instance of MakeDataWorse
        make_data_worse = MakeDataWorse(size=1, output_size=2)

        # Create test array
        input_array = np.ones((2, 2, 3))
        input_array[0, 0, :] = 0

        # Apply the transform
        output_image = make_data_worse._transform(
            Image.fromarray(input_array.astype("uint8")).convert("RGB"), {}
        )

        # Check if the output image has the expected size
        assert output_image.size == (2, 2)

        # the added zeros should have no effect after downsampling and then upsampling
        assert np.allclose(
            np.ones((2, 2, 3)).astype("uint8"), np.array(output_image).astype("uint8")
        )

    def test_tensor_input(self):
        """
        check that the transform works with tensor inputs
        """
        # Create an instance of MakeDataWorse
        make_data_worse = MakeDataWorse(size=1, output_size=2)

        # Create test tensors
        input_tensor = torch.randn(3, 2, 2)

        # Apply the transform
        output_tensor = make_data_worse._transform(input_tensor, {})

        assert isinstance(output_tensor, torch.Tensor)

        # Check if the output tensor has the expected size
        assert output_tensor.size() == (3, 2, 2)
