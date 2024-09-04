import logging
import argparse

import torch

from models.model import DLModel

logger = logging.getLogger()


def parse_args():
    """
    helper function to parse command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
    )
    parser.add_argument("--image-folder", type=str, required=True)
    parser.add_argument(
        "--result-folder",
        type=str,
        required=True,
    )
    return parser.parse_args()


def make_inferences(
    model_path: str,
    image_folder: str,
    result_folder: str,
    device: torch.device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    ),
):
    """
    Takes the output numpy files stored after training and plots them
    """

    # load the segmenter model
    model = DLModel.load_from_checkpoint(model_path)

    # inference logic goes here
