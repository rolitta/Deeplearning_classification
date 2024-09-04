"""
Entrypoint for training or inference
"""

import argparse
import logging
import os
import json

from lightning.pytorch import seed_everything

from train.train import train_model

logger = logging.getLogger()


def parse_args():
    parser = argparse.ArgumentParser(
        prog="template", description="Template training and inference utilities"
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO", help="Logging level (DEBUG, INFO, etc)"
    )

    # training arguments
    #########################################################
    parser.add_argument(
        "--max-epochs", type=int, default=26, help="Number of epochs to train for"
    )
    parser.add_argument(
        "--batch-size", type=int, default=4, help="Batch size for training"
    )
    parser.add_argument(
        "--data-folder",
        required=False,
        type=str,
        help="Path to the training chips",
    )
    parser.add_argument(
        "--model-folder",
        type=str,
        required=False,
        help="Path to save the trained model",
    )
    parser.add_argument(
        "--tensorboard-dir",
        type=str,
        required=True,
        help="Path to save tensorboard logs",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        required=False,
        help="Path to save model checkpoints",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="gpu",
        help="Device to train the model on (cpu, gpu, tpu, hpu, mps, auto)",
    )
    parser.add_argument(
        "--distributed",
        type=bool,
        default=False,
        help="Use distributed training",
    )
    parser.add_argument(
        "--use-deterministic-algorithms",
        type=bool,
        default=False,
        help="Use deterministic algorithms for reproducibility",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a checkpoint to resume training from",
    )
    parser.add_argument(
        "--amp",
        type=bool,
        default=True,
        help="Use automatic mixed precision training",
    )

    # debugging args
    parser.add_argument("--fast-dev-run", type=bool, default=False)

    return parser.parse_args()


def setup_for_sagemaker(args):
    """
    Set up for running in sagemaker
    """
    args.model_folder = os.getenv("SM_MODEL_DIR", "/opt/ml/model")
    work_dir = os.getenv("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data")
    args.data_folder = os.path.join(
        os.getenv("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"),
    )

    os.makedirs(args.data_folder, exist_ok=True)
    os.makedirs(args.model_folder, exist_ok=True)
    os.makedirs(args.tensorboard_dir, exist_ok=True)

    logger.info(
        "model_folder: %s\nwork_dir: %s\ndata_folder: %s",
        args.model_folder,
        work_dir,
        args.data_folder,
    )

    logger.debug(
        "Contents of %s:\n %s",
        args.data_folder,
        json.dumps(os.listdir(args.data_folder), indent=2),
    )

    # change the python working directory to the work directory so that run files
    # get saved to S3 as part of the sagemaker system
    os.chdir(work_dir)

    # logger.info(f"submit directory: {os.getenv('SAGEMAKER_SUBMIT_DIRECTORY')}")
    # submit_dir = os.getenv("SAGEMAKER_SUBMIT_DIRECTORY", "/opt/ml/code")

    return args


if __name__ == "__main__":
    # set seeds for reproducibility
    seed_everything(42, workers=True)

    args = parse_args()

    # set up the logger:
    match args.log_level:
        case "DEBUG":
            logging.basicConfig(level=logging.DEBUG)
        case "INFO":
            logging.basicConfig(level=logging.INFO)
        case "WARNING":
            logging.basicConfig(level=logging.WARNING)
        case "ERROR":
            logging.basicConfig(level=logging.ERROR)
        case "CRITICAL":
            logging.basicConfig(level=logging.CRITICAL)
        case _:
            raise ValueError(f"Invalid log level: {args.log_level}")

    # set up for sagemaker
    args = setup_for_sagemaker(args)

    # run the function specified by the subparser
    train_model(
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        data_folder=args.data_folder,
        model_folder=args.model_folder,
        tensorboard_dir=args.tensorboard_dir,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,  # “cpu”, “gpu”, “tpu”, “hpu”, “mps”, “auto”
        amp=args.amp,
        distributed=args.distributed,
        use_deterministic_algorithms=args.use_deterministic_algorithms,
        resume=args.resume,
        fast_dev_run=args.fast_dev_run,
    )
