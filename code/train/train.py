"""
Training logic.
"""

from pathlib import Path
import os
import logging

import torch.utils.data.distributed
import torch.optim.lr_scheduler
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger

from models.model import DLModel

from datasets.datamodule import DataModule

logger = logging.getLogger()


def train_model(
    max_epochs: int = 26,
    batch_size: int = 2,
    data_folder="data",
    model_folder=".",
    tensorboard_dir="/opt/ml/output/tensorboard",
    checkpoint_dir="/opt/ml/checkpoints",
    device="gpu" if torch.cuda.is_available() else "cpu",
    distributed: bool = False,
    use_deterministic_algorithms: bool = False,
    amp: bool = False,  # Use torch.cuda.amp for mixed precision training
    resume: str = "",  # "path of checkpoint"
    fast_dev_run: bool = False,
):
    """Train the segmentation model

    Note that if there are any weights in the checkpoint directory, the latest ones
    will be used as the starting point for training

    Parameters
    ----------
    max_epochs: int, default: 25
        The maximum number of epochs to train for
    init_learning_rate: float, default: 1e-6
        The learning rate used when starting to train
    max_learning_rate: float, default: 5e-4
        The maximum learning rate to scale the learning rate to
    val_size: float < 1, default: 0.1
        The ratio of the entire dataset to use for the validation set
    test_size: float < 1, default: 0.1
        The ratio of the entire dataset to use for the test set
    data_folder: pathlib.Path
        Path of the data folder, which should be set up as described in `data/README.md`
    model_folder: pathlib.Path | str, default: "."
        The folder to save the trained model to
    tensorboard_dir: pathlib.Path | str, default: "/opt/ml/output/tensorboard"
        The folder to save tensorboard logs to
    checkpoint_dir: pathlib.Path | str, default: "/opt/ml/checkpoints"
        The folder to save model checkpoints to.
    device: str. “cpu”, “gpu”, “tpu”, “hpu”, “mps”, “auto”, default: "gpu"
    num_dataloader_workers: int, default: 10
        The number of workers to use in the dataloaders. More workers can speed
        up training if the problem is IO bound, but also can increase CPU memory usage.
    starting_weights: str, default: None
        The S3 URI to download the starting weights from in case there are not already
        files in teh checkpoint directory. If None, no weights will be loaded other than
        checkpoints.
    starting_epoch: int, default: 0
        Epoch number associated with the starting_weights.
        If starting_weights is None, this value will be ignored.
        If there are checkpoint files in the checkpoint dir, this value
        (and the starting_weights) will be ignored
    weigth_decay: float, default: 1e-4
        weight decay
    norm_weight_decay: float, default: None
        weight decay for Normalization layers (default: None, same value as weight_decay)
    momentum: float, default: 0.9
        momentum
    """
    data_folder = Path(data_folder)
    model_folder = Path(model_folder)

    logger.debug("Contents of the data folder: %s", os.listdir(data_folder))

    # Data loading code
    data_module = DataModule(
        data_dir=str(data_folder),
        batch_size=batch_size,
    )

    if resume:
        model = DLModel.load_from_checkpoint(resume)
    else:
        data_module.setup("fit")
        model = DLModel()

    tb_logger = TensorBoardLogger(tensorboard_dir, name="lightning_logs")
    extra_params = {}
    if distributed:
        extra_params["use_distributed_sampler"] = True
    if amp:
        extra_params["precision"] = "bf16-mixed"
    if fast_dev_run:
        extra_params["fast_dev_run"] = 40
    if device == "gpu":
        # take advantage of GPU with Tensor cores
        torch.set_float32_matmul_precision("medium")

    logger.info("start training")
    trainer = Trainer(
        strategy="auto",
        deterministic=use_deterministic_algorithms,
        accelerator=device,
        default_root_dir=checkpoint_dir,
        enable_checkpointing=True,
        logger=[tb_logger],
        max_epochs=max_epochs,
        # limit_train_batches=20,
        # limit_val_batches=20,
        **extra_params,
    )
    trainer.fit(model, data_module)

    trainer.save_checkpoint(model_folder / "model.ckpt")
