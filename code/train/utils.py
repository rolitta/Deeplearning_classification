"""
Helper utils for model training
"""

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter  # type: ignore


def add_running_tb_logs(
    writer: SummaryWriter,
    running_loss: float,
    epoch: int,
    i: int,
    train_dataloader: DataLoader,
):
    """
    helper for logging metrics and sample images to tensorboard
    """
    # ...log the running loss
    writer.add_scalar(
        "Running training loss",
        running_loss / 1000,
        (epoch - 1) * len(train_dataloader) + i,
    )
