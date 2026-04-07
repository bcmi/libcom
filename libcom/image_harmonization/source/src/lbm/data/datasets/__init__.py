"""
A collection of :mod:`pytorch_lightning.LightningDataModule` used to train the models. In particular,
they can be used to create the dataloaders and setup the data pipelines.
"""

from .dataset import DataModule
from .datasets_config import DataModuleConfig

__all__ = ["DataModule", "DataModuleConfig"]
