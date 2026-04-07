from dataclasses import field
from typing import List, Literal, Optional, Union

from pydantic.dataclasses import dataclass

from ..config import BaseConfig


@dataclass
class TrainingConfig(BaseConfig):
    """
    Configuration for the training pipeline

    Args:

        experiment_id (str):
            The experiment id for the training run. If not provided, a random id will be generated.
        optimizer_name (str):
            The optimizer to use. Default is "AdamW". Choices are "Adam", "AdamW", "Adadelta", "Adagrad", "RMSprop", "SGD"
        optimizer_kwargs (Dict[str, Any])
            The optimizer kwargs. Default is [{}]
        learning_rate (float):
            The learning rate to use. Default is 1e-3
        lr_scheduler_name (str):
            The learning rate scheduler to use. Default is None. Choices are "StepLR", "CosineAnnealingLR",
            "CosineAnnealingWarmRestarts", "ReduceLROnPlateau", "ExponentialLR"
        lr_scheduler_kwargs (Dict[str, Any])
            The learning rate scheduler kwargs. Default is [{}]
        lr_scheduler_interval (str):
            The learning rate scheduler interval. Default is ["step"]. Choices are "step", "epoch"
        lr_scheduler_frequency (int):
            The learning rate scheduler frequency. Default is 1
        metrics (List[str])
            The metrics to use. Default is None
        tracking_metrics: Optional[List[str]]
            The metrics to track. Default is None
        backup_every (int):
            The frequency to backup the model. Default is 50.
        trainable_params (Union[str, List[str]]):
            Regexes indicateing the parameters to train.
            Default is [["./*"]] (i.e. all parameters are trainable)
        log_keys: Union[str, List[str]]:
            The keys to log when sampling from the model. Default is "txt"
        log_samples_model_kwargs (Dict[str, Any]):
            The kwargs for logging samples from the model. Default is {
                "max_samples": 4,
                "num_steps": 20,
                "input_shape": None,
            }
    """

    experiment_id: Optional[str] = None
    optimizer_name: Literal[
        "Adam", "AdamW", "Adadelta", "Adagrad", "RMSprop", "SGD"
    ] = field(default_factory=lambda: "AdamW")
    optimizer_kwargs: Optional[dict] = field(default_factory=lambda: {})
    learning_rate: float = field(default_factory=lambda: 1e-3)
    lr_scheduler_name: Optional[
        Literal[
            "StepLR",
            "CosineAnnealingLR",
            "CosineAnnealingWarmRestarts",
            "ReduceLROnPlateau",
            "ExponentialLR",
            None,
        ]
    ] = None
    lr_scheduler_kwargs: Optional[dict] = field(default_factory=lambda: {})
    lr_scheduler_interval: Optional[Literal["step", "epoch", None]] = "step"
    lr_scheduler_frequency: Optional[int] = 1
    metrics: Optional[List[str]] = None
    tracking_metrics: Optional[List[str]] = None
    backup_every: int = 50
    trainable_params: List[str] = field(default_factory=lambda: ["./*"])
    log_keys: Optional[Union[str, List[str]]] = "txt"
    log_samples_model_kwargs: Optional[dict] = field(
        default_factory=lambda: {
            "max_samples": 4,
            "num_steps": 20,
            "input_shape": None,
        }
    )
