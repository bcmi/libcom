"""
This module contains the training pipeline and the training configuration along with all relevant parts 
of the training pipeline such as loggers and callbacks.

The :mod:`cr.trainer` includes the following submodules:

- :mod:`cr.trainer.trainer`: the main training pipeline class for ClipDrop.
- :mod:`cr.trainer.training_config`: the configuration for the training pipeline.
- :mod:`cr.trainer.loggers`: the loggers for logging samples to wandb.


Examples
########

Train a model using the training pipeline

.. code-block:: python

    from cr.trainer import TrainingPipeline, TrainingConfig
    from cr.data import DataPipeline, DataConfig
    from pytorch_lightning import Trainer
    from cr.data.datasets import DataModule, DataModuleConfig

    # Create a model to train
    model = DummyModel()

    # Create a training configuration
    config = TrainingConfig(
        experiment_id="test",
        optimizers_name=["AdamW"],
        optimizers_kwargs=[{}],
        learning_rates=[1e-3],
        lr_schedulers_name=[None],
        lr_schedulers_kwargs=[{}],
        trainable_params=[["./*"]],
        log_keys="txt",
        log_samples_model_kwargs={
            "max_samples": 8,
            "num_steps": 20,
            "input_shape": (4, 32, 32),
            "guidance_scale": 7.5,
        }
    )

    # Create a training pipeline
    pipeline = TrainingPipeline(model=model, pipeline_config=config)
    
    # Create a DataModule
    data_module = DataModule(
            train_config=DataModuleConfig(
                shards_path_or_urls="your urls or paths",
                decoder="pil",
                shuffle_buffer_size=100,
                per_worker_batch_size=32,
                num_workers=4,
            ),
            train_filters_mappers=your_mappers_and_filters,
            eval_config=DataModuleConfig(
                shards_path_or_urls="your urls or paths",
                decoder="pil",
                shuffle_buffer_size=100,
                per_worker_batch_size=32,
                num_workers=4,
            ),
            eval_filters_mappers=your_mappers_and_filters,
        )

    # Create a trainer
    trainer = Trainer(
            accelerator="cuda",
            max_epochs=1,
            devices=1,
            log_every_n_steps=1,
            default_root_dir="your dir",
            max_steps=2,
        )

    # Train the model
    trainer.fit(pipeline, data_module)
"""

from .trainer import TrainingPipeline
from .training_config import TrainingConfig

__all__ = ["TrainingPipeline", "TrainingConfig"]
