from typing import Callable, List, Union

import pytorch_lightning as pl
import webdataset as wds
from webdataset import DataPipeline

from ..filters import BaseFilter, FilterWrapper
from ..mappers import BaseMapper, MapperWrapper
from .collation_fn import custom_collation_fn
from .datasets_config import DataModuleConfig


class DataPipeline:
    """
    DataPipeline class for creating a dataloader from a single configuration

    Args:

        config (DataModuleConfig):
            Configuration for the dataset

        filters_mappers (Union[List[Union[BaseMapper, BaseFilter, FilterWrapper, MapperWrapper]]):
            List of filters and mappers for the dataset. These will be sequentially applied.

        batched_filters_mappers (List[Union[BaseMapper, BaseFilter, FilterWrapper, MapperWrapper]]):
            List of batched transforms for the dataset. These will be sequentially applied.
    """

    def __init__(
        self,
        config: DataModuleConfig,
        filters_mappers: List[
            Union[BaseMapper, BaseFilter, FilterWrapper, MapperWrapper]
        ],
        batched_filters_mappers: List[
            Union[BaseMapper, BaseFilter, FilterWrapper, MapperWrapper]
        ] = None,
    ):
        self.config = config
        self.shards_path_or_urls = config.shards_path_or_urls
        self.filters_mappers = filters_mappers
        self.batched_filters_mappers = batched_filters_mappers or []

        if filters_mappers is None:
            filters_mappers = []

        # set processing pipeline
        self.processing_pipeline = [wds.decode(config.decoder, handler=config.handler)]
        self.processing_pipeline.extend(
            self._add_filters_mappers(
                filters_mappers=filters_mappers,
                handler=config.handler,
            )
        )

    def _add_filters_mappers(
        self,
        filters_mappers: List[
            Union[
                FilterWrapper,
                MapperWrapper,
            ]
        ],
        handler: Callable = wds.warn_and_continue,
    ) -> List[Union[FilterWrapper, MapperWrapper]]:
        tmp_pipeline = []
        for filter_mapper in filters_mappers:
            if isinstance(filter_mapper, FilterWrapper) or isinstance(
                filter_mapper, BaseFilter
            ):
                tmp_pipeline.append(wds.select(filter_mapper))
            elif isinstance(filter_mapper, MapperWrapper) or isinstance(
                filter_mapper, BaseMapper
            ):
                tmp_pipeline.append(wds.map(filter_mapper, handler=handler))
            elif isinstance(filter_mapper) or isinstance(filter_mapper):
                tmp_pipeline.append(wds.map(filter_mapper, handler=handler))
            else:
                raise ValueError("Unknown type of filter/mapper")
        return tmp_pipeline

    def setup(self):
        pipeline = [wds.SimpleShardList(self.shards_path_or_urls)]

        # shuffle before split by node
        if self.config.shuffle_before_split_by_node_buffer_size is not None:
            pipeline.append(
                wds.shuffle(
                    self.config.shuffle_before_split_by_node_buffer_size,
                    handler=self.config.handler,
                )
            )
        # split by node
        pipeline.append(wds.split_by_node)

        # shuffle before split by workers
        if self.config.shuffle_before_split_by_workers_buffer_size is not None:
            pipeline.append(
                wds.shuffle(
                    self.config.shuffle_before_split_by_workers_buffer_size,
                    handler=self.config.handler,
                )
            )
        # split by worker
        pipeline.extend(
            [
                wds.split_by_worker,
                wds.tarfile_to_samples(
                    handler=self.config.handler,
                    rename_files=self.config.rename_files_fn,
                ),
            ]
        )

        # shuffle before filter mappers
        if self.config.shuffle_before_filter_mappers_buffer_size is not None:
            pipeline.append(
                wds.shuffle(
                    self.config.shuffle_before_filter_mappers_buffer_size,
                    handler=self.config.handler,
                )
            )

        # apply filters and mappers
        pipeline.extend(self.processing_pipeline)

        # shuffle after filter mappers
        if self.config.shuffle_after_filter_mappers_buffer_size is not None:
            pipeline.append(
                wds.shuffle(
                    self.config.shuffle_after_filter_mappers_buffer_size,
                    handler=self.config.handler,
                ),
            )

        # batching
        pipeline.append(
            wds.batched(
                self.config.per_worker_batch_size,
                collation_fn=custom_collation_fn,
            )
        )

        # apply batched transforms
        pipeline.extend(
            self._add_filters_mappers(
                filters_mappers=self.batched_filters_mappers,
                handler=self.config.handler,
            )
        )

        # create the data pipeline
        pipeline = wds.DataPipeline(*pipeline, handler=self.config.handler)

        # set the pipeline
        self.pipeline = pipeline

    def dataloader(self):
        # return the loader
        return wds.WebLoader(
            self.pipeline,
            batch_size=None,
            num_workers=self.config.num_workers,
        )


class DataModule(pl.LightningDataModule):
    """
    Main DataModule class for creating data loaders and training/evaluating models

    Args:

        train_config (DataModuleConfig):
            Configuration for the training dataset

        train_filters_mappers (Union[List[Union[BaseMapper, BaseFilter, FilterWrapper, MapperWrapper]]):
            List of filters and mappers for the training dataset. These will be sequentially applied.

        train_batched_filters_mappers (List[Union[BaseMapper, BaseFilter, FilterWrapper, MapperWrapper]]):
            List of batched transforms for the training dataset. These will be sequentially applied.

        eval_config (DataModuleConfig):
            Configuration for the evaluation dataset

        eval_filters_mappers (List[Union[FilterWrapper, MapperWrapper]]):
            List of filters and mappers for the evaluation dataset.These will be sequentially applied.

        eval_batched_filters_mappers (List[Union[BaseMapper, BaseFilter, FilterWrapper, MapperWrapper]]):
            List of batched transforms for the evaluation dataset. These will be sequentially applied.
    """

    def __init__(
        self,
        train_config: DataModuleConfig,
        train_filters_mappers: List[
            Union[BaseMapper, BaseFilter, FilterWrapper, MapperWrapper]
        ] = None,
        train_batched_filters_mappers: List[
            Union[BaseMapper, BaseFilter, FilterWrapper, MapperWrapper]
        ] = None,
        eval_config: DataModuleConfig = None,
        eval_filters_mappers: List[Union[FilterWrapper, MapperWrapper]] = None,
        eval_batched_filters_mappers: List[
            Union[BaseMapper, BaseFilter, FilterWrapper, MapperWrapper]
        ] = None,
    ):
        super().__init__()

        self.train_config = train_config
        self.train_filters_mappers = train_filters_mappers
        self.train_batched_filters_mappers = train_batched_filters_mappers

        self.eval_config = eval_config
        self.eval_filters_mappers = eval_filters_mappers
        self.eval_batched_filters_mappers = eval_batched_filters_mappers

    def setup(self, stage=None):
        """
        Setup the data module and create the webdataset processing pipelines
        """

        # train pipeline
        self.train_pipeline = DataPipeline(
            config=self.train_config,
            filters_mappers=self.train_filters_mappers,
            batched_filters_mappers=self.train_batched_filters_mappers,
        )
        self.train_pipeline.setup()

        # eval pipeline
        if self.eval_config is not None:
            self.eval_pipeline = DataPipeline(
                config=self.eval_config,
                filters_mappers=self.eval_filters_mappers,
                batched_filters_mappers=self.eval_batched_filters_mappers,
            )
            self.eval_pipeline.setup()

    def train_dataloader(self):
        return self.train_pipeline.dataloader()

    def val_dataloader(self):
        return self.eval_pipeline.dataloader()
