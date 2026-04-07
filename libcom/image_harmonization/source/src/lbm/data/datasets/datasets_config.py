from typing import Callable, List, Optional, Union

import webdataset as wds
from pydantic.dataclasses import dataclass

from ...config import BaseConfig


@dataclass
class DataModuleConfig(BaseConfig):
    """
    Configuration for the DataModule

    Args:

        shards_path_or_urls (Union[str, List[str]]): The path or url to the shards. Defaults to None.
        per_worker_batch_size (int): The batch size for the dataset. Defaults to 16.
        num_workers (int): The number of workers to use. Defaults to 1.
        shuffle_before_split_by_node_buffer_size (Optional[int]): The buffer size for the shuffle before split by node. Defaults to 100.
        shuffle_before_split_by_workers_buffer_size (Optional[int]): The buffer size for the shuffle before split by workers. Defaults to 100.
        shuffle_before_filter_mappers_buffer_size (Optional[int]): The buffer size for the shuffle before filter mappers. Defaults to 1000.
        shuffle_after_filter_mappers_buffer_size (Optional[int]): The buffer size for the shuffle after filter mappers. Defaults to 1000.
        decoder (str): The decoder to use. Defaults to "pil".
        handler (Callable): A callable to handle the warnings. Defaults to wds.warn_and_continue.
        rename_files_fn (Optional[Callable[[str], str]]): A callable to rename the files. Defaults to None.
    """

    shards_path_or_urls: Union[str, List[str]] = None
    per_worker_batch_size: int = 16
    num_workers: int = 1
    shuffle_before_split_by_node_buffer_size: Optional[int] = 100
    shuffle_before_split_by_workers_buffer_size: Optional[int] = 100
    shuffle_before_filter_mappers_buffer_size: Optional[int] = 1000
    shuffle_after_filter_mappers_buffer_size: Optional[int] = 1000
    decoder: str = "pil"
    handler: Callable = wds.warn_and_continue
    rename_files_fn: Optional[Callable[[str], str]] = None

    def __post_init__(self):
        super().__post_init__()
        if self.rename_files_fn is not None:
            assert callable(self.rename_files_fn), "rename_files must be a callable"
