from typing import Any, Dict

from .mappers_config import BaseMapperConfig


class BaseMapper:
    """
    Base class for the mappers used to modify the samples in the data pipeline.

    Args:

        config (BaseMapperConfig):
            Configuration for the mapper.
    """

    def __init__(self, config: BaseMapperConfig):
        self.config = config
        self.key = config.key

        if config.output_key is None:
            self.output_key = config.key
        else:
            self.output_key = config.output_key

    def map(self, batch: Dict[str, Any], *args, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError
