from typing import Any, Dict

from .filters_config import BaseFilterConfig


class BaseFilter:
    """
    Base class for filters. This class should be subclassed to create a new filter.

    Args:

        config (BaseFilterConfig):
            Configuration for the filter
    """

    def __init__(self, config: BaseFilterConfig):
        self.verbose = config.verbose

    def __call__(self, sample: Dict[str, Any]) -> bool:
        """This function should be implemented by the subclass"""
        raise NotImplementedError
