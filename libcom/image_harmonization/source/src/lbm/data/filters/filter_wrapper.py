from typing import Any, Dict, List, Union

from .base import BaseFilter


class FilterWrapper:
    """
    Wrapper for multiple filters. This class allows to apply multiple filters to a batch of data.
    The filters are applied in the order they are passed to the wrapper.

    Args:

        filters (List[BaseFilter]):
            List of filters to apply to the batch of data
    """

    def __init__(
        self,
        filters: Union[List[BaseFilter], None] = None,
    ):
        self.filters = filters

    def __call__(self, batch: Dict[str, Any]) -> None:
        """
        Forward pass through all filters

        Args:

            batch: batch of data
        """
        filter_output = True
        for filter in self.filters:
            filter_output = filter(batch)
            if not filter_output:
                return False
        return True
