from typing import Any, Dict, List, Union

from .base import BaseMapper


class MapperWrapper:
    """
    Wrapper for the mappers to allow iterating over several mappers in one go.

    Args:

        mappers (Union[List[BaseMapper], None]): List of mappers to apply to the batch
    """

    def __init__(
        self,
        mappers: Union[List[BaseMapper], None] = None,
    ):
        self.mappers = mappers

    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Forward pass through all mappers

        Args:

            batch (Dict[str, Any]): batch of data
        """
        for mapper in self.mappers:
            batch = mapper(batch)
        return batch
