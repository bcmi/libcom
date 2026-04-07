import logging

from .base import BaseFilter
from .filters_config import KeyFilterConfig

logging.basicConfig(level=logging.INFO)


class KeyFilter(BaseFilter):
    """
    This filter checks if ALL the given keys are present in the sample

    Args:

        config (KeyFilterConfig): configuration for the filter
    """

    def __init__(self, config: KeyFilterConfig):
        super().__init__(config)
        keys = config.keys
        if isinstance(keys, str):
            keys = [keys]

        self.keys = set(keys)

    def __call__(self, batch: dict) -> bool:
        try:
            res = self.keys.issubset(set(batch.keys()))
            return res
        except Exception as e:
            if self.verbose:
                logging.error(f"Error in KeyFilter: {e}")
            return False
