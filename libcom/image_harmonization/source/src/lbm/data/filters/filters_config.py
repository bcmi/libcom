from typing import List, Union

from pydantic.dataclasses import dataclass

from ...config import BaseConfig


@dataclass
class BaseFilterConfig(BaseConfig):
    """
    Base configuration for filters

    Args:

        verbose (bool):
            If True, print debug information. Defaults to False"""

    verbose: bool = False


@dataclass
class KeyFilterConfig(BaseFilterConfig):
    """
    This filter checks if the keys are present in a sample.

    Args:

        keys (Union[str, List[str]]):
            Key or list of keys to check. Defaults to "txt"
    """

    keys: Union[str, List[str]] = "txt"
