"""Interface for the (non-JIT) DIA data."""

from abc import ABC, abstractmethod

import numpy as np


class DiaData(ABC):
    """Abstract class providing the interface for the (non-JIT) DIA data."""

    @property
    @abstractmethod
    def has_mobility(self) -> bool:
        """Whether the data contains mobility values."""

    @property
    @abstractmethod
    def has_ms1(self) -> bool:
        """Whether the data contains MS1 scans."""

    @property
    @abstractmethod
    def mobility_values(self) -> np.ndarray[tuple[int], np.dtype[np.float32]]:
        """Mobility values."""

    @property
    @abstractmethod
    def rt_values(self) -> np.ndarray[tuple[int], np.dtype[np.float32]]:
        """Retention time values."""

    @property
    @abstractmethod
    def cycle(self) -> np.ndarray[tuple[int, int, int, int], np.dtype[np.float64]]:
        """Cycle information."""
