"""Interface for DIA data."""

from abc import ABC, abstractmethod

import numpy as np

from alphadia.raw_data.jitclasses.alpharaw_jit import AlphaRawJIT
from alphadia.raw_data.jitclasses.bruker_jit import TimsTOFTransposeJIT

DiaDataJIT = TimsTOFTransposeJIT | AlphaRawJIT


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

    @abstractmethod
    def to_jitclass(self) -> DiaDataJIT:
        """Create a JIT-compatible class with the current state."""
