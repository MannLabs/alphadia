"""The features that the automatic optimizers track in each optimization round."""

from abc import ABC, abstractmethod

import pandas as pd

from alphadia.workflow.optimizers.optimization_lock import OptimizationLock


class OptimizationFeature(ABC):
    """The value that an AutomaticOptimizer makes as large as possible.

    A feature keeps its history column name and its calculation together, so the two
    cannot drift apart. A feature has no state: optimizers reference the class itself.
    """

    name: str

    @staticmethod
    @abstractmethod
    def measure(
        precursors_df: pd.DataFrame,
        fragments_df: pd.DataFrame,
        optlock: OptimizationLock,
    ) -> float:
        """Calculate the feature value for one optimization round.

        Parameters
        ----------
        precursors_df: pd.DataFrame
            The filtered precursor dataframe for the search.

        fragments_df: pd.DataFrame
            The filtered fragment dataframe for the search.

        optlock: OptimizationLock
            The optimization lock that holds the state of the current batch.

        """


class PrecursorProportionDetected(OptimizationFeature):
    """The share of the elution groups in the optimization lock that gave a precursor at 1% FDR.

    The optimization handler stops the search before any optimizer sees a lock without
    elution groups, so the denominator is never zero.
    """

    name = "precursor_proportion_detected"

    @staticmethod
    def measure(
        precursors_df: pd.DataFrame,
        fragments_df: pd.DataFrame,
        optlock: OptimizationLock,
    ) -> float:
        """See base class."""
        return len(precursors_df) / optlock.total_elution_groups


class MeanIsotopeIntensityCorrelation(OptimizationFeature):
    """The mean isotope intensity correlation of the precursors found at 1% FDR.

    Without precursors the mean is NaN. The optimizer searches for the maximum of the
    feature and pandas ignores NaN in that search, so such a round is never the optimum.
    """

    name = "mean_isotope_intensity_correlation"

    @staticmethod
    def measure(
        precursors_df: pd.DataFrame,
        fragments_df: pd.DataFrame,
        optlock: OptimizationLock,
    ) -> float:
        """See base class."""
        return precursors_df["isotope_intensity_correlation"].mean()
