"""Features tracked by the automatic optimizers across optimization rounds."""

from abc import ABC, abstractmethod

import pandas as pd

from alphadia.workflow.optimizers.optimization_lock import OptimizationLock


class OptimizationFeature(ABC):
    """The quantity an AutomaticOptimizer maximises across optimization rounds.

    Bundles the history column name with its computation so the two cannot drift
    apart, and gives every feature one definition of the empty search.

    Implementations must be stateless: a single instance is shared by all
    optimizers declaring the feature.
    """

    name: str

    @abstractmethod
    def measure(
        self,
        precursors_df: pd.DataFrame,
        fragments_df: pd.DataFrame,
        optlock: OptimizationLock,
    ) -> float | None:
        """Return the feature value, or None when this round admits no measurement.

        Parameters
        ----------
        precursors_df: pd.DataFrame
            The filtered precursor dataframe for the search.

        fragments_df: pd.DataFrame
            The filtered fragment dataframe for the search.

        optlock: OptimizationLock
            The optimization lock holding the state of the current batch.

        """


class PrecursorProportionDetected(OptimizationFeature):
    """Fraction of the batch's elution groups that yielded a precursor at 1% FDR.

    Undefined when the batch contributed no elution groups: there is nothing for a
    proportion to be *of*. Zero precursors out of a non-empty batch, by contrast, is
    a real measurement of zero and is recorded as such.
    """

    name = "precursor_proportion_detected"

    def measure(
        self,
        precursors_df: pd.DataFrame,
        fragments_df: pd.DataFrame,
        optlock: OptimizationLock,
    ) -> float | None:
        """See base class."""
        if optlock.total_elution_groups == 0:
            return None
        return len(precursors_df) / optlock.total_elution_groups


class MeanIsotopeIntensityCorrelation(OptimizationFeature):
    """Mean isotope intensity correlation over the detected precursors.

    Undefined with no precursors: pandas returns NaN, which propagates silently
    through the convergence comparisons rather than stopping them.
    """

    name = "mean_isotope_intensity_correlation"

    def measure(
        self,
        precursors_df: pd.DataFrame,
        fragments_df: pd.DataFrame,
        optlock: OptimizationLock,
    ) -> float | None:
        """See base class."""
        if precursors_df.empty:
            return None

        # An all-NaN column is as unmeasurable as an empty frame, and pandas reports
        # both as NaN rather than raising.
        mean_correlation = precursors_df["isotope_intensity_correlation"].mean()
        if pd.isna(mean_correlation):
            return None
        return float(mean_correlation)
