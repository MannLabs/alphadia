"""The features that the automatic optimizers track in each optimization round."""

from abc import ABC, abstractmethod

import pandas as pd

from alphadia.workflow.optimizers.optimization_lock import OptimizationLock


class OptimizationFeature(ABC):
    """The value that an AutomaticOptimizer makes as large as possible.

    A feature keeps its history column name and its calculation together. Thus the
    name and the calculation stay in agreement. Each feature also gives one
    definition of a round that has no data.

    A feature must not keep state. All optimizers that use a feature share one
    instance of it.
    """

    name: str

    @abstractmethod
    def measure(
        self,
        precursors_df: pd.DataFrame,
        fragments_df: pd.DataFrame,
        optlock: OptimizationLock,
    ) -> float | None:
        """Calculate the feature value for one optimization round.

        Parameters
        ----------
        precursors_df: pd.DataFrame
            The filtered precursor dataframe for the search.

        fragments_df: pd.DataFrame
            The filtered fragment dataframe for the search.

        optlock: OptimizationLock
            The optimization lock that holds the state of the current batch.

        Returns
        -------
        float | None
            The feature value, or None if this round has no data to measure.

        """


class PrecursorProportionDetected(OptimizationFeature):
    """The part of the elution groups in the batch that gave a precursor at 1% FDR.

    The value has no definition if the batch has no elution groups. There is no
    quantity to calculate a part of. But if the batch has elution groups and no
    precursor, the value is a correct measurement of zero. The optimizer records it.
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
    """The mean isotope intensity correlation of the precursors that were found.

    The value has no definition if there is no precursor. In this condition pandas
    gives NaN. A NaN does not stop the convergence calculations. It moves through
    them without a message.
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

        # A column that contains only NaN has no more data than an empty frame.
        # Pandas gives NaN for the two conditions and does not raise an error.
        mean_correlation = precursors_df["isotope_intensity_correlation"].mean()
        if pd.isna(mean_correlation):
            return None
        return float(mean_correlation)
