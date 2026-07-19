import json
import logging
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from alphadia.calibration.estimator import CalibrationEstimator
from alphadia.constants.keys import CalibCols, ConstantsClass
from alphadia.workflow.managers.base import BaseManager

logger = logging.getLogger()


EstimatorGroups = dict[str, dict[str, CalibrationEstimator]]


class CalibrationGroups(metaclass=ConstantsClass):
    """String constants for calibration groups."""

    FRAGMENT = "fragment"
    PRECURSOR = "precursor"


class CalibrationEstimators(metaclass=ConstantsClass):
    """String constants for calibration estimators."""

    MZ = "mz"
    RT = "rt"
    MOBILITY = "mobility"


# Transform factor for deviations expressed in parts-per-million (m/z calibration).
PPM_TRANSFORM = 1e6


class CalibrationManager(BaseManager):
    def __init__(
        self,
        path: None | str = None,
        load_from_file: bool = True,
        has_ms1: bool = True,
        has_mobility: bool = True,
        **kwargs,
    ):
        """Contains, updates and applies all calibrations for a single run.

        Calibrations are grouped into calibration groups. Each calibration group is
        applied to a single data structure (precursor dataframe, fragment dataframe).
        Each group contains multiple estimators which each calibrate a single
        property (mz, rt, mobility). The numeric model and its configuration live in
        the Rust backend; this manager only maps dataframe columns to estimators.

        Parameters
        ----------
        path : str, default=None
            Path where the current parameter set is saved to and loaded from.

        load_from_file : bool, default=True
            If True, the manager will be loaded from file if it exists.

        has_ms1 : bool, default=True
            If True, an MS1 (precursor mz) estimator is included in the precursor group.

        has_mobility : bool, default=True
            If True, a mobility estimator is included in the precursor group.

        kwargs :
             Passed to the parent class `BaseManager`.

        """

        super().__init__(path=path, load_from_file=load_from_file, **kwargs)

        self._has_mobility = has_mobility
        self._has_ms1 = has_ms1

        self.reporter.log_string(f"Initializing {self.__class__.__name__}")
        self.reporter.log_event("initializing", {"name": f"{self.__class__.__name__}"})

        if not self.is_loaded_from_file:
            self.all_fitted = False
            self.estimator_groups: EstimatorGroups = self._build_estimator_groups()

    @property
    def estimator_groups(self) -> EstimatorGroups:
        """List of calibration groups."""
        return self._estimator_groups

    @estimator_groups.setter
    def estimator_groups(self, value: EstimatorGroups):
        self._estimator_groups = value

    def _build_estimator_groups(self) -> EstimatorGroups:
        """Build the fixed set of calibration estimators.

        The configuration (number of kernels, ppm transform) is hardcoded in the
        Rust backend and selected by the estimator ``kind``. Estimators that are not
        available in the raw data (MS1, mobility) are skipped.
        """
        self.reporter.log_string("Setting up calibration estimators ..")

        fragment_group = {
            CalibrationEstimators.MZ: CalibrationEstimator(
                name=CalibrationEstimators.MZ,
                n_kernels=2,
                transform_deviation=PPM_TRANSFORM,
                input_column=CalibCols.MZ_LIBRARY,
                target_column=CalibCols.MZ_OBSERVED,
                output_column=CalibCols.MZ_CALIBRATED,
            )
        }

        precursor_group: dict[str, CalibrationEstimator] = {}

        if self._has_ms1:
            precursor_group[CalibrationEstimators.MZ] = CalibrationEstimator(
                name=CalibrationEstimators.MZ,
                n_kernels=2,
                transform_deviation=PPM_TRANSFORM,
                input_column=CalibCols.MZ_LIBRARY,
                target_column=CalibCols.MZ_OBSERVED,
                output_column=CalibCols.MZ_CALIBRATED,
            )
        else:
            self.reporter.log_string(
                f"Skipping estimator '{CalibrationEstimators.MZ}' in group '{CalibrationGroups.PRECURSOR}' as it is not available in the raw data",
            )

        precursor_group[CalibrationEstimators.RT] = CalibrationEstimator(
            name=CalibrationEstimators.RT,
            n_kernels=6,
            transform_deviation=None,
            input_column=CalibCols.RT_LIBRARY,
            target_column=CalibCols.RT_OBSERVED,
            output_column=CalibCols.RT_CALIBRATED,
        )

        if self._has_mobility:
            precursor_group[CalibrationEstimators.MOBILITY] = CalibrationEstimator(
                name=CalibrationEstimators.MOBILITY,
                n_kernels=2,
                transform_deviation=None,
                input_column=CalibCols.MOBILITY_LIBRARY,
                target_column=CalibCols.MOBILITY_OBSERVED,
                output_column=CalibCols.MOBILITY_CALIBRATED,
            )
        else:
            self.reporter.log_string(
                f"Skipping estimator '{CalibrationEstimators.MOBILITY}' in group '{CalibrationGroups.PRECURSOR}' as it is not available in the raw data",
            )

        self.reporter.log_string("Done setting up calibration estimators.")

        return {
            CalibrationGroups.FRAGMENT: fragment_group,
            CalibrationGroups.PRECURSOR: precursor_group,
        }

    def get_estimator(
        self, group_name: str, estimator_name: str
    ) -> CalibrationEstimator | None:
        """Get an estimator from a calibration group.

        Parameters
        ----------
        group_name : str
            Name of the calibration group

        estimator_name : str
            Name of the estimator

        Returns
        -------
        CalibrationEstimator | None
            The estimator object or None if not found

        """
        try:
            return self.estimator_groups[group_name][estimator_name]
        except KeyError:
            return None

    def fit(
        self,
        df: pd.DataFrame,
        group_name: str,
        plot: bool = True,
        figure_path: None | str = None,
    ):
        """Fit all estimators in a calibration group.

        Parameters
        ----------
        df : pandas.DataFrame
            Dataframe containing the input and target columns

        group_name : str
            Name of the calibration group

        plot: bool, default=True
            If True, a plot of the calibration is generated.

        figure_path: str, default=None
            If set, the generated plot is saved to the given path.

        """

        for estimator in self.estimator_groups[group_name].values():
            self.reporter.log_string(
                f"Fitting estimator '{estimator.name}' in calibration group '{group_name}' .."
            )
            estimator.fit(df, plot=plot, figure_path=figure_path)

        all_fitted = True
        for group in self.estimator_groups.values():
            for estimator in group.values():
                all_fitted &= estimator.is_fitted
        self.all_fitted = all_fitted

    def get_stats(self) -> dict[str, dict[str, dict[str, float]]]:
        """Return the fitted metrics for each estimator, grouped by calibration group.

        Shape: ``{group_name: {estimator_name: {metric: value}}}``. Estimators that
        have not been fitted (no metrics) are omitted. Used to export calibration
        statistics for the output without persisting the whole manager.
        """
        return {
            group_name: {
                estimator_name: asdict(estimator.metrics)
                for estimator_name, estimator in group.items()
                if estimator.metrics is not None
            }
            for group_name, group in self.estimator_groups.items()
        }

    def save_stats(self, path: str) -> None:
        """Write the calibration metrics (see `get_stats`) to a JSON file."""
        with Path(path).open("w") as f:
            json.dump(self.get_stats(), f)

    def predict(self, df: pd.DataFrame, group_name: str):
        """Predict all estimators in a calibration group.

        Parameters
        ----------
        df : pandas.DataFrame
            Dataframe containing the input column

        group_name : str
            Name of the calibration group

        """

        for estimator in self.estimator_groups[group_name].values():
            self.reporter.log_string(
                f"Predicting estimator '{estimator.name}' in calibration group '{group_name}' .."
            )
            estimator.predict(df, inplace=True)
