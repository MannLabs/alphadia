from abc import ABC

import pandas as pd

from alphadia.reporting import reporting
from alphadia.workflow.config import Config
from alphadia.workflow.managers.calibration_manager import (
    CalibrationEstimators,
    CalibrationGroups,
    CalibrationManager,
)
from alphadia.workflow.managers.fdr_manager import FDRManager
from alphadia.workflow.managers.optimization_manager import OptimizationManager
from alphadia.workflow.optimizers.base import BaseOptimizer


class TargetedOptimizer(BaseOptimizer, ABC):
    def __init__(
        self,
        initial_parameter: float,
        target_parameter: float,
        config: Config,
        optimization_manager: OptimizationManager,
        calibration_manager: CalibrationManager,
        fdr_manager: FDRManager,
        reporter: None | reporting.Pipeline | reporting.Backend = None,
    ):
        """This class optimizes the search parameter until it reaches a user-specified target value.

        Parameters
        ----------

        initial_parameter: float
            The parameter used for search in the first round of optimization.

        target_parameter: float
            Optimization will stop when this parameter is reached.

        See base class for other parameters.

        """
        super().__init__(
            config, optimization_manager, calibration_manager, fdr_manager, reporter
        )
        self._optimization_manager.update(**{self.parameter_name: initial_parameter})
        self.target_parameter = target_parameter
        self.update_factor = self._config["optimization"][self.parameter_name][
            "targeted_update_factor"
        ]
        self.update_percentile_range = self._config["optimization"][
            self.parameter_name
        ]["targeted_update_percentile_range"]
        self.has_converged = False

    def _check_convergence(self, proposed_parameter: float):
        """The optimization has converged if the proposed parameter is equal to or less than the target parameter and the a sufficient number of steps has been taken.

        Parameters
        ----------
        proposed_parameter: float
            The proposed parameter for the next round of optimization.

        Returns
        -------
        bool
            True if proposed parameter less than target and the current step is greater than the minimum required, False otherwise.


        """
        min_steps_reached = (
            self._num_prev_optimizations >= self._config["calibration"]["min_steps"]
        )
        return proposed_parameter <= self.target_parameter and min_steps_reached

    def _propose_new_parameter(self, df: pd.DataFrame):
        """See base class. The update rule is
            1) calculate the deviation of the predicted mz values from the observed mz values,
            2) take the mean of the endpoints of the central 95% of these deviations, and
            3) take the maximum of this value and the target parameter.
        This is implemented by the ci method for the estimator.
        """
        return self.update_factor * max(
            self._calibration_manager.get_estimator(
                self._estimator_group_name, self._estimator_name
            ).ci(df, self.update_percentile_range),
            self.target_parameter,
        )

    def step(
        self,
        precursors_df: pd.DataFrame,
        fragments_df: pd.DataFrame,
    ):
        """See base class."""
        if self.has_converged:
            self._reporter.log_string(
                f"✅ {self.parameter_name:<15}: {self._optimization_manager.__dict__[self.parameter_name]:.4f} <= {self.target_parameter:.4f}",
                verbosity="progress",
            )
            return
        self._num_prev_optimizations += 1
        new_parameter = self._propose_new_parameter(
            precursors_df
            if self._estimator_group_name == CalibrationGroups.PRECURSOR
            else fragments_df
        )
        just_converged = self._check_convergence(new_parameter)
        self._optimization_manager.update(**{self.parameter_name: new_parameter})
        self._optimization_manager.update(
            classifier_version=self._fdr_manager.current_version
        )

        if just_converged:
            self.has_converged = True
            self._reporter.log_string(
                f"✅ {self.parameter_name:<15}: {self._optimization_manager.__dict__[self.parameter_name]:.4f} <= {self.target_parameter:.4f}",
                verbosity="progress",
            )

        else:
            self._reporter.log_string(
                f"❌ {self.parameter_name:<15}: {self._optimization_manager.__dict__[self.parameter_name]:.4f} > {self.target_parameter:.4f} or insufficient steps taken.",
                verbosity="progress",
            )

    def skip(self):
        """See base class."""

    def plot(self):
        """See base class"""

    def _update_workflow(self):
        pass

    def _update_history(self, precursors_df: pd.DataFrame, fragments_df: pd.DataFrame):
        pass


class TargetedRTOptimizer(TargetedOptimizer):
    def __init__(
        self,
        initial_parameter: float,
        target_parameter: float,
        config: Config,
        optimization_manager: OptimizationManager,
        calibration_manager: CalibrationManager,
        fdr_manager: FDRManager,
        reporter: None | reporting.Pipeline | reporting.Backend = None,
    ):
        """See base class."""
        self.parameter_name = "rt_error"
        self._estimator_group_name = CalibrationGroups.PRECURSOR
        self._estimator_name = CalibrationEstimators.RT
        super().__init__(
            initial_parameter,
            target_parameter,
            config,
            optimization_manager,
            calibration_manager,
            fdr_manager,
            reporter,
        )


class TargetedMS2Optimizer(TargetedOptimizer):
    def __init__(
        self,
        initial_parameter: float,
        target_parameter: float,
        config: Config,
        optimization_manager: OptimizationManager,
        calibration_manager: CalibrationManager,
        fdr_manager: FDRManager,
        reporter: None | reporting.Pipeline | reporting.Backend = None,
    ):
        """See base class."""
        self.parameter_name = "ms2_error"
        self._estimator_group_name = CalibrationGroups.FRAGMENT
        self._estimator_name = CalibrationEstimators.MZ
        super().__init__(
            initial_parameter,
            target_parameter,
            config,
            optimization_manager,
            calibration_manager,
            fdr_manager,
            reporter,
        )


class TargetedMS1Optimizer(TargetedOptimizer):
    def __init__(
        self,
        initial_parameter: float,
        target_parameter: float,
        config: Config,
        optimization_manager: OptimizationManager,
        calibration_manager: CalibrationManager,
        fdr_manager: FDRManager,
        reporter: None | reporting.Pipeline | reporting.Backend = None,
    ):
        """See base class."""
        self.parameter_name = "ms1_error"
        self._estimator_group_name = CalibrationGroups.PRECURSOR
        self._estimator_name = CalibrationEstimators.MZ
        super().__init__(
            initial_parameter,
            target_parameter,
            config,
            optimization_manager,
            calibration_manager,
            fdr_manager,
            reporter,
        )


class TargetedMobilityOptimizer(TargetedOptimizer):
    def __init__(
        self,
        initial_parameter: float,
        target_parameter: float,
        config: Config,
        optimization_manager: OptimizationManager,
        calibration_manager: CalibrationManager,
        fdr_manager: FDRManager,
        reporter: None | reporting.Pipeline | reporting.Backend = None,
    ):
        """See base class."""
        self.parameter_name = "mobility_error"
        self._estimator_group_name = CalibrationGroups.PRECURSOR
        self._estimator_name = CalibrationEstimators.MOBILITY
        super().__init__(
            initial_parameter,
            target_parameter,
            config,
            optimization_manager,
            calibration_manager,
            fdr_manager,
            reporter,
        )
