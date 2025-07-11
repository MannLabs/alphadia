from abc import ABC, abstractmethod

import pandas as pd

from alphadia.reporting import reporting
from alphadia.workflow.config import Config
from alphadia.workflow.managers.calibration_manager import CalibrationManager
from alphadia.workflow.managers.fdr_manager import FDRManager
from alphadia.workflow.managers.optimization_manager import OptimizationManager


class BaseOptimizer(ABC):
    parameter_name: str | None
    _estimator_name: str | None
    _estimator_group_name: str | None
    _feature_name: str | None

    def __init__(
        self,
        config: Config,
        optimization_manager: OptimizationManager,
        calibration_manager: CalibrationManager,
        fdr_manager: FDRManager,
        reporter: None | reporting.Pipeline | reporting.Backend = None,
    ):
        """This class serves as a base class for the search parameter optimization process, which defines the parameters used for search.

        Parameters
        ----------

        workflow: peptidecentric.PeptideCentricWorkflow
            The workflow object, which includes the calibration, calibration_optimization and FDR managers which are used as part of optimization.

        reporter: None | reporting.Pipeline | reporting.Backend
            The reporter object used to log information about the optimization process. If None, a new LogBackend object is created.

        """
        self._optimization_manager = optimization_manager
        self._calibration_manager = calibration_manager
        self._fdr_manager = fdr_manager
        self._config = config
        self._reporter = reporting.LogBackend() if reporter is None else reporter
        self._num_prev_optimizations: int = 0

    @abstractmethod
    def step(self, precursors_df: pd.DataFrame, fragments_df: pd.DataFrame):
        """This method evaluates the progress of the optimization, and either concludes the optimization if it has converged or continues the optimization if it has not.
        This method includes the update rule for the optimization.

        Parameters
        ----------

        precursors_df: pd.DataFrame
            The filtered precursor dataframe for the search (see peptidecentric.PeptideCentricWorkflow.filter_dfs).

        fragments_df: pd.DataFrame
            The filtered fragment dataframe for the search (see peptidecentric.PeptideCentricWorkflow.filter_dfs).


        """

    @abstractmethod
    def skip(self):
        """Record skipping of optimization. Can be overwritten with an empty method if there is no need to record skips."""

    def proceed_with_insufficient_precursors(self, precursors_df, fragments_df):
        self._reporter.log_string(
            "No more batches to process. Will proceed to extraction using best parameters available in optimization manager.",
            verbosity="warning",
        )
        self._update_history(precursors_df, fragments_df)
        self._update_workflow()

        self._reporter.log_string(
            f"Using current optimal value for {self.parameter_name}: {self._optimization_manager.__dict__[self.parameter_name]:.2f}.",
            verbosity="warning",
        )

    @abstractmethod
    def plot(self):
        """Plots the progress of the optimization. Can be overwritten with an empty method if there is no need to plot the progress."""

    @abstractmethod
    def _update_workflow(self):
        """This method updates the optimization manager with the results of the optimization, namely:
        the classifier version,
        the optimal parameter,
        score cutoff,
        FWHM_RT,
        and FWHM_mobility

        """

    @abstractmethod
    def _update_history(self, precursors_df: pd.DataFrame, fragments_df: pd.DataFrame):
        """This method updates the history dataframe with relevant values.

        Parameters
        ----------
        precursors_df: pd.DataFrame
            The filtered precursor dataframe for the search.

        fragments_df: pd.DataFrame
            The filtered fragment dataframe for the search.

        """
