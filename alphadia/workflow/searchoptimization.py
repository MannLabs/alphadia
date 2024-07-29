# native imports
from abc import ABC, abstractmethod

import numpy as np

# alpha family imports
# third party imports
import pandas as pd

# alphadia imports
from alphadia.workflow import manager, reporting


class BaseOptimizer(ABC):
    def __init__(
        self,
        calibration_manager: manager.CalibrationManager,
        optimization_manager: manager.OptimizationManager,
        fdr_manager: manager.FDRManager,
        reporter: None | reporting.Pipeline | reporting.Backend = None,
    ):
        """This class serves as a base class for organizing the search parameter optimization process, which defines the parameters used for search.

        Parameters
        ----------

        calibration_manager: manager.CalibrationManager
            The calibration manager for the workflow, which is needed to update the search parameter between rounds of optimization

        optimization_manager: manager.OptimizationManager
            The optimization manager for the workflow, which is needed so the optimal parameter and classifier version can be saved to the manager

        fdr_manager: manager.FDRManager
            The FDR manager for the workflow, which is needed to update the optimal classifier version in the optimization manager

        """
        self.optimal_parameter = None
        self.calibration_manager = calibration_manager
        self.optimization_manager = optimization_manager
        self.fdr_manager = fdr_manager
        self.reporter = reporting.LogBackend() if reporter is None else reporter

    @abstractmethod
    def step(self, precursors_df: pd.DataFrame, fragments_df: pd.DataFrame):
        """This method evaluates the progress of the optimization, and either concludes the optimization if it has converged or continues the optimization if it has not.
        This method includes the update rule for the optimization.

        Parameters
        ----------

        precursors_df: pd.DataFrame
            The precursor dataframe for the search

        fragments_df: pd.DataFrame
            The fragment dataframe for the search


        """
        pass

    @abstractmethod
    def _update_parameter(self, df):
        """This method specifies the rule according to which the search parameter is updated between rounds of optimization. The rule is specific to the parameter being optimized.

        Parameters
        ----------

        df: pd.DataFrame
            The dataframe used to update the parameter. This could be the precursor or fragment dataframe, depending on the search parameter being optimized.


        """
        pass

    @abstractmethod
    def _check_convergence(self):
        """This method checks if the optimization has converged according to parameter-specific conditions and, if it has, sets the optimal parameter attribute and updates the optimization manager."""
        pass

    def has_converged(self):
        """If the optimal parameter has been set, the optimization must have converged and the method returns True. Otherwise, it returns False."""
        return self.optimal_parameter is not None


class RTOptimizer(BaseOptimizer):
    """TODO Finish this optimizer"""

    def __init__(
        self,
        initial_parameter: float,
        calibration_manager: manager.CalibrationManager,
        optimization_manager: manager.OptimizationManager,
        fdr_manager: manager.FDRManager,
        **kwargs,
    ):
        """See base class.

        Parameters
        ----------

        initial_parameter: float
            The parameter used for search in the first round of optimization.

        """
        super().__init__(
            calibration_manager, optimization_manager, fdr_manager, **kwargs
        )
        self.parameters = [initial_parameter]
        self.feature = []

    def _check_convergence(self):
        """Optimization should stop if continued narrowing of the TODO parameter is not improving the TODO feature value.
        This function checks if the previous rounds of optimization have led to a meaningful improvement in the TODO feature value.
        If so, it continues optimization and appends the proposed new parameter to the list of parameters. If not, it stops optimization and sets the optimal parameter attribute.

        Notes
        -----
            Because the check for an increase in TODO feature value requires two previous rounds, the function will also initialize for another round of optimization if there have been fewer than 3 rounds.


        """

        if (
            len(self.feature) > 2
            and self.feature[-1] < 1.1 * self.feature[-2]
            and self.feature[-1] < 1.1 * self.feature[-3]
        ):
            self.optimal_parameter = self.parameters[np.argmax(self.feature)]

            self.optimization_manager.fit({"rt_error": self.optimal_parameter})

    def _update_parameter(self, df: pd.DataFrame):
        """See base class. The update rule is
            1) calculate the deviation of the predicted mz values from the observed mz values,
            2) take the mean of the endpoints of the central 99% of these deviations, and
            3) multiply this value by 1.1.
        This is implemented by the ci method for the estimator.


        """
        proposed_parameter = 1.1 * self.calibration_manager.get_estimator(
            "precursor", "rt"
        ).ci(df, 0.99)

        return proposed_parameter

    def step(self, precursors_df: pd.DataFrame, fragments_df: pd.DataFrame):
        """See base class. The number of precursor identifications is used to track the progres of the optimization (stored in .feature) and determine whether it has converged."""
        if not self.has_converged():
            self.feature.append(len(precursors_df))
            self._check_convergence()

        if self.has_converged():  # Note this may change from the above statement since .optimal_parameter may be set in ._check_convergence
            self.reporter.log_string(
                f"✅ RT: optimization complete. Optimal parameter {self.optimal_parameter} found after {len(self.parameters)} searches.",
                verbosity="progress",
            )

        else:
            proposed_parameter = self._update_parameter(precursors_df)

            self.reporter.log_string(
                f"❌ RT: optimization incomplete after {len(self.parameters)} search(es). Will search with parameter {proposed_parameter}.",
                verbosity="progress",
            )

            self.parameters.append(proposed_parameter)
            self.optimization_manager.fit({"rt_error": proposed_parameter})


class MS2Optimizer(BaseOptimizer):
    def __init__(
        self,
        initial_parameter: float,
        calibration_manager: manager.CalibrationManager,
        optimization_manager: manager.OptimizationManager,
        fdr_manager: manager.FDRManager,
        **kwargs,
    ):
        """See base class.

        Parameters
        ----------
        initial_parameter: float
            The parameter used for search in the first round of optimization.


        """
        super().__init__(
            calibration_manager, optimization_manager, fdr_manager, **kwargs
        )
        self.history_df = pd.DataFrame(
            columns=["parameter", "precursor_ids", "classifier_version"]
        )
        self.proposed_parameter = initial_parameter

    def _check_convergence(self):
        """Optimization should stop if continued narrowing of the MS2 parameter is not improving the number of precursor identifications.
        This function checks if the previous rounds of optimization have led to a meaningful improvement in the number of identifications.
        If so, it continues optimization and appends the proposed new parameter to the list of parameters. If not, it stops optimization and sets the optimal parameter attribute.

        Notes
        -----
            Because the check for an increase in identifications requires two previous rounds, the function will also initialize for another round of optimization if there have been fewer than 3 rounds.


        """

        if (
            len(self.history_df) > 2
            and self.history_df["precursor_ids"].iloc[-1]
            < 1.1 * self.history_df["precursor_ids"].iloc[-2]
            and self.history_df["precursor_ids"].iloc[-1]
            < 1.1 * self.history_df["precursor_ids"].iloc[-3]
        ):
            self.optimal_parameter = self.history_df.loc[
                self.history_df["precursor_ids"].idxmax(), "parameter"
            ]
            self.optimization_manager.fit({"ms2_error": self.optimal_parameter})
            self.optimization_manager.fit(
                {
                    "classifier_version": self.history_df.loc[
                        self.history_df["precursor_ids"].idxmax(), "classifier_version"
                    ]
                }
            )

    def _update_parameter(self, df: pd.DataFrame):
        """See base class. The update rule is
            1) calculate the deviation of the predicted mz values from the observed mz values,
            2) take the mean of the endpoints of the central 99% of these deviations, and
            3) multiply this value by 1.1.
        This is implemented by the ci method for the estimator.


        """
        proposed_parameter = 1.1 * self.calibration_manager.get_estimator(
            "fragment", "mz"
        ).ci(df, 0.99)

        return proposed_parameter

    def step(self, precursors_df: pd.DataFrame, fragments_df: pd.DataFrame):
        """See base class. The number of precursor identifications is used to track the progres of the optimization (stored in .precursor_ids) and determine whether it has converged."""
        if not self.has_converged():
            new_row = pd.DataFrame(
                [
                    {
                        "parameter": float(
                            self.proposed_parameter
                        ),  # Ensure float dtype
                        "precursor_ids": int(len(precursors_df)),  # Ensure int dtype
                        "classifier_version": int(
                            self.fdr_manager.current_version
                        ),  # Ensure int dtype
                    }
                ]
            )
            self.history_df = pd.concat([self.history_df, new_row], ignore_index=True)
            self._check_convergence()

        if self.has_converged():  # Note this may change from the above statement since .optimal_parameter may be set in ._check_convergence
            self.reporter.log_string(
                f"✅ MS2: optimization complete. Optimal parameter {self.optimal_parameter} found after {len(self.history_df['parameter'])} searches.",
                verbosity="progress",
            )

        else:
            self.proposed_parameter = self._update_parameter(fragments_df)

            self.reporter.log_string(
                f"❌ MS2: optimization incomplete after {len(self.history_df['parameter'])} search(es). Will search with parameter {self.proposed_parameter}.",
                verbosity="progress",
            )

            self.optimization_manager.fit({"ms2_error": self.proposed_parameter})


class MS1Optimizer(BaseOptimizer):
    """TODO Finish this optimizer"""

    def __init__(
        self,
        initial_parameter: float,
        calibration_manager: manager.CalibrationManager,
        optimization_manager: manager.OptimizationManager,
        fdr_manager: manager.FDRManager,
        **kwargs,
    ):
        """See base class.

        Parameters
        ----------

        initial_parameter: float
            The parameter used for search in the first round of optimization.

        """
        super().__init__(
            calibration_manager, optimization_manager, fdr_manager, **kwargs
        )
        self.parameters = [initial_parameter]
        self.feature = []

    def _check_convergence(self):
        """Optimization should stop if continued narrowing of the TODO parameter is not improving the TODO feature value.
        This function checks if the previous rounds of optimization have led to a meaningful improvement in the TODO feature value.
        If so, it continues optimization and appends the proposed new parameter to the list of parameters. If not, it stops optimization and sets the optimal parameter attribute.

        Notes
        -----
            Because the check for an increase in TODO feature value requires two previous rounds, the function will also initialize for another round of optimization if there have been fewer than 3 rounds.


        """

        if (
            len(self.feature) > 2
            and self.feature[-1] < 1.1 * self.feature[-2]
            and self.feature[-1] < 1.1 * self.feature[-3]
        ):
            self.optimal_parameter = self.parameters[np.argmax(self.feature)]

            self.optimization_manager.fit({"ms1_error": self.optimal_parameter})

    def _update_parameter(self, df: pd.DataFrame):
        """See base class. The update rule is
            1) calculate the deviation of the predicted mz values from the observed mz values,
            2) take the mean of the endpoints of the central 99% of these deviations, and
            3) multiply this value by 1.1.
        This is implemented by the ci method for the estimator.


        """
        proposed_parameter = 1.1 * self.calibration_manager.get_estimator(
            "precursor", "mz"
        ).ci(df, 0.99)

        return proposed_parameter

    def step(self, precursors_df: pd.DataFrame, fragments_df: pd.DataFrame):
        """See base class. The number of precursor identifications is used to track the progres of the optimization (stored in .feature) and determine whether it has converged."""
        if not self.has_converged():
            self.feature.append(len(precursors_df))
            self._check_convergence()

        if self.has_converged():  # Note this may change from the above statement since .optimal_parameter may be set in ._check_convergence
            self.reporter.log_string(
                f"✅ MS1: optimization complete. Optimal parameter {self.optimal_parameter} found after {len(self.parameters)} searches.",
                verbosity="progress",
            )

        else:
            proposed_parameter = self._update_parameter(precursors_df)

            self.reporter.log_string(
                f"❌ MS1: optimization incomplete after {len(self.parameters)} search(es). Will search with parameter {proposed_parameter}.",
                verbosity="progress",
            )

            self.parameters.append(proposed_parameter)
            self.optimization_manager.fit({"ms1_error": proposed_parameter})


class MobilityOptimizer(BaseOptimizer):
    """TODO: Implement this class. It will be used to optimize the mobility parameter for the search."""

    pass
