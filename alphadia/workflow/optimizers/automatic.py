from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

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
from alphadia.workflow.optimizers.optimization_lock import OptimizationLock


class AutomaticOptimizer(BaseOptimizer, ABC):
    def __init__(
        self,
        initial_parameter: float,
        config: Config,
        optimization_manager: OptimizationManager,
        calibration_manager: CalibrationManager,
        fdr_manager: FDRManager,
        optlock: OptimizationLock,
        reporter: None | reporting.Pipeline | reporting.Backend = None,
    ):
        """This class automatically optimizes the search parameter and stores the progres of optimization in a dataframe, history_df.

        Parameters
        ----------
        initial_parameter: float
            The parameter used for search in the first round of optimization.

        See base class for other parameters.

        """
        super().__init__(
            config, optimization_manager, calibration_manager, fdr_manager, reporter
        )

        self._optlock = optlock

        self.history_df = pd.DataFrame()

        self._optimization_manager.update(**{self.parameter_name: initial_parameter})
        self.has_converged = False
        self._num_prev_optimizations = 0
        self._num_consecutive_skips = 0
        self.update_factor = self._config["optimization"][self.parameter_name][
            "automatic_update_factor"
        ]
        self.update_percentile_range = self._config["optimization"][
            self.parameter_name
        ]["automatic_update_percentile_range"]

        self._try_narrower_values = self._config["optimization"][self.parameter_name][
            "try_narrower_values"
        ]

        self._maximal_decrease = (
            self._config["optimization"][self.parameter_name]["maximal_decrease"]
            if self._try_narrower_values
            else None
        )

        self._favour_narrower_optimum = self._config["optimization"][
            self.parameter_name
        ]["favour_narrower_optimum"]

        self._maximum_decrease_from_maximum = (
            self._config["optimization"][self.parameter_name][
                "maximum_decrease_from_maximum"
            ]
            if self._favour_narrower_optimum
            else None
        )

    def step(
        self,
        precursors_df: pd.DataFrame,
        fragments_df: pd.DataFrame,
    ):
        """See base class. The feature is used to track the progres of the optimization and determine whether it has converged.
        It also resets the internal counter for the number of consecutive skips.
        """
        if self.has_converged:
            self._reporter.log_string(
                f"✅ {self.parameter_name:<15}: optimization already complete. Optimal parameter {self._optimization_manager.__dict__[self.parameter_name]} found after {len(self.history_df)} searches.",
                verbosity="progress",
            )
            return

        self._num_consecutive_skips = 0
        self._num_prev_optimizations += 1
        self._reporter.log_string(
            f"=== Optimization of {self.parameter_name} has been performed {self._num_prev_optimizations} time(s); minimum number is {self._config['calibration']['min_steps']} ===",
            verbosity="progress",
        )

        self._update_history(precursors_df, fragments_df)

        if self._just_converged:
            self.has_converged = True

            self._update_workflow()

            self._reporter.log_string(
                f"✅ {self.parameter_name:<15}: optimization just completed. Optimal parameter {self._optimization_manager.__dict__[self.parameter_name]:.4f} found after {len(self.history_df)} searches.",
                verbosity="progress",
            )

        else:
            new_parameter = self._propose_new_parameter(
                precursors_df
                if self._estimator_group_name == CalibrationGroups.PRECURSOR
                else fragments_df
            )

            self._optimization_manager.update(**{self.parameter_name: new_parameter})

            self._reporter.log_string(
                f"❌ {self.parameter_name:<15}: optimization incomplete after {len(self.history_df)} search(es). Will search with parameter {self._optimization_manager.__dict__[self.parameter_name]:.4f}.",
                verbosity="progress",
            )

    def skip(self):
        """Increments the internal counter for the number of consecutive skips and checks if the optimization should be stopped."""
        self._num_consecutive_skips += 1
        self._reporter.log_string(
            f"=== Optimization of {self.parameter_name} has been skipped {self._num_consecutive_skips} time(s); maximum number is {self._config['calibration']['max_skips']} ===",
            verbosity="progress",
        )
        if self._batch_substantially_bigger:
            self.has_converged = True
            self._update_workflow()
            self._reporter.log_string(
                f"✅ {self.parameter_name:<15}: optimization complete (batch_substantially_bigger). Optimal parameter {self._optimization_manager.__dict__[self.parameter_name]:.4f} found after {len(self.history_df)} searches.",
                verbosity="progress",
            )

    def plot(self):
        """Plot the value of the feature used to assess optimization progress against the parameter value, for each value tested."""
        fig, ax = plt.subplots()

        ax.vlines(
            x=self._optimization_manager.__dict__[self.parameter_name],
            ymin=0,
            ymax=self.history_df.loc[self._find_index_of_optimum(), self._feature_name],
            color="red",
            zorder=0,
            label=f"Optimal {self.parameter_name}",
        )

        sns.lineplot(
            data=self.history_df,
            x="parameter",
            y=self._feature_name,
            ax=ax,
        )
        sns.scatterplot(
            data=self.history_df,
            x="parameter",
            y=self._feature_name,
            ax=ax,
        )

        ax.set_xlabel(self.parameter_name)
        ax.xaxis.set_inverted(True)
        ax.set_ylim(bottom=0, top=self.history_df[self._feature_name].max() * 1.1)
        ax.legend(loc="upper left")

        plt.show()

    def _propose_new_parameter(self, df: pd.DataFrame):
        """This method specifies the rule according to which the search parameter is updated between rounds of optimization. The update rule is
            1) calculate the deviation of the predicted mz values from the observed mz values,
            2) take the mean of the endpoints of the central interval
                (determined by the self.update_percentile_range attribute, which determines the percentile taken expressed as a decimal) of these deviations, and
            3) multiply this value by self.update_factor.
        This is implemented by the ci method for the estimator.

        Parameters
        ----------

        df: pd.DataFrame
            The dataframe used to update the parameter. This could be the precursor or fragment dataframe, depending on the search parameter being optimized.

        Returns
        -------
        float
            The proposed new value for the search parameter.

        """
        return (
            self.update_factor
            * self._calibration_manager.get_estimator(  # TODO save only estimators?
                self._estimator_group_name, self._estimator_name
            ).ci(df, self.update_percentile_range)
        )

    def _update_history(self, precursors_df: pd.DataFrame, fragments_df: pd.DataFrame):
        """This method updates the history dataframe with relevant values.

        Parameters
        ----------
        precursors_df: pd.DataFrame
            The filtered precursor dataframe for the search.

        fragments_df: pd.DataFrame
            The filtered fragment dataframe for the search.

        """
        new_row = pd.DataFrame(
            [
                {
                    "parameter": self._optimization_manager.__dict__[
                        self.parameter_name
                    ],
                    self._feature_name: self._get_feature_value(
                        precursors_df, fragments_df
                    ),
                    "classifier_version": self._fdr_manager.current_version,  # TODO: only we need from fdr_manager
                    "score_cutoff": self._optimization_manager.score_cutoff,
                    "fwhm_rt": self._optimization_manager.fwhm_rt,
                    "fwhm_mobility": self._optimization_manager.fwhm_mobility,
                    "batch_idx": self._optlock.batch_idx,
                }
            ]
        )
        self.history_df = pd.concat([self.history_df, new_row], ignore_index=True)

    @property
    def _batch_substantially_bigger(self):
        """This function checks if the optimization has already been optimized sufficiently many times and if it has been skipped too many times at the current parameter value.
        (Being skipped indicates that the current parameter proposal significantly reduces the number of identified precursors and is unlikely to be optimal.)

        Returns
        -------
        bool
            True if the optimization has already been performed the minimum number of times and the maximum number of skips has been reached, False otherwise.

        """
        min_steps_reached = (
            self._num_prev_optimizations >= self._config["calibration"]["min_steps"]
        )
        max_skips_reached = (
            self._num_consecutive_skips > self._config["calibration"]["max_skips"]
        )
        return min_steps_reached and max_skips_reached

    @property
    def _just_converged(self):
        """Optimization should stop if continued narrowing of the parameter is not improving the feature value.
        If self._try_narrower_values is False:
            1) This function checks if the previous rounds of optimization have led to a meaningful improvement in the feature value.
            2) If so, it continues optimization and appends the proposed new parameter to the list of parameters. If not, it stops optimization and sets the optimal parameter attribute.
        If self._try_narrower_values is True:
            1) This function checks if the previous rounds of optimization have led to a meaningful disimprovement in the feature value or if the parameter has not changed substantially.
            2) If not, it continues optimization and appends the proposed new parameter to the list of parameters. If so, it stops optimization and sets the optimal parameter attribute.

        Notes
        -----
            Because the check for an increase in feature value requires two previous rounds, the function will also initialize for another round of optimization if there have been fewer than 3 rounds.
            This function may be overwritten in child classes.

        """
        if len(self.history_df) < 3:
            return False

        feature_history = self.history_df[self._feature_name]
        last_feature_value = feature_history.iloc[-1]
        second_last_feature_value = feature_history.iloc[-2]
        third_last_feature_value = feature_history.iloc[-3]

        if self._try_narrower_values:  # This setting can be useful for optimizing parameters for which many parameter values have similar feature values.
            min_steps_reached = (
                self._num_prev_optimizations >= self._config["calibration"]["min_steps"]
            )

            feature_substantially_decreased = (
                last_feature_value - second_last_feature_value
            ) / np.abs(second_last_feature_value) < -self._maximal_decrease and (
                last_feature_value - third_last_feature_value
            ) / np.abs(third_last_feature_value) < -self._maximal_decrease

            parameter_history = self.history_df["parameter"]

            last_parameter_value = parameter_history.iloc[-1]
            second_last_parameter_value = parameter_history.iloc[-2]
            parameter_not_substantially_changed = (
                np.abs(
                    (last_parameter_value - second_last_parameter_value)
                    / second_last_parameter_value
                )
                < 0.05
            )

            return min_steps_reached and (
                feature_substantially_decreased or parameter_not_substantially_changed
            )

        else:
            min_steps_reached = (
                self._num_prev_optimizations >= self._config["calibration"]["min_steps"]
            )

            feature_not_substantially_increased = (
                last_feature_value - second_last_feature_value
            ) / np.abs(second_last_feature_value) < 0.1 and (
                last_feature_value - third_last_feature_value
            ) / np.abs(third_last_feature_value) < 0.1

            return min_steps_reached and feature_not_substantially_increased

    def _find_index_of_optimum(self) -> int:
        """Finds the index of the row in the history dataframe with the optimal value of the feature used for optimization.
        if self._favour_narrower_parameter is False:
            The index at optimum is the index of the parameter value that maximizes the feature.
        if self._favour_narrower_parameter is True:
            The index at optimum is the index of the minimal parameter value whose feature value is at least self._maximum_decrease_from_maximum of the maximum value of the feature.

        Returns
        -------
        int
            The index of the row in the history dataframe with the optimal value of the feature used for optimization.
        Notes
        -----
            This method may be overwritten in child classes.

        """

        if len(self.history_df) == 0:
            raise ValueError(f"Optimizer: {self.parameter_name} has no history.")

        if len(self.history_df) == 1:
            # If there's only one row, return its index
            return self.history_df.index[0]

        if self._favour_narrower_optimum:  # This setting can be useful for optimizing parameters for which many parameter values have similar feature values.
            maximum_feature_value = self.history_df[self._feature_name].max()
            threshold = (
                maximum_feature_value
                - self._maximum_decrease_from_maximum * np.abs(maximum_feature_value)
            )

            rows_within_thresh_of_max = self.history_df[
                self.history_df[self._feature_name] > threshold
            ]

            if rows_within_thresh_of_max.empty:
                # If no rows meet the threshold, return the index of the max feature value
                return self.history_df[self._feature_name].idxmax()
            else:
                return rows_within_thresh_of_max["parameter"].idxmin()

        else:
            return self.history_df[self._feature_name].idxmax()

    def _update_workflow(self):
        """Updates the optimization manager with the results of the optimization, namely:
            the classifier version,
            the optimal parameter,
            score cutoff,
            FWHM_RT,
            and FWHM_mobility
        at the optimal parameter. Also updates the optlock with the batch index at the optimum.

        """
        index_of_optimum = self._find_index_of_optimum()

        optimal_parameter = self.history_df["parameter"].loc[index_of_optimum]
        self._optimization_manager.update(**{self.parameter_name: optimal_parameter})

        classifier_version_at_optimum = self.history_df["classifier_version"].loc[
            index_of_optimum
        ]
        self._optimization_manager.update(
            classifier_version=classifier_version_at_optimum
        )

        score_cutoff_at_optimum = self.history_df["score_cutoff"].loc[index_of_optimum]
        self._optimization_manager.update(score_cutoff=score_cutoff_at_optimum)

        fwhm_rt_at_optimum = self.history_df["fwhm_rt"].loc[index_of_optimum]
        self._optimization_manager.update(fwhm_rt=fwhm_rt_at_optimum)

        fwhm_mobility_at_optimum = self.history_df["fwhm_mobility"].loc[
            index_of_optimum
        ]
        self._optimization_manager.update(fwhm_mobility=fwhm_mobility_at_optimum)

        batch_index_at_optimum = self.history_df["batch_idx"].loc[index_of_optimum]
        # Take the batch index of the optimum, at the cost of potentially getting the batch library twice if this is the same as the current batch index.
        # The time impact of this is negligible and the benefits can be significant.
        self._optlock.batch_idx = batch_index_at_optimum

    @abstractmethod
    def _get_feature_value(
        self, precursors_df: pd.DataFrame, fragments_df: pd.DataFrame
    ):
        """Each parameter is optimized according to a particular feature. This method gets the value of that feature for a given round of optimization.

        Parameters
        ----------

        precursors_df: pd.DataFrame
            The precursor dataframe for the search

        fragments_df: pd.DataFrame
            The fragment dataframe for the search


        """


class AutomaticRTOptimizer(AutomaticOptimizer):
    def __init__(
        self,
        initial_parameter: float,
        config: Config,
        optimization_manager: OptimizationManager,
        calibration_manager: CalibrationManager,
        fdr_manager: FDRManager,
        optlock: OptimizationLock,
        reporter: None | reporting.Pipeline | reporting.Backend = None,
    ):
        """See base class. Optimizes retention time error."""
        self.parameter_name = "rt_error"
        self._estimator_group_name = CalibrationGroups.PRECURSOR
        self._estimator_name = CalibrationEstimators.RT
        self._feature_name = "precursor_proportion_detected"
        super().__init__(
            initial_parameter,
            config,
            optimization_manager,
            calibration_manager,
            fdr_manager,
            optlock,
            reporter,
        )

    def _get_feature_value(
        self, precursors_df: pd.DataFrame, fragments_df: pd.DataFrame
    ):
        return len(precursors_df) / self._optlock.total_elution_groups


class AutomaticMS2Optimizer(AutomaticOptimizer):
    def __init__(
        self,
        initial_parameter: float,
        config: Config,
        optimization_manager: OptimizationManager,
        calibration_manager: CalibrationManager,
        fdr_manager: FDRManager,
        optlock: OptimizationLock,
        reporter: None | reporting.Pipeline | reporting.Backend = None,
    ):
        """See base class. This class automatically optimizes the MS2 tolerance parameter by tracking the number of precursor identifications and stopping when further changes do not increase this number."""
        self.parameter_name = "ms2_error"
        self._estimator_group_name = CalibrationGroups.FRAGMENT
        self._estimator_name = CalibrationEstimators.MZ
        self._feature_name = "precursor_proportion_detected"
        super().__init__(
            initial_parameter,
            config,
            optimization_manager,
            calibration_manager,
            fdr_manager,
            optlock,
            reporter,
        )

    def _get_feature_value(
        self, precursors_df: pd.DataFrame, fragments_df: pd.DataFrame
    ):
        return len(precursors_df) / self._optlock.total_elution_groups


class AutomaticMS1Optimizer(AutomaticOptimizer):
    def __init__(
        self,
        initial_parameter: float,
        config: Config,
        optimization_manager: OptimizationManager,
        calibration_manager: CalibrationManager,
        fdr_manager: FDRManager,
        optlock: OptimizationLock,
        reporter: None | reporting.Pipeline | reporting.Backend = None,
    ):
        """See base class. Optimizes MS1 error."""
        self.parameter_name = "ms1_error"
        self._estimator_group_name = CalibrationGroups.PRECURSOR
        self._estimator_name = CalibrationEstimators.MZ
        self._feature_name = "mean_isotope_intensity_correlation"
        super().__init__(
            initial_parameter,
            config,
            optimization_manager,
            calibration_manager,
            fdr_manager,
            optlock,
            reporter,
        )

    def _get_feature_value(
        self, precursors_df: pd.DataFrame, fragments_df: pd.DataFrame
    ):
        return precursors_df.isotope_intensity_correlation.mean()


class AutomaticMobilityOptimizer(AutomaticOptimizer):
    def __init__(
        self,
        initial_parameter: float,
        config: Config,
        optimization_manager: OptimizationManager,
        calibration_manager: CalibrationManager,
        fdr_manager: FDRManager,
        optlock: OptimizationLock,
        reporter: None | reporting.Pipeline | reporting.Backend = None,
    ):
        """See base class. Optimizes mobility error."""
        self.parameter_name = "mobility_error"
        self._estimator_group_name = CalibrationGroups.PRECURSOR
        self._estimator_name = CalibrationEstimators.MOBILITY
        self._feature_name = "precursor_proportion_detected"
        super().__init__(
            initial_parameter,
            config,
            optimization_manager,
            calibration_manager,
            fdr_manager,
            optlock,
            reporter,
        )

    def _get_feature_value(
        self, precursors_df: pd.DataFrame, fragments_df: pd.DataFrame
    ):
        return len(precursors_df) / self._optlock.total_elution_groups
