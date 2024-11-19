# native imports
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np

# third party imports
import pandas as pd
import seaborn as sns

# alpha family imports
from alphabase.peptide.fragment import remove_unused_fragments
from alphabase.spectral_library.flat import SpecLibFlat

from alphadia.exceptions import NoOptimizationLockTargetError

# alphadia imports
from alphadia.workflow import reporting


class BaseOptimizer(ABC):
    def __init__(
        self,
        workflow,
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
        self.workflow = workflow
        self.reporter = reporting.LogBackend() if reporter is None else reporter
        self._num_prev_optimizations = 0

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
        self.workflow.reporter.log_string(
            "No more batches to process. Will proceed to extraction using best parameters available in optimization manager.",
            verbosity="warning",
        )
        self._update_history(precursors_df, fragments_df)
        self._update_workflow()

        self.workflow.reporter.log_string(
            f"Using current optimal value for {self.parameter_name}: {self.workflow.optimization_manager.__dict__[self.parameter_name]:.2f}.",
            verbosity="warning",
        )

    @abstractmethod
    def plot(self):
        """Plots the progress of the optimization. Can be overwritten with an empty method if there is no need to plot the progress."""

    @abstractmethod
    def _update_workflow():
        """This method updates the optimization manager with the results of the optimization, namely:
        the classifier version,
        the optimal parameter,
        score cutoff,
        FWHM_RT,
        and FWHM_mobility

        """

    @abstractmethod
    def _update_history():
        """This method updates the history dataframe with relevant values.

        Parameters
        ----------
        precursors_df: pd.DataFrame
            The filtered precursor dataframe for the search.

        fragments_df: pd.DataFrame
            The filtered fragment dataframe for the search.

        """


class AutomaticOptimizer(BaseOptimizer):
    def __init__(
        self,
        initial_parameter: float,
        workflow,
        reporter: None | reporting.Pipeline | reporting.Backend = None,
    ):
        """This class automatically optimizes the search parameter and stores the progres of optimization in a dataframe, history_df.

        Parameters
        ----------
        initial_parameter: float
            The parameter used for search in the first round of optimization.

        See base class for other parameters.

        """
        super().__init__(workflow, reporter)
        self.history_df = pd.DataFrame()
        self.workflow.optimization_manager.fit({self.parameter_name: initial_parameter})
        self.has_converged = False
        self._num_prev_optimizations = 0
        self._num_consecutive_skips = 0
        self.update_factor = workflow.config["optimization"][self.parameter_name][
            "automatic_update_factor"
        ]
        self.update_percentile_range = workflow.config["optimization"][
            self.parameter_name
        ]["automatic_update_percentile_range"]

        self._try_narrower_values = workflow.config["optimization"][
            self.parameter_name
        ]["try_narrower_values"]

        self._maximal_decrease = (
            workflow.config["optimization"][self.parameter_name]["maximal_decrease"]
            if self._try_narrower_values
            else None
        )

        self._favour_narrower_optimum = workflow.config["optimization"][
            self.parameter_name
        ]["favour_narrower_optimum"]

        self._maximum_decrease_from_maximum = (
            workflow.config["optimization"][self.parameter_name][
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
            self.reporter.log_string(
                f"✅ {self.parameter_name:<15}: optimization complete. Optimal parameter {self.workflow.optimization_manager.__dict__[self.parameter_name]} found after {len(self.history_df)} searches.",
                verbosity="progress",
            )
            return

        self._num_consecutive_skips = 0
        self._num_prev_optimizations += 1
        self.reporter.log_string(
            f"=== Optimization of {self.parameter_name} has been performed {self._num_prev_optimizations} time(s); minimum number is {self.workflow.config['calibration']['min_steps']} ===",
            verbosity="progress",
        )

        self._update_history(precursors_df, fragments_df)

        if self._just_converged:
            self.has_converged = True

            self._update_workflow()

            self.reporter.log_string(
                f"✅ {self.parameter_name:<15}: optimization complete. Optimal parameter {self.workflow.optimization_manager.__dict__[self.parameter_name]:.4f} found after {len(self.history_df)} searches.",
                verbosity="progress",
            )

        else:
            new_parameter = self._propose_new_parameter(
                precursors_df
                if self.estimator_group_name == "precursor"
                else fragments_df
            )

            self.workflow.optimization_manager.fit({self.parameter_name: new_parameter})

            self.reporter.log_string(
                f"❌ {self.parameter_name:<15}: optimization incomplete after {len(self.history_df)} search(es). Will search with parameter {self.workflow.optimization_manager.__dict__[self.parameter_name]:.4f}.",
                verbosity="progress",
            )

    def skip(self):
        """Increments the internal counter for the number of consecutive skips and checks if the optimization should be stopped."""
        self._num_consecutive_skips += 1
        self.reporter.log_string(
            f"=== Optimization of {self.parameter_name} has been skipped {self._num_consecutive_skips} time(s); maximum number is {self.workflow.config['calibration']['max_skips']} ===",
            verbosity="progress",
        )
        if self._batch_substantially_bigger:
            self.has_converged = True
            self._update_workflow()
            self.reporter.log_string(
                f"✅ {self.parameter_name:<15}: optimization complete. Optimal parameter {self.workflow.optimization_manager.__dict__[self.parameter_name]:.4f} found after {len(self.history_df)} searches.",
                verbosity="progress",
            )

    def plot(self):
        """Plot the value of the feature used to assess optimization progress against the parameter value, for each value tested."""
        fig, ax = plt.subplots()

        ax.vlines(
            x=self.workflow.optimization_manager.__dict__[self.parameter_name],
            ymin=0,
            ymax=self.history_df.loc[self._find_index_of_optimum(), self.feature_name],
            color="red",
            zorder=0,
            label=f"Optimal {self.parameter_name}",
        )

        sns.lineplot(
            data=self.history_df,
            x="parameter",
            y=self.feature_name,
            ax=ax,
        )
        sns.scatterplot(
            data=self.history_df,
            x="parameter",
            y=self.feature_name,
            ax=ax,
        )

        ax.set_xlabel(self.parameter_name)
        ax.xaxis.set_inverted(True)
        ax.set_ylim(bottom=0, top=self.history_df[self.feature_name].max() * 1.1)
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
        return self.update_factor * self.workflow.calibration_manager.get_estimator(
            self.estimator_group_name, self.estimator_name
        ).ci(df, self.update_percentile_range)

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
                    "parameter": self.workflow.optimization_manager.__dict__[
                        self.parameter_name
                    ],
                    self.feature_name: self._get_feature_value(
                        precursors_df, fragments_df
                    ),
                    "classifier_version": self.workflow.fdr_manager.current_version,
                    "score_cutoff": self.workflow.optimization_manager.score_cutoff,
                    "fwhm_rt": self.workflow.optimization_manager.fwhm_rt,
                    "fwhm_mobility": self.workflow.optimization_manager.fwhm_mobility,
                    "batch_idx": self.workflow.optlock.batch_idx,
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
            self._num_prev_optimizations
            >= self.workflow.config["calibration"]["min_steps"]
        )
        max_skips_reached = (
            self._num_consecutive_skips
            > self.workflow.config["calibration"]["max_skips"]
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

        feature_history = self.history_df[self.feature_name]
        last_feature_value = feature_history.iloc[-1]
        second_last_feature_value = feature_history.iloc[-2]
        third_last_feature_value = feature_history.iloc[-3]

        if self._try_narrower_values:  # This setting can be useful for optimizing parameters for which many parameter values have similar feature values.
            min_steps_reached = (
                self._num_prev_optimizations
                >= self.workflow.config["calibration"]["min_steps"]
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
                self._num_prev_optimizations
                >= self.workflow.config["calibration"]["min_steps"]
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
            maximum_feature_value = self.history_df[self.feature_name].max()
            threshold = (
                maximum_feature_value
                - self._maximum_decrease_from_maximum * np.abs(maximum_feature_value)
            )

            rows_within_thresh_of_max = self.history_df[
                self.history_df[self.feature_name] > threshold
            ]

            if rows_within_thresh_of_max.empty:
                # If no rows meet the threshold, return the index of the max feature value
                return self.history_df[self.feature_name].idxmax()
            else:
                return rows_within_thresh_of_max["parameter"].idxmin()

        else:
            return self.history_df[self.feature_name].idxmax()

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
        self.workflow.optimization_manager.fit({self.parameter_name: optimal_parameter})

        classifier_version_at_optimum = self.history_df["classifier_version"].loc[
            index_of_optimum
        ]
        self.workflow.optimization_manager.fit(
            {"classifier_version": classifier_version_at_optimum}
        )

        score_cutoff_at_optimum = self.history_df["score_cutoff"].loc[index_of_optimum]
        self.workflow.optimization_manager.fit(
            {"score_cutoff": score_cutoff_at_optimum}
        )

        fwhm_rt_at_optimum = self.history_df["fwhm_rt"].loc[index_of_optimum]
        self.workflow.optimization_manager.fit({"fwhm_rt": fwhm_rt_at_optimum})

        fwhm_mobility_at_optimum = self.history_df["fwhm_mobility"].loc[
            index_of_optimum
        ]
        self.workflow.optimization_manager.fit(
            {"fwhm_mobility": fwhm_mobility_at_optimum}
        )

        batch_index_at_optimum = self.history_df["batch_idx"].loc[index_of_optimum]
        # Take the batch index of the optimum, at the cost of potentially getting the batch library twice if this is the same as the current batch index.
        # The time impact of this is negligible and the benefits can be significant.
        self.workflow.optlock.batch_idx = batch_index_at_optimum

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


class TargetedOptimizer(BaseOptimizer):
    def __init__(
        self,
        initial_parameter: float,
        target_parameter: float,
        workflow,
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
        super().__init__(workflow, reporter)
        self.workflow.optimization_manager.fit({self.parameter_name: initial_parameter})
        self.target_parameter = target_parameter
        self.update_factor = workflow.config["optimization"][self.parameter_name][
            "targeted_update_factor"
        ]
        self.update_percentile_range = workflow.config["optimization"][
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
            self._num_prev_optimizations
            >= self.workflow.config["calibration"]["min_steps"]
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
            self.workflow.calibration_manager.get_estimator(
                self.estimator_group_name, self.estimator_name
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
            self.reporter.log_string(
                f"✅ {self.parameter_name:<15}: {self.workflow.optimization_manager.__dict__[self.parameter_name]:.4f} <= {self.target_parameter:.4f}",
                verbosity="progress",
            )
            return
        self._num_prev_optimizations += 1
        new_parameter = self._propose_new_parameter(
            precursors_df if self.estimator_group_name == "precursor" else fragments_df
        )
        just_converged = self._check_convergence(new_parameter)
        self.workflow.optimization_manager.fit({self.parameter_name: new_parameter})
        self.workflow.optimization_manager.fit(
            {"classifier_version": self.workflow.fdr_manager.current_version}
        )

        if just_converged:
            self.has_converged = True
            self.reporter.log_string(
                f"✅ {self.parameter_name:<15}: {self.workflow.optimization_manager.__dict__[self.parameter_name]:.4f} <= {self.target_parameter:.4f}",
                verbosity="progress",
            )

        else:
            self.reporter.log_string(
                f"❌ {self.parameter_name:<15}: {self.workflow.optimization_manager.__dict__[self.parameter_name]:.4f} > {self.target_parameter:.4f} or insufficient steps taken.",
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


class AutomaticRTOptimizer(AutomaticOptimizer):
    def __init__(
        self,
        initial_parameter: float,
        workflow,
        reporter: None | reporting.Pipeline | reporting.Backend = None,
    ):
        """See base class. Optimizes retention time error."""
        self.parameter_name = "rt_error"
        self.estimator_group_name = "precursor"
        self.estimator_name = "rt"
        self.feature_name = "precursor_proportion_detected"
        super().__init__(initial_parameter, workflow, reporter)

    def _get_feature_value(
        self, precursors_df: pd.DataFrame, fragments_df: pd.DataFrame
    ):
        return len(precursors_df) / self.workflow.optlock.total_elution_groups


class AutomaticMS2Optimizer(AutomaticOptimizer):
    def __init__(
        self,
        initial_parameter: float,
        workflow,
        reporter: None | reporting.Pipeline | reporting.Backend = None,
    ):
        """See base class. This class automatically optimizes the MS2 tolerance parameter by tracking the number of precursor identifications and stopping when further changes do not increase this number."""
        self.parameter_name = "ms2_error"
        self.estimator_group_name = "fragment"
        self.estimator_name = "mz"
        self.feature_name = "precursor_proportion_detected"
        super().__init__(initial_parameter, workflow, reporter)

    def _get_feature_value(
        self, precursors_df: pd.DataFrame, fragments_df: pd.DataFrame
    ):
        return len(precursors_df) / self.workflow.optlock.total_elution_groups


class AutomaticMS1Optimizer(AutomaticOptimizer):
    def __init__(
        self,
        initial_parameter: float,
        workflow,
        reporter: None | reporting.Pipeline | reporting.Backend = None,
    ):
        """See base class. Optimizes MS1 error."""
        self.parameter_name = "ms1_error"
        self.estimator_group_name = "precursor"
        self.estimator_name = "mz"
        self.feature_name = "mean_isotope_intensity_correlation"
        super().__init__(initial_parameter, workflow, reporter)

    def _get_feature_value(
        self, precursors_df: pd.DataFrame, fragments_df: pd.DataFrame
    ):
        return precursors_df.isotope_intensity_correlation.mean()


class AutomaticMobilityOptimizer(AutomaticOptimizer):
    def __init__(
        self,
        initial_parameter: float,
        workflow,
        reporter: None | reporting.Pipeline | reporting.Backend = None,
    ):
        """See base class. Optimizes mobility error."""
        self.parameter_name = "mobility_error"
        self.estimator_group_name = "precursor"
        self.estimator_name = "mobility"
        self.feature_name = "precursor_proportion_detected"
        super().__init__(initial_parameter, workflow, reporter)

    def _get_feature_value(
        self, precursors_df: pd.DataFrame, fragments_df: pd.DataFrame
    ):
        return len(precursors_df) / self.workflow.optlock.total_elution_groups


class TargetedRTOptimizer(TargetedOptimizer):
    def __init__(
        self,
        initial_parameter: float,
        target_parameter: float,
        workflow,
        reporter: None | reporting.Pipeline | reporting.Backend = None,
    ):
        """See base class."""
        self.parameter_name = "rt_error"
        self.estimator_group_name = "precursor"
        self.estimator_name = "rt"
        super().__init__(initial_parameter, target_parameter, workflow, reporter)


class TargetedMS2Optimizer(TargetedOptimizer):
    def __init__(
        self,
        initial_parameter: float,
        target_parameter: float,
        workflow,
        reporter: None | reporting.Pipeline | reporting.Backend = None,
    ):
        """See base class."""
        self.parameter_name = "ms2_error"
        self.estimator_group_name = "fragment"
        self.estimator_name = "mz"
        super().__init__(initial_parameter, target_parameter, workflow, reporter)


class TargetedMS1Optimizer(TargetedOptimizer):
    def __init__(
        self,
        initial_parameter: float,
        target_parameter: float,
        workflow,
        reporter: None | reporting.Pipeline | reporting.Backend = None,
    ):
        """See base class."""
        self.parameter_name = "ms1_error"
        self.estimator_group_name = "precursor"
        self.estimator_name = "mz"
        super().__init__(initial_parameter, target_parameter, workflow, reporter)


class TargetedMobilityOptimizer(TargetedOptimizer):
    def __init__(
        self,
        initial_parameter: float,
        target_parameter: float,
        workflow,
        reporter: None | reporting.Pipeline | reporting.Backend = None,
    ):
        """See base class."""
        self.parameter_name = "mobility_error"
        self.estimator_group_name = "precursor"
        self.estimator_name = "mobility"
        super().__init__(initial_parameter, target_parameter, workflow, reporter)


class OptimizationLock:
    def __init__(self, library: SpecLibFlat, config: dict):
        """Sets and updates the optimization lock, which is the data used for calibration and optimization of the search parameters.

        Parameters
        ----------
        library: alphabase.spectral_library.flat.SpecLibFlat
            The library object from the PeptideCentricWorkflow object, which includes the precursor and fragment library dataframes.

        config: dict
            The configuration dictionary from the PeptideCentricWorkflow object.
        """
        self._library = library
        self._config = config

        self.previously_calibrated = False
        self.has_target_num_precursors = False

        self.elution_group_order = library._precursor_df["elution_group_idx"].unique()
        rng = np.random.default_rng(seed=772)
        rng.shuffle(self.elution_group_order)

        self._precursor_target_count = self._config["calibration"][
            "optimization_lock_target"
        ]

        self.batch_idx = 0
        self.set_batch_plan()

        eg_idxes = self.elution_group_order[self.start_idx : self.stop_idx]
        self.set_batch_dfs(eg_idxes)

        self.feature_dfs = []
        self.fragment_dfs = []

    @property
    def features_df(self) -> pd.DataFrame:
        return pd.concat(self.feature_dfs)

    @property
    def fragments_df(self) -> pd.DataFrame:
        return pd.concat(self.fragment_dfs)

    @property
    def start_idx(self) -> int:
        if self.has_target_num_precursors:
            return 0
        elif self.batch_idx >= len(self.batch_plan):
            raise NoOptimizationLockTargetError()  # This should never be triggered since introduction of the BaseOptimizer.proceed_with_insufficient_precursors method and associated code, and could be removed.
        else:
            return self.batch_plan[self.batch_idx][0]

    @property
    def stop_idx(self) -> int:
        return self.batch_plan[self.batch_idx][1]

    def _get_exponential_batches(self, step):
        """Get the number of batches for a given step
        This plan has the shape:
        1, 2, 4, 8, 16, 32, 64, ...
        """
        return int(2**step)

    def set_batch_plan(self):
        """Gets an exponential batch plan based on the batch_size value in the config."""
        n_eg = len(self.elution_group_order)

        plan = []

        batch_size = self._config["calibration"]["batch_size"]
        step = 0
        start_idx = 0

        while start_idx < n_eg:
            n_batches = self._get_exponential_batches(step)
            stop_idx = min(start_idx + n_batches * batch_size, n_eg)
            plan.append((start_idx, stop_idx))
            step += 1
            start_idx = stop_idx

        self.batch_plan = plan

    def batches_remaining(self):
        return self.batch_idx + 1 < len(self.batch_plan)

    def update_with_extraction(
        self, feature_df: pd.DataFrame, fragment_df: pd.DataFrame
    ):
        """Extract features and fragments from the current batch of the optimization lock.

        Parameters
        ----------
        feature_df: pd.DataFrame
            The feature dataframe for the current batch of the optimization lock (from workflow.extract_batch).

        fragment_df: pd.DataFrame
            The fragment dataframe for the current batch of the optimization lock (from workflow.extract_batch).
        """

        self.feature_dfs += [feature_df]
        self.fragment_dfs += [fragment_df]

        self.total_elution_groups = self.features_df.elution_group_idx.nunique()

    def update_with_fdr(self, precursor_df: pd.DataFrame):
        """Calculates the number of precursors at 1% FDR for the current optimization lock and determines if it is sufficient to perform calibration and optimization.

        Parameters
        ----------
        precursor_df: pd.DataFrame
            The precursor dataframe for the current batch of the optimization lock (from workflow.perform_fdr).
        """

        self._precursor_at_fdr_count = np.sum(
            (precursor_df["qval"] < 0.01) & (precursor_df["decoy"] == 0)
        )
        self.has_target_num_precursors = (
            self._precursor_at_fdr_count >= self._precursor_target_count
        )

    def update_with_calibration(self, calibration_manager):
        """Updates the batch library with the current calibrated values using the calibration manager.

        Parameters
        ----------
        calibration_manager: manager.CalibrationManager
            The calibration manager object from the PeptideCentricWorkflow object.

        """
        calibration_manager.predict(
            self.batch_library._precursor_df,
            "precursor",
        )

        calibration_manager.predict(
            self.batch_library._fragment_df,
            "fragment",
        )

    def increase_batch_idx(self):
        """If the optimization lock does not contain enough precursors at 1% FDR, the optimization lock proceeds to include the next step in the batch plan in the library attribute.
        This is done by incrementing self.batch_idx.
        """
        self.batch_idx += 1

    def decrease_batch_idx(self):
        """If the optimization lock contains enough precursors at 1% FDR, checks if enough precursors can be obtained using a smaller library and updates the library attribute accordingly.
        If not, the same library is used as before.
        This is done by taking the smallest step in the batch plan which gives more precursors than the target number of precursors.
        """

        batch_plan_diff = np.array(
            [
                stop_at_given_idx
                - self.stop_idx
                * self._precursor_target_count
                / self._precursor_at_fdr_count
                for _, stop_at_given_idx in self.batch_plan
            ]
        )  # Calculate the difference between the number of precursors expected at the given idx and the target number of precursors for each idx in the batch plan.
        # get index of smallest value >= 0
        self.batch_idx = np.where(batch_plan_diff >= 0)[0][0]

    def update(self):
        """Updates the library to use for the next round of optimization, either adjusting it upwards or downwards depending on whether the target has been reached.
        If the target has been reached, the feature and fragment dataframes are reset
        """
        if self.has_target_num_precursors:
            self.decrease_batch_idx()
            self.feature_dfs = []
            self.fragment_dfs = []

        else:
            self.increase_batch_idx()

        eg_idxes = self.elution_group_order[self.start_idx : self.stop_idx]
        self.set_batch_dfs(eg_idxes)

    def reset_after_convergence(self, calibration_manager):
        """Resets the optimization lock after all optimizers in a given round of optimization have converged.

        Parameter
        ---------
        calibration_manager: manager.CalibrationManager
            The calibration manager object from the PeptideCentricWorkflow object.

        """
        self.has_target_num_precursors = True
        self.feature_dfs = []
        self.fragment_dfs = []
        self.set_batch_dfs()
        self.update_with_calibration(calibration_manager)

    def set_batch_dfs(self, eg_idxes: None | np.ndarray = None):
        """
        Sets the batch library to use for the next round of optimization, either adjusting it upwards or downwards depending on whether the target has been reached.

        Parameters
        ----------
        eg_idxes: None | np.ndarray
            The elution group indexes to use for the next round of optimization. If None, the eg_idxes for the current self.start_idx and self.stop_idx are used.
        """
        if eg_idxes is None:
            eg_idxes = self.elution_group_order[self.start_idx : self.stop_idx]
        self.batch_library = SpecLibFlat()
        self.batch_library._precursor_df, (self.batch_library._fragment_df,) = (
            remove_unused_fragments(
                self._library._precursor_df[
                    self._library._precursor_df["elution_group_idx"].isin(eg_idxes)
                ],
                (self._library._fragment_df,),
                frag_start_col="flat_frag_start_idx",
                frag_stop_col="flat_frag_stop_idx",
            )
        )
