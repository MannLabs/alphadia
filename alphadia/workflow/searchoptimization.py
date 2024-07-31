# native imports
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt

# alpha family imports
# third party imports
import pandas as pd
import seaborn as sns

# alphadia imports
from alphadia.workflow import reporting


class BaseOptimizer(ABC):
    def __init__(
        self,
        workflow,
        reporter: None | reporting.Pipeline | reporting.Backend = None,
    ):
        """This class serves as a base class for organizing the search parameter optimization process, which defines the parameters used for search.

        Parameters
        ----------

        workflow: peptidecentric.PeptideCentricWorkflow
            The workflow object that the optimization is being performed on.

        """
        self.optimal_parameter = None
        self.workflow = workflow
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


class AutomaticOptimizer(BaseOptimizer):
    def __init__(
        self,
        initial_parameter: float,
        workflow,
        **kwargs,
    ):
        """This class automatically optimizes the search parameter and stores the progres of optimization in a dataframe, history_df.

        Parameters
        ----------
        initial_parameter: float
            The parameter used for search in the first round of optimization.


        """
        super().__init__(workflow, **kwargs)
        self.history_df = pd.DataFrame()
        self.workflow.com.fit({self.parameter_name: initial_parameter})
        self.has_converged = False

    def step(self, precursors_df: pd.DataFrame, fragments_df: pd.DataFrame):
        """See base class. The feature is used to track the progres of the optimization (stored in .feature) and determine whether it has converged."""
        if self.has_converged:
            self.reporter.log_string(
                f"✅ {self.parameter_name:<15}: optimization complete. Optimal parameter {self.workflow.com.__dict__[self.parameter_name]} found after {len(self.history_df)} searches.",
                verbosity="progress",
            )
            return

        new_row = pd.DataFrame(
            [
                {
                    "parameter": float(
                        self.workflow.com.__dict__[self.parameter_name]
                    ),  # Ensure float dtype
                    self.feature_name: self._get_feature_value(
                        precursors_df, fragments_df
                    ),
                    "classifier_version": int(
                        self.workflow.fdr_manager.current_version
                    ),  # Ensure int dtype
                    "score_cutoff": float(self.workflow.com.score_cutoff),
                    "fwhm_rt": float(self.workflow.com.fwhm_rt),
                    "fwhm_mobility": float(self.workflow.com.fwhm_mobility),
                }
            ]
        )
        self.history_df = pd.concat([self.history_df, new_row], ignore_index=True)
        just_converged = self._check_convergence()

        if just_converged:
            self.has_converged = True

            index_of_optimum = self.history_df[self.feature_name].idxmax()

            optimal_parameter = self.history_df["parameter"].loc[index_of_optimum]
            classifier_version_at_optimum = self.history_df["classifier_version"].loc[
                index_of_optimum
            ]
            score_cutoff_at_optimum = self.history_df["score_cutoff"].loc[
                index_of_optimum
            ]
            fwhm_rt_at_optimum = self.history_df["fwhm_rt"].loc[index_of_optimum]
            fwhm_mobility_at_optimum = self.history_df["fwhm_mobility"].loc[
                index_of_optimum
            ]

            self.workflow.com.fit({self.parameter_name: optimal_parameter})
            self.workflow.com.fit({"classifier_version": classifier_version_at_optimum})
            self.workflow.com.fit({"score_cutoff": score_cutoff_at_optimum})
            self.workflow.com.fit({"fwhm_rt": fwhm_rt_at_optimum})
            self.workflow.com.fit({"fwhm_mobility": fwhm_mobility_at_optimum})

            self.reporter.log_string(
                f"✅ {self.parameter_name:<15}: optimization complete. Optimal parameter {self.workflow.com.__dict__[self.parameter_name]:.4f} found after {len(self.history_df)} searches.",
                verbosity="progress",
            )

        else:
            new_parameter = self._propose_new_parameter(
                precursors_df
                if self.estimator_group_name == "precursor"
                else fragments_df
            )

            self.workflow.com.fit({self.parameter_name: new_parameter})

            self.reporter.log_string(
                f"❌ {self.parameter_name:<15}: optimization incomplete after {len(self.history_df)} search(es). Will search with parameter {self.workflow.com.__dict__[self.parameter_name]:.4f}.",
                verbosity="progress",
            )

    def plot(self):
        """Plot the optimization of the RT error parameter."""
        fig, ax = plt.subplots()

        ax.axvline(
            x=self.workflow.com.__dict__[self.parameter_name],
            ymin=0,
            ymax=self.history_df[self.feature_name].max(),
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

    @abstractmethod
    def _propose_new_parameter(self, df):
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
        pass


class TargetedOptimizer(BaseOptimizer):
    def __init__(
        self,
        initial_parameter: float,
        target_parameter: float,
        workflow,
        **kwargs,
    ):
        """This class optimizes the search parameter until it reaches a user-specified target value.

        Parameters
        ----------

        initial_parameter: float
            The parameter used for search in the first round of optimization.

        target_parameter: float
            Optimization will stop when this parameter is reached.

        """
        super().__init__(workflow, **kwargs)
        self.workflow.com.fit({self.parameter_name: initial_parameter})
        self.target_parameter = target_parameter
        self.has_converged = False

    def _check_convergence(self, proposed_parameter):
        """The optimization has converged if the proposed parameter is equal to or less than the target parameter. At this point, the target parameter is saved as the optimal parameter.

        Parameters
        ----------
        proposed_parameter: float
            The proposed parameter for the next round of optimization.
        """

        return (
            proposed_parameter <= self.target_parameter
            and self.workflow.current_version
            > self.workflow.config["min_training_iterations"]
        )

    def _propose_new_parameter(self, df: pd.DataFrame):
        """See base class. The update rule is
            1) calculate the deviation of the predicted mz values from the observed mz values,
            2) take the mean of the endpoints of the central 95% of these deviations, and
            3) take the maximum of this value and the target parameter.
        This is implemented by the ci method for the estimator.
        """
        return max(
            self.workflow.calibration_manager.get_estimator(
                self.estimator_group_name, self.estimator_name
            ).ci(df, 0.95),
            self.target_parameter,
        )

    def step(self, precursors_df: pd.DataFrame, fragments_df: pd.DataFrame):
        """See base class."""
        if self.has_converged:
            self.reporter.log_string(
                f"✅ {self.parameter_name:<15}: {self.workflow.com.__dict__[self.parameter_name]:.4f} <= {self.target_parameter:.4f}",
                verbosity="progress",
            )
            return

        new_parameter = self._propose_new_parameter(
            precursors_df if self.estimator_group_name == "precursor" else fragments_df
        )
        just_converged = self._check_convergence(new_parameter)
        self.workflow.com.fit({self.parameter_name: new_parameter})
        self.workflow.com.fit(
            {"classifier_version": self.workflow.fdr_manager.current_version}
        )

        if just_converged:
            self.has_converged = True
            self.reporter.log_string(
                f"✅ {self.parameter_name:<15}: {self.workflow.com.__dict__[self.parameter_name]:.4f} <= {self.target_parameter:.4f}",
                verbosity="progress",
            )

        else:
            self.reporter.log_string(
                f"❌ {self.parameter_name:<15}: {self.workflow.com.__dict__[self.parameter_name]:.4f} > {self.target_parameter:.4f}",
                verbosity="progress",
            )


class AutomaticRTOptimizer(AutomaticOptimizer):
    def __init__(
        self,
        initial_parameter: float,
        workflow,
        **kwargs,
    ):
        """See base class.

        Parameters
        ----------

        initial_parameter: float
            The parameter used for search in the first round of optimization.

        """
        self.parameter_name = "rt_error"
        self.estimator_group_name = "precursor"
        self.estimator_name = "rt"
        self.feature_name = "precursor_count"
        super().__init__(initial_parameter, workflow, **kwargs)

    def _check_convergence(self):
        """Optimization should stop if continued optimization of the parameter is not improving the TODO feature value.
        This function checks if the previous rounds of optimization have led to a meaningful improvement in the TODO feature value.
        If so, it continues optimization and appends the proposed new parameter to the list of parameters. If not, it stops optimization and sets the optimal parameter attribute.

        Notes
        -----
            Because the check for an increase in TODO feature value requires two previous rounds, the function will also initialize for another round of optimization if there have been fewer than 3 rounds.


        """

        return (
            len(self.history_df) > 2
            and self.history_df[self.feature_name].iloc[-1]
            < 1.1 * self.history_df[self.feature_name].iloc[-2]
            and self.history_df[self.feature_name].iloc[-1]
            < 1.1 * self.history_df[self.feature_name].iloc[-3]
        )

    def _propose_new_parameter(self, df: pd.DataFrame):
        """See base class. The update rule is
            1) calculate the deviation of the predicted mz values from the observed mz values,
            2) take the mean of the endpoints of the central 99% of these deviations, and
            3) multiply this value by 1.1.
        This is implemented by the ci method for the estimator.


        """
        return 1.1 * self.workflow.calibration_manager.get_estimator(
            self.estimator_group_name, self.estimator_name
        ).ci(df, 0.99)

    def _get_feature_value(
        self, precursors_df: pd.DataFrame, fragments_df: pd.DataFrame
    ):
        return len(precursors_df)


class AutomaticMS2Optimizer(AutomaticOptimizer):
    def __init__(
        self,
        initial_parameter: float,
        workflow,
        **kwargs,
    ):
        """This class automatically optimizes the MS2 tolerance parameter by tracking the number of precursor identifications and stopping when further changes do not increase this number.

        Parameters
        ----------
        initial_parameter: float
            The parameter used for search in the first round of optimization.


        """
        self.parameter_name = "ms2_error"
        self.estimator_group_name = "fragment"
        self.estimator_name = "mz"
        self.feature_name = "precursor_count"
        super().__init__(initial_parameter, workflow, **kwargs)

    def _check_convergence(self):
        """Optimization should stop if continued narrowing of the MS2 parameter is not improving the number of precursor identifications.
        This function checks if the previous rounds of optimization have led to a meaningful improvement in the number of identifications.
        If so, it continues optimization and appends the proposed new parameter to the list of parameters. If not, it stops optimization and sets the optimal parameter attribute.

        Notes
        -----
            Because the check for an increase in identifications requires two previous rounds, the function will also initialize for another round of optimization if there have been fewer than 3 rounds.


        """

        return (
            len(self.history_df) > 2
            and self.history_df[self.feature_name].iloc[-1]
            < 1.1 * self.history_df[self.feature_name].iloc[-2]
            and self.history_df[self.feature_name].iloc[-1]
            < 1.1 * self.history_df[self.feature_name].iloc[-3]
        )

    def _propose_new_parameter(self, df: pd.DataFrame):
        """See base class. The update rule is
            1) calculate the deviation of the predicted mz values from the observed mz values,
            2) take the mean of the endpoints of the central 99% of these deviations, and
            3) multiply this value by 1.1.
        This is implemented by the ci method for the estimator.


        """
        return 1.1 * self.workflow.calibration_manager.get_estimator(
            self.estimator_group_name, self.estimator_name
        ).ci(df, 0.99)

    def _get_feature_value(
        self, precursors_df: pd.DataFrame, fragments_df: pd.DataFrame
    ):
        return len(precursors_df)


class AutomaticMS1Optimizer(AutomaticOptimizer):
    def __init__(
        self,
        initial_parameter: float,
        workflow,
        **kwargs,
    ):
        """See base class.

        Parameters
        ----------

        initial_parameter: float
            The parameter used for search in the first round of optimization.

        """
        self.parameter_name = "ms1_error"
        self.estimator_group_name = "precursor"
        self.estimator_name = "mz"
        self.feature_name = "precursor_count"
        super().__init__(initial_parameter, workflow, **kwargs)

    def _check_convergence(self):
        """Optimization should stop if continued narrowing of the parameter is not improving the TODO feature value.
        This function checks if the previous rounds of optimization have led to a meaningful improvement in the TODO feature value.
        If so, it continues optimization and appends the proposed new parameter to the list of parameters. If not, it stops optimization and sets the optimal parameter attribute.

        Notes
        -----
            Because the check for an increase in TODO feature value requires two previous rounds, the function will also initialize for another round of optimization if there have been fewer than 3 rounds.


        """

        return (
            len(self.history_df) > 2
            and self.history_df[self.feature_name].iloc[-1]
            < 1.1 * self.history_df[self.feature_name].iloc[-2]
            and self.history_df[self.feature_name].iloc[-1]
            < 1.1 * self.history_df[self.feature_name].iloc[-3]
        )

    def _propose_new_parameter(self, df: pd.DataFrame):
        """See base class. The update rule is
            1) calculate the deviation of the predicted mz values from the observed mz values,
            2) take the mean of the endpoints of the central 99% of these deviations, and
            3) multiply this value by 1.1.
        This is implemented by the ci method for the estimator.


        """
        return 1.1 * self.workflow.calibration_manager.get_estimator(
            self.estimator_group_name, self.estimator_name
        ).ci(df, 0.99)

    def _get_feature_value(
        self, precursors_df: pd.DataFrame, fragments_df: pd.DataFrame
    ):
        return len(precursors_df)


class AutomaticMobilityOptimizer(AutomaticOptimizer):
    def __init__(
        self,
        initial_parameter: float,
        workflow,
        **kwargs,
    ):
        """See base class.

        Parameters
        ----------

        initial_parameter: float
            The parameter used for search in the first round of optimization.

        """
        self.parameter_name = "mobility_error"
        self.estimator_group_name = "precursor"
        self.estimator_name = "mobility"
        self.feature_name = "precursor_count"
        super().__init__(initial_parameter, workflow, **kwargs)

    def _check_convergence(self):
        """Optimization should stop if continued narrowing of the parameter is not improving the TODO feature value.
        This function checks if the previous rounds of optimization have led to a meaningful improvement in the TODO feature value.
        If so, it continues optimization and appends the proposed new parameter to the list of parameters. If not, it stops optimization and sets the optimal parameter attribute.

        Notes
        -----
            Because the check for an increase in TODO feature value requires two previous rounds, the function will also initialize for another round of optimization if there have been fewer than 3 rounds.


        """

        return (
            len(self.history_df) > 2
            and self.history_df[self.feature_name].iloc[-1]
            < 1.1 * self.history_df[self.feature_name].iloc[-2]
            and self.history_df[self.feature_name].iloc[-1]
            < 1.1 * self.history_df[self.feature_name].iloc[-3]
        )

    def _propose_new_parameter(self, df: pd.DataFrame):
        """See base class. The update rule is
            1) calculate the deviation of the predicted mz values from the observed mz values,
            2) take the mean of the endpoints of the central 99% of these deviations, and
            3) multiply this value by 1.1.
        This is implemented by the ci method for the estimator.


        """
        return 1.1 * self.workflow.calibration_manager.get_estimator(
            self.estimator_group_name, self.estimator_name
        ).ci(df, 0.99)

    def _get_feature_value(
        self, precursors_df: pd.DataFrame, fragments_df: pd.DataFrame
    ):
        return len(precursors_df)


class TargetedRTOptimizer(TargetedOptimizer):
    """This class optimizes the RT search parameter until it reaches a user-specified target value."""

    def __init__(
        self,
        initial_parameter: float,
        target_parameter: float,
        workflow,
        **kwargs,
    ):
        """See base class."""
        self.parameter_name = "rt_error"
        self.estimator_group_name = "precursor"
        self.estimator_name = "rt"
        super().__init__(initial_parameter, target_parameter, workflow, **kwargs)


class TargetedMS2Optimizer(TargetedOptimizer):
    """This class optimizes the MS2 search parameter until it reaches a user-specified target value."""

    def __init__(
        self,
        initial_parameter: float,
        target_parameter: float,
        workflow,
        **kwargs,
    ):
        """See base class."""
        self.parameter_name = "ms2_error"
        self.estimator_group_name = "fragment"
        self.estimator_name = "mz"
        super().__init__(initial_parameter, target_parameter, workflow, **kwargs)


class TargetedMS1Optimizer(TargetedOptimizer):
    """This class optimizes the MS1 search parameter until it reaches a user-specified target value."""

    def __init__(
        self,
        initial_parameter: float,
        target_parameter: float,
        workflow,
        **kwargs,
    ):
        """See base class."""
        self.parameter_name = "ms1_error"
        self.estimator_group_name = "precursor"
        self.estimator_name = "mz"
        super().__init__(initial_parameter, target_parameter, workflow, **kwargs)


class TargetedMobilityOptimizer(TargetedOptimizer):
    """This class optimizes the mobility search parameter until it reaches a user-specified target value."""

    def __init__(
        self,
        initial_parameter: float,
        target_parameter: float,
        workflow,
        **kwargs,
    ):
        """See base class."""
        self.parameter_name = "mobility_error"
        self.estimator_group_name = "precursor"
        self.estimator_name = "mobility"
        super().__init__(initial_parameter, target_parameter, workflow, **kwargs)
