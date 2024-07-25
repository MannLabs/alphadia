# native imports
from functools import wraps

import numpy as np

# alpha family imports
# third party imports
import pandas as pd

# alphadia imports
from alphadia.workflow import reporting


def check_if_optimum_found(method):
    """If an optimum has been found, the optimizer should cease to execute its methods to check, initiate and update.
    This function can be used as a decorator and placed before the methods in an optimizer to prevent execution of the methods if an optimum has been found.
    If the optimum has been found, the decorator will print the end of optimization message and, for consistency with stop_or_continue methods, return True.
    """

    @wraps(method)
    def wrapper(self, *args, **kwargs):
        if self.optimal_tolerance is not None:
            self.end_of_optimization_message()
            return True
        return method(self, *args, **kwargs)

    return wrapper


class BaseOptimizer:
    def __init__(
        self,
        initial_tolerance: float,
        reporter: None | reporting.Pipeline | reporting.Backend = None,
    ):
        """This class serves as a base class for organizing the search parameter optimization process, which defines the tolerances used for search.

        Parameters
        ----------
        initial_tolerance: float
            The initial tolerance to start the optimization
        """
        self.proposed_new_tolerance = initial_tolerance
        self.tolerances = []
        self.round = -1
        self.optimal_tolerance = None
        self.reporter = reporting.LogBackend() if reporter is None else reporter

    def check_stopping_conditions(self):
        raise NotImplementedError(
            f"check_stopping_conditions() not implemented for {self.__class__.__name__}"
        )

    def update(self):
        raise NotImplementedError(
            f"update() not implemented for {self.__class__.__name__}"
        )

    def initiate(self):
        raise NotImplementedError(f"initiate() not implemented for {self.__class__}")


class RTOptimizer(BaseOptimizer):
    pass


class MS2Optimizer(BaseOptimizer):
    def __init__(self, initial_tolerance: float):
        super().__init__(initial_tolerance)
        self.estimator = "mz"
        self.df = "fragment"
        self.ids = []

    @check_if_optimum_found
    def check_stopping_conditions(self):
        """Optimization should stop if continued narrowing of the MS2 tolerance is not improving the number of precursor identifications.
        This function checks if the previous rounds of optimization have led to a meaningful improvement in the number of identifications.
        If so, it continues optimization and appends the proposed new tolerance to the list of tolerances. If not, it stops optimization and sets the optimal tolerance attribute.

        Notes
        -----
            Because the check for an increase in identifications requires two previous rounds, the function will also initialize for another round of optimization if the round variable is less than 2.

            Once the optimal tolerance is set, the optimizer will no longer check for improvements.

        Returns
        -------
        stop_optimizaton: bool
            True if there has been no recent improvement in the number of identifications and optimization should stop. False otherwise.

        """

        if (
            self.round < 2
            or self.ids[-1] > 1.1 * self.ids[-2]
            or self.ids[-1] > 1.1 * self.ids[-3]
        ):
            self.reporter.log_string(
                f"❌ MS2: optimization incomplete in round {self.round}. Will search with tolerance {self.proposed_new_tolerance}.",
                verbosity="progress",
            )
            stop_optimizaton = False

        else:
            self.optimal_tolerance = self.tolerances[np.argmax(self.ids)]
            self.end_of_optimization_message()
            stop_optimizaton = True

        return stop_optimizaton

    @check_if_optimum_found
    def initiate(self):
        """If the optimization continues (which will use the proposed new tolerance), the round variable should be incremented and the proposed new tolerance should be added to the list of tolerances."""
        self.tolerances.append(self.proposed_new_tolerance)
        self.round += 1

    @check_if_optimum_found
    def update(self, precursors_df: pd.DataFrame, proposed_new_tolerance: float):
        """This function appends the number of precursors identified to the list of identifications.
        It also saves the proposed new tolerance, which will be added to the list of tolerances if the optimization continues after check is called.

        Parameters
        ----------

        precursors_df: pd.DataFrame
            The filtered dataframe of precursors obtained after FDR correction.

        proposed_new_tolerance: float
            The tolerance which will be used for the next round of optimization (if optimization continues).

        """
        self.ids.append(len(precursors_df))
        self.proposed_new_tolerance = proposed_new_tolerance

    def end_of_optimization_message(self):
        self.reporter.log_string(
            f"✅ MS2: optimization complete. Optimal tolerance {self.optimal_tolerance} found in round {self.round}.",
            verbosity="progress",
        )


class MS1Optimizer(BaseOptimizer):
    pass


class MobilityOptimizer(BaseOptimizer):
    pass
