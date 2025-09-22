import numpy as np
import pandas as pd
from alphabase.peptide.fragment import remove_unused_fragments
from alphabase.spectral_library.flat import SpecLibFlat

from alphadia.exceptions import NoOptimizationLockTargetError
from alphadia.workflow.config import Config
from alphadia.workflow.managers.calibration_manager import CalibrationGroups


class OptimizationLock:
    def __init__(self, library: SpecLibFlat, config: Config):
        """Sets and updates the optimization lock, which is the data used for calibration and optimization of the search parameters.

        Parameters
        ----------
        library: alphabase.spectral_library.flat.SpecLibFlat
            The library object from the PeptideCentricWorkflow object, which includes the precursor and fragment library dataframes.

        config: Config
            The configuration object from the PeptideCentricWorkflow.
        """
        self._library = library

        self.previously_calibrated = False
        self.has_target_num_precursors = False

        self._elution_group_order = library._precursor_df["elution_group_idx"].unique()
        rng = np.random.default_rng(seed=772)
        rng.shuffle(self._elution_group_order)

        self._precursor_target_count = config["calibration"]["optimization_lock_target"]
        self._batch_size = config["calibration"]["batch_size"]

        self.batch_idx = 0
        self.batch_plan = self._get_batch_plan(
            len(self._elution_group_order),
            self._batch_size,
        )

        eg_idxes = self._elution_group_order[self.start_idx : self.stop_idx]

        self.batch_library: SpecLibFlat | None = None
        self.set_batch_dfs(eg_idxes)

        self._feature_dfs = []
        self._fragment_dfs = []

    @property
    def features_df(self) -> pd.DataFrame:
        return pd.concat(self._feature_dfs)

    @property
    def fragments_df(self) -> pd.DataFrame:
        return pd.concat(self._fragment_dfs)

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

    @staticmethod
    def _get_batch_plan(
        num_items: int, batch_size: int, *, fixed_start_idx: bool = False
    ) -> list[tuple[int, int]]:
        """Gets an exponential batch plan based on num_items and batch_size.

        The batch plan is a list of tuples, where each tuple contains the start and stop index of the elution groups to use for each step in the optimization lock.

        Parameters
        ----------
        num_items: int
            The total number of items to create a batch plan for.
        batch_size: int
            The batch size to use for each step in the optimization lock.
        fixed_start_idx: bool
            If True, the start index of each batch is fixed to 0, otherwise the start index is the stop index of the previous batch.

        returns
        -------
        list[tuple[int,int]]
            The batch plan as a list of tuples, where each tuple contains the start and stop index
        """

        plan = []

        step = 0
        start_idx = 0
        stop_idx = 0

        while stop_idx < num_items:
            n_batches = int(2**step)
            stop_idx = min(stop_idx + n_batches * batch_size, num_items)
            plan.append((start_idx, stop_idx))
            step += 1
            if not fixed_start_idx:
                start_idx = stop_idx

        return plan

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

        self._feature_dfs += [feature_df]
        self._fragment_dfs += [fragment_df]

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
        calibration_manager: CalibrationManager
            The calibration manager object from the PeptideCentricWorkflow object.

        """
        calibration_manager.predict(
            self.batch_library._precursor_df, CalibrationGroups.PRECURSOR
        )

        calibration_manager.predict(
            self.batch_library._fragment_df, CalibrationGroups.FRAGMENT
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
            self._feature_dfs = []
            self._fragment_dfs = []

        else:
            self.increase_batch_idx()

        eg_idxes = self._elution_group_order[self.start_idx : self.stop_idx]
        self.set_batch_dfs(eg_idxes)

    def reset_after_convergence(self, calibration_manager):
        """Resets the optimization lock after all optimizers in a given round of optimization have converged.

        Parameter
        ---------
        calibration_manager: CalibrationManager
            The calibration manager object from the PeptideCentricWorkflow object.

        """
        self.has_target_num_precursors = True
        self._feature_dfs = []
        self._fragment_dfs = []
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
            eg_idxes = self._elution_group_order[self.start_idx : self.stop_idx]
        self.batch_library = SpecLibFlat()
        # TODO using batch_library.precursor_df (no underscore) here will trigger the setter method, which will additionally call refine_precursor_df()
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
