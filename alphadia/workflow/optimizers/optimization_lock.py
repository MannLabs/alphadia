import numpy as np
import pandas as pd
from alphabase.peptide.fragment import remove_unused_fragments
from alphabase.spectral_library.flat import SpecLibFlat

from alphadia.exceptions import NoOptimizationLockTargetError
from alphadia.fdr.utils import is_identification
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

        self.has_target_num_precursors = False

        self._elution_group_order = library._precursor_df["elution_group_idx"].unique()
        rng = np.random.default_rng(seed=772)
        rng.shuffle(self._elution_group_order)

        self._precursor_target_count = config["calibration"]["optimization_lock_target"]
        self._batch_size = config["calibration"]["batch_size"]

        # None when the lock is not pruned or was already pruned.
        self.prune_qval: float | None = (
            config["calibration"]["prune_lock_qval"]
            if config["calibration"].get("prune_lock", False)
            else None
        )
        # Positions in `_elution_group_order` that pruning removed from the lock.
        self._is_pruned = np.zeros(len(self._elution_group_order), dtype=bool)
        self._elution_groups_below_prune_qval = np.array([], dtype=int)

        self.batch_idx = 0
        self.batch_plan = self._get_batch_plan(
            len(self._elution_group_order), self._batch_size
        )

        self.batch_library: SpecLibFlat | None = None
        self.set_batch_dfs(self._elution_groups(self.start_idx, self.stop_idx))

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
            raise NoOptimizationLockTargetError()
        else:
            return self.batch_plan[self.batch_idx][0]

    @property
    def stop_idx(self) -> int:
        return self.batch_plan[self.batch_idx][1]

    @property
    def num_pruned_elution_groups(self) -> int:
        """Return the number of elution groups of the lock that pruning removed."""
        return int(self._is_pruned[: self.stop_idx].sum())

    def _elution_groups(self, start_idx: int, stop_idx: int) -> np.ndarray:
        """Return the elution groups between two positions of the lock order, without the pruned ones."""
        return self._elution_group_order[start_idx:stop_idx][
            ~self._is_pruned[start_idx:stop_idx]
        ]

    @staticmethod
    def _get_batch_plan(num_items: int, batch_size: int) -> list[tuple[int, int]]:
        """Gets an exponential batch plan based on num_items and batch_size.

        The batch plan is a list of tuples, where each tuple contains the start and stop index of the elution groups to use for each step in the optimization lock.

        Parameters
        ----------
        num_items: int
            The total number of items to create a batch plan for.
        batch_size: int
            The batch size to use for each step in the optimization lock.

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

        # A pruned elution group stays in the lock as extracted without an identification, so the identification rate
        # of a round compares with the rounds before pruning.
        self.total_elution_groups = (
            self.features_df["elution_group_idx"].nunique()
            + self.num_pruned_elution_groups
        )

    def update_with_fdr(self, precursor_df: pd.DataFrame):
        """Calculates the number of precursors at 1% FDR for the current optimization lock and determines if it is sufficient to perform calibration and optimization.

        Parameters
        ----------
        precursor_df: pd.DataFrame
            The precursor dataframe for the current batch of the optimization lock (from workflow.perform_fdr).
        """

        self._precursor_at_fdr_count = np.sum(is_identification(precursor_df))
        self.has_target_num_precursors = (
            self._precursor_at_fdr_count >= self._precursor_target_count
        )

    def record_prune_candidates(self, precursor_df: pd.DataFrame) -> None:
        """Remember the elution groups with a candidate below `prune_qval`, which the next pruning keeps.

        Parameters
        ----------
        precursor_df: pd.DataFrame
            The precursor dataframe after FDR, with the q-values up to at least `prune_qval`.
        """
        if self.prune_qval is None:
            return
        self._elution_groups_below_prune_qval = precursor_df.loc[
            precursor_df["qval"] < self.prune_qval, "elution_group_idx"
        ].unique()

    def _prune(self) -> None:
        """Remove the elution groups of the lock without a candidate below `prune_qval` in the last search.

        Targets and decoys share the elution group, so a kept target keeps its decoy and the reverse.
        """
        lock = slice(0, self.stop_idx)
        self._is_pruned[lock] = ~np.isin(
            self._elution_group_order[lock], self._elution_groups_below_prune_qval
        )
        self.prune_qval = None

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

    def extend(self, calibration_manager=None) -> None:
        """Add the next batch of the plan to the lock and keep the batches extracted so far.

        Parameters
        ----------
        calibration_manager: CalibrationManager | None
            Calibrates the new batch like the earlier batches. None when no calibration exists yet.
        """
        self.increase_batch_idx()
        self.set_batch_dfs(self._elution_groups(self.start_idx, self.stop_idx))
        if calibration_manager is not None:
            self.update_with_calibration(calibration_manager)

    def resize(self, calibration_manager) -> str | None:
        """Resize the lock to the smallest batch of the plan that is expected to hold the target.

        The next search covers the whole lock again. With `calibration.prune_lock`, the first resize also prunes the
        lock, see `_prune`, and returns a summary of the pruning to log; otherwise it returns None.

        Parameters
        ----------
        calibration_manager: CalibrationManager
            Calibrates the resized lock.
        """
        self.decrease_batch_idx()
        self._feature_dfs = []
        self._fragment_dfs = []
        prune_summary = None
        if self.prune_qval is not None:
            self._prune()
            prune_summary = (
                f"Pruned the optimization lock: removed {self.num_pruned_elution_groups:,} of "
                f"{self.stop_idx:,} elution groups."
            )
        self.set_batch_dfs(self._elution_groups(0, self.stop_idx))
        self.update_with_calibration(calibration_manager)
        return prune_summary

    def grow(self, calibration_manager) -> None:
        """Add the next batch of the plan. The next search covers the whole lock again with the current calibration.

        A warm-up calibration changes the tolerances. The features of the earlier batches used the old tolerances, so
        the lock drops them and extracts again from its first elution group.

        Parameters
        ----------
        calibration_manager: CalibrationManager
            Calibrates the grown lock.
        """
        self.increase_batch_idx()
        self._feature_dfs = []
        self._fragment_dfs = []
        self.set_batch_dfs(self._elution_groups(0, self.stop_idx))
        self.update_with_calibration(calibration_manager)

    def set_batch_dfs(self, eg_idxes: None | np.ndarray = None):
        """
        Sets the batch library to use for the next round of optimization, either adjusting it upwards or downwards depending on whether the target has been reached.

        Parameters
        ----------
        eg_idxes: None | np.ndarray
            The elution group indexes to use for the next round of optimization. If None, the eg_idxes for the current self.start_idx and self.stop_idx are used, without the pruned ones.
        """
        if eg_idxes is None:
            eg_idxes = self._elution_groups(self.start_idx, self.stop_idx)
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
