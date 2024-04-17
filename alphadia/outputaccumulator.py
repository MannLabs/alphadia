"""
Output Accumulator
==================
This module contains classes to accumulate the information from the output folders of the alphadia pipeline
in a linear fashion. This is hugely useful when we have a large number of output folders and we want to accumulate the information from
all of them in a single object/Library which can be a challenge to do in a single go due to memory constraints.
The module is designed as broadcast-subscriber pattern where the AccumulationBroadcaster class loops over the output folders creating a
speclibBase object from each output folder and then broadcasts the information to the subscribers.

Classes
-------
BaseAccumulator
    Base class for accumulator classes, which are used to subscribe on the linear accumulation of a list of output folders.
    it has two methods update and post_process.

AccumulationBroadcaster
    Class that loops over output folders in a linear fashion to prevent having all the output folders in memory at the same time.

TransferLearningAccumulator
    Class that accumulates the information from the output folders for fine-tuning by selecting the top keep_top precursors and their fragments from all the output folders.


"""

from typing import List, Tuple
import numba as nb
import pandas as pd
import numpy as np
import os
import multiprocessing
import threading

from alphabase.spectral_library import base
from alphabase.spectral_library.flat import SpecLibFlat


import logging

logger = logging.getLogger()


class SpecLibFlatFromOutput(SpecLibFlat):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def parse_output_folder(
        self,
        folder: str,
        selected_precursor_columns: List[str] = [
            "precursor_idx",
            "sequence",
            "flat_frag_start_idx",
            "flat_frag_stop_idx",
            "charge",
            "rt_library",
            "rt_observed",
            "rt_calibrated",
            "mobility_library",
            "mobility_observed",
            "mz_library",
            "mz_observed",
            "mz_calibrated",
            "proteins",
            "genes",
            "mods",
            "mod_sites",
            "proba",
        ],
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Parse the output folder to get a precursor and fragment dataframe in the flat format.

        Parameters
        ----------
        folder : str
            The output folder to be parsed.
        selected_precursor_columns : list, optional
            The columns to be selected from the precursor dataframe, by default ['precursor_idx', 'sequence', 'flat_frag_start_idx', 'flat_frag_stop_idx', 'charge', 'rt_library', 'mobility_library', 'mz_library', 'proteins', 'genes', 'mods', 'mod_sites', 'proba']

        Returns
        -------
        pd.DataFrame
            The precursor dataframe.
        pd.DataFrame
            The fragment dataframe.


        """
        psm_df = pd.read_csv(os.path.join(folder, "psm.tsv"), sep="\t")
        frag_df = pd.read_csv(os.path.join(folder, "frag.tsv"), sep="\t")

        assert set(selected_precursor_columns).issubset(
            psm_df.columns
        ), f"selected_precursor_columns must be a subset of psm_df.columns didnt find {set(selected_precursor_columns) - set(psm_df.columns)}"
        psm_df = psm_df[selected_precursor_columns]
        # validate.precursors_flat_from_output(psm_df)

        self._precursor_df = pd.DataFrame()
        for col in psm_df.columns:
            self._precursor_df[col] = psm_df[col]

        # self._precursor_df.set_index('precursor_idx', inplace=True)
        # Change the data type of the mods column to string
        self._precursor_df["mods"] = self._precursor_df["mods"].astype(str)

        self._precursor_df["mod_sites"] = self._precursor_df["mod_sites"].astype(str)

        # Replace nan with empty string
        self._precursor_df["mods"] = self._precursor_df["mods"].replace("nan", "")
        self._precursor_df["mod_sites"] = self._precursor_df["mod_sites"].replace(
            "nan", ""
        )

        self.calc_precursor_mz()

        for col in ["rt", "mz", "mobility"]:
            if f"{col}_observed" in psm_df.columns:
                values = psm_df[f"{col}_observed"]
            elif "{col}_calibrated" in psm_df.columns:
                values = psm_df["{col}_calibrated"]
            else:
                values = psm_df[f"{col}_library"]
            self._precursor_df[col] = values

        # ----------------- Fragment -----------------

        self._fragment_df = frag_df[
            ["mz", "intensity", "precursor_idx", "frag_idx"]
        ].copy()

        for col in ["number", "type", "charge"]:
            if col in self.custom_fragment_df_columns:
                self._fragment_df.loc[:, col] = frag_df.loc[:, col]

        if "position" in self.custom_fragment_df_columns:
            if "position" in frag_df.columns:
                self._fragment_df.loc[:, "position"] = frag_df.loc[:, "position"]
            else:
                # Fragtypes from ascii to char
                available_frag_types = self._fragment_df["type"].unique()
                self.frag_types_as_char = {i: chr(i) for i in available_frag_types}

                mapped_frag_types = self._fragment_df["type"].map(
                    self.frag_types_as_char
                )
                a_b_c_fragments = mapped_frag_types.isin(["a", "b", "c"])
                x_y_z_fragments = mapped_frag_types.isin(["x", "y", "z"])

                precursor_idx_to_nAA = (
                    self._precursor_df[["precursor_idx", "nAA"]]
                    .set_index("precursor_idx")
                    .to_dict()["nAA"]
                )
                # For X,Y,Z frags calculate the position as being the nAA of the precursor - number of the fragment
                x_y_z_number = (
                    self._fragment_df.loc[x_y_z_fragments, "precursor_idx"].map(
                        precursor_idx_to_nAA
                    )
                    - self._fragment_df.loc[x_y_z_fragments, "number"]
                )
                self._fragment_df.loc[x_y_z_fragments, "position"] = x_y_z_number - 1

                # For A,B,C frags calculate the position as being the number of the fragment
                self._fragment_df.loc[a_b_c_fragments, "position"] = (
                    self._fragment_df.loc[a_b_c_fragments, "number"] - 1
                )

                # Change position to int
                self._fragment_df["position"] = self._fragment_df["position"].astype(
                    int
                )

        return self._precursor_df, self._fragment_df


class BaseAccumulator:
    """
    Base class for accumulator classes, which are used to subscribe on the linear accumulation of a list of output folders.
    """

    def update(self, info: base.SpecLibBase) -> None:
        """
        Called when a new output folder is obtained.

        Parameters
        ----------
        info : SpecLibBase
            The information from the output folder.

        """
        raise NotImplementedError("Subclasses must implement the update method")

    def post_process(self) -> None:
        """
        Called after all output folders have been processed.
        """

        raise NotImplementedError("Subclasses must implement the post_process method")


def process_folder(folder):
    """
    Process a folder and return the speclibase object.
    It does so by parsing the output folderto get SpecLibFlat object and then converting it to SpecLibBase object.
    And for now it assumes that the loss_type is 0 for all the fragments.

    Parameters
    ----------
    folder : str
        The folder to be processed.

    Returns
    -------
    SpecLibBase
        The SpecLibBase object obtained from the output folder.
    """
    specLibFlat_object = SpecLibFlatFromOutput()
    psm, frag_df = specLibFlat_object.parse_output_folder(folder)
    specLibFlat_object._fragment_df["loss_type"] = 0
    speclibase = specLibFlat_object.to_SpecLibBase()
    return speclibase


def error_callback(e):
    logger.error(e)


class AccumulationBroadcaster:
    """
    Class that loops over output folders in a linear fashion to only have one folder in memory at a time.
    And broadcasts the output of each folder to the subscribers.
    """

    def __init__(self, folders: list, number_of_processes: int):
        self._folders = folders
        self._number_of_processes = number_of_processes
        self._subscribers = []
        self._lock = (
            threading.Lock()
        )  # Lock to prevent two processes trying to update the same subscriber at the same time

    def subscribe(self, subscriber):
        assert isinstance(
            subscriber, BaseAccumulator
        ), f"subscriber must be an instance of BaseAccumulator, got {type(subscriber)}"
        self._subscribers.append(subscriber)

    def _update_subscriber(self, subscriber, speclibase):
        subscriber.update(speclibase)

    def _broadcast(self, result):
        speclibBase = result
        with self._lock:
            for sub in self._subscribers:
                self._update_subscriber(sub, speclibBase)

    def _post_process(self):
        for sub in self._subscribers:
            sub.post_process()

    def run(self):
        with multiprocessing.Pool(processes=self._number_of_processes) as pool:
            for folder in self._folders:
                _ = pool.apply_async(
                    process_folder,
                    (folder,),
                    callback=self._broadcast,
                    error_callback=error_callback,
                )
            pool.close()
            pool.join()
            self._post_process()


@nb.jit(nopython=True)
def _get_top_indices_from_freq(
    number_of_readings_per_precursor, keep_top, len_of_precursor_df
):
    """
    Get the indices of the top keep_top elements in the array number_of_readings_per_precursor.

    Parameters
    ----------
    number_of_readings_per_precursor : np.array
        The array of number of readings per precursor.
    keep_top : int
        The number of top elements to keep.
    len_of_precursor_df : int
        The length of the precursor_df.

    Returns
    -------
    np.array
        The indices of the top keep_top elements in the array number_of_readings_per_precursor.
    """
    indices = np.zeros(len_of_precursor_df, dtype=np.bool_)
    i = 0
    for n in number_of_readings_per_precursor:
        indices[i : i + min(n, keep_top)] = np.ones(min(n, keep_top), dtype=np.bool_)
        i += n

    return indices


class TransferLearningAccumulator(BaseAccumulator):
    def __init__(self, keep_top: int = 3, norm_w_calib: bool = True):
        """
        TransferLearningAccumulator is used to accumulate the information from the output folders for fine-tuning by selecting
        the top keep_top precursors and their fragments from all the output folders. The current measure of score is the probA

        Parameters
        ----------

        keep_top : int, optional
            The number of top precursors to keep, by default 3
        norm_w_calib : bool, optional
            Whether to normalize the retention time with the calibration, by default True

        """
        self._keep_top = keep_top
        self.consensus_speclibase = None
        self._norm_w_calib = norm_w_calib

    def update(self, speclibase: base.SpecLibBase):
        """
        Update the consensus_speclibase with the information from the speclibase.

        Parameters
        ----------
        speclibase : SpecLibBase
            The information from the output folder.


        """
        speclibase.hash_precursor_df()
        if self.consensus_speclibase is None:
            self.consensus_speclibase = speclibase
        else:
            # Append in basespeclib and modify to work do the same for additional dataframe

            self.consensus_speclibase.append(
                speclibase,
                dfs_to_append=["_precursor_df"]
                + [df for df in speclibase.available_dense_fragment_dfs()],
            )

        # Sort by modseqhash and proba in ascending order
        self.consensus_speclibase._precursor_df = (
            self.consensus_speclibase._precursor_df.sort_values(
                ["mod_seq_hash", "proba"], ascending=[True, True]
            )
        )

        # Select the top keep_top precursors

        # First get the numbero of readings per precursor such as mod_seq_hash _ maps to number of rows with the same mod_seq_hash
        number_of_readings_per_precursor = self.consensus_speclibase._precursor_df[
            "mod_seq_hash"
        ].value_counts(sort=False)
        keepIndices = _get_top_indices_from_freq(
            number_of_readings_per_precursor.values,
            self._keep_top,
            self.consensus_speclibase._precursor_df.shape[0],
        )
        assert (
            len(keepIndices) == self.consensus_speclibase._precursor_df.shape[0]
        ), f"keepIndices length {len(keepIndices)} must be equal to the length of the precursor_df {self.consensus_speclibase._precursor_df.shape[0]}"
        self.consensus_speclibase._precursor_df = (
            self.consensus_speclibase._precursor_df.iloc[keepIndices]
        )

        # Drop unused fragments
        self.consensus_speclibase.remove_unused_fragments()

    def post_process(self):
        """
        Post process the consensus_speclibase by normalizing the retention time and mobility.
        """
        if self._norm_w_calib:
            # rt normalization from observed rt
            deviation_from_calib = (
                self.consensus_speclibase.precursor_df["rt_observed"]
                - self.consensus_speclibase.precursor_df["rt_calibrated"]
            ) / self.consensus_speclibase.precursor_df["rt_calibrated"]

            self.consensus_speclibase.precursor_df[
                "rt_norm"
            ] = self.consensus_speclibase.precursor_df["rt_library"] * (
                1 + deviation_from_calib
            )
            # Normalize rt
            self.consensus_speclibase.precursor_df["rt_norm"] = (
                self.consensus_speclibase.precursor_df["rt_norm"]
                / self.consensus_speclibase.precursor_df["rt_norm"].max()
            )

            rt_observed_norm = (
                self.consensus_speclibase.precursor_df["rt_observed"]
                / self.consensus_speclibase.precursor_df["rt_observed"].max()
            )

            self.consensus_speclibase.precursor_df["rt_norm"] = (
                (1 - self.consensus_speclibase.precursor_df["rt_norm"])
                * self.consensus_speclibase.precursor_df["rt_norm"]
            ) + (self.consensus_speclibase.precursor_df["rt_norm"] * rt_observed_norm)

        else:
            self.consensus_speclibase.precursor_df["rt_norm"] = (
                self.consensus_speclibase.precursor_df["rt_observed"]
                / self.consensus_speclibase.precursor_df["rt_observed"].max()
            )