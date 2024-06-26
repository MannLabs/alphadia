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

import logging
import multiprocessing
import os
import threading

import numba as nb
import numpy as np
import pandas as pd
from alphabase.spectral_library import base
from alphabase.spectral_library.flat import SpecLibFlat
from tqdm import tqdm

logger = logging.getLogger()


class SpecLibFlatFromOutput(SpecLibFlat):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _calculate_fragment_position(self):
        """
        Calculate the position of the fragments based on the type and number of the fragment.
        """
        # Fragtypes from ascii to char
        available_frag_types = self._fragment_df["type"].unique()
        self.frag_types_as_char = {i: chr(i) for i in available_frag_types}

        mapped_frag_types = self._fragment_df["type"].map(self.frag_types_as_char)
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
        self._fragment_df["position"] = self._fragment_df["position"].astype(int)

    def parse_output_folder(
        self,
        folder: str,
        selected_precursor_columns: list[str] | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
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
        if selected_precursor_columns is None:
            selected_precursor_columns = [
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
                "decoy",
            ]
        psm_df = pd.read_parquet(os.path.join(folder, "psm.parquet"))
        frag_df = pd.read_parquet(os.path.join(folder, "frag.parquet"))

        assert set(
            selected_precursor_columns
        ).issubset(
            psm_df.columns
        ), f"selected_precursor_columns must be a subset of psm_df.columns didnt find {set(selected_precursor_columns) - set(psm_df.columns)}"
        psm_df = psm_df[selected_precursor_columns]
        # validate.precursors_flat_from_output(psm_df)

        # get foldername of the output folder
        foldername = os.path.basename(folder)
        psm_df["raw_name"] = foldername

        # remove decoy precursors
        psm_df = psm_df[psm_df["decoy"] == 0]

        self._precursor_df = pd.DataFrame()
        for col in psm_df.columns:
            self._precursor_df[col] = psm_df[col]

        self._precursor_df["decoy"] = self._precursor_df["decoy"].astype(int)
        self._precursor_df = psm_df[psm_df["decoy"] == 0].reset_index(drop=True)

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
        # Filer fragments that are not used in the precursors
        frag_df = frag_df[
            frag_df["precursor_idx"].isin(self._precursor_df["precursor_idx"])
        ]
        self._fragment_df = frag_df[
            ["mz", "intensity", "precursor_idx", "frag_idx", "correlation"]
        ].copy()

        for col in ["number", "type", "charge"]:
            if col in self.custom_fragment_df_columns:
                self._fragment_df.loc[:, col] = frag_df.loc[:, col]

        if "position" in self.custom_fragment_df_columns:
            if "position" in frag_df.columns:
                self._fragment_df.loc[:, "position"] = frag_df.loc[:, "position"]
            else:
                self._calculate_fragment_position()

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
    speclibflat_object = SpecLibFlatFromOutput()
    psm, frag_df = speclibflat_object.parse_output_folder(folder)
    speclibflat_object._fragment_df["loss_type"] = 0
    speclibase = speclibflat_object.to_SpecLibBase()
    # sort columns
    for dense_df_name in speclibase.available_dense_fragment_dfs():
        df = getattr(speclibase, dense_df_name)
        setattr(speclibase, dense_df_name, df[df.columns.sort_values()])

    return speclibase


def error_callback(e):
    logger.error(e, exc_info=True)


class AccumulationBroadcaster:
    """
    Class that loops over output folders in a linear fashion to only have one folder in memory at a time.
    And broadcasts the output of each folder to the subscribers.
    """

    def __init__(self, folders: list, number_of_processes: int):
        self._folders = folders
        self._number_of_processes = number_of_processes
        self._subscribers = []
        self._lock = threading.Lock()  # Lock to prevent two processes trying to update the same subscriber at the same time

    def subscribe(self, subscriber: BaseAccumulator):
        assert isinstance(
            subscriber, BaseAccumulator
        ), f"subscriber must be an instance of BaseAccumulator, got {type(subscriber)}"
        self._subscribers.append(subscriber)

    def _update_subscriber(
        self, subscriber: BaseAccumulator, speclibase: base.SpecLibBase
    ):
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
    number_of_readings_per_precursor: np.ndarray,
    keep_top: int,
    len_of_precursor_df: int,
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
        to_keep = min(n, keep_top)
        indices[i : i + to_keep] = np.ones(to_keep, dtype=np.bool_)
        i += n

    return indices


class TransferLearningAccumulator(BaseAccumulator):
    def __init__(
        self,
        keep_top: int = 3,
        norm_delta_max: bool = True,
        precursor_correlation_cutoff: float = 0.5,
        fragment_correlation_ratio: float = 0.75,
    ):
        """
        TransferLearningAccumulator is used to accumulate the information from the output folders for fine-tuning by selecting
        the top keep_top precursors and their fragments from all the output folders. The current measure of score is the probA

        Parameters
        ----------

        keep_top : int, optional
            The number of top precursors to keep, by default 3

        norm_w_calib : bool, optional
            If true, advanced normalization of retention times will be performed.
            Retention times are normalized using calibrated deviation from the library at the start of the gradient and max normalization at the end of the gradient.

            If false, max normalization will be performed, by default True

        precursor_correlation_cutoff : float, optional
            Only precursors with a median fragment correlation above this cutoff will be used for MS2 learning, by default 0.5

        fragment_correlation_ratio : float, optional
            The cutoff for the fragment correlation relative to the median fragment correlation for a precursor, by default 0.75

        """
        self._keep_top = keep_top
        self.consensus_speclibase = None
        self._norm_delta_max = norm_delta_max
        self._precursor_correlation_cutoff = precursor_correlation_cutoff
        self._fragment_correlation_ratio = fragment_correlation_ratio

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

        # First get the number of readings per precursor such as mod_seq_hash _ maps to number of rows with the same mod_seq_hash
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
        Post process the consensus_speclibase by normalizing retention times.
        """

        logger.info(
            "Performing quality control for transfer learning."
            + f"Normalize by delta: {self._norm_delta_max}"
            + f"Precursor correlation cutoff: {self._precursor_correlation_cutoff}"
            + f"Fragment correlation cutoff: {self._fragment_correlation_ratio}"
        )

        if self._norm_delta_max:
            self.consensus_speclibase = normalize_rt_delta_max(
                self.consensus_speclibase
            )
        else:
            self.consensus_speclibase = normalize_rt_max(self.consensus_speclibase)

        self.consensus_speclibase = ms2_quality_control(
            self.consensus_speclibase,
            self._precursor_correlation_cutoff,
            self._fragment_correlation_ratio,
        )


def normalize_rt_max(spec_lib_base: base.SpecLibBase) -> base.SpecLibBase:
    """
    Normalize the retention times of the precursors in the SpecLibBase object using max normalization.

    Parameters
    ----------

    spec_lib_base : SpecLibBase
        The SpecLibBase object to be normalized.

    Returns
    -------

    SpecLibBase
        The SpecLibBase object with the retention times normalized using max normalization.

    """

    spec_lib_base.precursor_df["rt_norm"] = (
        spec_lib_base.precursor_df["rt_observed"]
        / spec_lib_base.precursor_df["rt_observed"].max()
    )

    return spec_lib_base


def normalize_rt_delta_max(spec_lib_base: base.SpecLibBase) -> base.SpecLibBase:
    """
    Normalize the retention times of the precursors in the SpecLibBase object using delta max normalization.

    Parameters
    ----------

    spec_lib_base : SpecLibBase
        The SpecLibBase object to be normalized.

    Returns
    -------

    SpecLibBase
        The SpecLibBase object with the retention times normalized using delta max normalization.

    """

    # instead of a simple max normalization, we want to use a weighted average of the two normalizations
    # At the start of the retention time we will normalize using the calibrated deviation from the library
    # At the end of the retention time we will normalize using the max normalization

    precursor_df = spec_lib_base.precursor_df

    # caclulate max normalization
    max_norm = precursor_df["rt_observed"].values / np.max(
        precursor_df["rt_observed"].values
    )

    # calculate calibrated normalization
    deviation_from_calib = (
        precursor_df["rt_observed"].values - precursor_df["rt_calibrated"].values
    ) / precursor_df["rt_calibrated"].values
    calibrated_norm = precursor_df["rt_library"].values * (1 + deviation_from_calib)
    calibrated_norm = calibrated_norm / calibrated_norm.max()

    # use max norm as weight and combine the two normalizations
    spec_lib_base.precursor_df["rt_norm"] = (
        1 - max_norm
    ) * calibrated_norm + max_norm * max_norm

    return spec_lib_base


def ms2_quality_control(
    spec_lib_base: base.SpecLibBase,
    precursor_correlation_cutoff: float = 0.5,
    fragment_correlation_ratio: float = 0.75,
):
    """
    Perform quality control for transfer learning by filtering out precursors with low median fragment correlation and fragments with low correlation.

    Parameters
    ----------

    spec_lib_base : SpecLibBase
        The SpecLibBase object to be normalized.

    precursor_correlation_cutoff : float
        Only precursors with a median fragment correlation above this cutoff will be used for MS2 learning. Default is 0.5.

    fragment_correlation_ratio : float
        The cutoff for the fragment correlation relative to the median fragment correlation for a precursor. Default is 0.75.

    Returns
    -------

    SpecLibBase
        The SpecLibBase object with the precursors and fragments that pass the quality
        control filters.
    """

    use_for_ms2 = np.zeros(len(spec_lib_base.precursor_df), dtype=bool)

    precursor_df = spec_lib_base.precursor_df
    fragment_intensity_df = spec_lib_base.fragment_intensity_df
    fragment_correlation_df = spec_lib_base._fragment_correlation_df

    for i, (start_idx, stop_idx) in tqdm(
        enumerate(
            zip(
                precursor_df["frag_start_idx"],
                precursor_df["frag_stop_idx"],
                strict=True,
            )
        )
    ):
        # get XIC correlations and intensities for the precursor
        fragment_correlation_view = fragment_correlation_df.iloc[start_idx:stop_idx]
        flat_correlation = fragment_correlation_view.values.flatten()

        fragment_intensity_view = fragment_intensity_df.iloc[start_idx:stop_idx]
        flat_intensity = fragment_intensity_view.values.flatten()

        # calculate the median correlation for the precursor
        intensity_mask = flat_intensity > 0.0
        median_correlation = np.median(flat_correlation[intensity_mask])

        # use the precursor for MS2 learning if the median correlation is above the cutoff
        use_for_ms2[i] = median_correlation > precursor_correlation_cutoff

        fragment_intensity_view[:] = fragment_intensity_view * (
            fragment_correlation_view > median_correlation * fragment_correlation_ratio
        )

    spec_lib_base.precursor_df["use_for_ms2"] = use_for_ms2

    return spec_lib_base
