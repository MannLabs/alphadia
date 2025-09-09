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

from alphadia.constants.keys import CalibCols, SearchStepFiles

logger = logging.getLogger()


def build_speclibflat_from_quant(
    folder: str,
    mandatory_precursor_columns: list[str] | None = None,
    optional_precursor_columns: list[str] | None = None,
    charged_frag_types: list[str] | None = None,
) -> SpecLibFlat:
    """
    Build a SpecLibFlat object from quantification output data stored in a folder for transfer learning.


    Parameters
    ----------
    folder : str
        The output folder to be parsed.
    mandatory_precursor_columns : list[str], optional
        The columns to be selected from the precursor dataframe
    optional_precursor_columns : list[str], optional
        Additional optional columns to include if present

    Returns
    -------
    SpecLibFlat
        A spectral library object containing the parsed data
    """
    speclib = SpecLibFlat()

    if mandatory_precursor_columns is None:
        mandatory_precursor_columns = [
            "precursor_idx",
            "sequence",
            "flat_frag_start_idx",
            "flat_frag_stop_idx",
            "charge",
            CalibCols.RT_LIBRARY,
            CalibCols.RT_OBSERVED,
            CalibCols.MOBILITY_LIBRARY,
            CalibCols.MOBILITY_OBSERVED,
            CalibCols.MZ_LIBRARY,
            CalibCols.MZ_OBSERVED,
            "proteins",
            "genes",
            "mods",
            "mod_sites",
            "proba",
            "decoy",
        ]

    if optional_precursor_columns is None:
        optional_precursor_columns = [
            CalibCols.RT_CALIBRATED,
            CalibCols.MZ_CALIBRATED,
        ]

    psm_df = pd.read_parquet(os.path.join(folder, SearchStepFiles.PSM_FILE_NAME))
    frag_df = pd.read_parquet(
        os.path.join(folder, SearchStepFiles.FRAG_TRANSFER_FILE_NAME)
    )

    if not set(mandatory_precursor_columns).issubset(psm_df.columns):
        raise ValueError(
            f"mandatory_precursor_columns must be a subset of psm_df.columns didnt find {set(mandatory_precursor_columns) - set(psm_df.columns)}"
        )

    available_columns = sorted(
        list(
            set(mandatory_precursor_columns)
            | (set(optional_precursor_columns) & set(psm_df.columns))
        )
    )
    psm_df = psm_df[available_columns]

    psm_df["raw_name"] = os.path.basename(folder)

    psm_df["decoy"] = psm_df["decoy"].astype(int)
    psm_df = psm_df[psm_df["decoy"] == 0].reset_index(drop=True)

    speclib._precursor_df = psm_df.copy()

    speclib._precursor_df["mods"] = speclib._precursor_df["mods"].astype(str)
    speclib._precursor_df["mod_sites"] = speclib._precursor_df["mod_sites"].astype(str)
    speclib._precursor_df["mods"] = speclib._precursor_df["mods"].replace("nan", "")
    speclib._precursor_df["mod_sites"] = speclib._precursor_df["mod_sites"].replace(
        "nan", ""
    )

    speclib.calc_precursor_mz()

    for col in ["rt", "mz", "mobility"]:
        if f"{col}_observed" in psm_df.columns:
            values = psm_df[f"{col}_observed"]
        elif "{col}_calibrated" in psm_df.columns:
            values = psm_df["{col}_calibrated"]
        else:
            values = psm_df[f"{col}_library"]
        speclib._precursor_df[col] = values

    frag_df = frag_df[
        frag_df["precursor_idx"].isin(speclib._precursor_df["precursor_idx"])
    ]
    speclib._fragment_df = frag_df[
        [
            "mz",
            "intensity",
            "precursor_idx",
            "frag_idx",
            "correlation",
            "number",
            "type",
            "charge",
            "loss_type",
            "position",
        ]
    ].copy()

    return speclib.to_speclib_base(
        charged_frag_types=charged_frag_types,
        flat_columns=["intensity", "correlation"],
    )


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


def error_callback(e):
    logger.error(e, exc_info=True)


class AccumulationBroadcaster:
    """
    Class that loops over output folders in a linear fashion to only have one folder in memory at a time.
    And broadcasts the output of each folder to the subscribers.
    """

    def __init__(
        self, folder_list: list, number_of_processes: int, processing_kwargs: dict
    ):
        self._folder_list = folder_list
        self._number_of_processes = number_of_processes
        self._subscribers = []
        self._lock = threading.Lock()  # Lock to prevent two processes trying to update the same subscriber at the same time
        self._processing_kwargs = processing_kwargs

    def subscribe(self, subscriber: BaseAccumulator):
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
            for folder in self._folder_list:
                _ = pool.apply_async(
                    build_speclibflat_from_quant,
                    (folder,),
                    self._processing_kwargs,
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
                ["mod_seq_hash", "proba", "precursor_idx"],  # last sort to break ties
                ascending=[True, True, True],
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

        norm_delta_max = self._norm_delta_max
        if (
            CalibCols.RT_CALIBRATED
            not in self.consensus_speclibase.precursor_df.columns
        ):
            logger.warning(
                f"Column '{CalibCols.RT_CALIBRATED}' not found in the precursor_df, delta-max normalization will not be performed"
            )
            norm_delta_max = False

        logger.info("Performing quality control for transfer learning.")
        logger.info(f"Normalize by delta: {norm_delta_max}")
        logger.info(
            f"Precursor correlation cutoff: {self._precursor_correlation_cutoff}"
        )
        logger.info(f"Fragment correlation cutoff: {self._fragment_correlation_ratio}")

        if norm_delta_max:
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
        spec_lib_base.precursor_df[CalibCols.RT_OBSERVED]
        / spec_lib_base.precursor_df[CalibCols.RT_OBSERVED].max()
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

    # calculate max normalization
    max_norm = precursor_df[CalibCols.RT_OBSERVED].values / np.max(
        precursor_df[CalibCols.RT_OBSERVED].values
    )

    # calculate calibrated normalization
    deviation_from_calib = (
        precursor_df[CalibCols.RT_OBSERVED].values
        - precursor_df[CalibCols.RT_CALIBRATED].values
    ) / precursor_df[CalibCols.RT_CALIBRATED].values
    calibrated_norm = precursor_df[CalibCols.RT_LIBRARY].values * (
        1 + deviation_from_calib
    )
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
        median_correlation = (
            np.median(flat_correlation[intensity_mask]) if intensity_mask.any() else 0.0
        )

        # use the precursor for MS2 learning if the median correlation is above the cutoff
        use_for_ms2[i] = median_correlation > precursor_correlation_cutoff

        # Fix: Use iloc to modify the original DataFrame instead of the view
        spec_lib_base.fragment_intensity_df.iloc[start_idx:stop_idx] = (
            fragment_intensity_view.values
            * (
                fragment_correlation_view
                > median_correlation * fragment_correlation_ratio
            )
        )

    spec_lib_base.precursor_df["use_for_ms2"] = use_for_ms2

    return spec_lib_base
