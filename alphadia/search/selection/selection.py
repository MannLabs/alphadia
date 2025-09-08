"""Main candidate selection implementation for DIA data analysis."""

import logging

import alphatims.utils
import numba as nb
import numpy as np
import pandas as pd

from alphadia import utils
from alphadia.constants.keys import CalibCols
from alphadia.raw_data import DiaData, DiaDataJIT
from alphadia.search.jitclasses.fragment_container import FragmentContainer
from alphadia.search.selection import fft
from alphadia.search.selection.config_df import (
    CandidateContainer,
    CandidateSelectionConfig,
    CandidateSelectionConfigJIT,
    PrecursorFlatContainer,
    candidate_container_to_df,
)
from alphadia.search.selection.kernel import GaussianKernel
from alphadia.search.selection.utils import (
    amean1,
    assemble_isotope_mz,
    astd1,
    find_peaks_1d,
    find_peaks_2d,
    slice_manual,
    symetric_limits_2d,
    wrap0,
    wrap1,
)
from alphadia.utils import USE_NUMBA_CACHING
from alphadia.validation.schemas import fragments_flat_schema, precursors_flat_schema

logger = logging.getLogger()


@nb.njit(cache=USE_NUMBA_CACHING)
def _is_valid(
    dense_fragments: np.ndarray, dense_precursors: np.ndarray, kernel: np.ndarray
) -> bool:
    """Perform sanity checks and return False if any of them fails."""
    if dense_fragments.shape[0] == 0:
        # "Empty dense fragment matrix"
        return False

    if dense_precursors.shape[0] == 0:
        # "Empty dense precursor matrix"
        return False

    if dense_fragments.shape[2] % 2 != 0:
        # "Dense fragment matrix not divisible by 2"
        return False

    if dense_precursors.shape[2] % 2 != 0:
        # "Dense precursor matrix not divisible by 2"
        return False

    if (
        dense_precursors.shape[2] < kernel.shape[0]
        or dense_precursors.shape[3] < kernel.shape[1]
    ):
        # "Precursor matrix smaller than convolution kernel"
        return False

    if (
        dense_fragments.shape[2] < kernel.shape[0]
        or dense_fragments.shape[3] < kernel.shape[1]
    ):
        # "Fragment matrix smaller than convolution kernel"
        return False

    return True


@alphatims.utils.pjit(cache=USE_NUMBA_CACHING)
def _select_candidates_pjit(
    i: int,  # pjit decorator changes the passed argument from an iterable to single index
    jit_data: DiaDataJIT,
    precursor_container: PrecursorFlatContainer,
    fragment_container: FragmentContainer,
    config: CandidateSelectionConfigJIT,
    kernel: np.ndarray,
    candidate_container: CandidateContainer,
) -> None:
    """Select candidates for MS2 extraction based on MS1 features.

    Parameters
    ----------
    i : int
        Index of the precursor to process.
    jit_data : DiaDataJIT
        JIT-compiled data object containing the raw data.
    precursor_container : PrecursorFlatContainer
        Container holding precursor information.
    fragment_container : FragmentContainer
        Container holding fragment information.
    config : CandidateSelectionConfigJIT
        Configuration object containing parameters for candidate selection.
    kernel : np.ndarray
        Convolution kernel for smoothing the precursor and fragment data.
    candidate_container : CandidateContainer
        Container to store the selected candidates.

    Returns
    -------
    None, results are stored in `candidate_container`.

    """
    # prepare precursor isotope intensity
    # (n_isotopes)
    isotope_intensity = precursor_container.isotopes[i][: config.top_k_precursors]
    # (n_isotopes)
    isotope_mz = assemble_isotope_mz(
        precursor_container.mz[i], precursor_container.charge[i], isotope_intensity
    )

    fragment_idx_slices = np.array(
        [
            [
                precursor_container.frag_start_idx[i],
                precursor_container.frag_stop_idx[i],
                1,
            ]
        ],
        dtype=np.uint32,
    )

    fragment_container_slice = slice_manual(fragment_container, fragment_idx_slices)
    if config.exclude_shared_ions:
        fragment_container_slice.filter_by_cardinality(1)
    fragment_container_slice.sort_by_mz()

    if len(fragment_container_slice.precursor_idx) <= 3:
        return

    # start extraction of raw data
    rt = precursor_container.rt[i]
    mobility = precursor_container.mobility[i]

    frame_limits = jit_data.get_frame_indices_tolerance(
        rt, config.rt_tolerance, min_size=config.kernel_size
    )
    scan_limits = jit_data.get_scan_indices_tolerance(
        mobility, config.mobility_tolerance
    )

    # identify most abundant isotope
    # max_isotope_idx = np.argmax(isotope_intensity)
    quadrupole_mz = np.array([[isotope_mz[0], isotope_mz[-1]]], dtype=np.float32)

    dense_precursors, _ = jit_data.get_dense_intensity(
        frame_limits,
        scan_limits,
        isotope_mz,
        config.precursor_mz_tolerance,
        np.array([[-1.0, -1.0]], dtype=np.float32),
    )

    # shape = (2, n_fragments, n_observations, n_scans, n_frames), dtype = np.float32
    dense_fragments, _ = jit_data.get_dense_intensity(
        frame_limits,
        scan_limits,
        fragment_container_slice.mz,
        config.fragment_mz_tolerance,
        quadrupole_mz,
        custom_cycle=jit_data.cycle,
    )

    # FLAG: needed for debugging
    # self.dense_fragments = dense_fragments

    if not _is_valid(dense_fragments, dense_precursors, kernel):
        return

    if config.use_weighted_score:
        mean = config.feature_mean
        std = config.feature_std
        weights = config.feature_weight

    else:
        mean = None
        std = None
        weights = None

    _build_candidates(
        precursor_container.precursor_idx[i],
        candidate_container,
        precursor_container.candidate_start_idx[i],
        dense_precursors,
        dense_fragments,
        kernel,
        jit_data,
        config,
        scan_limits,
        frame_limits,
        candidate_count=config.candidate_count,
        weights=weights,
        mean=mean,
        std=std,
    )


@nb.njit(fastmath=True, cache=USE_NUMBA_CACHING)
def _build_features(
    smooth_precursor: np.ndarray, smooth_fragment: np.ndarray
) -> np.ndarray:
    n_features = 1

    features = np.zeros(
        (
            n_features,
            smooth_precursor.shape[2],
            smooth_fragment.shape[3],
        ),
        dtype=np.float32,
    )

    log_fragment = np.sum(np.log(smooth_fragment[0] + 1), axis=0)
    log_precursor = np.sum(np.log(smooth_precursor[0] + 1), axis=0)

    features[0] = log_fragment + log_precursor

    return features


@nb.njit(cache=USE_NUMBA_CACHING)
def _join_close_peaks(
    peak_scan_list: np.ndarray,
    peak_cycle_list: np.ndarray,
    peak_score_list: np.ndarray,
    scan_tolerance: int,
    cycle_tolerance: int,
) -> np.ndarray:
    """Join peaks that are close in scan and cycle space.

    Parameters
    ----------
    peak_scan_list : np.ndarray
        List of scan indices for each peak

    peak_cycle_list : np.ndarray
        List of cycle indices for each peak

    peak_score_list : np.ndarray
        List of scores for each peak

    scan_tolerance : int
        Maximum number of scans that two peaks can be apart to be considered close

    cycle_tolerance : int
        Maximum number of cycles that two peaks can be apart to be considered close

    Returns
    -------
    peak_mask : np.ndarray, dtype=np.bool_

    """
    n_peaks = peak_scan_list.shape[0]
    peak_mask = np.ones(n_peaks, dtype=np.bool_)
    for peak_idx in range(n_peaks):
        if not peak_mask[peak_idx]:
            continue
        scan = peak_scan_list[peak_idx]
        cycle = peak_cycle_list[peak_idx]
        score = peak_score_list[peak_idx]
        for other_peak_idx in range(peak_idx + 1, n_peaks):
            if not peak_mask[other_peak_idx]:
                continue
            other_scan = peak_scan_list[other_peak_idx]
            other_cycle = peak_cycle_list[other_peak_idx]
            other_score = peak_score_list[other_peak_idx]
            if (
                abs(scan - other_scan) <= scan_tolerance
                and abs(cycle - other_cycle) <= cycle_tolerance
            ):
                if score > other_score:
                    peak_mask[other_peak_idx] = False
                else:
                    peak_mask[peak_idx] = False

    return peak_mask


@nb.njit(cache=USE_NUMBA_CACHING)
def _join_overlapping_candidates(
    scan_limits_list: np.ndarray,
    cycle_limits_list: np.ndarray,
    p_scan_overlap: float = 0.01,
    p_cycle_overlap: float = 0.6,
) -> np.ndarray:
    """Identify overlapping candidates and join them into a single candidate.
    The limits of the candidates are updated in-place.

    Parameters
    ----------
    scan_limits_list : np.ndarray
        List of scan limits for each candidate

    cycle_limits_list : np.ndarray
        List of cycle limits for each candidate

    p_scan_overlap : float
        Minimum percentage of scan overlap to join two candidates

    p_cycle_overlap : float
        Minimum percentage of cycle overlap to join two candidates

    Returns
    -------
    joined_mask : np.ndarray, dtype=np.bool_
        Mask that indicates which candidates were joined

    """
    joined_mask = np.ones(len(scan_limits_list), dtype=np.bool_)

    for i in range(len(scan_limits_list)):
        # check if the candidate is already joined
        if joined_mask[i] == 0:
            continue

        # check if the candidate overlaps with any other candidate
        for j in range(i + 1, len(scan_limits_list)):
            # check if the candidate is already joined
            if joined_mask[j] == 0:
                continue

            # calculate the overlap of the area of the two candidates

            cycle_len = cycle_limits_list[i, 1] - cycle_limits_list[i, 0]
            cycle_overlap = (
                min(cycle_limits_list[i, 1], cycle_limits_list[j, 1])
                - max(cycle_limits_list[i, 0], cycle_limits_list[j, 0])
            ) / cycle_len

            scan_len = scan_limits_list[i, 1] - scan_limits_list[i, 0]
            scan_overlap = (
                min(scan_limits_list[i, 1], scan_limits_list[j, 1])
                - max(scan_limits_list[i, 0], scan_limits_list[j, 0])
            ) / scan_len

            # overlap must be positive in both dimensions
            if scan_overlap < 0 or cycle_overlap < 0:
                continue

            if cycle_overlap > p_cycle_overlap and scan_overlap > p_scan_overlap:
                # join the candidates
                scan_limits_list[i, 0] = min(
                    scan_limits_list[i, 0], scan_limits_list[j, 0]
                )
                scan_limits_list[i, 1] = max(
                    scan_limits_list[i, 1], scan_limits_list[j, 1]
                )
                cycle_limits_list[i, 0] = min(
                    cycle_limits_list[i, 0], cycle_limits_list[j, 0]
                )
                cycle_limits_list[i, 1] = max(
                    cycle_limits_list[i, 1], cycle_limits_list[j, 1]
                )
                joined_mask[j] = 0

    return joined_mask


@nb.njit(fastmath=True, cache=USE_NUMBA_CACHING)
def _build_candidates(
    precursor_idx: int,
    candidate_container: CandidateContainer,
    candidate_start_idx: int,
    dense_precursors: np.ndarray,
    dense_fragments: np.ndarray,
    kernel: np.ndarray,
    jit_data: DiaDataJIT,
    config: CandidateSelectionConfigJIT,
    scan_limits: np.ndarray,
    frame_limits: np.ndarray,
    candidate_count: int = 3,
    weights: np.ndarray | None = None,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
) -> None:
    cycle_length = jit_data.cycle.shape[1]

    feature_weights = np.ones(1) if weights is None else weights
    feature_weights = feature_weights.reshape(-1, 1, 1)

    smooth_precursor = fft.convolve_fourier(dense_precursors, kernel)
    smooth_fragment = fft.convolve_fourier(dense_fragments, kernel)

    if smooth_precursor.shape != dense_precursors.shape:
        print(
            f"smooth_precursor shape does not match dense_precursors shape {smooth_precursor.shape} != {dense_precursors.shape}"
        )
    if smooth_fragment.shape != dense_fragments.shape:
        print(
            f"smooth_fragment shape does not match dense_fragments shape {smooth_fragment.shape} != {dense_fragments.shape}"
        )

    feature_matrix = _build_features(smooth_precursor, smooth_fragment).astype(
        "float32"
    )

    # get mean and std to normalize features
    # if trained, use the mean and std from training, otherwise calculate the mean and std from the current data
    feature_mean = (
        amean1(feature_matrix).reshape(-1, 1, 1)
        if mean is None
        else mean.reshape(-1, 1, 1)
    )

    feature_std = (
        astd1(feature_matrix).reshape(-1, 1, 1)
        if std is None
        else std.reshape(-1, 1, 1)
    )

    if not (feature_std.shape == feature_mean.shape == feature_weights.shape):
        raise ValueError(
            f"feature_mean.shape={feature_mean.shape}, feature_std.shape={feature_std.shape} and feature_weights.shape={feature_weights.shape} must be equal"
        )

    feature_matrix_norm = (
        feature_weights * (feature_matrix - feature_mean) / (feature_std + 1e-6)
    )

    score = np.sum(feature_matrix_norm, axis=0)

    peak_scan_list, peak_cycle_list, peak_score_list = _find_peaks(
        score, candidate_count
    )

    peak_mask = _join_close_peaks(
        peak_scan_list, peak_cycle_list, peak_score_list, 3, 3
    )

    peak_scan_list = peak_scan_list[peak_mask]
    peak_cycle_list = peak_cycle_list[peak_mask]
    peak_score_list = peak_score_list[peak_mask]

    scan_limits_list = np.zeros((peak_scan_list.shape[0], 2), dtype="int32")
    cycle_limits_list = np.zeros((peak_cycle_list.shape[0], 2), dtype="int32")

    for candidate_rank, (scan_relative, cycle_relative) in enumerate(
        zip(peak_scan_list, peak_cycle_list)  # ('strict' not supported by numba yet)
    ):
        scan_limits_relative, cycle_limits_relative = symetric_limits_2d(
            score,
            scan_relative,
            cycle_relative,
            f_mobility=config.f_mobility,
            f_rt=config.f_rt,
            center_fraction=config.center_fraction,
            min_size_mobility=config.min_size_mobility,
            min_size_rt=config.min_size_rt,
            max_size_mobility=config.max_size_mobility,
            max_size_rt=config.max_size_rt,
        )

        scan_limits_list[candidate_rank] = scan_limits_relative
        cycle_limits_list[candidate_rank] = cycle_limits_relative

    # check if candidates overlapping candidates should be joined
    if config.join_close_candidates:
        mask = _join_overlapping_candidates(
            scan_limits_list,
            cycle_limits_list,
            p_scan_overlap=config.join_close_candidates_scan_threshold,
            p_cycle_overlap=config.join_close_candidates_cycle_threshold,
        )

        peak_scan_list = peak_scan_list[mask]
        peak_cycle_list = peak_cycle_list[mask]
        peak_score_list = peak_score_list[mask]
        scan_limits_list = scan_limits_list[mask]
        cycle_limits_list = cycle_limits_list[mask]

    # (n_candidates)
    candidate_rank_array = np.arange(peak_scan_list.shape[0], dtype=np.uint8)

    for (
        candidate_rank,
        scan_relative,
        cycle_relative,
        candidate_score,
        scan_limits_relative,
        cycle_limits_relative,
    ) in zip(
        candidate_rank_array,
        peak_scan_list,
        peak_cycle_list,
        peak_score_list,
        scan_limits_list,
        cycle_limits_list,
    ):  # ('strict' not supported by numba yet)
        # does not work anymore

        scan_limits_absolute = wrap1(
            scan_limits_relative + scan_limits[0, 0], jit_data.scan_max_index
        )
        frame_limits_absolute = wrap1(
            cycle_limits_relative * cycle_length + frame_limits[0, 0],
            jit_data.frame_max_index,
        )

        scan_absolute = wrap0(
            scan_relative + scan_limits[0, 0], jit_data.scan_max_index
        )
        frame_absolute = wrap0(
            cycle_relative * cycle_length + frame_limits[0, 0], jit_data.frame_max_index
        )

        candidate_index = candidate_start_idx + candidate_rank

        candidate_container.precursor_idx[candidate_index] = precursor_idx
        candidate_container.rank[candidate_index] = candidate_rank
        candidate_container.score[candidate_index] = candidate_score

        candidate_container.scan_center[candidate_index] = scan_absolute
        candidate_container.scan_start[candidate_index] = scan_limits_absolute[0]
        candidate_container.scan_stop[candidate_index] = scan_limits_absolute[1]

        candidate_container.frame_center[candidate_index] = frame_absolute
        candidate_container.frame_start[candidate_index] = frame_limits_absolute[0]
        candidate_container.frame_stop[candidate_index] = frame_limits_absolute[1]


@nb.njit(cache=USE_NUMBA_CACHING)
def _find_peaks(
    score: np.ndarray,
    candidate_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Identify distinct peaks."""
    #  check if there is a real ion mobility dimension
    if score.shape[0] <= 2:
        peak_scan_list, peak_cycle_list, peak_score_list = find_peaks_1d(
            score, top_n=candidate_count
        )
    else:
        peak_scan_list, peak_cycle_list, peak_score_list = find_peaks_2d(
            score, top_n=candidate_count
        )
    return peak_scan_list, peak_cycle_list, peak_score_list


class CandidateSelection:
    def __init__(
        self,
        dia_data: DiaData,
        precursors_flat: pd.DataFrame,
        fragments_flat: pd.DataFrame,
        config: CandidateSelectionConfig,
        rt_column: str,
        mobility_column: str,
        precursor_mz_column: str,
        fragment_mz_column: str,
        fwhm_rt: float = 5.0,
        fwhm_mobility: float = 0.012,
    ) -> None:
        """Select candidates for MS2 extraction based on MS1 features

        Parameters
        ----------
        dia_data : DiaData
            dia data object

        precursors_flat : pd.DataFrame
            flattened precursor dataframe

        fragments_flat : pd.DataFrame
            flattened fragment dataframe

        config : CandidateSelectionConfig
            config object

        rt_column : str
            name of the rt column in the precursor dataframe

        mobility_column : str
            name of the mobility column in the precursor dataframe

        precursor_mz_column : str
            name of the precursor mz column in the precursor dataframe

        fragment_mz_column : str
            name of the fragment mz column in the fragment dataframe

        fwhm_rt : float, optional
            full width at half maximum in RT dimension for the GaussianKernel, by default 5.0

        fwhm_mobility : float, optional
            full width at half maximum in mobility dimension for the GaussianKernel, by default 0.012

        """
        self.dia_data_jit: DiaDataJIT = dia_data.to_jitclass()

        self.precursors_flat = precursors_flat.sort_values("precursor_idx").reset_index(
            drop=True
        )
        self.fragments_flat = fragments_flat
        self.config_jit = config.to_jitclass()

        self.rt_column = rt_column
        self.precursor_mz_column = precursor_mz_column
        self.fragment_mz_column = fragment_mz_column
        self.mobility_column = mobility_column

        gaussian_filter = GaussianKernel(
            self.dia_data_jit,
            fwhm_rt=fwhm_rt,
            sigma_scale_rt=self.config_jit.sigma_scale_rt,
            fwhm_mobility=fwhm_mobility,
            sigma_scale_mobility=self.config_jit.sigma_scale_mobility,
            kernel_width=self.config_jit.kernel_size,
            kernel_height=min(
                self.config_jit.kernel_size, self.dia_data_jit.scan_max_index + 1
            ),
        )
        self.kernel = gaussian_filter.get_dense_matrix()

    def __call__(self, thread_count: int = 10, debug: bool = False) -> pd.DataFrame:
        """Perform candidate extraction workflow.
        1. First, elution groups are assembled based on the annotation in the flattened precursor dataframe.
        Each elution group is instantiated as an ElutionGroup Numba JIT object.
        Elution groups are stored in the ElutionGroupContainer Numba JIT object.

        2. Then, the elution groups are iterated over and the candidates are selected.
        The candidate selection is performed in parallel using the alphatims.utils.pjit function.

        3. Finally, the candidates are collected from the ElutionGroup,
        assembled into a pd.DataFrame and precursor information is appended.

        Returns
        -------
        pd.DataFrame
            dataframe containing the extracted candidates

        """
        logging.info("Starting candidate selection")

        precursor_container = self._assemble_precursor_container(self.precursors_flat)
        fragment_container = self._assemble_fragment_container()

        # initialize output container
        candidate_container = CandidateContainer(
            len(self.precursors_flat) * self.config_jit.candidate_count
        )

        iterator_len = len(self.precursors_flat)

        if debug:
            iterator_len = min(10, len(self.precursors_flat))
            thread_count = 1

        alphatims.utils.set_threads(thread_count)

        _select_candidates_pjit(
            range(iterator_len),  # type: ignore  # noqa: PGH003  # function is wrapped by pjit -> will be turned into single index and passed to the method
            self.dia_data_jit,
            precursor_container,
            fragment_container,
            self.config_jit,
            self.kernel,
            candidate_container,
        )

        candidate_df = candidate_container_to_df(candidate_container)

        candidate_with_precursors_df = candidate_df.merge(
            self.precursors_flat[["precursor_idx", "elution_group_idx", "decoy"]],
            on="precursor_idx",
            how="left",
        )

        return candidate_with_precursors_df

    def _assemble_fragment_container(self) -> FragmentContainer:
        # set cardinality to 1 if not present
        if "cardinality" in self.fragments_flat.columns:
            cardinality_values = self.fragments_flat["cardinality"].values
        else:
            logging.warning(
                "Fragment cardinality column not found in fragment dataframe. Setting cardinality to 1."
            )
            cardinality_values = np.ones(len(self.fragments_flat), dtype=np.uint8)

        self.fragments_flat["cardinality"] = cardinality_values

        # prepare jitclass compatible dtypes
        fragments_flat_schema.validate(
            self.fragments_flat, warn_on_critical_values=True
        )

        return FragmentContainer(
            self.fragments_flat[CalibCols.MZ_LIBRARY].values,
            self.fragments_flat[self.fragment_mz_column].values,
            self.fragments_flat["intensity"].values,
            self.fragments_flat["type"].values,
            self.fragments_flat["loss_type"].values,
            self.fragments_flat["charge"].values,
            self.fragments_flat["number"].values,
            self.fragments_flat["position"].values,
            self.fragments_flat["cardinality"].values,
        )

    def _assemble_precursor_container(
        self, precursors_flat: pd.DataFrame
    ) -> PrecursorFlatContainer:
        # prepare jitclass compatible dtypes
        precursors_flat_schema.validate(precursors_flat, warn_on_critical_values=True)

        available_isotopes = utils.get_isotope_columns(precursors_flat.columns)
        available_isotope_columns = [f"i_{i}" for i in available_isotopes]

        candidate_start_index = np.arange(
            0,
            len(precursors_flat) * self.config_jit.candidate_count,
            self.config_jit.candidate_count,
            dtype=np.uint32,
        )
        candidate_stop_index = (
            candidate_start_index + self.config_jit.candidate_count
        ).astype(np.uint32)

        return PrecursorFlatContainer(
            precursors_flat["precursor_idx"].values,
            precursors_flat["flat_frag_start_idx"].values,
            precursors_flat["flat_frag_stop_idx"].values,
            candidate_start_index,
            candidate_stop_index,
            precursors_flat["charge"].values,
            precursors_flat[self.rt_column].values,
            precursors_flat[self.mobility_column].values,
            precursors_flat[self.precursor_mz_column].values,
            precursors_flat[available_isotope_columns].values.copy(),
        )
