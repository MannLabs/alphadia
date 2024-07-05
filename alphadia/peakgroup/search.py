# native imports
import logging
import os

# alpha family imports
import alphatims

# third party imports
import numba as nb
import numpy as np
import pandas as pd

# alphadia imports
from alphadia import utils, validate
from alphadia.numba import config, fft, fragments, numeric
from alphadia.peakgroup.kernel import GaussianKernel
from alphadia.peakgroup.utils import assemble_isotope_mz

logger = logging.getLogger()


@nb.experimental.jitclass()
class HybridCandidateConfigJIT:
    """
    Numba compatible config object for the HybridCandidate class.
    Please see the documentation of the HybridCandidateConfig class for more information on the parameters and their default values.
    """

    rt_tolerance: nb.float64
    precursor_mz_tolerance: nb.float64
    fragment_mz_tolerance: nb.float64
    mobility_tolerance: nb.float64
    isotope_tolerance: nb.float64

    peak_len_rt: nb.float64
    sigma_scale_rt: nb.float64
    peak_len_mobility: nb.float64
    sigma_scale_mobility: nb.float64

    candidate_count: nb.int64
    top_k_precursors: nb.int64
    top_k_fragments: nb.int64
    exclude_shared_ions: nb.types.bool_
    kernel_size: nb.int64

    f_mobility: nb.float64
    f_rt: nb.float64
    center_fraction: nb.float64
    min_size_mobility: nb.int64
    min_size_rt: nb.int64
    max_size_mobility: nb.int64
    max_size_rt: nb.int64

    group_channels: nb.types.bool_
    use_weighted_score: nb.types.bool_

    join_close_candidates: nb.types.bool_
    join_close_candidates_scan_threshold: nb.float64
    join_close_candidates_cycle_threshold: nb.float64

    feature_std: nb.float64[::1]
    feature_mean: nb.float64[::1]
    feature_weight: nb.float64[::1]

    def __init__(
        self,
        rt_tolerance,
        precursor_mz_tolerance,
        fragment_mz_tolerance,
        mobility_tolerance,
        isotope_tolerance,
        peak_len_rt,
        sigma_scale_rt,
        peak_len_mobility,
        sigma_scale_mobility,
        candidate_count,
        top_k_precursors,
        top_k_fragments,
        exclude_shared_ions,
        kernel_size,
        f_mobility,
        f_rt,
        center_fraction,
        min_size_mobility,
        min_size_rt,
        max_size_mobility,
        max_size_rt,
        group_channels,
        use_weighted_score,
        join_close_candidates,
        join_close_candidates_scan_threshold,
        join_close_candidates_cycle_threshold,
        feature_std,
        feature_mean,
        feature_weight,
    ):
        self.rt_tolerance = rt_tolerance
        self.precursor_mz_tolerance = precursor_mz_tolerance
        self.fragment_mz_tolerance = fragment_mz_tolerance
        self.mobility_tolerance = mobility_tolerance
        self.isotope_tolerance = isotope_tolerance

        self.peak_len_rt = peak_len_rt
        self.sigma_scale_rt = sigma_scale_rt
        self.peak_len_mobility = peak_len_mobility
        self.sigma_scale_mobility = sigma_scale_mobility

        self.candidate_count = candidate_count
        self.top_k_precursors = top_k_precursors
        self.top_k_fragments = top_k_fragments
        self.exclude_shared_ions = exclude_shared_ions
        self.kernel_size = kernel_size

        self.f_mobility = f_mobility
        self.f_rt = f_rt
        self.center_fraction = center_fraction
        self.min_size_mobility = min_size_mobility
        self.min_size_rt = min_size_rt
        self.max_size_mobility = max_size_mobility
        self.max_size_rt = max_size_rt

        self.group_channels = group_channels
        self.use_weighted_score = use_weighted_score

        self.join_close_candidates = join_close_candidates
        self.join_close_candidates_scan_threshold = join_close_candidates_scan_threshold
        self.join_close_candidates_cycle_threshold = (
            join_close_candidates_cycle_threshold
        )

        self.feature_std = feature_std
        self.feature_mean = feature_mean
        self.feature_weight = feature_weight


class HybridCandidateConfig(config.JITConfig):
    jit_container = HybridCandidateConfigJIT

    def __init__(self):
        self.rt_tolerance = 60.0
        self.precursor_mz_tolerance = 10.0
        self.fragment_mz_tolerance = 15.0
        self.mobility_tolerance = 0.1
        self.isotope_tolerance = 0.01

        self.peak_len_rt = 10.0
        self.sigma_scale_rt = 0.1
        self.peak_len_mobility = 0.013
        self.sigma_scale_mobility = 1.0

        self.candidate_count = 5

        self.top_k_precursors = 3
        self.top_k_fragments = 12
        self.exclude_shared_ions = True
        self.kernel_size = 30

        # parameters used during peak identification
        self.f_mobility = 1.0
        self.f_rt = 0.99
        self.center_fraction = 0.5
        self.min_size_mobility = 8
        self.min_size_rt = 3
        self.max_size_mobility = 30
        self.max_size_rt = 15

        self.group_channels = False
        self.use_weighted_score = True

        self.join_close_candidates = True
        self.join_close_candidates_scan_threshold = 0.01
        self.join_close_candidates_cycle_threshold = 0.6

        # self.feature_std = np.array([ 1.2583724, 0.91052234, 1.2126098, 14.557817, 0.04327635, 0.24623954, 0.03225865, 1.2671406,1.,1,1,1 ], np.float64)
        self.feature_std = np.ones(1, np.float64)
        self.feature_mean = np.zeros(1, np.float64)
        self.feature_weight = np.ones(1, np.float64)
        # self.feature_weight[2] = 1.
        # self.feature_weight[1] = 1.

        # self.feature_weight[11] = 1.
        # self.feature_mean = np.array([ 2.967344, 1.2160938, 1.426444, 13.960179, 0.06620345, 0.44364494, 0.03138363, 3.1453438,1.,1,1,1 ], np.float64)
        # self.feature_weight = np.array([ 0.43898424,  0.97879761,  0.72262148, 0., 0.0,  0.3174245, 0.30102549,  0.44892641, 1.,1,1,1], np.float64)

    def validate(self):
        pass


@nb.experimental.jitclass
class PrecursorFlatDF:
    precursor_idx: nb.uint32[::1]

    frag_start_idx: nb.uint32[::1]
    frag_stop_idx: nb.uint32[::1]
    candidate_start_idx: nb.uint32[::1]
    candidate_stop_idx: nb.uint32[::1]

    charge: nb.uint8[::1]
    rt: nb.float32[::1]
    mobility: nb.float32[::1]
    mz: nb.float32[::1]
    isotopes: nb.float32[:, ::1]

    def __init__(
        self,
        precursor_idx,
        frag_start_idx,
        frag_stop_idx,
        candidate_start_idx,
        candidate_stop_idx,
        charge,
        rt,
        mobility,
        mz,
        isotopes,
    ):
        self.precursor_idx = precursor_idx

        self.frag_start_idx = frag_start_idx
        self.frag_stop_idx = frag_stop_idx
        self.candidate_start_idx = candidate_start_idx
        self.candidate_stop_idx = candidate_stop_idx

        self.charge = charge
        self.rt = rt
        self.mobility = mobility
        self.mz = mz
        self.isotopes = isotopes


@nb.experimental.jitclass
class CandidateDF:
    precursor_idx: nb.uint32[::1]
    rank: nb.uint8[::1]
    score: nb.float32[::1]

    scan_center: nb.uint32[::1]
    scan_start: nb.uint32[::1]
    scan_stop: nb.uint32[::1]

    frame_center: nb.uint32[::1]
    frame_start: nb.uint32[::1]
    frame_stop: nb.uint32[::1]

    def __init__(
        self,
        n_candidates,
    ):
        self.precursor_idx = np.zeros(n_candidates, dtype=np.uint32)
        self.rank = np.zeros(n_candidates, dtype=np.uint8)
        self.score = np.zeros(n_candidates, dtype=np.float32)

        self.scan_center = np.zeros(n_candidates, dtype=np.uint32)
        self.scan_start = np.zeros(n_candidates, dtype=np.uint32)
        self.scan_stop = np.zeros(n_candidates, dtype=np.uint32)

        self.frame_center = np.zeros(n_candidates, dtype=np.uint32)
        self.frame_start = np.zeros(n_candidates, dtype=np.uint32)
        self.frame_stop = np.zeros(n_candidates, dtype=np.uint32)

    def to_candidate_df(self, min_score=0):
        mask = self.score > min_score
        self.precursor_idx = self.precursor_idx[mask]
        self.rank = self.rank[mask]
        self.score = self.score[mask]

        self.scan_center = self.scan_center[mask]
        self.scan_start = self.scan_start[mask]
        self.scan_stop = self.scan_stop[mask]

        self.frame_center = self.frame_center[mask]
        self.frame_start = self.frame_start[mask]
        self.frame_stop = self.frame_stop[mask]

        return (
            self.precursor_idx,
            self.rank,
            self.score,
            self.scan_center,
            self.scan_start,
            self.scan_stop,
            self.frame_center,
            self.frame_start,
            self.frame_stop,
        )


@alphatims.utils.pjit()
def _executor(
    i,
    jit_data,
    precursor_container,
    candidate_container,
    fragment_container,
    config,
    kernel,
    debug,
):
    select_candidates(
        i,
        jit_data,
        precursor_container,
        candidate_container,
        fragment_container,
        config,
        kernel,
        debug,
    )


@nb.njit()
def select_candidates(
    i,
    jit_data,
    precursor_container,
    candidate_container,
    fragment_container,
    config,
    kernel,
    debug,
):
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

    fragment_container_slice = fragments.slice_manual(
        fragment_container, fragment_idx_slices
    )
    if config.exclude_shared_ions:
        fragment_container_slice.filter_by_cardinality(1)
    fragment_container_slice.sort_by_mz()

    if len(fragment_container_slice.precursor_idx) <= 3:
        return

    # start extraction of raw data
    rt = precursor_container.rt[i]
    mobility = precursor_container.mobility[i]

    frame_limits = jit_data.get_frame_indices_tolerance(rt, config.rt_tolerance)
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

    # perform sanity checks
    if dense_fragments.shape[0] == 0:
        # "Empty dense fragment matrix"
        return

    if dense_precursors.shape[0] == 0:
        # "Empty dense precursor matrix"
        return

    if dense_fragments.shape[2] % 2 != 0:
        # "Dense fragment matrix not divisible by 2"
        return

    if dense_fragments.shape[2] % 2 != 0:
        # "Dense fragment matrix not divisible by 2"
        return

    if (
        dense_precursors.shape[2] < kernel.shape[0]
        or dense_precursors.shape[3] < kernel.shape[1]
    ):
        # "Precursor matrix smaller than convolution kernel"
        return

    if (
        dense_fragments.shape[2] < kernel.shape[0]
        or dense_fragments.shape[3] < kernel.shape[1]
    ):
        # "Fragment matrix smaller than convolution kernel"
        return

    if config.use_weighted_score:
        mean = config.feature_mean
        std = config.feature_std
        weights = config.feature_weight

    else:
        mean = None
        std = None
        weights = None

    build_candidates(
        precursor_container.precursor_idx[i],
        candidate_container,
        precursor_container.candidate_start_idx[i],
        precursor_container.candidate_stop_idx[i],
        dense_precursors,
        dense_fragments,
        isotope_intensity,
        fragment_container_slice.intensity,
        kernel,
        jit_data,
        config,
        scan_limits,
        frame_limits,
        candidate_count=config.candidate_count,
        debug=debug,
        weights=weights,
        mean=mean,
        std=std,
    )


@nb.njit(fastmath=True)
def build_features(smooth_precursor, smooth_fragment):
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


@nb.njit
def join_close_peaks(
    peak_scan_list, peak_cycle_list, peak_score_list, scan_tolerance, cycle_tolerance
):
    """
    Join peaks that are close in scan and cycle space.

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


@nb.njit()
def join_overlapping_candidates(
    scan_limits_list, cycle_limits_list, p_scan_overlap=0.01, p_cycle_overlap=0.6
):
    """
    Identify overlapping candidates and join them into a single candidate.
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


@nb.njit(fastmath=True)
def build_candidates(
    precursor_idx,
    candidate_container,
    candidate_start_idx,
    candidate_stop_idx,
    dense_precursors,
    dense_fragments,
    precursor_intensity,
    fragment_intensity,
    kernel,
    jit_data,
    config,
    scan_limits,
    frame_limits,
    candidate_count=3,
    debug=False,
    weights=None,
    mean=None,
    std=None,
):
    cycle_length = jit_data.cycle.shape[1]

    feature_weights = np.ones(1) if weights is None else weights

    feature_weights = feature_weights.reshape(-1, 1, 1)

    smooth_precursor = fft.convolve_fourier(dense_precursors, kernel)
    smooth_fragment = fft.convolve_fourier(dense_fragments, kernel)

    if smooth_precursor.shape != dense_precursors.shape:
        print(smooth_precursor.shape, dense_precursors.shape)
        print("smooth_precursor shape does not match dense_precursors shape")
    if smooth_fragment.shape != dense_fragments.shape:
        print(smooth_fragment.shape, dense_fragments.shape)
        print("smooth_fragment shape does not match dense_fragments shape")

    feature_matrix = build_features(smooth_precursor, smooth_fragment).astype("float32")

    # get mean and std to normalize features
    # if trained, use the mean and std from training
    # otherwise calculate the mean and std from the current data
    if mean is None:
        feature_mean = utils.amean1(feature_matrix).reshape(-1, 1, 1)
    else:
        feature_mean = mean.reshape(-1, 1, 1)
    # feature_mean = feature_mean.reshape(-1,1,1)

    if std is None:
        feature_std = utils.astd1(feature_matrix).reshape(-1, 1, 1)
    else:
        feature_std = std.reshape(-1, 1, 1)
    # feature_std = feature_std.reshape(-1,1,1)

    # make sure that mean, std and weights have the same shape
    if not (feature_std.shape == feature_mean.shape == feature_weights.shape):
        raise ValueError(
            "feature_mean, feature_std and feature_weights must have the same shape"
        )

    feature_matrix_norm = (
        feature_weights * (feature_matrix - feature_mean) / (feature_std + 1e-6)
    )

    score = np.sum(feature_matrix_norm, axis=0)

    # identify distinct peaks
    #  check if there is a real ion mobility dimension
    if score.shape[0] <= 2:
        peak_scan_list, peak_cycle_list, peak_score_list = utils.find_peaks_1d(
            score, top_n=candidate_count
        )
    else:
        peak_scan_list, peak_cycle_list, peak_score_list = utils.find_peaks_2d(
            score, top_n=candidate_count
        )

    peak_mask = join_close_peaks(peak_scan_list, peak_cycle_list, peak_score_list, 3, 3)

    peak_scan_list = peak_scan_list[peak_mask]
    peak_cycle_list = peak_cycle_list[peak_mask]
    peak_score_list = peak_score_list[peak_mask]

    scan_limits_list = np.zeros((peak_scan_list.shape[0], 2), dtype="int32")
    cycle_limits_list = np.zeros((peak_cycle_list.shape[0], 2), dtype="int32")

    for candidate_rank, (scan_relative, cycle_relative) in enumerate(
        zip(peak_scan_list, peak_cycle_list)  # noqa: B905 ('strict' not supported by numba yet)
    ):
        scan_limits_relative, cycle_limits_relative = numeric.symetric_limits_2d(
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
        mask = join_overlapping_candidates(
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
    ):  # noqa: B905 ('strict' not supported by numba yet)
        # does not work anymore

        scan_limits_absolute = numeric.wrap1(
            scan_limits_relative + scan_limits[0, 0], jit_data.scan_max_index
        )
        frame_limits_absolute = numeric.wrap1(
            cycle_limits_relative * cycle_length + frame_limits[0, 0],
            jit_data.frame_max_index,
        )

        scan_absolute = numeric.wrap0(
            scan_relative + scan_limits[0, 0], jit_data.scan_max_index
        )
        frame_absolute = numeric.wrap0(
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


class HybridCandidateSelection:
    def __init__(
        self,
        dia_data,
        precursors_flat,
        fragments_flat,
        config,
        rt_column="rt_library",
        mobility_column="mobility_library",
        precursor_mz_column="mz_library",
        fragment_mz_column="mz_library",
        fwhm_rt=5.0,
        fwhm_mobility=0.012,
        feature_path=None,
    ):
        """select candidates for MS2 extraction based on MS1 features

        Parameters
        ----------

        dia_data : alphadia.data.bruker.TimsTOFDIA
            dia data object

        precursors_flat : pandas.DataFrame
            flattened precursor dataframe

        rt_column : str, optional
            name of the rt column in the precursor dataframe, by default 'rt_library'

        mobility_column : str, optional
            name of the mobility column in the precursor dataframe, by default 'mobility_library'

        precursor_mz_column : str, optional
            name of the precursor mz column in the precursor dataframe, by default 'mz_library'

        fragment_mz_column : str, optional
            name of the fragment mz column in the fragment dataframe, by default 'mz_library'

        Returns
        -------

        pandas.DataFrame
            dataframe containing the extracted candidates
        """
        self.dia_data = dia_data
        self.precursors_flat = precursors_flat.sort_values("precursor_idx").reset_index(
            drop=True
        )
        self.fragments_flat = fragments_flat

        self.rt_column = rt_column
        self.precursor_mz_column = precursor_mz_column
        self.fragment_mz_column = fragment_mz_column
        self.mobility_column = mobility_column

        gaussian_filter = GaussianKernel(
            dia_data,
            fwhm_rt=fwhm_rt,
            sigma_scale_rt=config.sigma_scale_rt,
            fwhm_mobility=fwhm_mobility,
            sigma_scale_mobility=config.sigma_scale_mobility,
            kernel_width=config.kernel_size,
            kernel_height=min(config.kernel_size, dia_data.scan_max_index + 1),
        )
        self.kernel = gaussian_filter.get_dense_matrix()

        self.available_isotopes = utils.get_isotope_columns(
            self.precursors_flat.columns
        )
        self.available_isotope_columns = [f"i_{i}" for i in self.available_isotopes]

        self.config = config
        self.feature_path = feature_path

    def __call__(self, thread_count=10, debug=False):
        """
        Perform candidate extraction workflow.
        1. First, elution groups are assembled based on the annotation in the flattened precursor dataframe.
        Each elution group is instantiated as an ElutionGroup Numba JIT object.
        Elution groups are stored in the ElutionGroupContainer Numba JIT object.

        2. Then, the elution groups are iterated over and the candidates are selected.
        The candidate selection is performed in parallel using the alphatims.utils.pjit function.

        3. Finally, the candidates are collected from the ElutionGroup,
        assembled into a pandas.DataFrame and precursor information is appended.
        """

        logging.info("Starting candidate selection")

        # initialize input container
        precursor_container = self.assemble_precursor_df(self.precursors_flat)
        candidate_container = CandidateDF(
            len(self.precursors_flat) * self.config.candidate_count
        )
        fragment_container = self.assemble_fragments()

        # if debug mode, only iterate over 10 elution groups
        iterator_len = (
            min(10, len(self.precursors_flat)) if debug else len(self.precursors_flat)
        )
        thread_count = 1 if debug else thread_count

        alphatims.utils.set_threads(thread_count)

        _executor(
            range(iterator_len),
            self.dia_data,
            precursor_container,
            candidate_container,
            fragment_container,
            self.config,
            self.kernel,
            debug,
        )

        return self.collect_candidates(candidate_container)

    def collect_candidates(self, candidate_container):
        candidate_df = pd.DataFrame(
            {
                key: value
                for key, value in zip(
                    [
                        "precursor_idx",
                        "rank",
                        "score",
                        "scan_center",
                        "scan_start",
                        "scan_stop",
                        "frame_center",
                        "frame_start",
                        "frame_stop",
                    ],
                    candidate_container.to_candidate_df(),
                    strict=True,
                )
            }
        )
        candidate_df = candidate_df.merge(
            self.precursors_flat[["precursor_idx", "elution_group_idx", "decoy"]],
            on="precursor_idx",
            how="left",
        )
        return candidate_df

    def assemble_fragments(self):
        # set cardinality to 1 if not present
        if "cardinality" in self.fragments_flat.columns:
            self.fragments_flat["cardinality"] = self.fragments_flat[
                "cardinality"
            ].values

        else:
            logging.warning(
                "Fragment cardinality column not found in fragment dataframe. Setting cardinality to 1."
            )
            self.fragments_flat["cardinality"] = np.ones(
                len(self.fragments_flat), dtype=np.uint8
            )

        # validate dataframe schema and prepare jitclass compatible dtypes
        validate.fragments_flat(self.fragments_flat)

        return fragments.FragmentContainer(
            self.fragments_flat["mz_library"].values,
            self.fragments_flat[self.fragment_mz_column].values,
            self.fragments_flat["intensity"].values,
            self.fragments_flat["type"].values,
            self.fragments_flat["loss_type"].values,
            self.fragments_flat["charge"].values,
            self.fragments_flat["number"].values,
            self.fragments_flat["position"].values,
            self.fragments_flat["cardinality"].values,
        )

    def assemble_precursor_df(self, precursors_flat):
        # validate dataframe schema and prepare jitclass compatible dtypes
        validate.precursors_flat(precursors_flat)

        available_isotopes = utils.get_isotope_columns(precursors_flat.columns)
        available_isotope_columns = [f"i_{i}" for i in available_isotopes]

        candidate_start_index = np.arange(
            0,
            len(precursors_flat) * self.config.candidate_count,
            self.config.candidate_count,
            dtype=np.uint32,
        )
        candidate_stop_index = (
            candidate_start_index + self.config.candidate_count
        ).astype(np.uint32)

        return PrecursorFlatDF(
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

    def assemble_candidates(self, elution_group_container):
        """
        Candidates are collected from the ElutionGroup objects and assembled into a pandas.DataFrame.

        Parameters
        ----------
        elution_group_container : ElutionGroupContainer
            container object containing a list of ElutionGroup objects

        Returns
        -------
        pandas.DataFrame
            dataframe containing the extracted candidates

        """

        candidates = []
        for i in range(len(elution_group_container)):
            for j in range(len(elution_group_container[i].candidates)):
                candidates.append(elution_group_container[i].candidates[j])

        candidate_attributes = [
            "elution_group_idx",
            "score_group_idx",
            "precursor_idx",
            "rank",
            "score",
            "precursor_mz",
            "decoy",
            "channel",
            "scan_center",
            "scan_start",
            "scan_stop",
            "frame_center",
            "frame_start",
            "frame_stop",
        ]
        candidate_df = pd.DataFrame(
            {
                attr: [getattr(c, attr) for c in candidates]
                for attr in candidate_attributes
            }
        )
        candidate_df = candidate_df.sort_values(by="precursor_idx")

        # add additiuonal columns for precursor information
        precursor_attributes = ["mz_calibrated", "mz_library"]

        precursor_pidx = self.precursors_flat["precursor_idx"].values
        candidate_pidx = candidate_df["precursor_idx"].values
        precursor_flat_lookup = np.searchsorted(
            precursor_pidx, candidate_pidx, side="left"
        )

        for attr in precursor_attributes:
            if attr in self.precursors_flat.columns:
                candidate_df[attr] = self.precursors_flat[attr].values[
                    precursor_flat_lookup
                ]

        # DEBUG: save features for training if desired.
        if self.feature_path is not None:
            feature_matrix = np.zeros(
                (len(candidates), len(candidates[0].features)), dtype=np.float32
            )
            for i in range(len(candidates)):
                feature_matrix[i, :] = candidates[i].features

            np.save(os.path.join(self.feature_path, "features.npy"), feature_matrix)

            sub_df = candidate_df[
                [
                    "elution_group_idx",
                    "score_group_idx",
                    "precursor_idx",
                    "rank",
                    "score",
                    "decoy",
                ]
            ]
            sub_df.to_csv(
                os.path.join(self.feature_path, "candidates.tsv"), index=False, sep="\t"
            )

        return candidate_df

    def append_precursor_information(self, df):
        """
        Append relevant precursor information to the candidates dataframe.

        Parameters
        ----------
        df : pandas.DataFrame
            dataframe containing the extracted candidates

        Returns
        -------
        pandas.DataFrame
            dataframe containing the extracted candidates with precursor information appended
        """

        # precursor_flat_lookup has an element for every candidate and contains the index of the respective precursor
        precursor_pidx = self.precursors_flat["precursor_idx"].values
        candidate_pidx = df["precursor_idx"].values
        precursor_flat_lookup = np.searchsorted(
            precursor_pidx, candidate_pidx, side="left"
        )

        df["decoy"] = self.precursors_flat["decoy"].values[precursor_flat_lookup]

        df["rt_library"] = self.precursors_flat["rt_library"].values[
            precursor_flat_lookup
        ]
        if self.rt_column == "rt_calibrated":
            df["rt_calibrated"] = self.precursors_flat["rt_calibrated"].values[
                precursor_flat_lookup
            ]

        df["mobility_library"] = self.precursors_flat["mobility_library"].values[
            precursor_flat_lookup
        ]
        if self.mobility_column == "mobility_calibrated":
            df["mobility_calibrated"] = self.precursors_flat[
                "mobility_calibrated"
            ].values[precursor_flat_lookup]

        df["flat_frag_start_idx"] = self.precursors_flat["flat_frag_start_idx"].values[
            precursor_flat_lookup
        ]
        df["flat_frag_stop_idx"] = self.precursors_flat["flat_frag_stop_idx"].values[
            precursor_flat_lookup
        ]
        df["charge"] = self.precursors_flat["charge"].values[precursor_flat_lookup]
        df["proteins"] = self.precursors_flat["proteins"].values[precursor_flat_lookup]
        df["genes"] = self.precursors_flat["genes"].values[precursor_flat_lookup]

        available_isotopes = utils.get_isotope_columns(self.precursors_flat.columns)
        available_isotope_columns = [f"i_{i}" for i in available_isotopes]

        for col in available_isotope_columns:
            df[col] = self.precursors_flat[col].values[precursor_flat_lookup]

        return df
