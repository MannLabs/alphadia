# native imports
import logging

logger = logging.getLogger()
import os
import time
import typing

# alphadia imports
from alphadia import utils
from alphadia.numba import fragments, numeric, config
from alphadia import validate, utils
from alphadia.data import bruker, thermo

# alpha family imports
import alphatims

# third party imports
import numba as nb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patches
import matplotlib as mpl


class GaussianFilter:
    def __init__(
        self,
        dia_data: typing.Union[
            bruker.TimsTOFTransposeJIT, bruker.TimsTOFTranspose, thermo.Thermo
        ],
        fwhm_rt: float = 10.0,
        sigma_scale_rt: float = 1.0,
        fwhm_mobility: float = 0.03,
        sigma_scale_mobility: float = 1.0,
        kernel_height: int = 30,
        kernel_width: int = 30,
    ):
        """
        Create a two-dimensional gaussian filter kernel for the RT and mobility dimensions of a DIA dataset.
        First, the observed standard deviation is scaled by a linear factor. Second, the standard deviation is scaled by the resolution of the respective dimension.

        This results in sigma_scale to be independent of the resolution of the data and FWHM of the peaks.

        Parameters
        ----------

        dia_data : typing.Union[bruker.TimsTOFTransposeJIT, bruker.TimsTOFTranspose]
            alphatims dia_data object.

        fwhm_rt : float
            Full width at half maximum in RT dimension of the peaks in the spectrum.

        sigma_scale_rt : float
            Scaling factor for the standard deviation in RT dimension.

        fwhm_mobility : float
            Full width at half maximum in mobility dimension of the peaks in the spectrum.

        sigma_scale_mobility : float
            Scaling factor for the standard deviation in mobility dimension.

        kernel_size : int
            Kernel shape in pixel. The kernel will be a square of size (kernel_size, kernel_size).
            Should be even and will be rounded up to the next even number if necessary.

        """
        self.dia_data = dia_data
        self.fwhm_rt = fwhm_rt
        self.sigma_scale_rt = sigma_scale_rt
        self.fwhm_mobility = fwhm_mobility
        self.sigma_scale_mobility = sigma_scale_mobility

        self.kernel_height = int(
            np.ceil(kernel_height / 2) * 2
        )  # make sure kernel size is even
        self.kernel_width = int(
            np.ceil(kernel_width / 2) * 2
        )  # make sure kernel size is even

    def determine_rt_sigma(self, cycle_length_seconds: float):
        """
        Determine the standard deviation of the gaussian kernel in RT dimension.
        The standard deviation will be sclaed to the resolution of the raw data.

        Parameters
        ----------

        cycle_length_seconds : float
            Cycle length of the duty cycle in seconds.

        Returns
        -------

        float
            Standard deviation of the gaussian kernel in RT dimension scaled to the resolution of the raw data.
        """
        # a normal distribution has a FWHM of 2.3548 sigma
        sigma = self.fwhm_rt / 2.3548
        sigma_scaled = sigma * self.sigma_scale_rt / cycle_length_seconds
        return sigma_scaled

    def determine_mobility_sigma(self, mobility_resolution: float):
        """
        Determine the standard deviation of the gaussian kernel in mobility dimension.
        The standard deviation will be sclaed to the resolution of the raw data.

        Parameters
        ----------

        mobility_resolution : float
            Resolution of the mobility dimension in 1/K_0.

        Returns
        -------

        float
            Standard deviation of the gaussian kernel in mobility dimension scaled to the resolution of the raw data.
        """

        if not self.dia_data.has_mobility:
            return 1.0

        # a normal distribution has a FWHM of 2.3548 sigma
        sigma = self.fwhm_mobility / 2.3548
        sigma_scaled = sigma * self.sigma_scale_mobility / mobility_resolution
        return sigma_scaled

    def get_kernel(self, verbose: bool = True):
        """
        Calculate the gaussian kernel for the given data set and parameters.

        Parameters
        ----------

        verbose : bool
            If True, log information about the data set and the kernel.

        Returns
        -------

        np.ndarray
            Two-dimensional gaussian kernel.

        """

        rt_datapoints = self.dia_data.cycle.shape[1]
        rt_resolution = np.mean(np.diff(self.dia_data.rt_values[::rt_datapoints]))

        mobility_datapoints = self.dia_data.cycle.shape[2]
        mobility_resolution = np.mean(np.diff(self.dia_data.mobility_values[::-1]))

        if verbose:
            pass
            logger.info(
                f"Duty cycle consists of {rt_datapoints} frames, {rt_resolution:.2f} seconds cycle time"
            )
            logger.info(
                f"Duty cycle consists of {mobility_datapoints} scans, {mobility_resolution:.5f} 1/K_0 resolution"
            )

        rt_sigma = self.determine_rt_sigma(rt_resolution)
        mobility_sigma = self.determine_mobility_sigma(mobility_resolution)

        if verbose:
            pass
            logger.info(
                f"FWHM in RT is {self.fwhm_rt:.2f} seconds, sigma is {rt_sigma:.2f}"
            )
            logger.info(
                f"FWHM in mobility is {self.fwhm_mobility:.3f} 1/K_0, sigma is {mobility_sigma:.2f}"
            )

        return self.gaussian_kernel_2d(
            self.kernel_width, self.kernel_height, rt_sigma, mobility_sigma
        ).astype(np.float32)

    @staticmethod
    def gaussian_kernel_2d(size_x: int, size_y: int, sigma_x: float, sigma_y: float):
        """
        Create a 2D gaussian kernel with a given size and standard deviation.

        Parameters
        ----------

        size : int
            Width and height of the kernel matrix.

        sigma_x : float
            Standard deviation of the gaussian kernel in x direction. This will correspond to the RT dimension.

        sigma_y : float
            Standard deviation of the gaussian kernel in y direction. This will correspond to the mobility dimension.

        Returns
        -------

        weights : np.ndarray, dtype=np.float32
            2D gaussian kernel matrix of shape (size, size).

        """
        # create indicies [-2, -1, 0, 1 ...]
        x, y = np.meshgrid(
            np.arange(-size_x // 2, size_x // 2), np.arange(-size_y // 2, size_y // 2)
        )
        xy = np.column_stack((x.flatten(), y.flatten())).astype("float32")

        # mean is always zero
        mu = np.array([[0.0, 0.0]])

        # sigma is set with no covariance
        sigma_mat = np.array([[sigma_x, 0.0], [0.0, sigma_y]])

        weights = utils.multivariate_normal(xy, mu, sigma_mat)
        return weights.reshape(size_y, size_x).astype(np.float32)


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


@nb.experimental.jitclass()
class Candidate:

    """A candidate is a region in the rt, mobility space which likely contains a given precursor.
    The class is numba JIT compatible and will receive all values during initialization.
    """

    elution_group_idx: nb.int64
    score_group_idx: nb.int64
    precursor_idx: nb.int64
    rank: nb.int64

    score: nb.float64
    precursor_mz: nb.float64
    decoy: nb.int8
    channel: nb.uint32
    features: nb.float32[::1]

    scan_center: nb.int64
    scan_start: nb.int64
    scan_stop: nb.int64

    frame_center: nb.int64
    frame_start: nb.int64
    frame_stop: nb.int64

    def __init__(
        self,
        elution_group_idx,
        score_group_idx,
        precursor_idx,
        rank,
        score,
        precursor_mz,
        decoy,
        channel,
        features,
        scan_center,
        scan_start,
        scan_stop,
        frame_center,
        frame_start,
        frame_stop,
    ):
        self.elution_group_idx = elution_group_idx
        self.score_group_idx = score_group_idx
        self.precursor_idx = precursor_idx
        self.rank = rank
        self.score = score
        self.precursor_mz = precursor_mz
        self.decoy = decoy
        self.channel = channel
        self.features = features
        self.scan_center = scan_center
        self.scan_start = scan_start
        self.scan_stop = scan_stop
        self.frame_center = frame_center
        self.frame_start = frame_start
        self.frame_stop = frame_stop


# define the numba type of the class for use in other numba functions
candidate_type = Candidate.class_type.instance_type


@nb.experimental.jitclass()
class HybridElutionGroup:
    # values which are shared by all precursors in the elution group
    # (1)
    score_group_idx: nb.uint32
    elution_group_idx: nb.uint32
    rt: nb.float32
    mobility: nb.float32
    charge: nb.uint8
    status_code: nb.uint8
    # 100: no fragment masses after grouping
    # 101: no precursor masses after grouping
    # 102: wrong quadrupole mz shape
    # 103: empty dense precursor matrix
    # 104: empty dense fragment matrix
    # 105: dense precursor matrix not divisible by 2
    # 106: dense fragment matrix not divisible by 2
    # 107: precursor or fragment matrix smaller than convolution kernel

    # values which are specific to each precursor in the elution group
    # (n_precursor)
    precursor_idx: nb.uint32[::1]
    precursor_channel: nb.uint32[::1]
    precursor_decoy: nb.uint8[::1]
    precursor_mz: nb.float32[::1]
    precursor_score_group: nb.int32[::1]
    precursor_abundance: nb.float32[::1]

    # (n_precursor, 2)
    precursor_frag_start_stop_idx: nb.uint32[:, ::1]

    # (n_precursor, n_isotopes)
    precursor_isotope_intensity: nb.float32[:, :]
    precursor_isotope_mz: nb.float32[:, ::1]

    frame_limits: nb.uint64[:, ::1]
    scan_limits: nb.uint64[:, ::1]
    precursor_tof_limits: nb.uint64[:, ::1]
    fragment_tof_limits: nb.uint64[:, ::1]

    candidate_precursor_idx: nb.uint32[::1]
    candidate_mass_error: nb.float64[::1]
    candidate_fraction_nonzero: nb.float64[::1]
    candidate_intensity: nb.float32[::1]

    candidate_scan_limit: nb.int64[:, ::1]
    candidate_frame_limit: nb.int64[:, ::1]

    candidate_scan_center: nb.int64[::1]
    candidate_frame_center: nb.int64[::1]

    candidates: nb.types.ListType(candidate_type)

    # only for debugging

    fragment_lib: fragments.FragmentContainer.class_type.instance_type

    dense_fragments: nb.float32[:, :, :, ::1]
    dense_precursors: nb.float32[:, :, :, ::1]

    score_group_fragment_mz: nb.float32[::1]
    score_group_precursor_mz: nb.float32[::1]

    score_group_precursor_intensity: nb.float32[::1]
    score_group_fragment_intensity: nb.float32[::1]

    def __init__(
        self,
        score_group_idx,
        elution_group_idx,
        precursor_idx,
        channel,
        frag_start_stop_idx,
        rt,
        mobility,
        charge,
        decoy,
        mz,
        isotope_intensity,
    ) -> None:
        """
        ElutionGroup jit class which contains all information about a single elution group.

        Parameters
        ----------
        elution_group_idx : int
            index of the elution group as encoded in the precursor dataframe

        precursor_idx : numpy.ndarray
            indices of the precursors in the precursor dataframe

        rt : float
            retention time of the elution group in seconds, shared by all precursors

        mobility : float
            mobility of the elution group, shared by all precursors

        charge : int
            charge of the elution group, shared by all precursors

        decoy : numpy.ndarray
            array of integers indicating whether the precursor is a decoy (1) or target (0)

        mz : numpy.ndarray
            array of m/z values of the precursors

        isotope_apex_offset : numpy.ndarray
            array of integers indicating the offset of the isotope apex from the precursor m/z.
        """

        self.score_group_idx = score_group_idx
        self.elution_group_idx = elution_group_idx
        self.precursor_idx = precursor_idx
        self.rt = rt
        self.mobility = mobility
        self.charge = charge

        self.precursor_decoy = decoy
        self.precursor_mz = mz
        self.precursor_channel = channel
        self.precursor_frag_start_stop_idx = frag_start_stop_idx
        self.precursor_isotope_intensity = isotope_intensity
        self.candidates = nb.typed.List.empty_list(candidate_type)

        self.status_code = 0

    def __str__(self):
        with nb.objmode(r="unicode_type"):
            r = f"ElutionGroup(\nelution_group_idx: {self.elution_group_idx},\nprecursor_idx: {self.precursor_idx}\n)"
        return r

    def sort_by_mz(self):
        """
        Sort all precursor arrays by m/z

        """
        mz_order = np.argsort(self.precursor_mz)
        self.precursor_mz = self.precursor_mz[mz_order]
        self.precursor_decoy = self.precursor_decoy[mz_order]
        self.precursor_idx = self.precursor_idx[mz_order]
        self.precursor_frag_start_stop_idx = self.precursor_frag_start_stop_idx[
            mz_order
        ]

    def assemble_isotope_mz(self):
        """
        Assemble the isotope m/z values from the precursor m/z and the isotope
        offsets.
        """
        offset = (
            np.arange(self.precursor_isotope_intensity.shape[1])
            * 1.0033548350700006
            / self.charge
        )
        self.precursor_isotope_mz = np.expand_dims(self.precursor_mz, 1).astype(
            np.float32
        ) + np.expand_dims(offset, 0).astype(np.float32)

    def trim_isotopes(self):
        divisor = self.precursor_isotope_intensity.shape[0]

        if divisor == 0:
            raise ZeroDivisionError("Cannot divide by zero")

        elution_group_isotopes = (
            np.sum(self.precursor_isotope_intensity, axis=0) / divisor
        )
        self.precursor_isotope_intensity = self.precursor_isotope_intensity[
            :, elution_group_isotopes > 0.01
        ]

    def set_status(self, status_code, status_message=None):
        if status_message is not None:
            pass
            # print(status_code, status_message)
        self.status_code = status_code

    def calculate_score_group_limits(self, precursor_mz, precursor_intensity):
        quadrupole_mz = np.zeros((1, 2))

        mask = precursor_intensity > 0

        quadrupole_mz[0, 0] = precursor_mz[mask].min()
        quadrupole_mz[0, 1] = precursor_mz[mask].max()

        return quadrupole_mz

    def process(self, jit_data, fragment_container, config, kernel, debug):
        # print(self.precursor_idx)
        # print(self.precursor_decoy)

        """
        Process the elution group and store the candidates.

        Parameters
        ----------

        jit_data : alphadia.bruker.TimsTOFJIT
            TimsTOFJIT object containing the raw data

        kernel : np.ndarray
            Matrix of size (20, 20) containing the smoothing kernel

        rt_tolerance : float
            tolerance in seconds

        mobility_tolerance : float
            tolerance in inverse mobility units

        mz_tolerance : float
            tolerance in part per million (ppm)

        candidate_count : int
            number of candidates to select per precursor.

        debug : bool
            if True, self.visualize_candidates() will be called after processing the elution group.
            Make sure to use debug mode only on a small number of elution groups (10) and with a single thread.
        """

        precursor_abundance = np.ones((len(self.precursor_decoy)), dtype=np.float32)
        precursor_abundance[self.precursor_channel == 0] = 10

        self.precursor_abundance = precursor_abundance

        self.sort_by_mz()
        self.trim_isotopes()
        self.assemble_isotope_mz()

        fragment_idx_slices = utils.make_slice_2d(self.precursor_frag_start_stop_idx)

        fragment_lib = fragments.slice_manual(fragment_container, fragment_idx_slices)
        if config.exclude_shared_ions:
            fragment_lib.filter_by_cardinality(1)
        fragment_lib.sort_by_mz()

        if len(fragment_lib.precursor_idx) <= 3:
            self.set_status(100, "No fragment masses after grouping")
            return

        # print('fragment_lib.precursor_idx',len(fragment_lib.precursor_idx))

        if debug:
            self.fragment_lib = fragment_lib

        frame_limits = jit_data.get_frame_indices_tolerance(
            self.rt, config.rt_tolerance
        )
        scan_limits = jit_data.get_scan_indices_tolerance(
            self.mobility, config.mobility_tolerance
        )

        # with nb.objmode():
        fragment_mz, fragment_intensity = fragments.get_ion_group_mapping(
            fragment_lib.precursor_idx,
            fragment_lib.mz,
            fragment_lib.intensity,
            fragment_lib.cardinality,
            self.precursor_abundance,
            top_k=config.top_k_fragments,
        )

        # print(nb.typeof(fragment_mz), fragment_mz.shape)
        # print(nb.typeof(fragment_intensity), fragment_intensity.shape)

        # return if no valid fragments are left after grouping

        # FLAG: needed for debugging
        self.score_group_fragment_mz = fragment_mz
        self.score_group_fragment_intensity = fragment_intensity

        # shape = (n_fragments, 3, ), dtype = np.int64
        # fragment_tof_limits = jit_data.get_tof_indices_tolerance(fragment_mz, mz_tolerance)

        isotope_mz = self.precursor_isotope_mz.flatten()

        isotope_intensity = self.precursor_isotope_intensity.flatten()

        # this is the precursor index for each isotope
        isotope_precursor = np.repeat(
            np.arange(0, self.precursor_isotope_mz.shape[0], dtype=np.int64),
            self.precursor_isotope_mz.shape[1],
        )

        order = np.argsort(isotope_mz)
        isotope_mz = isotope_mz[order]
        isotope_intensity = isotope_intensity[order]
        isotope_precursor = isotope_precursor[order]

        precursor_mz, precursor_intensity = fragments.get_ion_group_mapping(
            isotope_precursor,
            isotope_mz,
            isotope_intensity,
            np.ones(len(isotope_mz), dtype=np.uint8),
            self.precursor_abundance,
            top_k=config.top_k_precursors,
        )

        # FLAG: needed for debugging
        self.score_group_precursor_mz = precursor_mz
        self.score_group_precursor_intensity = precursor_intensity

        # shape = (n_precursor_isotopes, 3, ), dtype = np.int64
        # precursor_tof_limits = jit_data.get_tof_indices_tolerance(precursor_mz, mz_tolerance)

        quadrupole_mz = self.calculate_score_group_limits(
            precursor_mz,
            precursor_intensity,
        )

        # return if no valid precursors are left after grouping
        if len(precursor_mz) == 0:
            self.set_status(101, "No precursor masses after grouping")
            return

        #if jit_data.has_mobility:
        # shape = (2, n_fragments, n_observations, n_scans, n_frames), dtype = np.float32
        _dense_precursors, _ = jit_data.get_dense(
            frame_limits,
            scan_limits,
            precursor_mz,
            config.precursor_mz_tolerance,
            np.array([[-1.0, -1.0]], dtype=np.float32),
        )

        dense_precursors = _dense_precursors.sum(axis=2)

        # FLAG: needed for debugging
        # self.dense_precursors = dense_precursors

        if not quadrupole_mz.shape == (
            1,
            2,
        ):
            self.set_status(102, "Unexpected quadrupole_mz.shape")
            return

        #if jit_data.has_mobility:
        # shape = (2, n_fragments, n_observations, n_scans, n_frames), dtype = np.float32
        _dense_fragments, _ = jit_data.get_dense(
            frame_limits,
            scan_limits,
            fragment_mz,
            config.fragment_mz_tolerance,
            quadrupole_mz,
            custom_cycle=jit_data.cycle,
        )

        dense_fragments = _dense_fragments.sum(axis=2)

        # FLAG: needed for debugging
        # self.dense_fragments = dense_fragments

        # perform sanity checks
        if dense_fragments.shape[0] == 0:
            self.set_status(103, "Empty dense fragment matrix")
            return

        if dense_precursors.shape[0] == 0:
            self.set_status(104, "Empty dense precursor matrix")
            return

        if not dense_fragments.shape[2] % 2 == 0:
            self.set_status(105, "Dense fragment matrix not divisible by 2")
            return

        if not dense_fragments.shape[2] % 2 == 0:
            self.set_status(106, "Dense fragment matrix not divisible by 2")
            return

        if (
            dense_precursors.shape[2] < kernel.shape[0]
            or dense_precursors.shape[3] < kernel.shape[1]
        ):
            self.set_status(107, "Precursor matrix smaller than convolution kernel")
            return

        if (
            dense_fragments.shape[2] < kernel.shape[0]
            or dense_fragments.shape[3] < kernel.shape[1]
        ):
            self.set_status(108, "Fragment matrix smaller than convolution kernel")
            return

        if config.use_weighted_score:
            mean = config.feature_mean
            std = config.feature_std
            weights = config.feature_weight

        else:
            mean = None
            std = None
            weights = None #np.array([1, 1, 1, 1, 1, 1, 1, 1], np.float64)

        self.candidates = build_candidates(
            dense_precursors,
            dense_fragments,
            precursor_intensity,
            fragment_intensity,
            kernel,
            jit_data,
            config,
            self.elution_group_idx,
            self.score_group_idx,
            self.precursor_idx,
            self.precursor_decoy,
            self.precursor_channel,
            scan_limits,
            frame_limits,
            precursor_mz,
            candidate_count=config.candidate_count,
            debug=debug,
            weights=weights,
            mean=mean,
            std=std,
        )

        return

    def visualize_candidates(self, smooth_dense):
        """
        Visualize the candidates of the elution group using numba objmode.

        Parameters
        ----------

        dense : np.ndarray
            The raw, dense intensity matrix of the elution group.
            Shape: (2, n_precursors, n_observations ,n_scans, n_cycles)
            n_observations is indexed based on the 'precursor' index within a DIA cycle.

        smooth_dense : np.ndarray
            Dense data of the elution group after smoothing.
            Shape: (n_precursors, n_observations, n_scans, n_cycles)

        """
        with nb.objmode():
            n_precursors = len(self.precursor_idx)

            fig, axs = plt.subplots(n_precursors, 2, figsize=(10, n_precursors * 3))

            if axs.shape == (2,):
                axs = axs.reshape(1, 2)

            # iterate all precursors
            for j, idx in enumerate(self.precursor_idx):
                axs[j, 0].set_xlabel("cycle")
                axs[j, 0].set_ylabel("scan")
                axs[j, 0].set_title(
                    f"- RAW DATA - elution group: {self.elution_group_idx}, precursor: {idx}"
                )

                axs[j, 1].imshow(smooth_dense[j], aspect="auto")
                axs[j, 1].set_xlabel("cycle")
                axs[j, 1].set_ylabel("scan")
                axs[j, 1].set_title(
                    f"- Candidates - elution group: {self.elution_group_idx}, precursor: {idx}"
                )

                candidate_mask = self.candidate_precursor_idx == idx
                for k, (
                    scan_limit,
                    scan_center,
                    frame_limit,
                    frame_center,
                ) in enumerate(
                    zip(
                        self.candidate_scan_limit[candidate_mask],
                        self.candidate_scan_center[candidate_mask],
                        self.candidate_frame_limit[candidate_mask],
                        self.candidate_frame_center[candidate_mask],
                    )
                ):
                    axs[j, 1].scatter(frame_center, scan_center, c="r", s=10)

                    axs[j, 1].text(frame_limit[1], scan_limit[0], str(k), color="r")

                    axs[j, 1].add_patch(
                        patches.Rectangle(
                            (frame_limit[0], scan_limit[0]),
                            frame_limit[1] - frame_limit[0],
                            scan_limit[1] - scan_limit[0],
                            fill=False,
                            edgecolor="r",
                        )
                    )

            fig.tight_layout()
            plt.show()


@nb.experimental.jitclass()
class HybridElutionGroupContainer:
    elution_groups: nb.types.ListType(HybridElutionGroup.class_type.instance_type)

    def __init__(
        self,
        elution_groups,
    ) -> None:
        """
        Container class which contains a list of ElutionGroup objects.

        Parameters
        ----------
        elution_groups : nb.types.ListType(ElutionGroup.class_type.instance_type)
            List of ElutionGroup objects.

        """

        self.elution_groups = elution_groups

    def __getitem__(self, idx):
        return self.elution_groups[idx]

    def __len__(self):
        return len(self.elution_groups)


class HybridCandidateSelection(object):
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

        gaussian_filter = GaussianFilter(
            dia_data,
            fwhm_rt=fwhm_rt,
            sigma_scale_rt=config.sigma_scale_rt,
            fwhm_mobility=fwhm_mobility,
            sigma_scale_mobility=config.sigma_scale_mobility,
            kernel_width=config.kernel_size,
            kernel_height=min(config.kernel_size, dia_data.scan_max_index + 1),
        )
        self.kernel = gaussian_filter.get_kernel()

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
        elution_group_container = self.assemble_score_groups(self.precursors_flat)
        fragment_container = self.assemble_fragments()

        # if debug mode, only iterate over 10 elution groups
        iterator_len = (
            min(10, len(elution_group_container))
            if debug
            else len(elution_group_container)
        )
        thread_count = 1 if debug else thread_count

        alphatims.utils.set_threads(thread_count)

        _executor(
            range(iterator_len),
            elution_group_container,
            self.dia_data,
            fragment_container,
            self.config,
            self.kernel,
            debug,
        )

        # return elution_group_container

        df = self.assemble_candidates(elution_group_container)
        df = self.append_precursor_information(df)
        # self.log_stats(df)
        if debug:
            return elution_group_container, df

        logging.info("Finished candidate selection")

        del elution_group_container
        del fragment_container

        return df

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

    def assemble_score_groups(self, precursors_flat):
        """
        Assemble elution groups from precursor library.

        Parameters
        ----------

        precursors_flat : pandas.DataFrame
            Precursor library.

        Returns
        -------
        HybridElutionGroupContainer
            Numba jitclass with list of elution groups.
        """

        if len(precursors_flat) == 0:
            return

        available_isotopes = utils.get_isotope_columns(precursors_flat.columns)
        available_isotope_columns = [f"i_{i}" for i in available_isotopes]

        precursors_sorted = utils.calculate_score_groups(
            precursors_flat, self.config.group_channels
        ).copy()

        # validate dataframe schema and prepare jitclass compatible dtypes
        validate.precursors_flat(precursors_sorted)

        @nb.njit(debug=True)
        def assemble_njit(
            score_group_idx,
            elution_group_idx,
            precursor_idx,
            channel,
            flat_frag_start_stop_idx,
            rt_values,
            mobility_values,
            charge,
            decoy,
            precursor_mz,
            isotope_intensity,
        ):
            score_group = score_group_idx[0]
            score_group_start = 0
            score_group_stop = 0

            eg_list = []

            while score_group_stop < len(score_group_idx):
                score_group_stop += 1

                if score_group_idx[score_group_stop] != score_group:
                    eg_list.append(
                        HybridElutionGroup(
                            score_group,
                            elution_group_idx[score_group_start],
                            precursor_idx[score_group_start:score_group_stop],
                            channel[score_group_start:score_group_stop],
                            flat_frag_start_stop_idx[
                                score_group_start:score_group_stop
                            ],
                            rt_values[score_group_start],
                            mobility_values[score_group_start],
                            charge[score_group_start],
                            decoy[score_group_start:score_group_stop],
                            precursor_mz[score_group_start:score_group_stop],
                            isotope_intensity[score_group_start:score_group_stop],
                        )
                    )

                    score_group_start = score_group_stop
                    score_group = score_group_idx[score_group_start]

            egs = nb.typed.List(eg_list)
            return HybridElutionGroupContainer(egs)

        return assemble_njit(
            precursors_sorted["score_group_idx"].values,
            precursors_sorted["elution_group_idx"].values,
            precursors_sorted["precursor_idx"].values,
            precursors_sorted["channel"].values,
            precursors_sorted[
                ["flat_frag_start_idx", "flat_frag_stop_idx"]
            ].values.copy(),
            precursors_sorted[self.rt_column].values,
            precursors_sorted[self.mobility_column].values,
            precursors_sorted["charge"].values,
            precursors_sorted["decoy"].values,
            precursors_sorted[self.precursor_mz_column].values,
            precursors_sorted[available_isotope_columns].values.copy(),
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

        # save features for training if desired.
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


from tqdm import tqdm


@alphatims.utils.pjit()
def _executor(
    i,
    eg_container,
    jit_data,
    fragment_container,
    config,
    kernel,
    debug,
):
    """
    Helper function.
    Is decorated with alphatims.utils.pjit to enable parallel execution of HybridElutionGroup.process.
    """

    eg_container[i].process(jit_data, fragment_container, config, kernel, debug)


@nb.njit
def build_features(
    smooth_precursor, smooth_fragment, precursor_intensity, fragment_intensity
):
    n_features = 1  # 2

    features = np.zeros(
        (
            n_features,
            smooth_precursor.shape[2],
            smooth_fragment.shape[3],
        ),
        dtype=np.float32,
    )

    # top fragment
    #frag_order = np.argsort(fragment_intensity)[::-1]

    #precursor_kernel = precursor_intensity.reshape(-1, 1, 1)
    #sfragment_kernel = fragment_intensity.reshape(-1, 1, 1)

    #smooth_fragment = smooth_fragment[:, frag_order]

    # fragment_binary = smooth_fragment[0] > 2
    # fragment_binary_sum = np.sum(fragment_binary, axis=0)
    # fragment_binary_weighted = np.sum(fragment_binary * fragment_kernel, axis=0)

    # precursor_binary = smooth_precursor[0] > 2
    # precursor_binary_sum = np.sum(precursor_binary, axis=0)
    # precursor_binary_weighted = np.sum(precursor_binary * precursor_kernel, axis=0)

    # precursor_dot = np.sum(smooth_precursor[0] * precursor_kernel, axis=0)
    # precursor_dot_mean = np.mean(precursor_dot)
    # precursor_norm = precursor_dot/(precursor_dot_mean+0.001)

    # fragment_dot = np.sum(smooth_fragment[0] * fragment_kernel, axis=0)
    # fragment_dot_mean = np.mean(fragment_dot)
    # fragment_norm = fragment_dot/(fragment_dot_mean+0.001)

    log_fragment = np.sum(np.log(smooth_fragment[0] + 1), axis=0)
    log_precursor = np.sum(np.log(smooth_precursor[0] + 1), axis=0)

    # fragment_mass_error = np.sum(np.abs(smooth_fragment[1]), axis=0)
    # fragment_mass_error_max = np.max(fragment_mass_error)
    # fragment_mass_error_norm = 1-(fragment_mass_error/fragment_mass_error_max)

    # precursor_mass_error = np.sum(np.abs(smooth_precursor[1]), axis=0)
    # precursor_mass_error_max = np.max(precursor_mass_error)
    # precursor_mass_error_norm = 1-(precursor_mass_error/precursor_mass_error_max)

    # isotope score

    # isotope_score = np.zeros(smooth_precursor.shape[2:], dtype=np.float32)
    # n_isotopes = precursor_intensity.shape[0]
    # for i in range(n_isotopes-1):
    #    if precursor_intensity[i] <= precursor_intensity[i+1]:
    #        isotope_score += smooth_precursor[0,i] <= smooth_precursor[0,i+1]
    #    else:
    #        isotope_score += smooth_precursor[0,i] >= smooth_precursor[0,i+1]

    # profile correlation
    # top3_profiles = np.sum(smooth_fragment[0,:3], axis=1)
    # top3_correlation = utils.profile_correlation(top3_profiles)

    # features[0] = fragment_binary_weighted
    # features[1] = np.log(fragment_norm +1)
    # features[2] = np.log(precursor_norm +1)
    features[0] = (
        log_fragment + log_precursor
    )  # np.log(fragment_norm +1) + np.log(precursor_norm +1)
    # features[4] = fragment_mass_error_norm
    # features[5] = precursor_mass_error_norm
    # features[6] = fragment_mass_error_norm * precursor_mass_error_norm
    # features[7] = np.log(smooth_fragment[0][:3].sum(axis=0) + 1)
    # features[8] = isotope_score
    # features[9] = top3_correlation[0] * fragment_binary[0]
    # features[10] = top3_correlation[1] * fragment_binary[0]
    # eatures[11] = top3_correlation[2] * fragment_binary[0]

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


def plot_candidates(
    score, dense_fragments, candidates, jit_data, scan_limits, frame_limits
):
    if len(candidates) == 0:
        return
    print("plotting candidates:", candidates[0].precursor_idx)

    height_px = score.shape[0]
    width_px = score.shape[1]

    # if mobility information is present, the 2d plot is more prominent
    has_mobility = height_px > 2

    if has_mobility:
        fig_size = (max(width_px / 500 * 8, 5), height_px / 100 * 5)
        gridspec_kw = {"height_ratios": [1, 9]}
    else:
        fig_size = (max(width_px / 500 * 8, 5), 5)
        gridspec_kw = {"height_ratios": [19, 1]}

    fig, axs = plt.subplots(
        nrows=2, ncols=1, figsize=fig_size, gridspec_kw=gridspec_kw, sharex=True
    )
    axs0_twin = axs[0].twinx()
    # 1d plot
    axs[0].plot(np.sum(score, axis=0))

    fragment_profiles = dense_fragments[0].sum(axis=1)
    colors = mpl.cm.jet(np.linspace(0, 1, fragment_profiles.shape[0]))
    for i, profile in enumerate(fragment_profiles):
        axs0_twin.plot(profile, color=colors[i])

    # 2d plot
    axs[1].imshow(score, cmap="coolwarm", interpolation="none")
    axs[1].set_ylabel("mobility")
    axs[1].set_xlabel("retention time")

    axs0_ylim = axs[0].get_ylim()
    axs[0].set_ylim(0, axs0_ylim[1])
    axs0_ylim = axs[0].get_ylim()

    absolute_scan = np.array([c.scan_center for c in candidates])
    absolute_frame = np.array([c.frame_center for c in candidates])

    relative_scan = absolute_scan - scan_limits[0, 0]
    relative_frame = (absolute_frame - frame_limits[0, 0]) // jit_data.cycle.shape[1]

    plt.scatter(relative_frame, relative_scan, c="red", s=1)

    absolute_scan_start = np.array([c.scan_start for c in candidates])
    absolute_scan_stop = np.array([c.scan_stop for c in candidates])
    absolute_frame_start = np.array([c.frame_start for c in candidates])
    absolute_frame_stop = np.array([c.frame_stop for c in candidates])

    relative_scan_start = absolute_scan_start - scan_limits[0, 0]
    relative_scan_stop = absolute_scan_stop - scan_limits[0, 0]
    relative_frame_start = (
        absolute_frame_start - frame_limits[0, 0]
    ) // jit_data.cycle.shape[1]
    relative_frame_stop = (
        absolute_frame_stop - frame_limits[0, 0]
    ) // jit_data.cycle.shape[1]

    ax = axs[1]
    for i in range(len(candidates)):
        rect = patches.Rectangle(
            (relative_frame_start[i], relative_scan_start[i]),
            relative_frame_stop[i] - relative_frame_start[i] - 1,
            relative_scan_stop[i] - relative_scan_start[i] - 1,
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        axs[1].add_patch(rect)
        # align the rank i at the top right corner of the box
        axs[1].text(
            relative_frame_stop[i] + 3,
            relative_scan_start[i],
            f"{i}",
            horizontalalignment="left",
            verticalalignment="top",
            color="red",
        )

        rect = patches.Rectangle(
            (relative_frame_start[i], 0),
            relative_frame_stop[i] - relative_frame_start[i] - 1,
            axs0_ylim[1],
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        axs[0].add_patch(rect)
        # align the rank i at the top right corner of the box
        axs[0].text(
            relative_frame_stop[i] + 3,
            axs0_ylim[1],
            f"{i}",
            horizontalalignment="left",
            verticalalignment="top",
            color="red",
        )

    plt.show()


@nb.njit
def build_candidates(
    dense_precursors,
    dense_fragments,
    precursor_intensity,
    fragment_intensity,
    kernel,
    jit_data,
    config,
    elution_group_idx,
    score_group_idx,
    precursor_idx,
    precursor_decoy,
    precursor_channel,
    scan_limits,
    frame_limits,
    precursor_mz,
    candidate_count=3,
    debug=False,
    weights=None,
    mean=None,
    std=None,
):
    cycle_length = jit_data.cycle.shape[1]

    candidates = nb.typed.List.empty_list(candidate_type)

    if weights is None:
        feature_weights = np.ones(8)
    else:
        feature_weights = weights

    feature_weights = feature_weights.reshape(-1, 1, 1)

    smooth_precursor = numeric.convolve_fourier_a1(dense_precursors, kernel)
    smooth_fragment = numeric.convolve_fourier_a1(dense_fragments, kernel)

    if not smooth_precursor.shape == dense_precursors.shape:
        print(smooth_precursor.shape, dense_precursors.shape)
        print("smooth_precursor shape does not match dense_precursors shape")
    if not smooth_fragment.shape == dense_fragments.shape:
        print(smooth_fragment.shape, dense_fragments.shape)
        print("smooth_fragment shape does not match dense_fragments shape")

    feature_matrix = build_features(
        smooth_precursor,
        smooth_fragment,
        precursor_intensity,
        fragment_intensity,
    ).astype("float32")

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

    # works until here

    scan_limits_list = np.zeros((peak_scan_list.shape[0], 2), dtype="int32")
    cycle_limits_list = np.zeros((peak_cycle_list.shape[0], 2), dtype="int32")

    for candidate_rank, (scan_relative, cycle_relative, candidate_score) in enumerate(
        zip(peak_scan_list, peak_cycle_list, peak_score_list)
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

    for candidate_rank, (
        scan_relative,
        cycle_relative,
        candidate_score,
        scan_limits_relative,
        cycle_limits_relative,
    ) in enumerate(
        zip(
            peak_scan_list,
            peak_cycle_list,
            peak_score_list,
            scan_limits_list,
            cycle_limits_list,
        )
    ):
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

        features = np.zeros(feature_matrix.shape[0], dtype="float32")
        for j in range(feature_matrix.shape[0]):
            features[j] = numeric.get_mean0(
                feature_matrix[j], scan_relative, cycle_relative
            )

        mass_error = np.zeros(smooth_precursor.shape[1], dtype="float32")
        for j in range(smooth_precursor.shape[0]):
            mass_error[j] = numeric.get_mean_sparse0(
                smooth_precursor[1, j], scan_relative, cycle_relative, 110
            )

        # iterate all precursors within this score group
        for i, pidx in enumerate(precursor_idx):
            candidates.append(
                Candidate(
                    elution_group_idx,
                    score_group_idx,
                    pidx,
                    candidate_rank,
                    candidate_score,
                    precursor_mz[i],
                    precursor_decoy[i],
                    precursor_channel[i],
                    features,
                    scan_absolute,
                    scan_limits_absolute[0],
                    scan_limits_absolute[1],
                    frame_absolute,
                    frame_limits_absolute[0],
                    frame_limits_absolute[1],
                )
            )

    if debug:
        with nb.objmode():
            plot_candidates(
                score, dense_fragments, candidates, jit_data, scan_limits, frame_limits
            )

    return candidates
