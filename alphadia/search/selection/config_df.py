"""Configuration DataFrames for selection parameters."""

import logging

import numba as nb
import numpy as np
import pandas as pd

from alphadia.search.jitclasses.jit_config import JITConfig

logger = logging.getLogger()


@nb.experimental.jitclass()
class CandidateSelectionConfigJIT:
    """Numba compatible config object for the HybridCandidate class.
    Please see the documentation of the CandidateSelectionConfig class for more information on the parameters and their default values.
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


class CandidateSelectionConfig(
    JITConfig
):  # TODO rename to CandidateSelectionHyperparameters
    _jit_container_type = CandidateSelectionConfigJIT

    def __init__(self):
        super().__init__()

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
class PrecursorFlatContainer:
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
class CandidateContainer:
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

    def get_candidate_df_column_names(self) -> list[str]:
        """Get the column names for the candidate DataFrame."""
        return [
            "precursor_idx",
            "rank",
            "score",
            "scan_center",
            "scan_start",
            "scan_stop",
            "frame_center",
            "frame_start",
            "frame_stop",
        ]

    def get_candidate_df_data(self, min_score: int = 0) -> tuple[np.ndarray, ...]:
        """Prepare a tuple with the candidate data, filtering by minimum score."""
        mask = self.score > min_score

        return (
            self.precursor_idx[mask],
            self.rank[mask],
            self.score[mask],
            self.scan_center[mask],
            self.scan_start[mask],
            self.scan_stop[mask],
            self.frame_center[mask],
            self.frame_start[mask],
            self.frame_stop[mask],
        )


def candidate_container_to_df(candidate_container: CandidateContainer) -> pd.DataFrame:
    """Convert a CandidateContainer to pd.DataFrame."""
    return pd.DataFrame(
        {
            key: value
            for key, value in zip(
                candidate_container.get_candidate_df_column_names(),
                candidate_container.get_candidate_df_data(),
                strict=True,
            )
        }
    )
