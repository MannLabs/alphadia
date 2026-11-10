"""Output Handling for Candidate Scoring."""

# native imports
import logging

# alpha family imports
import numba as nb
import numpy as np

# third party imports
# alphadia imports
from alphadia.constants.settings import NUM_FEATURES

logger = logging.getLogger()


@nb.experimental.jitclass()
class OutputPsmDF:
    valid: nb.boolean[::1]
    precursor_idx: nb.uint32[::1]
    rank: nb.uint8[::1]

    features: nb.float32[:, ::1]

    fragment_precursor_idx: nb.uint32[:, ::1]
    fragment_rank: nb.uint8[:, ::1]

    fragment_mz_library: nb.float32[:, ::1]
    fragment_mz: nb.float32[:, ::1]
    fragment_mz_observed: nb.float32[:, ::1]

    fragment_height: nb.float32[:, ::1]
    fragment_intensity: nb.float32[:, ::1]

    fragment_mass_error: nb.float32[:, ::1]
    fragment_correlation: nb.float32[:, ::1]

    fragment_position: nb.uint8[:, ::1]
    fragment_number: nb.uint8[:, ::1]
    fragment_type: nb.uint8[:, ::1]
    fragment_charge: nb.uint8[:, ::1]
    fragment_loss_type: nb.uint8[:, ::1]

    def __init__(self, n_psm, top_k_fragments):
        self.valid = np.zeros(n_psm, dtype=np.bool_)
        self.precursor_idx = np.zeros(n_psm, dtype=np.uint32)
        self.rank = np.zeros(n_psm, dtype=np.uint8)

        self.features = np.zeros((n_psm, NUM_FEATURES), dtype=np.float32)

        self.fragment_precursor_idx = np.zeros(
            (n_psm, top_k_fragments), dtype=np.uint32
        )
        self.fragment_rank = np.zeros((n_psm, top_k_fragments), dtype=np.uint8)

        self.fragment_mz_library = np.zeros((n_psm, top_k_fragments), dtype=np.float32)
        self.fragment_mz = np.zeros((n_psm, top_k_fragments), dtype=np.float32)
        self.fragment_mz_observed = np.zeros((n_psm, top_k_fragments), dtype=np.float32)

        self.fragment_height = np.zeros((n_psm, top_k_fragments), dtype=np.float32)
        self.fragment_intensity = np.zeros((n_psm, top_k_fragments), dtype=np.float32)

        self.fragment_mass_error = np.zeros((n_psm, top_k_fragments), dtype=np.float32)
        self.fragment_correlation = np.zeros((n_psm, top_k_fragments), dtype=np.float32)

        self.fragment_position = np.zeros((n_psm, top_k_fragments), dtype=np.uint8)
        self.fragment_number = np.zeros((n_psm, top_k_fragments), dtype=np.uint8)
        self.fragment_type = np.zeros((n_psm, top_k_fragments), dtype=np.uint8)
        self.fragment_charge = np.zeros((n_psm, top_k_fragments), dtype=np.uint8)
        self.fragment_loss_type = np.zeros((n_psm, top_k_fragments), dtype=np.uint8)

    def to_fragment_df(self):
        mask = self.fragment_mz_library.flatten() > 0

        return (
            self.fragment_precursor_idx.flatten()[mask],
            self.fragment_rank.flatten()[mask],
            self.fragment_mz_library.flatten()[mask],
            self.fragment_mz.flatten()[mask],
            self.fragment_mz_observed.flatten()[mask],
            self.fragment_height.flatten()[mask],
            self.fragment_intensity.flatten()[mask],
            self.fragment_mass_error.flatten()[mask],
            self.fragment_correlation.flatten()[mask],
            self.fragment_position.flatten()[mask],
            self.fragment_number.flatten()[mask],
            self.fragment_type.flatten()[mask],
            self.fragment_charge.flatten()[mask],
            self.fragment_loss_type.flatten()[mask],
        )

    def to_precursor_df(self):
        return (
            self.precursor_idx[self.valid],
            self.rank[self.valid],
            self.features[self.valid],
        )
