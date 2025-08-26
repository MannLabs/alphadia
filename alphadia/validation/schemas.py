import logging

import numpy as np

from alphadia.constants.keys import CalibCols
from alphadia.validation.base import Optional, Required, Schema

logger = logging.getLogger()


precursors_flat_schema = Schema(
    "precursors_flat",
    [
        Required("elution_group_idx", np.uint32),
        Optional("score_group_idx", np.uint32),
        Required("precursor_idx", np.uint32),
        Required("channel", np.uint32),
        Required("decoy", np.uint8),
        Required("flat_frag_start_idx", np.uint32),
        Required("flat_frag_stop_idx", np.uint32),
        Required("charge", np.uint8),
        Required(CalibCols.RT_LIBRARY, np.float32),
        Optional(CalibCols.RT_CALIBRATED, np.float32),
        Required(CalibCols.MOBILITY_LIBRARY, np.float32),
        Optional(CalibCols.MOBILITY_CALIBRATED, np.float32),
        Required(CalibCols.MZ_LIBRARY, np.float32),
        Optional(CalibCols.MZ_CALIBRATED, np.float32),
        Required("proteins", object),
        Required("genes", object),
        *[Optional(f"i_{i}", np.float32) for i in range(10)],
    ],
)


fragments_flat_schema = Schema(
    "fragments_flat",
    [
        Required(CalibCols.MZ_LIBRARY, np.float32),
        Optional(CalibCols.MZ_CALIBRATED, np.float32),
        Required("intensity", np.float32),
        Required("cardinality", np.uint8),
        Required("type", np.uint8),
        Required("loss_type", np.uint8),
        Required("charge", np.uint8),
        Required("number", np.uint8),
        Required("position", np.uint8),
    ],
)


candidates_schema = Schema(
    "candidates_df",
    [
        Required("elution_group_idx", np.uint32),
        Required("precursor_idx", np.uint32),
        Required("rank", np.uint8),
        Required("scan_start", np.int64),
        Required("scan_stop", np.int64),
        Required("scan_center", np.int64),
        Required("frame_start", np.int64),
        Required("frame_stop", np.int64),
        Required("frame_center", np.int64),
        Optional("score", np.float32),
        Optional("score_group_idx", np.uint32),
        Optional("channel", np.uint8),
        Optional("decoy", np.uint8),
        Optional("flat_frag_start_idx", np.uint32),
        Optional("flat_frag_stop_idx", np.uint32),
        Optional(CalibCols.MZ_LIBRARY, np.float32),
        Optional(CalibCols.MZ_CALIBRATED, np.float32),
        *[Optional(f"i_{i}", np.float32) for i in range(10)],
    ],
)


features_schema = Schema(
    "candidate_features_df",
    [
        Required("precursor_idx", np.uint32),
        Required("elution_group_idx", np.uint32),
        Required("rank", np.uint8),
        Required("decoy", np.uint8),
        Required("channel", np.uint8),
        Required("charge", np.uint8),
        Required("flat_frag_start_idx", np.uint32),
        Required("flat_frag_stop_idx", np.uint32),
        Required("scan_center", np.int64),
        Required("scan_start", np.int64),
        Required("scan_stop", np.int64),
        Required("frame_center", np.int64),
        Required("frame_start", np.int64),
        Required("frame_stop", np.int64),
        Required(CalibCols.MZ_LIBRARY, np.float32),
        Optional(CalibCols.MZ_CALIBRATED, np.float32),
        Required(CalibCols.MZ_OBSERVED, np.float32),
        Required(CalibCols.RT_LIBRARY, np.float32),
        Optional(CalibCols.RT_CALIBRATED, np.float32),
        Required(CalibCols.RT_OBSERVED, np.float32),
        Required(CalibCols.MOBILITY_LIBRARY, np.float32),
        Optional(CalibCols.MOBILITY_CALIBRATED, np.float32),
        Required(CalibCols.MOBILITY_OBSERVED, np.float32),
        *[Optional(f"i_{i}", np.float32) for i in range(10)],
    ],
)


fragment_features_schema = Schema(
    "fragment_features_df",
    [
        Required("precursor_idx", np.uint32),
        Required("rank", np.uint8),
        Required("elution_group_idx", np.uint32),
        Required(CalibCols.MZ_LIBRARY, np.float32),
        Required(CalibCols.MZ_OBSERVED, np.float32),
        Required("mass_error", np.float32),
        Required("height", np.float32),
        Required("intensity", np.float32),
        Required("decoy", np.uint8),
    ],
)
