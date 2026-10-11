"""Module providing methods to read and process raw data in the following formats: Bruker hdf.

Note: this module partly duplicates the functionality of alphatims.bruker.TimsTOF.load_from_hdf, but is decoupled from alphatims to avoid making it a dependency of alphadia.
"""

import logging

import numpy as np
from alphabase.io.hdf import HDF_File, HDF_Group

from alphadia.exceptions import NotValidDiaDataError

logger = logging.getLogger()

HDF_FILE_EXTENSION = ".hdf"

# Root group name under which alphatims stores a serialized TimsTOF object.
ALPHATIMS_HDF_GROUP = "raw"

# Attributes that must be present after loading an alphatims HDF file for the
# transpose and JIT conversion to succeed. alphatims versions that subclass
# alpharaw's TimsTOFBase store exactly these `_`-prefixed names; older "classic"
# alphatims files use a different schema and will fail this check.
_REQUIRED_HDF_ATTRIBUTES = (
    "_accumulation_times",
    "_cycle",
    "_dia_mz_cycle",
    "_dia_precursor_cycle",
    "_frame_max_index",
    "_intensity_corrections",
    "_intensity_max_value",
    "_intensity_min_value",
    "_intensity_values",
    "_max_accumulation_time",
    "_mobility_max_value",
    "_mobility_min_value",
    "_mobility_values",
    "_mz_values",
    "_precursor_indices",
    "_precursor_max_index",
    "_push_indptr",
    "_quad_indptr",
    "_quad_max_mz_value",
    "_quad_min_mz_value",
    "_quad_mz_values",
    "_raw_quad_indptr",
    "_rt_values",
    "_scan_max_index",
    "_tof_indices",
    "_tof_max_index",
    "_use_calibrated_mz_values_as_default",
    "_zeroth_frame",
)


def _hdf_group_to_dict(group: HDF_Group) -> dict:
    """Recursively reconstruct a plain dict from an alphabase HDF group."""
    result = dict(group.metadata)
    for name in group.dataset_names:
        result[name] = getattr(group, name).values
    for name in group.dataframe_names:
        result[name] = getattr(group, name).values
    for name in group.group_names:
        result[name] = _hdf_group_to_dict(getattr(group, name))
    return result


def import_data_from_hdf_file(bruker_d_folder_name: str) -> dict:
    """Load a TimsTOF object serialized by alphatims into a dict.

    The file is expected to contain a `raw` group holding the `_`-prefixed
    attributes of an alpharaw ``TimsTOFBase`` (the schema written by
    ``alphatims.bruker.TimsTOF.save_as_hdf``).
    """
    raw_group = getattr(HDF_File(bruker_d_folder_name), ALPHATIMS_HDF_GROUP)
    loaded = _hdf_group_to_dict(raw_group)

    missing = [attr for attr in _REQUIRED_HDF_ATTRIBUTES if attr not in loaded]
    if missing:
        raise NotValidDiaDataError(
            f"HDF file {bruker_d_folder_name} is missing required attributes: "
            f"{', '.join(missing)}. Only alphatims HDF files written by a version "
            "that subclasses alpharaw's TimsTOFBase are supported."
        )

    # h5py reads arrays back as C-contiguous, but `quad_mz_values` is built
    # by alpharaw as a transpose (`np.stack([quad_low_values, quad_high_values]).T`,
    # i.e. Fortran-contiguous) and the JIT class expects that layout
    # (`float64[::1, :]`). Restore it to match the .d path.
    loaded["_quad_mz_values"] = np.asfortranarray(loaded["_quad_mz_values"])

    return loaded
