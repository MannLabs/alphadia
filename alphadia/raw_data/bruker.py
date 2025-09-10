"""Module providing methods to read and process raw data in the following formats: Bruker."""

import logging
import os

import alphatims.bruker
import alphatims.tempmmap as tm
import alphatims.utils
import numba as nb
import numpy as np

from alphadia.exceptions import NotValidDiaDataError
from alphadia.search.jitclasses.bruker_jit import TimsTOFTransposeJIT
from alphadia.utils import USE_NUMBA_CACHING

logger = logging.getLogger()


class TimsTOFTranspose(alphatims.bruker.TimsTOF):
    """Transposed TimsTOF data structure."""

    def __init__(
        self,
        bruker_d_folder_name: str,
        *,
        mz_estimation_from_frame: int = 1,
        mobility_estimation_from_frame: int = 1,
        slice_as_dataframe: bool = True,
        use_calibrated_mz_values_as_default: int = 0,
        use_hdf_if_available: bool = True,
        mmap_detector_events: bool = True,
        drop_polarity: bool = True,
        convert_polarity_to_int: bool = True,
    ):
        self.has_mobility = True
        self.has_ms1 = True
        self.mmap_detector_events = mmap_detector_events

        bruker_d_folder_name = bruker_d_folder_name.removesuffix("/")
        logger.info(f"Importing data from {bruker_d_folder_name}")
        if bruker_d_folder_name.endswith(".d"):
            bruker_hdf_file_name = f"{bruker_d_folder_name[:-2]}.hdf"
            hdf_file_exists = os.path.exists(bruker_hdf_file_name)
            if use_hdf_if_available and hdf_file_exists:
                self._import_data_from_hdf_file(
                    bruker_hdf_file_name,
                    mmap_detector_events,
                )
                self.bruker_hdf_file_name = bruker_hdf_file_name
            else:
                self.bruker_d_folder_name = os.path.abspath(bruker_d_folder_name)
                self._import_data_from_d_folder(
                    bruker_d_folder_name,
                    mz_estimation_from_frame,
                    mobility_estimation_from_frame,
                    drop_polarity,
                    convert_polarity_to_int,
                    mmap_detector_events,
                )

                try:
                    cycle_shape = self._cycle.shape[0]
                except AttributeError as e:
                    raise NotValidDiaDataError(
                        "Could not find cycle shape attribute."
                    ) from e
                else:
                    if cycle_shape != 1:
                        raise NotValidDiaDataError(
                            f"Unexpected cycle shape: {cycle_shape} (expected: 1)."
                        )

                self.transpose()

        elif bruker_d_folder_name.endswith(".hdf"):
            self._import_data_from_hdf_file(
                bruker_d_folder_name,
                mmap_detector_events,
            )
            self.bruker_hdf_file_name = bruker_d_folder_name
        else:
            raise NotImplementedError("ERROR: file extension not understood")

        if not hasattr(self, "version"):
            self._version = "N.A."
        if self.version != alphatims.__version__:
            logger.info(
                "WARNING: "
                f"AlphaTims version {self.version} was used to initialize "
                f"{bruker_d_folder_name}, while the current version of "
                f"AlphaTims is {alphatims.__version__}."
            )
        self.slice_as_dataframe = slice_as_dataframe
        self.use_calibrated_mz_values_as_default(use_calibrated_mz_values_as_default)

        # Precompile
        logger.info(f"Successfully imported data from {bruker_d_folder_name}")

    def transpose(self):
        # abort if transposed data is already present
        if hasattr(self, "_push_indices") and hasattr(self, "_tof_indptr"):
            logger.info("Transposed data already present, aborting")
            return

        logger.info("Transposing detector events")
        push_indices, tof_indptr, intensity_values = _transpose(
            self._tof_indices,
            self._push_indptr,
            len(self._mz_values),
            self._intensity_values,
        )
        logger.info("Finished transposing data")

        self._tof_indices = np.zeros(1, np.uint32)
        self._push_indptr = np.zeros(1, np.int64)

        if self.mmap_detector_events:
            self._push_indices = tm.clone(push_indices)
            self._tof_indptr = tm.clone(tof_indptr)
            self._intensity_values = tm.clone(intensity_values)
        else:
            self._push_indices = push_indices
            self._tof_indptr = tof_indptr
            self._intensity_values = intensity_values

    def _import_data_from_hdf_file(self, *args, **kwargs):
        raise NotImplementedError("Not implemented yet for TimsTOFTranspose")

    def to_jitclass(self) -> TimsTOFTransposeJIT:
        """Create a TimsTOFTransposeJIT with the current state of this class."""
        return TimsTOFTransposeJIT(
            self._accumulation_times,
            self._cycle,
            self._dia_mz_cycle,
            self._dia_precursor_cycle,
            self._frame_max_index,
            self._intensity_corrections,
            self._intensity_max_value,
            self._intensity_min_value,
            self._intensity_values,
            self._max_accumulation_time,
            self._mobility_max_value,
            self._mobility_min_value,
            self._mobility_values,
            self._mz_values,
            self._precursor_indices,
            self._precursor_max_index,
            # self._push_indptr,
            self._quad_indptr,
            self._quad_max_mz_value,
            self._quad_min_mz_value,
            self._quad_mz_values,
            self._raw_quad_indptr,
            self._rt_values,
            self._scan_max_index,
            # self._tof_indices,
            self._tof_max_index,
            self._use_calibrated_mz_values_as_default,
            self._zeroth_frame,
            self._push_indices,
            self._tof_indptr,
        )


@alphatims.utils.pjit(cache=USE_NUMBA_CACHING)
def _transpose_chunk(
    chunk_idx,  # pjit decorator changes the passed argument from an iterable to single index
    chunks,
    push_indices,
    push_indptr,
    tof_indices,
    tof_indptr,
    values,
    new_values,
    tof_indcount,
):
    tof_index_chunk_start = chunks[chunk_idx]
    tof_index_chunk_stop = chunks[chunk_idx + 1]

    for push_idx in range(len(push_indptr) - 1):
        start_push_indptr = push_indptr[push_idx]
        stop_push_indptr = push_indptr[push_idx + 1]

        for idx in range(start_push_indptr, stop_push_indptr):
            # new row
            tof_index = tof_indices[idx]
            if tof_index_chunk_start <= tof_index < tof_index_chunk_stop:
                push_indices[tof_indptr[tof_index] + tof_indcount[tof_index]] = push_idx
                new_values[tof_indptr[tof_index] + tof_indcount[tof_index]] = values[
                    idx
                ]
                tof_indcount[tof_index] += 1


@nb.njit(cache=USE_NUMBA_CACHING)
def _build_chunks(number_of_elements, num_chunks):
    # Calculate the number of chunks needed
    chunk_size = (number_of_elements + num_chunks - 1) // num_chunks

    chunks = [0]
    start = 0

    for _ in range(num_chunks):
        stop = min(start + chunk_size, number_of_elements)
        chunks.append(stop)
        start = stop

    return np.array(chunks)


@nb.njit(cache=USE_NUMBA_CACHING)
def _transpose(tof_indices, push_indptr, n_tof_indices, values):
    """The default alphatims data format consists of a sparse matrix where pushes are the rows, tof indices (discrete mz values) the columns and intensities the values.
    A lookup starts with a given push index p which points to the row. The start and stop indices of the row are accessed from dia_data.push_indptr[p] and dia_data.push_indptr[p+1].
    The tof indices are then accessed from dia_data.tof_indices[start:stop] and the corresponding intensities from dia_data.intensity_values[start:stop].

    The function transposes the data such that the tof indices are the rows and the pushes are the columns.
    This is usefull when accessing only a small number of tof indices (e.g. when extracting a single precursor) and the number of pushes is large (e.g. when extracting a whole run).

    Parameters
    ----------
    tof_indices : np.ndarray
        column indices (n_values)

    push_indptr : np.ndarray
        start stop values for each row (n_rows +1)

    n_tof_indices : int
        number of tof indices which is usually equal to len(dia_data.mz_values)

    values : np.ndarray
        values (n_values)

    threads : int
        number of threads to use

    Returns
    -------
    push_indices : np.ndarray
        row indices (n_values)

    tof_indptr : np.ndarray
        start stop values for each row (n_rows +1)

    new_values : np.ndarray
        values (n_values)

    """
    tof_indcount = np.zeros((n_tof_indices), dtype=np.uint32)

    # get new row counts
    for v in tof_indices:
        tof_indcount[v] += 1

    # get new indptr
    tof_indptr = np.zeros((n_tof_indices + 1), dtype=np.int64)

    for i in range(n_tof_indices):
        tof_indptr[i + 1] = tof_indptr[i] + tof_indcount[i]

    tof_indcount = np.zeros((n_tof_indices), dtype=np.uint32)

    # get new values
    push_indices = np.zeros((len(tof_indices)), dtype=np.uint32)
    new_values = np.zeros_like(values)

    chunks = _build_chunks(n_tof_indices, 20)

    with nb.objmode:
        alphatims.utils.set_threads(20)

        _transpose_chunk(
            range(len(chunks) - 1),  # type: ignore  # noqa: PGH003  # function is wrapped by pjit -> will be turned into single index and passed to the method
            chunks,
            push_indices,
            push_indptr,
            tof_indices,
            tof_indptr,
            values,
            new_values,
            tof_indcount,
        )

    return push_indices, tof_indptr, new_values
