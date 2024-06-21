# native imports
import logging
import platform
import re
from ctypes import Structure, c_double

# alphadia imports
# alpha family imports
import alphatims.bruker
import alphatims.utils
import matplotlib.patches as patches
import numba as nb
import numpy as np

# third party imports
import pandas as pd
import torch

logger = logging.getLogger()


ISOTOPE_DIFF = 1.0032999999999674


def get_torch_device(use_gpu: bool = False):
    """Get the torch device to be used.

    Parameters
    ----------

    use_gpu : bool, optional
        If True, use GPU if available, by default False

    Returns
    -------
    str
        Device to be used, either 'cpu', 'gpu' or 'mps'

    """

    device = "cpu"
    if use_gpu:
        if platform.system() == "Darwin":
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        else:
            device = "gpu" if torch.cuda.is_available() else "cpu"

    logger.info(f"Device set to {device}")

    return device


@nb.njit
def candidate_hash(precursor_idx, rank):
    # create a 64 bit hash from the precursor_idx, number and type
    # the precursor_idx is the lower 32 bits
    # the rank is the next 8 bits
    return precursor_idx + (rank << 32)


@nb.njit
def ion_hash(precursor_idx, number, type, charge):
    # create a 64 bit hash from the precursor_idx, number and type
    # the precursor_idx is the lower 32 bits
    # the number is the next 8 bits
    # the type is the next 8 bits
    # the last 8 bits are used to distinguish between different charges of the same precursor
    # this is necessary because I forgot to save the charge in the frag.tsv file :D
    return precursor_idx + (number << 32) + (type << 40) + (charge << 48)


@nb.njit
def extended_ion_hash(precursor_idx, rank, number, type, charge):
    # create a 64 bit hash from the precursor_idx, number and type
    # the precursor_idx is the lower 32 bits
    # the number is the next 8 bits
    # the type is the next 8 bits
    # the last 8 bits are used to distinguish between different charges of the same precursor
    # this is necessary because I forgot to save the charge in the frag.tsv file :D
    return precursor_idx + (rank << 32) + (number << 40) + (type << 48) + (charge << 56)


def wsl_to_windows(
    path: str | list | tuple,
) -> str | list | tuple:
    """Converts a WSL path to a Windows path.

    Parameters
    ----------
    path : str, list, tuple
        WSL path.

    Returns
    -------
    str, list, tuple
        Windows path.

    """

    if path is None:
        return None

    if isinstance(path, str):
        disk_match = re.search(r"^/mnt/[a-z]", path)

        if len(disk_match.group()) == 0:
            raise ValueError(
                "Could not find disk in path during wsl to windows conversion"
            )

        disk_letter = disk_match.group()[5].upper()

        return re.sub(r"^/mnt/[a-z]", f"{disk_letter}:", path).replace("/", "\\")

    elif isinstance(path, list | tuple):
        return [wsl_to_windows(p) for p in path]
    else:
        raise ValueError(f"Unsupported type {type(path)}")


def windows_to_wsl(
    path: str | list | tuple,
) -> str | list | tuple:
    """Converts a Windows path to a WSL path.

    Parameters
    ----------
    path : str, list, tuple
        Windows path.

    Returns
    -------
    str, list, tuple
        WSL path.

    """
    if path is None:
        return None

    if isinstance(path, str):
        disk_match = re.search(r"^[A-Z]:", path)

        if len(disk_match.group()) == 0:
            raise ValueError(
                "Could not find disk in path during windows to wsl conversion"
            )

        disk_letter = disk_match.group()[0].lower()

        return re.sub(r"^[A-Z]:", f"/mnt/{disk_letter}", path.replace("\\", "/"))

    elif isinstance(path, list | tuple):
        return [windows_to_wsl(p) for p in path]
    else:
        raise ValueError(f"Unsupported type {type(path)}")


def recursive_update(full_dict: dict, update_dict: dict):
    """recursively update a dict with a second dict. The dict is updated inplace.

    Parameters
    ----------
    full_dict : dict
        dict to be updated, is updated inplace.

    update_dict : dict
        dict with new values

    Returns
    -------
    None

    """
    for key, value in update_dict.items():
        if key in full_dict:
            if isinstance(value, dict):
                recursive_update(full_dict[key], update_dict[key])
            else:
                full_dict[key] = value
        else:
            full_dict[key] = value


def normal(x, mu, sigma):
    """ """
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-np.power((x - mu) / sigma, 2) / 2)


def plt_limits(mobility_limits, dia_cycle_limits):
    mobility_len = mobility_limits[1] - mobility_limits[0]
    dia_cycle_len = dia_cycle_limits[1] - dia_cycle_limits[0]

    rect = patches.Rectangle(
        (dia_cycle_limits[0], mobility_limits[0]),
        dia_cycle_len,
        mobility_len,
        linewidth=1,
        edgecolor="r",
        facecolor="none",
    )

    return rect


@alphatims.utils.njit()
def find_peaks_1d(a, top_n=3):
    """accepts a dense representation and returns the top three peaks"""

    scan = []
    dia_cycle = []
    intensity = []

    for p in range(2, a.shape[1] - 2):
        isotope_is_peak = (
            a[0, p - 2] < a[0, p - 1] < a[0, p] > a[0, p + 1] > a[0, p + 2]
        )

        if isotope_is_peak:
            intensity.append(a[0, p])
            scan.append(0)
            dia_cycle.append(p)

    scan = np.array(scan)
    dia_cycle = np.array(dia_cycle)
    intensity = np.array(intensity)

    idx = np.argsort(intensity)[::-1][:top_n]

    scan = scan[idx]
    dia_cycle = dia_cycle[idx]
    intensity = intensity[idx]

    return scan, dia_cycle, intensity


@alphatims.utils.njit()
def find_peaks_2d(a, top_n=3):
    """accepts a dense representation and returns the top three peaks"""
    scan = []
    dia_cycle = []
    intensity = []

    for s in range(2, a.shape[0] - 2):
        for p in range(2, a.shape[1] - 2):
            isotope_is_peak = (
                a[s - 2, p] < a[s - 1, p] < a[s, p] > a[s + 1, p] > a[s + 2, p]
            )
            isotope_is_peak &= (
                a[s, p - 2] < a[s, p - 1] < a[s, p] > a[s, p + 1] > a[s, p + 2]
            )

            if isotope_is_peak:
                intensity.append(a[s, p])
                scan.append(s)
                dia_cycle.append(p)

    scan = np.array(scan)
    dia_cycle = np.array(dia_cycle)
    intensity = np.array(intensity)

    idx = np.argsort(intensity)[::-1][:top_n]

    scan = scan[idx]
    dia_cycle = dia_cycle[idx]
    intensity = intensity[idx]

    return scan, dia_cycle, intensity


@alphatims.utils.njit()
def amean1(array):
    out = np.zeros(array.shape[0])
    for i in range(len(out)):
        out[i] = np.mean(array[i])
    return out


@alphatims.utils.njit()
def amean0(array):
    out = np.zeros(array.shape[1])
    for i in range(len(out)):
        out[i] = np.mean(array[:, i])
    return out


@alphatims.utils.njit()
def astd0(array):
    out = np.zeros(array.shape[1])
    for i in range(len(out)):
        out[i] = np.std(array[:, i])
    return out


@alphatims.utils.njit()
def astd1(array):
    out = np.zeros(array.shape[0])
    for i in range(len(out)):
        out[i] = np.std(array[i])
    return out


def get_isotope_columns(colnames):
    isotopes = []
    for col in colnames:
        if col[:2] == "i_":
            try:
                isotopes.append(int(col[2:]))
            except Exception:
                logging.warning(
                    f"Column {col} does not seem to be a valid isotope column"
                )

    isotopes = np.array(sorted(isotopes))

    if not np.all(np.diff(isotopes) == 1):
        logging.warning("Isotopes are not consecutive")

    return isotopes


def get_isotope_column_names(colnames):
    return [f"i_{i}" for i in get_isotope_columns(colnames)]


@alphatims.utils.njit()
def mass_range(mz_list, ppm_tolerance):
    out_mz = np.zeros((len(mz_list), 2), dtype=mz_list.dtype)
    out_mz[:, 0] = mz_list - ppm_tolerance * mz_list / (10**6)
    out_mz[:, 1] = mz_list + ppm_tolerance * mz_list / (10**6)
    return out_mz


def function_call(q):
    q.put("X" * 1000000)


def modify(n, x, s, A):
    n.value **= 2
    x.value **= 2
    s.value = s.value.upper()
    for a in A:
        a.x **= 2
        a.y **= 2


class Point(Structure):
    _fields_ = [("x", c_double), ("y", c_double)]


@alphatims.utils.njit()
def tile(a, n):
    return np.repeat(a, n).reshape(-1, n).T.flatten()


@alphatims.utils.njit
def make_slice_1d(start_stop):
    """Numba helper function to create a 1D slice object from a start and stop value.

        e.g. make_slice_1d([0, 10]) -> np.array([[0, 10, 1]], dtype='uint64')

    Parameters
    ----------
    start_stop : np.ndarray
        Array of shape (2,) containing the start and stop value.

    Returns
    -------
    np.ndarray
        Array of shape (1,3) containing the start, stop and step value.

    """
    return np.array([[start_stop[0], start_stop[1], 1]], dtype=start_stop.dtype)


@alphatims.utils.njit
def make_slice_2d(start_stop):
    """Numba helper function to create a 2D slice object from multiple start and stop value.

        e.g. make_slice_2d([[0, 10], [0, 10]]) -> np.array([[0, 10, 1], [0, 10, 1]], dtype='uint64')

    Parameters
    ----------
    start_stop : np.ndarray
        Array of shape (N, 2) containing the start and stop value for each dimension.

    Returns
    -------
    np.ndarray
        Array of shape (N, 3) containing the start, stop and step value for each dimension.

    """

    out = np.ones((start_stop.shape[0], 3), dtype=start_stop.dtype)
    out[:, 0] = start_stop[:, 0]
    out[:, 1] = start_stop[:, 1]
    return out


@alphatims.utils.njit
def fourier_filter(dense_stack, kernel):
    """Numba helper function to apply a gaussian filter to a dense stack.
    The filter is applied as convolution wrapping around the edges, calculated in fourier space.

    As there seems to be no easy option to perform 2d fourier transforms in numba, the numpy fft is used in object mode.
    During multithreading the GIL has to be acquired to use the numpy fft and is realeased afterwards.

    Parameters
    ----------

    dense_stack : np.ndarray
        Array of shape (2, n_precursors, n_observations ,n_scans, n_cycles) containing the dense stack.

    kernel : np.ndarray
        Array of shape (k0, k1) containing the gaussian kernel.

    Returns
    -------
    smooth_output : np.ndarray
        Array of shape (n_precursors, n_observations, n_scans, n_cycles) containing the filtered dense stack.

    """

    # make sure both dimensions are even
    scan_mod = dense_stack.shape[3] % 2
    frame_mod = dense_stack.shape[4] % 2

    scan_size = dense_stack.shape[3] - scan_mod
    frame_size = dense_stack.shape[4] - frame_mod

    smooth_output = np.zeros(
        (
            dense_stack.shape[1],
            dense_stack.shape[2],
            scan_size,
            frame_size,
        ),
        dtype="float32",
    )

    fourier_filter = np.fft.rfft2(kernel, smooth_output.shape[2:])

    for i in range(smooth_output.shape[0]):
        for j in range(smooth_output.shape[1]):
            layer = dense_stack[0, i, j, :scan_size, :frame_size]

            smooth_output[i, j] = np.fft.irfft2(np.fft.rfft2(layer) * fourier_filter)

    # with nb.objmode(smooth_output='float32[:,:,:,:]'):
    #    # roll back to original position
    #    k0 = kernel.shape[0]
    #    k1 = kernel.shape[1]
    #    smooth_output = np.roll(smooth_output, -k0//2, axis=2)
    #    smooth_output = np.roll(smooth_output, -k1//2, axis=3)

    return smooth_output


def calculate_score_groups(
    input_df: pd.DataFrame,
    group_channels: bool = False,
):
    """
    Calculate score groups for DIA multiplexing.

    On the candidate selection level, score groups are used to group ions across channels.
    On the scoring level, score groups are used to group channels of the same precursor and rank together.

    This function makes sure that all precursors within a score group have the same `elution_group_idx`, `decoy` status and `rank` if available.
    If `group_channels` is True, different channels of the same precursor will be grouped together.

    Parameters
    ----------

    input_df : pandas.DataFrame
        Precursor dataframe. Must contain columns 'elution_group_idx' and 'decoy'. Can contain 'rank' column.

    group_channels : bool
        If True, precursors from the same elution group will be grouped together while seperating different ranks and decoy status.

    Returns
    -------

    score_groups : pandas.DataFrame
        Updated precursor dataframe with score_group_idx column.

    Example
    -------

    A precursor with the same `elution_group_idx` might be grouped with other precursors if only the `channel` is different.
    Different `rank` and `decoy` status will always lead to different score groups.

    .. list-table::
        :widths: 25 25 25 25 25 25
        :header-rows: 1

        * - elution_group_idx
          - rank
          - decoy
          - channel
          - group_channels = False
          - group_channels = True

        * - 0
          - 0
          - 0
          - 0
          - 0
          - 0

        * - 0
          - 0
          - 0
          - 4
          - 1
          - 0

        * - 0
          - 1
          - 0
          - 0
          - 2
          - 1

        * - 0
          - 1
          - 1
          - 0
          - 3
          - 2

    """

    @nb.njit
    def channel_score_groups(elution_group_idx, decoy, rank):
        """
        Calculate score groups for channel grouping.

        Parameters
        ----------

        elution_group_idx : numpy.ndarray
            Elution group indices.

        decoy : numpy.ndarray
            Decoy status.

        rank : numpy.ndarray
            Rank of precursor.

        Returns
        -------

        score_groups : numpy.ndarray
            Score groups.
        """
        score_groups = np.zeros(len(elution_group_idx), dtype=np.uint32)
        current_group = 0
        current_eg = elution_group_idx[0]
        current_decoy = decoy[0]
        current_rank = rank[0]

        for i in range(len(elution_group_idx)):
            # if elution group, decoy status or rank changes, increase score group
            if (
                (elution_group_idx[i] != current_eg)
                or (decoy[i] != current_decoy)
                or (rank[i] != current_rank)
            ):
                current_group += 1
                current_eg = elution_group_idx[i]
                current_decoy = decoy[i]
                current_rank = rank[i]

            score_groups[i] = current_group
        return score_groups

    # sort by elution group, decoy and rank
    # if no rank is present, pretend rank 0
    if "rank" in input_df.columns:
        input_df = input_df.sort_values(by=["elution_group_idx", "decoy", "rank"])
        rank_values = input_df["rank"].values
    else:
        input_df = input_df.sort_values(by=["elution_group_idx", "decoy"])
        rank_values = np.zeros(len(input_df), dtype=np.uint32)

    if group_channels:
        input_df["score_group_idx"] = channel_score_groups(
            input_df["elution_group_idx"].values, input_df["decoy"].values, rank_values
        )
    else:
        input_df["score_group_idx"] = np.arange(len(input_df), dtype=np.uint32)

    return input_df.sort_values(by=["score_group_idx"]).reset_index(drop=True)


@nb.njit()
def profile_correlation(profile, tresh=3, shift=2, kernel_size=12):
    mask = np.sum((profile >= tresh).astype(np.int8), axis=0) == profile.shape[0]

    output = np.zeros(profile.shape, dtype=np.float32)

    start_index = 0

    while start_index < (len(mask) - kernel_size):
        if not mask[start_index]:
            start_index += shift
            continue

        slice = profile[:, start_index : start_index + kernel_size]
        correlation = amean0(np.corrcoef(slice))

        start = start_index + kernel_size // 2 - shift
        end = start_index + kernel_size // 2
        output[:, start : start_index + end] = correlation.reshape(-1, 1)
        start_index += shift

    return output


def merge_missing_columns(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    right_columns: list,
    on: list = None,
    how: str = "left",
):
    """Merge missing columns from right_df into left_df.

    Merging is performed only for columns not yet present in left_df.

    Parameters
    ----------

    left_df : pandas.DataFrame
        Left dataframe

    right_df : pandas.DataFrame
        Right dataframe

    right_columns : list
        List of columns to merge from right_df into left_df

    on : list, optional
        List of columns to merge on, by default None

    how : str, optional
        How to merge, by default 'left'

    Returns
    -------
    pandas.DataFrame
        Merged left dataframe

    """
    if isinstance(on, str):
        on = [on]

    if isinstance(right_columns, str):
        right_columns = [right_columns]

    missing_from_left = list(set(right_columns) - set(left_df.columns))
    missing_from_right = list(set(missing_from_left) - set(right_df.columns))

    if len(missing_from_left) == 0:
        return left_df

    if missing_from_right:
        raise ValueError(f"Columns {missing_from_right} must be present in right_df")

    if on is None:
        raise ValueError("Parameter on must be specified")

    if not all([col in left_df.columns for col in on]):
        raise ValueError(f"Columns {on} must be present in left_df")

    if not all([col in right_df.columns for col in on]):
        raise ValueError(f"Columns {on} must be present in right_df")

    if how not in ["left", "right", "inner", "outer"]:
        raise ValueError("Parameter how must be one of left, right, inner, outer")

    # merge
    return left_df.merge(right_df[on + missing_from_left], on=on, how=how)
