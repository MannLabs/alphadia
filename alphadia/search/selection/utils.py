"""Utility functions for candidate selection operations."""

import numba as nb
import numpy as np
from numba.extending import overload

from alphadia.search.jitclasses.fragment_container import FragmentContainer
from alphadia.search.selection.fft import NumbaContextOnly
from alphadia.utils import USE_NUMBA_CACHING


def assemble_isotope_mz(mono_mz, charge, isotope_intensity):
    """Assemble the isotope m/z values from the precursor m/z and the isotope
    offsets.
    """
    raise NumbaContextOnly(
        "This function should only be used in a numba context as it relies on numbas overloads."
    )


@overload(assemble_isotope_mz)
def _(mono_mz, charge, isotope_intensity):
    if not isinstance(mono_mz, nb.types.Float):
        return None

    if not isinstance(charge, nb.types.Integer):
        return None

    if not isinstance(isotope_intensity, nb.types.Array):
        return None

    if isotope_intensity.ndim != 1:
        return None

    def funcx_impl(mono_mz, charge, isotope_intensity):
        offset = np.arange(len(isotope_intensity)) * 1.0033548350700006 / charge
        isotope_mz = np.zeros(len(isotope_intensity), dtype=np.float32)
        isotope_mz[:] = mono_mz
        isotope_mz += offset
        return isotope_mz

    return funcx_impl


@nb.njit(cache=USE_NUMBA_CACHING)
def find_peaks_1d(
    a: np.ndarray, top_n: int = 3
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Accepts a dense representation and returns the top three peaks"""
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


@nb.njit(cache=USE_NUMBA_CACHING)
def find_peaks_2d(
    a: np.ndarray, top_n: int = 3
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Accepts a dense representation and returns the top three peaks"""
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


@nb.njit(cache=USE_NUMBA_CACHING)
def amean1(array):
    out = np.zeros(array.shape[0])
    for i in range(len(out)):
        out[i] = np.mean(array[i])
    return out


@nb.njit(cache=USE_NUMBA_CACHING)
def astd1(array):
    out = np.zeros(array.shape[0])
    for i in range(len(out)):
        out[i] = np.std(array[i])
    return out


@nb.njit(cache=USE_NUMBA_CACHING)
def slice_manual(inst, slices):
    precursor_idx = []
    fragments_mz_library = []
    fragments_mz = []
    fragments_intensity = []
    fragments_type = []
    fragments_loss_type = []
    fragments_charge = []
    fragments_number = []
    fragments_position = []
    fragments_cardinality = []

    precursor = np.arange(len(slices), dtype=np.uint32)

    for i, (start_idx, stop_idx, _step) in enumerate(slices):
        for j in range(start_idx, stop_idx):
            precursor_idx.append(precursor[i])
            fragments_mz_library.append(inst.mz_library[j])
            fragments_mz.append(inst.mz[j])
            fragments_intensity.append(inst.intensity[j])
            fragments_type.append(inst.type[j])
            fragments_loss_type.append(inst.loss_type[j])
            fragments_charge.append(inst.charge[j])
            fragments_number.append(inst.number[j])
            fragments_position.append(inst.position[j])
            fragments_cardinality.append(inst.cardinality[j])

    precursor_idx = np.array(precursor_idx, dtype=np.uint32)
    fragments_mz_library = np.array(fragments_mz_library, dtype=np.float32)
    fragment_mz = np.array(fragments_mz, dtype=np.float32)
    fragment_intensity = np.array(fragments_intensity, dtype=np.float32)
    fragment_type = np.array(fragments_type, dtype=np.uint8)
    fragment_loss_type = np.array(fragments_loss_type, dtype=np.uint8)
    fragment_charge = np.array(fragments_charge, dtype=np.uint8)
    fragment_number = np.array(fragments_number, dtype=np.uint8)
    fragment_position = np.array(fragments_position, dtype=np.uint8)
    fragment_cardinality = np.array(fragments_cardinality, dtype=np.uint8)

    f = FragmentContainer(
        fragments_mz_library,
        fragment_mz,
        fragment_intensity,
        fragment_type,
        fragment_loss_type,
        fragment_charge,
        fragment_number,
        fragment_position,
        fragment_cardinality,
    )

    f.precursor_idx = precursor_idx

    return f


@nb.njit(cache=USE_NUMBA_CACHING)
def wrap0(
    value,
    limit,
):
    if value < 0:
        return 0
    return min(value, limit)


@nb.njit(cache=USE_NUMBA_CACHING)
def wrap1(
    values,
    limit,
):
    for i in range(values.shape[0]):
        values[i] = wrap0(values[i], limit)
    return values


@nb.njit(cache=USE_NUMBA_CACHING)
def _symetric_limits_1d(
    array_1d,
    center,
    f=0.95,
    center_fraction=0.01,
    min_size=1,
    max_size=10,
):
    """Find the limits of a symetric peak in a 1D array.
    Allthough the edge is symetrically extended in both directions, the trailing edge will use the last valid value when it reaches the limits of the array.

    Parameters
    ----------
    array_1d : np.ndarray
        1D array of intensities

    center : int
        Index of the center of the peak

    f : float
        minimal required decrease in intensity relative to the trailing intensity

    center_fraction : float
        minimal required intensity relative to the center intensity

    min_size : int
        minimal distance of the trailing edge of the peak from the center

    max_size : int
        maximal distance of the trailing edge of the peak from the center

    Returns
    -------
    np.ndarray, dtype='int32', shape=(2,)
        Array of containing the start and stop index of the peak.
        The start index is inclusive, the stop index is exclusive.
        Providing an empty array will return np.array([center, center], dtype='int32').
        Providing a center outside the array will return np.array([center, center], dtype='int32').

    """
    if len(array_1d) == 0:
        return np.array([center, center], dtype="int32")

    if center < 0 or center >= array_1d.shape[0]:
        return np.array([center, center], dtype="int32")

    center_intensity = array_1d[center]
    trailing_intensity = center_intensity

    limit = min_size

    for s in range(min_size + 1, max_size):
        intensity = (
            array_1d[max(center - s, 0)] + array_1d[min(center + s, len(array_1d) - 1)]
        ) / 2
        if intensity < f * trailing_intensity:
            if intensity > center_intensity * center_fraction:
                limit = s
                trailing_intensity = intensity
            else:
                break

        else:
            break

    return np.array(
        [max(center - limit, 0), min(center + limit + 1, len(array_1d))], dtype="int32"
    )


@nb.njit(cache=USE_NUMBA_CACHING)
def symetric_limits_2d(
    a,
    scan_center,
    dia_cycle_center,
    f_mobility=0.95,
    f_rt=0.95,
    center_fraction=0.01,
    min_size_mobility=3,
    max_size_mobility=20,
    min_size_rt=1,
    max_size_rt=10,
):
    mobility_lower = max(0, scan_center - min_size_mobility)
    mobility_upper = min(a.shape[0], scan_center + min_size_mobility)
    dia_cycle_lower = max(0, dia_cycle_center - min_size_rt)
    dia_cycle_upper = min(a.shape[1], dia_cycle_center + min_size_rt)

    mobility_limits = _symetric_limits_1d(
        a[:, dia_cycle_lower:dia_cycle_upper].sum(axis=1),
        scan_center,
        f=f_mobility,
        center_fraction=center_fraction,
        min_size=min_size_mobility,
        max_size=max_size_mobility,
    )

    dia_cycle_limits = _symetric_limits_1d(
        a[mobility_lower:mobility_upper, :].sum(axis=0),
        dia_cycle_center,
        f=f_rt,
        center_fraction=center_fraction,
        min_size=min_size_rt,
        max_size=max_size_rt,
    )

    return mobility_limits, dia_cycle_limits
