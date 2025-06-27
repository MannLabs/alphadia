import numba as nb
import numpy as np
from numba.extending import overload

from alphadia.numba.fft import NumbaContextOnly
from alphadia.utils import USE_NUMBA_CACHING


def assemble_isotope_mz(mono_mz, charge, isotope_intensity):
    """
    Assemble the isotope m/z values from the precursor m/z and the isotope
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


@nb.njit(cache=USE_NUMBA_CACHING)
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
