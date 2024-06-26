import numba as nb
import numpy as np
from numba.extending import overload

from alphadia.numba.fft import NumbaContextOnly


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
        return

    if not isinstance(charge, nb.types.Integer):
        return

    if not isinstance(isotope_intensity, nb.types.Array):
        return

    if isotope_intensity.ndim != 1:
        return

    def funcx_impl(mono_mz, charge, isotope_intensity):
        offset = np.arange(len(isotope_intensity)) * 1.0033548350700006 / charge
        isotope_mz = np.zeros(len(isotope_intensity), dtype=np.float32)
        isotope_mz[:] = mono_mz
        isotope_mz += offset
        return isotope_mz

    return funcx_impl
