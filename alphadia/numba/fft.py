import numba as nb
import numpy as np
from numba.extending import overload
from rocket_fft import pocketfft
from rocket_fft.overloads import (
    decrease_shape,
    get_fct,
    increase_shape,
    ndshape_and_axes,
    resize,
    zeropad_or_crop,
)


class NumbaContextOnly(Exception):
    pass


def rfft2(x: np.array, s: None | tuple = None) -> np.array:
    """
    Numba function to compute the 2D real-to-complex FFT of a real array.

    Parameters
    ----------

    x : np.ndarray
        dtype = np.float32, ndim = 2, containing the input data.

    s : Union[None, tuple]
        Tuple of integers containing the shape of the output array.

    Returns
    -------

    np.ndarray
        dtype = np.complex64, ndim = 2, containing the 2D real-to-complex FFT of the input array.

    .. note::
        This function should only be used in a numba context as it relies on numba overloads.
    """

    raise NumbaContextOnly(
        "This function should only be used in a numba context as it relies on numbas overloads."
    )


@overload(rfft2, fastmath=True)
def _(x, s=None):
    if not isinstance(x, nb.types.Array):
        return

    if x.ndim != 2:
        return

    if x.dtype != nb.types.float32:
        return

    def funcx_impl(x, s=None):
        s, axes = ndshape_and_axes(x, s, (-2, -1))
        x = zeropad_or_crop(x, s, axes, nb.types.float32)
        shape = decrease_shape(x.shape, axes)
        out = np.empty(shape, dtype=nb.types.complex64)
        fct = get_fct(x, axes, None, True)
        pocketfft.numba_r2c(x, out, axes, True, fct, 1)
        return out

    return funcx_impl


def irfft2(x: np.array, s: None | tuple = None) -> np.array:
    """
    Numba function to compute the 2D complex-to-real FFT of a complex array.

    Parameters
    ----------

    x : np.ndarray
        dtype = np.complex64, ndim = 2, containing the input data.

    s : Union[None, tuple]
        Tuple of integers containing the shape of the output array.

    Returns
    -------

    np.ndarray
        dtype = np.float32, ndim = 2, containing the 2D complex-to-real FFT of the input array.

    .. note::
        This function should only be used in a numba context as it relies on numba overloads.
    """

    raise NumbaContextOnly(
        "This function should only be used in a numba context as it relies on numbas overloads."
    )


@overload(irfft2, fastmath=True)
def _(x, s=None):
    if not isinstance(x, nb.types.Array):
        return

    if x.ndim != 2:
        return

    if x.dtype != nb.types.complex64:
        return

    def funcx_impl(x, s=None):
        s, axes = ndshape_and_axes(x, s, (-2, -1))
        xin = zeropad_or_crop(x, s, axes, nb.types.complex64)
        shape = increase_shape(x.shape, axes)
        shape = resize(shape, x, s, axes)
        out = np.empty(shape, dtype=nb.types.float32)
        fct = get_fct(out, axes, None, False)
        pocketfft.numba_c2r(xin, out, axes, False, fct, 1)

        return out

    return funcx_impl


@nb.njit
def roll(a, delta0, delta1):
    b = np.zeros_like(a)
    b[delta0:, delta1:] = a[:-delta0, :-delta1]
    b[:delta0, delta1:] = a[-delta0:, :-delta1]
    b[delta0:, :delta1] = a[:-delta0, -delta1:]
    b[:delta0, :delta1] = a[-delta0:, -delta1:]

    return b


def convolve_fourier(dense, kernel):
    """
    Numba helper function to apply a gaussian filter to a 2d or 3d dense matrix.


    Parameters
    ----------

    dense : np.ndarray
        Array of shape (..., n_scans, n_frames)

    kernel : np.ndarray
        Array of shape (i, j)

    Returns
    -------

    np.ndarray
        Array of shape (..., n_scans, n_frames) containing the filtered dense stack.

    """

    raise NumbaContextOnly(
        "This function should only be used in a numba context as it relies on numbas overloads."
    )


@overload(convolve_fourier, fastmath=True)
def _(dense, kernel):
    if not isinstance(dense, nb.types.Array):
        return

    if not isinstance(kernel, nb.types.Array):
        return

    if kernel.ndim != 2:
        return

    if dense.ndim < 2:
        return

    if dense.ndim == 2:

        def funcx_impl(dense, kernel):
            k0, k1 = kernel.shape
            delta0, delta1 = -k0 // 2, -k1 // 2

            out = np.zeros_like(dense)
            fourier_filter = rfft2(kernel, dense.shape)
            layer = irfft2(rfft2(dense) * fourier_filter)
            out[delta0:, delta1:] = layer[:-delta0, :-delta1]
            out[:delta0, delta1:] = layer[-delta0:, :-delta1]
            out[delta0:, :delta1] = layer[:-delta0, -delta1:]
            out[:delta0, :delta1] = layer[-delta0:, -delta1:]

            return out

        return funcx_impl

    if dense.ndim == 3:

        def funcx_impl(dense, kernel):
            k0, k1 = kernel.shape
            delta0, delta1 = -k0 // 2, -k1 // 2

            out = np.zeros_like(dense)
            fourier_filter = rfft2(kernel, dense.shape[-2:])

            for i in range(dense.shape[0]):
                layer = irfft2(rfft2(dense[i]) * fourier_filter)
                out[i, delta0:, delta1:] = layer[:-delta0, :-delta1]
                out[i, :delta0, delta1:] = layer[-delta0:, :-delta1]
                out[i, delta0:, :delta1] = layer[:-delta0, -delta1:]
                out[i, :delta0, :delta1] = layer[-delta0:, -delta1:]

            return out

        return funcx_impl

    if dense.ndim == 4:

        def funcx_impl(dense, kernel):
            k0, k1 = kernel.shape
            delta0, delta1 = -k0 // 2, -k1 // 2

            out = np.zeros_like(dense)
            fourier_filter = rfft2(kernel, dense.shape[-2:])

            for i in range(dense.shape[0]):
                for j in range(dense.shape[1]):
                    layer = irfft2(rfft2(dense[i, j]) * fourier_filter)
                    out[i, j, delta0:, delta1:] = layer[:-delta0, :-delta1]
                    out[i, j, :delta0, delta1:] = layer[-delta0:, :-delta1]
                    out[i, j, delta0:, :delta1] = layer[:-delta0, -delta1:]
                    out[i, j, :delta0, :delta1] = layer[-delta0:, -delta1:]

            return out

        return funcx_impl
