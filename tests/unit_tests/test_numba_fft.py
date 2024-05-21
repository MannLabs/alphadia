import numba as nb
import numpy as np
import pytest

from alphadia.numba.fft import convolve_fourier, irfft2, rfft2


@pytest.mark.parametrize("shape", [(100, 2), (2, 100), (100, 100)])
def test_rfft2_np_agreement(shape, tol=1e-6):
    @nb.njit(cache=True)
    def njit_rfft2(x):
        return rfft2(x)

    x = np.random.rand(*shape).astype(np.float32)
    y = njit_rfft2(x)
    y2 = np.fft.rfft2(x)
    assert np.allclose(y, y2, atol=1e-3)


@pytest.mark.parametrize("shape", [(128, 128), (2, 2), (128, 2), (2, 2)])
def test_irfft2_np_agreement(shape, tol=1e-6):
    @nb.njit(cache=True)
    def njit_r2r(x):
        return irfft2(rfft2(x))

    x = np.random.rand(*shape).astype(np.float32)
    x2 = njit_r2r(x)
    x3 = np.fft.irfft2(np.fft.rfft2(x))
    assert np.allclose(x, x2, atol=1e-3)
    assert np.allclose(x, x3, atol=1e-3)
    assert np.allclose(x2, x3, atol=1e-3)


def test_conv_np_agreement():
    @nb.njit(cache=True)
    def conv(x, y):
        y = rfft2(y, x.shape)
        return irfft2(rfft2(x) * y)

    x = np.random.rand(128, 128).astype(np.float32)
    y = np.random.rand(32, 32).astype(np.float32)
    z1 = conv(x, y)
    z2 = np.fft.irfft2(np.fft.rfft2(x) * np.fft.rfft2(y, x.shape))

    assert np.allclose(z1, z2, atol=1e-3)


@pytest.mark.parametrize(
    "x, should_fail",
    [
        (np.random.rand(100, 100).astype(np.float32), False),
        (np.random.rand(100).astype(np.float32), True),
        (np.random.rand(100, 100).astype(np.float64), True),
        (np.random.rand(100, 100).astype(np.int32), True),
    ],
)
def test_fft_typing(x, should_fail):
    @nb.njit(cache=True)
    def njit_r2r(x):
        return irfft2(rfft2(x))

    if should_fail:
        with pytest.raises(nb.TypingError):
            njit_r2r(x)

    else:
        njit_r2r(x)


@pytest.mark.parametrize("shape", [(128, 128), (10, 128, 128), (10, 10, 128, 128)])
def test_convolve_fourier(shape):
    dense = np.zeros(shape).astype(np.float32)
    dense[..., shape[-2] // 2, shape[-1] // 2] = 1
    filter = np.ones((20, 20)).astype(np.float32)

    target = np.zeros(shape).astype(np.float32)
    lower_bound = shape[-2] // 2 - filter.shape[-2] // 2
    upper_bound = shape[-2] // 2 + filter.shape[-2] // 2
    target[..., lower_bound:upper_bound, lower_bound:upper_bound] = 1

    @nb.njit(cache=True)
    def nb_wrapper(dense, filter):
        return convolve_fourier(dense, filter)

    y1 = nb_wrapper(dense, filter)
    assert (y1 - target).max() < 1e-6


@pytest.mark.parametrize(
    "shape, kernel_shape, should_fail",
    [
        ((128, 128), (20, 20), False),
        ((1, 128, 128), (20, 20), False),
        ((10, 128, 128), (20, 20), False),
        ((2, 10, 128, 128), (20, 20), False),
        ((128, 128), (20, 20, 20), True),
        ((128, 128), (20,), True),
        ((128), (20, 20), True),
    ],
)
def test_convolve_fourier_typing(shape, kernel_shape, should_fail):
    dense = np.zeros(shape).astype(np.float32)
    filter = np.ones(kernel_shape).astype(np.float32)

    @nb.njit(cache=True)
    def nb_wrapper(dense, filter):
        return convolve_fourier(dense, filter)

    if should_fail:
        with pytest.raises(nb.TypingError):
            nb_wrapper(dense, filter)

    else:
        nb_wrapper(dense, filter)
