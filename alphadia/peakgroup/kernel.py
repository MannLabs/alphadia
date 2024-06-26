# native imports
import logging

import numba as nb

# alpha family imports
# third party imports
import numpy as np

# alphadia imports
from alphadia.data import alpharaw, bruker

logger = logging.getLogger()


@nb.njit()
def multivariate_normal(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray):
    """multivariate normal distribution, probability density function

    Most likely an absolutely inefficient implementation of the multivariate normal distribution.
    Is only used for creating the gaussian kernel and will only be used a few times for small kernels.

    Parameters
    ----------

    x : np.ndarray
        `(N, D,)`

    mu : np.ndarray
        `(1, D,)`

    sigma : np.ndarray
        `(D, D,)`

    Returns
    -------

    np.ndarray, float32
        array of shape `(N,)` with the density at each point

    """

    k = mu.shape[0]
    dx = x - mu

    # implementation is not very efficient for large N as the N x N matrix will created only for storing the diagonal
    a = np.exp(-1 / 2 * np.diag(dx @ np.linalg.inv(sigma) @ dx.T))
    b = (np.pi * 2) ** (-k / 2) * np.linalg.det(sigma) ** (-1 / 2)
    return a * b


class GaussianKernel:
    def __init__(
        self,
        dia_data: bruker.TimsTOFTransposeJIT
        | bruker.TimsTOFTranspose
        | alpharaw.AlphaRaw,
        fwhm_rt: float = 10.0,
        sigma_scale_rt: float = 1.0,
        fwhm_mobility: float = 0.03,
        sigma_scale_mobility: float = 1.0,
        kernel_height: int = 30,
        kernel_width: int = 30,
    ):
        """
        Create a two-dimensional gaussian filter kernel for the RT and mobility dimensions of a DIA dataset.
        First, the observed standard deviation is scaled by a linear factor. Second, the standard deviation is scaled by the resolution of the respective dimension.

        This results in sigma_scale to be independent of the resolution of the data and FWHM of the peaks.

        Parameters
        ----------

        dia_data : typing.Union[bruker.TimsTOFTransposeJIT, bruker.TimsTOFTranspose]
            alphatims dia_data object.

        fwhm_rt : float
            Full width at half maximum in RT dimension of the peaks in the spectrum.

        sigma_scale_rt : float
            Scaling factor for the standard deviation in RT dimension.

        fwhm_mobility : float
            Full width at half maximum in mobility dimension of the peaks in the spectrum.

        sigma_scale_mobility : float
            Scaling factor for the standard deviation in mobility dimension.

        kernel_size : int
            Kernel shape in pixel. The kernel will be a square of size (kernel_size, kernel_size).
            Should be even and will be rounded up to the next even number if necessary.

        """
        self.dia_data = dia_data
        self.fwhm_rt = fwhm_rt
        self.sigma_scale_rt = sigma_scale_rt
        self.fwhm_mobility = fwhm_mobility
        self.sigma_scale_mobility = sigma_scale_mobility

        self.kernel_height = int(
            np.ceil(kernel_height / 2) * 2
        )  # make sure kernel size is even
        self.kernel_width = int(
            np.ceil(kernel_width / 2) * 2
        )  # make sure kernel size is even

    def determine_rt_sigma(self, cycle_length_seconds: float):
        """
        Determine the standard deviation of the gaussian kernel in RT dimension.
        The standard deviation will be sclaed to the resolution of the raw data.

        Parameters
        ----------

        cycle_length_seconds : float
            Cycle length of the duty cycle in seconds.

        Returns
        -------

        float
            Standard deviation of the gaussian kernel in RT dimension scaled to the resolution of the raw data.
        """
        # a normal distribution has a FWHM of 2.3548 sigma
        sigma = self.fwhm_rt / 2.3548
        sigma_scaled = sigma * self.sigma_scale_rt / cycle_length_seconds
        return sigma_scaled

    def determine_mobility_sigma(self, mobility_resolution: float):
        """
        Determine the standard deviation of the gaussian kernel in mobility dimension.
        The standard deviation will be sclaed to the resolution of the raw data.

        Parameters
        ----------

        mobility_resolution : float
            Resolution of the mobility dimension in 1/K_0.

        Returns
        -------

        float
            Standard deviation of the gaussian kernel in mobility dimension scaled to the resolution of the raw data.
        """

        if not self.dia_data.has_mobility:
            return 1.0

        # a normal distribution has a FWHM of 2.3548 sigma
        sigma = self.fwhm_mobility / 2.3548
        sigma_scaled = sigma * self.sigma_scale_mobility / mobility_resolution
        return sigma_scaled

    def get_dense_matrix(self, verbose: bool = True):
        """
        Calculate the gaussian kernel for the given data set and parameters.

        Parameters
        ----------

        verbose : bool
            If True, log information about the data set and the kernel.

        Returns
        -------

        np.ndarray
            Two-dimensional gaussian kernel.

        """

        rt_datapoints = self.dia_data.cycle.shape[1]
        rt_resolution = np.mean(np.diff(self.dia_data.rt_values[::rt_datapoints]))

        mobility_datapoints = self.dia_data.cycle.shape[2]
        mobility_resolution = np.mean(np.diff(self.dia_data.mobility_values[::-1]))

        if verbose:
            pass
            logger.info(
                f"Duty cycle consists of {rt_datapoints} frames, {rt_resolution:.2f} seconds cycle time"
            )
            logger.info(
                f"Duty cycle consists of {mobility_datapoints} scans, {mobility_resolution:.5f} 1/K_0 resolution"
            )

        rt_sigma = self.determine_rt_sigma(rt_resolution)
        mobility_sigma = self.determine_mobility_sigma(mobility_resolution)

        if verbose:
            pass
            logger.info(
                f"FWHM in RT is {self.fwhm_rt:.2f} seconds, sigma is {rt_sigma:.2f}"
            )
            logger.info(
                f"FWHM in mobility is {self.fwhm_mobility:.3f} 1/K_0, sigma is {mobility_sigma:.2f}"
            )

        return self.gaussian_kernel_2d(
            self.kernel_width, self.kernel_height, rt_sigma, mobility_sigma
        ).astype(np.float32)

    @staticmethod
    def gaussian_kernel_2d(size_x: int, size_y: int, sigma_x: float, sigma_y: float):
        """
        Create a 2D gaussian kernel with a given size and standard deviation.

        Parameters
        ----------

        size : int
            Width and height of the kernel matrix.

        sigma_x : float
            Standard deviation of the gaussian kernel in x direction. This will correspond to the RT dimension.

        sigma_y : float
            Standard deviation of the gaussian kernel in y direction. This will correspond to the mobility dimension.

        Returns
        -------

        weights : np.ndarray, dtype=np.float32
            2D gaussian kernel matrix of shape (size, size).

        """
        # create indicies [-2, -1, 0, 1 ...]
        x, y = np.meshgrid(
            np.arange(-size_x // 2, size_x // 2), np.arange(-size_y // 2, size_y // 2)
        )
        xy = np.column_stack((x.flatten(), y.flatten())).astype("float32")

        # mean is always zero
        mu = np.array([[0.0, 0.0]])

        # sigma is set with no covariance
        sigma_mat = np.array([[sigma_x, 0.0], [0.0, sigma_y]])

        weights = multivariate_normal(xy, mu, sigma_mat)
        return weights.reshape(size_y, size_x).astype(np.float32)
