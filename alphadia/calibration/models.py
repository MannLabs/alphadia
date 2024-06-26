import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import PolynomialFeatures


class LOESSRegression(BaseEstimator, RegressorMixin):
    """scikit-learn estimator which implements a LOESS style local polynomial regression. The number of basis functions or kernels can be explicitly defined which allows for faster and cheaper training and inference.

    Parameters
    ----------

    n_kernels : int
        default = 6, The number of local polynomial functions used to approximate the data. The location and extend of the kernels will be distributed to contain an equal number of datapoints in the training set.

    kernel_size : float
        default = 2, A factor increasing the kernel size to overlap with the neighboring kernel.

    polynomial_degree : int
        default = 2, Degree of the polynomial functions used for the local approximation.

    uniform : bool
        default = False, If True, the kernels are distributed uniformly over the input space.
        If False, the kernels are distributed to contain an equal number of datapoints.
        For every kernel at least polynomial_degree + 1 datapoints are required.

    """

    def __init__(
        self,
        n_kernels: int = 6,
        kernel_size: float = 2.0,
        polynomial_degree: int = 2,
        uniform=False,
    ):
        self.n_kernels = n_kernels
        self.kernel_size = kernel_size
        self.polynomial_degree = polynomial_degree
        self.uniform = uniform

    def _more_tags(self):
        return {"X_types": ["1darray"]}

    def _intervals_uniform(self, x: np.ndarray):
        """Determine the intervals of the kernels.
        The kernels are distributed uniformly over the input space.

        Parameters
        ----------
        x : numpy.ndarray
            float, of shape (n_datapoints)

        Returns
        -------
        numpy.ndarray, float
            of shape (n_kernels, 2)
        """

        minval = x[0]
        maxval = x[-1]

        interval_size = (maxval - minval) / self.n_kernels

        start = np.arange(minval, maxval, interval_size) - (interval_size / 2) * (
            self.kernel_size - 1
        )
        stop = start + interval_size + (interval_size) * (self.kernel_size - 1)
        return np.column_stack([start, stop])

    def _kernel_indices_uniform(self, x: np.ndarray):
        """Determine the indices of the datapoints belonging to each kernel.
        The kernels are distributed uniformly over the input space.

        Parameters
        ----------
        x : numpy.ndarray
            float, of shape (n_datapoints)

        Returns
        -------
        numpy.ndarray, int
            of shape (n_kernels, 2)

        """

        indices = np.searchsorted(x, self._intervals_uniform(x))

        return indices.astype(int)

    def _kernel_indices_density(self, x: np.ndarray):
        """Determine the indices of the datapoints belonging to each kernel.
        The kernels are distributed to contain an equal number of datapoints.

        Parameters
        ----------
        x : numpy.ndarray
            float, of shape (n_datapoints)

        Returns
        -------
        numpy.ndarray, int
            of shape (n_kernels, 2)

        """

        num_datapoints = len(x)
        interval_size = num_datapoints // self.n_kernels

        start = np.arange(0, self.n_kernels) * interval_size
        end = start + interval_size

        interval_extension = (interval_size * self.kernel_size - interval_size) // 2

        start = start - interval_extension
        start = np.maximum(0, start)

        end = end + interval_extension
        end = np.minimum(num_datapoints, end)

        return np.column_stack([start, end]).astype(int)

    def fit(self, x: np.ndarray, y: np.ndarray):
        """fit the model passed on provided training data.

        Parameters
        ----------

        x : numpy.ndarray
            float, of shape (n_samples,) or (n_samples, 1), Training data. Note that only a single feature is supported at the moment.

        y : numpy.ndarray, float
            of shape (n_samples,) or (n_samples, 1) Target values.

        Returns
        -------

        self: object
            Returns the fitted estimator.

        """

        # As required by scikit-learn estimator guidelines
        self.n_features_in_ = 1

        # === start === sanity checks ===
        # Does not yet work with more than one input dimension
        # axis-wise scaling and improved distance function need to be implemented
        if len(x.shape) > 1 and x.shape[1] > 1:
            raise ValueError(
                "Input arrays with more than one feature not yet supported. Please provide a matrix of shape (n_datapoints, 1) or (n_datapoints,)"
            )

        # at least two datapoints required
        if len(x.flat) < 2:
            raise ValueError("At least two datapoints required for fitting.")

        # sanity check for number of datapoints, reduce n_kernels if needed
        degrees_freedom = (1 + self.polynomial_degree) * self.n_kernels

        if len(x.flat) < degrees_freedom:
            print(
                f"Curve fitting with {self.n_kernels} kernels and polynomials of {self.polynomial_degree} degree requires at least {degrees_freedom} datapoints."
            )

            self.n_kernels = np.max([len(x.flat) // (1 + self.polynomial_degree), 1])

            print(f"Number of kernels will be reduced to {self.n_kernels} kernels.")

        # sanity check for number of datapoints, reduce degree of polynomial if necessary
        degrees_freedom = (1 + self.polynomial_degree) * self.n_kernels
        if len(x.flat) < degrees_freedom:
            self.polynomial_degree = len(x.flat) - 1

            print(f"Polynomial degree will be reduced to {self.polynomial_degree}.")

        # reshape both arrays to column arrays
        if len(x.shape) == 1:
            x = x[..., np.newaxis]

        if len(y.shape) == 1:
            y = y[..., np.newaxis]

        # remove outliers by using only the 0.5 to 99.5 percentile
        percentiles = np.percentile(x, [0.1, 99.9])
        mask = (percentiles[0] < x[:, 0]) & (x[:, 0] < percentiles[1])
        x = x[mask, ...]
        y = y[mask, ...]

        # === end === sanity checks ===

        # create flat version of the array for
        idx_sorted = np.argsort(x.flat)
        x_sorted = x.flat[idx_sorted]

        # stores if uniform training is still possible this round
        uniform = self.uniform

        # === start === kernel indices ===
        # get kernel indices matrix of shape (n_kernels, 2)
        if uniform:
            kernel_indices = self._kernel_indices_uniform(x_sorted)

            # check number of datapoints per kernel
            if np.any(np.diff(kernel_indices) < (1 + self.polynomial_degree)):
                print(
                    "Too few datapoints per kernel. Uniform kernels will be replaced by density kernels."
                )
                uniform = False

        # a second if statement is used instead of an if-else to account for failed uniform training
        if not uniform:
            kernel_indices = self._kernel_indices_density(x_sorted)

        # === end === kernel indices ===

        # === start === calculate kernel dimensions ===
        if uniform:
            start_stop = self._intervals_uniform(x_sorted)
            self.scale_mean = np.mean(start_stop, axis=1)
            self.scale_max = np.max(start_stop, axis=1) - self.scale_mean

        else:
            # scale max and scale mean will then be used for calculating the weighht matrix
            self.scale_mean = np.zeros(self.n_kernels)
            self.scale_max = np.zeros(self.n_kernels)

            # scale mean and max are calculated and contain the scaling before applying the kernel
            for i, area in enumerate(kernel_indices):
                area_slice = slice(*area)
                self.scale_mean[i] = x_sorted[area_slice].mean()
                self.scale_max[i] = np.max(
                    np.abs(x_sorted[area_slice] - self.scale_mean[i])
                )

        # === end === calculate kernel dimensions ===

        # from here on, the original column arrays are used
        w = self._get_weight_matrix(x)

        # build design matrix
        polynomial_transform = PolynomialFeatures(self.polynomial_degree)
        x_design = polynomial_transform.fit_transform(x)
        number_of_dimensions = len(x_design[0])

        self.beta = np.zeros((number_of_dimensions, self.n_kernels))

        for i, weights in enumerate(w.T):
            loadings = np.linalg.inv(x_design.T * weights @ x_design) @ x_design.T
            self.beta[:, i] = np.ravel((loadings * weights) @ y)

        return self

    def predict(self, x: np.ndarray):
        """Predict using the LOESS model.

        Parameters
        ----------

        x : numpy.ndarray
            float, of shape (n_samples,) or (n_samples, 1) Feature data. Note that only a single feature is supported at the moment.

        Returns
        -------

        y : numpy.ndarray, float
        of shape (n_samples,)
            Target values.

        """

        if len(x.shape) == 1:
            x = x[..., np.newaxis]

        w = self._get_weight_matrix(x)
        polynomial_transform = PolynomialFeatures(self.polynomial_degree)
        x_design = polynomial_transform.fit_transform(x)

        return np.sum(x_design @ self.beta * w, axis=1)

    def _get_weight_matrix(self, x: np.ndarray):
        """Applies the fitted scaling parameter and the kernel to yield a weight matrix.

        The weight matrix is calculated based on the self.scale_mean and self.scale_max parameters which need to be calculated before calling this function.
        They define the center and extend of the tricubic kernels. The first and last column are one-padded at the start and beginning to allow for extrapolation.

        Parameters
        ----------

        x: numpy.ndarray
            Numpy array of shape (n_datapoints, 1) which should be transformed to weights.


        Returns
        -------

        numpy.ndarray
            Weight matrix with the shape (n_datapoints, n_kernels).

        """
        w = np.tile(x, (1, self.n_kernels))

        w = w - self.scale_mean
        w = w / self.scale_max

        # apply weighting kernel
        w = _apply_kernel(w)

        w = w / np.sum(w, axis=1, keepdims=True)

        return w


def _apply_kernel(w):
    num_cols = w.shape[1]

    if num_cols == 1:
        return np.ones(w.shape)

    if num_cols == 2:
        w[:, 0] = _left_open_tricubic(w[:, 0])
        w[:, 1] = _right_open_tricubic(w[:, 1])

        return w

    if num_cols > 2:
        w[:, 0] = _left_open_tricubic(w[:, 0])
        w[:, 1:-1] = _tricubic(w[:, 1:-1])
        w[:, -1] = _right_open_tricubic(w[:, -1])

        return w


def _tricubic(x, EPSILON=1e-6):
    """tricubic weight kernel"""
    epsilon = EPSILON
    mask = np.abs(x) <= 1
    return mask * (np.power(1 - np.power(np.abs(x), 3), 3) + epsilon)


def _left_open_tricubic(x):
    """tricubic weight kernel which weights assigns 1 to values x < 0"""
    y = _tricubic(x)
    y[x < 0] = 1
    return y


def _right_open_tricubic(x):
    """tricubic weight kernel which weights assigns 1 to values x > 0"""
    y = _tricubic(x)
    y[x > 0] = 1
    return y
