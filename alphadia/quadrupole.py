# native imports

# alphadia imports
# alpha family imports
import alphatims.utils

# third party imports
import numba as nb
import numpy as np
from numba.experimental import jitclass
from scipy.optimize import curve_fit

from alphadia import utils


@alphatims.utils.njit
def logistic(x: np.array, mu: float, sigma: float):
    """Numba implementation of the logistic function

    Parameters
    ----------

    x : np.array
        Input array of shape `(n_samples,)`

    mu : float
        Mean of the logistic function

    sigma : float
        Standard deviation of the logistic function

    Returns
    -------

    np.array
        Logistic function evaluated for every element in x of shape `(n_samples,)`

    """
    a = (x - mu) / sigma
    y = 1 / (1 + np.exp(-a))
    return y


@alphatims.utils.njit
def logistic_rectangle(mu1, mu2, sigma1, sigma2, x):
    y = logistic(x, mu1, sigma1) - logistic(x, mu2, sigma2)
    return y


@alphatims.utils.njit
def linear(x, m, b):
    return m * x + b


@jitclass
class SimpleQuadrupoleJit:
    # original cycle as defined in the Bruker file
    cycle: nb.float64[:, :, :, ::1]

    # calibrated cycle which covers the 1% treshold of the quadrupole
    cycle_calibrated: nb.float64[:, :, :, ::1]

    dia_mz_cycle_calibrated: nb.float64[:, ::1]

    # left and right sigma of the logistic function
    # shared across all precursors and scans
    sigma: nb.float64[::1]

    # left and right delta mu of the logistic function
    # shared across all precursors and scans
    delta_mu: nb.float64[::1]

    def __init__(self, cycle):
        """
        Jitclass for predicting quadrupole transfer efficiency.
        Only used to store and predict the quadrupole transfer efficiency.

        Fitting is performed by an outside wrapper.

        Parameters
        ----------
        cycle : np.ndarray
            The dia cycle as defined in the Bruker file
        """
        self.cycle = cycle
        self.sigma = np.array([0.2, 0.2])
        self.delta_mu = np.array([0.0, 0.0])

    def predict(self, P, S, X):
        """
        Predict the quadrupole transfer efficiency

        Parameters
        ----------

        P : np.ndarray
            Precursor index for N datapoints

        S : np.ndarray
            Scan index for N datapoints

        X : np.ndarray
            m/z value for N datapoints

        Returns
        -------
        np.ndarray
            Quadrupole transfer efficiency for N datapoints

        """

        mu1l = [0.0]
        mu2l = [0.0]
        for i in range(len(P)):
            c = P[i]
            s = S[i]
            # print(self.cycle[0, c, s, 0])
            mu1l.append(self.cycle[0, c, s, 0])
            mu2l.append(self.cycle[0, c, s, 1])

        mu1 = np.array(mu1l)[1:] + self.delta_mu[0]
        mu2 = np.array(mu2l)[1:] + self.delta_mu[1]

        return logistic_rectangle(mu1, mu2, self.sigma[0], self.sigma[1], X)

    def set_cycle_calibrated(self, cycle_calibrated):
        self.cycle_calibrated = cycle_calibrated
        self.dia_mz_cycle_calibrated = np.reshape(
            cycle_calibrated, (cycle_calibrated.shape[1] * cycle_calibrated.shape[2], 2)
        )

    def get_dia_mz_cycle(self, lower_mz, upper_mz):
        expanded_cycle = expand_cycle(self.cycle_calibrated, lower_mz, upper_mz)
        return np.reshape(
            expanded_cycle, (expanded_cycle.shape[1] * expanded_cycle.shape[2], 2)
        )


class SimpleQuadrupole:
    def __init__(
        self,
        cycle,
    ):
        """
        Wrapper for fitting the quadrupole transfer efficiency.

        Parameters
        ----------
        cycle : np.ndarray
            The dia cycle as defined in the Bruker file

        Properties
        ----------
        jit : SimpleQuadrupoleJit
            Jitclass for predicting quadrupole transfer efficiency.

        """
        self.cycle = cycle
        self.jit = SimpleQuadrupoleJit(cycle)
        self.jit.set_cycle_calibrated(self.get_calibrated_cycle())

    def get_params(self, deep: bool = True):
        return super().get_params(deep)

    def set_params(self, **params):
        return super().set_params(**params)

    def _more_tags(self):
        return {"X_types": ["2darray"]}

    def fit(self, P, S, X, y):
        """
        Fit the quadrupole transfer efficiency.

        Parameters
        ----------
        P : np.ndarray
            Precursor index for N datapoints

        S : np.ndarray
            Scan index for N datapoints

        X : np.ndarray
            m/z value for N datapoints

        y : np.ndarray
            Quadrupole transfer efficiency for N datapoints

        Returns
        -------
        self : SimpleQuadrupole
            Fitted SimpleQuadrupole object

        """

        mu1 = self.jit.cycle[0, P, S, 0]
        mu2 = self.jit.cycle[0, P, S, 1]
        X_train = np.stack([mu1, mu2, X], axis=1)

        def _wrapper(X, sigma1, sigma2, delta_mu1, delta_mu2):
            mu1 = X[:, 0] + delta_mu1
            mu2 = X[:, 1] + delta_mu2
            x = X[:, 2]
            return logistic_rectangle(mu1, mu2, sigma1, sigma2, x)

        p0 = np.concatenate([self.jit.sigma, self.jit.delta_mu])

        popt, pcov = curve_fit(_wrapper, X_train, y, p0=p0)

        self.jit.sigma = popt[:2]
        self.jit.delta_mu = popt[2:]

        self.jit.set_cycle_calibrated(self.get_calibrated_cycle())

        return self

    def predict(self, P, S, X):
        """
        Fit the quadrupole transfer efficiency.

        Parameters
        ----------
        P : np.ndarray
            Precursor index for N datapoints

        S : np.ndarray
            Scan index for N datapoints

        X : np.ndarray
            m/z value for N datapoints

        """

        return self.jit.predict(P, S, X)

    def get_calibrated_cycle(self, treshold=0.01):
        """
        Calculate an updated cycle based on the fitted quadrupole transfer efficiency and the treshold.
        """
        non_zero_cycle = self.jit.cycle[self.jit.cycle > 0]

        lowest_mz = np.min(non_zero_cycle)
        highest_mz = np.max(non_zero_cycle)
        mz_width = highest_mz - lowest_mz

        mz_space = np.linspace(
            lowest_mz - mz_width * 0.1, highest_mz + mz_width * 0.1, 2000
        )

        new_cycle = self.jit.cycle.copy()
        n_precursor = self.jit.cycle.shape[1]
        n_scan = self.jit.cycle.shape[2]

        for precursor in range(n_precursor):
            for scan in range(n_scan):
                if self.jit.cycle[0, precursor, scan, 0] <= 0:
                    continue

                intensity = self.jit.predict(
                    np.array([precursor]), np.array([scan]), mz_space
                )
                q_range = mz_space[intensity > treshold]

                new_cycle[0, precursor, scan, 0] = np.min(q_range)
                new_cycle[0, precursor, scan, 1] = np.max(q_range)

        return new_cycle


@alphatims.utils.njit
def quadrupole_transfer_function_single(
    quadrupole_calibration_jit, observation_indices, scan_indices, isotope_mz
):
    """
    Calculate quadrupole transfer function for a given set of observations and scans.

    Parameters
    ----------
    quadrupole_calibration_jit : alphadia.quadrupole.SimpleQuadrupoleJit
        Quadrupole calibration jit object

    observation_indices : np.ndarray
        Array of observation indices, shape (n_observations,)

    scan_indices : np.ndarray
        Array of scan indices, shape (n_scans,)

    isotope_mz : np.ndarray
        Array of precursor isotope m/z values, shape (n_isotopes)

    Returns
    -------

    intensity : np.ndarray
        Array of predicted intensity values, shape (n_isotopes, n_observations, n_scans)

    """

    n_isotopes = isotope_mz.shape[0]
    n_observations = observation_indices.shape[0]
    n_scans = scan_indices.shape[0]

    mz_column = np.repeat(isotope_mz, n_scans * n_observations)
    observation_column = utils.tile(np.repeat(observation_indices, n_scans), n_isotopes)
    scan_column = utils.tile(scan_indices, n_isotopes * n_observations)

    intensity = quadrupole_calibration_jit.predict(
        observation_column, scan_column, mz_column
    )
    return intensity.reshape(n_isotopes, n_observations, n_scans)


@nb.njit
def calculate_template_single(qtf, dense_precursor_mz, isotope_intensity):
    # select only the intensity channel
    # expand observation dimension to the number of fragment observations
    precursor_mz = dense_precursor_mz[0]

    # unravel precursors and isotopes
    # precursor_mz = precursor_mz.reshape(n_isotopes, 1, n_scans, n_frames)

    # expand add frame dimension to qtf
    # (n_isotopes, n_observations, n_scans, n_frames)
    qtf_exp = np.expand_dims(qtf, axis=-1)

    # (n_isotopes, n_observations, n_scans, n_frames)
    isotope_exp = isotope_intensity.reshape(-1, 1, 1, 1)

    template = precursor_mz * isotope_exp * qtf_exp
    template = template.sum(axis=0)

    # (n_observations, n_scans, n_frames)
    return template.astype(np.float32)


@nb.njit
def calculate_observation_importance(
    template,
):
    observation_importance = np.sum(np.sum(template, axis=2), axis=2)
    return observation_importance / np.sum(observation_importance, axis=1).reshape(
        -1, 1
    )


@nb.njit
def calculate_observation_importance_single(
    template,
):
    observation_importance = np.sum(np.sum(template, axis=-1), axis=-1)
    if np.sum(observation_importance) == 0:
        return np.ones_like(observation_importance) / observation_importance.shape[0]
    else:
        return observation_importance / np.sum(observation_importance)


@nb.njit
def expand_cycle(cycle, lower_mz, upper_mz):
    new_cycle = cycle.copy()

    for i in range(cycle.shape[0]):
        for j in range(cycle.shape[1]):
            new_cycle[i, j, :, 0] -= lower_mz * (new_cycle[i, j, :, 0] > 0)
            new_cycle[i, j, :, 1] += upper_mz * (new_cycle[i, j, :, 1] > 0)

    return new_cycle
