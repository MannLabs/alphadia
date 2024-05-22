import pytest
import numpy as np
from alphadia.calibration.models import LOESSRegression


def _noisy_1d(x):
    y = np.sin(x)
    y_err = np.random.normal(y, 0.5)
    return y + y_err + 0.5 * x


@pytest.mark.parametrize("uniform", [True, False])
@pytest.mark.parametrize("n_kernels", [1, 10])
@pytest.mark.parametrize("polynomial_degree", [1, 2])
def test_loess_regression(uniform, n_kernels, polynomial_degree):
    x_train = np.linspace(0, 15, 100)
    y_train = _noisy_1d(x_train)
    x_test = np.linspace(0, 15, 10)
    y_test = (
        LOESSRegression(
            n_kernels=n_kernels, polynomial_degree=polynomial_degree, uniform=uniform
        )
        .fit(x_train, y_train)
        .predict(x_test)
    )

    assert len(y_test) == 10
