from alphadia.extraction import calibration

from sklearn.linear_model import LinearRegression
from alphabase.statistics.regression import LOESSRegression

import numpy as np
import pandas as pd

import pytest

import tempfile
import os

def test_uninitialized_calibration():

    library_mz = np.linspace(100, 1000, 100)
    observed_mz = library_mz + np.random.normal(0, 0.1, 100) + library_mz * 0.001
    mz_df = pd.DataFrame({'library_mz': library_mz, 'observed_mz': observed_mz})

    # test that an error is raised if the calibration is not initialized
    mz_calibration = calibration.Calibration()
    with pytest.raises(ValueError):
        mz_calibration.fit(mz_df)

def test_fit_predict_linear():

    library_mz = np.linspace(100, 1000, 100)
    observed_mz = library_mz + np.random.normal(0, 0.1, 100) + library_mz * 0.001
    mz_df = pd.DataFrame({'library_mz': library_mz, 'observed_mz': observed_mz})
    
    mz_calibration = calibration.Calibration(
        name = 'mz_calibration',
        function = LinearRegression(),
        input_columns=['library_mz'],
        target_columns=['observed_mz'],
        output_columns=['calibrated_mz']
    )

    mz_calibration.fit(mz_df, plot=False)
    mz_calibration.predict(mz_df)

    assert 'calibrated_mz' in mz_df.columns

def test_fit_predict_loess():

    library_mz = np.linspace(100, 1000, 100)
    observed_mz = library_mz + np.random.normal(0, 0.1, 100) + library_mz * 0.001
    mz_df = pd.DataFrame({'library_mz': library_mz, 'observed_mz': observed_mz})
    
    mz_calibration = calibration.Calibration(
        name = 'mz_calibration',
        function = LOESSRegression(),
        input_columns=['library_mz'],
        target_columns=['observed_mz'],
        output_columns=['calibrated_mz']
    )

    mz_calibration.fit(mz_df, plot=False)
    mz_calibration.predict(mz_df)

    assert 'calibrated_mz' in mz_df.columns


def test_save_load():

    library_mz = np.linspace(100, 1000, 100)
    observed_mz = library_mz + np.random.normal(0, 0.1, 100) + library_mz * 0.001
    mz_df = pd.DataFrame({'library_mz': library_mz, 'observed_mz': observed_mz})

    mz_calibration = calibration.Calibration(
        name = 'mz_calibration',
        function = LinearRegression(),
        input_columns=['library_mz'],
        target_columns=['observed_mz'],
        output_columns=['calibrated_mz']
    )

    df_original = mz_df.copy()
    df_loaded = mz_df.copy()

    mz_calibration.fit(mz_df)
    mz_calibration.predict(df_original)

    path = os.path.join(tempfile.tempdir, 'mz_calibration.pkl')
    mz_calibration.save(path)

    mz_calibration_loaded = calibration.Calibration().load(path)
    mz_calibration_loaded.predict(df_loaded)

    assert np.allclose(df_original['calibrated_mz'], df_loaded['calibrated_mz'])