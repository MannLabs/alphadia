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

    mz_calibration_loaded = calibration.Calibration()
    mz_calibration_loaded.load(path)
    mz_calibration_loaded.predict(df_loaded)

    assert np.allclose(df_original['calibrated_mz'], df_loaded['calibrated_mz'])

def test_manager_loading():

    """ Test the calibration manager.
    """
    # initialize the calibration manager
    calibration_manager = calibration.CalibrationManager()

    # load the config from a dictionary. The dictionary could be loaded from a yaml file
    calibration_manager.load_config([
    {
        'name': 'precursor',
        'estimators': [
            {
                'name': 'mz',
                'model': 'LinearRegression',

                'input_columns': ['mz_library'],
                'target_columns': ['mz_observed'],
                'output_columns': ['mz_calibrated'],
                'transform_deviation': 1e6
            },
            {
                'name': 'rt',
                'model': 'LOESSRegression',
                'model_args': {
                    'n_kernels': 2
                },
                'input_columns': ['rt_library'],
                'target_columns': ['rt_observed'],
                'output_columns': ['rt_calibrated'],
                'transform_deviation': None
            },
        ]
    },
    {
        'name': 'fragment',
        'estimators': [
            {
                'name': 'mz',
                'model': 'LinearRegression',

                'input_columns': ['mz_library'],
                'target_columns': ['mz_observed'],
                'output_columns': ['mz_calibrated'],
                'transform_deviation': 1e6
            }]
    }
    ])

    assert len(calibration_manager.estimator_groups) == 2
    assert len(calibration_manager.get_estimator_names('precursor')) == 2
    assert len(calibration_manager.get_estimator_names('fragment')) == 1

    assert calibration_manager.get_estimator('precursor', 'mz').name == 'mz'
    assert calibration_manager.get_estimator('precursor', 'rt').name == 'rt'
    assert calibration_manager.get_estimator('fragment', 'mz').name == 'mz'

    assert isinstance(calibration_manager.get_estimator('precursor', 'mz'), calibration.Calibration)
    assert isinstance(calibration_manager.get_estimator('precursor', 'rt'), calibration.Calibration)
    assert isinstance(calibration_manager.get_estimator('fragment', 'mz'), calibration.Calibration)

    assert isinstance(calibration_manager.get_estimator('precursor', 'mz').function , LinearRegression)
    assert isinstance(calibration_manager.get_estimator('precursor', 'rt').function , LOESSRegression)
    assert isinstance(calibration_manager.get_estimator('fragment', 'mz').function , LinearRegression)

    # create some test data and make sure estimation works
    mz_library = np.linspace(100, 1000, 1000)
    mz_observed = mz_library + np.random.normal(0, 0.001, 1000) + mz_library * 0.00001 + 0.005

    rt_library = np.linspace(0, 100, 1000)
    rt_observed = rt_library + np.random.normal(0, 0.5, 1000) + np.sin(rt_library * 0.05)

    test_df = pd.DataFrame({
        'mz_library': mz_library, 
        'mz_observed': mz_observed,
        'rt_library': rt_library,
        'rt_observed': rt_observed
    })

    calibration_manager.fit_predict(test_df, 'precursor', plot=False)

    assert 'mz_calibrated' in test_df.columns
    assert 'rt_calibrated' in test_df.columns

    # save and load the calibration manager
    # make sure the loaded calibration manager works

    temp_path = path = os.path.join(tempfile.tempdir, 'calibration.pkl')
    calibration_manager.save(temp_path)

    calibration_manager_loaded = calibration.CalibrationManager()
    calibration_manager_loaded.load(temp_path)

    test_df.drop(columns=['mz_calibrated', 'rt_calibrated'], inplace=True)
    calibration_manager_loaded.predict(test_df, 'precursor')

    assert 'mz_calibrated' in test_df.columns
    assert 'rt_calibrated' in test_df.columns