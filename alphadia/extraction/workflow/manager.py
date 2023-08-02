import os
import typing
import pickle
import logging

import pandas as pd
import numpy as np
logger = logging.getLogger()

import alphadia
from alphadia.extraction import calibration

class BaseManager():

    def __init__(
            self,
            path : typing.Union[None, str] = None,
            load_from_file : bool = True
        ):

        """Base class for all managers which handle parts of the workflow.
        
        Parameters
        ----------

        path : str, optional
            Path to the manager pickle on disk.

        load_from_file : bool, optional
            If True, the manager will be loaded from file if it exists.
        """

        self._path = path
        self.is_loaded_from_file = False
        self.is_fitted = False

        self._version = alphadia.__version__

        if load_from_file:
            self.load()

    @property
    def path(self):
        """Path to the manager pickle on disk.
        """
        return self._path

    @property
    def is_loaded_from_file(self):
        """Check if the calibration manager was loaded from file.
        """
        return self._is_loaded_from_file
    
    @is_loaded_from_file.setter
    def is_loaded_from_file(self, value):
        self._is_loaded_from_file = value
    
    @property
    def is_fitted(self):
        """Check if all estimators in all calibration groups are fitted.
        """
        return self._is_fitted
    
    @is_fitted.setter
    def is_fitted(self, value):
        self._is_fitted = value
    
    def save(self):
        """Save the state to pickle file.
        """
        if self.path is not None:
            try:
                with open(self.path, 'wb') as f:
                    pickle.dump(self, f)
            except:
                logging.error(f'Failed to save {self.__class__.__name__} to {self.path}')

    def load(self):

        """Load the state from pickle file.
        """
        if self.path is not None:
            if os.path.exists(self.path):
                try:
                    with open(self.path, 'rb') as f:
                        loaded_state = pickle.load(f)

                        if loaded_state._version == self._version:
                            self.__dict__.update(loaded_state.__dict__)
                            self.is_loaded_from_file = True
                        else:
                            logging.warning(f'Version mismatch while loading {self.__class__}: {loaded_state._version} != {self._version}. Will not load.')
                except:
                    logging.error(f'Failed to load {self.__class__.__name__} from {self.path}')
                else:
                    logging.info(f'Loaded {self.__class__.__name__} from {self.path}')
            else:
                logging.warning(f'{self.__class__.__name__} not found at: {self.path}')

    
    def fit(self):
        """Fit the workflow property of the manager.
        """
        raise NotImplementedError(f'fit() not implemented for {self.__class__.__name__}')
    
    def predict(self):
        """Return the predictions of the workflow property of the manager.
        """
        raise NotImplementedError(f'predict() not implemented for {self.__class__.__name__}')
    
    def fit_predict(self):
        """Fit and return the predictions of the workflow property of the manager.
        """
        raise NotImplementedError(f'fit_predict() not implemented for {self.__class__.__name__}')
    
class CalibrationManager(BaseManager):

    def __init__(
            self,
            config : typing.Union[None, dict] = None,
            path : typing.Union[None, str] = None,
            load_from_file : bool = True
        ):

        """Contains, updates and applies all calibrations for a single run.

        Calibrations are grouped into calibration groups. Each calibration group is applied to a single data structure (precursor dataframe, fragment fataframe, etc.). Each calibration group contains multiple estimators which each calibrate a single property (mz, rt, etc.). Each estimator is a `Calibration` object which contains the estimator function.
        
        Parameters
        ----------

        config : typing.Union[None, dict], default=None
            Calibration config dict. If None, the default config is used.

        path : str, default=None
            Path where the current parameter set is saved to and loaded from.

        load_from_file : bool, default=True
            If True, the manager will be loaded from file if it exists.
        
        """
        logging.info('========= Initializing Calibration Manager =========')
        super().__init__(path=path, load_from_file=load_from_file)

        if not self.is_loaded_from_file:
            self.estimator_groups = []
            self.load_config(config)
        
        logging.info('====================================================')


    @property
    def estimator_groups(self):
        """List of calibration groups.
        """
        return self._estimator_groups
    
    @estimator_groups.setter
    def estimator_groups(self, value):
        self._estimator_groups = value


    def load_config(self, config : dict):
        """Load calibration config from config Dict.

        each calibration config is a list of calibration groups which consist of multiple estimators.
        For each estimator the `model` and `model_args` are used to request a model from the calibration_model_provider and to initialize it.
        The estimator is then initialized with the `Calibration` class and added to the group.

        Parameters
        ----------

        config : dict
            Calibration config dict

        Example
        -------

        Create a calibration manager with a single group and a single estimator:

        .. code-block:: python

            calibration_manager = calibration.CalibrationManager()
            calibration_manager.load_config([{
                'name': 'mz_calibration',
                'estimators': [
                    {
                        'name': 'mz',
                        'model': 'LOESSRegression',
                        'model_args': {
                            'n_kernels': 2
                        },
                        'input_columns': ['mz_library'],
                        'target_columns': ['mz_observed'],
                        'output_columns': ['mz_calibrated'],
                        'transform_deviation': 1e6
                    },
                    
                ]
            }])
        
        """
        
        logging.info('loading calibration config')
        logging.info(f'found {len(config)} calibration groups')
        for group in config:
            logging.info(f'Calibration group :{group["name"]}, found {len(group["estimators"])} estimator(s)')
            for estimator in group['estimators']:
                try:
                    template = calibration.calibration_model_provider.get_model(estimator['model'])
                    model_args = estimator['model_args'] if 'model_args' in estimator else {}
                    estimator['function'] = template(**model_args)
                except Exception as e:
                    logging.error(f'Could not load estimator {estimator["name"]}: {e}')

            group_copy = {'name': group['name']} 
            group_copy['estimators'] = [calibration.Calibration(**x) for x in group['estimators']]
            self.estimator_groups.append(group_copy)

    def get_group_names(self):
        """Get the names of all calibration groups.

        Returns
        -------
        list of str
            List of calibration group names
        """

        return [x['name'] for x in self.estimator_groups]

    def get_group(self, group_name : str):
        """Get the calibration group by name.

        Parameters
        ----------

        group_name : str
            Name of the calibration group

        Returns
        -------
        dict
            Calibration group dict with `name` and `estimators` keys\
        
        """
        for group in self.estimator_groups:
            if group['name'] == group_name:
                return group

        logging.error(f'could not get_group: {group_name}')
        return None
    
    def get_estimator_names(self, group_name : str):
        """Get the names of all estimators in a calibration group.

        Parameters
        ----------

        group_name : str
            Name of the calibration group

        Returns
        -------
        list of str
            List of estimator names
        """

        group = self.get_group(group_name)
        if group is not None:
            return [x.name for x in group['estimators']]
        logging.error(f'could not get_estimator_names: {group_name}')
        return None

    def get_estimator(self, group_name : str, estimator_name : str):

        """Get an estimator from a calibration group.

        Parameters
        ----------

        group_name : str
            Name of the calibration group

        estimator_name : str
            Name of the estimator

        Returns
        -------
        Calibration
            The estimator object

        """
        group = self.get_group(group_name)
        if group is not None:
            for estimator in group['estimators']:
                if estimator.name == estimator_name:
                    return estimator
        logging.error(f'could not get_estimator: {group_name}, {estimator_name}')
        return None

    def fit(
        self, 
        df : pd.DataFrame, 
        group_name : str, 
        *args,
        **kwargs
        ):
        """Fit all estimators in a calibration group.

        Parameters
        ----------

        df : pandas.DataFrame
            Dataframe containing the input and target columns

        group_name : str
            Name of the calibration group

        """ 

        if len(self.estimator_groups) == 0:
            raise ValueError('No estimators defined')

        group_idx = [i for i, x in enumerate(self.estimator_groups) if x['name'] == group_name]
        if len(group_idx) == 0:
            raise ValueError(f'No group named {group_name} found')

        # only iterate over the first group with the given name
        for group in group_idx:
            for estimator in self.estimator_groups[group]['estimators']:
                logging.info(f'calibration group: {group_name}, fitting {estimator.name} estimator ')
                estimator.fit(df, *args, neptune_key=f'{group_name}_{estimator.name}', **kwargs)

        is_fitted = True
        # check if all estimators are fitted
        for group in self.estimator_groups:
            for estimator in group['estimators']:
                is_fitted = is_fitted and estimator.is_fitted

        self.is_fitted = is_fitted and len(self.estimator_groups) > 0

    def predict(
            self, 
            df : pd.DataFrame, 
            group_name : str, 
            *args, 
            **kwargs):
        
        """Predict all estimators in a calibration group.

        Parameters
        ----------

        df : pandas.DataFrame
            Dataframe containing the input and target columns

        group_name : str
            Name of the calibration group

        """

        if len(self.estimator_groups) == 0:
            raise ValueError('No estimators defined')

        group_idx = [i for i, x in enumerate(self.estimator_groups) if x['name'] == group_name]
        if len(group_idx) == 0:
            raise ValueError(f'No group named {group_name} found')
        for group in group_idx:
            for estimator in self.estimator_groups[group]['estimators']:
                logging.info(f'calibration group: {group_name}, predicting {estimator.name}')
                estimator.predict(df, inplace=True, *args, **kwargs)

    def fit_predict(
            self,
            df : pd.DataFrame,
            group_name : str,
            plot : bool = True,
        ):
        """Fit and predict all estimators in a calibration group.

        Parameters
        ----------

        df : pandas.DataFrame
            Dataframe containing the input and target columns

        group_name : str
            Name of the calibration group

        plot : bool, default=True
            If True, a plot of the calibration is generated.

        """
        self.fit(df, group_name, plot=plot)
        self.predict(df, group_name)

class OptimizationManager(BaseManager):

    def __init__(
        self,
        initial_parameters : dict,
        path : typing.Union[None, str] = None,
        load_from_file : bool = True
    ):

        logging.info('========= Initializing Optimization Manager =========')
        super().__init__(path=path, load_from_file=load_from_file)

        if not self.is_loaded_from_file:
            self.__dict__.update(initial_parameters)
            
            for key, value in initial_parameters.items():
                logging.info(f'initial parameter: {key} = {value}')
        
        logging.info('====================================================')

    def fit(self, update_dict):
        """Update the parameters dict with the values in update_dict.
        """
        self.__dict__.update(update_dict)
        self.is_fitted = True
        self.save()

    def predict(self):
        """Return the parameters dict.
        """
        return self.parameters
    
    def fit_predict(self, update_dict):
        """Update the parameters dict with the values in update_dict and return the parameters dict.
        """
        self.fit(update_dict)
        return self.predict()
    