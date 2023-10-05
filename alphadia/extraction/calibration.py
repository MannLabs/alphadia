# native imports
import os
import logging
import yaml 
import typing
import pickle

# alphadia imports
from alphadia.extraction.plotting.utils import density_scatter

# alpha family imports
import alphatims.bruker
import alphatims.utils
from alphabase.statistics.regression import LOESSRegression

# third party imports
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

import sklearn.base
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline


class Calibration():
    def __init__(self, 
                name : str = '',
                function : object = None,
                input_columns : typing.List[str] = [],
                target_columns : typing.List[str] = [],
                output_columns : typing.List[str] = [],
                transform_deviation : typing.Union[None, float] = None,
                **kwargs):
        """A single estimator for a property (mz, rt, etc.).

        Calibration is performed by modeling the deviation of an input values (e.g. mz_library) from an observed property (e.g. mz_observed) using a function (e.g. LinearRegression). Once calibrated, calibrated values (e.g. mz_calibrated) can be predicted from input values (e.g. mz_library). Additional explaining variabels can be added to the input values (e.g. rt_library) to improve the calibration.

        Parameters
        ----------

        name : str
            Name of the estimator for logging and plotting e.g. 'mz'
        
        function : object
            The estimator object instance which must have a fit and predict method.
            This will usually be a sklearn estimator or a custom estimator.

        input_columns : list of str
            The columns of the dataframe that are used as input for the estimator e.g. ['mz_library']. 
            The first column is the property which should be calibrated, additional columns can be used as explaining variables e.g. ['mz_library', 'rt_library'].

        target_columns : list of str
            The columns of the dataframe that are used as target for the estimator e.g. ['mz_observed'].
            At the moment only one target column is supported.

        output_columns : list of str
            The columns of the dataframe that are used as output for the estimator e.g. ['mz_calibrated'].
            At the moment only one output column is supported.
        
        transform_deviation : typing.List[Union[None, float]]
            If set to a valid float, the deviation is expressed as a fraction of the input value e.g. 1e6 for ppm.
            If set to None, the deviation is expressed in absolute units.

        """
        
        self.name = name
        self.function = function
        self.input_columns = input_columns
        self.target_columns = target_columns
        self.output_columns = output_columns
        self.transform_deviation = float(transform_deviation) if transform_deviation is not None else None
        self.is_fitted = False

    def __repr__(self) -> str:
        return f'<Calibration {self.name}, is_fitted: {self.is_fitted}>'

    def save(self, file_name: str):
        """Save the estimator to pickle file.

        Parameters
        ----------

        file_name : str
            Path to the pickle file

        """

        with open(file_name, 'wb') as f:
            pickle.dump(self, f)

    def load(self, file_name: str):
        """Load the estimator from pickle file.

        Parameters
        ----------

        file_name : str
            Path to the pickle file

        """

        with open(file_name, 'rb') as f:
            loaded_calibration = pickle.load(f)
            self.__dict__.update(loaded_calibration.__dict__)

    def validate_columns(
            self, 
            dataframe : pd.DataFrame
        ):
        """Validate that the input and target columns are present in the dataframe.

        Parameters
        ----------
        dataframe : pandas.DataFrame
            Dataframe containing the input and target columns

        Returns
        -------
        bool
            True if all columns are present, False otherwise

        """

        valid = True

        if len(self.target_columns) > 1 :
            logging.warning('Only one target column supported')
            valid = False

        required_columns = set(self.input_columns + self.target_columns)
        if not required_columns.issubset(dataframe.columns):
            logging.warning(f'{self.name}, at least one column {required_columns} not found in dataframe')
            valid = False

        return valid

    def fit(
            self, 
            dataframe : pd.DataFrame,
            plot : bool = False, 
            **kwargs
        ):
        """Fit the estimator based on the input and target columns of the dataframe.

        Parameters
        ----------

        dataframe : pandas.DataFrame
            Dataframe containing the input and target columns

        plot : bool, default=False
            If True, a plot of the calibration is generated.
        
        Returns
        -------

        np.ndarray
            Array of shape (n_input_columns, ) containing the mean absolute deviation of the residual deviation at the given confidence interval

        """

        if not self.validate_columns(dataframe):
            logging.warning(f'{self.name} calibration was skipped')
            return

        if self.function is None:
            raise ValueError('No estimator function provided')

        input_values = dataframe[self.input_columns].values
        target_value = dataframe[self.target_columns].values

        try:
            self.function.fit(input_values, target_value)
            self.is_fitted = True
        except Exception as e:
            logging.error(f'Could not fit estimator {self.name}: {e}')
            return

        if plot == True:
            self.plot(dataframe, **kwargs)


    def predict(self, dataframe, inplace=True):
        """Perform a prediction based on the input columns of the dataframe.

        Parameters
        ----------
        dataframe : pandas.DataFrame
            Dataframe containing the input and target columns

        inplace : bool, default=True
            If True, the prediction is added as a new column to the dataframe. If False, the prediction is returned as a numpy array.

        Returns
        -------
        np.ndarray
            Array of shape (n_samples, ) containing the prediction

        """

        if self.is_fitted == False:
            logging.warning(f'{self.name} prediction was skipped as it has not been fitted yet')
            return
        
        if not set(self.input_columns).issubset(dataframe.columns):
            logging.warning(f'{self.name} calibration was skipped as input column {self.input_columns} not found in dataframe')
            return

        input_values = dataframe[self.input_columns].values
        
        if inplace:
            dataframe[self.output_columns[0]] = self.function.predict(input_values)
        else:
            return self.function.predict(input_values)
        
    def fit_predict(
        self,
        dataframe : pd.DataFrame,
        plot : bool = False,
        inplace : bool = True
        ):
        """Fit the estimator and perform a prediction based on the input columns of the dataframe.

        Parameters
        ----------

        dataframe : pandas.DataFrame
            Dataframe containing the input and target columns
        
        plot : bool, default=False
            If True, a plot of the calibration is generated.

        inplace : bool, default=True
            If True, the prediction is added as a new column to the dataframe. If False, the prediction is returned as a numpy array.

        """
        self.fit(dataframe, plot=plot)
        return self.predict(dataframe, inplace=inplace)

    def deviation(self, dataframe : pd.DataFrame):
        """ Calculate the deviations between the input, target and calibrated values.

        Parameters
        ----------
        dataframe : pandas.DataFrame
            Dataframe containing the input and target columns

        Returns 
        -------
        np.ndarray 
            Array of shape (n_samples, 3 + n_input_columns). 
            The second dimension contains the observed deviation, calibrated deviation, residual deviation and the input values.
        
        """

        # the first column is the unclaibrated input property
        # all other columns are explaining variables
        input_values = dataframe[self.input_columns].values

        # the first column is the unclaibrated input property
        uncalibrated_values = input_values[:, [0]]

        # only one target column is supported
        target_values = dataframe[self.target_columns].values[:, [0]]
        input_transform = self.transform_deviation

        calibrated_values = self.predict(dataframe, inplace=False)
        if calibrated_values.ndim == 1:
            calibrated_values = calibrated_values[:, np.newaxis]

        # only one output column is supported
        calibrated_dim = calibrated_values[:, [0]]

        # deviation is the difference between the (observed) target value and the uncalibrated input value
        observed_deviation = target_values - uncalibrated_values
        if input_transform is not None:
            observed_deviation = observed_deviation/uncalibrated_values * float(input_transform)

        # calibrated deviation is the explained difference between the (calibrated) target value and the uncalibrated input value
        calibrated_deviation = calibrated_dim - uncalibrated_values
        if input_transform is not None:
            calibrated_deviation = calibrated_deviation/uncalibrated_values * float(input_transform)

        # residual deviation is the unexplained difference between the (observed) target value and the (calibrated) target value
        residual_deviation = observed_deviation - calibrated_deviation

        return np.concatenate([observed_deviation, calibrated_deviation, residual_deviation, input_values], axis=1)

    def ci(self, dataframe, ci : float = 0.95):
        """Calculate the residual deviation at the given confidence interval.

        Parameters
        ----------

        dataframe : pandas.DataFrame
            Dataframe containing the input and target columns

        ci : float, default=0.95
            confidence interval

        Returns
        -------

        float
            the confidence interval of the residual deviation after calibration
        """
        
        if not 0 < ci < 1:
            raise ValueError('Confidence interval must be between 0 and 1')
        
        if not self.is_fitted:
            return 0

        ci_percentile = [100*(1-ci)/2, 100*(1+ci)/2]
        
        deviation = self.deviation(dataframe)
        residual_deviation = deviation[:, 2]
        return np.mean(np.abs(np.percentile(residual_deviation, ci_percentile)))

    def get_transform_unit(
            self, 
            transform_deviation : typing.Union[None, float]
        ):

        """Get the unit of the deviation based on the transform deviation.
        
        Parameters
        ----------
        
        transform_deviation : typing.Union[None, float]
            If set to a valid float, the deviation is expressed as a fraction of the input value e.g. 1e6 for ppm.
            
        Returns
        -------
        str
            The unit of the deviation

        """
        if transform_deviation is not None:
            if np.isclose(transform_deviation,1e6):
                return '(ppm)'
            elif np.isclose(transform_deviation,1e2):
                return '(%)'
            else:
                return f'({transform_deviation})'
        else:
            return '(absolute)'


    def plot(
            self, 
            dataframe : pd.DataFrame, 
            figure_path : str = None,
            #neptune_run : str = None, 
            #neptune_key :str = None, 
            **kwargs
        ):

        """Plot the data and calibration model.

        Parameters
        ----------

        dataframe : pandas.DataFrame
            Dataframe containing the input and target columns

        figure_path : str, default=None
            If set, the figure is saved to the given path.

        neptune_run : str, default=None
            If set, the figure is logged to the given neptune run.

        neptune_key : str, default=None
            key under which the figure is logged to the neptune run.

        """

        deviation = self.deviation(dataframe)

        n_input_properties = deviation.shape[1] - 3

        transform_unit = self.get_transform_unit(self.transform_deviation)

        fig, axs = plt.subplots(n_input_properties, 2, figsize=(6.5, 3.5*n_input_properties), squeeze=False)

        for input_property in range(n_input_properties):

            # plot the relative observed deviation
            density_scatter(
                deviation[:, 3+input_property], 
                deviation[:, 0],
                axis=axs[input_property, 0],  
                s=1
            )

            # plot the calibration model 
            x_values = deviation[:, 3+input_property]
            y_values = deviation[:, 1]
            order = np.argsort(x_values)
            x_values = x_values[order]
            y_values = y_values[order]

            axs[input_property, 0].plot(x_values, y_values, color='red')

            # plot the calibrated deviation

            density_scatter(
                deviation[:, 3+input_property],
                deviation[:, 2],
                axis=axs[input_property, 1],
                s=1
            )

            for ax, dim in zip(axs[input_property, :],[0,2]):
                ax.set_xlabel(self.input_columns[input_property])
                ax.set_ylabel(f'observed deviation {transform_unit}')
                
                # get absolute y value and set limites to plus minus absolute y
                y = deviation[:, dim] 
                y_abs = np.abs(y)
                ax.set_ylim(-y_abs.max()*1.05, y_abs.max()*1.05)

        fig.tight_layout()

        # log figure to neptune ai
        #if neptune_run is not None and neptune_key is not None:
        #    neptune_run[f'calibration/{neptune_key}'].log(fig)

        #if figure_path is not None:
            
        #    i = 0
        #    file_name = os.path.join(figure_path, f'calibration_{neptune_key}_{i}.png')
        #    while os.path.exists(file_name):
        #        file_name = os.path.join(figure_path, f'calibration_{neptune_key}_{i}.png')
        #        i += 1

        #    fig.savefig(file_name)
            
        plt.show()  

        plt.close()
        
class CalibrationManager():

    def __init__(
            self,
            config : typing.Union[None, dict] = None,
            path : typing.Union[None, str] = None,
            load_calibration : bool = True):

        """Contains, updates and applies all calibrations for a single run.

        Calibrations are grouped into calibration groups. Each calibration group is applied to a single data structure (precursor dataframe, fragment fataframe, etc.). Each calibration group contains multiple estimators which each calibrate a single property (mz, rt, etc.). Each estimator is a `Calibration` object which contains the estimator function.
        
        Parameters
        ----------

        config : typing.Union[None, dict], default=None
            Calibration config dict. If None, the default config is used.

        path : str, default=None
            Path where the current parameter set is saved to and loaded from.

        load_calibration : bool, default=True
            If True, the calibration manager is loaded from the given path.
        
        """
        self._is_loaded_from_file = False
        self.estimator_groups = []
        self.path = path

        logging.info('========= Initializing Calibration Manager =========')

        self.load_config(config)
        if load_calibration:
            self.load()

        logging.info('====================================================')

    @property
    def is_loaded_from_file(self):
        """Check if the calibration manager was loaded from file.
        """
        return self._is_loaded_from_file
    
    @property
    def is_fitted(self):
        """Check if all estimators in all calibration groups are fitted.
        """

        is_fitted = True
        for group in self.estimator_groups:
            for estimator in group['estimators']:
                if not estimator.is_fitted:
                    is_fitted = False
                    break
        
        return is_fitted and len(self.estimator_groups) > 0

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
                    template = calibration_model_provider.get_model(estimator['model'])
                    model_args = estimator['model_args'] if 'model_args' in estimator else {}
                    estimator['function'] = template(**model_args)
                except Exception as e:
                    logging.error(f'Could not load estimator {estimator["name"]}: {e}')

            group_copy = {'name': group['name']} 
            group_copy['estimators'] = [Calibration(**x) for x in group['estimators']]
            self.estimator_groups.append(group_copy)

    def save(self):
        """Save the calibration manager state to pickle file.
        """
        if self.path is not None:
            with open(self.path, 'wb') as f:
                pickle.dump(self, f)

    def load(self):
        """Load the calibration manager from pickle file.
        """
        if self.path is not None and os.path.exists(self.path):
            try:
                with open(self.path, 'rb') as f:
                    loaded_state = pickle.load(f)
                    self.__dict__.update(loaded_state.__dict__)
                    self._is_loaded_from_file = True
            except:
                logging.warning(f'Could not load calibration manager from {self.path}')
            else:
                logging.info(f'Loaded calibration manager from {self.path}')
        else:
            logging.warning(f'Calibration manager path {self.path} does not exist')

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
        for group in group_idx:
            for estimator in self.estimator_groups[group]['estimators']:
                logging.info(f'calibration group: {group_name}, fitting {estimator.name} estimator ')
                estimator.fit(df, *args, neptune_key=f'{group_name}_{estimator.name}', **kwargs)

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

class CalibrationModelProvider:
    def __init__(self):

        """Provides a collection of scikit-learn compatible models for calibration.       
        """
        self.model_dict = {}

    def __repr__(self) -> str:
        string = '<CalibrationModelProvider, \n[\n'
        for key, value in self.model_dict.items():
            string += f' \t {key}: {value}\n'
        string += ']>'
        return string

    def register_model(
            self, 
            model_name : str, 
            model_template : sklearn.base.BaseEstimator
        ):
        """Register a model template with a given name.

        Parameters
        ----------
        model_name : str
            Name of the model

        model_template : sklearn.base.BaseEstimator
            The model template which must have a fit and predict method.

        """
        self.model_dict[model_name] = model_template

    def get_model(self, model_name : str):
        """Get a model template by name.

        Parameters
        ----------

        model_name : str
            Name of the model

        Returns
        -------
        sklearn.base.BaseEstimator
            The model template which must have a fit and predict method.

        """

        if model_name not in self.model_dict:
            raise ValueError(f'Unknown model {model_name}')
        else:
            return self.model_dict[model_name]

def PolynomialRegression(degree=2, include_bias=False):
    return Pipeline([
        ('poly', PolynomialFeatures(degree=degree, include_bias=include_bias)),
        ('linear', LinearRegression())
    ])

calibration_model_provider = CalibrationModelProvider()
calibration_model_provider.register_model('LinearRegression', LinearRegression)
calibration_model_provider.register_model('LOESSRegression', LOESSRegression)
calibration_model_provider.register_model('PolynomialRegression', PolynomialRegression)