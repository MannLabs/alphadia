# native imports
import os
import logging
from unittest.mock import DEFAULT
import yaml 

# alphadia imports
from alphadia.extraction.utils import density_scatter

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


class Calibration():
    def __init__(self, 
                name=None ,
                function  = None,
                input_columns=[],
                target_columns=[],
                output_columns=[],
                transform_deviation = None,
                is_fitted = False,
                **kwargs):
        
        self.name = name
        self.function = function
        self.input_columns = input_columns
        self.target_columns = target_columns
        self.output_columns = output_columns

        try:    
            self.transform_deviation = float(transform_deviation)
        except:
            self.transform_deviation = None

        self.is_fitted = is_fitted

    def validate_columns(
            self, 
            dataframe
        ):

        valid = True

        if len (self.input_columns) != 1:
            logging.warning('Only one input column supported')
            valid = False

        if len (self.target_columns) != 1:
            logging.warning('Only one target column supported')
            valid = False
        required_columns = set(self.input_columns + self.target_columns)
        if not required_columns.issubset(dataframe.columns):
            logging.warning(f'{self.name}, at least one column {required_columns} not found in dataframe')
            valid = False

        return valid

    def fit(
            self, 
            dataframe,
            plot=False, 
            report_ci=0.95, 
            **kwargs
        ):
        if not self.validate_columns(dataframe):
            logging.warning(f'{self.name} calibration was skipped')
            return

        if self.function is None:
            raise ValueError('No estimator function provided')

        input_values = dataframe[self.input_columns].values
        target_value = dataframe[self.target_columns].values

        self.function.fit(input_values, target_value)
        self.is_fitted = True

        if plot == True:
            self.plot(dataframe, **kwargs)

        return self.ci(dataframe, float(report_ci))

    def predict(self, dataframe, inplace=False):
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


    def deviation(self, dataframe):
        """ Calculate the deviation between the predicted and the target values

        Parameters
        ----------
        dataframe : pandas.DataFrame
            Dataframe containing the input and target columns

        Returns 
        -------
        np.ndarray 
            Array of shape (n_input_columns, 6, n_samples). 
            The second dimension corresponds to the input_values, target_values, calibrated_values, observed_deviation, calibrated_deviation, residual_deviation
        
        """

        input_values = dataframe[self.input_columns].values
        target_value = dataframe[self.target_columns].values
        calibrated_values = self.predict(dataframe, inplace=False)

        deviation_list = []

        for dimension in range(input_values.shape[1]):
            input_dim = input_values[:, dimension]
            target_dim = target_value[:, dimension]
            calibrated_dim = calibrated_values

            order = np.argsort(input_dim)

            input_dim = input_dim[order]
            target_dim = target_dim[order]
            calibrated_dim = calibrated_dim[order]

            # by default the observed deviation of the (measured) target value from the input value is expressed in absolute units
            # if transform_deviation is set to a valid float like 10e6, the deviation is expressed as a fraction of the input value
            observed_deviation = target_dim - input_dim
            if self.transform_deviation is not None:
                observed_deviation = observed_deviation/input_dim * float(self.transform_deviation)

            # calibrated deviation is the part of the deviation that is explained by the calibration
            calibrated_deviation = calibrated_dim - input_dim
            if self.transform_deviation is not None:
                calibrated_deviation = calibrated_deviation/input_dim * float(self.transform_deviation)

            # residual deviation is the part of the deviation that is not explained by the calibration
            residual_deviation = observed_deviation - calibrated_deviation

            deviation_list.append(np.stack((input_dim, target_dim, calibrated_dim, observed_deviation, calibrated_deviation, residual_deviation), axis=0))
        return np.stack(deviation_list, axis=0)

    def ci(self, dataframe, ci, for_plotting=False):
        
        if not 0 < ci < 1:
            raise ValueError('Confidence interval must be between 0 and 1')

        ci_percentile = [100*(1-ci)/2, 100*(1+ci)/2]
        
        deviation = self.deviation(dataframe)
        residual_deviation = deviation[:, 5, :]
        return np.mean(np.abs(np.percentile(residual_deviation, ci_percentile, axis=1)), axis=0)

    def get_transform_unit(self):
        if self.transform_deviation is not None:
            if np.isclose(self.transform_deviation,1e6):
                return '(ppm)'
            elif np.isclose(self.transform_deviation,1e2):
                return '(%)'
            else:
                return f'({self.transform_deviation})'
        else:
            return '(absolute)'


    def plot(
            self, 
            dataframe, 
            figure_path = None,
            neptune_run = None, 
            neptune_key = None, 
            **kwargs
        ):

        deviation = self.deviation(dataframe)

        n_input_properties = deviation.shape[0]

        fig, axs = plt.subplots(n_input_properties, 2, figsize=(6.5, 3.5*n_input_properties), squeeze=False)

        for input_property in range(n_input_properties):

            # plot the relative observed deviation
            density_scatter(
                deviation[input_property, 0, :], 
                deviation[input_property, 3, :],
                axis=axs[input_property, 0],  
                s=1
            )

            # plot the calibration model
            axs[input_property, 0].plot(
                deviation[input_property, 0, :],
                deviation[input_property, 4, :],
                color='red'
            )

            # plot the calibrated deviation

            density_scatter(
                deviation[input_property, 0, :],
                deviation[input_property, 5, :],
                axis=axs[input_property, 1],
                s=1
            )

            for ax, dim in zip(axs[input_property, :],[3,5]):
                ax.set_xlabel(self.input_columns[input_property])
                ax.set_ylabel(f'observed deviation {self.get_transform_unit()}')
                
                # get absolute y value and set limites to plus minus absolute y
                y = deviation[input_property, dim, :] 
                y_abs = np.abs(y)
                ax.set_ylim(-y_abs.max()*1.05, y_abs.max()*1.05)

        fig.tight_layout()

        # log figure to neptune ai
        if neptune_run is not None and neptune_key is not None:
            neptune_run[f'calibration/{neptune_key}'].log(fig)

        if figure_path is not None:
            
            i = 0
            file_name = os.path.join(figure_path, f'calibration_{neptune_key}_{i}.png')
            while os.path.exists(file_name):
                file_name = os.path.join(figure_path, f'calibration_{neptune_key}_{i}.png')
                i += 1

            fig.savefig(file_name)
            
        else:
            plt.show()  

        plt.close()

        

class RunCalibration():


    def __init__(self):
        self.estimator_groups = []

    def load_groups(self, estimator_groups):
        for group in estimator_groups:
            group['estimators'] = [Calibration(**x) for x in group['estimators']]
            
        self.estimator_groups = estimator_groups

    def get_group(self, group_name):
        for group in self.estimator_groups:
            if group['name'] == group_name:
                return group

        logging.error(f'could not get_group: {group_name}')
        return None

    def get_estimator(self, group_name, estimator_name):
        group = self.get_group(group_name)
        if group is not None:
            for estimator in group['estimators']:
                if estimator.name == estimator_name:
                    return estimator
        logging.error(f'could not get_estimator: {group_name}, {estimator_name}')
        return None

    def fit(self, df, group_name, *args, **kwargs):

        if len(self.estimator_groups) == 0:
            raise ValueError('No estimators defined')

        group_idx = [i for i, x in enumerate(self.estimator_groups) if x['name'] == group_name]
        if len(group_idx) == 0:
            raise ValueError(f'No group named {group_name} found')
        for group in group_idx:
            for estimator in self.estimator_groups[group]['estimators']:
                logging.info(f'calibration group: {group_name}, fitting {estimator.name} estimator ')
                estimator.fit(df, *args, neptune_key=f'{group_name}_{estimator.name}', **kwargs)

       
    def predict(self, df, group_name, *args, **kwargs):
        if len(self.estimator_groups) == 0:
            raise ValueError('No estimators defined')

        group_idx = [i for i, x in enumerate(self.estimator_groups) if x['name'] == group_name]
        if len(group_idx) == 0:
            raise ValueError(f'No group named {group_name} found')
        for group in group_idx:
            for estimator in self.estimator_groups[group]['estimators']:
                logging.info(f'calibration group: {group_name}, predicting {estimator.name}')
                estimator.predict(df, inplace=True, *args, **kwargs)

    def load_config(self, config):
        """Load calibration config from yaml file.
        each calibration config is a list of calibration groups which consist of multiple estimators.
        For each estimator the `model` and `model_args` are used to request a model from the calibration_model_provider and to initialize it.
        The estimator is then initialized with the `Calibration` class and added to the group.
        """

        logging.info(f'found {len(config["calibration"])} calibration groups')
        for group in config["calibration"]:
            logging.info(f'({group["name"]}) found {len(group["estimators"])} estimator(s)')
            for estimator in group['estimators']:
                try:
                    template = calibration_model_provider.get_model(estimator['model'])
                    estimator['function'] = template(**estimator['model_args'])
                except Exception as e:
                    logging.error(f'Could not load estimator {estimator["name"]}: {e}')

            group_copy = {'name': group['name']} 
            group_copy['estimators'] = [Calibration(**x) for x in group['estimators']]
            self.estimator_groups.append(group_copy)

        
class CalibrationModelProvider:

    def __init__(self):
        self.model_dict = {}

    def __repr__(self) -> str:
        return str(self.model_dict)

    def register_model(self, model_name, model_template):
        self.model_dict[model_name] = model_template

    def get_model(self, model_name):
        if model_name not in self.model_dict:
            raise ValueError(f'Unknown model {model_name}')
        else:
            return self.model_dict[model_name]

calibration_model_provider = CalibrationModelProvider()
calibration_model_provider.register_model('LinearRegression', LinearRegression)
calibration_model_provider.register_model('LOESSRegression', LOESSRegression)