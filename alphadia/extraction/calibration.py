# internal imports
from unittest.mock import DEFAULT
import alphatims.bruker
import alphatims.utils
from matplotlib import pyplot as plt

import sklearn.base

import os
import logging

# external imports
import pandas as pd
import numpy as np
from . import calibration


from sklearn.linear_model import LinearRegression
from scipy.stats import gaussian_kde

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
        self.transform_deviation = transform_deviation

        self.is_fitted = is_fitted

    def validate_columns(
            self, 
            dataframe
        ):

        valid = True

        if len (self.input_columns) != 1:
            logging.error('Warning: Only one input column supported')
            valid = False

        if len (self.target_columns) != 1:
            logging.error('Warning: Only one target column supported')
            valid = False
        required_columns = set(self.input_columns + self.target_columns)
        if not required_columns.issubset(dataframe.columns):
            logging.error(f'Warning: {self.name}, at least one column {required_columns} not found in dataframe')
            valid = False

        return valid

    def fit(self, dataframe, plot=False, report_ci=0.95):
        if not self.validate_columns(dataframe):
            logging.error(f'Warning: {self.name} calibration was skipped')
            return

        if self.function is None:
            raise ValueError('No estimator function provided')

        input_values = dataframe[self.input_columns].values
        target_value = dataframe[self.target_columns].values

        self.function.fit(input_values, target_value)
        self.is_fitted = True

        if plot == True:
            self.plot(dataframe)

        return self.ci(dataframe, float(report_ci))

    def predict(self, dataframe, inplace=False):
        if self.is_fitted == False:
            logging.error(f'Warning: {self.name} prediction was skipped as it hasn not been fitted yet')
            return
        
        if not set(self.input_columns).issubset(dataframe.columns):
            logging.error(f'Warning: {self.name} calibration was skipped as input column {self.input_columns} not found in dataframe')
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


    def plot(self, dataframe):

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
        plt.show()
        pass

       


class RunCalibration():

    estimator_groups = []


    def __init__(self, estimator_groups):
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
                estimator.fit(df, *args, **kwargs)

       
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


class GlobalCalibration():

    # template column names
    precursor_calibration_targets = {
        'precursor_mz':('mz_predicted', 'mz_calibrated'),
        'mobility':('mobility_predicted', 'mobility_observed'),
        'rt':('rt_predicted', 'rt_observed'),
    }

    fragment_calibration_targets = {
        'fragment_mz':('mz_predicted', 'mz_calibrated'),
    }

    def __init__(self, extraction_plan):
        self.prediction_targets = {}
        self.estimator_template = []
        self.extraction_plan = extraction_plan
        

    def __str__(self):

        output = ''
        
        num_run_mappings = len(self.extraction_plan.runs)
        output += f'Calibration for {num_run_mappings} runs: \n'

        for run in self.extraction_plan.runs:
            output += '\t' + run.__str__()

        return output

    def print(self):
        print(self)

    def set_extraction_plan(self, extraction_plan):
        self.extraction_plan = extraction_plan

    def set_estimators(self, estimator_template = {}):
        self.estimator_template=estimator_template

    def fit(self):
        """A calibration is fitted based on the preferred precursors presented by the extraction plan. 
        This is done for all runs found within the calibration df. 
        As the calibration df can change and (should) increase during recalibration, the runs need to be fixed. """
         
        calibration_df = self.extraction_plan.get_calibration_df()

        # contains all source - target coliumn names
        # is created based on self.prediction_target and will look somehwat like this:
        # prediction_target = {
        #    'mz':('precursor_mz', 'mz'),
        #    'mobility':('mobility_pred', 'mobility'),
        #    'rt':('rt_pred', 'rt'),
        # }


        # check what calibratable properties exist
        for property, columns in self.precursor_calibration_targets.items():
            if set(columns).issubset(calibration_df.columns):
                self.prediction_targets[property] = columns

            else:
                logging.info(f'calibrating {property} not possible as required columns are missing' )

        # reset estimators and initialize them based on the estimator template
        self.estimators = []

        for i, run in enumerate(self.extraction_plan.runs):
            new_estimators = {}
            for property in self.prediction_targets.keys():
                new_estimators[property] = sklearn.base.clone(self.estimator_template[property])
            self.estimators.append(new_estimators)

        # load all runs found in the extraction plan
        for run in self.extraction_plan.runs:
            run_index = run['index']
            run_name = run['name']

            calibration_df_run = calibration_df[calibration_df['raw_name'] == run_name]
            num_dp = len(calibration_df_run)
            logging.info(f'Calibrating run {run_index} {run_name} with {num_dp} entries')
            self.fit_run_wise(run_index,run_name , calibration_df_run, self.prediction_targets)

    def fit_run_wise(self, 
                    run_index, 
                    run_name, 
                    calibration_df, 
                    prediction_target):
        
        run_df = calibration_df[calibration_df['raw_name'] == run_name]

        for property, columns in prediction_target.items():

            estimator = self.estimators[run_index][property]

            source_column = columns[0]
            target_column = columns[1]

            source_values = run_df[source_column].values
            target_value = run_df[target_column].values

            estimator.fit(source_values, target_value)

    def predict(self, run, property, values):
        
        # translate run name to index
        if isinstance(run, str):
            run_index = -1

            for run_mapping in self.runs:
                if run_mapping['name'] == run:
                    run_index = run_mapping['index']
            
            if run_index == -1:
                raise ValueError(f'No run found with name {run}')

        else:
            run_index = run

        return self.estimators[run_index][property].predict(values)


    def plot(self, *args, save_name=None, **kwargs):
        logging.info('plotting calibration curves')
        ipp = 4
        calibration_df = self.extraction_plan.get_calibration_df()
    
        ax_labels = {'mz': ('mz','ppm'),
                    'rt': ('rt_pred','RT (seconds)'),
                    'mobility': ('mobility', 'mobility')}
        # check if 

        for run_mapping in self.extraction_plan.runs:

            run_df = calibration_df[calibration_df['raw_name'] == run_mapping['name']]
            print(len(run_df))

            estimators = self.estimators[run_mapping['index']]

            fig, axs = plt.subplots(ncols=len(estimators), nrows=1, figsize=(len(estimators)*ipp,ipp))
            for i, (property, estimator) in enumerate(estimators.items()):
                
                target_column, measured_column = self.prediction_targets[property]
                target_values = run_df[target_column].values
                measured_values = run_df[measured_column].values

                # plotting
                axs[i].set_title(property)

                calibration_space = np.linspace(np.min(target_values),np.max(target_values),1000)
                calibration_curve = estimator.predict(calibration_space)

                if property == 'mz':
                    measured_values = (target_values - measured_values) / target_values * 10**6
                    calibration_curve = (calibration_space - calibration_curve) / calibration_space * 10**6
                #axs[i].scatter(target_values,measured_values)
                density_scatter(target_values,measured_values,axs[i], s=2, **kwargs)
                axs[i].plot(calibration_space,calibration_curve, c='r')
                axs[i].set_xlabel(ax_labels[property][0])
                axs[i].set_ylabel(ax_labels[property][1])

            fig.suptitle(run_mapping['name'])
            fig.tight_layout()

            if save_name is not None:
                loaction = os.path.join(save_name, f"{run_mapping['name']}.png")
                fig.savefig(loaction, dpi=300)
            plt.show()


def density_scatter(x, y, axis, **kwargs):

    # Calculate the point density
    xy = np.vstack([x,y])
    z = gaussian_kde(xy)(xy)

    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]

    axis.scatter(x, y, c=z, **kwargs)
