# internal imports
from unittest.mock import DEFAULT
import alphatims.bruker
import alphatims.utils
from matplotlib import pyplot as plt

import sklearn.base


import logging

# external imports
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from scipy.stats import gaussian_kde

class GlobalCalibration():

    # template column names
    possible_prediction_targets = {
        'mz':('precursor_mz', 'mz'),
        'mobility':('mobility_pred', 'mobility'),
        'rt':('rt_pred', 'rt'),
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
        for property, columns in self.possible_prediction_targets.items():
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

        print(self.estimators)

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


    def plot(self):
        logging.info('plotting calibration curves')
        ipp = 4
        calibration_df = self.extraction_plan.get_calibration_df()
    
        ax_labels = {'mz': ('mz','ppm'),
                    'rt': ('rt_pred','RT (seconds)'),
                    'mobility': ('mobility', 'mobility')}
        # check if 

        for run_mapping in self.extraction_plan.runs:

            estimators = self.estimators[run_mapping['index']]

            fig, axs = plt.subplots(ncols=len(estimators), nrows=1, figsize=(len(estimators)*ipp,ipp))
            for i, (property, estimator) in enumerate(estimators.items()):
                
                target_column, measured_column = self.prediction_targets[property]
                target_values = calibration_df[target_column].values
                measured_values = calibration_df[measured_column].values

                # plotting
                axs[i].set_title(property)

                calibration_space = np.linspace(np.min(target_values),np.max(target_values),1000)
                calibration_curve = estimator.predict(calibration_space)

                if property == 'mz':
                    measured_values = (target_values - measured_values) / target_values * 10**6
                    calibration_curve = (calibration_space - calibration_curve) / calibration_space * 10**6

                density_scatter(target_values,measured_values,axs[i], s=2)
                axs[i].plot(calibration_space,calibration_curve, c='r')
                axs[i].set_xlabel(ax_labels[property][0])
                axs[i].set_ylabel(ax_labels[property][1])

            fig.suptitle(run_mapping['name'])
            fig.tight_layout()
            plt.show()


def density_scatter(x, y, axis, **kwargs):

    # Calculate the point density
    xy = np.vstack([x,y])
    z = gaussian_kde(xy)(xy)

    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]

    axis.scatter(x, y, c=z, **kwargs)
