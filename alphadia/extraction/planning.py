from typing_extensions import Self

from matplotlib.style import library
import alphadia.annotation

import pandas as pd 
import logging
from pathlib import Path


import alphabase.psm_reader
import alphabase.peptide.precursor
import alphabase.peptide.fragment
from alphabase.spectral_library.flat import SpecLibFlat
from alphabase.spectral_library.base import SpecLibBase
from alphabase.spectral_library.reader import SWATHLibraryReader

from alphadia.extraction.data import TimsTOFDIA

import yaml
import os

from . import calibration

import numpy as np
import hashlib

def recursive_update(full_dict, update_dict):
    """recursively update a dict with a second dict
    The dict is updated inplace
    """
    for key, value in update_dict.items():
        if key in full_dict.keys():
            if isinstance(value, dict):
                recursive_update(full_dict[key], update_dict[key])
            else:
                full_dict[key] = value
        else:
            full_dict[key] = value

class Plan:

    def __init__(self, 
                raw_data: list, 
                config_update=None):
        """initialize a dia extraction plan

        Parameters
        ----------
        raw_data : list
            list of input file locations

        config_update : dict or str, optional
            dict or yaml file to update the default config, by default None

        """
        
        # default yaml config location under /misc/config/config.yaml
        yaml_file = os.path.join(os.path.dirname(__file__), '..','..','misc','config','default.yaml')
        with open(yaml_file, 'r') as f:
            self._config = yaml.safe_load(f)

        # config can be updated with a dict or a yaml file
        if isinstance(config_update, dict):
            print(config_update)
            recursive_update(self._config, config_update)

        elif isinstance(config_update, str):
            try:
                with open(config_update, 'r') as f:
                    config_update = yaml.safe_load(f)
                    recursive_update(self._config, config_update)
            except:
                logging.error(f'Could not load config file {config_update}')

        self.raw_data = raw_data

    @property
    def config(self):
        return self._config

    def from_spec_lib_base(self, speclib_base):

        self.speclib = SpecLibFlat()
        self.speclib.parse_base_library(speclib_base)

        self.rename_columns(self.speclib._precursor_df, 'precursor_columns')
        self.rename_columns(self.speclib._fragment_df, 'fragment_columns')


        if 'rt_type' in self.config:
            self.rt_type = self.config['rt_type']
            logging.info(f'forcing rt_type {self.rt_type} from config file')
        else:
            self.rt_type = self.check_rt_type()
            logging.info(f'rt_type automatically determined as {self.rt_type}')


    def load_speclib(self, speclib_path, mode='dense'):
        if mode == 'dense':
            speclib_dense = SpecLibBase()
            speclib_dense.load_hdf(speclib_path, load_mod_seq=True)

            self.speclib = SpecLibFlat()
            self.speclib.parse_base_library(speclib_dense)

        elif mode == 'flat':
            self.speclib = SpecLibFlat()
            self.speclib.load_hdf(speclib_path, load_mod_seq=True)

        elif mode == 'swath':
            speclib_dense = SWATHLibraryReader()
            speclib_dense.import_file(speclib_path)

            if 'decoy' not in speclib_dense.precursor_df.columns:
                logging.info('adding decoys')
                speclib_dense.decoy = 'diann'
                speclib_dense.append_decoy_sequence()
                speclib_dense.calc_precursor_mz()

            self.speclib = SpecLibFlat()
            self.speclib.parse_base_library(speclib_dense)

        self.rename_columns(self.speclib.precursor_df, 'precursor_columns')
        self.rename_columns(self.speclib.fragment_df, 'fragment_columns')

        if 'rt_type' in self.config:
            self.rt_type = self.config['rt_type']
            logging.info(f'forcing rt_type {self.rt_type} from config file')
        else:
            self.rt_type = self.check_rt_type()
            logging.info(f'rt_type automatically determined as {self.rt_type}')

    def check_rt_type(self):
        # check if retention times are in seconds, convert to seconds if necessary
        # possible options: 'seconds', 'minutes', 'norm', 'irt'

        rt_type = 'unknown'

        rt_series = self.speclib.precursor_df['rt_library']

        if rt_series.min() < 0:
            rt_type = 'irt'
        
        elif 0 <= rt_series.min() <= 1:
            rt_type = 'norm'

        elif rt_series.max() < self.config['rt_heuristic']:
            rt_type = 'minutes'

        elif rt_series.max() > self.config['rt_heuristic']:
            rt_type = 'seconds'

        if rt_type == 'unknown':
            logging.warning("""Could not determine retention time typ. 
                            Raw values will be used. 
                            Please specify extraction.rt_type with the possible values ('irt', 'norm, 'minutes', 'seconds',) in the config file.""")

        return rt_type


    def rename_columns(self, precursor_flat, group):
        logging.info(f'renaming {group} columns')
        # precursor columns
        if group in self.config['extraction']:
            for key, value in self.config['extraction'][group].items():
                # column which should be created already exists
                if key in precursor_flat.columns:
                    continue
                # column does not yet exist
                else:
                    for candidate_columns in value:
                        if candidate_columns in precursor_flat.columns:
                            precursor_flat.rename(columns={candidate_columns: key}, inplace=True)
                            # break after first match
                            break
        else:
            logging.error(f'no {group} columns specified in extraction config')

    def get_run_data(self):
        for raw_location in self.raw_data:
            raw = TimsTOFDIA(raw_location)
            raw_name = Path(raw_location).stem

            if self.rt_type == 'seconds' or self.rt_type == 'unknown':
                yield raw, raw_name, self.speclib.precursor_df, self.speclib.fragment_df
            
            elif self.rt_type == 'minutes':
                precursor_df = self.speclib.precursor_df.copy()
                precursor_df['rt_library'] *= 60

                yield raw, raw_name, precursor_df, self.speclib.fragment_df

            elif self.rt_type == 'irt':
                raise NotImplementedError()
            
            elif self.rt_type == 'norm':
                precursor_df = self.speclib.precursor_df.copy()

                # the normalized rt is transformed to extend from the center of the lowest to the center of the highest rt window
                rt_min = self.config['extraction']['initial_rt_tolerance']/2
                rt_max = raw.rt_max_value - (self.config['extraction']['initial_rt_tolerance']/2)
                precursor_df['rt_library'] = precursor_df['rt_library'] * (rt_max - rt_min) + rt_min

                yield raw, raw_name, precursor_df, self.speclib.fragment_df
                

        

"""
class ExtractionPlan():

    def __init__(self, psm_reader_name, decoy_type='diann'):
        self.psm_reader_name = psm_reader_name
        self.runs = []
        self.speclib = alphabase.spectral_library.library_base.SpecLibBase(decoy=decoy_type)

    def set_precursor_df(self, precursor_df):
        self.speclib.precursor_df = precursor_df

        logging.info('Initiate run mapping')

        # init run mapping
        for i, raw_name in enumerate(self.speclib.precursor_df['raw_name'].unique()):
            logging.info(f'run: {i} , name: {raw_name}')
            self.runs.append(
                {
                    "name": raw_name, 
                    'index': i, 
                    'path': os.path.join(self.data_path, f'{raw_name}.d')
                }
            )

        self.process_psms()

    def has_decoys(self):
        if 'decoy' in self.speclib.precursor_df.columns:
            return self.speclib.precursor_df['decoy'].sum() > 0
        else:
            return False

    def process_psms(self):

        # rename columns
        # all columns are expected to be observed values
        self.speclib._precursor_df.rename(
            columns={
                "rt": "rt_observed", 
                "mobility": "mobility_observed",
                "mz": "mz_observed",
                "precursor_mz": "mz_predicted",
                }, inplace=True
        )

        if not self.has_decoys():
            logging.info('no decoys were found, decoys will be generated using alphaPeptDeep')
            self.speclib.append_decoy_sequence()
            self.speclib._precursor_df.drop(['mz_predicted'],axis=1, inplace=True)
            self.speclib._precursor_df = alphabase.peptide.precursor.update_precursor_mz(self.speclib._precursor_df)
            self.speclib._precursor_df.rename(columns={"precursor_mz": "mz_predicted",}, inplace=True )

        model_mgr = peptdeep.pretrained_models.ModelManager()
        model_mgr.nce = 30
        model_mgr.instrument = 'timsTOF'

        # check if retention times are in seconds, convert to seconds if necessary
        RT_HEURISTIC = 180
        if self.speclib._precursor_df['rt_observed'].max() < RT_HEURISTIC:
            logging.info('retention times are most likely in minutes, will be converted to seconds')
            self.speclib._precursor_df['rt_observed'] *= 60

        #if not 'mz_predicted' in self.speclib._precursor_df.columns:
        #    logging.info('precursor mz column not found, column is being generated')
        #    self.speclib._precursor_df = alphabase.peptide.precursor.update_precursor_mz(self.speclib._precursor_df)
            

        if not 'rt_predicted' in self.speclib._precursor_df.columns:
            logging.info('rt prediction not found, column is being generated using alphaPeptDeep')
            self.speclib._precursor_df = model_mgr.predict_all(
                self.speclib._precursor_df,
                predict_items=['rt']
            )['precursor_df']
        
        self.speclib._precursor_df.drop(['rt_norm','rt_norm_pred'],axis=1, inplace=True)
        self.speclib.precursor_df.rename(
            columns={
                "rt_pred": "rt_predicted",
                }, inplace=True
        )
            

        if not 'mobility_pred' in self.speclib._precursor_df.columns:
            logging.info('mobility prediction not found, column is being generated using alphaPeptDeep')
            self.speclib._precursor_df = model_mgr.predict_all(
                self.speclib._precursor_df,
                predict_items=['mobility']
            )['precursor_df']

        self.speclib._precursor_df.drop(['ccs_pred','ccs'],axis=1, inplace=True)
        self.speclib.precursor_df.rename(
            columns={
                "mobility_pred": "mobility_predicted",
                }, inplace=True
        )

        self.speclib._precursor_df.drop(['precursor_mz'],axis=1, inplace=True)

    def get_calibration_df(self):
        # Used by the calibration class to get the first set of precursors used for calibration.
        # Returns a filtered subset of the precursor_df based on metrics like the q-value, target channel etc.
        
        calibration_df = self.speclib.precursor_df.copy()
        calibration_df = calibration_df[calibration_df['fdr'] < 0.01]
        calibration_df = calibration_df[calibration_df['decoy'] == 0]

        return calibration_df

    def validate(self):
        #Validate extraction plan before proceeding
        

        logging.info('Validating extraction plan')

        if not hasattr(self,'precursor_df'):
            logging.error('No precursor_df found')
            return

        if not hasattr(self,'fragment_mz_df'):
            logging.error('No fragment_mz_df found')

        if not hasattr(self,'fragment_intensity_df'):
            logging.error('No fragment_intensity_df found')

        # check if all mandatory columns were found
        mandatory_precursor_df_columns = ['raw_name', 
                            'decoy',
                            'charge',
                            'frag_start_idx',
                            'frag_end_idx',
                            'precursor_mz',
                            'rt_pred',
                            'mobility_pred',
                            'mz_values',
                            'rt_values',
                            'mobility_values',
                            'fdr']

        for item in mandatory_precursor_df_columns:
            if not item in self.precursor_df.columns.to_list():
                logging.error(f'The mandatory column {item} was missing from the precursor_df')

        logging.info('Extraction plan succesfully validated')

    def set_library(self, lib: peptdeep.protein.fasta.FastaLib):
        self.lib = lib

    def set_data_path(self, folder):
        self.data_path = folder

    def set_calibration(self, estimators):
        
        self.calibration = alphadia.extraction.calibration.GlobalCalibration(self)
        self.calibration.set_estimators(estimators)

    def add_normalized_properties(self):

        # initialize normalized properties with zeros
        for property in self.calibration.prediction_targets:
            self.speclib._precursor_df[f'{property}_norm'] = 0

            for i, run in enumerate(self.runs):
                run_mask = self.speclib.precursor_df['raw_name'] == run['name']
                run_speclib = self.speclib.precursor_df[run_mask]
                
                # predicted value like rt_pred or mobility_pred
                source_column = self.calibration.prediction_targets[property][0]
                # measured value like rt or mobility
                target_column = self.calibration.prediction_targets[property][1]

                target = run_speclib[target_column].values
                source = run_speclib[source_column].values
                source_calibrated = self.calibration.predict(i, property, source)
                target_deviation = target / source_calibrated

                self.speclib._precursor_df.loc[run_mask, f'{property}_norm'] = target_deviation

            # make sure there are no zero values
            zero_vals = np.sum(self.speclib._precursor_df[f'{property}_norm'] == 0)
            if zero_vals > 0:
                logging.warning(f'normalisied property {property} has not been set for {zero_vals} entries')

        for run in self.runs:
            run_speclib = self.speclib.precursor_df[self.speclib.precursor_df['raw_name'] == run['name']]

            pass

    def build_run_precursor_df(self, run_index):
        
        #build run specific speclib which combines entries from other runs
        

        self.speclib.hash_precursor_df()

        # IDs from the own run are already calibrated
        run_name = self.runs[run_index]['name']
        run_precursor_df = self.speclib.precursor_df[self.speclib.precursor_df['raw_name'] == run_name].copy()
        run_precursor_df['same_run'] = 1
        existing_precursors = run_precursor_df['mod_seq_charge_hash'].values

        # assemble IDs from other runs
        other_speclib = self.speclib.precursor_df[self.speclib.precursor_df['raw_name'] != run_name]
        other_speclib = other_speclib[~other_speclib['mod_seq_charge_hash'].isin(existing_precursors)]

        # TODO sloooooow, needs to be optimized
        extra_precursors = []
        grouped = other_speclib.groupby('mod_seq_charge_hash')
        for name, group in grouped:
            group_dict = group.to_dict('records')

            out_dict = group_dict[0]
            for property in self.calibration.prediction_targets:
                out_dict[f'{property}_norm'] = group[f'{property}_norm'].median()

            extra_precursors.append(out_dict)

        nonrun_precursor_df = pd.DataFrame(extra_precursors)
        nonrun_precursor_df['same_run'] = 0
        new_precursor_df = pd.concat([run_precursor_df, nonrun_precursor_df]).reset_index(drop=True)

        # apply run specific calibration function
        for property, columns in self.calibration.prediction_targets.items():

            source_column = columns[0]
            target_column = columns[1]
            
            new_precursor_df[target_column] = self.calibration.predict(run_index,property,new_precursor_df[source_column].values)*new_precursor_df[f'{property}_norm']

        # flatten out the mz_values and intensity_values
            
        # flatten precursor
        precursors_flat, fragments_flat = alphabase.peptide.fragment.flatten_fragments(
            new_precursor_df,
            self.speclib.fragment_mz_df,
            self.speclib.fragment_intensity_df,
            intensity_treshold = 0
        )

        fragments_flat.rename(
            columns={
                "mz": "mz_predicted",
                }, inplace=True
        )

        if 'precursor_mz' in self.calibration.estimators[run_index].keys():
            logging.info('Performing precursor_mz calibration')
            source_column, target_column = self.calibration.precursor_calibration_targets['precursor_mz']
            precursors_flat[target_column] = self.calibration.predict(run_index, 'precursor_mz', precursors_flat[source_column].values)    
        else:
            logging.info('No precursor_mz calibration found, using predicted values')

        if 'fragment_mz' in self.calibration.estimators[run_index].keys():
            logging.info('Performing fragment_mz calibration')
            source_column, target_column = self.calibration.fragment_calibration_targets['fragment_mz']
            fragments_flat[target_column] = self.calibration.predict(run_index, 'fragment_mz', fragments_flat[source_column].values)    
        else:
            logging.info('No fragment_mz calibration found, using predicted values')

        return precursors_flat, fragments_flat


class LibraryManager():

    def __init__(self, decoy_type='diann'):
        self.runs = []
        self.speclib = alphabase.spectral_library.library_base.SpecLibBase(decoy=decoy_type)

    def set_precursor_df(self, precursor_df):
        self.speclib.precursor_df = precursor_df

        logging.info('Initiate run mapping')

        # init run mapping
        for i, raw_name in enumerate(self.speclib.precursor_df['raw_name'].unique()):
            logging.info(f'run: {i} , name: {raw_name}')
            self.runs.append(
                {
                    "name": raw_name, 
                    'index': i, 
                    'path': os.path.join(self.data_path, f'{raw_name}.d')
                }
            )

        self.process_psms()

    def has_decoys(self):
        if 'decoy' in self.speclib.precursor_df.columns:
            return self.speclib.precursor_df['decoy'].sum() > 0
        else:
            return False

    def process_psms(self):

        # rename columns
        # all columns are expected to be observed values
        self.speclib._precursor_df.rename(
            columns={
                "rt": "rt_library", 
                "mobility": "mobility_library",
                "mz": "mz_library",
                "precursor_mz": "mz_library",
                }, inplace=True
        )

        if not self.has_decoys():
            logging.info('no decoys were found, decoys will be generated using alphaPeptDeep')
            self.speclib.append_decoy_sequence()
            self.speclib._precursor_df.drop(['mz_library'],axis=1, inplace=True)
            self.speclib._precursor_df = alphabase.peptide.precursor.update_precursor_mz(self.speclib._precursor_df)
            self.speclib._precursor_df.rename(columns={"precursor_mz": "mz_library",}, inplace=True )

        # check if retention times are in seconds, convert to seconds if necessary
        RT_HEURISTIC = 180
        if self.speclib._precursor_df['rt_library'].max() < RT_HEURISTIC:
            logging.info('retention times are most likely in minutes, will be converted to seconds')
            self.speclib._precursor_df['rt_library'] *= 60

        #if not 'mz_predicted' in self.speclib._precursor_df.columns:
        #    logging.info('precursor mz column not found, column is being generated')
        #    self.speclib._precursor_df = alphabase.peptide.precursor.update_precursor_mz(self.speclib._precursor_df)
        if 'precursor_mz' in self.speclib._precursor_df.columns:
            self.speclib._precursor_df.drop(['precursor_mz'],axis=1, inplace=True)


    def set_library(self, lib: peptdeep.protein.fasta.FastaLib):
        self.lib = lib

    def set_data_path(self, folder):
        self.data_path = folder

    def build_run_precursor_df(self, run_index):
        
        # build run specific speclib which combines entries from other runs
        

        self.speclib.hash_precursor_df()

        # IDs from the own run are already calibrated
        run_name = self.runs[run_index]['name']
        run_precursor_df = self.speclib.precursor_df[self.speclib.precursor_df['raw_name'] == run_name].copy()
             
        # flatten precursor
        precursors_flat, fragments_flat = alphabase.peptide.fragment.flatten_fragments(
            run_precursor_df,
            self.speclib.fragment_mz_df,
            self.speclib.fragment_intensity_df,
            intensity_treshold = 0
        )

        fragments_flat.rename(
            columns={
                "mz": "mz_library"
                }, inplace=True
        )


        return precursors_flat, fragments_flat
"""