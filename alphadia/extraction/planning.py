# native imports
from typing_extensions import Self
import logging
import socket
from pathlib import Path
import yaml
import os 
from datetime import datetime
import hashlib
from typing import Union, List, Dict, Tuple, Optional

logger = logging.getLogger()
from alphadia.extraction import processlogger
    

# alphadia imports
from alphadia.extraction import data, validate, utils
from alphadia.extraction.workflow import peptidecentric, base

import alphadia

# alpha family imports
import alphatims

from alphabase.peptide import fragment
from alphabase.spectral_library.flat import SpecLibFlat
from alphabase.spectral_library.base import SpecLibBase

# third party imports
import numpy as np
import pandas as pd 
import neptune.new as neptune
from neptune.new.types import File
import os, psutil

class Plan:

    def __init__(self, 
            output_folder : str,
            raw_file_list: List,
            spectral_library : SpecLibBase,
            config_path : Union[str, None] = None,
            config_update_path : Union[str, None] = None,
            config_update : Union[Dict, None] = None
        ) -> None:
        """Highest level class to plan a DIA Search. 
        Owns the input file list, speclib and the config.
        Performs required manipulation of the spectral library like transforming RT scales and adding columns.

        Parameters
        ----------
        raw_data : list
            list of input file locations

        config_path : str, optional
            yaml file containing the default config.

        config_update_path : str, optional
           yaml file to update the default config.

        config_update : dict, optional
            dict to update the default config. Can be used for debugging purposes etc.

        """
        self.output_folder = output_folder
        processlogger.init_logging(self.output_folder)
        logger = logging.getLogger()

        logger.progress('      _   _      _         ___ ___   _   ')
        logger.progress('     /_\ | |_ __| |_  __ _|   \_ _| /_\  ')
        logger.progress('    / _ \| | \'_ \\ \' \/ _` | |) | | / _ \ ')
        logger.progress('   /_/ \_\_| .__/_||_\__,_|___/___/_/ \_\\')
        logger.progress('           |_|                            ')
        logger.progress('')

        self.raw_file_list = raw_file_list

        # default config path is not defined in the function definition to account for for different path separators on different OS
        if config_path is None:
            # default yaml config location under /misc/config/config.yaml
            config_path = os.path.join(os.path.dirname(__file__), '..','..','misc','config','default.yaml')

        # 1. load default config
        with open(config_path, 'r') as f:
            logger.info(f'loading default config from {config_path}')
            self.config = yaml.safe_load(f)

        # 2. load update config from yaml file
        if config_update_path is not None:
            logger.info(f'loading config update from {config_update_path}')
            with open(config_update_path, 'r') as f:
                config_update_fromyaml = yaml.safe_load(f)
            utils.recursive_update(self.config, config_update_fromyaml)

        # 3. load update config from dict
        if config_update is not None:
            logger.info(f'Applying config update from dict')
            utils.recursive_update(self.config, config_update)

        if not 'output' in self.config:
            self.config['output'] = output_folder

        logger.progress(f'version: {alphadia.__version__}')
        # print hostname, date with day format and time
        logger.progress(f'hostname: {socket.gethostname()}')
        now = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
        logger.progress(f'date: {now}')
        
        self.from_spec_lib_base(spectral_library)

    @property
    def raw_file_list(
            self
        ) -> List[str]:
        """List of input files locations.
        """
        return self._raw_file_list
    
    @raw_file_list.setter
    def raw_file_list(
            self, 
            raw_file_list : List[str]
        ):
        self._raw_file_list = raw_file_list

    @property
    def config(
            self
        ) -> dict:
        """Dict with all configuration parameters for the extraction.
        """
        return self._config
    
    @config.setter
    def config(
            self, 
            config : dict
        ) -> None:
        self._config = config

    @property
    def spectral_library(
            self
        ) -> SpecLibFlat:
        """Flattened Spectral Library."""
        return self._spectral_library
    
    @spectral_library.setter
    def spectral_library(
            self,
            spectral_library : SpecLibFlat
        ) -> None:
        self._spectral_library = spectral_library

    def from_spec_lib_base(self, speclib_base):

        speclib_base._fragment_cardinality_df = fragment.calc_fragment_cardinality(speclib_base.precursor_df, speclib_base._fragment_mz_df)

        speclib = SpecLibFlat(min_fragment_intensity=0.0001, keep_top_k_fragments=100)
        speclib.parse_base_library(speclib_base, custom_df={'cardinality':speclib_base._fragment_cardinality_df})

        self.from_spec_lib_flat(speclib)

    def from_spec_lib_flat(self, speclib_flat):

        self.spectral_library = speclib_flat

        self.rename_columns(self.spectral_library._precursor_df, 'precursor_columns')
        self.rename_columns(self.spectral_library._fragment_df, 'fragment_columns')

        self.log_library_stats()

        self.add_precursor_columns(self.spectral_library.precursor_df)

        output_columns = [
            'nAA',
            'elution_group_idx',
            'precursor_idx',
            'decoy' ,
            'flat_frag_start_idx',
            'flat_frag_stop_idx' ,
            'charge',
            'rt_library',
            'mobility_library',
            'mz_library',
            'sequence',
            'genes',
            'proteins',
            'uniprot_ids',
            'channel'
        ]
        
        existing_columns = self.spectral_library.precursor_df.columns
        output_columns += [f'i_{i}' for i in utils.get_isotope_columns(existing_columns)]
        existing_output_columns = [c for c in output_columns if c in existing_columns]

        self.spectral_library.precursor_df = self.spectral_library.precursor_df[existing_output_columns].copy()
        self.spectral_library.precursor_df = self.spectral_library.precursor_df.sort_values('elution_group_idx')
        self.spectral_library.precursor_df = self.spectral_library.precursor_df.reset_index(drop=True)

    def log_library_stats(self):

        logger.info(f'========= Library Stats =========')
        logger.info(f'Number of precursors: {len(self.spectral_library.precursor_df):,}')

        if 'decoy' in self.spectral_library.precursor_df.columns:
            n_targets = len(self.spectral_library.precursor_df.query('decoy == False'))
            n_decoys = len(self.spectral_library.precursor_df.query('decoy == True'))
            logger.info(f'\tthereof targets:{n_targets:,}')
            logger.info(f'\tthereof decoys: {n_decoys:,}')
        else:
            logger.warning(f'no decoy column was found')

        if 'elution_group_idx' in self.spectral_library.precursor_df.columns:
            n_elution_groups = len(self.spectral_library.precursor_df['elution_group_idx'].unique())
            average_precursors_per_group = len(self.spectral_library.precursor_df)/n_elution_groups
            logger.info(f'Number of elution groups: {n_elution_groups:,}')
            logger.info(f'\taverage size: {average_precursors_per_group:.2f}')

        else:
            logger.warning(f'no elution_group_idx column was found')

        if 'proteins' in self.spectral_library.precursor_df.columns:
            n_proteins = len(self.spectral_library.precursor_df['proteins'].unique())
            logger.info(f'Number of proteins: {n_proteins:,}')
        else:
            logger.warning(f'no proteins column was found')

        if 'channel' in self.spectral_library.precursor_df.columns:
            channels = self.spectral_library.precursor_df['channel'].unique()
            n_channels = len(channels)
            logger.info(f'Number of channels: {n_channels:,} ({channels})')

        else:
            logger.warning(f'no channel column was found, will assume only one channel')

        isotopes = utils.get_isotope_columns(self.spectral_library.precursor_df.columns)

        if len(isotopes) > 0:
            logger.info(f'Isotopes Distribution for {len(isotopes)} isotopes')

        logger.info(f'=================================')    

    def rename_columns(self, dataframe, group):
        logger.info(f'renaming {group} columns')
        # precursor columns
        if group in self.config['library_parsing']:
            for key, value in self.config['library_parsing'][group].items():
                # column which should be created already exists
                if key in dataframe.columns:
                    continue
                # column does not yet exist
                else:
                    for candidate_columns in value:
                        if candidate_columns in dataframe.columns:
                            dataframe.rename(columns={candidate_columns: key}, inplace=True)
                            # break after first match
                            break
        else:
            logger.error(f'no {group} columns specified in extraction config')

    def add_precursor_columns(self, dataframe):

        if not 'precursor_idx' in dataframe.columns:
            dataframe['precursor_idx'] = np.arange(len(dataframe))
            logger.warning(f'no precursor_idx column found, creating one')

        if not 'elution_group_idx' in dataframe.columns:
            dataframe['elution_group_idx'] = self.get_elution_group_idx(dataframe, strategy='precursor')
            logger.warning(f'no elution_group_idx column found, creating one')

        if not 'channel' in dataframe.columns:
            dataframe['channel'] = 0
            logger.warning(f'no channel column found, creating one')

    def get_elution_group_idx(self, dataframe, strategy='precursor'):

        if strategy == 'precursor':
            return dataframe['precursor_idx']

        else:
            raise NotImplementedError(f'elution group strategy {strategy} not implemented')

    def get_run_data(self):
        """Generator for raw data and spectral library."""

        if self.spectral_library is None:
            raise ValueError('no spectral library loaded')

        # iterate over raw files and yield raw data and spectral library
        for raw_location in self.raw_file_list:

            raw_name = Path(raw_location).stem
            yield raw_name, raw_location, self.spectral_library
                
    def run(self, 
            figure_path = None,
            neptune_token = None, 
            neptune_tags = [],
            keep_decoys = False,
            fdr = 0.01,
            ):

        for raw_name, dia_path, speclib in self.get_run_data():
            try:
                workflow = peptidecentric.PeptideCentricWorkflow(
                    raw_name,
                    self.config,
                    dia_path,
                    speclib
                )
   
                workflow.calibration()
                df = workflow.extraction()
                df = df[df['qval'] <= fdr]               

                if self.config['multiplexing']['multiplexed_quant']:
                    df = workflow.requantify(df)

                df['run'] = raw_name

                df.to_csv(os.path.join(workflow.path, 'psm.tsv'), sep='\t', index=False)
                del workflow
            
            except Exception as e:
                logger.exception(e)
                continue

        self.build_output()

    def build_output(self):

        output_path = self.config['output']
        temp_path = os.path.join(output_path, base.TEMP_FOLDER)

        psm_df = []
        stat_df = []

        for raw_name, dia_path, speclib in self.get_run_data():
            run_path = os.path.join(temp_path, raw_name)
            run_df = pd.read_csv(os.path.join(run_path, 'psm.tsv'), sep='\t')
            
            psm_df.append(run_df)
            stat_df.append(build_stat_df(run_df))

        psm_df = pd.concat(psm_df)
        stat_df = pd.concat(stat_df)

        psm_df.to_csv(os.path.join(output_path, 'psm.tsv'), sep='\t', index=False)
        stat_df.to_csv(os.path.join(output_path, 'stat.tsv'), sep='\t', index=False)


def build_stat_df(run_df):

    run_stat_df = []
    for name, group in run_df.groupby('channel'):
        run_stat_df.append({
            'run': run_df['run'].iloc[0],
            'channel': name,
            'precursors': np.sum(group['qval'] <= 0.01),
        })
    
    return pd.DataFrame(run_stat_df)
            