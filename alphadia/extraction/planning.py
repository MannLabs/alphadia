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
from alphadia.extraction import data, plexscoring
from alphadia.extraction.calibration import CalibrationManager
from alphadia.extraction.scoring import fdr_correction, channel_fdr_correction
from alphadia.extraction import utils, validate
from alphadia.extraction.hybridselection import HybridCandidateSelection, HybridCandidateConfig
import alphadia

# alpha family imports
import alphatims

from alphabase.peptide import fragment
from alphabase.spectral_library.flat import SpecLibFlat

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

        logger.progress(f'version: {alphadia.__version__}')
        # print hostname, date with day format and time
        logger.progress(f'hostname: {socket.gethostname()}')
        now = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
        logger.progress(f'date: {now}')
        

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
    def speclib(
            self
        ) -> SpecLibFlat:
        """Flattened Spectral Library."""
        return self._speclib
    
    @speclib.setter
    def speclib(
            self,
            speclib : SpecLibFlat
        ) -> None:
        self._speclib = speclib

    
    def norm_to_rt(
            self,
            dia_data : alphatims.bruker.TimsTOF, 
            norm_values : np.ndarray, 
            active_gradient_start : Union[float,None] = None, 
            active_gradient_stop : Union[float,None] = None,
            mode = None
        ):
        """Convert normalized retention time values to absolute retention time values.

        Parameters
        ----------
        dia_data : alphatims.bruker.TimsTOF
            TimsTOF object containing the DIA data.

        norm_values : np.ndarray
            Array of normalized retention time values.

        active_gradient_start : float, optional
            Start of the active gradient in seconds, by default None. 
            If None, the value from the config is used. 
            If not defined in the config, it is set to zero.

        active_gradient_stop : float, optional
            End of the active gradient in seconds, by default None.
            If None, the value from the config is used.
            If not defined in the config, it is set to the last retention time value.

        mode : str, optional
            Mode of the gradient, by default None.
            If None, the value from the config is used which should be 'tic' by default

        """

        # retrive the converted absolute intensities
        data = dia_data.frames.query('MsMsType == 0')[[
            'Time', 'SummedIntensities']
        ]
        time = data['Time'].values
        intensity = data['SummedIntensities'].values

        # determine if the gradient start and stop are defined in the config
        if active_gradient_start is None:
            if 'active_gradient_start' in self.config['calibration']:
                lower_rt = self.config['calibration']['active_gradient_start']
            else:
                lower_rt = time[0] + self.config['extraction_initial']['initial_rt_tolerance']/2
        else:
            lower_rt = active_gradient_start

        if active_gradient_stop is None:
            if 'active_gradient_stop' in self.config['calibration']:
                upper_rt = self.config['calibration']['active_gradient_stop']
            else:
                upper_rt = time[-1] - (self.config['extraction_initial']['initial_rt_tolerance']/2)
        else:
            upper_rt = active_gradient_stop

        # make sure values are really norm values
        norm_values = np.interp(norm_values, [norm_values.min(),norm_values.max()], [0,1])

        # determine the mode based on the config or the function parameter
        if mode is None:
            mode = self.config['calibration']['norm_rt_mode'] if 'norm_rt_mode' in self.config['calibration'] else 'tic'
        else:
            mode = mode.lower()

        if mode == 'linear':
            return np.interp(norm_values, [0,1], [lower_rt,upper_rt])
            
        elif mode == 'tic':
            # get lower and upper rt slice
            lower_idx = np.searchsorted(time, lower_rt)
            upper_idx = np.searchsorted(time, upper_rt, side='right')
            time = time[lower_idx:upper_idx]
            intensity = intensity[lower_idx:upper_idx]
            cum_intensity = np.cumsum(intensity)/np.sum(intensity)
            return np.interp(norm_values, cum_intensity, time)

        else:
            raise ValueError(f'Unknown norm_rt_mode {mode}')

    def from_spec_lib_base(self, speclib_base):

        speclib_base._fragment_cardinality_df = fragment.calc_fragment_cardinality(speclib_base.precursor_df, speclib_base._fragment_mz_df)

        speclib = SpecLibFlat(min_fragment_intensity=0.0001, keep_top_k_fragments=100)
        speclib.parse_base_library(speclib_base, custom_df={'cardinality':speclib_base._fragment_cardinality_df})

        self.from_spec_lib_flat(speclib)

    def from_spec_lib_flat(self, speclib_flat):

        self.speclib = speclib_flat

        self.rename_columns(self.speclib._precursor_df, 'precursor_columns')
        self.rename_columns(self.speclib._fragment_df, 'fragment_columns')

        self.log_library_stats()

        self.add_precursor_columns(self.speclib.precursor_df)

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
        
        existing_columns = self.speclib.precursor_df.columns
        output_columns += [f'i_{i}' for i in utils.get_isotope_columns(existing_columns)]
        existing_output_columns = [c for c in output_columns if c in existing_columns]

        self.speclib.precursor_df = self.speclib.precursor_df[existing_output_columns]
        self.speclib.precursor_df = self.speclib.precursor_df.sort_values('elution_group_idx')
        self.speclib.precursor_df = self.speclib.precursor_df.reset_index(drop=True)

    def log_library_stats(self):

        logger.info(f'========= Library Stats =========')
        logger.info(f'Number of precursors: {len(self.speclib.precursor_df):,}')

        if 'decoy' in self.speclib.precursor_df.columns:
            n_targets = len(self.speclib.precursor_df.query('decoy == False'))
            n_decoys = len(self.speclib.precursor_df.query('decoy == True'))
            logger.info(f'\tthereof targets:{n_targets:,}')
            logger.info(f'\tthereof decoys: {n_decoys:,}')
        else:
            logger.warning(f'no decoy column was found')

        if 'elution_group_idx' in self.speclib.precursor_df.columns:
            n_elution_groups = len(self.speclib.precursor_df['elution_group_idx'].unique())
            average_precursors_per_group = len(self.speclib.precursor_df)/n_elution_groups
            logger.info(f'Number of elution groups: {n_elution_groups:,}')
            logger.info(f'\taverage size: {average_precursors_per_group:.2f}')

        else:
            logger.warning(f'no elution_group_idx column was found')

        if 'proteins' in self.speclib.precursor_df.columns:
            n_proteins = len(self.speclib.precursor_df['proteins'].unique())
            logger.info(f'Number of proteins: {n_proteins:,}')
        else:
            logger.warning(f'no proteins column was found')

        if 'channel' in self.speclib.precursor_df.columns:
            channels = self.speclib.precursor_df['channel'].unique()
            n_channels = len(channels)
            logger.info(f'Number of channels: {n_channels:,} ({channels})')

        else:
            logger.warning(f'no channel column was found, will assume only one channel')

        
        
        isotopes = utils.get_isotope_columns(self.speclib.precursor_df.columns)

        if len(isotopes) > 0:
            logger.info(f'Isotopes Distribution for {len(isotopes)} isotopes')

        logger.info(f'=================================')


        
    def get_rt_type(self, speclib):
        """check the retention time type of a spectral library
    

        Parameters
        ----------
        speclib : SpecLibBase
            spectral library

        Returns
        -------
        str
            retention time type, possible options: 'unknown','seconds', 'minutes', 'norm', 'irt'
        
        """

        rt_type = 'unknown'

        rt_series = speclib.precursor_df['rt_library']

        if rt_series.min() < 0:
            rt_type = 'irt'
        
        elif 0 <= rt_series.min() <= 1:
            rt_type = 'norm'

        elif rt_series.max() < self.config['library_loading']['rt_heuristic']:
            rt_type = 'minutes'

        elif rt_series.max() > self.config['library_loading']['rt_heuristic']:
            rt_type = 'seconds'

        if rt_type == 'unknown':
            logger.warning("""Could not determine retention time typ. 
                            Raw values will be used. 
                            Please specify extraction.rt_type with the possible values ('irt', 'norm, 'minutes', 'seconds',) in the config file.""")

        return rt_type
    

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
        
        # get retention time format
        if 'rt_type' in self.config['library_loading']:
            rt_type = self.config['library_loading']['rt_type']
            logger.info(f'forcing rt_type {rt_type} from config file')
        else:
            rt_type = self.get_rt_type(self.speclib)
            logger.info(f'rt_type automatically determined as {rt_type}')

        # iterate over raw files and yield raw data and spectral library
        for raw_location in self.raw_file_list:
            raw = data.TimsTOFTranspose(raw_location)
            raw_name = Path(raw_location).stem

            precursor_df = self.speclib.precursor_df.copy()
            precursor_df['raw_name'] = raw_name

            if rt_type == 'seconds' or rt_type == 'unknown':
                yield raw.jitclass(), precursor_df, self.speclib.fragment_df
            
            elif rt_type == 'minutes':
                precursor_df['rt_library'] *= 60

                yield raw.jitclass(), precursor_df, self.speclib.fragment_df

            elif rt_type == 'irt' or rt_type == 'norm':

                precursor_df['rt_library'] = self.norm_to_rt(raw, precursor_df['rt_library'].values) 

                yield raw.jitclass(), precursor_df, self.speclib.fragment_df
                
    def run(self, 
            figure_path = None,
            neptune_token = None, 
            neptune_tags = [],
            keep_decoys = False,
            fdr = 0.01,
            ):

        dataframes = []

        for dia_data, precursors_flat, fragments_flat in self.get_run_data():

            raw_name = precursors_flat.iloc[0]['raw_name']

            try:

                workflow = Workflow(
                    self.config, 
                    dia_data, 
                    precursors_flat, 
                    fragments_flat, 
                    figure_path = figure_path,
                    neptune_token = neptune_token,
                    neptune_tags = neptune_tags
                    )
   
                workflow.calibration()

                df = workflow.extraction(keep_decoys = keep_decoys)
                df = df[df['qval'] <= fdr]

                if self.config['multiplexing']['multiplexed_quant']:
                    df = workflow.requantify(df)

                df['run'] = raw_name
                dataframes.append(df)

                del workflow
            
            except Exception as e:
                logger.exception(e)
                continue

        out_df = pd.concat(dataframes)
        out_df.to_csv(os.path.join(self.output_folder, f'alpha_psms.tsv'), sep='\t', index=False)

class Workflow:
    def __init__(
            self, 
            config, 
            dia_data,
            precursors_flat, 
            fragments_flat,
            figure_path = None,
            neptune_token = None,
            neptune_tags=[]
        ):

        self.config = config
        self.dia_data = dia_data
        self.raw_name = precursors_flat.iloc[0]['raw_name']

        
        if self.config["library_loading"]["channel_filter"] == '':
            allowed_channels = precursors_flat['channel'].unique()
        else:
            allowed_channels = [int(c) for c in self.config["library_loading"]["channel_filter"].split(',')]
            logger.progress(f'Applying channel filter using only: {allowed_channels}')
        
        self.precursors_flat_raw = precursors_flat.copy()
        self.precursors_flat = self.precursors_flat_raw[self.precursors_flat_raw['channel'].isin(allowed_channels)].copy()
        self.fragments_flat = fragments_flat

        self.figure_path = figure_path
 
        if neptune_token is not None:
            
            try:
                self.run = neptune.init_run(
                    project="MannLabs/alphaDIA",
                    api_token=neptune_token
                )

                self.run['version'] = alphadia.__version__
                self.run["sys/tags"].add(neptune_tags)
                self.run['host'] = socket.gethostname()
                self.run['raw_file'] = self.raw_name
                self.run['config'].upload(File.from_content(yaml.dump(self.config)))
            except:
                logger.error("initilizing neptune session failed!")
                self.run = None
        else:
            self.run = None

        self.calibration_manager = CalibrationManager()
        self.calibration_manager.load_config(self.config['calibration_manager'])

        # initialize the progress dict
        self.progress = {
            'current_epoch': 0,
            'current_step': 0,
            'ms1_error': self.config['extraction_initial']['initial_ms1_tolerance'],
            'ms2_error': self.config['extraction_initial']['initial_ms2_tolerance'],
            'rt_error': self.config['extraction_initial']['initial_rt_tolerance'],
            'mobility_error': self.config['extraction_initial']['initial_mobility_tolerance'],
            'column_type': 'library',
            'num_candidates': self.config['extraction_initial']['initial_num_candidates'],
            'recalibration_target': self.config['calibration']['recalibration_target'],
            'accumulated_precursors': 0,
            'accumulated_precursors_0.01FDR': 0,
            'accumulated_precursors_0.001FDR': 0,
            'cycle_fwhm': 5,
            'mobility_fwhm': 0.015
        }

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

    def get_exponential_batches(self, step):
        """Get the number of batches for a given step
        This plan has the shape:
        1, 1, 1, 2, 4, 8, 16, 32, 64, ...
        """
        return int(2 ** max(step - 3,0))

    def get_batch_plan(self):
        n_eg = self.precursors_flat['elution_group_idx'].nunique()

        plan = []

        batch_size = self.config['calibration']['batch_size']
        step = 0
        start_index = 0

        while start_index < n_eg:
            n_batches = self.get_exponential_batches(step)
            stop_index = min(start_index + n_batches * batch_size, n_eg)
            plan.append((start_index, stop_index))
            step += 1
            start_index = stop_index

        return plan

    def start_of_calibration(self):

        self.batch_plan = self.get_batch_plan()

        

    def start_of_epoch(self, current_epoch):
        self.progress['current_epoch'] = current_epoch

        if self.run is not None:
            self.run["eval/epoch"].log(current_epoch)

        self.elution_group_order = self.precursors_flat['elution_group_idx'].sample(frac=1).values


        self.calibration_manager.predict(self.precursors_flat, 'precursor')
        self.calibration_manager.predict(self.fragments_flat, 'fragment')

        # make updates to the progress dict depending on the epoch
        if self.progress['current_epoch'] > 0:
            self.progress['recalibration_target'] = self.config['calibration']['recalibration_target'] * (1+current_epoch)

    def start_of_step(self, current_step, start_index, stop_index):
        self.progress['current_step'] = current_step
        if self.run is not None:
            self.run["eval/step"].log(current_step)

            for key, value in self.progress.items():
                self.run[f"eval/{key}"].log(value)

        logger.progress(f'=== Epoch {self.progress["current_epoch"]}, step {current_step}, extracting elution groups {start_index} to {stop_index} ===')

    def check_epoch_conditions(self):

        continue_calibration = False

        if self.progress['ms1_error'] > self.config['extraction_target']['target_ms1_tolerance']:
            continue_calibration = True

        if self.progress['ms2_error'] > self.config['extraction_target']['target_ms2_tolerance']:
            continue_calibration = True

        if self.progress['rt_error'] > self.config['extraction_target']['target_rt_tolerance']:
            continue_calibration = True

        if self.progress['mobility_error'] > self.config['extraction_target']['target_mobility_tolerance']:
            continue_calibration = True

        if self.progress['current_epoch'] < self.config['calibration']['min_epochs']:
            continue_calibration = True

        return continue_calibration

    def calibration(self):
        
        self.start_of_calibration()
        for current_epoch in range(self.config['calibration']['max_epochs']):
            self.start_of_epoch(current_epoch)
        
            
            if self.check_epoch_conditions():
                pass
            else:
                break
        
            features = []
            fragments = []
            for current_step, (start_index, stop_index) in enumerate(self.batch_plan):
                self.start_of_step(current_step, start_index, stop_index)

                eg_idxes = self.elution_group_order[start_index:stop_index]
                batch_df = self.precursors_flat[self.precursors_flat['elution_group_idx'].isin(eg_idxes)]
                
                feature_df, fragment_df = self.extract_batch(batch_df)
                features += [feature_df]
                fragments += [fragment_df]
                features_df = pd.concat(features)
                fragments_df = pd.concat(fragments)

                logger.info(f'number of dfs in features: {len(features)}, total number of features: {len(features_df)}')
                precursor_df = self.fdr_correction(features_df)
                #precursor_df = self.fdr_correction(precursor_df)

                if self.check_recalibration(precursor_df):
                    self.recalibration(precursor_df, fragments_df)
                    break
                else:
                    pass
            
            self.end_of_epoch()

        

        if 'final_full_calibration' in self.config['calibration']:
            if self.config['calibration']['final_full_calibration']:
                logger.info('Performing final calibration with all precursors')
                features_df, fragments_df = self.extract_batch(self.precursors_flat)
                precursor_df = self.fdr_correction(features_df)
                self.recalibration(precursor_df, fragments_df)

        self.end_of_calibration()


    def end_of_epoch(self):
        pass

    def end_of_calibration(self):
        self.calibration_manager.predict(self.precursors_flat, 'precursor')
        self.calibration_manager.predict(self.fragments_flat, 'fragment')
        pass

    def recalibration(self, precursor_df, fragments_df):
        precursor_df_filtered = precursor_df[precursor_df['qval'] < 0.001]
        precursor_df_filtered = precursor_df_filtered[precursor_df_filtered['decoy'] == 0]

        self.calibration_manager.fit(
            precursor_df_filtered,
            'precursor', 
            plot = True, 
            figure_path = self.figure_path,
            neptune_run = self.run
        )

        m1_70 = self.calibration_manager.get_estimator('precursor', 'mz').ci(precursor_df_filtered, 0.70)
        m1_99 = self.calibration_manager.get_estimator('precursor', 'mz').ci(precursor_df_filtered, 0.95)
        rt_70 = self.calibration_manager.get_estimator('precursor', 'rt').ci(precursor_df_filtered, 0.70)
        rt_99 = self.calibration_manager.get_estimator('precursor', 'rt').ci(precursor_df_filtered, 0.95)
        mobility_70 = self.calibration_manager.get_estimator('precursor', 'mobility').ci(precursor_df_filtered, 0.70)
        mobility_99 = self.calibration_manager.get_estimator('precursor', 'mobility').ci(precursor_df_filtered, 0.95)

        #top_intensity_precursors = precursor_df_filtered.sort_values(by=['intensity'], ascending=False)
        median_precursor_intensity = precursor_df_filtered['weighted_ms1_intensity'].median()
        top_intensity_precursors = precursor_df_filtered[precursor_df_filtered['weighted_ms1_intensity'] > median_precursor_intensity]
        fragments_df_filtered = fragments_df[fragments_df['precursor_idx'].isin(top_intensity_precursors['precursor_idx'])]
        median_fragment_intensity = fragments_df_filtered['intensity'].median()
        fragments_df_filtered = fragments_df_filtered[fragments_df_filtered['intensity'] > median_fragment_intensity].head(50000)

        self.calibration_manager.fit(
            fragments_df_filtered,
            'fragment', 
            plot=True, 
            figure_path = self.figure_path,
            neptune_run = self.run
        )

        m2_70 = self.calibration_manager.get_estimator('fragment', 'mz').ci(fragments_df_filtered, 0.70)
        m2_99 = self.calibration_manager.get_estimator('fragment', 'mz').ci(fragments_df_filtered, 0.95)

        self.progress["ms1_error"] = max(m1_99, self.config['extraction_target']['target_ms1_tolerance'])
        self.progress["ms2_error"] = max(m2_99, self.config['extraction_target']['target_ms2_tolerance'])
        self.progress["rt_error"] = max(rt_99, self.config['extraction_target']['target_rt_tolerance'])
        self.progress["mobility_error"] = max(mobility_99, self.config['extraction_target']['target_mobility_tolerance'])
        self.progress["column_type"] = 'calibrated'
        self.progress['cycle_fwhm'] = precursor_df_filtered['cycle_fwhm'].median()
        self.progress['mobility_fwhm'] = precursor_df_filtered['mobility_fwhm'].median()

        if self.run is not None:
            precursor_df_fdr = precursor_df_filtered[precursor_df_filtered['qval'] < 0.01]
            self.run["eval/precursors"].log(len(precursor_df_fdr))
            self.run['eval/99_ms1_error'].log(m1_99)
            self.run['eval/99_ms2_error'].log(m2_99)
            self.run['eval/99_rt_error'].log(rt_99)
            self.run['eval/99_mobility_error'].log(mobility_99)


    
    def check_recalibration(self, precursor_df):
        self.progress['accumulated_precursors'] = len(precursor_df)
        self.progress['accumulated_precursors_0.01FDR'] = len(precursor_df[precursor_df['qval'] < 0.01])
        self.progress['accumulated_precursors_0.001FDR'] = len(precursor_df[precursor_df['qval'] < 0.001])

        logger.progress(f'=== checking if recalibration conditions were reached, target {self.progress["recalibration_target"]} precursors ===')

        logger.progress(f'Accumulated precursors: {self.progress["accumulated_precursors"]:,}, 0.01 FDR: {self.progress["accumulated_precursors_0.01FDR"]:,}, 0.001 FDR: {self.progress["accumulated_precursors_0.001FDR"]:,}')

        perform_recalibration = False

        if self.progress['accumulated_precursors_0.001FDR'] > self.progress['recalibration_target']:
            perform_recalibration = True
           
        if self.progress['current_step'] == len(self.batch_plan) -1:
            perform_recalibration = True

        return perform_recalibration

    
    def fdr_correction(self, features_df):
        return fdr_correction(features_df, figure_path=self.figure_path, neptune_run=self.run)
        

    def extract_batch(self, batch_df):
        logger.progress(f'MS1 error: {self.progress["ms1_error"]}, MS2 error: {self.progress["ms2_error"]}, RT error: {self.progress["rt_error"]}, Mobility error: {self.progress["mobility_error"]}')
        
        config = HybridCandidateConfig()
        config.update(self.config['selection_config'])
        config.update({
            'rt_tolerance':self.progress["rt_error"],
            'mobility_tolerance': self.progress["mobility_error"],
            'candidate_count': self.progress["num_candidates"],
            'precursor_mz_tolerance': self.progress["ms1_error"],
            'fragment_mz_tolerance': self.progress["ms2_error"]
        })
        
        extraction = HybridCandidateSelection(
            self.dia_data,
            batch_df,
            self.fragments_flat,
            config.jitclass(),
            rt_column = f'rt_{self.progress["column_type"]}',
            mobility_column = f'mobility_{self.progress["column_type"]}',
            precursor_mz_column = f'mz_{self.progress["column_type"]}',
            fragment_mz_column = f'mz_{self.progress["column_type"]}',
            fwhm_rt = self.progress['cycle_fwhm'],
            fwhm_mobility = self.progress['mobility_fwhm'],
            thread_count=self.config['general']['thread_count']
        )
        candidates_df = extraction()

        config = plexscoring.CandidateConfig()
        config.update(self.config['scoring_config'])
        config.update({
            'precursor_mz_tolerance': self.progress["ms1_error"],
            'fragment_mz_tolerance': self.progress["ms2_error"]
        })

        candidate_scoring = plexscoring.CandidateScoring(
            self.dia_data,
            self.precursors_flat,
            self.fragments_flat,
            config = config,
            rt_column = f'rt_{self.progress["column_type"]}',
            mobility_column = f'mobility_{self.progress["column_type"]}',
            precursor_mz_column = f'mz_{self.progress["column_type"]}',
            fragment_mz_column = f'mz_{self.progress["column_type"]}',
        )

        features_df, fragments_df = candidate_scoring(candidates_df, thread_count=10, debug=False)

        return features_df, fragments_df
       
    def extraction(
            self,
            keep_decoys=False):

        if self.run is not None:
            for key, value in self.progress.items():
                self.run[f"eval/{key}"].log(value)

        self.progress["num_candidates"] = self.config['extraction_target']['target_num_candidates']
        self.progress["ms1_error"] = self.config['extraction_target']['target_ms1_tolerance']
        self.progress["ms2_error"] = self.config['extraction_target']['target_ms2_tolerance']
        self.progress["rt_error"] = self.config['extraction_target']['target_rt_tolerance']
        self.progress["mobility_error"] = self.config['extraction_target']['target_mobility_tolerance']
        self.progress["column_type"] = 'calibrated'

        self.calibration_manager.predict(self.precursors_flat, 'precursor')
        self.calibration_manager.predict(self.fragments_flat, 'fragment')

        features_df, fragments_df = self.extract_batch(self.precursors_flat)
        #features_df = features_df[features_df['fragment_coverage'] > 0.1]
        precursor_df = self.fdr_correction(features_df)
        #precursor_df = self.fdr_correction(precursor_df)

        if not keep_decoys:
            precursor_df = precursor_df[precursor_df['decoy'] == 0]
        precursors_05 = len(precursor_df[(precursor_df['qval'] < 0.05) & (precursor_df['decoy'] == 0)])
        precursors_01 = len(precursor_df[(precursor_df['qval'] < 0.01) & (precursor_df['decoy'] == 0)])
        precursors_001 = len(precursor_df[(precursor_df['qval'] < 0.001) & (precursor_df['decoy'] == 0)])

        if self.run is not None:
            self.run["eval/precursors"].log(precursors_01)
            self.run.stop()

        logger.progress(f'=== extraction finished, 0.05 FDR: {precursors_05:,}, 0.01 FDR: {precursors_01:,}, 0.001 FDR: {precursors_001:,} ===')

        return precursor_df   

    def requantify(
            self,
            psm_df
        ):

        self.calibration_manager.predict(self.precursors_flat_raw, 'precursor')
        self.calibration_manager.predict(self.fragments_flat, 'fragment')

        reference_candidates = plexscoring.candidate_features_to_candidates(psm_df)

        if not 'multiplexing' in self.config:
            raise ValueError('no multiplexing config found')
        
        logger.progress(f'=== Multiplexing {len(reference_candidates):,} precursors ===')

        original_channels = psm_df['channel'].unique().tolist()
        logger.progress(f'original channels: {original_channels}')
        
        reference_channel = self.config['multiplexing']['reference_channel']
        logger.progress(f'reference channel: {reference_channel}')

        target_channels = [int(c) for c in self.config['multiplexing']['target_channels'].split(',')]
        logger.progress(f'target channels: {target_channels}')

        decoy_channel = self.config['multiplexing']['decoy_channel']
        logger.progress(f'decoy channel: {decoy_channel}')

        channels = list(set(original_channels + [reference_channel] + target_channels + [decoy_channel]))
        multiplexed_candidates = plexscoring.multiplex_candidates(reference_candidates, self.precursors_flat_raw, channels=channels)
        
        channel_count_lib = self.precursors_flat_raw['channel'].value_counts()
        channel_count_multiplexed = multiplexed_candidates['channel'].value_counts()
        ## log channels with less than 100 precursors
        for channel in channels:
            if channel not in channel_count_lib:
                logger.warning(f'channel {channel} not found in library')
            if channel not in channel_count_multiplexed:
                logger.warning(f'channel {channel} could not be mapped to existing IDs.')        

        logger.progress(f'=== Requantifying {len(multiplexed_candidates):,} precursors ===')

        config = plexscoring.CandidateConfig()
        config.max_cardinality = 1
        config.score_grouped = True
        config.reference_channel = 0

        multiplexed_scoring = plexscoring.CandidateScoring(
            self.dia_data,
            self.precursors_flat_raw,
            self.fragments_flat,
            config=config
        )

        multiplexed_features, fragments = multiplexed_scoring(multiplexed_candidates)

        return channel_fdr_correction(multiplexed_features)