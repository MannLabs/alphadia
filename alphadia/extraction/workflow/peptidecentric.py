import os
import logging
logger = logging.getLogger()
from typing import Union

import numpy as np
import pandas as pd

from alphadia.extraction import plexscoring, scoring, hybridselection
from alphadia.extraction.workflow import manager, base
from alphabase.spectral_library.base import SpecLibBase

class PeptideCentricWorkflow(base.WorkflowBase):
    
    def __init__(
        self,
        instance_name: str,
        config: dict,
        dia_data_path: str,
        spectral_library: SpecLibBase,
        ) -> None:
        
        super().__init__(
            instance_name, 
            config,
            dia_data_path,
            spectral_library,
        )

        logger.progress(f'Initializing workflow {self.instance_name}')

        self.init_neptune()
        self.init_calibration_optimization_manager()
        self.init_spectral_library()
        
    @property
    def calibration_optimization_manager(self):
        """ Is used during the iterative optimization of the calibration parameters.
        Should not be stored on disk.
        """
        return self._calibration_optimization_manager
    
    @property
    def com(self):
        """alias for calibration_optimization_manager"""
        return self.calibration_optimization_manager
    
    def init_neptune(self):
        pass

    def init_calibration_optimization_manager(self):
        self._calibration_optimization_manager = manager.OptimizationManager({
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
            'accumulated_precursors_01FDR': 0,
            'accumulated_precursors_001FDR': 0,
        })

    def init_spectral_library(self):
        # apply channel filter
        if self.config["library_loading"]["channel_filter"] == '':
            allowed_channels = self.spectral_library.precursor_df['channel'].unique()
        else:
            allowed_channels = [int(c) for c in self.config["library_loading"]["channel_filter"].split(',')]
            logger.progress(f'Applying channel filter using only: {allowed_channels}')

        # normalize spectral library rt to file specific TIC profile
        self.spectral_library._precursor_df['rt_library'] = self.norm_to_rt(self.dia_data, self.spectral_library._precursor_df['rt_library'].values) 
        
        # filter spectral library to only contain precursors from allowed channels
        # save original precursor_df for later use
        self.spectral_library.precursor_df_unfiltered = self.spectral_library.precursor_df.copy()
        self.spectral_library._precursor_df = self.spectral_library.precursor_df_unfiltered[self.spectral_library.precursor_df_unfiltered['channel'].isin(allowed_channels)].copy()

    def norm_to_rt(
        self,
        dia_data,
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
    
    def get_exponential_batches(self, step):
        """Get the number of batches for a given step
        This plan has the shape:
        1, 1, 1, 2, 4, 8, 16, 32, 64, ...
        """
        return int(2 ** max(step - 3,0))

    def get_batch_plan(self):
        n_eg = self.spectral_library._precursor_df['elution_group_idx'].nunique()

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
        self.com.current_epoch = current_epoch

        if self.run is not None:
            self.run["eval/epoch"].log(current_epoch)

        self.elution_group_order = self.spectral_library.precursor_df['elution_group_idx'].sample(frac=1).values


        self.calibration_manager.predict(self.spectral_library._precursor_df, 'precursor')
        self.calibration_manager.predict(self.spectral_library._fragment_df, 'fragment')

        # make updates to the progress dict depending on the epoch
        if self.com.current_epoch > 0:
            self.com.recalibration_target= self.config['calibration']['recalibration_target'] * (1+current_epoch)

    def start_of_step(self, current_step, start_index, stop_index):
        self.com.current_step = current_step
        if self.run is not None:
            self.run["eval/step"].log(current_step)

            for key, value in self.progress.items():
                self.run[f"eval/{key}"].log(value)

        logger.progress(f'=== Epoch {self.com.current_epoch}, step {current_step}, extracting elution groups {start_index} to {stop_index} ===')

    def check_epoch_conditions(self):

        continue_calibration = False

        if self.com.ms1_error > self.config['extraction_target']['target_ms1_tolerance']:
            continue_calibration = True

        if self.com.ms2_error > self.config['extraction_target']['target_ms2_tolerance']:
            continue_calibration = True

        if self.com.rt_error > self.config['extraction_target']['target_rt_tolerance']:
            continue_calibration = True

        if self.com.mobility_error > self.config['extraction_target']['target_mobility_tolerance']:
            continue_calibration = True

        if self.com.current_epoch < self.config['calibration']['min_epochs']:
            continue_calibration = True

        return continue_calibration

    def calibration(self):
        
        if self.calibration_manager.is_fitted and self.calibration_manager.is_loaded_from_file:
            logger.progress('Skipping calibration as existing calibration was found')
            return
        
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
                batch_df = self.spectral_library._precursor_df[self.spectral_library._precursor_df['elution_group_idx'].isin(eg_idxes)]
                
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
                features_df, fragments_df = self.extract_batch(self.spectral_library._precursor_df)
                precursor_df = self.fdr_correction(features_df)
                self.recalibration(precursor_df, fragments_df)

        self.end_of_calibration()


    def end_of_epoch(self):
        pass

    def end_of_calibration(self):
        self.calibration_manager.predict(self.spectral_library._precursor_df, 'precursor')
        self.calibration_manager.predict(self.spectral_library._fragment_df, 'fragment')
        self.calibration_manager.save()
        pass

    def recalibration(self, precursor_df, fragments_df):
        precursor_df_filtered = precursor_df[precursor_df['qval'] < 0.001]
        precursor_df_filtered = precursor_df_filtered[precursor_df_filtered['decoy'] == 0]

        self.calibration_manager.fit(
            precursor_df_filtered,
            'precursor', 
            plot = True, 
            #figure_path = self.figure_path,
            #neptune_run = self.run
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
            #figure_path = self.figure_path,
            #neptune_run = self.run
        )

        m2_70 = self.calibration_manager.get_estimator('fragment', 'mz').ci(fragments_df_filtered, 0.70)
        m2_99 = self.calibration_manager.get_estimator('fragment', 'mz').ci(fragments_df_filtered, 0.95)

        self.com.fit({
            'ms1_error': max(m1_99, self.config['extraction_target']['target_ms1_tolerance']),
            'ms2_error': max(m2_99, self.config['extraction_target']['target_ms2_tolerance']),
            'rt_error': max(rt_99, self.config['extraction_target']['target_rt_tolerance']),
            'mobility_error': max(mobility_99, self.config['extraction_target']['target_mobility_tolerance']),
            'column_type': 'calibrated',
            'num_candidates': self.config['extraction_target']['target_num_candidates'],
        })

        self.optimization_manager.fit({
            'fwhm_rt': precursor_df_filtered['cycle_fwhm'].median(),
            'fwhm_mobility': precursor_df_filtered['mobility_fwhm'].median(),
        })

        if self.run is not None:
            precursor_df_fdr = precursor_df_filtered[precursor_df_filtered['qval'] < 0.01]
            self.run["eval/precursors"].log(len(precursor_df_fdr))
            self.run['eval/99_ms1_error'].log(m1_99)
            self.run['eval/99_ms2_error'].log(m2_99)
            self.run['eval/99_rt_error'].log(rt_99)
            self.run['eval/99_mobility_error'].log(mobility_99)
    
    def check_recalibration(self, precursor_df):
        self.com.accumulated_precursors = len(precursor_df)
        self.com.accumulated_precursors_01FDR = len(precursor_df[precursor_df['qval'] < 0.01])
        self.com.accumulated_precursors_001FDR = len(precursor_df[precursor_df['qval'] < 0.001])

        logger.progress(f'=== checking if recalibration conditions were reached, target {self.com.recalibration_target} precursors ===')

        self.log_precursor_df(precursor_df)

        perform_recalibration = False

        if self.com.accumulated_precursors_001FDR > self.com.recalibration_target:
            perform_recalibration = True
           
        if self.com.current_step == len(self.batch_plan) -1:
            perform_recalibration = True

        return perform_recalibration

    def fdr_correction(self, features_df):
        return scoring.fdr_correction(features_df, competetive_scoring=self.config['fdr']['competetive_scoring'])

    def extract_batch(self, batch_df):
        logger.progress(f'MS1 error: {self.com.ms1_error}, MS2 error: {self.com.ms2_error}, RT error: {self.com.rt_error}, Mobility error: {self.com.mobility_error}')
        
        config = hybridselection.HybridCandidateConfig()
        config.update(self.config['selection_config'])
        config.update({
            'rt_tolerance':self.com.rt_error,
            'mobility_tolerance': self.com.mobility_error,
            'candidate_count': self.com.num_candidates,
            'precursor_mz_tolerance': self.com.ms1_error,
            'fragment_mz_tolerance': self.com.ms2_error,
            'exclude_shared_ions': self.config['library_loading']['exclude_shared_ions']
        })
        
        extraction = hybridselection.HybridCandidateSelection(
            self.dia_data.jitclass(),
            batch_df,
            self.spectral_library.fragment_df,
            config.jitclass(),
            rt_column = f'rt_{self.com.column_type}',
            mobility_column = f'mobility_{self.com.column_type}',
            precursor_mz_column = f'mz_{self.com.column_type}',
            fragment_mz_column = f'mz_{self.com.column_type}',
            fwhm_rt = self.optimization_manager.fwhm_rt,
            fwhm_mobility = self.optimization_manager.fwhm_mobility,
            thread_count=self.config['general']['thread_count']
        )
        candidates_df = extraction()

        config = plexscoring.CandidateConfig()
        config.update(self.config['scoring_config'])
        config.update({
            'precursor_mz_tolerance': self.com.ms1_error,
            'fragment_mz_tolerance': self.com.ms2_error,
            'exclude_shared_ions': self.config['library_loading']['exclude_shared_ions']
        })

        candidate_scoring = plexscoring.CandidateScoring(
            self.dia_data.jitclass(),
            self.spectral_library._precursor_df,
            self.spectral_library._fragment_df,
            config = config,
            rt_column = f'rt_{self.com.column_type}',
            mobility_column = f'mobility_{self.com.column_type}',
            precursor_mz_column = f'mz_{self.com.column_type}',
            fragment_mz_column = f'mz_{self.com.column_type}',
        )

        features_df, fragments_df = candidate_scoring(candidates_df, thread_count=10, debug=False)

        return features_df, fragments_df
       
    def extraction(
            self,
            keep_decoys=False):

        if self.run is not None:
            for key, value in self.com.__dict__.items():
                self.run[f"eval/{key}"].log(value)

        self.com.fit({
            'num_candidates': self.config['extraction_target']['target_num_candidates'],
            'ms1_error': self.config['extraction_target']['target_ms1_tolerance'],
            'ms2_error': self.config['extraction_target']['target_ms2_tolerance'],
            'rt_error': self.config['extraction_target']['target_rt_tolerance'],
            'mobility_error': self.config['extraction_target']['target_mobility_tolerance'],
            'column_type': 'calibrated'
        })

        self.calibration_manager.predict(self.spectral_library._precursor_df, 'precursor')
        self.calibration_manager.predict(self.spectral_library._fragment_df, 'fragment')

        features_df, fragments_df = self.extract_batch(self.spectral_library._precursor_df)
        #features_df = features_df[features_df['fragment_coverage'] > 0.1]
        precursor_df = self.fdr_correction(features_df)
        #precursor_df = self.fdr_correction(precursor_df)

        if not keep_decoys:
            precursor_df = precursor_df[precursor_df['decoy'] == 0]

        self.log_precursor_df(precursor_df)
        
        #precursors_05 = len(precursor_df[(precursor_df['qval'] < 0.05) & (precursor_df['decoy'] == 0)])
        precursors_01 = len(precursor_df[(precursor_df['qval'] < 0.01) & (precursor_df['decoy'] == 0)])
        #precursors_001 = len(precursor_df[(precursor_df['qval'] < 0.001) & (precursor_df['decoy'] == 0)])

        if self.run is not None:
            self.run["eval/precursors"].log(precursors_01)
            self.run.stop()


        return precursor_df
    
    def log_precursor_df(self, precursor_df):
        total_precursors = len(precursor_df)

        target_precursors = len(precursor_df[precursor_df['decoy'] == 0])
        target_precursors_percentages = target_precursors / total_precursors * 100
        decoy_precursors = len(precursor_df[precursor_df['decoy'] == 1])
        decoy_precursors_percentages = decoy_precursors / total_precursors * 100

        logger.progress(f'========================= Precursor FDR =========================')
        logger.progress(f'Total precursors accumulated: {total_precursors:,}')
  
        for channel in precursor_df['channel'].unique():
            precursor_05fdr = len(precursor_df[(precursor_df['qval'] < 0.05) & (precursor_df['decoy'] == 0) & (precursor_df['channel'] == channel)])
            precursor_01fdr = len(precursor_df[(precursor_df['qval'] < 0.01) & (precursor_df['decoy'] == 0) & (precursor_df['channel'] == channel)])
            precursor_001fdr = len(precursor_df[(precursor_df['qval'] < 0.001) & (precursor_df['decoy'] == 0) & (precursor_df['channel'] == channel)])

            logger.progress(f'Channel {channel:>3}:\t 0.05 FDR: {precursor_05fdr:>5,}; 0.01 FDR: {precursor_01fdr:>5,}; 0.001 FDR: {precursor_001fdr:>5,}')

        logger.progress(f'=================================================================')

    def requantify(
            self,
            psm_df
        ):

        self.calibration_manager.predict(self.spectral_library.precursor_df_unfiltered, 'precursor')
        self.calibration_manager.predict(self.spectral_library._fragment_df, 'fragment')

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
            self.dia_data.jitclass(),
            self.spectral_library.precursor_df_unfiltered,
            self.spectral_library.fragment_df,
            config=config,
            precursor_mz_column='mz_calibrated',
            fragment_mz_column='mz_calibrated',
            rt_column='rt_calibrated',
            mobility_column='mobility_calibrated'
        )

        multiplexed_features, fragments = multiplexed_scoring(multiplexed_candidates)

        return scoring.channel_fdr_correction(multiplexed_features)