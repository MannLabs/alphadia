
# native imports
import logging
import os
logger = logging.getLogger()

import pandas as pd
import numpy as np

class SearchPlanOutput:

    def __init__(self, config, output_folder):
        
        self._config = config
        self._output_folder = output_folder

    @property
    def config(self):
        return self._config
    
    @property
    def output_folder(self):
        return self._output_folder
    
    def build_output(self, folder_list):

        self.build_precursor_table(folder_list)
        self.build_fragment_table(folder_list)
        self.build_library(folder_list)

    def build_precursor_table(self, folder_list):
        """Build precursor table from search plan output
        """

        psm_df_list = []
        stat_df_list = []
        
        for folder in folder_list:
            raw_name = os.path.basename(folder)
            psm_path = os.path.join(folder, 'psm.tsv')

            logger.progress(f'Building output for {raw_name}')

            if not os.path.exists(psm_path):
                logger.warning(f'no psm file found for {raw_name}')
                continue
            run_df = pd.read_csv(psm_path, sep='\t')
            
            psm_df_list.append(run_df)
            stat_df_list.append(build_stat_df(run_df))

        logger.progress('Building combined output')
        psm_df = pd.concat(psm_df_list)
        stat_df = pd.concat(stat_df_list)

        psm_df = perform_protein_grouping(psm_df, group_column=self.config['fdr']['group_level'])
        psm_df = perform_protein_fdr(psm_df)
        #psm_df = psm_df[psm_df['pg_qval'] <= self.config['fdr']['fdr']]

        logger.progress('Writing combined output to disk')
        psm_df.to_csv(os.path.join(self.output_folder, 'psm.tsv'), sep='\t', index=False, float_format='%.6f')
        stat_df.to_csv(os.path.join(self.output_folder, 'stat.tsv'), sep='\t', index=False, float_format='%.6f')

        logger.info(f'Finished building output')

    def build_fragment_table(self, folder_list):
        """Build fragment table from search plan output
        """
        logger.warning("Fragment table not implemented yet")

    def build_library(self, folder_list):
        """Build spectral library from search plan output
        """
        logger.warning("Spectral library not implemented yet")

def build_stat_df(run_df):

    run_stat_df = []
    for name, group in run_df.groupby('channel'):
        run_stat_df.append({
            'run': run_df['run'].iloc[0],
            'channel': name,
            'precursors': np.sum(group['qval'] <= 0.01),
            'proteins': group[group['qval'] <= 0.01]['proteins'].nunique(),
            'ms1_accuracy': np.mean(group['weighted_mass_error']),
            'fwhm_rt': np.mean(group['cycle_fwhm']),
            'fwhm_mobility': np.mean(group['mobility_fwhm']),
        })
    
    return pd.DataFrame(run_stat_df)

def perform_protein_grouping(psm_df, group_column = 'proteins'):
    """Perform protein grouping on PSM dataframe
    """

    return psm_df

def perform_protein_fdr(psm_df):
    """Perform protein FDR on PSM dataframe
    """

    return psm_df

