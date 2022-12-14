# internal imports
from .utils import indices_to_slices

# alpha imports
import alphatims.bruker
import alphatims.utils
import alphatims.utils
import alphatims.tempmmap

# external imports
import logging
import pandas as pd
import numpy as np

class ElutionGroupAssembler():

    def __init__(self):
        pass

    def set_dia_data(self, dia_data):
        self.dia_data = dia_data

    def set_library(self, library):
        self.library = library

    def run(self):

        logging.info('Assemble elution groups and build elution group indexer.')

        if not hasattr(self,'dia_data'):
            raise AttributeError('No dia_data property set. Please call set_dia_data first.')
        
        if not hasattr(self,'library'):
            raise AttributeError('No library property set. Please call set_library first.')

        # create fixed library df from library
        # TODO, is there a way to sort the precursor df while maintaining the peptdeep library class?

        self.precursor_df= self.library.precursor_df.copy()
        self.precursor_df = self.precursor_df[self.precursor_df['decoy'] == 0]

        self.precursor_df['mods_sequence'] = self.precursor_df['mods']
        self.precursor_df['mods_sequence'] += self.precursor_df['sequence']
        self.precursor_df['mods_sequence'] += self.precursor_df['charge'].map(str)
        self.precursor_df['elution_group_idxes'] = self.precursor_df.groupby('mods_sequence').ngroup()

        self.precursor_df.sort_values(by='elution_group_idxes',ascending=True, inplace=True)

        self.elution_groups = alphatims.tempmmap.clone(indices_to_slices(self.precursor_df['elution_group_idxes'].values))

        num_elution_groups = len(self.elution_groups)
        num_precursors = len(self.precursor_df)

        logging.info(f'Assembled {num_elution_groups} elution groups from {num_precursors} precursors.')