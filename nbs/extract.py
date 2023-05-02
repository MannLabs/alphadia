import os, psutil
os.environ['NUMEXPR_MAX_THREADS'] = '20'
import logging
import logging
from alphadia.extraction import processlogger

import matplotlib
matplotlib.use('Agg')

import neptune.new as neptune

from alphabase.spectral_library.base import SpecLibBase
from alphadia.extraction.planning import Plan

try:
    neptune_token = os.environ['NEPTUNE_TOKEN']
except KeyError:
    logging.error('NEPTUNE_TOKEN environtment variable not set')

# spectral library location
# requires an alphabase spectral library with decoys
lib_location = '/Users/georgwallmann/Documents/data/raw_data/Alpha_dia_benchmarking/diaPASEF_vs_synchroPASEF/library_48_fractions_MSFragger.hdf'

# output location
# a file called alpha psm will be written to this location
output_location = '/Users/georgwallmann/Documents/data/alphadia_runs/2023_02_12_diaPASEF_vs_synchroPASEF/48_fraction_msfragger/'

raw_files = ['/Users/georgwallmann/Documents/data/raw_data/Alpha_dia_benchmarking/diaPASEF_vs_synchroPASEF/20221221_TIMS05_PaSk_SA_HeLa_Evo05_200ng_21min_IM0713_diaPASEF_S4-A1_1_500.d',
             '/Users/georgwallmann/Documents/data/raw_data/Alpha_dia_benchmarking/diaPASEF_vs_synchroPASEF/20221221_TIMS05_PaSk_SA_HeLa_Evo05_200ng_21min_IM0713_diaPASEF_S4-A2_1_504.d',
             '/Users/georgwallmann/Documents/data/raw_data/Alpha_dia_benchmarking/diaPASEF_vs_synchroPASEF/20221221_TIMS05_PaSk_SA_HeLa_Evo05_200ng_21min_IM0713_diaPASEF_S4-A3_1_508.d',
             '/Users/georgwallmann/Documents/data/raw_data/Alpha_dia_benchmarking/diaPASEF_vs_synchroPASEF/20221221_TIMS05_PaSk_SA_HeLa_Evo05_200ng_21min_IM0713_diaPASEF_S4-A4_1_512.d',
             '/Users/georgwallmann/Documents/data/raw_data/Alpha_dia_benchmarking/diaPASEF_vs_synchroPASEF/20221221_TIMS05_PaSk_SA_HeLa_Evo05_200ng_21min_IM0713_SyP_classical_5bins_S2-A1_1_449.d',
             '/Users/georgwallmann/Documents/data/raw_data/Alpha_dia_benchmarking/diaPASEF_vs_synchroPASEF/20221221_TIMS05_PaSk_SA_HeLa_Evo05_200ng_21min_IM0713_SyP_classical_5bins_S2-A2_1_453.d',
             '/Users/georgwallmann/Documents/data/raw_data/Alpha_dia_benchmarking/diaPASEF_vs_synchroPASEF/20221221_TIMS05_PaSk_SA_HeLa_Evo05_200ng_21min_IM0713_SyP_classical_5bins_S2-A3_1_457.d',
             '/Users/georgwallmann/Documents/data/raw_data/Alpha_dia_benchmarking/diaPASEF_vs_synchroPASEF/20221221_TIMS05_PaSk_SA_HeLa_Evo05_200ng_21min_IM0713_SyP_classical_5bins_S2-A4_1_464.d']

processlogger.init_logging(output_location)

test_lib = SpecLibBase()
test_lib.load_hdf(lib_location, load_mod_seq=True)

plan = Plan(raw_files)
plan.from_spec_lib_base(test_lib)
plan.run(output_location, keep_decoys=True, fdr=1.0)