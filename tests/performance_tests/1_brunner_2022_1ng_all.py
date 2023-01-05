
import logging
logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

import os
import neptune.new as neptune
import pathlib
import socket

import matplotlib
matplotlib.use('Agg')

from alphadia.extraction.planning import Plan
from alphadia.extraction.calibration import RunCalibration
from alphadia.extraction.data import TimsTOFDIA
from alphadia.extraction.testing import update_datashare
from alphadia.extraction.scoring import fdr_correction, unpack_fragment_info, MS2ExtractionWorkflow
from alphadia.extraction.candidateselection import MS1CentricCandidateSelection
from alphabase.spectral_library.base import SpecLibBase

if __name__ == "__main__":

    # set up logging
    logging.getLogger().setLevel(logging.INFO)
    logging.info("Starting diann psm extraction performance test")

    yaml_file = '/Users/georgwallmann/Documents/git/alphadia/misc/config/diann_lib.yaml'

    raw_files = ['/Users/georgwallmann/Documents/data/brunner2022/diaPASEF_repetitions/20200827_TIMS04_EVO07_AnBr_1ng_dia_rep01_400s_30min_S1-D1_1_2944.d',
                '/Users/georgwallmann/Documents/data/brunner2022/diaPASEF_repetitions/20200827_TIMS04_EVO07_AnBr_1ng_dia_rep01_400s_30min_S1-D3_1_2946.d',]
    x = [
                
                
                '/Users/georgwallmann/Documents/data/brunner2022/diaPASEF_repetitions/20200827_TIMS04_EVO07_AnBr_1ng_dia_rep02_400s_30min_S1-D4_1_2947.d',
                '/Users/georgwallmann/Documents/data/brunner2022/diaPASEF_repetitions/20200827_TIMS04_EVO07_AnBr_1ng_dia_rep02_400s_30min_S1-D5_1_2948.d',
                '/Users/georgwallmann/Documents/data/brunner2022/diaPASEF_repetitions/20200827_TIMS04_EVO07_AnBr_1ng_dia_rep02_400s_30min_S1-D6_1_2949.d',
                '/Users/georgwallmann/Documents/data/brunner2022/diaPASEF_repetitions/20200827_TIMS04_EVO07_AnBr_1ng_dia_rep03_400s_30min_S1-D7_1_2950.d',
                '/Users/georgwallmann/Documents/data/brunner2022/diaPASEF_repetitions/20200827_TIMS04_EVO07_AnBr_1ng_dia_rep03_400s_30min_S1-D8_1_2951.d',
                '/Users/georgwallmann/Documents/data/brunner2022/diaPASEF_repetitions/20200827_TIMS04_EVO07_AnBr_1ng_dia_rep03_400s_30min_S1-D9_1_2952.d',
                '/Users/georgwallmann/Documents/data/brunner2022/diaPASEF_repetitions/20200827_TIMS04_EVO07_AnBr_1ng_dia_rep04_400s_30min_S1-D10_1_2953.d',
                '/Users/georgwallmann/Documents/data/brunner2022/diaPASEF_repetitions/20200827_TIMS04_EVO07_AnBr_1ng_dia_rep04_400s_30min_S1-D11_1_2954.d',
                '/Users/georgwallmann/Documents/data/brunner2022/diaPASEF_repetitions/20200827_TIMS04_EVO07_AnBr_1ng_dia_rep04_400s_30min_S1-D12_1_2955.d',
                '/Users/georgwallmann/Documents/data/brunner2022/diaPASEF_repetitions/20200827_TIMS04_EVO07_AnBr_1ng_dia_rep05_400s_30min_S1-E1_1_2956.d',
                '/Users/georgwallmann/Documents/data/brunner2022/diaPASEF_repetitions/20200827_TIMS04_EVO07_AnBr_1ng_dia_rep05_400s_30min_S1-E2_1_2957.d',
                '/Users/georgwallmann/Documents/data/brunner2022/diaPASEF_repetitions/20200827_TIMS04_EVO07_AnBr_1ng_dia_rep05_400s_30min_S1-E3_1_2958.d',
                ]
 

    # get test dir from environment variable
    try:
        test_dir = os.environ['TEST_DATA_DIR']
    except KeyError:
        logging.error('TEST_DATA_DIR environtment variable not set')
        raise KeyError from None
    
    logging.info(f"Test data directory: {test_dir}")
    dependencies = {
        'folder_name': '1_brunner_2022_1ng_all',
        'file_list': [
            'https://datashare.biochem.mpg.de/s/cvHN0uZT3szGCcl',
            'https://datashare.biochem.mpg.de/s/jIaJoIEdom6bH5W',
            'https://datashare.biochem.mpg.de/s/sGoClvBpzk5RZ5G',
            'https://datashare.biochem.mpg.de/s/L3RMCPxMdqIA125',
            'https://datashare.biochem.mpg.de/s/W6WxxFfYYn10jxT',
            'https://datashare.biochem.mpg.de/s/dZWu3S0uuMo1jh9',
            'https://datashare.biochem.mpg.de/s/CCEDzP71kv9XwFS',
            'https://datashare.biochem.mpg.de/s/EJwUQbipZFTuB26'
        ]
    }

    output_dir = os.path.join(test_dir, dependencies['folder_name'])
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    dependency_list = dependencies['file_list']
    for element in dependency_list:
        update_datashare(element, output_dir)

    file_names = ['20200827_TIMS04_EVO07_AnBr_1ng_dia_rep01_400s_30min_S1-D1_1_2944.d',
                '20200827_TIMS04_EVO07_AnBr_1ng_dia_rep01_400s_30min_S1-D2_1_2945.d']
    raw_files = [os.path.join(output_dir, file) for file in file_names]

    print(raw_files)

    test_lib = SpecLibBase()
    test_lib_location = os.path.join(output_dir, 'out_lib.hdf')
    test_lib.load_hdf(test_lib_location, load_mod_seq=True)

    config_path = os.path.join(output_dir, 'diann_lib.yaml')
    plan = Plan(raw_files, config_update_path=config_path)
    plan.from_spec_lib_base(test_lib)
    plan.run(output_dir, log_neptune=True, neptune_tags=['1_brunner_2022_1ng_all'])