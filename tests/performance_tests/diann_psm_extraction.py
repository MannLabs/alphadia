
import logging
logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

import os
import neptune.new as neptune
import pathlib
import socket



from alphadia.extraction.planning import Plan
from alphadia.extraction.calibration import RunCalibration
from alphadia.extraction.data import TimsTOFDIA
from alphadia.extraction.testing import update_datashare
from alphadia.extraction.scoring import fdr_correction, unpack_fragment_info, MS2ExtractionWorkflow
from alphadia.extraction.candidateselection import MS1CentricCandidateSelection

if __name__ == "__main__":

    # set up neptune logging
    try:
        neptune_token = os.environ['NEPTUNE_TOKEN']
    except KeyError:
        logging.error('NEPTUNE_TOKEN environtment variable not set')
        raise KeyError from None
    
    run = neptune.init_run(
        project="MannLabs/alphaDIA",
        api_token=neptune_token
    )
    run['version'] = 'alpha_0.1'
    run["sys/tags"].add(["0_brunner_2022_1ng_extraction"])
    run['host'] = socket.gethostname()

    # set up logging
    
    logging.getLogger().setLevel(logging.INFO)
    logging.info("Starting diann psm extraction performance test")


    # get test dir from environment variable
    try:
        test_dir = os.environ['TEST_DATA_DIR']
    except KeyError:
        logging.error('TEST_DATA_DIR environtment variable not set')
        raise KeyError from None
    
    logging.info(f"Test data directory: {test_dir}")

    dependencies = {
        'folder_name': '0_brunner_2022_1ng_extraction',
        'file_list': [
            'https://datashare.biochem.mpg.de/s/LypobC5QM9HLl89',
            'https://datashare.biochem.mpg.de/s/H8y7zzQvdEkb42E'
        ]
    }

    output_dir = os.path.join(test_dir, dependencies['folder_name'])
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    dependency_list = dependencies['file_list']
    for element in dependency_list:
        update_datashare(element, output_dir)

    # annotate library with predicted fragments
    psm_lib_location = os.path.join(output_dir,'brunner_2022_1ng_rep01.hdf')
    raw_location = os.path.join(output_dir,'20200827_TIMS04_EVO07_AnBr_1ng_dia_rep01_400s_30min_S1-D1_1_2944.d')

    script_location = pathlib.Path(__file__).parent.resolve()
    yaml_file = os.path.join(script_location, '..','..', 'misc','config','default.yaml')

    run["config"].upload(yaml_file)
    
    plan = Plan(yaml_file)
    plan.load_speclib(psm_lib_location, dense=True)
    calibration = RunCalibration()
    conf = calibration.load_yaml(yaml_file)

    profiling = []


    dia_data = TimsTOFDIA(raw_location)

    initial_ms1_error = 80
    initial_ms2_error = 120
    initial_rt_error = 30

    target_ms1_error = 10
    target_ms2_error = 15
    target_rt_error = 10


    mobility_99 = 0.03

    iteration = 0

    precursors_flat, fragments_flat = plan.speclib.precursor_df, plan.speclib.fragment_df

    while initial_ms1_error >= target_ms1_error or iteration < 1:
        logging.info(f'Starting Iteration {iteration}, RT error {initial_rt_error:.2f}s,MS1 error {initial_ms1_error:.2f} ppm, MS2 error {initial_ms2_error:.2f} ppm')
        
        
        calibration.predict(precursors_flat, 'precursor')
        calibration.predict(fragments_flat, 'fragment')

        run["eval/iteration"].log(iteration)
        if iteration == 0:
            column_type = 'library'
            num_candidates = 2
        else:
            column_type = 'calibrated'
            num_candidates = 1

        extraction = MS1CentricCandidateSelection(
            dia_data,
            precursors_flat, 
            rt_column = f'rt_{column_type}',
            mobility_column = f'mobility_{column_type}',
            precursor_mz_column = f'mz_{column_type}',
            rt_tolerance=initial_rt_error,
            mobility_tolerance=mobility_99,
            num_candidates=num_candidates,
            num_isotopes=2,
            mz_tolerance=initial_ms1_error,
        )
        candidates_df = extraction()
        
                
        candidates_filtered = candidates_df[candidates_df['fraction_nonzero'] > 0.0]
        extraction = MS2ExtractionWorkflow(
            dia_data,
            precursors_flat, 
            candidates_filtered,
            fragments_flat,
            coarse_mz_calibration = False,
            rt_column = f'rt_{column_type}',
            mobility_column = f'mobility_{column_type}',
            precursor_mz_column = f'mz_{column_type}',
            fragment_mz_column = f'mz_{column_type}',
            precursor_mass_tolerance=initial_ms1_error,
            fragment_mass_tolerance=initial_ms2_error,
        )
        
        
        features_df = extraction()
        features_df['decoy'] = precursors_flat['decoy'].values[features_df['index'].values]
        features_df['charge'] = precursors_flat['charge'].values[features_df['index'].values]
        features_df['nAA'] = precursors_flat['nAA'].values[features_df['index'].values]

        
        features_df = fdr_correction(features_df, neptune_run=run)


        feature_filtered = features_df[features_df['qval'] < 0.01]
        run["eval/precursors"].log(len(feature_filtered))
        logging.info(f'Found {len(feature_filtered):,} features with qval < 0.01')
        
        calibration.fit(feature_filtered,'precursor', plot=True, neptune_run=run)
        m1_70 = calibration.get_estimator('precursor', 'mz').ci(features_df, 0.7)[0]
        m1_99 = calibration.get_estimator('precursor', 'mz').ci(features_df, 0.99)[0]
        rt_70 = calibration.get_estimator('precursor', 'rt').ci(features_df, 0.7)[0]
        rt_99 = calibration.get_estimator('precursor', 'rt').ci(features_df, 0.7)[0]
        mobility_99 = calibration.get_estimator('precursor', 'mobility').ci(features_df, 0.99)[0]
        
        run["eval/99_ms1_error"].log(m1_99)
        run["eval/99_rt_error"].log(rt_99)
        run["eval/99_mobility_error"].log(mobility_99)

        fragment_calibration_df = unpack_fragment_info(feature_filtered)
        fragment_calibration_df = fragment_calibration_df.sort_values(by=['intensity'], ascending=True).head(10000)

        calibration.fit(fragment_calibration_df,'fragment', plot=True, neptune_run=run)
        m2_70 = calibration.get_estimator('fragment', 'mz').ci(fragment_calibration_df, 0.7)[0]
        m2_99 = calibration.get_estimator('fragment', 'mz').ci(fragment_calibration_df, 0.99)[0]

        run["eval/99_ms2_error"].log(m2_99)

        if initial_ms1_error == target_ms1_error and initial_ms2_error == target_ms2_error:
            logging.info(f'Ending iteration {iteration}, target_reached')
            break

        initial_ms1_error = max(m1_70, target_ms1_error, initial_ms1_error * 0.6)
        initial_ms2_error = max(m2_70, target_ms2_error, initial_ms2_error * 0.6)
        initial_rt_error = max(rt_70, target_rt_error, initial_rt_error * 0.6)

        iteration += 1