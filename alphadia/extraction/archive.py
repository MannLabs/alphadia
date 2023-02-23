class MS2ExtractionWorkflow_():
    
    def __init__(self, 
            dia_data,
            precursors_flat, 
            candidates,
            fragments_flat,
            num_precursor_isotopes=2,
            precursor_mass_tolerance=20,
            fragment_mass_tolerance=100,
            coarse_mz_calibration=False,
            include_fragment_info=True,
            rt_column = 'rt_library',
            mobility_column = 'mobility_library',
            precursor_mz_column = 'mz_library',
            fragment_mz_column = 'mz_library',
            debug=False
                   
        ):

        self.dia_data = dia_data
        self.precursors_flat = precursors_flat
        self.candidates = candidates


        self.fragments_mz_library = fragments_flat['mz_library'].values.copy()
        self.fragments_mz = fragments_flat[fragment_mz_column].values.copy()
        self.fragments_intensity = fragments_flat['intensity'].values.copy().astype('float64')
        self.fragments_type = fragments_flat['type'].values.copy().astype('int8')

        self.num_precursor_isotopes = num_precursor_isotopes
        self.precursor_mass_tolerance = precursor_mass_tolerance
        self.fragment_mass_tolerance = fragment_mass_tolerance
        self.include_fragment_info = include_fragment_info

        self.rt_column = rt_column
        self.mobility_column = mobility_column
        self.precursor_mz_column = precursor_mz_column
        self.fragment_mz_column = fragment_mz_column

        self.debug = debug

        # check if rough calibration is possible
        if 'mass_error' in self.candidates.columns and coarse_mz_calibration:

            target_indices = np.nonzero(precursors_flat['decoy'].values == 0)[0]
            target_df = candidates[candidates['index'].isin(target_indices)]
            
            correction = np.mean(target_df['mass_error'])
            logging.info(f'rough calibration will be performed {correction:.2f} ppm')

            self.fragments_mz = fragments_flat[fragment_mz_column].values + fragments_flat[fragment_mz_column].values/(10**6)*correction


    def __call__(self):

        logging.info(f'performing MS2 extraction for {len(self.candidates):,} candidates')

        features = []
        candidate_iterator = self.candidates.to_dict(orient="records")
        if not self.debug:
            for i, candidate_dict in tqdm(enumerate(candidate_iterator), total=len(candidate_iterator)):
                features += self.get_features(i, candidate_dict)
        else:
            for i, candidate_dict in enumerate(candidate_iterator):
                logging.info(f'Only a single candidate is allowed in debugging mode')
                return self.get_features(i, candidate_dict)
      
            
        features = pd.DataFrame(features)
        if len(self.candidates) > 0:
            logging.info(f'MS2 extraction was able to extract {len(features):,} sets of features for {len(features)/len(self.candidates)*100:.2f}% of candidates')
        else:
            logging.warning(f'zero candidates were provided for MS2 extraction')
        

        return features

    def get_features(self, i, candidate_dict):

        c_precursor_index = candidate_dict['index']

        c_charge = self.precursors_flat.charge.values[c_precursor_index]

        c_mz_predicted = candidate_dict['mz_library']

        # observed mz
        c_mz = candidate_dict[self.precursor_mz_column]

        c_frag_start_idx = self.precursors_flat.frag_start_idx.values[c_precursor_index]
        c_frag_stop_idx = self.precursors_flat.frag_stop_idx.values[c_precursor_index]

        c_fragments_mzs = self.fragments_mz[c_frag_start_idx:c_frag_stop_idx]
        #print('fragment_mz', c_fragments_mzs)
        
        c_fragments_order = np.argsort(c_fragments_mzs)
        
        c_fragments_mzs = c_fragments_mzs[c_fragments_order]
        #print('fragment_mz ordered', c_fragments_mzs)
        
        c_intensity = self.fragments_intensity[c_frag_start_idx:c_frag_stop_idx][c_fragments_order]
        c_fragments_type = self.fragments_type[c_frag_start_idx:c_frag_stop_idx][c_fragments_order]

        fragment_limits = utils.mass_range(c_fragments_mzs, self.fragment_mass_tolerance)
        fragment_tof_limits = utils.make_np_slice(
            self.dia_data.return_tof_indices(
                fragment_limits,
            )
        )

        scan_limits = np.array([[candidate_dict['scan_start'],candidate_dict['scan_stop'],1]])
        frame_limits = np.array([[candidate_dict['frame_start'],candidate_dict['frame_stop'],1]])
    
        quadrupole_limits = np.array([[c_mz_predicted,c_mz_predicted]])



        dense_fragments = self.dia_data.get_dense(
            frame_limits,
            scan_limits,
            fragment_tof_limits,
            quadrupole_limits
        )

        #return dense_fragments, (c_fragments_mzs, c_intensity, c_fragments_type)

        # calculate fragment values
        fragment_intensity = np.sum(dense_fragments[0], axis=(1,2))
        intensity_mask = np.nonzero(fragment_intensity > 10)[0]

        dense_fragments = dense_fragments[:,intensity_mask]
        c_fragments_mzs = c_fragments_mzs[intensity_mask]
    
        c_intensity = c_intensity[intensity_mask]
        c_fragments_type = c_fragments_type[intensity_mask]
        fragment_intensity = fragment_intensity[intensity_mask]

        if len(intensity_mask) < 3:
            return []

        num_isotopes = 3
        
        # get dense precursor
        precursor_isotopes = utils.calc_isotopes_center(
            c_mz,
            c_charge,
            self.num_precursor_isotopes
        )
        precursor_isotope_limits = utils.mass_range(
                precursor_isotopes,
                self.precursor_mass_tolerance
        )
        precursor_tof_limits =utils.make_np_slice(
            self.dia_data.return_tof_indices(
                precursor_isotope_limits,
            )
        )
        dense_precursor = self.dia_data.get_dense(
            frame_limits,
            scan_limits,
            precursor_tof_limits,
            np.array([[-1.,-1.]])
        )

        if self.debug:
            
            visualize_dense_fragments(dense_precursor)
            visualize_dense_fragments(dense_fragments)
  
        #print(np.sum(dense_fragments[0],axis=(1,2)))
        #print(np.sum(dense_precursor[0],axis=(1,2)))

        #return {}

        # ========= assembling general features =========

        frame_center = candidate_dict['frame_center']
        rt_center = self.dia_data.rt_values[frame_center]
        rt_lib = self.precursors_flat[self.rt_column].values[c_precursor_index]

        candidate_dict['rt_diff'] = rt_center - rt_lib

        scan_center = candidate_dict['scan_center']
        mobility_center = self.dia_data.mobility_values[scan_center]
        mobility_lib = self.precursors_flat[self.mobility_column].values[c_precursor_index]

        candidate_dict['mobility_diff'] = mobility_center - mobility_lib

        # ========= assembling precursor features =========
        theoreticsl_precursor_isotopes = utils.calc_isotopes_center(
            c_mz,
            c_charge,
            self.num_precursor_isotopes
        )

        precursor_mass_err, precursor_fraction, precursor_observations = utils.calculate_mass_deviation(
                dense_precursor[1], 
                theoreticsl_precursor_isotopes, 
                size=5
        )

        precursor_intensity = np.sum(dense_precursor[0], axis=(1,2))

        # monoisotopic precursor intensity
        candidate_dict['mono_precursor_intensity'] = precursor_intensity[0]

        # monoisotopic precursor mass error
        candidate_dict['mono_precursor_mass_error'] = precursor_mass_err[0]

        # monoisotopic precursor observations
        candidate_dict['mono_precursor_observations'] = precursor_observations[0]

        # monoisotopic precursor fraction
        candidate_dict['mono_precursor_fraction'] = precursor_fraction[0]

        # highest intensity isotope
        candidate_dict['top_precursor_isotope'] = np.argmax(precursor_intensity, axis=0)

        # highest intensity isotope mass error
        candidate_dict['top_precursor_intensity'] = precursor_intensity[candidate_dict['top_precursor_isotope']]

        # precursor mass error
        candidate_dict['top_precursor_mass_error'] = precursor_mass_err[candidate_dict['top_precursor_isotope']]

        # ========= assembling fragment features =========


        fragment_mass_err, fragment_fraction, fragment_observations = utils.calculate_mass_deviation(
                dense_fragments[1], 
                c_fragments_mzs, 
                size=5
        )

        precursor_sum = np.sum(dense_precursor[0],axis=0)

        
        correlations = utils.calculate_correlations(
                np.sum(dense_precursor[0],axis=0), 
                dense_fragments[0]
        )

        precursor_correlations = np.mean(correlations[0:2], axis=0)
        fragment_correlations = np.mean(correlations[2:4], axis=0)

        corr_sum = np.mean(correlations,axis=0)

        # the fragment order is given by the sum of the correlations
        fragment_order = np.argsort(corr_sum)[::-1]

        # number of fragments above intensity threshold
        candidate_dict['num_fragments'] = len(fragment_order)

        # number of fragments with precursor correlation above 0.5
        candidate_dict['num_fragments_pcorr_7'] = np.sum(precursor_correlations[fragment_order] > 0.7)
        candidate_dict['num_fragments_pcorr_5'] = np.sum(precursor_correlations[fragment_order] > 0.5)
        candidate_dict['num_fragments_pcorr_3'] = np.sum(precursor_correlations[fragment_order] > 0.3)
        candidate_dict['num_fragments_pcorr_2'] = np.sum(precursor_correlations[fragment_order] > 0.2)
        candidate_dict['num_fragments_pcorr_1'] = np.sum(precursor_correlations[fragment_order] > 0.1)

        # number of fragments with precursor correlation above 0.5
        candidate_dict['num_fragments_fcorr_7'] = np.sum(precursor_correlations[fragment_order] > 0.7)
        candidate_dict['num_fragments_fcorr_5'] = np.sum(precursor_correlations[fragment_order] > 0.5)
        candidate_dict['num_fragments_fcorr_3'] = np.sum(fragment_correlations[fragment_order] > 0.3)
        candidate_dict['num_fragments_fcorr_2'] = np.sum(fragment_correlations[fragment_order] > 0.2)
        candidate_dict['num_fragments_fcorr_1'] = np.sum(fragment_correlations[fragment_order] > 0.1)

        # mean precursor correlation for top n fragments
        candidate_dict['mean_pcorr_top_5'] = np.mean(precursor_correlations[fragment_order[0:5]])
        candidate_dict['mean_pcorr_top_10'] = np.mean(precursor_correlations[fragment_order[0:10]])
        candidate_dict['mean_pcorr_top_15'] = np.mean(precursor_correlations[fragment_order[0:15]])

        # mean correlation for top n fragments
        candidate_dict['mean_fcorr_top_5'] = np.mean(fragment_correlations[fragment_order[0:5]])
        candidate_dict['mean_fcorr_top_10'] = np.mean(fragment_correlations[fragment_order[0:10]])
        candidate_dict['mean_fcorr_top_15'] = np.mean(fragment_correlations[fragment_order[0:15]])

        #  ======== assembling fragment intensity features ========

        candidate_dict['fragment_intensity_top_5'] = np.mean(fragment_intensity[fragment_order[0:5]])
        candidate_dict['fragment_intensity_top_10'] = np.mean(fragment_intensity[fragment_order[0:10]])
        candidate_dict['fragment_intensity'] = np.mean(fragment_intensity)

        candidate_dict['fragment_rank_score_1d'] = local_rank_score_1d(fragment_intensity, c_intensity)
        candidate_dict['fragment_rank_score_2d'] = local_rank_score_2d(fragment_intensity, c_intensity)

        candidate_dict['fragment_dot_product'] = np.dot(fragment_intensity, c_intensity)
        candidate_dict['fragment_similarity'] = cosine_similarity_float(fragment_intensity, c_intensity)

        fragment_intensity_weighted = calc_fragment_shape(dense_fragments[0, fragment_order])
        candidate_dict['intensity_weighted_top_5'] = np.mean(fragment_intensity_weighted[0:5])
        candidate_dict['intensity_weighted_top_10'] = np.mean(fragment_intensity_weighted[0:10])
        candidate_dict['intensity_weighted'] = np.mean(fragment_intensity_weighted)

        fragment_shape = fragment_intensity_weighted / np.sum(dense_fragments[0], axis = (1,2))
        candidate_dict['fragment_shape_top_5'] = np.mean(fragment_shape[0:5])
        candidate_dict['fragment_shape_top_10'] = np.mean(fragment_shape[0:10])
        candidate_dict['fragment_shape'] = np.mean(fragment_shape)

        # ========= assembling individual fragment information =========

        if self.include_fragment_info:
            mz_library = self.fragments_mz_library[c_frag_start_idx:c_frag_stop_idx]      
            #print('fragment_mz ordered by correlation',c_fragments_mzs[fragment_order])
            #print(mz_library)
            mz_library = mz_library[c_fragments_order]
            #print(mz_library)
            mz_library = mz_library[intensity_mask]
            #print(mz_library)
            mz_library = mz_library[fragment_order]
            #print(mz_library)

            # this will be mz_predicted in the first round and mz_calibrated as soon as calibration has been locked in
            mz_used = c_fragments_mzs[fragment_order]
            mass_error = fragment_mass_err[fragment_order]
    
            mz_observed = mz_used + mass_error * 1e-6 * mz_used

            candidate_dict['fragment_mz_library_list'] = ';'.join([ f'{el:.4f}' for el in mz_library[:15]])
            candidate_dict['fragment_mz_observed_list'] = ';'.join([ f'{el:.4f}' for el in mz_observed[:15]])
            candidate_dict['mass_error_list'] = ';'.join([ f'{el:.3f}' for el in mass_error[:15]])
            #candidate_dict['mass_list'] = ';'.join([ f'{el:.3f}' for el in c_fragments_mzs[fragment_order][:10]])
            candidate_dict['intensity_list'] = ';'.join([ f'{el:.3f}' for el in fragment_intensity[fragment_order][:15]])
            candidate_dict['type_list'] = ';'.join([ f'{el:.3f}' for el in c_fragments_type[fragment_order][:15]])

        return [candidate_dict]
    
def unpack_fragment_info(candidate_scoring_df):

    all_precursor_indices = []
    all_fragment_mz_library = []
    all_fragment_mz_observed = []
    all_mass_errors = []
    all_intensities = []

    for precursor_index, fragment_mz_library_list, fragment_mz_observed_list, mass_error_list, intensity_list in zip(
        candidate_scoring_df['index'].values,
        candidate_scoring_df.fragment_mz_library_list.values, 
        candidate_scoring_df.fragment_mz_observed_list.values, 
        candidate_scoring_df.mass_error_list.values,
        candidate_scoring_df.intensity_list.values
    ):
        fragment_masses = [float(i) for i in fragment_mz_library_list.split(';')]
        all_fragment_mz_library += fragment_masses

        all_fragment_mz_observed += [float(i) for i in fragment_mz_observed_list.split(';')]
        
        all_mass_errors += [float(i) for i in mass_error_list.split(';')]
        all_intensities += [float(i) for i in intensity_list.split(';')]
        all_precursor_indices += [precursor_index] * len(fragment_masses)

    fragment_calibration_df = pd.DataFrame({
        'precursor_index': all_precursor_indices,
        'mz_library': all_fragment_mz_library,
        'mz_observed': all_fragment_mz_observed,
        'mass_error': all_mass_errors,
        'intensity': all_intensities
    })

    return fragment_calibration_df.dropna().reset_index(drop=True)

def assemble_fragment_df(fragment_index_tuple: Tuple[int,int], lib):

    """ Collects all fragments encoded for an precursor and places them in a dataframe
    
    """
    # FUTURE, reduce minimum fragment intensity treshold in library generation

    # get fragments from df based on precursor fragemnt slicing
    fragment_slice = slice(*fragment_index_tuple)

    fragment_mz_slice = lib.fragment_mz_df.iloc[fragment_slice]
    fragment_intensity_slice = lib.fragment_intensity_df.iloc[fragment_slice]

    num_fragments = fragment_index_tuple[1]-fragment_index_tuple[0]
    # FUTURE, same for whole lib
    num_cols = len(fragment_mz_slice.columns)
    start_index = 1

    #fragment series number
    fragment_series = np.tile(np.arange(start_index,start_index + num_fragments),num_cols)

    #fragment mzs
    fragment_mz_flat = fragment_mz_slice.values.flatten(order='F')

    #fragment intensity
    fragment_intensity_flat = fragment_intensity_slice.values.flatten(order='F')

    # fragment series and charge
    fragment_ion_type = []
    for column in fragment_mz_slice.columns:
        fragment_ion_type += [column]*num_fragments

    # assemble dataframe
    df = pd.DataFrame({'fragment_index': fragment_series,
                        'fragment_type': fragment_ion_type,
                        'fragment_mz': fragment_mz_flat,
                        'fragment_intensity': fragment_intensity_flat,
                        })

    df = df[df['fragment_intensity'] > 0]

    return df

def dense(dia_data: alphatims.bruker.TimsTOF, 
        frame_index_tuple: Tuple[int], 
        scan_index_tuple: Tuple[int], 
        quad_tuple: Tuple[Union[float, int]],
        mz_tuple_list: List[Tuple[float]], 
        background: str = 'ones'):
    """Retrive a list of mass slices with common frame index, scan index and quadrupole values

    Returns
    -------
    numpy.ndarray: Numpy array of shape (N, S, F) with N: number of mass slices, S: number of scans, F: number of frames

    """

    cycle_length = dia_data.cycle.shape[1]

    number_of_ions = len(mz_tuple_list)
    scan_size = scan_index_tuple[1]-scan_index_tuple[0] + 1
    frame_size = (frame_index_tuple[1]-frame_index_tuple[0])//cycle_length + 1

    dense_mat = np.ones([number_of_ions, scan_size, frame_size])

    for i, mz_tuple in enumerate(mz_tuple_list):
        
        data_df = dia_data[slice(*frame_index_tuple),slice(*scan_index_tuple),quad_tuple,slice(*mz_tuple)]
      
        cycle_index = data_df['frame_indices']//cycle_length
        cycle_index_local = cycle_index - frame_index_tuple[0] // cycle_length

        scan_index = data_df['scan_indices'].values
        scan_index_local = scan_index-scan_index_tuple[0]

        intensity_arr = data_df['intensity_values'].values

        for j, (cycle, scan, intensity) in enumerate(zip(cycle_index_local, scan_index_local, intensity_arr)):
            dense_mat[i,scan,cycle]+=intensity

    return dense_mat

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

class GlobalCalibration():

    # template column names
    precursor_calibration_targets = {
        'precursor_mz':('mz_predicted', 'mz_calibrated'),
        'mobility':('mobility_predicted', 'mobility_observed'),
        'rt':('rt_predicted', 'rt_observed'),
    }

    fragment_calibration_targets = {
        'fragment_mz':('mz_predicted', 'mz_calibrated'),
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
        for property, columns in self.precursor_calibration_targets.items():
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


    def plot(self, *args, save_name=None, **kwargs):
        logging.info('plotting calibration curves')
        ipp = 4
        calibration_df = self.extraction_plan.get_calibration_df()
    
        ax_labels = {'mz': ('mz','ppm'),
                    'rt': ('rt_pred','RT (seconds)'),
                    'mobility': ('mobility', 'mobility')}
        # check if 

        for run_mapping in self.extraction_plan.runs:

            run_df = calibration_df[calibration_df['raw_name'] == run_mapping['name']]
            print(len(run_df))

            estimators = self.estimators[run_mapping['index']]

            fig, axs = plt.subplots(ncols=len(estimators), nrows=1, figsize=(len(estimators)*ipp,ipp))
            for i, (property, estimator) in enumerate(estimators.items()):
                
                target_column, measured_column = self.prediction_targets[property]
                target_values = run_df[target_column].values
                measured_values = run_df[measured_column].values

                # plotting
                axs[i].set_title(property)

                calibration_space = np.linspace(np.min(target_values),np.max(target_values),1000)
                calibration_curve = estimator.predict(calibration_space)

                if property == 'mz':
                    measured_values = (target_values - measured_values) / target_values * 10**6
                    calibration_curve = (calibration_space - calibration_curve) / calibration_space * 10**6
                #axs[i].scatter(target_values,measured_values)
                density_scatter(target_values,measured_values,axs[i], s=2, **kwargs)
                axs[i].plot(calibration_space,calibration_curve, c='r')
                axs[i].set_xlabel(ax_labels[property][0])
                axs[i].set_ylabel(ax_labels[property][1])

            fig.suptitle(run_mapping['name'])
            fig.tight_layout()

            if save_name is not None:
                loaction = os.path.join(save_name, f"{run_mapping['name']}.png")
                fig.savefig(loaction, dpi=300)
            plt.show()


