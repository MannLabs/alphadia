# native imports

# alphadia imports
from alphadia.extraction import utils

# alpha family imports

# third party imports
import numpy as np
import numba as nb 


@nb.njit
def center_of_mass(
    single_dense_representation,
):

    frame_accumulation = 0
    scan_accumulation = 0
    intensity_accumulation = 0

    scans, frames = np.nonzero(single_dense_representation > 0)
    if len (scans) == 0:
        return 0, 0
    for scan, frame in zip(scans, frames):

        intensity = single_dense_representation[scan, frame]
        intensity_accumulation += intensity
        frame_accumulation += frame * intensity
        scan_accumulation += scan * intensity

    weighted_frame = frame_accumulation / intensity_accumulation if intensity_accumulation > 0 else 0
    weighted_scan = scan_accumulation / intensity_accumulation if intensity_accumulation > 0 else 0
    return weighted_scan, weighted_frame

@nb.njit
def center_of_mass_1d(
    dense_representation,
):
    scan = np.zeros(dense_representation.shape[0])
    frame = np.zeros(dense_representation.shape[0])
    for i in range(dense_representation.shape[0]):
        s, f = center_of_mass(dense_representation[i])
        scan[i], frame[i] = s, f
    return scan, frame

@nb.njit
def center_of_mass_2d(
    dense_representation,
):
    scan = np.zeros((dense_representation.shape[0], dense_representation.shape[1]))
    frame = np.zeros((dense_representation.shape[0], dense_representation.shape[1]))
    for i in range(dense_representation.shape[0]):
        for j in range(dense_representation.shape[1]):
            s, f = center_of_mass(dense_representation[i, j])
            scan[i, j], frame[i, j] = s, f
    return scan, frame

@nb.njit
def weighted_center_of_mass(
    single_dense_representation,
):
   
    intensity = [0.]

    scans, frames = np.nonzero(single_dense_representation > 0)

    if len (scans) == 0:
        return 0, 0, 0, 0

    for scan, frame in zip(scans, frames):
        intensity.append(single_dense_representation[scan, frame])

    intensity_arr = np.array(intensity)[1:]
    intensity_sum = np.sum(intensity_arr)

    scan_mean = np.sum(scans * intensity_arr) / intensity_sum if intensity_sum > 0 else 0
    frame_mean = np.sum(frames * intensity_arr) / intensity_sum if intensity_sum > 0 else 0
    frame_var_weighted = np.sum((frames - frame_mean) ** 2 * intensity_arr) 
    scan_var_weighted = np.sum((scans - scan_mean) ** 2 * intensity_arr)

    frame_var_weighted = frame_var_weighted / intensity_sum if intensity_sum > 0 else 0
    scan_var_weighted = scan_var_weighted / intensity_sum if intensity_sum > 0 else 0

    return scan_mean, frame_mean, scan_var_weighted, frame_var_weighted

@nb.njit
def weighted_center_of_mass_1d(
    dense_representation,
):
    scan = np.zeros(dense_representation.shape[0])
    frame = np.zeros(dense_representation.shape[0])
    frame_var_weighted = np.zeros(dense_representation.shape[0])
    scan_var_weighted = np.zeros(dense_representation.shape[0])

    for i in range(dense_representation.shape[0]):
        scan[i], frame[i], scan_var_weighted[i], frame_var_weighted[i] = weighted_center_of_mass(dense_representation[i])
    return scan, frame, scan_var_weighted, frame_var_weighted

@nb.njit
def weighted_center_of_mass_2d(
    dense_representation,
):
    scan = np.zeros((dense_representation.shape[0], dense_representation.shape[1]))
    frame = np.zeros((dense_representation.shape[0], dense_representation.shape[1]))
    scan_var_weighted = np.zeros((dense_representation.shape[0], dense_representation.shape[1]))
    frame_var_weighted = np.zeros((dense_representation.shape[0], dense_representation.shape[1]))


    for i in range(dense_representation.shape[0]):
        for j in range(dense_representation.shape[1]):
            scan[i, j], frame[i, j], scan_var_weighted[i,j], frame_var_weighted[i,j] = weighted_center_of_mass(dense_representation[i, j])
    return scan, frame, scan_var_weighted, frame_var_weighted


@nb.njit
def weighted_center_mean(
    single_dense_representation,
    scan_center,
    frame_center
):  
    
    values = 0
    weights = 0

    scans, frames = np.nonzero(single_dense_representation > 0)
    if len (scans) == 0:
        return 0

    for scan, frame in zip(scans, frames):

        value = single_dense_representation[scan, frame]
        distance = np.sqrt((scan - scan_center)**2 + (frame - frame_center)**2)
        weight = np.exp(-0.1 * distance)
        values += value * weight
        weights += weight

    return values / weights if weights > 0 else 0

@nb.njit
def weighted_center_mean_1d(
    dense_representation,
    scan_center,
    frame_center
    ):

    values = np.zeros(dense_representation.shape[0])

    for i in range(dense_representation.shape[0]):
        values[i] = weighted_center_mean(dense_representation[i], scan_center[i], frame_center[i])
    return values


@nb.njit
def weighted_center_mean_2d(
    dense_representation,
    scan_center,
    frame_center
    ):
    values = np.zeros((dense_representation.shape[0], dense_representation.shape[1]))
    for i in range(dense_representation.shape[0]):
        for j in range(dense_representation.shape[1]):
            values[i,j] = weighted_center_mean(dense_representation[i,j], scan_center[i,j], frame_center[i,j])

    return values

@nb.njit
def center_sum(
    single_dense_representation,
    scan_center,
    frame_center,
    window_radius = 2
):  
    lower_scan_limit = max(np.floor(scan_center - window_radius), 0)
    upper_scan_limit = min(np.ceil(scan_center + window_radius), single_dense_representation.shape[0])
    lower_frame_limit = max(np.floor(frame_center - window_radius), 0)
    upper_frame_limit = min(np.ceil(frame_center + window_radius), single_dense_representation.shape[1])

    window = single_dense_representation[
        int(lower_scan_limit):int(upper_scan_limit),
        int(lower_frame_limit):int(upper_frame_limit)
    ]
    s = window.shape[0] * window.shape[1]

    intensity = window.sum()
    fraction_nonzero = (window > 0).sum() / s

    return intensity, fraction_nonzero

@nb.njit
def center_sum_1d(
    dense_representation,
    scan_center,
    frame_center,
    window_radius = 2
):
    intensity = np.zeros(dense_representation.shape[0])
    fraction_nonzero = np.zeros(dense_representation.shape[0])
    for i in range(dense_representation.shape[0]):
        s, f = center_sum(dense_representation[i], scan_center[0], frame_center[0], window_radius=window_radius)
        intensity[i], fraction_nonzero[i] = s, f
    return intensity, fraction_nonzero

@nb.njit
def center_sum_2d(
    dense_representation,
    scan_center,
    frame_center,
    window_radius = 2
):
    intensity = np.zeros((dense_representation.shape[0], dense_representation.shape[1]))
    fraction_nonzero = np.zeros((dense_representation.shape[0], dense_representation.shape[1]))
    for i in range(dense_representation.shape[0]):
        for j in range(dense_representation.shape[1]):
            s, f = center_sum(dense_representation[i, j], scan_center[i, j], frame_center[i, j], window_radius=window_radius)
            intensity[i, j], fraction_nonzero[i, j] = s, f
    return intensity, fraction_nonzero




@nb.njit
def masked_mean_a0(array, mask):
    """
    takes an array of shape (a, b) and a mask of shape (a, b)
    and returns an array of shape (b) where each element is the mean of the corresponding masked column in the array
    """
    mean = np.zeros(mask.shape[1])
    for i in range(array.shape[1]):
        masked_array = array[mask[:,i],i]
        if len(masked_array) > 0:
            mean[i] = np.mean(masked_array)
        else:
            mean[i] = 0
    return mean


@nb.njit
def masked_mean_a1(array, mask):
    """
    takes an array of shape (a, b) and a mask of shape (a, b)
    and returns an array of shape (a) where each element is the mean of the corresponding masked row in the array
    """
    mean = np.zeros(array.shape[0])
    for i in range(array.shape[0]):
        masked_array = array[i][mask[i]]
        if len(masked_array) > 0:
            mean[i] = np.mean(masked_array)
        else:
            mean[i] = 0
    return mean



float_array = nb.types.float32[:]

@nb.njit
def cosine_similarity_a1(template_intensity, fragments_intensity):

    fragment_norm = np.sqrt(np.sum(np.power(fragments_intensity,2),axis=-1))
    template_norm = np.sqrt(np.sum(np.power(template_intensity,2),axis=-1))

    div = (fragment_norm * template_norm) + 0.0001

    return np.sum(fragments_intensity * template_intensity,axis=-1) / div

@nb.njit
def frame_profile_2d(x):
    return np.sum(x, axis=2)

@nb.njit
def frame_profile_1d(x):
    return np.sum(x, axis=1)

@nb.njit
def scan_profile_2d(x):
    return np.sum(x, axis=3)

@nb.njit
def scan_profile_1d(x):
    return np.sum(x, axis=2)

@nb.njit
def or_envelope_1d(x):
    res = x.copy()
    for a0 in range(x.shape[0]):
        for i in range(1, x.shape[1] - 1):
            if (x[a0, i] < x[a0, i-1]) or (x[a0, i] < x[a0, i+1]):
                res[a0, i] = (x[a0, i-1] + x[a0, i+1]) / 2
    return res

@nb.njit
def or_envelope_2d(x):
    res = x.copy()
    for a0  in range(x.shape[0]):
        for a1 in range(x.shape[1]):
            for i in range(1, x.shape[2] - 1):
                if (x[a0, a1, i] < x[a0, a1, i-1]) or (x[a0, a1, i] < x[a0, a1, i+1]):
                    res[a0, a1, i] = (x[a0, a1, i-1] + x[a0, a1, i+1]) / 2
    return res

@nb.njit
def weighted_mean_a1(array, weight_mask):
    """
    takes an array of shape (a, b) and a mask of shape (a, b)
    and returns an array of shape (a) where each element is the weighted mean of the corresponding masked row in the array.

    Parameters
    ----------

    array: np.ndarray
        array of shape (a, b)

    weight_mask: np.ndarray
        array of shape (a, b)

    Returns
    -------
    np.ndarray
        array of shape (a)
    """

    mask = weight_mask > 0

    mean = np.zeros(array.shape[0])
    for i in range(array.shape[0]):
        masked_array = array[i][mask[i]]
        if len(masked_array) > 0:
            local_weight_mask = weight_mask[i][mask[i]]/np.sum(weight_mask[i][mask[i]])
            mean[i] = np.sum(masked_array*local_weight_mask)
        else:
            mean[i] = 0
    return mean

@nb.njit
def weighted_correlation(
    fragment_profile,
    template_profile,
    mask
):
    n_fragemts = fragment_profile.shape[0]
    n_observations = fragment_profile.shape[1]

    weighted_fragment_correlation = np.zeros((n_fragemts, n_observations))
    template_correlation = np.zeros((n_fragemts, n_observations))

    for i_observations in range(n_observations):
        correlation = np.corrcoef(fragment_profile[:,i_observations], template_profile[0,i_observations])

        # expand 1d mask to two dimensional mask
        mask_2d = np.expand_dims(mask[:,i_observations], axis=-1)*mask[:, i_observations]
        np.fill_diagonal(mask_2d, 0)
        weighted_fragment_correlation[:,i_observations] = weighted_mean_a1(correlation[:-1,:-1], mask_2d)
        
        mask_bool = mask[:,i_observations] > 0
        template_correlation[:,i_observations] = correlation[-1,:-1]
        template_correlation[:,i_observations][~mask_bool] = 0
   
    return weighted_fragment_correlation, template_correlation

@nb.njit
def weighted_correlation_single(
    fragment_profile,
    template_profile,
    mask
):
    n_fragemts = fragment_profile.shape[0]
    n_observations = fragment_profile.shape[1]

    weighted_fragment_correlation = np.zeros((n_fragemts, n_observations))
    template_correlation = np.zeros((n_fragemts, n_observations))

    for i_observations in range(n_observations):
        # TODO: if fragment profiles are empty, lots of NaNs are produced
        correlation = np.corrcoef(fragment_profile[:,i_observations], template_profile[i_observations])
        
        # expand 1d mask to two dimensional mask
        mask_2d = np.expand_dims(mask[:,i_observations], axis=-1)*mask[:, i_observations]
        np.fill_diagonal(mask_2d, 0)
        weighted_fragment_correlation[:,i_observations] = weighted_mean_a1(correlation[:-1,:-1], mask_2d)
        
        mask_bool = mask[:,i_observations] > 0
        template_correlation[:,i_observations] = correlation[-1,:-1]
        template_correlation[:,i_observations][~mask_bool] = 0
   
    return weighted_fragment_correlation, template_correlation

#@nb.njit
def weighted_correlation_double(
    fragment_profile,
    template_profile,
    mask
):
    n_fragemts = fragment_profile.shape[0]
    n_observations = fragment_profile.shape[1]

    weighted_fragment_correlation = np.zeros((n_fragemts, n_observations))
    template_correlation = np.zeros((n_fragemts, n_observations))

    for i_observations in range(n_observations):
        # TODO: if fragment profiles are empty, lots of NaNs are produced
        correlation = np.corrcoef(fragment_profile[:,i_observations], template_profile[:,i_observations])
        
        # expand 1d mask to two dimensional mask
        mask_2d = np.expand_dims(mask[:,i_observations], axis=-1)*mask[:, i_observations]
        np.fill_diagonal(mask_2d, 0)
        weighted_fragment_correlation[:,i_observations] = weighted_mean_a1(correlation[:-1,:-1], mask_2d)
        
        mask_bool = mask[:,i_observations] > 0
        template_correlation[:,i_observations] = correlation[-1,:-1]
        template_correlation[:,i_observations][~mask_bool] = 0
   
    return weighted_fragment_correlation, template_correlation

@nb.njit
def build_features(
        dense_fragments,
        dense_precursors,
        template,
        isotope_intensity,
        isotope_mz,
        fragments
    ):

    features = nb.typed.Dict.empty(
        key_type=nb.types.unicode_type,
        value_type=nb.types.float32
    )

    fragment_features = nb.typed.Dict.empty(
        key_type=nb.types.unicode_type,
        value_type=float_array
    )


    n_precursors = isotope_intensity.shape[0]
    n_isotopes = isotope_intensity.shape[1]
    n_fragments = dense_fragments.shape[1]
    n_observations = dense_fragments.shape[2]

    # ============= PRECURSOR FEATURES =============

    # all metrics are calculated with respect to the expected center of mass, which is calculated from the template
    # (n_precursor, n_observations)
    (
        expected_scan_center, 
        expected_frame_center, 
        expected_scan_variance, 
        expected_frame_variance
    ) = weighted_center_of_mass_2d(template)

    # expand the expected center of mass to n_precursor * n_isotopes
    # (n_precursor, n_observations)
    p_expected_scan_center = utils.tile(expected_scan_center, n_precursors*n_isotopes).reshape(n_precursors*n_isotopes, -1)
    p_expected_frame_center = utils.tile(expected_frame_center, n_precursors*n_isotopes).reshape(n_precursors*n_isotopes, -1)

    # (n_precursor, n_isotopes)
    observed_precursor_mz = weighted_center_mean_2d(
        dense_precursors[1],
        p_expected_scan_center,
        p_expected_frame_center
    ).reshape(n_precursors, n_isotopes)
    
    # create weight matrix to exclude empty isotopes
    i, j = np.nonzero(observed_precursor_mz == 0)
    precursor_weights = isotope_intensity.copy()
    for i, j in zip(i, j):
        precursor_weights[i, j] = 0

    observed_precursor_intensity = weighted_center_mean_2d(
        dense_precursors[0],
        p_expected_scan_center,
        p_expected_frame_center
    ).reshape(n_precursors, n_isotopes)

    # sum precursor
    sum_precursor_intensity = np.sum(np.sum(dense_precursors[0], axis=-1), axis=-1).astype(np.float32)
    sum_fragment_intensity = np.sum(np.sum(dense_fragments[0], axis=-1), axis=-1).astype(np.float32)
    

    # (n_precursor, n_isotopes) 
    mass_error_array = (observed_precursor_mz - isotope_mz) / isotope_mz * 1e6

    # (n_precursor)
    mass_error = weighted_mean_a1(mass_error_array, precursor_weights)
    total_precursor_intensity = np.sum(observed_precursor_intensity * isotope_intensity, axis=-1)

    features['precursor_mass_error'] = np.mean(mass_error.astype(np.float32))
    features['mz_library'] = isotope_mz[0,0]
    features['mz_observed'] = isotope_mz[0,0] + features['precursor_mass_error'] * 1e-6 * isotope_mz[0,0]
    features['precursor_isotope_correlation'] = np.corrcoef(isotope_intensity.flatten(), observed_precursor_intensity.flatten())[0,1]
    features['sum_precursor_intensity'] = np.log10(total_precursor_intensity[0])

    # ============= FRAGMENT FEATURES =============
    
    # expand the expected center of mass to the number of fragments
    # (n_fragments, n_observations)
    f_expected_scan_center = utils.tile(expected_scan_center, n_fragments).reshape(n_fragments, -1)
    f_expected_frame_center = utils.tile(expected_frame_center, n_fragments).reshape(n_fragments, -1)

    # create fragment masks for filtering
    # (n_fragments, n_observations)
    total_fragment_intensity = np.sum(np.sum(dense_fragments[0], axis=-1), axis=-1)
    total_template_intensity = np.sum(np.sum(template, axis=-1), axis=-1)
    fragment_mask_2d = total_fragment_intensity > 0
    fragment_mask_1d = np.sum(fragment_mask_2d, axis=-1) > 0

    # get the observed fragment mz and intensity
    # (n_fragments, n_observations)
    observed_fragment_mz = weighted_center_mean_2d(
        dense_fragments[1],
        f_expected_scan_center,
        f_expected_frame_center
    )
    
    # (n_fragments, n_observations)
    observed_fragment_intensity, observed_fragment_nonzero = center_sum_2d(
        dense_fragments[0],
        f_expected_scan_center,
        f_expected_frame_center
    )
    
    # get rid of the observation dimension by performing a masked mean
    # (n_fragments)
    observed_fragment_mz_mean = masked_mean_a1(observed_fragment_mz, fragment_mask_2d)
    observed_fragment_intensity_mean = masked_mean_a1(observed_fragment_intensity, fragment_mask_2d)
    observed_fragment_nonzero = masked_mean_a1(observed_fragment_nonzero, fragment_mask_2d)

    peak_fragment_mask_2d = observed_fragment_intensity > 0
    peak_fragment_mask_1d = np.sum(peak_fragment_mask_2d, axis=-1) > 0

    mass_error = (observed_fragment_mz_mean - fragments.mz) / fragments.mz * 1e6

    features['n_fragments_matched'] = np.sum(observed_fragment_intensity_mean > 0)
    features['n_fragments'] = len(fragment_mask_1d)
    features['fraction_fragments'] = features['n_fragments_matched']/features['n_fragments']


    features['intensity_correlation'] = np.corrcoef(fragments.intensity[fragment_mask_1d], observed_fragment_intensity_mean[fragment_mask_1d].astype(np.float32))[0,1]
    features['sum_fragment_intensity'] = np.log10(np.sum(observed_fragment_intensity_mean[fragment_mask_1d]))
    features['mean_fragment_intensity'] = np.log10(np.mean(observed_fragment_intensity_mean[fragment_mask_1d]))
    features['mean_fragment_nonzero'] = np.mean(observed_fragment_nonzero[fragment_mask_1d])

    features['n_observations'] = float(n_observations)

    features['mean_observation_score'] = 0
    features['var_observation_score'] = 1

    if np.sum(peak_fragment_mask_1d) > 0:
        if n_observations > 1:
            observation_score = cosine_similarity_a1(total_template_intensity, observed_fragment_intensity[peak_fragment_mask_1d]).astype(np.float32)
            features['mean_observation_score'] = np.mean(observation_score)
            features['var_observation_score'] = np.var(observation_score)

    fragment_features['mz_library'] = fragments.mz_library[fragment_mask_1d]
    fragment_features['mz_observed'] = observed_fragment_mz_mean[fragment_mask_1d].astype(np.float32)
    fragment_features['mass_error'] = mass_error[fragment_mask_1d].astype(np.float32)
    fragment_features['intensity'] = observed_fragment_intensity_mean[fragment_mask_1d].astype(np.float32)
    fragment_features['type'] = fragments.type[fragment_mask_1d].astype(np.float32)

    

    return features, fragment_features

@nb.njit()
def precursor_features(
    isotope_mz: np.ndarray,
    isotope_intensity: np.ndarray,
    dense_precursors: np.ndarray,
    observation_importance,
    template: np.ndarray,
    ):

    feature_dict = nb.typed.Dict.empty(
        key_type=nb.types.unicode_type,
        value_type=nb.types.float32
    )

    n_isotopes = isotope_intensity.shape[0]
    n_observations = dense_precursors.shape[2]

    # ============= PRECURSOR FEATURES =============

    #(1, n_observations)
    observation_importance_reshaped = observation_importance.reshape(1, -1)

    # (n_isotopes, n_observations)
    sum_precursor_intensity = np.sum(np.sum(dense_precursors[0], axis=-1), axis=-1).astype(np.float32)

    # (n_isotopes)
    weighted_sum_precursor_intensity = np.sum(sum_precursor_intensity * observation_importance_reshaped, axis=-1).astype(np.float32)

    

    feature_dict['mono_ms1_intensity'] = weighted_sum_precursor_intensity[0]
    feature_dict['top_ms1_intensity'] = weighted_sum_precursor_intensity[np.argmax(isotope_intensity)]
    feature_dict['sum_ms1_intensity'] = np.sum(weighted_sum_precursor_intensity)
    feature_dict['weighted_ms1_intensity'] = np.sum(weighted_sum_precursor_intensity*isotope_intensity)

    expected_scan_center = utils.tile(dense_precursors.shape[3], n_isotopes*n_observations).reshape(n_isotopes, -1)
    expected_frame_center = utils.tile(dense_precursors.shape[2], n_isotopes*n_observations).reshape(n_isotopes, -1)

    # (n_isotopes)
    observed_precursor_height = weighted_center_mean_2d(
        dense_precursors[0],
        expected_scan_center,
        expected_frame_center
    )[:,0]

    # (n_isotopes)
    observed_precursor_mz = weighted_center_mean_2d(
        dense_precursors[1],
        expected_scan_center,
        expected_frame_center
    )[:,0]

    mz_mask = observed_precursor_mz > 0

    # (n_isotopes) 
    mass_error_array = (observed_precursor_mz - isotope_mz) / isotope_mz * 1e6
    weighted_mass_error = np.sum(mass_error_array[mz_mask] * isotope_intensity[mz_mask])

    feature_dict['weighted_mass_deviation'] = weighted_mass_error
    feature_dict['weighted_mass_error'] = np.abs(weighted_mass_error)
    feature_dict['mz_library'] = isotope_mz[0]
    feature_dict['mz_observed'] = isotope_mz[0] + weighted_mass_error * 1e-6 * isotope_mz[0]

    feature_dict['mono_ms1_height'] = observed_precursor_height[0]
    feature_dict['top_ms1_height'] = observed_precursor_height[np.argmax(isotope_intensity)]
    feature_dict['sum_ms1_height'] = np.sum(observed_precursor_height)
    feature_dict['weighted_ms1_height'] = np.sum(observed_precursor_height*isotope_intensity)

    feature_dict['isotope_intensity_correlation'] = np.corrcoef(isotope_intensity, np.sum(sum_precursor_intensity, axis=-1))[0,1]
    feature_dict['isotope_height_correlation'] = np.corrcoef(isotope_intensity, observed_precursor_height)[0,1]

    return feature_dict

@nb.njit()
def location_features(
    jit_data,
    scan_start,
    scan_stop,
    scan_center,
    frame_start,
    frame_stop,
    frame_center,
    ):

    feature_dict = nb.typed.Dict.empty(
        key_type=nb.types.unicode_type,
        value_type=nb.types.float32
    )

    

    feature_dict['base_width_mobility'] = jit_data.mobility_values[scan_start] - jit_data.mobility_values[scan_stop-1 ]
    feature_dict['base_width_rt'] = jit_data.rt_values[frame_stop-1] - jit_data.rt_values[frame_start]
    feature_dict['rt_observed'] = jit_data.rt_values[frame_center]
    feature_dict['mobility_observed'] = jit_data.mobility_values[scan_center]

    return feature_dict

nb_float32_array = nb.types.Array(nb.types.float32, 1, 'C')

@nb.njit()
def fragment_features(
    dense_fragments: np.ndarray,
    observation_importance: np.ndarray,
    template: np.ndarray,
    fragments: np.ndarray,
):  
    
    feature_dict = nb.typed.Dict.empty(
        key_type=nb.types.unicode_type,
        value_type=nb.types.float32
    )

    fragment_feature_dict = nb.typed.Dict.empty(
        key_type=nb.types.unicode_type,
        value_type=float_array
    )
    
    n_observations = observation_importance.shape[0]
    n_fragments = dense_fragments.shape[1]
    feature_dict['n_observations'] = float(n_observations)

    #(1, n_observations)
    observation_importance_reshaped = observation_importance.reshape(1, -1)

    # (n_fragments)
    fragment_intensity_norm = fragments.intensity / np.sum(fragments.intensity)   

    if n_fragments == 0:
        print(n_fragments)

    # (n_observations)
    (
        expected_scan_center, 
        expected_frame_center, 
        expected_scan_variance, 
        expected_frame_variance
    ) = weighted_center_of_mass_1d(template)

    # expand the expected center of mass to the number of fragments
    # (n_fragments, n_observations)
    f_expected_scan_center = utils.tile(expected_scan_center, n_fragments).reshape(n_fragments, -1)
    f_expected_frame_center = utils.tile(expected_frame_center, n_fragments).reshape(n_fragments, -1)

    # create fragment masks for filtering
    # (n_fragments, n_observations)
    sum_fragment_intensity = np.sum(np.sum(dense_fragments[0], axis=-1), axis=-1)
    # (n_fragments)
    observed_fragment_intensity = np.sum(sum_fragment_intensity*observation_importance_reshaped, axis=-1)

    # (n_observations)
    sum_template_intensity = np.sum(np.sum(template, axis=-1), axis=-1)

    # get the observed fragment mz and intensity
    # (n_fragments, n_observations)
    observed_fragment_mz = weighted_center_mean_2d(
        dense_fragments[1],
        f_expected_scan_center,
        f_expected_frame_center
    )
    # (n_fragments)
    observed_fragment_mz_mean = np.sum(observed_fragment_mz*observation_importance_reshaped, axis=-1)
    
    # (n_fragments, n_observations)
    o_fragment_height = weighted_center_mean_2d(
        dense_fragments[0],
        f_expected_scan_center,
        f_expected_frame_center
    )
    # (n_fragments)
    observed_fragment_height = np.sum(o_fragment_height*observation_importance_reshaped, axis=-1)

    if np.sum(observed_fragment_intensity) == 0.0:
        feature_dict['intensity_correlation'] = 0.0
    else:
        feature_dict['intensity_correlation'] = np.corrcoef(
            observed_fragment_intensity,
            fragment_intensity_norm
        )[0,1]
    
    if np.sum(observed_fragment_height) == 0.0:
        feature_dict['height_correlation'] = 0.0
    else:
        feature_dict['height_correlation'] = np.corrcoef(
            observed_fragment_height,
            fragment_intensity_norm
        )[0,1]

    feature_dict['intensity_fraction'] = np.sum(observed_fragment_intensity > 0.0) / n_fragments
    feature_dict['height_fraction'] = np.sum(observed_fragment_height > 0.0) / n_fragments

    feature_dict['intensity_fraction_weighted'] = np.sum(fragment_intensity_norm[observed_fragment_intensity > 0.0])
    feature_dict['height_fraction_weighted'] = np.sum(fragment_intensity_norm[observed_fragment_height > 0.0])

    fragment_mask = observed_fragment_intensity > 0

    if np.sum(fragment_mask) == 0:
        feature_dict['mean_observation_score'] = 0.0
    else:

        sum_template_intensity_expanded = sum_template_intensity.reshape(1, -1)
        observation_score = cosine_similarity_a1(
            sum_template_intensity_expanded, 
            sum_fragment_intensity[fragment_mask]
            
            ).astype(np.float32)
        feature_dict['mean_observation_score'] = np.mean(observation_score)

    

    # ============= FRAGMENT TYPE FEATURES =============

    b_ion_mask = fragments.type == 98
    y_ion_mask = fragments.type == 121

    weighted_b_ion_intensity = observed_fragment_intensity[b_ion_mask]
    weighted_y_ion_intensity = observed_fragment_intensity[y_ion_mask]

    feature_dict['sum_b_ion_intensity'] = np.log(np.sum(weighted_b_ion_intensity)+1) if len(weighted_b_ion_intensity) > 0 else 0.0
    feature_dict['sum_y_ion_intensity'] = np.log(np.sum(weighted_y_ion_intensity)+1) if len(weighted_y_ion_intensity) > 0 else 0.0
    feature_dict['diff_b_y_ion_intensity'] =feature_dict['sum_b_ion_intensity'] - feature_dict['sum_y_ion_intensity']

    # ============= FRAGMENT FEATURES =============

    mz_mask = observed_fragment_mz_mean > 0.0
    
    mass_error = (observed_fragment_mz_mean - fragments.mz) / fragments.mz * 1e6

    fragment_feature_dict['mz_library'] = fragments.mz_library
    fragment_feature_dict['mz_observed'] = observed_fragment_mz_mean[mz_mask].astype(np.float32)
    fragment_feature_dict['mass_error'] = mass_error[mz_mask].astype(np.float32)
    fragment_feature_dict['height'] = observed_fragment_height[mz_mask].astype(np.float32)
    fragment_feature_dict['intensity'] = observed_fragment_intensity[mz_mask].astype(np.float32)
    fragment_feature_dict['type'] = fragments.type[mz_mask].astype(np.float32)
    
    return feature_dict, fragment_feature_dict

@nb.njit
def profile_features(
        dia_data,
        fragment_intensity,
        fragment_type,
        observation_importance,
        fragments_scan_profile,
        fragments_frame_profile,
        template_scan_profile,
        template_frame_profile, 
        scan_start,
        scan_stop,
        frame_start,
        frame_stop,
    ):

    n_observations = len(observation_importance)

    feature_dict = nb.typed.Dict.empty(
        key_type=nb.types.unicode_type,
        value_type=nb.types.float32
    )

    fragment_mask_2d = np.sum(fragments_scan_profile, axis=-1) > 0
    fragment_mask_1d = np.sum(np.sum(fragments_scan_profile, axis=-1), axis=-1) > 0
    
    fragment_weights_2d = fragment_mask_2d.astype(np.int8)* np.expand_dims(fragment_intensity, axis=-1)

    if np.sum(fragment_mask_1d) < 3:
        feature_dict['fragment_scan_correlation'] = 0.0
        feature_dict['top3_scan_correlation'] = 0.0
        feature_dict['fragment_frame_correlation'] = 0.0
        feature_dict['top3_frame_correlation'] = 0.0
        feature_dict['template_scan_correlation'] = 0.0
        feature_dict['template_frame_correlation'] = 0.0
        feature_dict['template_frame_correlation'] = 0.0
        feature_dict['top3_b_ion_correlation'] = 0.0
        feature_dict['top3_y_ion_correlation'] = 0.0
        feature_dict['cycle_fwhm'] = 0.0
        feature_dict['mobility_fwhm'] = 0.0

        return feature_dict
    
    non_zero_fragment_norm = fragment_intensity[fragment_mask_1d]/np.sum(fragment_intensity[fragment_mask_1d])
    fragment_idx_sorted = np.argsort(non_zero_fragment_norm)[::-1]

    # (n_fragments, n_observations)
    fragment_scan_correlation, template_scan_correlation = weighted_correlation_single(
        fragments_scan_profile,
        template_scan_profile.reshape(n_observations, -1),
        fragment_weights_2d,
    )

    # (n_fragments, n_observations)
    fragment_frame_correlation, template_frame_correlation = weighted_correlation_single(
        fragments_frame_profile,
        template_frame_profile.reshape(n_observations, -1),
        fragment_weights_2d,
    )

    fragment_scan_mean_list = np.sum(fragment_scan_correlation[fragment_mask_1d] * observation_importance.reshape(1,-1), axis = -1)
    fragment_scan_mean_agg = np.sum(fragment_scan_mean_list * non_zero_fragment_norm)
    feature_dict['fragment_scan_correlation'] = fragment_scan_mean_agg
    feature_dict['top3_scan_correlation'] = fragment_scan_mean_list[fragment_idx_sorted[:3]].mean()

    fragment_frame_mean_list = np.sum(fragment_frame_correlation[fragment_mask_1d] * observation_importance.reshape(1,-1), axis = -1)
    fragment_frame_mean_agg = np.sum(fragment_frame_mean_list * non_zero_fragment_norm)
    feature_dict['fragment_frame_correlation'] = fragment_frame_mean_agg
    feature_dict['top3_frame_correlation'] = fragment_frame_mean_list[fragment_idx_sorted[:3]].mean()

    template_scan_mean_list = np.sum(template_scan_correlation[fragment_mask_1d] * observation_importance.reshape(1,-1), axis = -1)
    template_scan_mean_agg = np.sum(template_scan_mean_list * non_zero_fragment_norm)
    feature_dict['template_scan_correlation'] = template_scan_mean_agg

    template_frame_mean_list = np.sum(template_frame_correlation[fragment_mask_1d] * observation_importance.reshape(1,-1), axis = -1)
    template_frame_mean_agg = np.sum(template_frame_mean_list * non_zero_fragment_norm)
    feature_dict['template_frame_correlation'] = template_frame_mean_agg

    # ============= FRAGMENT TYPE FEATURES =============

    fragment_type_filtered = fragment_type[fragment_mask_1d]
    b_ion_mask = fragment_type_filtered == 98
    y_ion_mask = fragment_type_filtered == 121

    if len(fragment_type_filtered[b_ion_mask]) > 0:
        feature_dict['top3_b_ion_correlation'] = fragment_frame_mean_list[b_ion_mask][fragment_idx_sorted[:3]].mean()
    else:
        feature_dict['top3_b_ion_correlation'] = 0.0

    if len(fragment_type_filtered[y_ion_mask]) > 0:
        feature_dict['top3_y_ion_correlation'] = fragment_frame_mean_list[y_ion_mask][fragment_idx_sorted[:3]].mean()
    else:
        feature_dict['top3_y_ion_correlation'] = 0.0

    # ============= FWHM =============

    # (n_fragments, n_observations)
    cycle_fwhm = np.zeros((
        fragments_frame_profile.shape[0], 
        fragments_frame_profile.shape[1], ),
        dtype=np.float32
    )

    rt_width = dia_data.rt_values[frame_stop-1] - dia_data.rt_values[frame_start]

    for i_fragment in range(fragments_frame_profile.shape[0]):
        for i_observation in range(fragments_frame_profile.shape[1]):
            max_intensity = np.max(fragments_frame_profile[i_fragment, i_observation])
            half_max = max_intensity / 2
            n_values_above = np.sum(fragments_frame_profile[i_fragment, i_observation] > half_max)
            fraction_above = n_values_above / len(fragments_frame_profile[i_fragment, i_observation])

            cycle_fwhm[i_fragment, i_observation] = fraction_above * rt_width

    cycle_fwhm_mean_list = np.sum(cycle_fwhm * observation_importance.reshape(1,-1), axis = -1)
    cycle_fwhm_mean_agg = np.sum(cycle_fwhm_mean_list[fragment_mask_1d] * non_zero_fragment_norm)

    feature_dict['cycle_fwhm'] = cycle_fwhm_mean_agg    

    # (n_fragments, n_observations)
    mobility_fwhm = np.zeros((
        fragments_scan_profile.shape[0], 
        fragments_scan_profile.shape[1], ),
        dtype=np.float32
    )

    mobility_width = dia_data.mobility_values[scan_start] - dia_data.mobility_values[scan_stop-1]

    for i_fragment in range(fragments_scan_profile.shape[0]):
        for i_observation in range(fragments_scan_profile.shape[1]):
            max_intensity = np.max(fragments_scan_profile[i_fragment, i_observation])
            half_max = max_intensity / 2
            n_values_above = np.sum(fragments_scan_profile[i_fragment, i_observation] > half_max)
            fraction_above = n_values_above / len(fragments_scan_profile[i_fragment, i_observation])

            mobility_fwhm[i_fragment, i_observation] = fraction_above * mobility_width

    mobility_fwhm_mean_list = np.sum(mobility_fwhm * observation_importance.reshape(1,-1), axis = -1)
    mobility_fwhm_mean_agg = np.sum(mobility_fwhm_mean_list[fragment_mask_1d] * non_zero_fragment_norm)

    feature_dict['mobility_fwhm'] = mobility_fwhm_mean_agg

    return feature_dict

@nb.njit
def reference_features(
    reference_observation_importance,
    reference_fragments_scan_profile,
    reference_fragments_frame_profile,
    reference_template_scan_profile,
    reference_template_frame_profile,
    observation_importance,
    fragments_scan_profile,
    fragments_frame_profile,
    template_scan_profile,
    template_frame_profile,
    fragment_lib_intensity,
    ):

    feature_dict = nb.typed.Dict.empty(
        key_type=nb.types.unicode_type,
        value_type=nb.types.float32
    )

    n_observation = reference_observation_importance.shape[0]
    n_fragments = reference_fragments_scan_profile.shape[0]
    fragment_idx_sorted = np.argsort(fragment_lib_intensity)[::-1]

    if reference_fragments_scan_profile.shape[0] == 0 or fragments_scan_profile.shape[0] == 0:
        feature_dict['reference_intensity_correlation'] = 0

        feature_dict['mean_reference_scan_cosine'] = 0
        feature_dict['top3_reference_scan_cosine'] = 0
        feature_dict['mean_reference_frame_cosine'] = 0
        feature_dict['top3_reference_frame_cosine'] = 0
        feature_dict['mean_reference_template_scan_cosine'] = 0
        feature_dict['top3_reference_template_scan_cosine'] = 0
        feature_dict['mean_reference_template_frame_cosine'] = 0
        feature_dict['top3_reference_template_frame_cosine'] = 0

        return feature_dict

    # ============= Fragment Intensity =============

    reference_fragment_intensity = np.sum(np.sum(reference_fragments_scan_profile, axis = -1) * reference_observation_importance.reshape(1,-1), axis = -1)
    fragment_intensity = np.sum(np.sum(fragments_scan_profile, axis = -1)* observation_importance.reshape(1,-1), axis = -1)

    total_fragment_intensity = np.sum(fragment_intensity)

    reference_intensity_correlation = 0

    if total_fragment_intensity > 0:
        reference_intensity_correlation = correlation = np.corrcoef(
            reference_fragment_intensity,
            fragment_intensity
        )[0,1]

    feature_dict['reference_intensity_correlation'] = reference_intensity_correlation

    # ============= Fragment Profile =============

    reference_scan_profile = np.sum(reference_fragments_scan_profile*reference_observation_importance.reshape(1,-1,1), axis = 1)
    scan_profile = np.sum(fragments_scan_profile*observation_importance.reshape(1,-1,1), axis = 1)

    scan_similarity = cosine_similarity_a1(
        reference_scan_profile,
        scan_profile
    )

    feature_dict['mean_reference_scan_cosine'] = np.mean(scan_similarity)
    feature_dict['top3_reference_scan_cosine'] = scan_similarity[fragment_idx_sorted[:3]].mean()

    reference_frame_profile = np.sum(reference_fragments_frame_profile*reference_observation_importance.reshape(1,-1,1), axis = 1)
    frame_profile = np.sum(fragments_frame_profile*observation_importance.reshape(1,-1,1), axis = 1)

    frame_similarity = cosine_similarity_a1(
        reference_frame_profile,
        frame_profile
    )

    feature_dict['mean_reference_frame_cosine'] = np.mean(frame_similarity)
    feature_dict['top3_reference_frame_cosine'] = frame_similarity[fragment_idx_sorted[:3]].mean()



    # ============= Template Profile =============

    reference_template_scan_profile = np.sum(reference_template_scan_profile*reference_observation_importance.reshape(1,-1,1), axis = 1)
    template_scan_profile = np.sum(template_scan_profile*observation_importance.reshape(1,-1,1), axis = 1)

    scan_similarity = cosine_similarity_a1(
        reference_template_scan_profile,
        template_scan_profile
    )

    feature_dict['mean_reference_template_scan_cosine'] = np.mean(scan_similarity)

    reference_template_frame_profile = np.sum(reference_template_frame_profile*reference_observation_importance.reshape(1,-1,1), axis = 1)
    template_frame_profile = np.sum(template_frame_profile*observation_importance.reshape(1,-1,1), axis = 1)

    frame_similarity = cosine_similarity_a1(
        reference_template_frame_profile,
        template_frame_profile
    )

    feature_dict['mean_reference_template_frame_cosine'] = np.mean(frame_similarity)

    return feature_dict

@nb.njit
def rank_features(
    current_candidate_idx,
    candidate_list
    ):
        feature_dict = nb.typed.Dict.empty(
            key_type=nb.types.unicode_type,
            value_type=nb.types.float32
        )
        
        max_features = ['mean_reference_template_frame_cosine',
                        'mean_reference_template_scan_cosine',
                        'top3_reference_frame_cosine',
                        'mean_reference_frame_cosine',
                        'top3_reference_scan_cosine',
                        'mean_reference_scan_cosine',
                        'reference_intensity_correlation',
                        'top3_b_ion_correlation',
                        'top3_y_ion_correlation',
                        'top3_frame_correlation',
                        'fragment_frame_correlation',
                        'weighted_ms1_intensity',
                        'isotope_intensity_correlation',
                        'isotope_pattern_correlation',
                        'mono_ms1_intensity'
        ]

        min_features = ['weighted_mass_error',
                        ]

        for feature in max_features:
            if feature in candidate_list[current_candidate_idx].features:
                rank = 0
                count = 0

                for i_candidate in range(len(candidate_list)):
                    if i_candidate != current_candidate_idx:
                        if feature in candidate_list[i_candidate].features:
                            if candidate_list[i_candidate].features[feature] < candidate_list[current_candidate_idx].features[feature]:
                                rank += 1
                            count += 1

            if count > 0:
                feature_dict[feature+'_rank'] = rank / count
            else:
                feature_dict[feature+'_rank'] = 1


        for feature in min_features:
            if feature in candidate_list[current_candidate_idx].features:
                rank = 0
                count = 0

                for i_candidate in range(len(candidate_list)):
                    if i_candidate != current_candidate_idx:
                        if feature in candidate_list[i_candidate].features:
                            if candidate_list[i_candidate].features[feature] > candidate_list[current_candidate_idx].features[feature]:
                                rank += 1
                            count += 1

            if count > 0:
                feature_dict[feature+'_rank'] = rank / count
            else:
                feature_dict[feature+'_rank'] = 1

        return feature_dict
        
        pass