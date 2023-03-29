import numba as nb
from numba.extending import overload_method, overload
import numpy as np

@nb.experimental.jitclass()
class FragmentContainer:

    mz_library: nb.float32[::1]
    mz: nb.float32[::1]
    intensity: nb.float32[::1]
    type: nb.uint8[::1]
    loss_type: nb.uint8[::1]
    charge: nb.uint8[::1]
    number: nb.uint8[::1]
    position: nb.uint8[::1]
    precursor_idx: nb.uint32[::1]
    cardinality: nb.uint8[::1]
       
    def __init__(
        self,
        mz_library,
        mz,
        intensity,
        type,
        loss_type,
        charge,
        number,
        position,
        cardinality
    ):

        self.mz_library = mz_library
        self.mz = mz
        self.intensity = intensity
        self.type = type
        self.loss_type = loss_type
        self.charge = charge
        self.number = number
        self.position = position
        self.precursor_idx = np.zeros(len(mz), dtype=np.uint32)
        self.cardinality = cardinality

    def __len__(self):
        return len(self.mz)

    def sort_by_mz(self):
        """
        Sort all arrays by m/z
        
        """
        mz_order = np.argsort(self.mz)
        self.precursor_idx = self.precursor_idx[mz_order]
        self.mz_library = self.mz_library[mz_order]
        self.mz = self.mz[mz_order]
        self.intensity = self.intensity[mz_order]
        self.type = self.type[mz_order]
        self.loss_type = self.loss_type[mz_order]
        self.charge = self.charge[mz_order]
        self.number = self.number[mz_order]
        self.position = self.position[mz_order]
        self.cardinality = self.cardinality[mz_order]
        

@overload_method(nb.types.misc.ClassInstanceType, 'slice', )
def slice(inst, slices):
    
    if inst is FragmentContainer.class_type.instance_type:

        
        def impl(inst, slices):
            precursor_idx = []
            fragments_mz_library = []
            fragments_mz = []
            fragments_intensity = []
            fragments_type = []
            fragments_loss_type = []
            fragments_charge = []
            fragments_number = []
            fragments_position = []
            fragments_cardinality = []

            precursor = np.arange(len(slices), dtype=np.uint32)

            for i, (start_idx, stop_idx, step) in enumerate(slices):
                for j in range(start_idx, stop_idx):
                    
                    precursor_idx.append(precursor[i])
                    fragments_mz_library.append(inst.mz_library[j])
                    fragments_mz.append(inst.mz[j])
                    fragments_intensity.append(inst.intensity[j])
                    fragments_type.append(inst.type[j])
                    fragments_loss_type.append(inst.loss_type[j])
                    fragments_charge.append(inst.charge[j])
                    fragments_number.append(inst.number[j])
                    fragments_position.append(inst.position[j])
                    fragments_cardinality.append(inst.cardinality[j])

            precursor_idx = np.array(precursor_idx, dtype=np.uint32)
            fragments_mz_library = np.array(fragments_mz_library, dtype=np.float32)
            fragment_mz = np.array(fragments_mz, dtype=np.float32)
            fragment_intensity = np.array(fragments_intensity, dtype=np.float32)
            fragment_type = np.array(fragments_type, dtype=np.uint8)
            fragment_loss_type = np.array(fragments_loss_type, dtype=np.uint8)
            fragment_charge = np.array(fragments_charge, dtype=np.uint8)
            fragment_number = np.array(fragments_number, dtype=np.uint8)
            fragment_position = np.array(fragments_position, dtype=np.uint8)
            fragment_cardinality = np.array(fragments_cardinality, dtype=np.uint8)

            f = FragmentContainer(
                fragments_mz_library,
                fragment_mz,
                fragment_intensity,
                fragment_type,
                fragment_loss_type,
                fragment_charge,
                fragment_number,
                fragment_position,
                fragment_cardinality
            )

            f.precursor_idx = precursor_idx

            return f
        return impl
    
import numba as nb
@nb.njit
def get_ion_group_mapping(
    ion_precursor, 
    ion_mz, 
    ion_intensity, 
    precursor_abundance, 
    precursor_group,
    exclude_shared=False,
    top_k=20
    ):
    """
    Can be used to group a set of ions by and return the expected, summed intensity distribution for each group.

    Parameters
    ----------

    ion_precursor : np.ndarray
        Array of precursor indices for each ion with shape (n_fragments).

    ion_mz : np.ndarray
        Array of m/z values for each ion (n_fragments).

    ion_intensity : np.ndarray
        Array of intensity values for each ion (n_fragments).

    precursor_abundance : np.ndarray
        Array of precursor abundances with shape (n_precursors).

    precursor_group : np.ndarray
        Array of precursor groups with shape (n_precursors).

    exclude_shared : bool, optional, default=False
        If True, ions that are shared between multiple precursor groups are excluded from the calculation.

    top_k : int, optional, default=20
        Number of ions to consider per precursor group.

    Returns
    -------

    score_group_intensity : np.ndarray
        Array of summed intensity values for each group with shape (n_groups, n_unique_fragments).

    score_group_mz : np.ndarray
        Array of m/z values for each group with shape (n_groups, n_unique_fragments).


    """

    if not len(precursor_abundance) == len(precursor_group):
        raise ValueError('precursor_abundance and precursor_group must have the same length')
    
    if not len(ion_mz) == len(ion_intensity) == len(ion_precursor):
        raise ValueError('ion_mz, ion_intensity and ion_precursor must have the same length')

    EPSILON = 1e-6

    grouped_mz = []

    # keep track of the ions per group
    top_k_count = np.zeros(precursor_group.max()+1)

    score_group_intensity = np.zeros((precursor_group.max()+1, len(ion_mz)), dtype=np.float32)

    for i, (precursor, mz, intensity) in enumerate(zip(ion_precursor, ion_mz, ion_intensity)):

        score_group_idx = precursor_group[precursor]

        if top_k_count[score_group_idx] < top_k:
            top_k_count[score_group_idx] += 1

            if len(grouped_mz) == 0:
                grouped_mz.append(mz)
                
            elif np.abs(grouped_mz[-1] - mz) > EPSILON:
                grouped_mz.append(mz)

            idx = len(grouped_mz) - 1
            
            priot_abundance = precursor_abundance[precursor]

            score_group_intensity[score_group_idx, idx] += intensity * priot_abundance

    score_group_intensity = score_group_intensity[:, :len(grouped_mz)].copy()

    grouped_mz = np.array(grouped_mz)

    # normalize each score group to 1
    for row in score_group_intensity:
        if np.max(row) > 0:
            row /= row.max()

    if exclude_shared:
        score_group_count = np.ceil(score_group_intensity).astype(np.uint8).sum(axis=0)

        score_group_intensity = score_group_intensity[:, score_group_count <= 1]
        grouped_mz = grouped_mz[score_group_count <= 1]

        # make sure to exit if all ions are shared
        if len(grouped_mz) == 0:
            return grouped_mz, score_group_intensity

        #normalization has to be done again
        # it was needed in the first place so np.ceil evaluates to 1
        # normalize each score group to 1
        for row in score_group_intensity:
            if row.max() > 0:
                row /= row.max()

    return grouped_mz, score_group_intensity