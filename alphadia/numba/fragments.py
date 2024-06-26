# native imports

# alphadia imports

# alpha family imports

# third party imports
import numba as nb
import numpy as np
from numba.extending import overload_method


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
        cardinality,
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

    # def __repr__(self) -> str:
    #    return f"FragmentContainer with {len(self)} fragments"

    def __str__(self) -> str:
        return f"FragmentContainer with {len(self)} fragments"

    def sort_by_mz(self):
        """
        Sort the fragments in-place by m/z
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

    def filter_by_cardinality(self, max_cardinality: nb.uint8):
        """
        Remove fragements which appear in more than max_cardinality precursors
        """
        mask = self.cardinality <= max_cardinality
        self.precursor_idx = self.precursor_idx[mask]
        self.mz_library = self.mz_library[mask]
        self.mz = self.mz[mask]
        self.intensity = self.intensity[mask]
        self.type = self.type[mask]
        self.loss_type = self.loss_type[mask]
        self.charge = self.charge[mask]
        self.number = self.number[mask]
        self.position = self.position[mask]
        self.cardinality = self.cardinality[mask]

    def filter_top_k(self, top_k):
        """
        Filter out all fragments that are not of cardinality 1 or 2
        """
        mask = self.intensity.argsort()[::-1][:top_k]
        self.precursor_idx = self.precursor_idx[mask]
        self.mz_library = self.mz_library[mask]
        self.mz = self.mz[mask]
        self.intensity = self.intensity[mask]
        self.type = self.type[mask]
        self.loss_type = self.loss_type[mask]
        self.charge = self.charge[mask]
        self.number = self.number[mask]
        self.position = self.position[mask]
        self.cardinality = self.cardinality[mask]

    def filter_by_min_mz(self, min_mz):
        """
        Remove all fragments with m/z below min_mz
        """
        mask = self.mz >= min_mz
        self.precursor_idx = self.precursor_idx[mask]
        self.mz_library = self.mz_library[mask]
        self.mz = self.mz[mask]
        self.intensity = self.intensity[mask]
        self.type = self.type[mask]
        self.loss_type = self.loss_type[mask]
        self.charge = self.charge[mask]
        self.number = self.number[mask]
        self.position = self.position[mask]
        self.cardinality = self.cardinality[mask]

    def apply_mask(self, mask):
        """
        Apply a boolean mask to the fragment container
        """
        self.precursor_idx = self.precursor_idx[mask]
        self.mz_library = self.mz_library[mask]
        self.mz = self.mz[mask]
        self.intensity = self.intensity[mask]
        self.type = self.type[mask]
        self.loss_type = self.loss_type[mask]
        self.charge = self.charge[mask]
        self.number = self.number[mask]
        self.position = self.position[mask]
        self.cardinality = self.cardinality[mask]

        if np.sum(mask) > 0:
            self.intensity = self.intensity / np.sum(self.intensity)


@overload_method(
    nb.types.misc.ClassInstanceType,
    "slice",
)
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

            for i, (start_idx, stop_idx, _step) in enumerate(slices):
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
                fragment_cardinality,
            )

            f.precursor_idx = precursor_idx

            return f

        return impl


@nb.njit()
def slice_manual(inst, slices):
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

    for i, (start_idx, stop_idx, _step) in enumerate(slices):
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
        fragment_cardinality,
    )

    f.precursor_idx = precursor_idx

    return f


@nb.njit
def get_ion_group_mapping(
    ion_precursor,
    ion_mz,
    ion_intensity,
    ion_cardinality,
    precursor_abundance,
    top_k=20,
    max_cardinality=10,
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

    ion_cardinality : np.ndarray
        Array of cardinality values for each ion (n_fragments). This is the number of occurences across precursors in the same elution group.

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

    if (
        not len(ion_mz)
        == len(ion_intensity)
        == len(ion_precursor)
        == len(ion_cardinality)
    ):
        raise ValueError(
            "ion_mz, ion_intensity, ion_precursor and ion cardinality must have the same length"
        )

    cardinality_mask = ion_cardinality <= max_cardinality
    ion_mz = ion_mz[cardinality_mask]
    ion_intensity = ion_intensity[cardinality_mask]
    ion_precursor = ion_precursor[cardinality_mask]
    ion_cardinality = ion_cardinality[cardinality_mask]

    EPSILON = 1e-6

    grouped_mz = []

    score_group_intensity = np.zeros((len(ion_mz)), dtype=np.float32)

    for precursor, mz, intensity in zip(ion_precursor, ion_mz, ion_intensity):  # noqa: B905 ('strict' not supported by numba yet
        # score_group_idx = precursor_group[precursor]

        if len(grouped_mz) == 0 or np.abs(grouped_mz[-1] - mz) > EPSILON:
            grouped_mz.append(mz)

        idx = len(grouped_mz) - 1
        score_group_intensity[idx] += intensity * precursor_abundance[precursor]

    score_group_intensity = score_group_intensity[: len(grouped_mz)].copy()
    grouped_mz = np.array(grouped_mz)

    # normalize each score group to 1
    if np.max(score_group_intensity) > 0:
        score_group_intensity /= score_group_intensity.max()

    indices = np.argsort(score_group_intensity)[::-1][
        : min(top_k, len(score_group_intensity))
    ]
    indices = np.sort(indices)

    grouped_mz = grouped_mz[indices]
    score_group_intensity = score_group_intensity[indices]

    return grouped_mz, score_group_intensity
