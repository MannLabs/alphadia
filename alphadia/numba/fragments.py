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
    if inst is not FragmentContainer.class_type.instance_type:
        return None

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
