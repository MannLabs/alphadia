import logging
import os

import numpy as np
from alphabase.protein import fasta
from alphabase.spectral_library.base import SpecLibBase

from alphadia.libtransform.base import ProcessingStep
from alphadia.utils import get_isotope_columns

logger = logging.getLogger()


class PrecursorInitializer(ProcessingStep):
    def __init__(self, *args, **kwargs) -> None:
        """Initialize alphabase spectral library with precursor information.
        Expects a `SpecLibBase` object as input and will return a `SpecLibBase` object.
        This step is required for all spectral libraries and will add the `precursor_idx`,`decoy`, `channel` and `elution_group_idx` columns to the precursor dataframe.
        """
        super().__init__(*args, **kwargs)

    def validate(self, input: SpecLibBase) -> bool:
        """Validate the input object. It is expected that the input is a `SpecLibBase` object."""
        valid = isinstance(input, SpecLibBase)

        if len(input.precursor_df) == 0:
            logger.error("Input library has no precursor information")
            valid = False

        if len(input.fragment_intensity_df) == 0:
            logger.error("Input library has no fragment intensity information")
            valid = False

        if len(input.fragment_mz_df) == 0:
            logger.error("Input library has no fragment mz information")
            valid = False

        return valid

    def forward(self, input: SpecLibBase) -> SpecLibBase:
        """Initialize the precursor dataframe with the `precursor_idx`, `decoy`, `channel` and `elution_group_idx` columns."""
        if "decoy" not in input.precursor_df.columns:
            input.precursor_df["decoy"] = 0

        if "channel" not in input.precursor_df.columns:
            input.precursor_df["channel"] = 0

        if "elution_group_idx" not in input.precursor_df.columns:
            input.precursor_df["elution_group_idx"] = np.arange(len(input.precursor_df))

        if "precursor_idx" not in input.precursor_df.columns:
            input.precursor_df["precursor_idx"] = np.arange(len(input.precursor_df))

        return input


class AnnotateFasta(ProcessingStep):
    def __init__(
        self,
        fasta_path_list: list[str],
        drop_unannotated: bool = True,
        drop_decoy: bool = True,
    ) -> None:
        """Annotate the precursor dataframe with protein information from a FASTA file.
        Expects a `SpecLibBase` object as input and will return a `SpecLibBase` object.

        Parameters
        ----------
        fasta_path_list : List[str]
            List of paths to FASTA files. Multiple files can be provided and will be merged into a single protein dataframe.

        drop_unannotated : bool, optional
            Drop all precursors which could not be annotated by the FASTA file. Default is True.

        """
        super().__init__()
        self.fasta_path_list = fasta_path_list
        self.drop_unannotated = drop_unannotated
        self.drop_decoy = drop_decoy

    def validate(self, input: SpecLibBase) -> bool:
        """Validate the input object. It is expected that the input is a `SpecLibBase` object and that all FASTA files exist."""
        valid = isinstance(input, SpecLibBase)

        for path in self.fasta_path_list:
            if not os.path.exists(path):
                logger.error(
                    f"Annotation by FASTA failed, input path {path} does not exist"
                )
                valid = False

        return valid

    def forward(self, input: SpecLibBase) -> SpecLibBase:
        """Annotate the precursor dataframe with protein information from a FASTA file."""
        protein_df = fasta.load_fasta_list_as_protein_df(self.fasta_path_list)

        if self.drop_decoy and "decoy" in input.precursor_df.columns:
            logger.info("Dropping decoys from input library before annotation")
            input._precursor_df = input._precursor_df[input._precursor_df["decoy"] == 0]

        input._precursor_df = fasta.annotate_precursor_df(
            input.precursor_df, protein_df
        )

        if self.drop_unannotated and "cardinality" in input._precursor_df.columns:
            input._precursor_df = input._precursor_df[
                input._precursor_df["cardinality"] > 0
            ]

        return input


class IsotopeGenerator(ProcessingStep):
    def __init__(self, n_isotopes: int = 4, mp_process_num: int = 8) -> None:
        """Generate isotope information for the spectral library.
        Expects a `SpecLibBase` object as input and will return a `SpecLibBase` object.

        Parameters
        ----------
        n_isotopes : int, optional
            Number of isotopes to generate. Default is 4.

        """
        super().__init__()
        self.n_isotopes = n_isotopes
        self.mp_process_num = mp_process_num

    def validate(self, input: SpecLibBase) -> bool:
        """Validate the input object. It is expected that the input is a `SpecLibBase` object."""
        return isinstance(input, SpecLibBase)

    def forward(self, input: SpecLibBase) -> SpecLibBase:
        """Generate isotope information for the spectral library."""
        existing_isotopes = get_isotope_columns(input.precursor_df.columns)

        if len(existing_isotopes) > 0:
            logger.warning(
                "Input library already contains isotope information. Skipping isotope generation. \n Please note that isotope generation outside of alphabase is not supported."
            )
            return input

        input.calc_precursor_isotope_intensity(
            max_isotope=self.n_isotopes,
            mp_process_num=self.mp_process_num,
        )
        return input


class RTNormalization(ProcessingStep):
    def __init__(self) -> None:
        """Normalize the retention time of the spectral library.
        Expects a `SpecLibBase` object as input and will return a `SpecLibBase` object.
        """
        super().__init__()

    def validate(self, input: SpecLibBase) -> bool:
        """Validate the input object. It is expected that the input is a `SpecLibBase` object."""
        valid = isinstance(input, SpecLibBase)

        if not any(
            [
                col in input.precursor_df.columns
                for col in ["rt", "rt_norm", "rt_norm_pred"]
            ]
        ):
            logger.error(
                "Input library has no RT information. Please enable RT prediction or provide RT information."
            )
            valid = False
        return valid

    def forward(self, input: SpecLibBase) -> SpecLibBase:
        """Normalize the retention time of the spectral library."""
        if len(input.precursor_df) == 0:
            logger.warning(
                "Input library has no precursor information. Skipping RT normalization"
            )
            return input

        if "rt" not in input.precursor_df.columns and (
            "rt_norm" in input.precursor_df.columns
            or "rt_norm_pred" in input.precursor_df.columns
        ):
            logger.warning(
                "Input library already contains normalized RT information. Skipping RT normalization"
            )
            return input

        percentiles = np.percentile(input.precursor_df["rt"], [0.1, 99.9])
        input._precursor_df["rt"] = np.clip(
            input._precursor_df["rt"], percentiles[0], percentiles[1]
        )

        return input
