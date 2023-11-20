# native imports
from pathlib import Path
import logging

logger = logging.getLogger()
import os
import typing

# alphadia imports
from alphadia import utils

# alpha family imports
from alphabase.peptide import fragment
from alphabase.protein import fasta
from alphabase.spectral_library.flat import SpecLibFlat
from alphabase.spectral_library.base import SpecLibBase
from alphabase.spectral_library.reader import LibraryReaderBase
from alphabase.spectral_library.decoy import decoy_lib_provider

# third party imports
import numpy as np
import pandas as pd


class ProcessingStep:
    def __init__(self) -> None:
        """Base class for processing steps. Each implementation must implement the `validate` and `forward` method.
        Processing steps can be chained together in a ProcessingPipeline."""
        pass

    def __call__(self, input: typing.Any) -> typing.Any:
        """Run the processing step on the input object."""
        logger.info(f"Running {self.__class__.__name__}")
        if self.validate(input):
            return self.forward(input)
        else:
            logger.critical(
                f"Input {input} failed validation for {self.__class__.__name__}"
            )
            raise ValueError(
                f"Input {input} failed validation for {self.__class__.__name__}"
            )

    def validate(self, input: typing.Any) -> bool:
        """Validate the input object."""
        raise NotImplementedError("Subclasses must implement this method")

    def forward(self, input: typing.Any) -> typing.Any:
        """Run the processing step on the input object."""
        raise NotImplementedError("Subclasses must implement this method")


class ProcessingPipeline:
    def __init__(self, steps: typing.List[ProcessingStep]) -> None:
        """Processing pipeline for loading and transforming spectral libraries.
        The pipeline is a list of ProcessingStep objects. Each step is called in order and the output of the previous step is passed to the next step.

        Example:
        ```
        pipeline = ProcessingPipeline([
            DynamicLoader(),
            PrecursorInitializer(),
            AnnotateFasta(fasta_path_list),
            IsotopeGenerator(),
            DecoyGenerator(),
            RTNormalization()
        ])

        library = pipeline(input_path)
        ```
        """
        self.steps = steps

    def __call__(self, input: typing.Any) -> typing.Any:
        """Run the pipeline on the input object."""
        for step in self.steps:
            input = step(input)
        return input


class DynamicLoader(ProcessingStep):
    def __init__(self, modification_mapping={}) -> None:
        """Load a spectral library from a file. The file type is dynamically inferred from the file ending.
        Expects a `str` as input and will return a `SpecLibBase` object.

        Supported file types are:

        **Alphabase hdf5 files**
        The library is loaded into a `SpecLibBase` object and immediately returned.

        **Long format csv files**
        The classical spectral library format as returned by MSFragger.
        It will be imported and converted to a `SpecLibBase` format. This might require additional parsing information.
        """
        self.modification_mapping = modification_mapping

    def validate(self, input: str) -> bool:
        """Validate the input object. It is expected that the input is a path to a file which exists."""
        valid = True
        valid &= isinstance(input, str) or isinstance(input, Path)

        if not os.path.exists(input):
            logger.error(f"Input path {input} does not exist")
            valid = False

        return valid

    def forward(self, input_path: str) -> SpecLibBase:
        """Load the spectral library from the input path. The file type is dynamically inferred from the file ending."""
        # get ending of file
        file_type = Path(input_path).suffix

        if file_type in [".hdf5", ".h5", ".hdf"]:
            library = SpecLibBase()
            library.load_hdf(input_path, load_mod_seq=True)

        elif file_type in [".csv", ".tsv"]:
            library = LibraryReaderBase()
            library.add_modification_mapping(self.modification_mapping)
            library.import_file(input_path)

        else:
            raise ValueError(f"File type {file_type} not supported")

        return library


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
            logger.error(f"Input library has no precursor information")
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
        fasta_path_list: typing.List[str],
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
            logger.info(f"Dropping decoys from input library before annotation")
            input._precursor_df = input._precursor_df[input._precursor_df["decoy"] == 0]

        input._precursor_df = fasta.annotate_precursor_df(
            input.precursor_df, protein_df
        )

        if self.drop_unannotated and "cardinality" in input._precursor_df.columns:
            input._precursor_df = input._precursor_df[
                input._precursor_df["cardinality"] > 0
            ]

        return input


class DecoyGenerator(ProcessingStep):
    def __init__(self, decoy_type: str = "diann") -> None:
        """Generate decoys for the spectral library.
        Expects a `SpecLibBase` object as input and will return a `SpecLibBase` object.

        Parameters
        ----------

        decoy_type : str, optional
            Type of decoys to generate. Currently only `pseudo_reverse` and `diann` are supported. Default is `diann`.

        """

        super().__init__()
        self.decoy_type = decoy_type

    def validate(self, input: SpecLibBase) -> bool:
        """Validate the input object. It is expected that the input is a `SpecLibBase` object."""
        return isinstance(input, SpecLibBase)

    def forward(self, input: SpecLibBase) -> SpecLibBase:
        """Generate decoys for the spectral library."""

        if "decoy" not in input.precursor_df.columns:
            input.precursor_df["decoy"] = 0

        decoy_values = input.precursor_df["decoy"].unique()
        if len(decoy_values) > 1:
            logger.warning(
                f"Input library already contains decoys. Skipping decoy generation. \n Please note that decoys generated outside of alphabase are not supported."
            )
            return input

        decoy_lib = decoy_lib_provider.get_decoy_lib(self.decoy_type, input.copy())
        decoy_lib.decoy_sequence()
        decoy_lib.calc_precursor_mz()
        decoy_lib.remove_unused_fragments()
        decoy_lib.calc_fragment_mz_df()
        decoy_lib._precursor_df["decoy"] = 1

        input.append(decoy_lib)
        input._precursor_df.sort_values("elution_group_idx", inplace=True)
        input._precursor_df.reset_index(drop=True, inplace=True)
        input.precursor_df["precursor_idx"] = np.arange(len(input.precursor_df))
        input.remove_unused_fragments()

        return input


class IsotopeGenerator(ProcessingStep):
    def __init__(self, n_isotopes: int = 4) -> None:
        """Generate isotope information for the spectral library.
        Expects a `SpecLibBase` object as input and will return a `SpecLibBase` object.

        Parameters
        ----------

        n_isotopes : int, optional
            Number of isotopes to generate. Default is 4.

        """
        super().__init__()
        self.n_isotopes = n_isotopes

    def validate(self, input: SpecLibBase) -> bool:
        """Validate the input object. It is expected that the input is a `SpecLibBase` object."""
        return isinstance(input, SpecLibBase)

    def forward(self, input: SpecLibBase) -> SpecLibBase:
        """Generate isotope information for the spectral library."""
        existing_isotopes = utils.get_isotope_columns(input.precursor_df.columns)

        if len(existing_isotopes) > 0:
            logger.warning(
                f"Input library already contains isotope information. Skipping isotope generation. \n Please note that isotope generation outside of alphabase is not supported."
            )
            return input

        input.calc_precursor_isotope_intensity(max_isotope=self.n_isotopes)
        return input


class RTNormalization(ProcessingStep):
    def __init__(self) -> None:
        """Normalize the retention time of the spectral library.
        Expects a `SpecLibBase` object as input and will return a `SpecLibBase` object.
        """
        super().__init__()

    def validate(self, input: SpecLibBase) -> bool:
        """Validate the input object. It is expected that the input is a `SpecLibBase` object."""
        return isinstance(input, SpecLibBase)

    def forward(self, input: SpecLibBase) -> SpecLibBase:
        """Normalize the retention time of the spectral library."""
        if len(input.precursor_df) == 0:
            logger.warning(
                f"Input library has no precursor information. Skipping RT normalization"
            )
            return input

        if "rt" not in input.precursor_df.columns:
            if (
                "rt_norm" in input.precursor_df.columns
                or "rt_norm_pred" in input.precursor_df.columns
            ):
                logger.warning(
                    f"Input library already contains normalized RT information. Skipping RT normalization"
                )
                return input
            else:
                logger.warning(
                    f"Input library has no RT information. Skipping RT normalization"
                )
                return input

        percentiles = np.percentile(input.precursor_df["rt"], [0.1, 99.9])
        input._precursor_df["rt"] = np.clip(
            input._precursor_df["rt"], percentiles[0], percentiles[1]
        )

        return input


class FlattenLibrary(ProcessingStep):
    def __init__(
        self, top_k_fragments: int = 12, min_fragment_intensity: float = 0.01
    ) -> None:
        """Convert a `SpecLibBase` object into a `SpecLibFlat` object.

        Parameters
        ----------

        top_k_fragments : int, optional
            Number of top fragments to keep. Default is 12.

        min_fragment_intensity : float, optional
            Minimum intensity threshold for fragments. Default is 0.01.

        """
        self.top_k_fragments = top_k_fragments
        self.min_fragment_intensity = min_fragment_intensity

        super().__init__()

    def validate(self, input: SpecLibBase) -> bool:
        """Validate the input object. It is expected that the input is a `SpecLibBase` object."""
        return isinstance(input, SpecLibBase)

    def forward(self, input: SpecLibBase) -> SpecLibFlat:
        """Convert a `SpecLibBase` object into a `SpecLibFlat` object."""

        input._fragment_cardinality_df = fragment.calc_fragment_cardinality(
            input.precursor_df, input._fragment_mz_df
        )
        output = SpecLibFlat(
            min_fragment_intensity=self.min_fragment_intensity,
            keep_top_k_fragments=self.top_k_fragments,
        )
        output.parse_base_library(
            input, custom_df={"cardinality": input._fragment_cardinality_df}
        )
        return output


class InitFlatColumns(ProcessingStep):
    def __init__(self) -> None:
        """Initialize the columns of a `SpecLibFlat` object for alphadia search.
        Calibratable columns are `mz_library`, `rt_library` and `mobility_library` will be initialized with the first matching column in the input dataframe.
        """

        super().__init__()

    def validate(self, input: SpecLibFlat) -> bool:
        """Validate the input object. It is expected that the input is a `SpecLibFlat` object."""
        return isinstance(input, SpecLibFlat)

    def forward(self, input: SpecLibFlat) -> SpecLibFlat:
        """Initialize the columns of a `SpecLibFlat` object for alphadia search."""

        precursor_columns = {
            "mz_library": ["mz_library", "mz", "precursor_mz"],
            "rt_library": [
                "rt_library",
                "rt",
                "rt_norm",
                "rt_pred",
                "rt_norm_pred",
                "irt",
            ],
            "mobility_library": ["mobility_library", "mobility"],
        }

        fragment_columns = {
            "mz_library": ["mz_library", "mz", "predicted_mz"],
        }

        for column_mapping, df in [
            (precursor_columns, input.precursor_df),
            (fragment_columns, input.fragment_df),
        ]:
            for key, value in column_mapping.items():
                for candidate_columns in value:
                    if candidate_columns in df.columns:
                        df.rename(columns={candidate_columns: key}, inplace=True)
                        # break after first match
                        break

        if "mobility_library" not in input.precursor_df.columns:
            input.precursor_df["mobility_library"] = 0

        return input


class LogFlatLibraryStats(ProcessingStep):
    def __init__(self) -> None:
        """Log basic statistics of a `SpecLibFlat` object."""
        super().__init__()

    def validate(self, input: SpecLibFlat) -> bool:
        """Validate the input object. It is expected that the input is a `SpecLibFlat` object."""
        return isinstance(input, SpecLibFlat)

    def forward(self, input: SpecLibFlat) -> SpecLibFlat:
        """Validate the input object. It is expected that the input is a `SpecLibFlat` object."""

        logger.info(f"============ Library Stats ============")
        logger.info(f"Number of precursors: {len(input.precursor_df):,}")

        if "decoy" in input.precursor_df.columns:
            n_targets = len(input.precursor_df.query("decoy == False"))
            n_decoys = len(input.precursor_df.query("decoy == True"))
            logger.info(f"\tthereof targets:{n_targets:,}")
            logger.info(f"\tthereof decoys: {n_decoys:,}")
        else:
            logger.warning(f"no decoy column was found")

        if "elution_group_idx" in input.precursor_df.columns:
            n_elution_groups = len(input.precursor_df["elution_group_idx"].unique())
            average_precursors_per_group = len(input.precursor_df) / n_elution_groups
            logger.info(f"Number of elution groups: {n_elution_groups:,}")
            logger.info(f"\taverage size: {average_precursors_per_group:.2f}")

        else:
            logger.warning(f"no elution_group_idx column was found")

        if "proteins" in input.precursor_df.columns:
            n_proteins = len(input.precursor_df["proteins"].unique())
            logger.info(f"Number of proteins: {n_proteins:,}")
        else:
            logger.warning(f"no proteins column was found")

        if "channel" in input.precursor_df.columns:
            channels = input.precursor_df["channel"].unique()
            n_channels = len(channels)
            logger.info(f"Number of channels: {n_channels:,} ({channels})")

        else:
            logger.warning(f"no channel column was found, will assume only one channel")

        isotopes = utils.get_isotope_columns(input.precursor_df.columns)

        if len(isotopes) > 0:
            logger.info(f"Isotopes Distribution for {len(isotopes)} isotopes")

        logger.info(f"=======================================")

        return input
