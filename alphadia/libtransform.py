# native imports
import logging
import os
import typing
from pathlib import Path

# third party imports
import numpy as np
import pandas as pd

# alpha family imports
from alphabase.peptide import fragment
from alphabase.peptide.fragment import get_charged_frag_types
from alphabase.protein import fasta
from alphabase.protein.fasta import protease_dict
from alphabase.spectral_library.base import SpecLibBase
from alphabase.spectral_library.decoy import decoy_lib_provider
from alphabase.spectral_library.flat import SpecLibFlat
from alphabase.spectral_library.reader import LibraryReaderBase
from peptdeep.pretrained_models import ModelManager
from peptdeep.protein.fasta import PredictSpecLibFasta

# alphadia imports
from alphadia import utils, validate

logger = logging.getLogger()


class ProcessingStep:
    def __init__(self) -> None:
        """Base class for processing steps. Each implementation must implement the `validate` and `forward` method.
        Processing steps can be chained together in a ProcessingPipeline."""
        pass

    def __call__(self, *args: typing.Any) -> typing.Any:
        """Run the processing step on the input object."""
        logger.info(f"Running {self.__class__.__name__}")
        if self.validate(*args):
            return self.forward(*args)
        else:
            logger.critical(
                f"Input {input} failed validation for {self.__class__.__name__}"
            )
            raise ValueError(
                f"Input {input} failed validation for {self.__class__.__name__}"
            )

    def validate(self, *args: typing.Any) -> bool:
        """Validate the input object."""
        raise NotImplementedError("Subclasses must implement this method")

    def forward(self, *args: typing.Any) -> typing.Any:
        """Run the processing step on the input object."""
        raise NotImplementedError("Subclasses must implement this method")


class ProcessingPipeline:
    def __init__(self, steps: list[ProcessingStep]) -> None:
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
    def __init__(self, modification_mapping: dict | None = None) -> None:
        """Load a spectral library from a file. The file type is dynamically inferred from the file ending.
        Expects a `str` as input and will return a `SpecLibBase` object.

        Supported file types are:

        **Alphabase hdf5 files**
        The library is loaded into a `SpecLibBase` object and immediately returned.

        **Long format csv files**
        The classical spectral library format as returned by MSFragger.
        It will be imported and converted to a `SpecLibBase` format. This might require additional parsing information.
        """
        if modification_mapping is None:
            modification_mapping = {}
        self.modification_mapping = modification_mapping

    def validate(self, input: str) -> bool:
        """Validate the input object. It is expected that the input is a path to a file which exists."""
        valid = True
        valid &= isinstance(input, str | Path)

        if not os.path.exists(input):
            logger.error(f"Input path {input} does not exist")
            valid = False

        return valid

    def forward(self, input_path: str) -> SpecLibBase:
        """Load the spectral library from the input path. The file type is dynamically inferred from the file ending."""
        # get ending of file
        file_type = Path(input_path).suffix

        if file_type in [".hdf5", ".h5", ".hdf"]:
            logger.info(f"Loading {file_type} library from {input_path}")
            library = SpecLibBase()
            library.load_hdf(input_path, load_mod_seq=True)

        elif file_type in [".csv", ".tsv"]:
            logger.info(f"Loading {file_type} library from {input_path}")
            library = LibraryReaderBase()
            library.add_modification_mapping(self.modification_mapping)
            library.import_file(input_path)

        else:
            raise ValueError(f"File type {file_type} not supported")

        return library


class FastaDigest(ProcessingStep):
    def __init__(
        self,
        enzyme: str = "trypsin",
        fixed_modifications: list[str] | None = None,
        variable_modifications: list[str] | None = None,
        missed_cleavages: int = 1,
        precursor_len: list[int] | None = None,
        precursor_charge: list[int] | None = None,
        precursor_mz: list[int] | None = None,
        max_var_mod_num: int = 1,
    ) -> None:
        """Digest a FASTA file into a spectral library.
        Expects a `List[str]` object as input and will return a `SpecLibBase` object.
        """
        if precursor_mz is None:
            precursor_mz = [400, 1200]
        if precursor_charge is None:
            precursor_charge = [2, 4]
        if precursor_len is None:
            precursor_len = [7, 35]
        if variable_modifications is None:
            variable_modifications = ["Oxidation@M", "Acetyl@Prot N-term"]
        if fixed_modifications is None:
            fixed_modifications = ["Carbamidomethyl@C"]
        super().__init__()
        self.enzyme = enzyme
        self.fixed_modifications = fixed_modifications
        self.variable_modifications = variable_modifications
        self.missed_cleavages = missed_cleavages
        self.precursor_len = precursor_len
        self.precursor_charge = precursor_charge
        self.precursor_mz = precursor_mz
        self.max_var_mod_num = max_var_mod_num

    def validate(self, input: list[str]) -> bool:
        if not isinstance(input, list):
            logger.error("Input fasta list is not a list")
            return False
        if len(input) == 0:
            logger.error("Input fasta list is empty")
            return False

        return True

    def forward(self, input: list[str]) -> SpecLibBase:
        frag_types = get_charged_frag_types(["b", "y"], 2)

        model_mgr = ModelManager()

        fasta_lib = PredictSpecLibFasta(
            model_mgr,
            protease=protease_dict[self.enzyme],
            charged_frag_types=frag_types,
            var_mods=self.variable_modifications,
            fix_mods=self.fixed_modifications,
            max_missed_cleavages=self.missed_cleavages,
            max_var_mod_num=self.max_var_mod_num,
            peptide_length_max=self.precursor_len[1],
            peptide_length_min=self.precursor_len[0],
            precursor_charge_min=self.precursor_charge[0],
            precursor_charge_max=self.precursor_charge[1],
            precursor_mz_min=self.precursor_mz[0],
            precursor_mz_max=self.precursor_mz[1],
            decoy=None,
        )
        logger.info("Digesting fasta file")
        fasta_lib.get_peptides_from_fasta_list(input)
        logger.info("Adding modifications")
        fasta_lib.add_modifications()

        fasta_lib.precursor_df["proteins"] = fasta_lib.precursor_df[
            "protein_idxes"
        ].apply(
            lambda x: ";".join(
                [
                    fasta_lib.protein_df["protein_id"].values[int(i)]
                    for i in x.split(";")
                ]
            )
        )
        fasta_lib.precursor_df["genes"] = fasta_lib.precursor_df["protein_idxes"].apply(
            lambda x: ";".join(
                [fasta_lib.protein_df["gene_org"].values[int(i)] for i in x.split(";")]
            )
        )

        fasta_lib.add_charge()
        fasta_lib.hash_precursor_df()
        fasta_lib.calc_precursor_mz()
        fasta_lib.precursor_df = fasta_lib.precursor_df[
            (fasta_lib.precursor_df["precursor_mz"] > self.precursor_mz[0])
            & (fasta_lib.precursor_df["precursor_mz"] < self.precursor_mz[1])
        ]

        logger.info("Removing non-canonical amino acids")
        forbidden = ["B", "J", "X", "Z"]

        masks = []
        for aa in forbidden:
            masks.append(fasta_lib.precursor_df["sequence"].str.contains(aa))
        mask = np.logical_or.reduce(masks)
        fasta_lib.precursor_df = fasta_lib.precursor_df[~mask]

        logger.info(
            f"Fasta library contains {len(fasta_lib.precursor_df):,} precursors"
        )

        return fasta_lib


class PeptDeepPrediction(ProcessingStep):
    def __init__(
        self,
        use_gpu: bool = True,
        mp_process_num: int = 8,
        fragment_mz: list[int] | None = None,
        nce: int = 25,
        instrument: str = "Lumos",
        checkpoint_folder_path: str | None = None,
        fragment_types: list[str] | None = None,
        max_fragment_charge: int = 2,
    ) -> None:
        """Predict the retention time of a spectral library using PeptDeep.

        Parameters
        ----------

        use_gpu : bool, optional
            Use GPU for prediction. Default is True.

        mp_process_num : int, optional
            Number of processes to use for prediction. Default is 8.

        fragment_mz : List[int], optional
            MZ range for fragment prediction. Default is [100, 2000].

        nce : int, optional
            Normalized collision energy for prediction. Default is 25.

        instrument : str, optional
            Instrument type for prediction. Default is "Lumos". Must be a valid PeptDeep instrument.

        checkpoint_folder_path : str, optional
            Path to a folder containing PeptDeep models. If not provided, the default models will be used.

        fragment_types : List[str], optional
            Fragment types to predict. Default is ["b", "y"].

        max_fragment_charge : int, optional
            Maximum charge state to predict. Default is 2.
        """
        if fragment_types is None:
            fragment_types = ["b", "y"]
        if fragment_mz is None:
            fragment_mz = [100, 2000]
        super().__init__()
        self.use_gpu = use_gpu
        self.fragment_mz = fragment_mz
        self.nce = nce
        self.instrument = instrument
        self.mp_process_num = mp_process_num
        self.checkpoint_folder_path = checkpoint_folder_path

        self.fragment_types = fragment_types
        self.max_fragment_charge = max_fragment_charge

    def validate(self, input: list[str]) -> bool:
        return True

    def forward(self, input: SpecLibBase) -> SpecLibBase:
        charged_frag_types = get_charged_frag_types(
            self.fragment_types, self.max_fragment_charge
        )

        input.charged_frag_types = charged_frag_types

        device = utils.get_torch_device(self.use_gpu)

        model_mgr = ModelManager(device=device)
        if self.checkpoint_folder_path is not None:
            logging.info(f"Loading PeptDeep models from {self.checkpoint_folder_path}")
            model_mgr.load_external_models(
                ms2_model_file=os.path.join(self.checkpoint_folder_path, "ms2.pth"),
                rt_model_file=os.path.join(self.checkpoint_folder_path, "rt.pth"),
                ccs_model_file=os.path.join(self.checkpoint_folder_path, "ccs.pth"),
            )

        model_mgr.nce = self.nce
        model_mgr.instrument = self.instrument

        logger.info("Predicting RT, MS2 and mobility")
        res = model_mgr.predict_all(
            input.precursor_df,
            predict_items=["rt", "ms2", "mobility"],
            frag_types=charged_frag_types,
            process_num=self.mp_process_num,
        )

        if "fragment_mz_df" in res:
            logger.info("Adding fragment mz information")
            input._fragment_mz_df = res["fragment_mz_df"][charged_frag_types]

        if "fragment_intensity_df" in res:
            logger.info("Adding fragment intensity information")
            input._fragment_intensity_df = res["fragment_intensity_df"][
                charged_frag_types
            ]

        if "precursor_df" in res:
            logger.info("Adding precursor information")
            input._precursor_df = res["precursor_df"]

        return input


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


class DecoyGenerator(ProcessingStep):
    def __init__(self, decoy_type: str = "diann", mp_process_num: int = 8) -> None:
        """Generate decoys for the spectral library.
        Expects a `SpecLibBase` object as input and will return a `SpecLibBase` object.

        Parameters
        ----------

        decoy_type : str, optional
            Type of decoys to generate. Currently only `pseudo_reverse` and `diann` are supported. Default is `diann`.

        """

        super().__init__()
        self.decoy_type = decoy_type
        self.mp_process_num = mp_process_num

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
                "Input library already contains decoys. Skipping decoy generation. \n Please note that decoys generated outside of alphabase are not supported."
            )
            return input

        decoy_lib = decoy_lib_provider.get_decoy_lib(self.decoy_type, input.copy())

        decoy_lib.charged_frag_types = input.charged_frag_types
        decoy_lib.decoy_sequence(mp_process_num=self.mp_process_num)
        decoy_lib.calc_precursor_mz()
        decoy_lib.remove_unused_fragments()
        decoy_lib.calc_fragment_mz_df()
        decoy_lib._precursor_df["decoy"] = 1

        # keep original precursor_idx and only create new ones for decoys
        start_precursor_idx = input.precursor_df["precursor_idx"].max() + 1
        decoy_lib._precursor_df["precursor_idx"] = np.arange(
            start_precursor_idx, start_precursor_idx + len(decoy_lib.precursor_df)
        )

        input.append(decoy_lib)
        input._precursor_df.sort_values("elution_group_idx", inplace=True)
        input._precursor_df.reset_index(drop=True, inplace=True)
        input.precursor_df["precursor_idx"] = np.arange(len(input.precursor_df))
        input.remove_unused_fragments()

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
        existing_isotopes = utils.get_isotope_columns(input.precursor_df.columns)

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
            "mobility_library": ["mobility_library", "mobility", "mobility_pred"],
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
            logger.warning("Library contains no ion mobility annotations")

        validate.precursors_flat_schema(input.precursor_df)
        validate.fragments_flat_schema(input.fragment_df)

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

        logger.info("============ Library Stats ============")
        logger.info(f"Number of precursors: {len(input.precursor_df):,}")

        if "decoy" in input.precursor_df.columns:
            n_targets = len(input.precursor_df.query("decoy == False"))
            n_decoys = len(input.precursor_df.query("decoy == True"))
            logger.info(f"\tthereof targets:{n_targets:,}")
            logger.info(f"\tthereof decoys: {n_decoys:,}")
        else:
            logger.warning("no decoy column was found")

        if "elution_group_idx" in input.precursor_df.columns:
            n_elution_groups = len(input.precursor_df["elution_group_idx"].unique())
            average_precursors_per_group = len(input.precursor_df) / n_elution_groups
            logger.info(f"Number of elution groups: {n_elution_groups:,}")
            logger.info(f"\taverage size: {average_precursors_per_group:.2f}")

        else:
            logger.warning("no elution_group_idx column was found")

        if "proteins" in input.precursor_df.columns:
            n_proteins = len(input.precursor_df["proteins"].unique())
            logger.info(f"Number of proteins: {n_proteins:,}")
        else:
            logger.warning("no proteins column was found")

        if "channel" in input.precursor_df.columns:
            channels = input.precursor_df["channel"].unique()
            n_channels = len(channels)
            logger.info(f"Number of channels: {n_channels:,} ({channels})")

        else:
            logger.warning("no channel column was found, will assume only one channel")

        isotopes = utils.get_isotope_columns(input.precursor_df.columns)

        if len(isotopes) > 0:
            logger.info(f"Isotopes Distribution for {len(isotopes)} isotopes")

        logger.info("=======================================")

        return input


class MbrLibraryBuilder(ProcessingStep):
    def __init__(self, fdr=0.01) -> None:
        super().__init__()
        self.fdr = fdr

    def validate(self, psm_df, base_library) -> bool:
        """Validate the input object. It is expected that the input is a `SpecLibFlat` object."""
        return True

    def forward(self, psm_df, base_library):
        psm_df = psm_df[psm_df["qval"] <= self.fdr]
        psm_df = psm_df[psm_df["decoy"] == 0]

        rt_df = psm_df.groupby("elution_group_idx", as_index=False).agg(
            rt=pd.NamedAgg(column="rt_observed", aggfunc="median"),
            pg=pd.NamedAgg(column="pg", aggfunc="first"),
        )

        mbr_spec_lib = base_library.copy()
        if "rt" in mbr_spec_lib._precursor_df.columns:
            mbr_spec_lib._precursor_df.drop(columns=["rt"], inplace=True)

        mbr_spec_lib._precursor_df = mbr_spec_lib._precursor_df.merge(
            rt_df, on="elution_group_idx", how="right"
        )
        mbr_spec_lib._precursor_df["genes"] = mbr_spec_lib._precursor_df["pg"]
        mbr_spec_lib._precursor_df["proteins"] = mbr_spec_lib._precursor_df["pg"]

        mbr_spec_lib._precursor_df.drop(columns=["pg"], inplace=True)

        mbr_spec_lib.remove_unused_fragments()

        return mbr_spec_lib
