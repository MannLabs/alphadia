import logging

import numpy as np
from alphabase.peptide.fragment import get_charged_frag_types
from alphabase.protein.fasta import protease_dict
from alphabase.spectral_library.base import SpecLibBase
from peptdeep.pretrained_models import ModelManager
from peptdeep.protein.fasta import PredictSpecLibFasta

from alphadia.libtransform.base import ProcessingStep

logger = logging.getLogger()


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
