import tempfile

import numpy as np
import pandas as pd
import pytest
from alphabase.spectral_library.base import SpecLibBase

from alphadia.libtransform.base import ProcessingPipeline, ProcessingStep
from alphadia.libtransform.decoy import DecoyGenerator
from alphadia.libtransform.flatten import (
    FlattenLibrary,
    InitFlatColumns,
    LogFlatLibraryStats,
)
from alphadia.libtransform.harmonize import (
    AnnotateFasta,
    IsotopeGenerator,
    PrecursorInitializer,
    RTNormalization,
)
from alphadia.libtransform.loader import DynamicLoader
from alphadia.libtransform.multiplex import MultiplexLibrary
from alphadia.libtransform.prediction import DecoyPrediction


def test_library_transform():
    fasta = """
>sp|Q9CX84|RGS19_MOUSE Regulator of G-protein signaling 19 OS=Mus musculus OX=10090 GN=Rgs19 PE=1 SV=2
LMHSPTGRRRKK

>sp|P39935|TIF4631_YEAST Translation initiation factor eIF-4G 1 OS=Saccharomyces cerevisiae (strain ATCC 204508 / S288c) OX=559292 GN=TIF4631 YGR254W PE=1 SV=2
KSKSSGEHLDLKSGEHLDLKLMHSPTGR

""".strip()

    library = """PrecursorMz	ProductMz	Annotation	ProteinId	GeneName	PeptideSequence	ModifiedPeptideSequence	PrecursorCharge	LibraryIntensity	NormalizedRetentionTime	PrecursorIonMobility	FragmentType	FragmentCharge	FragmentSeriesNumber	FragmentLossType
300.156968	333.188096	y3^1	Q9CX84	Rgs19	LMHSPTGR	LMHSPTGR	3	4311.400524927019	-25.676406886060136		y	1	3
300.156968	430.24086	y4^1	Q9CX84	Rgs19	LMHSPTGR	LMHSPTGR	3	7684.946735600609	-25.676406886060136		y	1	4
300.156968	517.27289	y5^1	Q9CX84	Rgs19	LMHSPTGR	LMHSPTGR	3	10000.0	-25.676406886060136		y	1	5
300.159143	313.187033	y5^2	P39935	TIF4631	SGEHLDLK	SGEHLDLK	3	4817.867861369569	29.42456033403839		y	2	5
300.159143	375.223813	y3^1	P39935	TIF4631	SGEHLDLK	SGEHLDLK	3	8740.775194419808	29.42456033403839		y	1	3
300.159143	406.219062	y7^2	P39935	TIF4631	SGEHLDLK	SGEHLDLK	3	2026.7157241363188	29.42456033403839		y	2	7
300.159143	488.307878	y4^1	P39935	TIF4631	SGEHLDLK	SGEHLDLK	3	10000.0	29.42456033403839		y	1	4
300.159143	625.36679	y5^1	P39935	TIF4631	SGEHLDLK	SGEHLDLK	3	6782.1533255969025	29.42456033403839		y	1	5
300.159143	639.273285	b6^1	P39935	TIF4631	SGEHLDLK	SGEHLDLK	3	1844.4293802287832	29.42456033403839		b	1	6
"""

    # create temp file
    temp_lib = tempfile.NamedTemporaryFile(suffix=".tsv", delete=False)
    temp_lib.write(library.encode())
    temp_lib.close()

    # create temp fasta
    temp_fasta = tempfile.NamedTemporaryFile(suffix=".fasta", delete=False)
    temp_fasta.write(fasta.encode())
    temp_fasta.close()

    import_pipeline = ProcessingPipeline(
        [
            DynamicLoader(),
            PrecursorInitializer(),
            AnnotateFasta([temp_fasta.name]),
            IsotopeGenerator(n_isotopes=4),
            RTNormalization(),
        ]
    )

    # the prepare pipeline is used to prepare an alphabase compatible spectral library for extraction
    prepare_pipeline = ProcessingPipeline(
        [
            DecoyGenerator(decoy_type="diann"),
            FlattenLibrary(),
            InitFlatColumns(),
            LogFlatLibraryStats(),
        ]
    )

    speclib = import_pipeline(temp_lib.name)
    speclib = prepare_pipeline(speclib)

    assert len(speclib.precursor_df) == 4
    assert np.all(
        [
            col in speclib.precursor_df.columns
            for col in [
                "mz_library",
                "rt_library",
                "mobility_library",
                "i_0",
                "i_1",
                "i_2",
                "i_3",
            ]
        ]
    )
    speclib.precursor_df.sort_values("cardinality", inplace=True, ascending=False)

    assert speclib.precursor_df["decoy"].sum() == 2
    assert np.all(speclib.precursor_df["cardinality"] == [2, 2, 1, 1])


def test_multiplex_library():
    # given
    repeat = 2
    peptides = ["AGHCEWQMK"] * repeat
    mods = ["mTRAQ@K"] * repeat
    sites = ["0;9"] * repeat

    precursor_df = pd.DataFrame(
        {"sequence": peptides, "mods": mods, "mod_sites": sites}
    )
    precursor_df["nAA"] = precursor_df["sequence"].str.len()
    precursor_df["charge"] = [2, 3]

    test_lib = SpecLibBase()
    test_lib.precursor_df = precursor_df
    test_lib.calc_precursor_mz()
    test_lib.calc_fragment_mz_df()

    test_multiplex_mapping = [
        {"channel_name": 0, "modifications": {"mTRAQ@K": "mTRAQ@K"}},
        {
            "channel_name": "magic_channel",
            "modifications": {"mTRAQ@K": "mTRAQ:13C(3)15N(1)@K"},
        },
        {"channel_name": 1337, "modifications": {"mTRAQ@K": "mTRAQ:13C(6)15N(2)@K"}},
    ]

    # when
    multiplexer = MultiplexLibrary(test_multiplex_mapping)
    result_lib = multiplexer.forward(test_lib)

    # then
    assert result_lib.precursor_df["sequence"].shape == (6,)
    assert result_lib.precursor_df["charge"].nunique() == 2
    assert result_lib.precursor_df["frag_stop_idx"].nunique() == 6

    for channel in [0, 1337, "magic_channel"]:
        assert (
            result_lib.precursor_df[
                result_lib.precursor_df["channel"] == channel
            ].shape[0]
            == repeat
        )

    for modification in ["mTRAQ@K", "mTRAQ:13C(3)15N(1)@K", "mTRAQ:13C(6)15N(2)@K"]:
        assert (
            result_lib.precursor_df[
                result_lib.precursor_df["mods"].str.contains(modification, regex=False)
            ].shape[0]
            == repeat
        )


def test_precursor_initializer_drop_decoys():
    """Test that PrecursorInitializer drops decoys when drop_decoys=True."""
    speclib = SpecLibBase()
    speclib._precursor_df = pd.DataFrame(
        {
            "sequence": ["PEPTIDE", "ANOTHERPEP", "DECOYSEQ", "DECOYSEQ2"],
            "mods": ["", "", "", ""],
            "mod_sites": ["", "", "", ""],
            "decoy": [0, 0, 1, 1],
        }
    )
    speclib._fragment_mz_df = pd.DataFrame({"b_z1": [1.0, 2.0, 3.0, 4.0]})
    speclib._fragment_intensity_df = pd.DataFrame(
        {"b_z1": [100.0, 200.0, 300.0, 400.0]}
    )
    speclib._precursor_df["frag_start_idx"] = [0, 1, 2, 3]
    speclib._precursor_df["frag_stop_idx"] = [1, 2, 3, 4]

    initializer = PrecursorInitializer(drop_decoys=True)
    result = initializer(speclib)

    assert len(result.precursor_df) == 2
    assert (result.precursor_df["decoy"] == 0).all()
    assert "PEPTIDE" in result.precursor_df["sequence"].values
    assert "ANOTHERPEP" in result.precursor_df["sequence"].values
    assert "DECOYSEQ" not in result.precursor_df["sequence"].values


def test_precursor_initializer_keep_decoys():
    """Test that PrecursorInitializer keeps decoys when drop_decoys=False (default)."""
    speclib = SpecLibBase()
    speclib._precursor_df = pd.DataFrame(
        {
            "sequence": ["PEPTIDE", "ANOTHERPEP", "DECOYSEQ", "DECOYSEQ2"],
            "mods": ["", "", "", ""],
            "mod_sites": ["", "", "", ""],
            "decoy": [0, 0, 1, 1],
        }
    )
    speclib._fragment_mz_df = pd.DataFrame({"b_z1": [1.0, 2.0, 3.0, 4.0]})
    speclib._fragment_intensity_df = pd.DataFrame(
        {"b_z1": [100.0, 200.0, 300.0, 400.0]}
    )
    speclib._precursor_df["frag_start_idx"] = [0, 1, 2, 3]
    speclib._precursor_df["frag_stop_idx"] = [1, 2, 3, 4]

    initializer = PrecursorInitializer(drop_decoys=False)
    result = initializer(speclib)

    assert len(result.precursor_df) == 4
    assert result.precursor_df["decoy"].sum() == 2


def test_shuffle_decoy_keeps_the_termini_and_the_composition():
    from alphadia.libtransform.decoy import ShuffleDecoyGenerator

    generator = ShuffleDecoyGenerator()
    sequence = "ACDEFGHIKLMNPQRSTVWY"

    decoy = generator._decoy(sequence)

    assert decoy != sequence
    assert decoy[0] == sequence[0]
    assert decoy[-1] == sequence[-1]
    assert sorted(decoy) == sorted(sequence)
    # seeded by the sequence, so every process derives the same decoy
    assert ShuffleDecoyGenerator()._decoy(sequence) == decoy


def test_shuffle_decoy_is_registered_with_alphabase():
    from alphabase.spectral_library.decoy import decoy_lib_provider

    from alphadia.libtransform import decoy  # noqa: F401 # registers on import

    assert "shuffle" in decoy_lib_provider.decoy_dict


class _ConstantPrediction(ProcessingStep):
    """Stands in for peptdeep: every intensity becomes one, every rt_pred one half."""

    def __init__(self, drop_first: bool = False):
        super().__init__()
        self.drop_first = drop_first

    def validate(self, input: SpecLibBase) -> bool:
        return True

    def forward(self, input: SpecLibBase) -> SpecLibBase:
        if self.drop_first:
            input._precursor_df = input.precursor_df.iloc[1:].copy()
            input.remove_unused_fragments()
        input._fragment_intensity_df = pd.DataFrame(
            1.0,
            index=input.fragment_mz_df.index,
            columns=input.fragment_mz_df.columns,
        )
        input._precursor_df["rt_pred"] = 0.5
        return input


def _library_with_decoys() -> SpecLibBase:
    precursor_df = pd.DataFrame(
        {
            "sequence": ["PEPTIDEK", "ANOTHERPEPTIDER"],
            "mods": ["", ""],
            "mod_sites": ["", ""],
            "charge": [2, 3],
            "decoy": [0, 0],
            "rt_pred": [0.1, 0.2],
        }
    )
    precursor_df["nAA"] = precursor_df["sequence"].str.len()
    precursor_df["precursor_idx"] = np.arange(len(precursor_df))
    precursor_df["elution_group_idx"] = np.arange(len(precursor_df))
    precursor_df["channel"] = 0
    library = SpecLibBase()
    library.precursor_df = precursor_df
    library.calc_precursor_mz()
    library.calc_fragment_mz_df()
    library._fragment_intensity_df = pd.DataFrame(
        0.2, index=library.fragment_mz_df.index, columns=library.fragment_mz_df.columns
    )
    return DecoyGenerator(decoy_type="diann", mp_process_num=1)(library)


def test_decoy_prediction_replaces_only_the_decoys():
    library = _library_with_decoys()
    n_targets = int((library.precursor_df["decoy"] == 0).sum())
    n_decoys = int((library.precursor_df["decoy"] == 1).sum())
    assert n_decoys == n_targets

    result = DecoyPrediction(_ConstantPrediction())(library)

    df = result.precursor_df
    assert (df["decoy"] == 0).sum() == n_targets
    assert (df["decoy"] == 1).sum() == n_decoys
    assert df.loc[df["decoy"] == 0, "rt_pred"].tolist() == [0.1, 0.2]
    assert (df.loc[df["decoy"] == 1, "rt_pred"] == 0.5).all()
    for _, row in df.iterrows():
        intensities = result.fragment_intensity_df.iloc[
            row["frag_start_idx"] : row["frag_stop_idx"]
        ]
        assert len(intensities) == row["nAA"] - 1
        assert (intensities == (1.0 if row["decoy"] else 0.2)).all().all()
    assert len(result.fragment_mz_df) == len(result.fragment_intensity_df)


def test_decoy_prediction_rejects_a_prediction_that_drops_decoys():
    library = _library_with_decoys()

    with pytest.raises(ValueError, match="dropped 1 decoys"):
        DecoyPrediction(_ConstantPrediction(drop_first=True)).forward(library)


def test_decoy_prediction_needs_decoys():
    library = _library_with_decoys()
    library._precursor_df = library.precursor_df[library.precursor_df["decoy"] == 0]

    assert not DecoyPrediction(_ConstantPrediction()).validate(library)
