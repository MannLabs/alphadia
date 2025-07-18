import tempfile

import numpy as np
import pandas as pd
from alphabase.spectral_library.base import SpecLibBase

from alphadia.libtransform.base import ProcessingPipeline
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


def test_library_transform():
    fasta = """
>sp|Q9CX84|RGS19_MOUSE Regulator of G-protein signaling 19 OS=Mus musculus OX=10090 GN=Rgs19 PE=1 SV=2
LMHSPTGRRRKK

>sp|P39935|TIF4631_YEAST Translation initiation factor eIF-4G 1 OS=Saccharomyces cerevisiae (strain ATCC 204508 / S288c) OX=559292 GN=TIF4631 YGR254W PE=1 SV=2
KSKSSGEHLDLKSGEHLDLKLMHSPTGR

"""

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
