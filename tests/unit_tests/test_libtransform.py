import tempfile
import numpy as np
import pandas as pd

from alphabase.constants import _const
import tempfile
from alphadia import libtransform


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

    import_pipeline = libtransform.ProcessingPipeline(
        [
            libtransform.DynamicLoader(),
            libtransform.PrecursorInitializer(),
            libtransform.AnnotateFasta([temp_fasta.name]),
            libtransform.IsotopeGenerator(n_isotopes=4),
            libtransform.RTNormalization(),
        ]
    )

    # the prepare pipeline is used to prepare an alphabase compatible spectral library for extraction
    prepare_pipeline = libtransform.ProcessingPipeline(
        [
            libtransform.DecoyGenerator(decoy_type="diann"),
            libtransform.FlattenLibrary(),
            libtransform.InitFlatColumns(),
            libtransform.LogFlatLibraryStats(),
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
