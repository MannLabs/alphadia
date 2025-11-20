# Output format
## Single-Step Search
For a standard single-step search, all output files are written directly to the output directory specified with the `-o` flag:

```
output/
├── stats.tsv
├── precursors.parquet
├── pg.matrix.parquet
├── internal.tsv
├── speclib.hdf
├── speclib.mbr.hdf
├── frozen_config.yaml
├── quant/
│   ├── <raw_file_1>/
│   │   ├── psm.parquet
│   │   └── frag.parquet
│   ├── <raw_file_2>/
│   │   ├── psm.parquet
│   │   └── frag.parquet
│   └── ...
└── figures/
```

## Multi-Step Search
AlphaDIA supports multi-step searches to improve identification rates through transfer learning and match-between-runs (MBR). When these features are enabled, the output is organized into subdirectories, with each step producing its own intermediate results. The output of the final step will always be in the root of the output directory.

### Transfer Learning Step (`transfer_step_enabled: true`)
When transfer learning is enabled, an initial search is performed to train sample-specific PeptDeep models. The intermediate results are stored in `output/transfer/`, which include:
- Training data for the neural network models (`speclib.transfer.hdf`)
- Trained PeptDeep models (`peptdeep.transfer/`)
- Statistics from the transfer learning process (`stats.transfer.tsv`)

The final results will still be saved in the `output` folder.

### Match Between Runs (`mbr_step_enabled: true`)
When MBR is enabled, a two-pass search strategy is used. The first pass performs an initial search to build a sample-specific MBR library. Intermediate results are stored in `output/library/`, including:
- The MBR library built from first-pass identifications (`speclib.mbr.hdf`)
- Statistics from the library building step

The final results will still be saved in the `output` folder.

## Output Files

### Overview

| File | Description |
|------|-------------|
| `precursors.parquet` | Main output with precursor-level information, quantification, and scoring |
| `stats.tsv` | Summary statistics and quality metrics per run/channel |
| `pg.matrix.parquet` | Protein group quantification matrix across all samples |
| `peptide.matrix.parquet` | Peptide-level quantification matrix (if enabled) |
| `precursor.matrix.parquet` | Precursor-level quantification matrix (if enabled) |
| `internal.tsv` | Internal statistics and metadata from the search |
| `speclib.hdf` | Input spectral library (may be reannotated or predicted) |
| `speclib.mbr.hdf` | MBR library containing all identified precursors |
| `speclib.transfer.hdf` | Fragment quantities extracted from search results for transfer learning |
| `frozen_config.yaml` | Complete configuration snapshot for reproducibility |
| `quant/` | Per-file quantification data for checkpointing |
| `figures/` | Quality control figures and visualizations |

### `precursors.parquet`
The main output file containing precursor-level identifications with scoring, quantification, and metadata.

Format: one row per identified precursor per run.

#### Columns

| Column | Description | Unit |
|--------|-------------|------|
| **Raw Level** | | |
| `raw.name` | Name of the raw file/run | - |
| **Precursor Level** | | |
| `precursor.idx` | Unique index for the precursor in the library (consistent only within a search; may vary across searches due to filtering or raw files) | - |
| `precursor.elution_group_idx` | Index of the elution group (precursors eluting together; consistent only within a search) | - |
| `precursor.sequence` | Peptide sequence | - |
| `precursor.charge` | Precursor charge state | - |
| `precursor.mods` | Modification types (e.g., Phospho@S; semicolon-separated) | - |
| `precursor.mod_sites` | Modification positions in the sequence (e.g., 5; semicolon-separated, corresponds to mods) | - |
| `precursor.mod_seq_hash` | Hash of modified sequence (peptide level; stable across searches for comparison) | - |
| `precursor.mod_seq_charge_hash` | Hash of modified sequence with charge (precursor level; stable across searches for comparison) | - |
| `precursor.rank` | Rank of this precursor in the search candidates | - |
| `precursor.naa` | Number of amino acids in the sequence | count |
| `precursor.mz.library` | Calculated (theoretical) m/z based on peptide sequence and modifications | - |
| `precursor.mz.observed` | Observed m/z | - |
| `precursor.mz.calibrated` | Calibrated m/z | - |
| `precursor.rt.library` | Library-annotated retention time (predicted or empirical) | seconds |
| `precursor.rt.observed` | Observed retention time | seconds |
| `precursor.rt.calibrated` | Calibrated retention time | seconds |
| `precursor.rt.fwhm` | Full width at half maximum of the RT peak | seconds |
| `precursor.mobility.library` | Library-annotated ion mobility (predicted or empirical) | mobility units |
| `precursor.mobility.observed` | Observed ion mobility | mobility units |
| `precursor.mobility.calibrated` | Calibrated ion mobility | mobility units |
| `precursor.mobility.fwhm` | Full width at half maximum of the mobility peak | mobility units |
| `precursor.intensity` | Quantified intensity (LFQ intensity if enabled) | arbitrary units |
| `precursor.qval` | Q-value (FDR-corrected p-value) | - |
| `precursor.proba` | Decoy probability score from classifier (range 0-1). Lower scores indicate higher probability of a target hit. | - |
| `precursor.score` | Raw score from scoring function | - |
| `precursor.channel` | Channel number (0 for label-free) | - |
| `precursor.decoy` | Decoy flag (0=target, 1=decoy) | - |
| **Peptide Level** | | |
| `peptide.intensity` | Peptide-level intensity (if peptide-level LFQ enabled) | arbitrary units |
| **Protein Group Level** | | |
| `pg.name` | Protein group identifier | - |
| `pg.proteins` | Protein accessions in the group (semicolon-separated) | - |
| `pg.genes` | Gene names associated with the protein group | - |
| `pg.master_protein` | Representative protein in the group | - |
| `pg.qval` | Protein group q-value | - |
| `pg.intensity` | Protein group intensity (if LFQ enabled) | arbitrary units |

**Notes:**
- Mobility columns are only present for ion mobility data
- LFQ intensities are only present when label-free quantification is enabled
- Decoy precursors (`decoy=1`) are typically filtered out unless `keep_decoys` is enabled
- The `precursor.proba` value represents the decoy probability score (lower is better for target hits)
- **Identifiers for comparison**: Use `precursor.mod_seq_hash` (peptide level) or `precursor.mod_seq_charge_hash` (precursor level) to match identifications across different searches. These hashes are stable and based on the modified sequence, making them suitable for comparing results between runs, experiments, or analysis versions. In contrast, `precursor.idx` and `precursor.elution_group_idx` are search-specific and should not be used for cross-search comparisons

### `stats.tsv`
The `stats.tsv` file contains summary statistics and quality metrics for each run and channel in the analysis.
It provides insights into the search results, calibration quality, and general performance metrics.

Format: one row per run/channel combination.

#### Columns

| Column | Description | Unit |
|--------|-------------|------|
| **Raw Level** | | |
| `raw.name` | Name of the raw file/run | - |
| `raw.gradient_length` | Total duration of the gradient | seconds |
| `raw.cycle_length` | Number of scans per cycle | count |
| `raw.cycle_duration` | Average duration of each cycle | seconds |
| `raw.cycle_number` | Total number of cycles in the run | count |
| `raw.ms2_range_min` | Minimum MS2 m/z value measured | - |
| `raw.ms2_range_max` | Maximum MS2 m/z value measured | - |
| **Search Level** | | |
| `search.channel` | Channel number (0 for label-free, or channel numbers for multiplexed data) | - |
| `search.precursors` | Number of identified precursors in this run/channel | count |
| `search.proteins` | Number of unique protein groups identified in this run/channel | count |
| `search.fwhm_rt` | Mean full width at half maximum of peaks in retention time | seconds |
| `search.fwhm_mobility` | Mean FWHM of peaks in mobility dimension (ion mobility data only) | mobility units |
| **Optimization Level** | | |
| `optimization.ms2_error` | Final MS2 mass error tolerance used | ppm |
| `optimization.ms1_error` | Final MS1 mass error tolerance used | ppm |
| `optimization.rt_error` | Final retention time tolerance used | seconds |
| `optimization.mobility_error` | Final ion mobility tolerance used (ion mobility data only) | mobility units |
| **Calibration Level** | | |
| `calibration.ms2_bias` | Median mass bias for fragment ions | ppm |
| `calibration.ms2_variance` | Median mass variance for fragment ions | ppm |
| `calibration.ms1_bias` | Median mass bias for precursor ions | ppm |
| `calibration.ms1_variance` | Median mass variance for precursor ions | ppm |

**Notes:**
- Some columns may be NaN if the corresponding measurements or calibrations were not performed
- For label-free data, there will typically be one row per run with channel=0
- For multiplexed data, there will be multiple rows per run (one for each channel)
- **Important**: The `search.precursors` and `search.proteins` counts represent **identification** statistics (precursors/proteins that passed protein FDR), while the quantification matrices (see below) contain **quantification** statistics (a subset that passed additional quality filters for LFQ). Typically ~3-4% of identified precursors may lack quantification values due to insufficient fragment quality, poor correlation, or failing directLFQ thresholds. This is expected behavior and indicates the difference between identification (broader) and quantification (stricter quality requirements).


## `pg.matrix.parquet`
The protein group quantification matrix provides protein-level quantification across all samples.
It contains one row per protein group and one column per sample.

**Important**: This matrix contains only protein groups with valid quantification values. The number of non-zero entries per sample may be slightly lower (~0.3-0.8%) than the `search.proteins` count in `stats.tsv`, which reports all identified proteins. The difference represents proteins that were identified but could not be quantified due to insufficient fragment data or quality.

## `peptide.matrix.parquet`
The peptide quantification matrix provides peptide-level quantification across all samples (when peptide-level LFQ is enabled).
It contains one row per peptide and one column per sample.

**Important**: This matrix contains only peptides with valid quantification values. Peptides that were identified but failed quality filters for LFQ will have missing (NaN) values or may be absent from the matrix entirely.

## `precursor.matrix.parquet`
The precursor quantification matrix provides precursor-level quantification across all samples (when precursor-level LFQ is enabled).
It contains one row per precursor and one column per sample.

**Important**: This matrix contains only precursors with valid quantification values. The number of non-zero entries per sample will be lower (~3-4%) than the `search.precursors` count in `stats.tsv`. The difference represents precursors that were identified but failed quantification quality filters such as:
- Poor fragment quality or correlation (below `min_correlation` threshold)
- Insufficient fragments (fewer than `min_k_fragments`)
- Insufficient non-missing values (below `min_nonnan` threshold for directLFQ)

This is expected behavior and reflects the distinction between identification (passing FDR) and quantification (passing additional quality requirements).

## `internal.tsv`
Internal statistics and timing information from the search process.

## `speclib.hdf`
The input spectral library as it was loaded and preprocessed. This includes isotope calculation, library prediction, retention time normalization, and FASTA annotation.

## `speclib.mbr.hdf`
The match-between-runs (MBR) output library containing all precursors which were identified in the search for second-step quantification.

- All high-confidence precursor identifications from the search
- Empirically optimized retention times and ion mobilities
- Only created when `general.save_mbr_library: true` in the configuration

## `speclib.transfer.hdf`
The transfer learning library containing training data for sample-specific model refinement.

- High-confidence precursor identifications
- All requested fragment types (not just top fragments)
- Observed intensities (not predicted values)
- Empirically measured retention times and ion mobilities

## `quant/` folder
The `quant/` folder contains per-file quantification results used for checkpointing and distributed searches.

Structure:
```
quant/
├── <raw_file_1>/
│   ├── psm.parquet      # PSM-level data for the raw file
│   └── frag.parquet     # Fragment-level data for the raw file
├── <raw_file_2>/
│   ├── psm.parquet
│   └── frag.parquet
└── ...
```

If the files `psm.parquet` and `frag.parquet` are available in the `quant/` folder, these values will be reused when `reuse_quant` is enabled in the configuration. This allows for efficient re-analysis without re-extracting quantification data from raw files.

See the [restarting documentation](./command-line.md#restarting) for more details on using the `--quant-dir` parameter and `reuse_quant` configuration.
