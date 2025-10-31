# Output Format Breaking Changes

This document describes breaking changes to the output column names introduced in PR #709 and PR #726.

## Table of Contents

1. [Precursor Output (precursors.parquet)](#precursor-output-precursorsparquet)
2. [Precursor Matrix Output (precursor.matrix.parquet)](#precursor-matrix-output-precursormatrixparquet)
3. [Peptide Matrix Output (peptide.matrix.parquet)](#peptide-matrix-output-peptidematrixparquet)
4. [Protein Group Matrix Output (pg.matrix.parquet)](#protein-group-matrix-output-pgmatrixparquet)
5. [Statistics Output (stat.tsv)](#statistics-output-stattsv)
6. [Internal Table (internal.tsv)](#internal-table-internaltsv)
7. [References](#references)


## Precursor Output (precursors.parquet)

### Core Identification

| Old Name | New Name | Status |
|----------|----------|--------|
| `precursor_idx` | `precursor.idx` | Renamed |
| `elution_group_idx` | `precursor.elution_group_idx` | Renamed |
| `sequence` | `precursor.sequence` | Renamed |
| `charge` | `precursor.charge` | Renamed |
| `mods` | `precursor.mods` | Renamed |
| `mod_sites` | `precursor.mod_sites` | Renamed |
| `mod_seq_hash` | `precursor.mod_seq_hash` | Renamed |
| `mod_seq_charge_hash` | `precursor.mod_seq_charge_hash` | Renamed |
| `rank` | `precursor.rank` | Renamed |
| `naa` | `precursor.naa` | Renamed |

### Mass Measurements

| Old Name | New Name | Status |
|----------|----------|--------|
| `mz_library` | `precursor.mz.library` | Renamed |
| `mz` | **DROPPED** | Removed |
| `mz_observed` | `precursor.mz.observed` | Renamed |
| `mz_calibrated` | **DROPPED** | Removed |

### Retention Time Measurements

| Old Name | New Name | Status |
|----------|----------|--------|
| `rt_library` | `precursor.rt.library` | Renamed |
| `rt` | **DROPPED** | Removed |
| `rt_observed` | `precursor.rt.observed` | Renamed |
| `rt_calibrated` | `precursor.rt.calibrated` | Renamed |
| `cycle_fwhm` | `precursor.rt.fwhm` | Renamed |

### Mobility Measurements

| Old Name | New Name | Status |
|----------|----------|--------|
| `mobility_library` | `precursor.mobility.library` | Renamed |
| `mobility_observed` | `precursor.mobility.observed` | Renamed |
| `mobility_calibrated` | **DROPPED** | Removed |
| `mobility_fwhm` | `precursor.mobility.fwhm` | Renamed |

### Quality Scores

| Old Name | New Name | Status |
|----------|----------|--------|
| `qval` | `precursor.qval` | Renamed |
| `proba` | `precursor.proba` | Renamed |
| `score` | `precursor.score` | Renamed |

### Experimental Metadata

| Old Name | New Name | Status |
|----------|----------|--------|
| `channel` | `precursor.channel` | Renamed |
| `decoy` | `precursor.decoy` | Renamed |

### Protein Group Information

| Old Name | New Name | Status |
|----------|----------|--------|
| `pg` | `pg.name` | Renamed |
| `proteins` | `pg.proteins` | Renamed |
| `genes` | `pg.genes` | Renamed |
| `pg_master` | `pg.master_protein` | Renamed |
| `pg_qval` | `pg.qval` | Renamed |

### Raw File Metadata

| Old Name | New Name | Status |
|----------|----------|--------|
| `run` | `raw.name` | Renamed |

### Quantification

| Old Name   | New Name | Status |
|------------|----------|--------|
| `intensity` | `pg.intensity` |  Aggregated protein group intensity |
| N/A        | `precursor.intensity` | **NEW** - Precursor-level intensity |
| N/A        | `peptide.intensity` | **NEW** - Aggregated peptide-level intensity |

**Note**: These columns provide direct access to quantification data in the precursors file, making it easier to filter and analyze quantitative results without requiring joins to matrix files.

### Internal/Technical Columns (DROPPED)

The following columns were internal to AlphaDIA and have been removed from the output:

| Old Name | Status |
|----------|--------|
| `flat_frag_start_idx` | **DROPPED** |
| `flat_frag_stop_idx` | **DROPPED** |
| `scan_start` | **DROPPED** |
| `scan_center` | **DROPPED** |
| `scan_stop` | **DROPPED** |
| `frame_start` | **DROPPED** |
| `frame_center` | **DROPPED** |
| `frame_stop` | **DROPPED** |
| `i_0` | **DROPPED** |
| `i_1` | **DROPPED** |
| `i_2` | **DROPPED** |
| `i_3` | **DROPPED** |

## Precursor Matrix Output (precursor.matrix.parquet)

This file contains quantification data at the precursor level across samples.

### Changes

| Old Name | New Name | Status |
|----------|----------|--------|
| `mod_seq_charge_hash` | `precursor.mod_seq_charge_hash` | Renamed |
| N/A | `pg.name` | **NEW** - Added for easier joining |
| N/A | `precursor.sequence` | **NEW** - Added for easier interpretation |
| N/A | `precursor.mods` | **NEW** - Added for easier interpretation |
| N/A | `precursor.mod_sites` | **NEW** - Added for easier interpretation |
| N/A | `precursor.charge` | **NEW** - Added for easier interpretation |

**Note**: Additional columns now include precursor identification information (sequence, mods, mod_sites, charge, pg.name) to make the matrix file more self-contained and easier to work with without requiring joins.

Sample intensity columns (e.g., `20231017_OA2_TiHe_ADIAMA_HeLa_200ng_Evo011_21min_F-40_05`) remain unchanged.

## Peptide Matrix Output (peptide.matrix.parquet)

This file contains quantification data at the peptide level across samples.

Similar structure to precursor.matrix.parquet with appropriate peptide-level columns. Sample intensity columns remain unchanged.

## Protein Group Matrix Output (pg.matrix.parquet)

This file contains quantification data at the protein group level across samples.

### Changes

| Old Name | New Name | Status |
|----------|----------|--------|
| `pg` | `pg.name` | Renamed |

Sample intensity columns (e.g., `20231017_OA2_TiHe_ADIAMA_HeLa_200ng_Evo011_21min_F-40_05`) remain unchanged.

## Statistics Output (stat.tsv)

### Raw File Metadata

| Old Name | New Name | Status |
|----------|----------|--------|
| `run` | `raw.name` | Renamed |
| `raw.gradient_min_m` | **DROPPED** | Removed |
| `raw.gradient_max_m` | **DROPPED** | Removed |
| `raw.gradient_length_m` | `raw.gradient_length` | Renamed (suffix removed) |
| `raw.msms_range_min` | `raw.ms2_range_min` | Renamed (msms → ms2) |
| `raw.msms_range_max` | `raw.ms2_range_max` | Renamed (msms → ms2) |

The following raw file metadata columns remain unchanged:
- `raw.cycle_length`
- `raw.cycle_duration`
- `raw.cycle_number`

### Search Statistics

| Old Name | New Name | Status |
|----------|----------|--------|
| `channel` | `search.channel` | Renamed |
| `precursors` | `search.precursors` | Renamed |
| `proteins` | `search.proteins` | Renamed |
| `fwhm_rt` | `search.fwhm_rt` | Renamed |
| `fwhm_mobility` | `search.fwhm_mobility` | Renamed |

### Calibration Statistics

| Old Name | New Name | Status |
|----------|----------|--------|
| `calibration.ms2_median_accuracy` | `calibration.ms2_bias` | Renamed (accuracy → bias) |
| `calibration.ms2_median_precision` | `calibration.ms2_variance` | Renamed (precision → variance) |
| `calibration.ms1_median_accuracy` | `calibration.ms1_bias` | Renamed (accuracy → bias) |
| `calibration.ms1_median_precision` | `calibration.ms1_variance` | Renamed (precision → variance) |

**Note**: The terminology change from "accuracy/precision" to "bias/variance" reflects more statistically accurate terminology.

### Optimization Statistics

Optimization statistics columns (prefixed with `optimization.*`) retain their names:
- `optimization.ms1_error`
- `optimization.ms2_error`
- `optimization.rt_error`
- `optimization.mobility_error`

## Internal Table (internal.tsv)

The internal table remains **unchanged**:
- `run`
- `duration_total`
- `duration_load`
- `duration_optimization`
- `duration_extraction`




## References

- PR #709: https://github.com/MannLabs/alphadia/pull/709 - Semantic output names for precursor/peptide/protein levels
- PR #726: https://github.com/MannLabs/alphadia/pull/726 - Semantic stat keys
