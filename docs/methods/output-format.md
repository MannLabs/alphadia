# Output format
AlphaDIA writes tab-separated values (TSV) files.

## `stats.tsv`
The `stats.tsv` file contains summary statistics and quality metrics for each run and channel in the analysis.
It provides insights into the search results, calibration quality, and general performance metrics.

Format: one row per run/channel combination.

### Columns

#### Basic Information
- `run`: Name of the raw file/run
- `channel`: Channel number (0 for label-free data, or channel numbers for multiplexed data)

#### Search Results Statistics
- `precursors`: Number of identified precursors in this run/channel
- `proteins`: Number of unique protein groups identified in this run/channel

#### Peak Width Statistics
These columns are only present if the data contains the relevant measurements:
- `fwhm_rt`: Mean full width at half maximum (FWHM) of peaks in retention time dimension
- `fwhm_mobility`: Mean FWHM of peaks in mobility dimension (only for ion mobility data)

#### Optimization Statistics
These metrics show the final values used for various search parameters after optimization:

- `optimization.ms2_error`: Final MS2 mass error tolerance used
- `optimization.ms1_error`: Final MS1 mass error tolerance used
- `optimization.rt_error`: Final retention time tolerance used
- `optimization.mobility_error`: Final ion mobility tolerance used (only for ion mobility data)

#### Calibration Statistics
These metrics show the quality of mass calibration:

##### MS2 Level
- `calibration.ms2_median_accuracy`: Median mass accuracy for fragment ions
- `calibration.ms2_median_precision`: Median mass precision for fragment ions

##### MS1 Level
- `calibration.ms1_median_accuracy`: Median mass accuracy for precursor ions
- `calibration.ms1_median_precision`: Median mass precision for precursor ions

### Notes

- All FWHM values are reported in the native units of the measurement (minutes for RT, mobility units for IM)
- Mass accuracy values are typically reported in parts per million (ppm)
- Some columns may be NaN if the corresponding measurements or calibrations were not performed
- For label-free data, there will typically be one row per run with channel=0
- For multiplexed data, there will be multiple rows per run (one for each channel)

#### Raw File Statistics
These metrics are derived from the raw data file analysis:

- `raw.gradient_min_m`: Minimum retention time in the run (minutes)
- `raw.gradient_max_m`: Maximum retention time in the run (minutes)
- `raw.gradient_length_m`: Total duration of the run (minutes)
- `raw.cycle_length`: Number of scans per cycle
- `raw.cycle_duration`: Average duration of each cycle in seconds
- `raw.cycle_number`: Total number of cycles in the run
- `raw.msms_range_min`: Minimum MS2 m/z value measured
- `raw.msms_range_max`: Maximum MS2 m/z value measured


## `precursors.tsv`
TODO

## `pg.matrix.tsv`
TODO

## `internal.tsv`
TODO

## `speclib.mbr.hdf `
TODO

## `quant folder`
TODO
