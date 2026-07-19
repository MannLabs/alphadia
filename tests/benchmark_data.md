# Benchmark datasets

Larger example/benchmark datasets for running and profiling alphaDIA searches are
hosted on S3 at `s3://aplusia-public-data/` (access requires AWS credentials).

| Prefix | Contents |
| --- | --- |
| `helaft_5ng_4min/` | 6 × HeLa 5 ng, 3.9 min raw files (`..._Pierce_HeLa_01–06_repeat.raw`) plus an example `config.yaml` |
| `speclibs/` | HeLa spectral libraries — `speclib_hela_entr.hdf` (base) and `speclib_hela_entr_flat.hdf` (flat) |
| `runs/` | Outputs and logs from a prior search |

Point `library_path` at the **base** library (`speclib_hela_entr.hdf`); repoint the
`raw_paths` and `output_directory` in `config.yaml` to your local copies. To use the
pre-flattened `speclib_hela_entr_flat.hdf` instead, set `general.input_library_type: flat`
— this requires `general.mbr_step_enabled: false`, since flat libraries are incompatible
with the MBR step.
