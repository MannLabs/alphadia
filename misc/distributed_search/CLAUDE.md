# Distributed AlphaDIA Search Pipeline

## Overview

This is a parallel, multi-step proteomics analysis pipeline designed for HPC systems with Slurm. It distributes raw file processing across multiple nodes while maintaining the biological validity of the staged workflow.

## Directory Structure

```
distributed_search/
├── outer.sh                    # Main orchestration script
├── inner.sh                    # Worker script for each chunk
├── parse_parameters.py         # Splits raw files into chunks, generates configs
├── speclib_config.py           # Library prediction config generator
├── discover_project_files.py   # Raw file discovery utility
├── search.config               # Main configuration file (user edits this)
├── first_config.yaml           # First search parameters template
├── second_config.yaml          # Second search/MBR/LFQ parameters template
└── README.md                   # Setup documentation
```

## 5-Stage Workflow

```
1. PREDICT LIBRARY    (optional, single task) - generates spectral library from FASTA
        ↓
2. FIRST SEARCH       (parallelized into concurrent single tasks) - initial search across N chunks
        ↓
3. MBR LIBRARY BUILD  (single task) - aggregates first search, builds focused MBR library
        ↓
4. SECOND SEARCH      (parallelized into concurrent single tasks) - search with MBR library across N chunks
        ↓
5. LFQ QUANTIFICATION (single task) - final protein/precursor quantification
```

## Script Interactions

### outer.sh (Orchestrator)
- Reads `search.config` for paths and settings
- Creates output directory structure: `ad_search_<name>/1_predicted_speclib/`, `2_first_search/`, `3_mbr_library/`, `4_second_search/`, `5_lfq/`
- Submits each stage as Slurm jobs with `--wait` to ensure sequential execution
- Passes environment variables to inner.sh (target_directory, quant_dir, N_CPUS)

### parse_parameters.py (Chunk Generator)
- Calculates `chunk_size = ceil(num_rawfiles / num_nodes)`
- Creates `chunk_N/` directories with chunk-specific `config.yaml`
- Copies spectral library to each chunk (avoids concurrent read issues)
- Returns number of chunks (used for Slurm array size)
- Key args: `--nnodes`, `--reuse_quant`, `--library_path`

### inner.sh (Worker)
- Executed by each Slurm array element
- Uses `SLURM_ARRAY_TASK_ID` to find its chunk directory
- Runs `alphadia --config config.yaml` with thread count from `N_CPUS`
- Optionally aggregates quant outputs to `quant_dir`

### speclib_config.py
- Generates config for library prediction from FASTA
- Disables raw file processing, enables `library_prediction.enabled: True`

### discover_project_files.py
- Utility to find raw files matching regex patterns
- Outputs 2-column CSV: project | filepath

## Usage

```bash
# Basic usage with 45 parallel nodes (--files is required)
sbatch outer.sh --files file_list.csv --nnodes 45 --cpus 12 --mem 250G

# Skip library prediction (use existing library)
sbatch outer.sh --files file_list.csv --nnodes 45 --predict_library 0

# Only run first search
sbatch outer.sh --files file_list.csv --nnodes 45 --mbr_library 0 --second_search 0 --lfq 0
```

### Command-Line Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--files` | (required) | CSV file with raw file paths |
| `--search_config` | search.config | Search configuration file |
| `--nnodes` | 1 | Number of parallel nodes |
| `--cpus` | 32 | CPUs per task (thread_count) |
| `--mem` | '250G' | RAM per task |
| `--predict_library` | 1 | Enable library prediction |
| `--first_search` | 1 | Enable first search |
| `--mbr_library` | 1 | Enable MBR library building |
| `--second_search` | 1 | Enable second search |
| `--lfq` | 1 | Enable LFQ quantification |

## Configuration

### search.config (Main Config)
```bash
input_directory="/path/to/configs"        # Where config files live
target_directory="/path/to/output"        # Output directory
library_path="/path/to/speclib.hdf"       # Spectral library
fasta_path="/path/to/proteins.fasta"      # For library prediction
first_search_config_filename="first_config.yaml"
second_search_config_filename="second_config.yaml"
```

Note: The input CSV file is now specified via `--files` command-line argument instead of in search.config.

### Input CSV Format
Two columns: project name and absolute path to raw file
```
project,filepath
MyProject,/absolute/path/to/file1.raw
MyProject,/absolute/path/to/file2.raw
```

### YAML Config Templates
- `first_config.yaml`: Must have `mbr_step_enabled: false`
- `second_config.yaml`: Must have `inference_strategy: library`, `target_num_candidates: 5`, `mbr_step_enabled: false`

## Output Structure

```
ad_search_<basename>/
├── 1_predicted_speclib/
│   └── speclib.hdf
├── 2_first_search/
│   ├── chunk_0/
│   ├── chunk_1/
│   └── ...
├── 3_mbr_library/
│   └── chunk_0/
│       └── speclib.mbr.hdf
├── 4_second_search/
│   ├── chunk_0/
│   └── ...
└── 5_lfq/
    └── chunk_0/
        ├── precursor_table.csv
        └── protein_table.csv
```

## Key Implementation Details

1. **Parallelization**: First and second searches run as Slurm array jobs; MBR and LFQ are single-node aggregation steps
2. **Library copying**: Library is copied to each chunk directory to avoid concurrent HDF5 read issues
3. **Quant aggregation**: `quant_dir` environment variable directs quantification outputs to shared location for MBR/LFQ stages
4. **Chunk balancing**: `parse_parameters.py` ensures balanced distribution of raw files across nodes
5. **Environment**: Requires a mamba/conda environment with mono and alphadia installed activated when launching the shell scripts.
