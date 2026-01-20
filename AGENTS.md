# AGENTS.md

This file provides guidance to AI coding agents when working with code in this repository.

## Project Overview

alphaDIA is a proteomics search engine for DIA (Data-Independent Acquisition) mass spectrometry data. It supports empirical and predicted spectral library searches with transfer learning capabilities. Part of the AlphaPept ecosystem from Mann Labs.

Key features:
- Spectral library search (empirical and predicted)
- Transfer learning for RT, mobility, and MS2 models
- Label-free quantification (directLFQ, QuantSelect)
- Multi-step search (transfer → library → MBR)

## Build and Development Commands

### Environment Setup
```bash
conda create -n alphadia_env python=3.11 -y
conda activate alphadia_env
pip install -e ".[stable,development]"
```

### Running Tests
```bash
# All unit and integration tests (from tests/ directory)
cd tests && pytest

# Unit tests only
cd tests && pytest unit_tests

# Integration tests only
cd tests && pytest integration_tests

# Single test file
cd tests && pytest unit_tests/test_cli.py

# Single test function
cd tests && pytest unit_tests/test_cli.py::test_function_name

# Skip slow tests
cd tests && pytest -k "not slow"

# With coverage
cd tests && coverage run --source=../alphadia -m pytest && coverage html

# End-to-end tests (these take several minutes)
cd tests && ./run_e2e_tests.sh basic <name of conda environment>
```

### Linting and Type Checking
```bash
# Run pre-commit hooks (ruff format, ruff lint, ty type check)
pre-commit run --all-files

# Install pre-commit hooks (once)
pre-commit install
```

The project uses:
- **ruff** for formatting and linting
- **ty** (Astral's type checker) for type checking (some directories excluded, see pyproject.toml)

### Running alphaDIA
```bash
# Check installation
alphadia --check

# Run search
alphadia --config path/to/config.yaml --output /path/to/output

# Command help
alphadia -h
```

## Architecture Overview

### Entry Points
- `alphadia/cli.py` - CLI entry point, parses arguments and launches `SearchPlan`
- `alphadia = "alphadia.cli:run"` - Package entry point defined in pyproject.toml

### Core Search Pipeline

```
SearchPlan (search_plan.py)
    └── orchestrates multi-step searches (transfer → library → MBR)
    └── SearchStep (search_step.py)
            └── owns config, spectral library, raw file list
            └── PeptideCentricWorkflow (workflow/peptidecentric/)
                    └── per-raw-file workflow execution
                    └── calibration, optimization, extraction, FDR
```

**SearchPlan**: Top-level orchestrator for single or multi-step searches. Handles step sequencing and passing optimized parameters between steps.

**SearchStep**: Manages a single search step. Owns library loading/building, config initialization, and iterates over raw files.

**WorkflowBase** (workflow/base.py): Base class for per-file workflows. Manages calibration_manager, optimization_manager, timing_manager, and raw data loading.

### Key Modules

- `alphadia/libtransform/` - Library transformation pipeline (loading, prediction, decoys, flattening)
- `alphadia/search/` - Core search algorithms (python version, the rust version is imported from `alphadia_search_rs`)
  - `search/scoring/` - Candidate scoring and feature extraction
  - `search/selection/` - Candidate selection using FFT-based methods
  - `search/jitclasses/` - Numba JIT-compiled data structures
- `alphadia/fdr/` - FDR control and classifiers
- `alphadia/calibration/` - RT, m/z, mobility calibration
- `alphadia/outputtransform/` - Output processing and quantification
- `alphadia/workflow/managers/` - Workflow component managers (calibration, optimization, FDR, timing)
- `alphadia/constants/` - Default config (`default.yaml`), keys, settings

### Configuration System

Config is hierarchical with override order:
1. `alphadia/constants/default.yaml` (base defaults)
2. User config file (`--config`)
3. CLI parameters (`--file`, `--library`, etc.)
4. Extra config (for multi-step orchestration)

Config class in `workflow/config.py` handles merging and validation.

### External Dependencies

Core scientific stack from AlphaPept ecosystem (https://github.com/MannLabs):
- **alphabase** - Base spectral library classes (`SpecLibBase`, `SpecLibFlat`)
- **alpharaw** - Raw file reading (Thermo, Sciex)
- **alphatims** - Bruker TimsTOF support
- **alphapeptdeep** - Deep learning models for property prediction
- **directlfq** - Label-free quantification

## Test Structure

- `tests/unit_tests/` - Fast unit tests mirroring `alphadia/` structure
- `tests/integration_tests/` - Integration tests requiring test data
- `tests/e2e_tests/` - End-to-end tests with real data
- `tests/performance_tests/` - Performance benchmarks

pytest marker: `@pytest.mark.slow` for slow tests
