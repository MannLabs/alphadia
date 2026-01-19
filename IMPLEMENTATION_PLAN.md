# Type Hints Implementation Plan

**Objective**: Add and fix type hints across the alphadia codebase to achieve zero type errors.

**Strategy**: Fix existing incorrect type hints first, then add missing ones. Work bottom-up through the dependency tree, starting with foundational modules and ending with entry points. Largely ignore the `search/` package as requested.

**Current Status**: 741 total diagnostics
- In `search/` package: ~341 errors (mostly ignored)
- Outside `search/` and tests: ~230 errors (our focus)
- In tests: ~170 errors

---

## Stage 1: Foundation Modules
**Goal**: Fix type hints in constants, exceptions, and base utilities
**Success Criteria**: Zero type errors in these modules
**Tests**: Run `uvx ty check --output-format concise` after each file
**Status**: Not Started

### Files to Fix:
1. ✅ `constants/keys.py` - 0 errors (already correct)
2. ✅ `exceptions.py` - 0 errors (already correct)
3. `utils.py` - Check if has errors

**Notes**: These are leaf modules with minimal dependencies.

---

## Stage 2: Core Infrastructure
**Goal**: Fix reporting and logging modules with custom logging levels
**Success Criteria**: Zero type errors in reporting/* modules
**Tests**: Verify custom PROGRESS logging level is properly typed
**Status**: Not Started

### Files to Fix:
1. `reporting/logging.py` - 10 errors
   - Custom PROGRESS level attribute issues
   - Need to add proper type stubs or protocol
2. `reporting/reporting.py` - 16 errors
   - `matplotlib.ticker`, `matplotlib.figure`, `matplotlib.image` module access
   - Default parameter type mismatches
   - Backend `__enter__`/`__exit__` issues

**Key Issues**:
- Logger.progress attribute doesn't exist on standard Logger
- Need to define custom logger protocol or use proper typing

---

## Stage 3: Calibration Module
**Goal**: Fix type hints in calibration module
**Success Criteria**: Zero type errors in calibration/*
**Tests**: Ensure numpy array types are properly annotated
**Status**: Not Started

### Files to Fix:
1. `calibration/estimator.py` - 7 errors
   - set[str] vs list[str] mismatch (line 137-138)
   - None subscripting issues (line 258, 261)
   - numpy floating type returns (line 293, 325)
2. `calibration/models.py` - 2 errors
   - numpy floating subscripting (line 205)
3. `calibration/plot.py` - 6 errors
   - Axes vs None default parameter
   - ndarray vs Axes argument issues

**Key Issues**:
- Proper numpy array typing with numpy.typing.NDArray
- Matplotlib Axes optional parameters
- numpy scalar types (floating[Any])

---

## Stage 4: FDR Module
**Goal**: Fix type hints in FDR classification modules
**Success Criteria**: Zero type errors in fdr/*
**Tests**: Check classifier interfaces are properly typed
**Status**: Not Started

### Files to Fix:
1. `fdr/_fdrx/base.py` - 4 errors
   - BaseEstimator attribute warnings (fit, predict_proba)
2. `fdr/_fdrx/models/two_step_classifier.py` - 7 errors
   - list[str] | None argument issues
   - Classifier attribute assignments
   - DataFrame | None return type mismatches
3. `fdr/classifiers.py` - 7 errors
   - tuple[int, int | float] vs tuple[int | float, int | float]
   - FeedForwardNN attribute access
   - DataFrame vs ndarray arguments
   - Custom __setattr__ attribute assignments
4. `fdr/fdr.py` - 3 errors
   - ndarray default None parameter
   - Classifier/TwoStepClassifier attribute access
5. `fdr/plotting.py` - 1 error
   - matplotlib.ticker access

**Key Issues**:
- sklearn BaseEstimator typing (use protocols or TypeVar)
- Optional types and None handling
- Neural network model typing

---

## Stage 5: Library Transformation Modules
**Goal**: Fix type hints in libtransform/* modules
**Success Criteria**: Zero type errors in libtransform/*
**Tests**: Check SpecLibBase/SpecLibFlat type hierarchies
**Status**: Not Started

### Files to Fix:
1. `libtransform/fasta_digest.py` - 1 error
   - None vs str argument (line 79)
2. `libtransform/flatten.py` - 2 errors
   - SpecLibBase attribute access (_fragment_cardinality_df)
3. `libtransform/harmonize.py` - 2 errors
   - numpy floating subscripting (line 192)
4. `libtransform/multiplex.py` - 2 errors
   - str vs object with precursor_df attribute

**Key Issues**:
- SpecLibBase vs SpecLibFlat inheritance hierarchy
- Private attribute typing
- String vs object type confusion

---

## Stage 6: Output Transformation Modules
**Goal**: Fix type hints in outputtransform/* modules
**Success Criteria**: Zero type errors in outputtransform/*
**Tests**: Check dataframe transformations are properly typed
**Status**: Not Started

### Files to Fix:
1. `outputtransform/outputaccumulator.py` - 5 errors
   - SpecLibFlat vs SpecLibBase return type
   - Optional SpecLibBase attribute/argument issues
2. `outputtransform/search_plan_output.py` - 13 errors
   - Logger.progress attribute
   - Optional types (str | None, SpecLibBase | None)
   - Config vs dict[Unknown, Unknown] type mismatches

**Key Issues**:
- SpecLib type hierarchy consistency
- Config object typing (Config vs dict)
- Custom logger typing

---

## Stage 7: Workflow Modules
**Goal**: Fix type hints in workflow/* modules (largest effort)
**Success Criteria**: Zero type errors in workflow/*
**Tests**: Integration tests should pass
**Status**: Not Started

### Files to Fix:
1. `workflow/config.py` - 5 errors
2. `workflow/base.py` - 1 error
3. `workflow/managers/calibration_manager.py` - 7 errors
4. `workflow/managers/optimization_manager.py` - 8 errors
5. `workflow/managers/fdr_manager.py` - 3 errors
6. `workflow/managers/raw_file_manager.py` - 1 error
7. `workflow/optimizers/base.py` - 1 error
8. `workflow/optimizers/targeted.py` - 12 errors
9. `workflow/optimizers/automatic.py` - 18 errors
10. `workflow/optimizers/optimization_lock.py` - 2 errors
11. `workflow/peptidecentric/peptidecentric.py` - 2 errors
12. `workflow/peptidecentric/utils.py` - 2 errors
13. `workflow/peptidecentric/library_init.py` - 3 errors
14. `workflow/peptidecentric/optimization_handler.py` - 11 errors
15. `workflow/peptidecentric/extraction_handler.py` - 14 errors
16. `workflow/peptidecentric/multiplexing_requantification_handler.py` - 6 errors
17. `workflow/peptidecentric/recalibration_handler.py` - 1 error
18. `workflow/peptidecentric/ng/ng_mapper.py` - 5 errors

**Key Issues**:
- Complex workflow orchestration typing
- Manager and handler interfaces
- Config object propagation

---

## Stage 8: Entry Points
**Goal**: Fix type hints in search orchestration and CLI
**Success Criteria**: Zero type errors in cli.py, search_plan.py, search_step.py
**Tests**: Run full CLI with `--check` flag
**Status**: Not Started

### Files to Fix:
1. `cli.py` - 1 error
   - Line 175: `_get_from_args_or_config` return type (str vs Unknown | None)
2. `search_plan.py` - 5 errors
3. `search_step.py` - 10 errors

**Key Issues**:
- CLI argument parsing types
- Config merging logic types
- Integration with all other modules

---

## Stage 9: Transfer Learning (Optional - Lower Priority)
**Goal**: Fix type hints in transferlearning module
**Success Criteria**: Zero type errors in transferlearning/*
**Tests**: Transfer learning tests pass
**Status**: Not Started

### Files to Fix:
1. `transferlearning/train.py` - 22 errors (highest count)

**Notes**: Can be deferred if not critical path.

---

## Common Patterns to Address

### 1. Optional Parameters with Wrong Defaults
```python
# WRONG
def foo(ax: Axes = None): ...

# RIGHT
def foo(ax: Axes | None = None): ...
```

### 2. Numpy Array Typing
```python
# Import
from numpy.typing import NDArray
import numpy as np

# Use
def foo(arr: NDArray[np.float64]) -> NDArray[np.float64]: ...
```

### 3. Matplotlib Types
```python
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# Access via pyplot or proper imports
fig: Figure = plt.figure()
```

### 4. Custom Logger Typing
```python
# Option 1: Protocol
from typing import Protocol

class ProgressLogger(Protocol):
    def progress(self, msg: str, *args, **kwargs) -> None: ...

# Option 2: Type stub or cast
```

### 5. SpecLib Type Hierarchy
```python
# Be explicit about SpecLibBase vs SpecLibFlat
# Check if private attributes should be in base class or subclass
```

### 6. Config Typing
```python
# Use Config type, not dict, when Config object expected
from alphadia.workflow.config import Config

def foo(config: Config) -> None: ...  # NOT dict
```

### 7. Fix for ProcessingStep (libtransform/base.py)

Change from:
```python
def validate(self, *args: typing.Any) -> bool:
def forward(self, *args: typing.Any) -> typing.Any:
```

To:
```python
def validate(self, input: typing.Any) -> bool:
def forward(self, input: typing.Any) -> typing.Any:
```

This allows subclasses to narrow the `input` parameter type while remaining compatible.

### 8. Fix for Backend (reporting/reporting.py)

Use explicit parameters instead of `*args`:
```python
def log_figure(self, name: str, figure: typing.Any, extension: str = "png", **kwargs: Any) -> None:
def log_event(self, name: str, value: typing.Any, exception: Exception | None = None, **kwargs: Any) -> None:
def log_metric(self, name: str, value: float | str, **kwargs: Any) -> None:
```

### 9. Fix for CalibrationModelProvider.register_model

Change type annotation to accept both types and factory functions:
```python
def register_model(
    self, model_name: str, model_template: type[CalibrationModel] | Callable[..., CalibrationModel]
) -> None:
```

---

## Testing Strategy

After each file/stage:
1. Run `uvx ty check --output-format concise` to verify fixes
2. Count remaining errors: `uvx ty check --output-format concise 2>&1 | grep "Found" | tail -1`
3. Run relevant unit tests if they exist
4. Commit with clear message indicating stage and file

---

## Progress Tracking

- [x] Stage 1: Foundation Modules (0 errors) ✅ **COMPLETE**
- [x] Stage 2: Core Infrastructure (26 errors → 0) ✅ **COMPLETE**
  - Custom Logger.progress method with TYPE_CHECKING
  - matplotlib imports and type fixes
  - Backend context manager protocol
- [x] Stage 3: Calibration Module (15 errors → 0) ✅ **COMPLETE**
  - NumPy scalar to Python float conversions
  - matplotlib Axes optional parameters
  - None checks for optional returns
- [x] Stage 4: FDR Module (excl. _fdrx/) (9 errors → 0) ✅ **COMPLETE**
  - nn.Module custom __setattr__ type: ignore
  - tuple return type int casting
  - Optional ndarray parameters
- [x] Stage 4b: cli.py (1 error → 0) ✅ **COMPLETE**
  - Return type str | None fix
- [x] Stage 5: Library Transformation (7 errors → 0) ✅ **COMPLETE**
  - SpecLibBase dynamically added attributes
  - numpy.percentile type annotations
  - Parameter type corrections (str → SpecLibBase)
- [ ] Stage 6: Output Transformation (~18 errors) **SKIPPED** (per user request)
- [ ] Stage 7: Workflow Modules (~100+ errors - largest stage)
- [x] Stage 8: Entry Points (search_plan.py, search_step.py) (13 errors → 0) ✅ **COMPLETE**
  - Logger.progress type checking with TYPE_CHECKING pattern
  - None subscripting with assertions for control flow
  - Path | None argument types with assertions
  - Return type annotations for Generator
- [ ] Stage 9: Transfer Learning (~22 errors - optional/deferred)

**Progress**:
- **Fixed**: ~78 errors across 6 completed stages
- **Remaining**: ~152 errors (excluding search/, fdr/_fdrx/, tests/)
- **Total Original**: ~230 errors
- **Completion**: ~34% of non-search/test errors fixed

---

## Detailed Error Analysis (from ty check)

**Updated Scope**: 47 type errors within the alphadia package (excluding `ignore/`, `misc/`, notebooks)

### 1. Invalid Method Override Errors (29 errors) - HIGH PRIORITY

**Location**: `libtransform/*.py` and `reporting/reporting.py`

**Root Cause**: Base class methods use `*args: Any` signatures, but subclasses use specific typed parameters, violating the Liskov Substitution Principle.

**Files affected**:
- `libtransform/base.py` - `ProcessingStep.validate()` and `ProcessingStep.forward()`
- `libtransform/decoy.py`, `fasta_digest.py`, `flatten.py`, `harmonize.py`, `loader.py`, `mbr.py`, `multiplex.py`, `prediction.py`
- `reporting/reporting.py` - `Backend.log_figure()`, `log_event()`, `log_metric()`

**Approach**: Change base class signatures from `*args: Any` to use a single `input: Any` parameter that subclasses can narrow.

### 2. Unused `type: ignore` Comments (11 warnings) - EASY

**Location**: Various files - these were likely added for a different type checker.

**Files**:
- `calibration/models.py:204`
- `calibration/plot.py:124, 125`
- `fdr/classifiers.py:507, 508, 510, 511`
- `fragcomp/fragcomp.py:279`
- `libtransform/harmonize.py:190`
- `raw_data/bruker.py:263`
- `reporting/reporting.py:545`
- `search_plan.py:121`

**Approach**: Remove unused `# type: ignore` directives.

### 3. Specific Type Errors (5 errors)

| File | Line | Error | Fix |
|------|------|-------|-----|
| `calibration/estimator.py` | 384 | Function passed instead of type | Widen type signature to accept `Callable` |
| `calibration/models.py` | 323 | sum() overload mismatch | Add explicit type annotation or cast |
| `fdr/plotting.py` | 138 | datetime.now(UTC) issue | Use `datetime.now(timezone.utc)` |
| `reporting/reporting.py` | 657 | Duplicate parameter | Fix the function call |

### 4. Possibly Missing Attributes (2 warnings)

| File | Line | Issue |
|------|------|-------|
| `fdr/classifiers.py` | 277 | `state_dict` might be missing |
| `search_step.py` | 529 | `log_string` might be missing |

**Approach**: Add proper null/type guards or narrow the types.

---

## Recommended Implementation Order

### Phase 1: Core Infrastructure (fixes cascade to many files)
1. `libtransform/base.py` - Fix base class signatures (resolves 22 override errors)
2. `reporting/reporting.py` - Fix Backend base class (resolves 4 override errors)

### Phase 2: Remove Stale Ignores
3. Remove all unused `# type: ignore` comments (11 files)

### Phase 3: Specific Fixes
4. `calibration/estimator.py` - Fix register_model type
5. `calibration/models.py` - Fix sum() call
6. `fdr/plotting.py` - Fix datetime.now()
7. `fdr/classifiers.py` - Fix possibly-missing-attribute
8. `search_step.py` - Fix possibly-missing-attribute

### Phase 4: Verification
9. Run `uvx ty check --output-format concise` to verify zero errors
10. Fix any remaining issues that surface
