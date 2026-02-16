# Hidden Decoys (Entrapment) Implementation Plan

## Concept

A random N% of generated decoys are marked as "hidden decoys". They keep `decoy=0` throughout calibration and optimization (fully invisible — participate in FDR training as targets, influence calibration). At the final extraction step per raw file, their `decoy` flag is flipped to `1` before FDR computation.

## New column: `is_hidden_decoy` (bool)

Lives on `precursor_df` in the spectral library. `True` = hidden decoy, `False` = everything else.

## Changes

### 1. Config — `alphadia/constants/default.yaml`

Add under `fdr:` section (after `enable_nn_hyperparameter_tuning`):

```yaml
  # Fraction of decoys to treat as hidden entrapment targets (0.0 = disabled)
  hidden_decoy_fraction: 0.0
```

Default `0.0` = no behavior change for existing users.

### 2. DecoyGenerator — `alphadia/libtransform/decoy.py`

- Add `hidden_decoy_fraction: float = 0.0` constructor parameter.
- In `forward()`, after setting `decoy_lib._precursor_df["decoy"] = 1` (line 48):
  - Initialize `is_hidden_decoy = False` on both target and decoy DataFrames.
  - If `hidden_decoy_fraction > 0`, randomly select N% of decoys, flip their `decoy` to `0`, set `is_hidden_decoy = True`.
  - Use `np.random.default_rng(seed=42)` for reproducibility.

### 3. Wire config to DecoyGenerator — `alphadia/search_step.py`

At line 358, pass `hidden_decoy_fraction=self.config["fdr"]["hidden_decoy_fraction"]` to `DecoyGenerator`.

### 4. Column propagation — Python backend

`alphadia/search/scoring/scoring.py` line 93-107: Add `"is_hidden_decoy"` to `DEFAULT_PRECURSOR_COLUMNS` (after `"decoy"`).

### 5. Column propagation — NG/Rust backend

`alphadia/workflow/peptidecentric/ng/ng_mapper.py`:
- `parse_candidates()` line 129: Add `"is_hidden_decoy"` to the merge column list.
- `to_features_df()` line 176: Add `"is_hidden_decoy"` to the merge column list.

### 6. Reveal — `alphadia/workflow/peptidecentric/peptidecentric.py`

In `extraction()`, insert reveal logic before FDR computation for each backend:

**Python backend** (before line 219, the `fit_predict` call):
```python
if "is_hidden_decoy" in precursor_quantified_w_features_df.columns:
    precursor_quantified_w_features_df.loc[
        precursor_quantified_w_features_df["is_hidden_decoy"], "decoy"
    ] = 1
```

**Rust/NG backend** (before line 250, the `perform_fdr_and_filter_candidates` call):
```python
if "is_hidden_decoy" in precursor_w_features_df.columns:
    precursor_w_features_df.loc[
        precursor_w_features_df["is_hidden_decoy"], "decoy"
    ] = 1
```

### 7. No changes needed in

- `fdr_manager.py` — hidden decoys naturally flow as targets during cal/opt.
- `optimization_handler.py` `_filter_dfs()` — hidden decoys pass `decoy==0` filter naturally.
- `optimization_lock.py` — hidden decoys count as targets, which is intended.
- `outputtransform/` — after reveal, hidden decoys have `decoy=1` and are filtered by existing logic.

## Files to modify (summary)

| File | Change |
|------|--------|
| `alphadia/constants/default.yaml` | Add `hidden_decoy_fraction: 0.0` |
| `alphadia/libtransform/decoy.py` | Core logic: select hidden decoys, set column |
| `alphadia/search_step.py` | Pass config to DecoyGenerator |
| `alphadia/search/scoring/scoring.py` | Add `is_hidden_decoy` to `DEFAULT_PRECURSOR_COLUMNS` |
| `alphadia/workflow/peptidecentric/ng/ng_mapper.py` | Add `is_hidden_decoy` to two merge column lists |
| `alphadia/workflow/peptidecentric/peptidecentric.py` | Reveal hidden decoys before final FDR |

## Edge cases

- **`hidden_decoy_fraction=0.0`**: No hidden decoys created, `is_hidden_decoy` column still added (all `False`). Zero behavior change.
- **Small library**: `int(n_decoys * fraction)` could be 0 → log warning, degrade to standard behavior.
- **Calibration contamination**: Hidden decoys that pass 1% FDR have high target similarity by definition; impact on calibration models is bounded and small.
- **MBR step**: Uses `DecoyGenerator` with default `hidden_decoy_fraction=0.0` — not affected.
- **Column missing**: Guard `if "is_hidden_decoy" in df.columns` ensures no crash if column dropped.

## Verification

1. Run unit tests: `cd tests && pytest unit_tests/libtransform/`
2. Run full unit test suite: `cd tests && pytest unit_tests/`
3. Manual verification: Run with `hidden_decoy_fraction: 0.2` on a test dataset and check:
   - Log output shows N hidden decoys created
   - Final output has fewer identified precursors (some hidden decoys revealed)
   - With `keep_decoys: true`, hidden decoys visible in output with `is_hidden_decoy=True`
