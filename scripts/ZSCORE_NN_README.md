# ZScoreNNClassifier

Two-stage FDR classifier: z-score pre-filter on rank 0, then NN on survivors.
Drop-in replacement for `BinaryClassifierLegacyNewBatching`.

## How it works

1. **Z-score stage**: Computes z-score from 5 features on rank 0 targets vs decoys. Finds score threshold at 50% FDR. All candidates below threshold are rejected.
2. **NN stage**: Trains `BinaryClassifierLegacyNewBatching` on survivors only (all features except rank). Non-survivors get probability `[0, 1]` (certain decoy).

## Prerequisites

Candidates must be pre-filtered with `n_matched_strict >= 3` **before** passing to the classifier. This hard filter is not performed inside the classifier — it is a candidate selection concern handled upstream.

```python
df = df[df["n_matched_strict"] >= 3]
```

## Usage with `perform_fdr`

```python
from alphadia.fdr.fdr import perform_fdr
from zscore_nn_classifier import ZScoreNNClassifier

# 'rank' must be in available_columns
feature_names = [c for c in df.columns if c not in exclude_cols]
assert "rank" in feature_names

classifier = ZScoreNNClassifier(
    available_columns=feature_names,
    # NN kwargs forwarded to BinaryClassifierLegacyNewBatching:
    test_size=0.001,
    batch_size=5000,
    learning_rate=0.001,
    epochs=10,
    experimental_hyperparameter_tuning=True,
)

psm_df = perform_fdr(
    classifier,
    feature_names,
    df_target,
    df_decoy,
    competitive=True,
    group_channels=False,
)
```

## Important: `rank` column

The `rank` column **must** be included in `available_columns` passed to `perform_fdr`.
It is used internally by the z-score stage to identify rank 0 candidates, then stripped before NN training/prediction.

When integrating into alphadia's `fdr_manager`, ensure `rank` is not excluded from the feature columns:

```python
# In fdr_manager or extraction_handler, make sure rank is available:
available_columns = list(set(features_df.columns).intersection(set(self.feature_columns)))
# feature_columns must include "rank"
```

## Z-score features

Default 5 features (optimized on HeLa 21min):
- `num_over_0`
- `delta_rt` (abs-transformed)
- `idf_corr_mass_gaussian`
- `intensity_correlation`
- `idf_hyperscore`

Override via `zscore_features` parameter.

## Performance

Tested on HeLa 21min (14.7M candidates, 3 ranks):

| Method | @1% FDR | @5% FDR | Time |
|---|---|---|---|
| NN only | 95,215 | 123,264 | 73s |
| ZScoreNNClassifier | 95,199 | 116,792 | 16s |

Same @1% accuracy, 4.5x faster due to pre-filtering.

## Standalone script

`two_stage_fdr.py` in `z-score-two-step/` runs the same pipeline without the alphadia classifier interface:

```bash
python two_stage_fdr.py features_rescored.parquet
```
