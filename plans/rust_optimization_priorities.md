# Rust Optimization Priorities

Profiled on HeLa 21min single-file search (Astral, ~5.6M library precursors).
All timings from zscore-NN run with flat lib, 7ppm ms2, 2 candidates.

## Time Budget (per-file workflow: ~147s)

| Component | Time | % | What it does |
|---|---|---|---|
| Rust extraction/scoring | 27s | 18% | Already in Rust |
| FDR: NN training | 30s | 20% | PyTorch feed-forward NN on survivors |
| FDR: data prep/q-values | 42s | 29% | Pandas dropna, to_numpy, sort, cumsum, q-value calc |
| Score filtering | 18s | 12% | Pandas boolean index on score cutoff |
| Z-score pre-filter | 11s | 7% | NumPy z-score on 5 features, threshold |
| Calibration fit+predict | 9s | 6% | LOESS regression (2-6 kernels) |
| Optimization overhead | 9s | 6% | Between-step bookkeeping |

Output phase (~115s additional): LFQ 98s, protein FDR 5s, I/O 12s.

## Ranked Optimization Opportunities

### 1. FDR Data Prep & Q-Value Pipeline (~42s -> ~5s)

**Current**: `fdr.py:perform_fdr` — pandas dropna, column selection, to_numpy conversion,
train/test split, post-prediction sort, cumulative FDR calculation, q-value assignment.
Called 18 times per file; final batch dominates (~13M candidates).

**Why Rust**: All operations are array-based. Pandas overhead (index management, object
creation, sort with multiple keys) is significant at 13M rows. The q-value calculation
(`get_q_values`) does sort + cumsum + monotonic correction — textbook vectorized Rust.

**Approach**: Single Rust function taking raw feature arrays + labels, returning q-values
and probabilities. Eliminates all pandas intermediate objects.

**Effort**: Medium. No external dependencies. Well-defined input/output contract.

### 2. NN Training (~30s -> ~5-10s)

**Current**: `classifiers.py:BinaryClassifierLegacyNewBatching` — 4-layer PyTorch NN
(100/50/20/5 units), BatchNorm, ReLU, Dropout. Adam optimizer, BCE loss. 10 epochs on
~10M samples (80% of 13M after z-score filter survivors).

**Why Rust**: PyTorch Python overhead: tensor allocation, GIL contention, per-batch
Python loop. The network is tiny (175 total hidden units) — dominated by data movement
not compute. The z-score pre-filter already reduces to ~300K survivors for the final
batch, making this less critical than before.

**Approach**: `tch-rs` (Rust PyTorch bindings) or pure ndarray implementation. The
network is simple enough for a custom implementation (matmul + activations).

**Effort**: High. Need to replicate training loop, BatchNorm state, hyperparameter
tuning. Alternatively, keep PyTorch but move data prep to Rust (see #1).

**Note**: With z-score pre-filter active, NN trains on ~300K not 13M, so this dropped
from 312s to 30s already. Further Rust optimization has diminishing returns.

### 3. Score Filtering (~18s -> ~2s)

**Current**: `extraction_handler.py:_apply_score_cutoff` — pandas boolean indexing on
`candidates_df["score"] > cutoff`. Applied to ~13M-row DataFrame in final extraction.

**Why Rust**: Simple threshold comparison on a single column. Currently requires full
DataFrame copy. Could be integrated into the Rust extraction backend as a post-processing
step, eliminating the Python-side DataFrame entirely.

**Approach**: Add score cutoff parameter to Rust `CandidateScoring`/`CandidateFeatureCollection`,
filter before returning to Python. Zero-copy.

**Effort**: Low. Minimal code change in Rust backend.

### 4. Z-Score Pre-Filter (~11s -> ~1s)

**Current**: `zscore_nn_classifier.py` — NumPy z-score on 5 features across all candidates.
`np.nan_to_num`, mean/std computation, element-wise scoring, threshold comparison.

**Why Rust**: Pure SIMD-friendly computation. 13M candidates x 5 features = 65M ops.
NumPy already vectorized but Python overhead for array allocation and NaN handling adds up.
Rayon parallelism across candidates is trivial.

**Approach**: Rust function: `zscore_filter(features: &[f64], means: &[f64], stds: &[f64],
signs: &[f64], threshold: f64) -> Vec<bool>`. Could combine with score filtering (#3).

**Effort**: Low. No dependencies. Pure arithmetic.

### 5. Calibration LOESS (~9s -> ~3s)

**Current**: `calibration/models.py:LOESSRegression` — tricubic kernel weighting,
polynomial feature expansion, per-kernel weighted least squares via `np.linalg.inv`.
2-6 kernels, applied to 1K-10K PSMs.

**Why Rust**: Small data but called 12 times per file. NumPy `linalg.inv` is already
BLAS-backed so compute is fast. Overhead is in Python object creation (sklearn
PolynomialFeatures, weight matrix broadcasting).

**Approach**: Pure Rust LOESS with inline polynomial expansion. Use `ndarray-linalg`.

**Effort**: Medium. Need to replicate tricubic kernel + weighted regression.
Limited impact (~6s total savings).

### 6. LFQ (~98s -> ~40s)

**Current**: External `directLFQ` package. Already parallelized across cores.
Runs after the per-file workflow in the output phase.

**Why Rust**: Would require reimplementing the entire directLFQ algorithm. Already
uses multiprocessing. Bottleneck is likely the per-protein estimation loop.

**Approach**: Not recommended as standalone effort. Better to profile directLFQ
first. Consider if `polars` could replace pandas operations within it.

**Effort**: Very high. External package dependency.

### 7. Protein FDR (~5s -> ~2s)

**Current**: `protein_fdr.py` — pandas groupby aggregation to protein level,
sklearn MLPClassifier training on ~10K proteins, q-value calculation.

**Why Rust**: Small dataset (10K proteins). Pandas groupby is the main overhead.
MLPClassifier trains in <1s on this scale.

**Approach**: Rust groupby aggregation. Keep sklearn for the tiny NN.

**Effort**: Low but minimal impact.

## Recommended Implementation Order

**Phase 1 — Quick wins (estimated -25s, ~1 week)**
1. Score filtering in Rust backend (#3)
2. Z-score pre-filter in Rust (#4)

**Phase 2 — FDR pipeline (estimated -35s, ~2 weeks)**
3. FDR data prep & q-value pipeline (#1)

**Phase 3 — Optional further gains (estimated -20s, ~2-3 weeks)**
4. NN training via tch-rs or custom Rust NN (#2)
5. Calibration LOESS (#5)

**Expected result**: Per-file workflow from ~147s to ~70-80s (~2x speedup).
Combined with flat lib loading, total search time from ~283s to ~190-200s.
