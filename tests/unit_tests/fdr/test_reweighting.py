import numpy as np

from alphadia.fdr.reweighting import JUNK_RANK, decoy_weights


def _shifted_decoys(n: int = 6000, shift: float = 1.0, seed: int = 0):
    """False targets sit at 0 in the first feature, decoys at `shift`; ranks are random."""
    rng = np.random.default_rng(seed)
    x = np.vstack([rng.normal(0.0, 1.0, (n, 3)), rng.normal(0.0, 1.0, (n, 3))])
    x[n:, 0] += shift
    y = np.concatenate([np.zeros(n), np.ones(n)])
    rank = rng.integers(0, JUNK_RANK + 2, size=2 * n)
    return x, y, rank


def test_decoy_weights_favour_the_decoys_that_look_like_false_targets():
    x, y, rank = _shifted_decoys()

    weight = decoy_weights(x, y, rank, random_state=0)

    assert np.all(weight[y == 0] == 1.0)
    np.testing.assert_allclose(weight[y == 1].mean(), 1.0)
    decoys = np.flatnonzero(y == 1)
    low, high = decoys[x[decoys, 0] < 0.5], decoys[x[decoys, 0] > 1.5]
    assert weight[low].mean() > 2 * weight[high].mean()


def test_decoy_weights_extrapolate_to_the_best_candidates():
    x, y, rank = _shifted_decoys()

    weight = decoy_weights(x, y, rank, random_state=0)

    rank0 = (y == 1) & (rank == 0)
    assert np.corrcoef(weight[rank0], -x[rank0, 0])[0, 1] > 0.7


def test_decoy_weights_are_uniform_when_too_few_candidates_are_false(caplog):
    x, y, _rank = _shifted_decoys(n=500)

    weight = decoy_weights(x, y, np.zeros(len(y), dtype=int), random_state=0)

    assert np.all(weight == 1.0)
    assert "Too few" in caplog.text
