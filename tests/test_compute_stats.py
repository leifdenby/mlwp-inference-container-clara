"""The statistics are those mllam-data-prep computes, from blocks."""
import numpy as np
import pytest

from clara_inference.artifact.compute_stats import RunningStats, SplitStats


def test_running_stats_combines_exactly():
    rng = np.random.default_rng(0)
    x = rng.normal(5.0, 3.0, size=(100, 7, 4))
    stats = RunningStats(4)
    for block in np.array_split(x, 6):
        stats.add(block)
    flat = x.reshape(-1, 4)
    np.testing.assert_allclose(stats.mean, flat.mean(axis=0), rtol=1e-12)
    np.testing.assert_allclose(stats.std, flat.std(axis=0), rtol=1e-12)


@pytest.mark.parametrize("block", [3, 5, 17, 100])
def test_split_stats_as_mllam_data_prep(block):
    """
    mllam-data-prep: mean/std of the values, mean of the first and std of
    the second time differences, over (time, grid_index)
    """
    rng = np.random.default_rng(1)
    n_times = 40
    times = np.arange(n_times)
    x = rng.normal(size=(n_times, 6, 3)).cumsum(axis=0) * [1.0, 10.0, 1e4]
    stats = SplitStats(3)
    start = 0
    while True:
        end = min(start + block, n_times)
        stats.add_block(times[start:end], x[start:end])
        if end == n_times:
            break
        start = end - 2  # overlap by two time steps
    result = stats.result()
    d1 = np.diff(x, axis=0)
    d2 = np.diff(d1, axis=0)
    expected = dict(
        mean=x.mean(axis=(0, 1)),
        std=x.std(axis=(0, 1)),
        diff_mean=d1.mean(axis=(0, 1)),
        diff_std=d2.std(axis=(0, 1)),
    )
    for op, values in expected.items():
        np.testing.assert_allclose(result[op], values, rtol=1e-10, err_msg=op)


def test_split_stats_needs_overlap():
    stats = SplitStats(1)
    x = np.zeros((5, 2, 1))
    stats.add_block(np.arange(5), x)
    with pytest.raises(ValueError, match="overlap"):
        stats.add_block(np.arange(4, 9), x)
