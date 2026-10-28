"""Blocks of the resumable DANRA statistics combine to the whole split's."""
import numpy as np
import pytest
import xarray as xr

from clara_inference.artifact import danra_stats
from clara_inference.artifact.compute_stats import SplitStats

STEP = np.timedelta64(3, "h")
SOURCE_TIMES = np.datetime64("2016-01-01T00:00", "ns") + STEP * np.arange(60)


def test_chunk_blocks_aligned_to_source_chunks():
    blocks = danra_stats.chunk_blocks(
        SOURCE_TIMES, 8, SOURCE_TIMES[5], SOURCE_TIMES[30]
    )
    indices = [
        tuple(int(np.searchsorted(SOURCE_TIMES, t)) for t in b) for b in blocks
    ]
    assert indices == [(5, 7), (8, 15), (16, 23), (24, 30)]


def test_chunk_blocks_needs_source_times():
    with pytest.raises(ValueError, match="not a source time"):
        danra_stats.chunk_blocks(
            SOURCE_TIMES,
            8,
            SOURCE_TIMES[0] + np.timedelta64(1, "h"),
            SOURCE_TIMES[10],
        )


def test_split_stats_arrays_round_trip():
    rng = np.random.default_rng(3)
    stats = SplitStats(3)
    stats.add_block(np.arange(10), rng.normal(size=(10, 4, 3)))
    arrays = stats.to_arrays(prefix="state_")
    restored = SplitStats.from_arrays(arrays, prefix="state_")
    for op, values in stats.result().items():
        np.testing.assert_array_equal(restored.result()[op], values)


@pytest.fixture
def synthetic_source(monkeypatch):
    """`_extract` from a synthetic (time, grid_index, feature) variable."""
    rng = np.random.default_rng(4)
    x = rng.normal(size=(SOURCE_TIMES.size, 5, 2)).cumsum(axis=0)
    feature = xr.DataArray(["a", "b"], dims=["state_feature"])

    def _extract(config, start, end, max_retries):
        sel = (SOURCE_TIMES >= start) & (SOURCE_TIMES <= end)
        return {"state": (SOURCE_TIMES[sel], x[sel], feature)}

    monkeypatch.setattr(danra_stats, "_extract", _extract)
    return x


@pytest.mark.parametrize("sub_block_steps", [1, 3, 8])
@pytest.mark.parametrize("lose_tail", [None, 1, 2])
@pytest.mark.parametrize("i_start", [3, 7])
def test_blocks_combine_exactly(
    synthetic_source, sub_block_steps, lose_tail, i_start
):
    """
    As the whole split at once, also when the first block has a single time
    (`i_start` 7) and when a block's tail is missing (e.g. after resuming),
    which then reads up to two steps more (but not before the split)
    """
    x = synthetic_source
    i_end = 40
    whole = SplitStats(2)
    whole.add_block(SOURCE_TIMES[i_start : i_end + 1], x[i_start : i_end + 1])

    split_start = SOURCE_TIMES[i_start]
    blocks = danra_stats.chunk_blocks(
        SOURCE_TIMES, 8, split_start, SOURCE_TIMES[i_end]
    )
    combined, tail = None, None
    for k, (first, last) in enumerate(blocks):
        if k == lose_tail:
            tail = None
        stats, _, tail = danra_stats._process_block(
            None, first, last, STEP, sub_block_steps, tail, split_start, 0
        )
        if combined is None:
            combined = stats["state"]
        else:
            combined.merge(stats["state"])
    for op, values in whole.result().items():
        np.testing.assert_allclose(
            combined.result()[op], values, rtol=1e-10, err_msg=op
        )
