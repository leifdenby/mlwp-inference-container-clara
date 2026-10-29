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


MINI_CONFIG = """
schema_version: v0.5.0
dataset_version: v0.1.0
output:
  variables:
    state: [time, grid_index, state_feature]
    forcing: [time, grid_index, forcing_feature]
  coord_ranges:
    time: {{start: 2016-01-01T00:00, end: 2016-01-07T00:00, step: PT3H}}
  chunking: {{time: 1, state_feature: 20}}
  splitting:
    dim: time
    splits:
      train:
        start: 2016-01-01T00:00
        end: 2016-01-07T00:00
        compute_statistics:
          ops: [mean, std, diff_mean, diff_std]
          dims: [grid_index, time]
inputs:
  sl:
    path: {path}
    dims: [time, x, y]
    variables: [t2m, u10m]
    dim_mapping:
      time: {{method: rename, dim: time}}
      grid_index: {{method: stack, dims: [x, y]}}
      state_feature:
        method: stack_variables_by_var_name
        name_format: "{{var_name}}"
    target_output_variable: state
  pl:
    path: {path}
    dims: [time, x, y, pressure]
    variables:
      t:
        pressure: {{values: [500, 850], units: hPa}}
    dim_mapping:
      time: {{method: rename, dim: time}}
      state_feature:
        method: stack_variables_by_var_name
        dims: [pressure]
        name_format: "{{var_name}}{{pressure}}"
      grid_index: {{method: stack, dims: [x, y]}}
    target_output_variable: state
  forcing:
    path: {path}
    dims: [time, x, y]
    derived_variables:
      hour_of_day_sin:
        kwargs: {{time: ds_input.time, component: sin}}
        function: mllam_data_prep.ops.derive_variable.time_components.calculate_hour_of_day
    dim_mapping:
      time: {{method: rename, dim: time}}
      grid_index: {{method: stack, dims: [x, y]}}
      forcing_feature:
        method: stack_variables_by_var_name
        name_format: "{{var_name}}"
    target_output_variable: forcing
"""


def test_extract_matches_the_config_chunking(tmp_path):
    """
    `_extract` (with its own output chunking) gives the values of
    mllam-data-prep with the config's
    """
    import mllam_data_prep as mdp

    from clara_inference.artifact.compute_stats import _block_config

    rng = np.random.default_rng(5)
    times = SOURCE_TIMES[:48]
    shape = (times.size, 4, 6)
    coords = dict(
        time=times, x=np.arange(4.0), y=np.arange(6.0), pressure=[500, 850]
    )
    source = xr.Dataset(
        {
            "t2m": (("time", "x", "y"), rng.normal(size=shape)),
            "u10m": (("time", "x", "y"), rng.normal(size=shape)),
            "t": (
                ("time", "pressure", "x", "y"),
                rng.normal(size=(times.size, 2, 4, 6)),
            ),
        },
        coords=coords,
    ).chunk(time=16, pressure=1, x=2, y=3)
    source.to_zarr(tmp_path / "source.zarr")
    fp_config = tmp_path / "config.yaml"
    fp_config.write_text(MINI_CONFIG.format(path=tmp_path / "source.zarr"))
    config = mdp.Config.from_yaml_file(fp_config)

    start, end = times[5], times[30]
    extracted = danra_stats._extract(config, start, end, 0)
    expected = mdp.create_dataset(
        config=_block_config(
            config, danra_stats._py(start), danra_stats._py(end)
        )
    )
    for category in ["state", "forcing"]:
        got_times, x, feature = extracted[category]
        np.testing.assert_array_equal(got_times, expected.time.values)
        np.testing.assert_array_equal(
            feature.values, expected[f"{category}_feature"].values
        )
        np.testing.assert_array_equal(
            x, expected[category].transpose("time", "grid_index", ...).values
        )


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
