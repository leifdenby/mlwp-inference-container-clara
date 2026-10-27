"""
Tests for `clara_inference.upload_outputs`, against fsspec's in-memory
filesystem instead of S3.

    uv run pytest tests
"""
import fsspec
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from clara_inference import upload_outputs as uo


def _outputs(tmp_path, n_times=3, value=1.0):
    outputs = tmp_path / "outputs"
    times = pd.date_range("2026-10-01T06:00", periods=n_times, freq="3h")
    for name in uo.OUTPUT_NAMES:
        ds = xr.Dataset(
            {"t": (("time", "y", "x"), np.full((n_times, 2, 3), value))},
            coords=dict(time=times),
        )
        # one chunk per time: t/0.0.0, t/1.0.0, ...
        ds = ds.chunk(time=1)
        ds.to_zarr(outputs / name, mode="w", consolidated=True)
    return outputs


def test_target_prefix():
    assert (
        uo.target_prefix("s3://clara-forecasts/", "2026-10-01T00:00")
        == "s3://clara-forecasts/2026-10-01T000000Z"
    )
    assert (
        uo.target_prefix("s3://b/runs", np.datetime64("2026-10-01T06:00"))
        == "s3://b/runs/2026-10-01T060000Z"
    )


def test_upload_replaces_earlier_upload(tmp_path):
    fs = fsspec.filesystem("memory")
    prefix = "memory://bucket/forecasts/"
    # an earlier, longer forecast for the same analysis time
    earlier = uo.upload_outputs(
        _outputs(tmp_path / "a", n_times=9), "2026-10-01T00:00", prefix, fs=fs
    )
    assert fs.exists(f"{earlier}/single_levels.zarr/t/8.0.0")
    target = uo.upload_outputs(
        _outputs(tmp_path / "b", n_times=3, value=2.0),
        "2026-10-01T00:00",
        prefix,
        fs=fs,
    )
    assert target == "memory://bucket/forecasts/2026-10-01T000000Z"
    for name in uo.OUTPUT_NAMES:
        ds = xr.open_zarr(fs.get_mapper(f"{target}/{name}"))
        # only the new upload, no stale chunks of the earlier one
        assert ds.time.size == 3 and float(ds.t.max()) == 2.0
        assert not fs.exists(f"{target}/{name}/t/8.0.0")
    fs.rm("/bucket", recursive=True)


def test_upload_needs_outputs(tmp_path):
    with pytest.raises(FileNotFoundError):
        uo.upload_outputs(
            tmp_path,
            "2026-10-01T00:00",
            "memory://bucket/",
            fs=fsspec.filesystem("memory"),
        )
