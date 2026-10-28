"""Converting neural-lam's ensemble prediction to DANRA-like datasets."""
import numpy as np
import pytest
import xarray as xr

from clara_inference import convert_output as co

LEVELS = [100, 200, 400, 600, 700, 850, 925, 1000]


def _prediction(n_members):
    features = co.SINGLE_LEVEL_VARS + [
        f"{v}{p}" for v in co.PRESSURE_LEVEL_VARS for p in LEVELS
    ]
    nx, ny, nt = 4, 3, 2
    dims = ["start_time", "elapsed_forecast_duration", "x", "y"]
    shape = [1, nt, nx, ny]
    if n_members:
        dims.insert(2, "member")
        shape.insert(2, n_members)
    dims.append("state_feature")
    shape.append(len(features))
    # value = 1000 * member + feature index
    data = np.zeros(shape, "f4") + np.arange(len(features))
    if n_members:
        data = data + 1000 * np.arange(n_members)[:, None, None, None]
    coords = dict(
        start_time=[np.datetime64("2026-10-26T09:00", "ns")],
        elapsed_forecast_duration=np.array([3, 6], "timedelta64[h]").astype(
            "timedelta64[ns]"
        ),
        x=np.arange(nx) * 2500.0,
        y=np.arange(ny) * 2500.0,
        state_feature=features,
    )
    return xr.Dataset({"state": (dims, data)}, coords=coords), features


def _grid():
    x, y = np.arange(4) * 2500.0, np.arange(3) * 2500.0
    lon, lat = np.meshgrid(x / 1e5, y / 1e5)
    return xr.Dataset(
        coords=dict(x=x, y=y, lat=(("y", "x"), lat), lon=(("y", "x"), lon))
    )


@pytest.mark.parametrize("n_members", [0, 3])
def test_convert_members(n_members):
    ds_pred, features = _prediction(n_members)
    ds_sl, ds_pl = co.convert(
        ds_pred, _grid(), np.datetime64("2026-10-26T06:00"), {"a": 1}
    )
    n = max(n_members, 1)
    assert ds_sl.t2m.dims == (co.MEMBER_DIM, "time", "y", "x")
    assert ds_pl.t.dims == (co.MEMBER_DIM, "time", "pressure", "y", "x")
    assert ds_sl[co.MEMBER_DIM].values.tolist() == list(range(n))
    assert ds_sl.attrs["ensemble_size"] == n
    # valid times: start_time + elapsed_forecast_duration
    assert ds_sl.time.values.tolist() == [
        np.datetime64("2026-10-26T12:00", "ns").astype(int),
        np.datetime64("2026-10-26T15:00", "ns").astype(int),
    ] or list(ds_sl.time.values) == [
        np.datetime64("2026-10-26T12:00", "ns"),
        np.datetime64("2026-10-26T15:00", "ns"),
    ]
    # the right member and feature end up in each variable
    i_t850 = features.index("t850")
    member = n - 1
    np.testing.assert_array_equal(
        ds_pl.t.sel(pressure=850).isel({co.MEMBER_DIM: member}).values,
        i_t850 + 1000 * member * (n_members > 0),
    )
    assert ds_sl.t2m.attrs["units"] == "K"
    assert ds_pl.r.attrs["units"] == "1"


@pytest.mark.parametrize("clip", [False, True])
def test_clip_to_physical_ranges_only_if_asked(clip):
    """Feature values are 0..N-1 (+1000 per member): r and swavr0m > range"""
    ds_pred, _ = _prediction(2)
    # negative values too (swavr0m is feature 6)
    ds_pred["state"] = ds_pred.state - 10.0
    ds_sl, ds_pl = co.convert(
        ds_pred, _grid(), np.datetime64("2026-10-26T06:00"), {"a": 1}, clip=clip
    )
    unclipped_sl, unclipped_pl = co.convert(
        ds_pred, _grid(), np.datetime64("2026-10-26T06:00"), {"a": 1}
    )
    if clip:
        assert float(ds_pl.r.min()) >= 0 and float(ds_pl.r.max()) <= 1
        assert float(ds_sl.swavr0m.min()) >= 0
        assert "r clipped to [0.0, 1.0]" in ds_pl.attrs["postprocessing"]
        assert "swavr0m clipped to [0.0, inf]" in ds_sl.attrs["postprocessing"]
        assert ds_pl.r.attrs["units"] == "1"
    else:
        assert float(ds_pl.r.max()) > 1 and float(ds_sl.swavr0m.min()) < 0
        assert "postprocessing" not in ds_sl.attrs
        assert "postprocessing" not in ds_pl.attrs
    # other variables are never changed
    np.testing.assert_array_equal(ds_sl.t2m.values, unclipped_sl.t2m.values)
    np.testing.assert_array_equal(ds_pl.t.values, unclipped_pl.t.values)
