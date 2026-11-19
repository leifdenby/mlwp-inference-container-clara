"""The interior inputs taken from DANRA (hindcasts)."""
import datetime
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from clara_inference import danra_interior as di

H = datetime.timedelta(hours=1)
T0 = np.datetime64("1999-12-01T00:00", "ns")
DANRA_LEVELS = [
    1000,
    950,
    925,
    900,
    850,
    800,
    700,
    600,
    500,
    400,
    300,
    250,
    200,
    100,
]


def _create_danra(n_days: int = 5) -> tuple[xr.Dataset, xr.Dataset]:
    """
    Create small DANRA-like single and pressure level datasets, 3-hourly from
    1999-12-01, whose values are the hours since 1999-12-01.

    Parameters
    ----------
    n_days : int
        Number of days.

    Returns
    -------
    tuple[xr.Dataset, xr.Dataset]
        Single levels (with extra variables and the statics) and pressure
        levels (with extra variables and levels), with DANRA's coordinates.
    """
    times = T0 + np.arange(8 * n_days) * np.timedelta64(3, "h")
    hours = ((times - T0) / np.timedelta64(1, "h")).astype("f4")
    nx, ny = 4, 3
    x, y = np.arange(nx) * 2500.0, np.arange(ny) * 2500.0
    lon, lat = np.meshgrid(x / 1e5, y / 1e5)
    coords = dict(
        time=times,
        x=x,
        y=y,
        lat=(("y", "x"), lat),
        lon=(("y", "x"), lon),
        danra_projection=0,
    )
    ds_sl = xr.Dataset(
        {
            v: (
                ("time", "y", "x"),
                np.broadcast_to(hours[:, None, None], (times.size, ny, nx)),
            )
            for v in di.SL_VARS + ["cape_column"]
        },
        coords=coords,
    )
    ds_sl["lsm"] = (("y", "x"), np.ones((ny, nx), "f4"))
    ds_sl["orography"] = (("y", "x"), np.zeros((ny, nx), "f4"))
    ds_pl = xr.Dataset(
        {
            v: (
                ("time", "pressure", "y", "x"),
                np.broadcast_to(
                    hours[:, None, None, None],
                    (times.size, len(DANRA_LEVELS), ny, nx),
                ),
            )
            for v in di.PL_VARS + ["ciwc"]
        },
        coords=dict(coords, pressure=DANRA_LEVELS),
    )
    return ds_sl, ds_pl


def test_get_valid_times() -> None:
    valid_times = di.get_valid_times(
        datetime.datetime(1999, 12, 2, 21), 27 * H
    )
    # analysis time .. +27 h, plus INTERIOR_EXTRA_STEPS
    assert valid_times[0] == np.datetime64("1999-12-02T21:00", "ns")
    assert valid_times[-1] == np.datetime64("1999-12-04T06:00", "ns")
    assert valid_times.size == 27 // 3 + di.INTERIOR_EXTRA_STEPS + 1


def test_select_danra_interior() -> None:
    ds_sl, ds_pl = _create_danra()
    valid_times = di.get_valid_times(
        datetime.datetime(1999, 12, 2, 21), 27 * H
    )
    ds_sl_out, ds_pl_out = di.select_danra_interior(ds_sl, ds_pl, valid_times)
    assert sorted(ds_sl_out.data_vars) == sorted(di.SL_VARS + di.STATIC_VARS)
    assert sorted(ds_pl_out.data_vars) == sorted(di.PL_VARS)
    assert ds_pl_out.pressure.values.tolist() == di.LEVELS
    # the coordinates the datastore config uses, not DANRA's others
    assert {"lat", "lon"} <= set(ds_sl_out.coords)
    assert "danra_projection" not in ds_sl_out.coords
    # unchanged values at the valid times (45 h after T0 is 02T21)
    np.testing.assert_array_equal(
        ds_sl_out.t2m.isel(x=0, y=0).values,
        45 + 3 * np.arange(valid_times.size),
    )
    assert ds_sl_out.lsm.dims == ("y", "x")


def test_select_danra_interior_missing_times() -> None:
    ds_sl, ds_pl = _create_danra(n_days=3)
    valid_times = di.get_valid_times(
        datetime.datetime(1999, 12, 2, 21), 27 * H
    )
    with pytest.raises(ValueError, match="DANRA has no valid times"):
        di.select_danra_interior(ds_sl, ds_pl, valid_times)


def test_extract_danra_interior(tmp_path: Path) -> None:
    ds_sl, ds_pl = _create_danra()
    danra_root = tmp_path / "danra"
    ds_sl.chunk(time=16).to_zarr(danra_root / "single_levels.zarr")
    ds_pl.chunk(time=16).to_zarr(danra_root / "pressure_levels.zarr")
    di.extract_danra_interior(
        datetime.datetime(1999, 12, 2, 21),
        27 * H,
        tmp_path / "inputs",
        danra_root=str(danra_root),
    )
    ds_sl_out = xr.open_zarr(
        tmp_path / "inputs" / "interior_single_levels.zarr"
    )
    ds_pl_out = xr.open_zarr(
        tmp_path / "inputs" / "interior_pressure_levels.zarr"
    )
    assert ds_sl_out.time.size == ds_pl_out.time.size == 12
    assert ds_pl_out.u.sel(pressure=850).dims == ("time", "y", "x")
    assert ds_sl_out.attrs["source"] == str(danra_root)
