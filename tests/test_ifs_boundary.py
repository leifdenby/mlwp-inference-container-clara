"""The IFS boundary: cycle choice, lead times, contract, grid and order."""
import datetime

import cartopy.crs as ccrs
import numpy as np
import pytest
import torch
import xarray as xr

from clara_inference import ifs_boundary as ib

H = datetime.timedelta(hours=1)


def _cycles(*hours_since_2026_10_25):
    t0 = np.datetime64("2026-10-25T00:00", "ns")
    return [t0 + np.timedelta64(h, "h") for h in hours_since_2026_10_25]


@pytest.mark.parametrize(
    "analysis_hour, expected",
    [
        # as neural-lam picks the cycle in training (see module docstring):
        # T=00 -> previous day 12, T=06 -> 00, T=12 -> 00, T=18 -> 12
        (24, 12),
        (30, 24),
        (36, 24),
        (42, 36),
        # T=03 (3 h after the 00 cycle) -> 00
        (27, 24),
    ],
)
def test_select_cycle(analysis_hour, expected):
    cycles = _cycles(0, 6, 12, 18, 24, 30, 36, 42)
    analysis_time = datetime.datetime(2026, 10, 25) + analysis_hour * H
    assert ib.select_cycle(cycles, analysis_time) == _cycles(expected)[0]


def test_select_cycle_only_00_12():
    # 06/18 UTC cycles are not used (CLARA was fine-tuned with 00/12)
    cycles = _cycles(6, 18)
    with pytest.raises(ValueError, match="00/12 UTC"):
        ib.select_cycle(cycles, datetime.datetime(2026, 10, 25, 23))


def test_required_lead_times():
    cycle = _cycles(24)[0]
    analysis_time = datetime.datetime(2026, 10, 26, 6)
    leads = ib.required_lead_times(cycle, analysis_time, 18 * H)
    hours = (leads / np.timedelta64(1, "h")).astype(int).tolist()
    # last interior time 06 + 18 + 2 * 3 = 30 h after the cycle -> 5 steps,
    # plus BOUNDARY_EXTRA_STEPS
    assert hours == list(range(0, 6 * (5 + ib.BOUNDARY_EXTRA_STEPS) + 1, 6))


def _boundary_grid():
    lats = np.arange(60.0, 49.9, -0.25)
    lons = np.arange(0, 360, 0.25)
    ds = xr.Dataset(
        {
            "land_sea_mask": (
                ("latitude", "longitude"),
                np.ones((lats.size, lons.size), "f4"),
            ),
            "geopotential_at_surface": (
                ("latitude", "longitude"),
                np.zeros((lats.size, lons.size), "f4"),
            ),
        },
        coords=dict(latitude=lats, longitude=lons),
    )
    return ds


def _ifs(cycles, leads_h, lon_180=True, lat_ascending=True):
    lats = np.arange(40.0, 72.01, 0.25)
    if not lat_ascending:
        lats = lats[::-1]
    lons = np.arange(-30.0, 50.01, 0.25)
    if not lon_180:
        lons = lons % 360
    shape_sl = (len(cycles), len(leads_h), lats.size, lons.size)
    dims_sl = ("time", "prediction_timedelta", "latitude", "longitude")
    lon2d, lat2d = np.meshgrid(lons, lats)
    # a field identifying the point, to check the reordering
    field = (lat2d * 1000 + (lon2d % 360)).astype("f4")
    data_vars = {
        v: (dims_sl, np.broadcast_to(field, shape_sl)) for v in ib.SL_VARS
    }
    for v in ib.PL_VARS:
        data_vars[v] = (
            ("time", "prediction_timedelta", "level", "latitude", "longitude"),
            np.broadcast_to(
                field, shape_sl[:2] + (len(ib.LEVELS),) + shape_sl[2:]
            ),
        )
    return xr.Dataset(
        data_vars,
        coords=dict(
            time=np.array(cycles, dtype="datetime64[ns]"),
            prediction_timedelta=np.array(leads_h, dtype="timedelta64[h]")
            .astype("timedelta64[ns]"),
            level=ib.LEVELS,
            latitude=lats,
            longitude=lons,
        ),
    )


@pytest.mark.parametrize("lon_180", [True, False])
@pytest.mark.parametrize("lat_ascending", [True, False])
def test_prepare_boundary_input(lon_180, lat_ascending):
    ds_ifs = _ifs(_cycles(12, 24), range(0, 61, 6), lon_180, lat_ascending)
    ds_grid = _boundary_grid()
    ds = ib.prepare_boundary_input(
        ds_ifs, ds_grid, datetime.datetime(2026, 10, 26, 6), 18 * H
    )
    assert ds.time.values.tolist() == [_cycles(24)[0].astype(int)] or (
        ds.time.values[0] == _cycles(24)[0]
    )
    # on the boundary grid, in its (training) order
    np.testing.assert_array_equal(ds.latitude, ds_grid.latitude)
    np.testing.assert_array_equal(ds.longitude, ds_grid.longitude)
    t2m = ds["2m_temperature"].isel(time=0, prediction_timedelta=0)
    lon2d, lat2d = np.meshgrid(ds.longitude, ds.latitude)
    covered = (lon2d <= 50) | (lon2d >= 330)
    np.testing.assert_allclose(
        t2m.values[covered], (lat2d * 1000 + lon2d)[covered]
    )
    assert np.isnan(t2m.values[~covered]).all()
    for v in ib.STATIC_VARS:
        assert v in ds and ds[v].dims == ("latitude", "longitude")
    assert ds.attrs["boundary_cycle"] == "2026-10-26T00:00"


def test_prepare_boundary_input_missing_lead_times():
    ds_ifs = _ifs(_cycles(24), range(0, 25, 6))
    with pytest.raises(ValueError, match="no lead times"):
        ib.prepare_boundary_input(
            ds_ifs,
            _boundary_grid(),
            datetime.datetime(2026, 10, 26, 6),
            18 * H,
        )


def test_contract_missing_variable():
    ds_ifs = _ifs(_cycles(24), range(0, 61, 6)).drop_vars("surface_pressure")
    with pytest.raises(ValueError, match="surface_pressure"):
        ib.check_contract(ds_ifs)


def test_check_point_order(tmp_path):
    projection = ccrs.LambertConformal(
        central_longitude=25.0,
        central_latitude=56.7,
        standard_parallels=(56.7, 56.7),
    )
    lat = np.array([60.0, 59.75, 60.0])
    lon = np.array([10.0, 10.0, 10.25])
    x, y, _ = projection.transform_points(ccrs.PlateCarree(), lon, lat).T
    mesh = np.zeros((4, 2))
    nodes = np.concatenate([np.stack([x, y], axis=1), mesh]).astype("f4")
    torch.save(torch.tensor(nodes), tmp_path / "g2m_node_features.pt")
    ds = xr.Dataset(
        coords=dict(
            latitude=("grid_index", lat), longitude=("grid_index", lon)
        )
    )
    ib.check_point_order(ds, tmp_path, projection)
    with pytest.raises(ValueError, match="don't match"):
        ib.check_point_order(ds.isel(grid_index=[1, 0, 2]), tmp_path, projection)
