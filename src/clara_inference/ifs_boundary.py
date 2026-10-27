"""
The IFS boundary forecast CLARA is run with: the contract an IFS zarr must
follow, choosing the forecast cycle, and bringing it into the layout and
point order of the boundary CLARA was fine-tuned with.

## The IFS boundary contract

An IFS forecast zarr (local path or S3) with

- dims `time` (forecast cycle, 00/12 UTC cycles are used), and
  `prediction_timedelta` (lead time, **every 6 h** from 0 h), `latitude` and
  `longitude` on the 0.25 deg (WeatherBench2) grid, plus `level` (hPa) for
  pressure level variables. Longitudes in -180..180 or 0..360 and either
  latitude order are accepted
- covering the 400 km margin around the DANRA domain (lat 43.25..67.75,
  lon -22.0..44.0); points outside it are not used
- the variables (ERA5/WeatherBench2 names and units) in `SL_VARS` and
  `PL_VARS`, the latter at the levels in `LEVELS`

`dev-utils/make_ifs_boundary_zarr.py` makes such a zarr from ECMWF open data.

## Cycle and lead times

The model was fine-tuned with the 00/12 UTC IFS cycles, and neural-lam takes
the boundary from the latest cycle strictly before the second initial state
(analysis time + 3 h), stepping back one cycle if that leaves less than one
6 h boundary step before it (`WeatherDataset`, forecast-format boundaries).
So in training, a forecast from T used the latest 00/12 UTC cycle at or
before T - 3 h; `select_cycle` picks that cycle, and only that cycle is given
to neural-lam.

## Point order

neural-lam's boundary nodes are the boundary datastore's grid points in
order, and the graph shipped in the artifact (built in training) connects
them to the mesh by index. In training the points were ordered by longitude
(0..360, increasing) and, within a longitude, by latitude (decreasing). The
IFS fields are therefore put on that grid (`grids/<boundary>.grid.zarr` in
the artifact, the ERA5 statics on the boundary latitude band in training
order) before the datastore is created, and `check_point_order` checks the
resulting boundary points against the graph.
"""
import datetime
from pathlib import Path

import numpy as np
import pyproj
import torch
import xarray as xr
from loguru import logger

SL_VARS = [
    "mean_sea_level_pressure",
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "surface_pressure",
]
PL_VARS = [
    "geopotential",
    "temperature",
    "specific_humidity",
    "u_component_of_wind",
    "v_component_of_wind",
    "vertical_velocity",
]
STATIC_VARS = ["land_sea_mask", "geopotential_at_surface"]
LEVELS = [100, 200, 400, 600, 700, 850, 925, 1000]  # hPa

LEAD_STEP = datetime.timedelta(hours=6)
CYCLE_HOURS = (0, 12)
INTERIOR_STEP = datetime.timedelta(hours=3)
# as in regrid_dini.py: interior steps beyond analysis time + duration
INTERIOR_EXTRA_STEPS = 2
# boundary steps beyond the last interior time: neural-lam's window for a
# state at t covers the boundary step at or before t and
# `num_future_boundary_steps=1` after it (exclusive end, so +1)
BOUNDARY_EXTRA_STEPS = 2
# maximum distance (m) between the boundary points and the graph's boundary
# nodes for the point order check (float32 positions: ~0.1 m)
MAX_POSITION_DIFF_M = 10.0


def select_cycle(cycles, analysis_time):
    """
    The forecast cycle to use for a forecast from `analysis_time`: the latest
    00/12 UTC cycle at or before analysis_time - 3 h (see module docstring).
    """
    latest = np.datetime64(analysis_time - INTERIOR_STEP, "ns")
    cycles = np.asarray(cycles, dtype="datetime64[ns]")
    hours = (cycles - cycles.astype("datetime64[D]")) // np.timedelta64(
        1, "h"
    )
    usable = cycles[(cycles <= latest) & np.isin(hours, CYCLE_HOURS)]
    if usable.size == 0:
        raise ValueError(
            f"no 00/12 UTC IFS cycle at or before {latest} (analysis time "
            f"{analysis_time} - 3h) in the boundary zarr (cycles: "
            f"{[str(c)[:16] for c in cycles]})"
        )
    return usable.max()


def required_lead_times(cycle, analysis_time, forecast_duration):
    """Lead times (0, 6h, ...) neural-lam needs from `cycle`."""
    last_interior = (
        np.datetime64(analysis_time, "ns")
        + np.timedelta64(forecast_duration, "ns")
        + INTERIOR_EXTRA_STEPS * np.timedelta64(INTERIOR_STEP, "ns")
    )
    step = np.timedelta64(LEAD_STEP, "ns")
    n = int(np.ceil((last_interior - cycle) / step)) + BOUNDARY_EXTRA_STEPS
    return np.arange(n + 1) * step


def check_contract(ds):
    """Check variables, dims and levels of an IFS boundary zarr."""
    problems = []
    for v in SL_VARS + PL_VARS:
        if v not in ds:
            problems.append(f"missing variable {v}")
    for dim in ["time", "prediction_timedelta", "latitude", "longitude"]:
        if dim not in ds.dims:
            problems.append(f"missing dimension {dim}")
    if "level" in ds.dims:
        missing = sorted(set(LEVELS) - set(ds.level.values.tolist()))
        if missing:
            problems.append(f"missing levels {missing} hPa")
    else:
        problems.append("missing dimension level")
    if problems:
        raise ValueError(
            "the IFS boundary zarr doesn't follow the contract (see "
            "clara_inference/ifs_boundary.py): " + "; ".join(problems)
        )


def _to_grid(ds, ds_grid):
    """
    Put `ds` on the latitude/longitude of `ds_grid` (exact 0.25 deg grid
    points, NaN where `ds` doesn't cover them), longitudes in 0..360.
    """
    ds = ds.assign_coords(longitude=ds.longitude % 360).sortby("longitude")
    ds = ds.sortby("latitude")
    return ds.reindex(
        latitude=ds_grid.latitude.values,
        longitude=ds_grid.longitude.values,
        method="nearest",
        tolerance=1e-3,
    )


def prepare_boundary_input(ds, ds_grid, analysis_time, forecast_duration):
    """
    The boundary input for the boundary datastore config: the selected cycle
    and lead times of an IFS forecast (contract layout) on the boundary grid
    in training order, with the ERA5 statics of the training boundary.
    """
    check_contract(ds)
    cycle = select_cycle(ds.time.values, analysis_time)
    leads = required_lead_times(cycle, analysis_time, forecast_duration)
    missing = sorted(
        set(leads.tolist()) - set(ds.prediction_timedelta.values.tolist())
    )
    if missing:
        raise ValueError(
            f"the IFS cycle {str(cycle)[:16]} has no lead times "
            f"{[str(np.timedelta64(m, 'ns').astype('timedelta64[h]')) for m in missing]}"
            f", needed for a {forecast_duration} forecast from "
            f"{analysis_time} (lead times every 6 h up to "
            f"{leads[-1].astype('timedelta64[h]')})"
        )
    ds = ds[SL_VARS + PL_VARS].sel(
        time=[cycle], prediction_timedelta=leads, level=LEVELS
    )
    ds = _to_grid(ds, ds_grid).astype("float32")
    for v in STATIC_VARS:
        ds[v] = ds_grid[v]
    ds.attrs = dict(
        boundary_cycle=str(cycle)[:16],
        boundary_lead_times=f"0..{leads[-1].astype('timedelta64[h]')} "
        "every 6h",
    )
    logger.info(
        f"IFS boundary: cycle {str(cycle)[:16]}, lead times "
        f"{ds.attrs['boundary_lead_times']}"
    )
    return ds


def boundary_grid_from_points(lat, lon, ds_statics):
    """
    The boundary grid (`grids/<boundary>.grid.zarr` in the artifact): the
    0.25 deg latitude band covering the training boundary points (`lat`,
    `lon`), all longitudes 0..360, in training order (latitude decreasing),
    with the ERA5 statics (from `ds_statics`, e.g. WeatherBench2 ERA5).
    """
    lats = np.arange(np.max(lat), np.min(lat) - 0.125, -0.25)
    lons = np.arange(0, 360, 0.25)
    ds = ds_statics[STATIC_VARS]
    ds = ds.assign_coords(longitude=ds.longitude % 360)
    ds = ds.sortby("longitude").sortby("latitude")
    ds = ds.reindex(
        latitude=lats, longitude=lons, method="nearest", tolerance=1e-3
    )
    ds = ds.drop_vars([c for c in ds.coords if c not in ds.dims])
    return ds.astype("float32").load()


def check_point_order(ds_boundary, graph_dir, projection):
    """
    Check that the points of a boundary datastore (`latitude`/`longitude`
    per grid_index, as created by mllam-data-prep) are the graph's boundary
    (g2m source) nodes, in the same order. `projection` is the interior
    (graph) cartopy projection.
    """
    node_xy = torch.load(
        Path(graph_dir) / "g2m_node_features.pt", weights_only=True
    )
    node_xy = (node_xy[0] if isinstance(node_xy, list) else node_xy).numpy()
    lat = ds_boundary.latitude.values.astype("f8")
    lon = ds_boundary.longitude.values.astype("f8")
    transformer = pyproj.Transformer.from_crs(
        "EPSG:4326", projection, always_xy=True
    )
    x, y = transformer.transform(lon, lat)
    n = lat.size
    if node_xy.shape[0] < n:
        raise ValueError(
            f"the graph has {node_xy.shape[0]} g2m nodes, fewer than the "
            f"{n} boundary points"
        )
    diff = np.hypot(x - node_xy[:n, 0], y - node_xy[:n, 1])
    if not (diff <= MAX_POSITION_DIFF_M).all():
        raise ValueError(
            f"the boundary points don't match the graph's boundary nodes: "
            f"{int((diff > MAX_POSITION_DIFF_M).sum())} of {n} points are "
            f"further than {MAX_POSITION_DIFF_M} m from their node (max "
            f"{diff.max():.0f} m). The boundary must have the training "
            "points in training order"
        )
    logger.info(
        f"boundary points match the graph's {n} boundary nodes (max "
        f"distance {diff.max():.2f} m)"
    )
