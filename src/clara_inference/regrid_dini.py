"""
Regrid a DINI (HARMONIE, 2 km Lambert) control forecast to the inputs ANNA
expects:

- interior: the DANRA grid (2.5 km Lambert, 589 x 789) with DANRA variable
  names and conventions, written as `{output}/interior_single_levels.zarr` and
  `{output}/interior_pressure_levels.zarr` (read through the
  `danra_model1_config` datastore config)
- boundary: a regular 0.25 deg lat/lon box with ERA5 (WeatherBench2) variable
  names and conventions, following the contract in
  `anna_inference/boundary/ifs_7deg_model1_config.yaml`, written as
  `{output}/boundary.zarr`
  (read through the `dini_7deg_model1_config` datastore config)

Conventions (checked against the data, see INFERENCE_PLAN.md step 4):
- DINI and DANRA winds are both *grid-relative* (despite DINI's
  eastward_wind/northward_wind labels), on differently rotated Lambert grids.
  Winds are rotated to earth-relative on the DINI grid, interpolated, and then
  rotated to DANRA grid-relative (interior) or kept earth-relative (boundary,
  as ERA5/IFS)
- DINI relative humidity is in %, DANRA's is a fraction (though labelled %)
- DINI `orography` is surface altitude (m), DANRA's is surface geopotential;
  the interior statics are taken from DANRA itself (`--danra-grid`)
- `lwavr0m`/`swavr0m` are the same (net surface) fluxes in both
- boundary specific humidity is derived from relative humidity, temperature and
  pressure, and vertical velocity (omega, Pa/s) from DINI's geometric vertical
  velocity `tw` (m/s) with the hydrostatic approximation

Interpolation is bilinear in DINI's own projection (from the `crs_wkt` stored
with the DINI data); target points outside the DINI domain are handled with
`--outside-domain` (the boundary box extends north of DINI).

Usage:
    # once: cache the DANRA grid and statics
    python -m anna_inference.regrid_dini create-danra-grid \\
        --output danra_model1_config.grid.zarr

    python -m anna_inference.regrid_dini regrid \\
        --dini-root s3://harmonie-zarr/dini/control/2026-09-26T180000Z/ \\
        --danra-grid danra_model1_config.grid.zarr \\
        --forecast-duration PT18H --output inference_workdir/inputs
"""
import argparse
import datetime
import time
from pathlib import Path

import dask
import dask.diagnostics
import isodate
import numpy as np
import pyproj
import scipy.ndimage
import xarray as xr
from loguru import logger

DANRA_SL_URL = (
    "https://object-store.os-api.cci1.ecmwf.int"
    "/danra/v0.5.0/single_levels.zarr/"
)

GRAVITY = 9.80665  # m s-2
R_DRY = 287.05  # J kg-1 K-1
EPSILON = 0.621981  # R_dry / R_vapour

LEVELS = [100, 200, 400, 600, 700, 850, 925, 1000]  # hPa, as used by ANNA
INTERIOR_STEP = datetime.timedelta(hours=3)
BOUNDARY_STEP = datetime.timedelta(hours=6)
# interior time steps needed beyond analysis_time + forecast_duration: one for
# `num_future_forcing_steps=1`, and one more because neural-lam's
# WeatherDataset counts samples as
# `n_times - (2 + ar_steps) - num_future_forcing_steps`, one step more
# conservative than the data a sample actually uses
INTERIOR_EXTRA_STEPS = 2

# ERA5 (WeatherBench2) 0.25 deg box covering the boundary points ANNA was
# trained with (lat 40.50..71.50, lon -26.25..39.50), as in the contract in
# anna_inference/boundary/ifs_7deg_model1_config.yaml. The east edge must be
# exactly 39.5E, the edge of the training ERA5 subset: the cropping margin
# reaches further east, so a wider box would add boundary points not used in
# training
BOUNDARY_LATS = np.arange(72.0, 40.0 - 0.125, -0.25)
BOUNDARY_LONS = np.arange(-27.0, 39.5 + 0.125, 0.25)
# DINI grid spacing, used to convert the boundary smoothing scale to grid cells
DINI_DX_KM = 2.0

INTERIOR_SL_VARS = ["pres_seasurface", "t2m", "pres0m", "lwavr0m", "swavr0m"]
INTERIOR_PL_VARS = ["z", "t", "r", "tw"]  # plus winds u, v
BOUNDARY_SL_VARS = {
    "pres_seasurface": "mean_sea_level_pressure",
    "t2m": "2m_temperature",
    "pres0m": "surface_pressure",
}  # plus 10m winds
BOUNDARY_PL_VARS = {"z": "geopotential", "t": "temperature"}  # plus u, v, q, w

# units of the output variables (metadata only, they end up as the
# `*_feature_units` coordinates of the inference datastores). NB: relative
# humidity is a fraction, even though DANRA labels it "%"
UNITS = {
    "pres_seasurface": "Pa",
    "t2m": "K",
    "pres0m": "Pa",
    "lwavr0m": "W m**-2",
    "swavr0m": "W m**-2",
    "u10m": "m s**-1",
    "v10m": "m s**-1",
    "z": "m**2 s**-2",
    "t": "K",
    "r": "1",
    "tw": "m s**-1",
    "u": "m s**-1",
    "v": "m s**-1",
    "mean_sea_level_pressure": "Pa",
    "2m_temperature": "K",
    "surface_pressure": "Pa",
    "10m_u_component_of_wind": "m s**-1",
    "10m_v_component_of_wind": "m s**-1",
    "geopotential": "m**2 s**-2",
    "temperature": "K",
    "u_component_of_wind": "m s**-1",
    "v_component_of_wind": "m s**-1",
    "specific_humidity": "kg kg**-1",
    "vertical_velocity": "Pa s**-1",
    "land_sea_mask": "1",
    "geopotential_at_surface": "m**2 s**-2",
}


def _set_units(ds):
    for v in ds.data_vars:
        if v in UNITS:
            ds[v].attrs["units"] = UNITS[v]
    return ds


def grid_x_axis_angle(lat, lon):
    """
    Angle (radians, counter-clockwise from east) of the grid x-axis at each
    point of a 2D (y, x) grid, from finite differences of lat/lon along x.
    """
    dlon = np.deg2rad(np.gradient(lon, axis=1))
    dlon = (dlon + np.pi) % (2 * np.pi) - np.pi
    dlat = np.deg2rad(np.gradient(lat, axis=1))
    return np.arctan2(dlat, dlon * np.cos(np.deg2rad(lat)))


def grid_to_earth(u, v, theta):
    """Rotate grid-relative wind to earth-relative (eastward, northward)."""
    return (
        u * np.cos(theta) - v * np.sin(theta),
        u * np.sin(theta) + v * np.cos(theta),
    )


def earth_to_grid(u, v, theta):
    """Rotate earth-relative wind to grid-relative."""
    return (
        u * np.cos(theta) + v * np.sin(theta),
        -u * np.sin(theta) + v * np.cos(theta),
    )


def saturation_vapour_pressure(t):
    """
    Saturation vapour pressure (Pa) as in the ECMWF IFS (and so ERA5): over
    water above 0 C, over ice below -23 C and a quadratic mix in between.
    """
    t0, t_ice = 273.16, 250.16
    es_water = 611.21 * np.exp(17.502 * (t - t0) / (t - 32.19))
    es_ice = 611.21 * np.exp(22.587 * (t - t0) / (t + 0.7))
    alpha = np.clip((t - t_ice) / (t0 - t_ice), 0.0, 1.0) ** 2
    return alpha * es_water + (1 - alpha) * es_ice


def specific_humidity(rh_fraction, t, p):
    """Specific humidity (kg/kg) from relative humidity (0-1), T (K), p (Pa)."""
    e = np.clip(rh_fraction, 0.0, None) * saturation_vapour_pressure(t)
    return EPSILON * e / (p - (1 - EPSILON) * e)


def omega_from_w(w, t, q, p):
    """Pressure vertical velocity (Pa/s) from geometric w (m/s), hydrostatic."""
    t_virtual = t * (1 + (1 / EPSILON - 1) * q)
    rho = p / (R_DRY * t_virtual)
    return -rho * GRAVITY * w


class BilinearInterpolator:
    """
    Bilinear interpolation from a regular (y, x) grid to arbitrary target
    points given in the same projected coordinates. Weights are computed once
    and applied to any number of fields.
    """

    def __init__(self, x_src, y_src, x_tgt, y_tgt, outside="error"):
        dx, dy = x_src[1] - x_src[0], y_src[1] - y_src[0]
        fi = (np.asarray(x_tgt) - x_src[0]) / dx
        fj = (np.asarray(y_tgt) - y_src[0]) / dy
        self.shape = np.shape(x_tgt)
        self.inside = (
            (fi >= 0)
            & (fi <= x_src.size - 1)
            & (fj >= 0)
            & (fj <= y_src.size - 1)
        )
        n_outside = int((~self.inside).sum())
        if n_outside > 0:
            msg = (
                f"{n_outside} of {self.inside.size} target points are outside "
                "the source grid"
            )
            if outside == "error":
                raise ValueError(msg)
            logger.warning(
                f"{msg}, filling with the nearest source grid point"
            )
        fi = np.clip(fi, 0, x_src.size - 1)
        fj = np.clip(fj, 0, y_src.size - 1)
        self.i0 = np.minimum(np.floor(fi).astype(int), x_src.size - 2)
        self.j0 = np.minimum(np.floor(fj).astype(int), y_src.size - 2)
        self.wx = fi - self.i0
        self.wy = fj - self.j0
        self.n_outside = n_outside

    def __call__(self, field):
        """Interpolate a (..., y, x) field (the last two axes)."""
        f = np.asarray(field)
        i0, j0, wx, wy = self.i0, self.j0, self.wx, self.wy
        return (
            f[..., j0, i0] * (1 - wx) * (1 - wy)
            + f[..., j0, i0 + 1] * wx * (1 - wy)
            + f[..., j0 + 1, i0] * (1 - wx) * wy
            + f[..., j0 + 1, i0 + 1] * wx * wy
        )

    def apply(self, da, out_dims):
        """
        Interpolate a (lazy) DataArray with (y, x) dims, block by block with
        dask; the target points get dims `out_dims`.
        """
        return xr.apply_ufunc(
            self,
            da,
            input_core_dims=[["y", "x"]],
            output_core_dims=[list(out_dims)],
            dask="parallelized",
            output_dtypes=[np.float64],
            dask_gufunc_kwargs=dict(
                output_sizes=dict(zip(out_dims, self.shape))
            ),
        )


def _open_dini(dini_root):
    root = str(dini_root).rstrip("/")
    ds_sl = xr.open_zarr(f"{root}/single_levels.zarr")
    ds_pl = xr.open_zarr(f"{root}/pressure_levels.zarr")
    return ds_sl, ds_pl


def _dini_transformer(ds_sl):
    crs = pyproj.CRS.from_wkt(ds_sl["dini_projection"].attrs["crs_wkt"])
    return pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)


def _valid_times(ds, analysis_time, duration, step):
    """
    Valid times analysis_time, +step, ... up to (at least) analysis_time +
    duration.
    """
    n = int(np.ceil(duration / step))
    times = [
        np.datetime64(analysis_time + k * step, "ns") for k in range(n + 1)
    ]
    missing = sorted(set(times) - set(ds.time.values))
    if missing:
        raise ValueError(
            f"DINI forecast doesn't contain valid times {missing} "
            f"(available: {ds.time.values[0]} .. {ds.time.values[-1]})"
        )
    return times


def create_danra_grid(output, danra_url=DANRA_SL_URL):
    """
    Cache the DANRA grid (x, y, lat, lon) and statics (lsm, orography) that
    ANNA was trained with.
    """
    ds = xr.open_zarr(danra_url)
    ds_grid = ds[["lsm", "orography"]].load()
    ds_grid = ds_grid.drop_vars(
        [v for v in ds_grid.coords if v not in ("x", "y", "lat", "lon")]
    )
    ds_grid.attrs = dict(
        source=danra_url, description="DANRA v0.5.0 grid and statics"
    )
    ds_grid.to_zarr(output, mode="w", consolidated=True)
    logger.info(f"wrote DANRA grid and statics to {output}")


def _regrid_interior(
    ds_sl, ds_pl, danra_grid, times, transformer, theta_dini, outside
):
    lat, lon = danra_grid.lat.values, danra_grid.lon.values
    x_tgt, y_tgt = transformer.transform(lon, lat)
    interp = BilinearInterpolator(
        ds_sl.x.values, ds_sl.y.values, x_tgt, y_tgt, outside=outside
    )
    # interpolated fields get dims (..., y_out, x_out), the DANRA grid
    out_dims = ("y_out", "x_out")
    theta_danra = xr.DataArray(grid_x_axis_angle(lat, lon), dims=out_dims)

    # lazy: nothing is read until the caller computes the result
    sl_in = ds_sl.sel(time=times)
    pl_in = ds_pl.sel(time=times, pressure=LEVELS)

    def _winds(ds, u_name, v_name):
        u_e, v_e = grid_to_earth(ds[u_name], ds[v_name], theta_dini)
        return earth_to_grid(
            interp.apply(u_e, out_dims),
            interp.apply(v_e, out_dims),
            theta_danra,
        )

    sl = {v: interp.apply(sl_in[v], out_dims) for v in INTERIOR_SL_VARS}
    sl["u10m"], sl["v10m"] = _winds(sl_in, "u10m", "v10m")
    pl = {v: interp.apply(pl_in[v], out_dims) for v in INTERIOR_PL_VARS}
    pl["u"], pl["v"] = _winds(pl_in, "u", "v")
    sl = {v: da.astype("f4").data for v, da in sl.items()}
    pl = {
        v: da.transpose("time", "pressure", *out_dims).astype("f4").data
        for v, da in pl.items()
    }
    # DANRA relative humidity is a fraction (despite being labelled "%")
    pl["r"] = pl["r"] / 100.0

    dims = ("time", "y", "x")
    coords = dict(
        time=("time", np.array(times)),
        x=danra_grid.x.values,
        y=danra_grid.y.values,
        lat=(("y", "x"), lat),
        lon=(("y", "x"), lon),
    )
    ds_sl_out = xr.Dataset(
        {v: (dims, a) for v, a in sl.items()}, coords=coords
    )
    # statics exactly as in training (DANRA orography is surface geopotential)
    ds_sl_out["lsm"] = danra_grid.lsm
    ds_sl_out["orography"] = danra_grid.orography
    ds_pl_out = xr.Dataset(
        {v: (("time", "pressure") + dims[1:], a) for v, a in pl.items()},
        coords=dict(coords, pressure=("pressure", LEVELS, {"units": "hPa"})),
    )
    return _set_units(ds_sl_out), _set_units(ds_pl_out), interp.n_outside


def smooth(field, n_cells):
    """
    Box average over n_cells x n_cells grid cells (edges: nearest value), to
    bring km-scale DINI fields to the scale of the 0.25 deg ERA5 grid before
    sampling them there. Point samples of 2 km fields keep small-scale
    variability (e.g. vertical velocity over orography) that 0.25 deg ERA5
    doesn't have.

    Smooths over the last two (y, x) axes only.
    """
    field = np.asarray(field)
    if n_cells <= 1:
        return field
    return scipy.ndimage.uniform_filter(
        field.astype("f8"),
        size=(1,) * (field.ndim - 2) + (n_cells, n_cells),
        mode="nearest",
    )


def smooth_lazy(da, n_cells):
    """`smooth` of a (lazy) DataArray with (y, x) dims, block by block."""
    return xr.apply_ufunc(
        smooth,
        da,
        n_cells,
        input_core_dims=[["y", "x"], []],
        output_core_dims=[["y", "x"]],
        dask="parallelized",
        output_dtypes=[np.float64],
    )


def _regrid_boundary(
    ds_sl,
    ds_pl,
    analysis_time,
    times,
    transformer,
    theta_dini,
    outside,
    smoothing_km,
):
    lon2d, lat2d = np.meshgrid(BOUNDARY_LONS, BOUNDARY_LATS)
    x_tgt, y_tgt = transformer.transform(lon2d, lat2d)
    interp = BilinearInterpolator(
        ds_sl.x.values, ds_sl.y.values, x_tgt, y_tgt, outside=outside
    )
    # odd number of cells so the box average is centred
    n_cells = (
        2 * int(round(smoothing_km / DINI_DX_KM / 2)) + 1
        if smoothing_km > 0
        else 1
    )
    logger.info(
        f"boundary: smoothing over {n_cells}x{n_cells} DINI grid cells before sampling"
    )

    out_dims = ("latitude", "longitude")

    def sample(field):
        return interp(smooth(field, n_cells))

    def sample_lazy(da):
        return interp.apply(smooth_lazy(da, n_cells), out_dims)

    # lazy: nothing is read until the caller computes the result
    sl_in = ds_sl.sel(time=times)
    pl_in = ds_pl.sel(time=times, pressure=LEVELS)

    def _earth_winds(ds, u_name, v_name):
        u_e, v_e = grid_to_earth(ds[u_name], ds[v_name], theta_dini)
        return sample_lazy(u_e), sample_lazy(v_e)

    sl = {
        v_era: sample_lazy(sl_in[v_dini])
        for v_dini, v_era in BOUNDARY_SL_VARS.items()
    }
    (
        sl["10m_u_component_of_wind"],
        sl["10m_v_component_of_wind"],
    ) = _earth_winds(sl_in, "u10m", "v10m")
    pl = {
        v_era: sample_lazy(pl_in[v_dini])
        for v_dini, v_era in BOUNDARY_PL_VARS.items()
    }
    (
        pl["u_component_of_wind"],
        pl["v_component_of_wind"],
    ) = _earth_winds(pl_in, "u", "v")
    # derive on the DINI grid, then interpolate
    t_k = pl_in["t"]
    p_pa = xr.DataArray(np.array(LEVELS) * 100.0, dims="pressure")
    q = specific_humidity(pl_in["r"] / 100.0, t_k, p_pa)
    w = omega_from_w(pl_in["tw"], t_k, q, p_pa)
    pl["specific_humidity"] = sample_lazy(q)
    pl["vertical_velocity"] = sample_lazy(w)
    # layout (analysis time, lead time[, level], latitude, longitude)
    sl = {v: da.astype("f4").data[None] for v, da in sl.items()}
    pl = {
        v: da.transpose("time", "pressure", *out_dims).astype("f4").data[None]
        for v, da in pl.items()
    }

    dims_sl = ("time", "prediction_timedelta", "latitude", "longitude")
    dims_pl = (
        "time",
        "prediction_timedelta",
        "level",
        "latitude",
        "longitude",
    )
    ds = xr.Dataset(
        {v: (dims_sl, a) for v, a in sl.items()}
        | {v: (dims_pl, a) for v, a in pl.items()},
        coords=dict(
            time=("time", [np.datetime64(analysis_time, "ns")]),
            prediction_timedelta=(
                "prediction_timedelta",
                np.array(
                    [t - np.datetime64(analysis_time, "ns") for t in times]
                ),
            ),
            level=("level", LEVELS, {"units": "hPa"}),
            latitude=("latitude", BOUNDARY_LATS),
            longitude=("longitude", BOUNDARY_LONS),
        ),
    )
    ds["land_sea_mask"] = (
        ("latitude", "longitude"),
        sample(ds_sl["lsm"].values),
    )
    ds["geopotential_at_surface"] = (
        ("latitude", "longitude"),
        sample(ds_sl["orography"].values) * GRAVITY,
    )
    return _set_units(ds), interp


def regrid(
    dini_root,
    danra_grid,
    forecast_duration,
    output,
    interior=True,
    boundary=True,
    outside_domain="nearest",
    boundary_smoothing_km=25.0,
):
    ds_sl, ds_pl = _open_dini(dini_root)
    analysis_time = ds_sl.time.values[0].astype("datetime64[s]").item()
    logger.info(f"DINI forecast from {analysis_time}, output to {output}")
    transformer = _dini_transformer(ds_sl)
    theta_dini = xr.DataArray(
        grid_x_axis_angle(ds_sl.lat.values, ds_sl.lon.values), dims=("y", "x")
    )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    # the regridded datasets are built lazily and computed together at the
    # end, so that dask reads the DINI chunks concurrently (and only once
    # when both the interior and the boundary use them)
    outputs = {}

    if interior:
        # two interior steps beyond the forecast are needed, see
        # INTERIOR_EXTRA_STEPS
        times = _valid_times(
            ds_sl,
            analysis_time,
            forecast_duration + INTERIOR_EXTRA_STEPS * INTERIOR_STEP,
            INTERIOR_STEP,
        )
        ds_sl_out, ds_pl_out, n_outside = _regrid_interior(
            ds_sl,
            ds_pl,
            xr.open_zarr(danra_grid).load(),
            times,
            transformer,
            theta_dini,
            outside="error",
        )
        attrs = dict(
            source=str(dini_root),
            description="DINI regridded to the DANRA grid (regrid_dini.py)",
        )
        ds_sl_out.attrs = ds_pl_out.attrs = attrs
        outputs["interior_single_levels.zarr"] = ds_sl_out
        outputs["interior_pressure_levels.zarr"] = ds_pl_out
        logger.info(f"interior: {len(times)} time steps")

    if boundary:
        # one extra boundary step for `num_future_boundary_steps=1`
        times = _valid_times(
            ds_sl,
            analysis_time,
            forecast_duration + BOUNDARY_STEP,
            BOUNDARY_STEP,
        )
        ds_b, interp = _regrid_boundary(
            ds_sl,
            ds_pl,
            analysis_time,
            times,
            transformer,
            theta_dini,
            outside_domain,
            boundary_smoothing_km,
        )
        ds_b.attrs = dict(
            source=str(dini_root),
            description=(
                "DINI regridded to the ERA5 0.25 deg boundary box in the IFS "
                "boundary contract layout (regrid_dini.py)"
            ),
            n_points_outside_dini_filled_nearest=interp.n_outside,
            smoothing_km=boundary_smoothing_km,
        )
        outputs["boundary.zarr"] = ds_b
        logger.info(f"boundary: {len(times)} lead times")

    logger.info("reading and regridding DINI")
    start = time.perf_counter()
    with dask.diagnostics.ProgressBar(dt=10):
        computed = dask.compute(*outputs.values())
    logger.info(f"regridded in {time.perf_counter() - start:.0f} s")
    for name, ds in zip(outputs, computed):
        ds.to_zarr(output / name, mode="w", consolidated=True)
        logger.info(f"wrote {output / name}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p = subparsers.add_parser(
        "create-danra-grid", help="cache DANRA grid and statics"
    )
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--danra-url", default=DANRA_SL_URL)

    p = subparsers.add_parser("regrid", help="regrid a DINI forecast")
    p.add_argument(
        "--dini-root", required=True, help=".../dini/control/{analysis}/"
    )
    p.add_argument("--danra-grid", required=True, type=Path)
    p.add_argument(
        "--forecast-duration",
        required=True,
        type=isodate.parse_duration,
        help="ISO8601 duration, e.g. PT18H",
    )
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--no-interior", action="store_true")
    p.add_argument("--no-boundary", action="store_true")
    p.add_argument(
        "--outside-domain",
        choices=["nearest", "error"],
        default="nearest",
        help=(
            "boundary points outside the DINI domain (~25%% of the training "
            "boundary points, mostly in the east and north)"
        ),
    )
    p.add_argument(
        "--boundary-smoothing-km",
        type=float,
        default=25.0,
        help=(
            "box-average DINI over this scale before sampling the 0.25 deg "
            "boundary (0: off)"
        ),
    )
    args = parser.parse_args()

    if args.command == "create-danra-grid":
        create_danra_grid(args.output, args.danra_url)
    else:
        regrid(
            dini_root=args.dini_root,
            danra_grid=args.danra_grid,
            forecast_duration=args.forecast_duration,
            output=args.output,
            interior=not args.no_interior,
            boundary=not args.no_boundary,
            outside_domain=args.outside_domain,
            boundary_smoothing_km=args.boundary_smoothing_km,
        )


if __name__ == "__main__":
    main()
