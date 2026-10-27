"""
Regrid a DINI (HARMONIE, 2 km Lambert) control forecast to the interior
inputs CLARA expects: the DANRA grid (2.5 km Lambert, 589 x 789) with DANRA
variable names and conventions, every 3 h, written as
`{output}/interior_single_levels.zarr` and
`{output}/interior_pressure_levels.zarr` (read through the
`danra_model1_config_ifs_finetune` datastore config).

Adapted from ANNA (dmidk/mlwp-inference-container-anna), without the DINI
boundary: CLARA's boundary is an IFS forecast (see `ifs_boundary.py`).

Conventions (as established for ANNA):
- DINI and DANRA winds are both *grid-relative* (despite DINI's
  eastward_wind/northward_wind labels), on differently rotated Lambert grids.
  Winds are rotated to earth-relative on the DINI grid, interpolated, and then
  rotated to DANRA grid-relative
- DINI relative humidity is in %, DANRA's is a fraction (though labelled %)
- the interior statics (lsm, orography) are taken from DANRA itself
  (`--danra-grid`); DINI `orography` is surface altitude, DANRA's is surface
  geopotential
- `lwavr0m`/`swavr0m` are the same (net surface) fluxes in both

Interpolation is bilinear in DINI's own projection (from the `crs_wkt` stored
with the DINI data); the DANRA grid lies within DINI.

Usage:
    # once (artifact building): cache the DANRA grid and statics
    python -m clara_inference.regrid_dini create-danra-grid \\
        --output danra_model1_config.grid.zarr

    python -m clara_inference.regrid_dini regrid \\
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
import xarray as xr
from loguru import logger

from clara_inference import storage

DANRA_SL_URL = (
    "https://object-store.os-api.cci1.ecmwf.int"
    "/danra/v0.5.0/single_levels.zarr/"
)

LEVELS = [100, 200, 400, 600, 700, 850, 925, 1000]  # hPa, as used by CLARA
INTERIOR_STEP = datetime.timedelta(hours=3)
# interior time steps needed beyond analysis_time + forecast_duration: one for
# `num_future_forcing_steps=1`, and one more because neural-lam's
# WeatherDataset counts samples as
# `n_times - (2 + ar_steps) - num_future_forcing_steps`, one step more
# conservative than the data a sample actually uses
INTERIOR_EXTRA_STEPS = 2

INTERIOR_SL_VARS = ["pres_seasurface", "t2m", "pres0m", "lwavr0m", "swavr0m"]
INTERIOR_PL_VARS = ["z", "t", "r", "tw"]  # plus winds u, v

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
    ds_sl = storage.open_zarr(f"{root}/single_levels.zarr")
    ds_pl = storage.open_zarr(f"{root}/pressure_levels.zarr")
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
    CLARA was trained with.
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


def _regrid_interior(ds_sl, ds_pl, danra_grid, times, transformer, theta_dini):
    lat, lon = danra_grid.lat.values, danra_grid.lon.values
    x_tgt, y_tgt = transformer.transform(lon, lat)
    interp = BilinearInterpolator(
        ds_sl.x.values, ds_sl.y.values, x_tgt, y_tgt, outside="error"
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
    return _set_units(ds_sl_out), _set_units(ds_pl_out)


def regrid(dini_root, danra_grid, forecast_duration, output):
    ds_sl, ds_pl = _open_dini(dini_root)
    analysis_time = ds_sl.time.values[0].astype("datetime64[s]").item()
    logger.info(f"DINI forecast from {analysis_time}, output to {output}")
    transformer = _dini_transformer(ds_sl)
    theta_dini = xr.DataArray(
        grid_x_axis_angle(ds_sl.lat.values, ds_sl.lon.values), dims=("y", "x")
    )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    # two interior steps beyond the forecast are needed, see
    # INTERIOR_EXTRA_STEPS
    times = _valid_times(
        ds_sl,
        analysis_time,
        forecast_duration + INTERIOR_EXTRA_STEPS * INTERIOR_STEP,
        INTERIOR_STEP,
    )
    ds_sl_out, ds_pl_out = _regrid_interior(
        ds_sl,
        ds_pl,
        xr.open_zarr(danra_grid).load(),
        times,
        transformer,
        theta_dini,
    )
    attrs = dict(
        source=str(dini_root),
        description="DINI regridded to the DANRA grid (regrid_dini.py)",
    )
    ds_sl_out.attrs = ds_pl_out.attrs = attrs
    outputs = {
        "interior_single_levels.zarr": ds_sl_out,
        "interior_pressure_levels.zarr": ds_pl_out,
    }
    logger.info(f"interior: {len(times)} time steps, every {INTERIOR_STEP}")

    # computed together, so that dask reads the DINI chunks concurrently
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
    args = parser.parse_args()

    if args.command == "create-danra-grid":
        create_danra_grid(args.output, args.danra_url)
    else:
        regrid(
            dini_root=args.dini_root,
            danra_grid=args.danra_grid,
            forecast_duration=args.forecast_duration,
            output=args.output,
        )


if __name__ == "__main__":
    main()
