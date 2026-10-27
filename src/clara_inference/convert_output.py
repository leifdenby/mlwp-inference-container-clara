"""
Convert the neural-lam evaluation output of ANNA into DANRA-like datasets.

neural-lam (`train_model --eval test --save_eval_to_zarr_path ...`) writes the
predicted (de-normalised) state as `state(start_time,
elapsed_forecast_duration, state_feature, x, y)`. This splits the state
features back into variables:

- `single_levels.zarr`: pres_seasurface, t2m, u10m, v10m, pres0m, lwavr0m,
  swavr0m with dims (time, y, x)
- `pressure_levels.zarr`: z, t, r, u, v, tw with dims (time, pressure, y, x)

where `time` is the valid time, on the DANRA grid (with lat/lon from the
DANRA grid in the inference artifact). Conventions are those of DANRA/ANNA:
winds are relative to the DANRA (Lambert) grid and relative humidity is a
fraction (0-1).

The grid is described with CF metadata as in the DINI zarrs: x/y in
projection metres, and a `danra_projection` grid mapping variable (CF Lambert
parameters and WKT) named by every field's `grid_mapping`. The projection
comes from the interior datastore config (`extra.projection`, as neural-lam
reads it). Viewers such as Gridlook need it to recognise the Lambert grid;
without it they take x/y for lon/lat in degrees. NB: DANRA's own
`danra_projection` has wrong CF parameters (e.g. central meridian 0.025
instead of 25), so it isn't copied.

Usage:
    python -m anna_inference.convert_output --prediction inference_output.zarr \\
        --danra-grid inference_artifact/grids/danra_model1_config.grid.zarr \\
        --interior-config inference_artifact/configs/danra_model1_config.yaml \\
        --analysis-time 2026-10-01T00:00 --output-dir outputs/
"""
import argparse
import re
from pathlib import Path

import cartopy.crs as ccrs
import numpy as np
import xarray as xr
import yaml
from loguru import logger

SINGLE_LEVEL_VARS = [
    "pres_seasurface",
    "t2m",
    "u10m",
    "v10m",
    "pres0m",
    "lwavr0m",
    "swavr0m",
]
PRESSURE_LEVEL_VARS = ["z", "t", "r", "u", "v", "tw"]
UNITS = {
    "pres_seasurface": "Pa",
    "t2m": "K",
    "u10m": "m s**-1",
    "v10m": "m s**-1",
    "pres0m": "Pa",
    "lwavr0m": "W m**-2",
    "swavr0m": "W m**-2",
    "z": "m**2 s**-2",
    "t": "K",
    "r": "1",
    "u": "m s**-1",
    "v": "m s**-1",
    "tw": "m s**-1",
}
NOTES = {
    "u10m": "relative to the DANRA (Lambert) grid x-axis",
    "v10m": "relative to the DANRA (Lambert) grid y-axis",
    "u": "relative to the DANRA (Lambert) grid x-axis",
    "v": "relative to the DANRA (Lambert) grid y-axis",
    "r": "fraction (0-1), not percent",
    "lwavr0m": "net surface long-wave radiation flux",
    "swavr0m": "net surface short-wave radiation flux",
    "tw": "geometric vertical velocity",
}
GRID_MAPPING = "danra_projection"


def projection_cf_attrs(interior_config):
    """
    CF grid mapping attributes (incl. `crs_wkt`) of the projection in an mdp
    datastore config's `extra.projection`, built as neural-lam does.
    """
    with open(interior_config) as f:
        projection = yaml.safe_load(f)["extra"]["projection"]
    kwargs = dict(projection["kwargs"])
    if "globe" in kwargs:
        kwargs["globe"] = ccrs.Globe(**kwargs["globe"])
    crs = getattr(ccrs, projection["class_name"])(**kwargs)
    return {
        k: list(v) if isinstance(v, tuple) else v
        for k, v in crs.to_cf().items()
    }


def convert(ds_pred, ds_grid, analysis_time, projection_attrs):
    """Split the predicted state features into single/pressure level datasets."""
    if ds_pred.start_time.size != 1:
        raise ValueError(
            f"expected a single forecast, got {ds_pred.start_time.size} start times"
        )
    da = ds_pred.state.isel(start_time=0)
    valid_time = (
        ds_pred.start_time.values[0] + da.elapsed_forecast_duration
    ).values
    da = da.assign_coords(time=("elapsed_forecast_duration", valid_time))
    da = da.swap_dims(elapsed_forecast_duration="time").drop_vars(
        "elapsed_forecast_duration"
    )
    features = [str(f) for f in da.state_feature.values]

    # with the grid's CF attrs (x/y: projection_{x,y}_coordinate in m)
    coords = dict(
        x=("x", ds_grid.x.values, ds_grid.x.attrs),
        y=("y", ds_grid.y.values, ds_grid.y.attrs),
        lat=(("y", "x"), ds_grid.lat.values, ds_grid.lat.attrs),
        lon=(("y", "x"), ds_grid.lon.values, ds_grid.lon.attrs),
    )
    attrs = dict(
        source="ANNA forecast (DANRA ML LAM model, arXiv:2504.09340)",
        analysis_time=str(np.datetime64(analysis_time, "s")),
        start_time=str(ds_pred.start_time.values[0]),
    )

    def _var(feature):
        return (
            da.sel(state_feature=feature)
            .drop_vars("state_feature")
            .transpose("time", "y", "x")
        )

    ds_sl = xr.Dataset({v: _var(v) for v in SINGLE_LEVEL_VARS}).assign_coords(
        **coords
    )

    levels = sorted(
        {
            int(m.group(1))
            for f in features
            if (m := re.fullmatch(r"z(\d+)", f))
        }
    )
    ds_pl = xr.Dataset(
        {
            v: xr.concat(
                [_var(f"{v}{p}") for p in levels],
                dim=xr.DataArray(levels, dims="pressure", name="pressure"),
            ).transpose("time", "pressure", "y", "x")
            for v in PRESSURE_LEVEL_VARS
        }
    ).assign_coords(**coords)
    ds_pl.pressure.attrs["units"] = "hPa"

    expected = set(SINGLE_LEVEL_VARS) | {
        f"{v}{p}" for v in PRESSURE_LEVEL_VARS for p in levels
    }
    if expected != set(features):
        raise ValueError(
            f"unexpected state features: {set(features) ^ expected}"
        )

    for ds in [ds_sl, ds_pl]:
        ds.attrs = attrs
        for v in ds.data_vars:
            ds[v].attrs["units"] = UNITS[v]
            ds[v].attrs["grid_mapping"] = GRID_MAPPING
            # neural-lam's encoding (`coordinates: time`) would keep xarray
            # from writing `coordinates: lat lon` on the variable
            ds[v].encoding.pop("coordinates", None)
            if v in NOTES:
                ds[v].attrs["comment"] = NOTES[v]
        ds[GRID_MAPPING] = xr.DataArray(0, attrs=projection_attrs)
        # time units as in the DINI zarrs
        ds.time.encoding["units"] = "seconds since 1970-01-01"
    return ds_sl, ds_pl


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--prediction", required=True, type=Path)
    parser.add_argument("--danra-grid", required=True, type=Path)
    parser.add_argument(
        "--interior-config",
        required=True,
        type=Path,
        help="interior (DANRA) datastore config, for the grid's projection",
    )
    parser.add_argument("--analysis-time", required=True, type=np.datetime64)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    ds_pred = xr.open_zarr(args.prediction)
    ds_grid = xr.open_zarr(args.danra_grid).load()
    ds_sl, ds_pl = convert(
        ds_pred,
        ds_grid,
        args.analysis_time,
        projection_cf_attrs(args.interior_config),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, ds in [("single_levels", ds_sl), ("pressure_levels", ds_pl)]:
        fp = args.output_dir / f"{name}.zarr"
        ds.to_zarr(fp, mode="w", consolidated=True)
        logger.info(
            f"wrote {fp} ({ds.time.size} valid times {ds.time.values})"
        )


if __name__ == "__main__":
    main()
