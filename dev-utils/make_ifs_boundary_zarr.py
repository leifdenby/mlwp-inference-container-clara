"""
Make an IFS boundary zarr for CLARA (the contract in
`clara_inference/ifs_boundary.py`) from ECMWF open data: one 00/12 UTC
forecast cycle, lead times every 6 h, on the native 0.25 deg open-data grid
cropped to a box around the boundary (lat 40..72, lon -30..50).

Adapted from `scripts/download_ifs_open_data_forecast.py` in real-prob-lam
(which wrote 3-hourly lead times; CLARA was fine-tuned with 6-hourly ones).

Open data is only retained for a few days. The cycle defaults to the latest
00/12 UTC cycle available.

Usage:
    uv run python dev-utils/make_ifs_boundary_zarr.py \\
        --cycle 2026-10-26T00:00Z --max-lead-hours 48 \\
        --output dev-data/ifs/2026-10-26T000000Z/ifs.zarr
"""
import argparse
import datetime
import shutil
from pathlib import Path

import earthkit.data as ekd
import ecmwf.opendata
import isodate
import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar

SL_PARAMS = {
    "msl": "mean_sea_level_pressure",
    "2t": "2m_temperature",
    "10u": "10m_u_component_of_wind",
    "10v": "10m_v_component_of_wind",
    "sp": "surface_pressure",
}
PL_PARAMS = {
    "gh": "geopotential",  # geopotential height, converted to geopotential
    "t": "temperature",
    "q": "specific_humidity",
    "u": "u_component_of_wind",
    "v": "v_component_of_wind",
    "w": "vertical_velocity",
}
LEVELS = [100, 200, 400, 600, 700, 850, 925, 1000]
LAT_RANGE = (40.0, 72.0)
LON_RANGE = (-30.0, 50.0)
GRAVITY = 9.80665


def latest_cycle(source):
    """The latest 00/12 UTC cycle available from open data."""
    client = ecmwf.opendata.Client(source=source, model="ifs", resol="0p25")
    return max(
        client.latest(stream="oper", type="fc", time=hour) for hour in (0, 12)
    )


def _fetch(cycle, steps, source, **kwargs):
    fieldlist = ekd.from_source(
        "ecmwf-open-data",
        source=source,
        date=int(cycle.strftime("%Y%m%d")),
        time=cycle.hour,
        step=steps,
        stream="oper",
        type="fc",
        resol="0p25",
        **kwargs,
    )
    ds = fieldlist.to_xarray(
        time_dims=["forecast_reference_time", "step"],
        ensure_dims=["forecast_reference_time", "step"]
        + (["level"] if kwargs.get("levtype") == "pl" else []),
    )
    lon = ds.longitude.values
    ds = ds.assign_coords(longitude=np.where(lon >= 180, lon - 360, lon))
    ds = ds.sortby("longitude").sortby("latitude")
    return ds.sel(latitude=slice(*LAT_RANGE), longitude=slice(*LON_RANGE))


def make_boundary(cycle, max_lead_hours, source):
    if cycle.hour not in (0, 12) or cycle.minute:
        raise ValueError(f"CLARA uses 00/12 UTC cycles, got {cycle}")
    steps = list(range(0, max_lead_hours + 1, 6))
    print(f"IFS open data ({source}): cycle {cycle}, steps {steps} h")
    ds_sl = _fetch(cycle, steps, source, levtype="sfc", param=list(SL_PARAMS))
    ds_pl = _fetch(
        cycle,
        steps,
        source,
        levtype="pl",
        param=list(PL_PARAMS),
        levelist=LEVELS,
    )
    ds_pl["gh"] = ds_pl["gh"] * GRAVITY
    ds = xr.merge(
        [ds_sl.rename_vars(SL_PARAMS), ds_pl.rename_vars(PL_PARAMS)],
        compat="override",
    )
    ds["geopotential"].attrs.update(units="m**2 s**-2")
    ds = ds.rename(
        forecast_reference_time="time", step="prediction_timedelta"
    )
    ds = ds.assign_coords(
        time=ds.time.values.astype("datetime64[ns]"),
        prediction_timedelta=ds.prediction_timedelta.values.astype(
            "timedelta64[ns]"
        ),
    )
    keep = list(SL_PARAMS.values()) + list(PL_PARAMS.values())
    ds = ds[keep].drop_vars(
        [c for c in ds.coords if c not in ds[keep].dims]
    )
    for name in ds.variables:
        ds[name].encoding = {}
    ds.attrs = dict(
        source=f"ECMWF open data ({source}), IFS oper fc 0p25",
        description="IFS boundary for CLARA (dev-utils/make_ifs_boundary_zarr.py)",
    )
    return ds.astype("float32").chunk({"time": 1, "prediction_timedelta": 1})


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--cycle",
        help="00/12 UTC cycle, ISO8601 (default: the latest available)",
    )
    parser.add_argument("--max-lead-hours", type=int, default=48)
    parser.add_argument(
        "--source",
        choices=["ecmwf", "aws", "azure", "google"],
        default="azure",
        help="open-data mirror (default: %(default)s)",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.cycle:
        cycle = isodate.parse_datetime(args.cycle)
        if cycle.tzinfo is not None:
            cycle = cycle.astimezone(datetime.timezone.utc).replace(
                tzinfo=None
            )
    else:
        cycle = latest_cycle(args.source)
    if args.output.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output} exists (--overwrite)")
        shutil.rmtree(args.output)
    ds = make_boundary(cycle, args.max_lead_hours, args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with ProgressBar():
        ds.to_zarr(args.output, mode="w", consolidated=True)
    print(f"wrote {args.output}: {dict(ds.sizes)}")


if __name__ == "__main__":
    main()
