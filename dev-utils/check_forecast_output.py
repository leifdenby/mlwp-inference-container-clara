"""
Sanity-check an ANNA forecast written by `python -m anna_inference`: the
valid times, that all values are finite and within physical ranges, and how
far the forecast is from the DINI forecast it was started from (regridded
to the DANRA grid,
`inputs/interior_*_levels.zarr` in the workdir) at the same valid times.

ANNA and DINI are both forecasts, so they differ, but after a few hours the
differences should be of the size of typical short-range forecast
differences (e.g. ~1-2 K in t2m, a few m/s in the winds). Much larger ones
point at a problem with the inputs (units, wind directions, normalisation).

Usage (from the repository root):
    uv run python \\
        dev-utils/check_forecast_output.py <workdir>
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr

# plausible ranges (min, max) per variable, in the output units
RANGES = dict(
    pres_seasurface=(9.0e4, 1.08e5),
    pres0m=(5.0e4, 1.08e5),
    t2m=(210.0, 330.0),
    u10m=(-60.0, 60.0),
    v10m=(-60.0, 60.0),
    lwavr0m=(-300.0, 100.0),
    swavr0m=(-50.0, 1200.0),
    t=(180.0, 330.0),
    r=(-0.05, 1.05),
    u=(-120.0, 120.0),
    v=(-120.0, 120.0),
    tw=(-10.0, 10.0),
)


def _check(name, ds_out, ds_dini, problems):
    print(f"\n{name}:")
    common = np.intersect1d(ds_out.time.values, ds_dini.time.values)
    for var in ds_out.data_vars:
        da = ds_out[var].load()
        finite = bool(np.isfinite(da).all())
        vmin, vmax = float(da.min()), float(da.max())
        line = f"  {var:16s} [{vmin:11.4g}, {vmax:11.4g}]"
        if not finite:
            problems.append(f"{var} has non-finite values")
        if var in RANGES:
            lo, hi = RANGES[var]
            if vmin < lo or vmax > hi:
                problems.append(
                    f"{var} outside [{lo}, {hi}]: [{vmin:.4g}, {vmax:.4g}]"
                )
                line += "  OUT OF RANGE"
        if var in ds_dini and common.size:
            diff = da.sel(time=common) - ds_dini[var].sel(time=common)
            dims = [d for d in diff.dims if d not in ("time", "pressure")]
            rmse = np.sqrt((diff**2).mean(dims))
            bias = diff.mean(dims)
            if "pressure" in diff.dims:
                for p in diff.pressure.values:
                    r = float(rmse.sel(pressure=p).isel(time=-1))
                    b = float(bias.sel(pressure=p).isel(time=-1))
                    line += f"\n      {int(p):4d} hPa: rmse {r:9.4g} bias {b:+9.4g}"
            else:
                r, b = float(rmse.isel(time=-1)), float(bias.isel(time=-1))
                line += f"  vs DINI: rmse {r:9.4g} bias {b:+9.4g}"
        print(line)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("workdir", type=Path)
    args = parser.parse_args()
    outputs, inputs = args.workdir / "outputs", args.workdir / "inputs"
    problems = []

    for name in ["single_levels", "pressure_levels"]:
        ds_out = xr.open_zarr(outputs / f"{name}.zarr")
        ds_dini = xr.open_zarr(inputs / f"interior_{name}.zarr")
        if "pressure" in ds_dini.dims:
            ds_dini = ds_dini.sel(pressure=ds_out.pressure.values)
        if name == "single_levels":
            analysis = np.datetime64(ds_out.attrs["analysis_time"])
            times = ds_out.time.values
            hours = ((times - analysis) / np.timedelta64(1, "h")).tolist()
            print(f"analysis time {analysis}, lead times (h): {hours}")
            if hours[0] != 6 or np.any(np.diff(hours) != 3):
                problems.append(f"unexpected lead times {hours}")
            common = np.intersect1d(times, ds_dini.time.values)
            print(
                f"compared with DINI at {[str(t)[:16] for t in common]}, "
                "values at the last of these"
            )
        _check(name, ds_out, ds_dini, problems)

    print()
    for problem in problems:
        print("PROBLEM:", problem)
    print("OK" if not problems else f"{len(problems)} problem(s)")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
