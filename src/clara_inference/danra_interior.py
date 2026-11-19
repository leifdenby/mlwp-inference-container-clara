"""
Take CLARA's interior inputs (initial states, and the "targets" neural-lam
needs alongside them) from DANRA itself, for hindcasts of past cases.

DANRA is the reanalysis CLARA was trained on, so nothing is regridded or
converted: the variables, levels and statics of the interior datastore config
are copied for the valid times of the forecast into
`{output}/interior_single_levels.zarr` and
`{output}/interior_pressure_levels.zarr`, the files `regrid_dini.py` writes
for a DINI forecast. The rest of the pipeline is the same for both.

NB: DANRA's zarrs are chunked by 256 time steps (32 days), so even a short
forecast reads a month of each variable (about 13 GB for 1999-12).

Usage:
    python -m clara_inference.danra_interior \\
        --analysis-time 1999-12-02T21:00 --forecast-duration PT27H \\
        --output inference_workdir/inputs
"""
import argparse
import datetime
import time
from pathlib import Path

import dask
import dask.diagnostics
import isodate
import numpy as np
import xarray as xr
from loguru import logger

from clara_inference import storage
from clara_inference.input_sources import DANRA_ROOT
from clara_inference.regrid_dini import (
    INTERIOR_EXTRA_STEPS,
    INTERIOR_PL_VARS,
    INTERIOR_SL_VARS,
    INTERIOR_STEP,
    LEVELS,
)

SL_VARS = INTERIOR_SL_VARS + ["u10m", "v10m"]
PL_VARS = INTERIOR_PL_VARS + ["u", "v"]
STATIC_VARS = ["lsm", "orography"]
# the coordinates the interior datastore config uses (lat/lon for the TOA
# radiation forcing)
COORDS = ["time", "x", "y", "pressure", "lat", "lon"]


def get_valid_times(
    analysis_time: datetime.datetime, forecast_duration: datetime.timedelta
) -> np.ndarray:
    """
    Get the interior valid times of a forecast.

    Parameters
    ----------
    analysis_time : datetime.datetime
        The time of the first initial state (naive UTC).
    forecast_duration : datetime.timedelta
        The forecast duration from `analysis_time`.

    Returns
    -------
    np.ndarray
        `analysis_time` every 3 h up to `analysis_time + forecast_duration`
        and the extra steps neural-lam needs (`INTERIOR_EXTRA_STEPS`), as
        datetime64[ns].
    """
    n_steps = int(forecast_duration / INTERIOR_STEP) + INTERIOR_EXTRA_STEPS
    return np.array(
        [
            np.datetime64(analysis_time + k * INTERIOR_STEP, "ns")
            for k in range(n_steps + 1)
        ]
    )


def select_danra_interior(
    ds_sl: xr.Dataset, ds_pl: xr.Dataset, valid_times: np.ndarray
) -> tuple[xr.Dataset, xr.Dataset]:
    """
    Select CLARA's interior inputs from the DANRA zarrs (lazily).

    Parameters
    ----------
    ds_sl : xr.Dataset
        DANRA single levels (with the statics `lsm` and `orography`).
    ds_pl : xr.Dataset
        DANRA pressure levels.
    valid_times : np.ndarray
        The valid times to select.

    Returns
    -------
    tuple[xr.Dataset, xr.Dataset]
        The single level (state variables and statics) and pressure level
        datasets, with DANRA's own names, units and attributes.

    Raises
    ------
    ValueError
        If DANRA doesn't have all the valid times.
    """
    missing = sorted(
        set(valid_times.tolist()) - set(ds_sl.time.values.tolist())
    )
    if missing:
        missing_str = [str(np.datetime64(m, "ns"))[:16] for m in missing]
        raise ValueError(
            f"DANRA has no valid times {missing_str} "
            f"(available: {str(ds_sl.time.values[0])[:16]} .. "
            f"{str(ds_sl.time.values[-1])[:16]})"
        )
    ds_sl_out = ds_sl[SL_VARS + STATIC_VARS].sel(time=valid_times)
    ds_pl_out = ds_pl[PL_VARS].sel(time=valid_times, pressure=LEVELS)
    ds_sl_out, ds_pl_out = (
        ds.drop_vars([c for c in ds.coords if c not in COORDS])
        for ds in (ds_sl_out, ds_pl_out)
    )
    # as for the DINI inputs, one zarr chunk per time step (DANRA's 256-step
    # chunks don't carry over)
    for ds in (ds_sl_out, ds_pl_out):
        for v in ds.variables:
            ds[v].encoding = {}
    return ds_sl_out, ds_pl_out


def extract_danra_interior(
    analysis_time: datetime.datetime,
    forecast_duration: datetime.timedelta,
    output: Path,
    danra_root: str = DANRA_ROOT,
) -> None:
    """
    Write CLARA's interior inputs for a forecast from DANRA.

    Parameters
    ----------
    analysis_time : datetime.datetime
        The time of the first initial state (naive UTC).
    forecast_duration : datetime.timedelta
        The forecast duration from `analysis_time`.
    output : Path
        Directory for `interior_single_levels.zarr` and
        `interior_pressure_levels.zarr`.
    danra_root : str
        Directory (or URL) with DANRA's `single_levels.zarr` and
        `pressure_levels.zarr`.
    """
    root = danra_root.rstrip("/")
    ds_sl = storage.open_zarr(f"{root}/single_levels.zarr")
    ds_pl = storage.open_zarr(f"{root}/pressure_levels.zarr")
    valid_times = get_valid_times(analysis_time, forecast_duration)
    ds_sl_out, ds_pl_out = select_danra_interior(ds_sl, ds_pl, valid_times)
    attrs = dict(
        source=root,
        description="DANRA, the interior inputs of CLARA as in training "
        "(danra_interior.py)",
    )
    ds_sl_out.attrs = ds_pl_out.attrs = attrs
    outputs = {
        "interior_single_levels.zarr": ds_sl_out,
        "interior_pressure_levels.zarr": ds_pl_out,
    }
    logger.info(
        f"DANRA interior: {valid_times.size} time steps "
        f"{str(valid_times[0])[:16]} .. {str(valid_times[-1])[:16]}, "
        f"output to {output}"
    )
    # computed together, so that dask reads the DANRA chunks concurrently
    start = time.perf_counter()
    with dask.diagnostics.ProgressBar(dt=10):
        computed = dask.compute(*outputs.values())
    logger.info(f"read in {time.perf_counter() - start:.0f} s")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for name, ds in zip(outputs, computed):
        ds.to_zarr(output / name, mode="w", consolidated=True)
        logger.info(f"wrote {output / name}")


def _parse_analysis_time(s: str) -> datetime.datetime:
    t = isodate.parse_datetime(s)
    if t.tzinfo is not None:
        t = t.astimezone(datetime.timezone.utc).replace(tzinfo=None)
    return t


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--analysis-time",
        required=True,
        type=_parse_analysis_time,
        help="time of the first initial state, ISO8601 (UTC if no timezone)",
    )
    parser.add_argument(
        "--forecast-duration",
        required=True,
        type=isodate.parse_duration,
        help="ISO8601 duration, e.g. PT18H",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--danra-root",
        default=DANRA_ROOT,
        help="DANRA zarrs root (default: %(default)s)",
    )
    args = parser.parse_args()
    extract_danra_interior(
        analysis_time=args.analysis_time,
        forecast_duration=args.forecast_duration,
        output=args.output,
        danra_root=args.danra_root,
    )


if __name__ == "__main__":
    main()
