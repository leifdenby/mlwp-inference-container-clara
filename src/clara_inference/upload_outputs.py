"""
Upload the forecast outputs (`outputs/{single_levels,pressure_levels}.zarr`
in the workdir) to S3, as `{prefix}/{analysis time}/{name}.zarr`, with the
analysis time in the form of the DINI zarr paths (e.g. 2026-10-01T000000Z).

An earlier upload under the same analysis time is replaced (its objects are
removed first, so no stale chunks remain), and the uploaded zarrs are opened
from S3 and checked against the local ones.

Credentials as for reading DINI (e.g. AWS_PROFILE).

Usage:
    python -m anna_inference.upload_outputs \\
        --outputs inference_workdir/outputs \\
        --analysis-time 2026-10-01T00:00 --prefix s3://anna-forecasts/
"""
import argparse
from pathlib import Path

import numpy as np
import s3fs
import xarray as xr
from loguru import logger

OUTPUT_NAMES = ["single_levels.zarr", "pressure_levels.zarr"]


def target_prefix(prefix, analysis_time):
    """`{prefix}/{analysis time as in the DINI paths}` (no trailing /)."""
    t = np.datetime64(analysis_time, "s").item()
    return f"{prefix.rstrip('/')}/{t:%Y-%m-%dT%H%M%SZ}"


def upload_outputs(outputs, analysis_time, prefix, fs=None):
    fs = fs or s3fs.S3FileSystem()
    target = target_prefix(prefix, analysis_time)
    for name in OUTPUT_NAMES:
        local, remote = Path(outputs) / name, f"{target}/{name}"
        if not local.exists():
            raise FileNotFoundError(local)
        if fs.exists(remote):
            logger.info(f"removing the earlier upload {remote}")
            fs.rm(remote, recursive=True)
        logger.info(f"uploading {local} -> {remote}")
        fs.put(str(local), remote, recursive=True)
        # check: the uploaded zarr opens and has the same contents
        ds_remote = xr.open_zarr(fs.get_mapper(remote))
        ds_local = xr.open_zarr(local)
        xr.testing.assert_identical(ds_remote.load(), ds_local.load())
        logger.info(
            f"uploaded {remote} ({ds_remote.time.size} valid times, checked)"
        )
    return target


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--outputs", required=True, type=Path)
    parser.add_argument("--analysis-time", required=True, type=np.datetime64)
    parser.add_argument("--prefix", required=True, help="e.g. s3://bucket/")
    args = parser.parse_args()
    upload_outputs(args.outputs, args.analysis_time, args.prefix)


if __name__ == "__main__":
    main()
