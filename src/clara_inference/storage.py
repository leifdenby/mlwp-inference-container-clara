"""
Opening the input zarrs (DINI, IFS), locally or on S3.

S3 inputs are read with the usual AWS configuration (credentials, profile,
endpoint; see run.py for the per-side options). Without any AWS credentials
they are read anonymously, which works for public buckets such as DINI's
`harmonie-zarr`.
"""
import botocore.session
import xarray as xr
from loguru import logger


def _have_aws_credentials():
    return botocore.session.get_session().get_credentials() is not None


def open_zarr(path, **kwargs):
    """`xr.open_zarr`, anonymous for S3 without AWS credentials."""
    path = str(path)
    if path.startswith("s3://") and not _have_aws_credentials():
        logger.info(f"no AWS credentials, reading {path} anonymously")
        kwargs["storage_options"] = {
            **kwargs.get("storage_options", {}),
            "anon": True,
        }
    return xr.open_zarr(path, **kwargs)
