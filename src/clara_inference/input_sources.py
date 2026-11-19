"""
The sources CLARA's inputs can come from, and how the outputs describe them.

Operationally, the initial states are a DINI control forecast regridded to
the DANRA grid and the boundary is an IFS forecast, as in fine-tuning. For
hindcasts of past cases (before DINI and the IFS open data), the initial
states can be taken from DANRA itself, the reanalysis CLARA was trained on,
and the boundary from ERA5 (WeatherBench2), laid out as a forecast (see
`ifs_boundary.era5_as_forecast`).

Only the standard library is used here, so that `--help` stays fast.
"""

# the DANRA zarrs of the training configs (`single_levels.zarr`,
# `pressure_levels.zarr` under this root)
DANRA_ROOT = "https://object-store.os-api.cci1.ecmwf.int/danra/v0.5.0/"
# the ERA5 zarr of the training configs (the boundary statics, and the
# boundary of the pre-training before the IFS fine-tuning)
ERA5_PATH = "gs://weatherbench2/datasets/era5/1959-2022-6h-1440x721.zarr"

# `initial_state_source`/`boundary_source` attributes of the outputs
INITIAL_STATE_SOURCES = {
    "dini": "DINI control forecast (analysis and +3 h) regridded to the "
    "DANRA grid",
    "danra": "DANRA v0.5.0 reanalysis (no regridding)",
}
BOUNDARY_SOURCES = {
    "ifs": "IFS forecast (00/12 UTC cycle), as in fine-tuning",
    "era5": "ERA5 reanalysis (WeatherBench2) at the valid times, laid out "
    "as a forecast from the 00/12 UTC cycle: OUT OF DISTRIBUTION for this "
    "checkpoint, which was fine-tuned with an IFS forecast boundary",
}
