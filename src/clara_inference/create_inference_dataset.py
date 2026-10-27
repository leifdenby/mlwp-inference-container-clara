"""
Create the inference datastores and neural-lam config for running ANNA
from regridded DINI (interior) and DINI or IFS (boundary) data.

Inputs:
- the inference artifact (`anna_inference.artifact`), which describes the
  model as trained: its neural-lam config (named in `configs/model.yaml`),
  the interior (DANRA) and boundary (ERA5) datastore configs and their
  training statistics
- the operational boundary datastore config for the boundary source
  (`anna_inference/boundary/{dini,ifs}_7deg_model1_config.yaml`, see
  `model_configs.py`)
- the regridded interior (`interior_{single,pressure}_levels.zarr` from
  `regrid_dini.py`)
- the boundary forecast in the IFS contract layout: `boundary.zarr` from
  `regrid_dini.py` (DINI boundary) or a converted IFS forecast. It is turned
  into the valid-time layout of the training (ERA5) boundary first, see
  `boundary_to_valid_time`

Written to the inference workdir (neural-lam opens `{name}.zarr` next to each
`{name}.yaml`):
- `danra_model1_config.{yaml,zarr}`: interior datastore covering
  analysis_time .. analysis_time + forecast_duration (+ one step for
  `num_future_forcing_steps=1`), with the DANRA *training* statistics merged
  in (statistics are not computed from the inference data)
- `{dini,ifs}_7deg_model1_config.input.zarr`: the boundary forecast in
  valid-time layout
- `{dini,ifs}_7deg_model1_config.{yaml,zarr}`: boundary datastore, cropped to
  the boundary points around the interior; normalised with the ERA5 training
  statistics in the artifact (`overload_stats_path`)
- `config.yaml`: neural-lam config referencing the above, derived from the
  training config (`model_configs.inference_neural_lam_config`)

NB: mllam-data-prep configs must never be written with sorted keys, that
would reorder inputs/variables and so the features the checkpoint expects.

Usage:
    python -m anna_inference.create_inference_dataset --artifact inference_artifact \\
        --interior-dir inference_workdir/inputs \\
        --boundary inference_workdir/inputs/boundary.zarr --boundary-source dini \\
        --analysis-time 2026-09-26T18:00 --forecast-duration PT18H \\
        --workdir inference_workdir
"""
import argparse
import datetime
import os
from pathlib import Path

import isodate
import mllam_data_prep as mdp
import mllam_data_prep.config as mdp_config
import numpy as np
import xarray as xr
import yaml
from loguru import logger

from anna_inference.model_configs import (
    BOUNDARY_SOURCES,
    boundary_config_path,
    inference_neural_lam_config,
    training_configs,
)

# interior time step and the boundary time step ANNA was trained with
INTERIOR_STEP = datetime.timedelta(hours=3)
BOUNDARY_STEP = datetime.timedelta(hours=6)
# interior time steps needed beyond analysis_time + forecast_duration: one for
# `num_future_forcing_steps=1`, and one more because neural-lam's
# WeatherDataset counts samples as
# `n_times - (2 + ar_steps) - num_future_forcing_steps`, one step more
# conservative than the data a sample actually uses (same as in
# regrid_dini.py)
INTERIOR_EXTRA_STEPS = 2
# shortest forecast: neural-lam's ARModel (when loading the checkpoint) builds
# a WeatherDataset per split with its default `ar_steps=3`, which needs
# 2 + 3 + 1 + 1 interior time steps (see INTERIOR_EXTRA_STEPS), i.e. a
# forecast of at least 12h. Shorter forecasts fail in train_model
MIN_FORECAST_DURATION = datetime.timedelta(hours=12)


def ar_steps_for(forecast_duration):
    """
    Autoregressive steps for a forecast: the initial states are at the
    analysis time and +3h, the predictions at +6h .. +forecast_duration.
    """
    return int(forecast_duration / INTERIOR_STEP) - 1


def _floor_boundary_step(t):
    """The boundary time (00/06/12/18 UTC) at or before `t`."""
    midnight = datetime.datetime.combine(t.date(), datetime.time())
    return t - (t - midnight) % BOUNDARY_STEP


def _time_range(start, end):
    """Times from `start` to `end` (inclusive) every BOUNDARY_STEP."""
    n = int((end - start) / BOUNDARY_STEP) + 1
    return [start + i * BOUNDARY_STEP for i in range(n)]


def boundary_times(analysis_time, forecast_duration):
    """
    The boundary (valid) times of a forecast: those the model uses, and the
    times neural-lam requires the boundary datastore to cover.

    For each prediction at time t (analysis_time + 6h .. + forecast_duration)
    neural-lam takes the boundary at the last boundary time at or before t,
    plus one boundary step either side (`num_{past,future}_boundary_steps=1`)
    - in training the 00/06/12/18 UTC ERA5 times. neural-lam's coverage check
    is more conservative: it requires the boundary to cover one boundary step
    before the first interior time and after the last one (which includes
    INTERIOR_EXTRA_STEPS). Those extra times are never used and are filled
    with copies of the nearest used time.

    Returns (used, required): lists of datetimes, `used` within `required`.
    """
    first_prediction = analysis_time + 2 * INTERIOR_STEP
    last_prediction = analysis_time + forecast_duration
    used = _time_range(
        _floor_boundary_step(first_prediction) - BOUNDARY_STEP,
        _floor_boundary_step(last_prediction) + BOUNDARY_STEP,
    )
    interior_end = last_prediction + INTERIOR_EXTRA_STEPS * INTERIOR_STEP
    required_start = min(
        used[0], _floor_boundary_step(analysis_time - BOUNDARY_STEP)
    )
    # the first boundary time at or after interior_end + one boundary step
    required_end = _floor_boundary_step(interior_end + BOUNDARY_STEP)
    if required_end < interior_end + BOUNDARY_STEP:
        required_end += BOUNDARY_STEP
    required = _time_range(required_start, max(required_end, used[-1]))
    return used, required


def boundary_to_valid_time(ds, analysis_time, forecast_duration):
    """
    Turn a boundary forecast in the IFS contract layout (`time` = cycle,
    `prediction_timedelta` = lead time) into the valid-time layout of the ERA5
    boundary the model was trained with (`time` = valid time, 6-hourly at
    00/06/12/18 UTC), as read by the DINI/IFS boundary datastore configs.

    The latest cycle at or before `analysis_time` is used. The times the
    model uses (see `boundary_times`) must be among its valid times; the
    extra times neural-lam's coverage check requires are copies of the
    nearest used time.
    """
    used, required = boundary_times(analysis_time, forecast_duration)
    cycles = ds.time.values[ds.time.values <= np.datetime64(analysis_time)]
    if cycles.size == 0:
        raise ValueError(
            f"no boundary cycle at or before the analysis time "
            f"{analysis_time} (have {ds.time.values})"
        )
    cycle = cycles.max()

    time_dependent = [v for v in ds.data_vars if "time" in ds[v].dims]
    ds_cycle = ds[time_dependent].sel(time=cycle).drop_vars("time")
    valid_times = cycle + ds_cycle.prediction_timedelta.values
    ds_cycle = (
        ds_cycle.assign_coords(time=("prediction_timedelta", valid_times))
        .swap_dims(prediction_timedelta="time")
        .drop_vars("prediction_timedelta")
    )

    used_ns = np.array(used, dtype="datetime64[ns]")
    missing = sorted(set(used_ns) - set(ds_cycle.time.values))
    if missing:
        raise ValueError(
            f"the boundary cycle {cycle} has no data at "
            f"{[str(t)[:16] for t in missing]}, needed for a "
            f"{forecast_duration} forecast from {analysis_time} (boundary "
            f"times {used[0]} .. {used[-1]}, 6-hourly at 00/06/12/18 UTC)"
        )
    # take each required time from the nearest used time
    source_times = np.clip(
        np.array(required, dtype="datetime64[ns]"), used_ns[0], used_ns[-1]
    )
    ds_valid = ds_cycle.sel(time=source_times).assign_coords(
        time=np.array(required, dtype="datetime64[ns]")
    )
    ds_valid = xr.merge(
        [ds_valid, ds.drop_vars(time_dependent + ["time"], errors="ignore")],
        combine_attrs="override",
    )
    ds_valid = ds_valid.drop_vars("prediction_timedelta", errors="ignore")
    ds_valid = ds_valid.load()
    padded = [t for t in required if t < used[0] or t > used[-1]]
    ds_valid.attrs.update(
        boundary_cycle=str(cycle)[:19],
        boundary_times_used=f"{used[0]:%Y-%m-%dT%H:%M} .. "
        f"{used[-1]:%Y-%m-%dT%H:%M}",
        boundary_times_padded=", ".join(f"{t:%Y-%m-%dT%H:%M}" for t in padded),
    )
    return ds_valid


SPLIT_NAMES = ["train", "val", "test"]
INTERIOR_INPUT_FILES = dict(
    danra_sl_state="interior_single_levels.zarr",
    danra_pl_state="interior_pressure_levels.zarr",
    danra_static="interior_single_levels.zarr",
    danra_forcing="interior_single_levels.zarr",
)


def _write_config(config, fp):
    # NB: sort_keys=False, sorting would change the feature order
    config.to_yaml_file(fp, sort_keys=False)


def _create_interior_datastore(
    artifact, configs, interior_dir, analysis_time, interior_end, workdir
):
    name = configs.name(configs.interior_datastore)
    config = mdp.Config.from_yaml_file(
        artifact / "configs" / configs.interior_datastore
    )
    for input_name, input_config in config.inputs.items():
        input_config.path = str(
            (interior_dir / INTERIOR_INPUT_FILES[input_name]).resolve()
        )

    start, end = analysis_time.isoformat(), interior_end.isoformat()
    config.output.coord_ranges["time"].start = start
    config.output.coord_ranges["time"].end = end
    # neural-lam requires train/val/test splits, the forecast is run on "test".
    # Statistics are not computed from the inference data, the training
    # statistics are merged in below
    config.output.splitting.splits = {
        name: mdp_config.Split(start=start, end=end) for name in SPLIT_NAMES
    }

    fp_config = workdir / f"{name}.yaml"
    _write_config(config, fp_config)

    ds = mdp.create_dataset(config=config)
    ds_stats = xr.open_zarr(artifact / "stats" / f"{name}.stats.zarr").load()
    for category in ["state", "forcing", "static"]:
        dim = f"{category}_feature"
        if dim in ds_stats.dims and list(ds_stats[dim].values) != list(
            ds[dim].values
        ):
            raise ValueError(
                f"{category} features of the inference dataset don't match "
                "those of the training statistics"
            )
    # only merge the statistics themselves, the feature metadata coordinates
    # (units, long names) describe the inference data
    stats_vars = [v for v in ds_stats.data_vars if "__train__" in v]
    ds_stats = ds_stats[stats_vars].reset_coords(drop=True)
    ds = xr.merge([ds, ds_stats], join="exact", combine_attrs="override")

    n_missing = {
        v: int(ds[v].isnull().sum()) for v in ["state", "forcing", "static"]
    }
    if any(n_missing.values()):
        raise ValueError(
            f"missing values in the interior datastore: {n_missing}"
        )
    ds.to_zarr(workdir / f"{name}.zarr", mode="w", consolidated=True)
    logger.info(
        f"interior datastore: {ds.time.size} time steps {start} .. {end}, "
        f"{ds.grid_index.size} grid points"
    )
    return fp_config


def _create_boundary_datastore(
    boundary_source,
    fp_boundary,
    fp_interior_config,
    analysis_time,
    forecast_duration,
    workdir,
):
    # the operational boundary datastore config (package data)
    fp_source_config = boundary_config_path(boundary_source)
    name = fp_source_config.stem
    # the datastore config reads the boundary in the valid-time layout of the
    # training (ERA5) boundary
    ds_valid = boundary_to_valid_time(
        xr.open_zarr(fp_boundary), analysis_time, forecast_duration
    )
    fp_valid = workdir / f"{name}.input.zarr"
    ds_valid.to_zarr(fp_valid, mode="w", consolidated=True)
    logger.info(
        f"boundary ({boundary_source}) from cycle "
        f"{ds_valid.attrs['boundary_cycle']}: valid times used "
        f"{ds_valid.attrs['boundary_times_used']}, padded (unused) "
        f"{ds_valid.attrs['boundary_times_padded']}"
    )

    config = mdp.Config.from_yaml_file(fp_source_config)
    for input_config in config.inputs.values():
        input_config.path = str(fp_valid.resolve())
    # crop around the (small) inference interior dataset, rather than the
    # training interior dataset. The path is resolved relative to the CWD by
    # mllam-data-prep, so make it absolute
    config.output.domain_cropping.interior_dataset_config_path = str(
        fp_interior_config.resolve()
    )

    fp_config = workdir / f"{name}.yaml"
    _write_config(config, fp_config)
    ds = mdp.create_dataset(config=config)

    n_missing = {v: int(ds[v].isnull().sum()) for v in ["forcing", "static"]}
    if any(n_missing.values()):
        raise ValueError(
            f"missing values in the boundary datastore: {n_missing}"
        )
    ds.to_zarr(workdir / f"{name}.zarr", mode="w", consolidated=True)
    logger.info(
        f"boundary datastore ({boundary_source}): {ds.time.size} times "
        f"{str(ds.time.values[0])[:16]} .. {str(ds.time.values[-1])[:16]}, "
        f"{ds.grid_index.size} grid points"
    )
    return fp_config


def _create_neural_lam_config(
    artifact, configs, fp_interior_config, fp_boundary_config, workdir
):
    """
    The training neural-lam config with the inference datastores, and the
    boundary normalised with the training (ERA5) boundary statistics: the
    stats datastore next to the training boundary config in the artifact.
    neural-lam joins `overload_stats_path` onto the config directory, so it
    is absolute.
    """
    nl_config = inference_neural_lam_config(
        yaml.safe_load(
            (artifact / "configs" / configs.neural_lam_config).read_text()
        ),
        interior_config=fp_interior_config.name,
        boundary_config=fp_boundary_config.name,
        stats_config=(
            artifact / "configs" / configs.boundary_datastore
        ).resolve(),
    )
    fp_config = workdir / "config.yaml"
    fp_config.write_text(yaml.safe_dump(nl_config, sort_keys=False))
    logger.info(f"neural-lam config written to {fp_config}")
    return fp_config


def create_inference_datasets(
    artifact,
    interior_dir,
    boundary,
    boundary_source,
    analysis_time,
    forecast_duration,
    workdir,
):
    artifact, interior_dir, workdir = (
        Path(artifact),
        Path(interior_dir),
        Path(workdir),
    )
    workdir.mkdir(parents=True, exist_ok=True)
    configs = training_configs(artifact / "configs")
    logger.info(
        f"training configs: {configs}; {boundary_source} boundary: "
        f"{boundary_config_path(boundary_source)}"
    )
    # the initial states are at analysis_time and +3h, so the first prediction
    # is at +6h; see MIN_FORECAST_DURATION for the minimum
    if (
        forecast_duration < MIN_FORECAST_DURATION
        or forecast_duration % INTERIOR_STEP
    ):
        raise ValueError(
            f"forecast duration must be a multiple of {INTERIOR_STEP} and at "
            f"least {MIN_FORECAST_DURATION}, got {forecast_duration}"
        )
    interior_end = (
        analysis_time
        + forecast_duration
        + INTERIOR_EXTRA_STEPS * INTERIOR_STEP
    )

    fp_interior_config = _create_interior_datastore(
        artifact, configs, interior_dir, analysis_time, interior_end, workdir
    )
    fp_boundary_config = _create_boundary_datastore(
        boundary_source,
        boundary,
        fp_interior_config,
        analysis_time,
        forecast_duration,
        workdir,
    )
    logger.info(
        f"forecast {analysis_time} + {forecast_duration}: run neural-lam with "
        f"--ar_steps_eval {ar_steps_for(forecast_duration)} --eval_init_times "
        "(empty: no init-time filtering)"
    )
    return _create_neural_lam_config(
        artifact,
        configs,
        fp_interior_config,
        fp_boundary_config,
        workdir,
    )


def _parse_analysis_time(s):
    t = isodate.parse_datetime(s)
    if t.tzinfo is not None:
        t = t.astimezone(datetime.timezone.utc).replace(tzinfo=None)
    return t


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--artifact", type=Path, default=Path("inference_artifact")
    )
    parser.add_argument(
        "--interior-dir",
        required=True,
        type=Path,
        help="regrid_dini.py output",
    )
    parser.add_argument(
        "--boundary",
        required=True,
        help="boundary forecast zarr (IFS contract)",
    )
    parser.add_argument(
        "--boundary-source", choices=list(BOUNDARY_SOURCES), default="dini"
    )
    parser.add_argument(
        "--analysis-time",
        required=True,
        type=_parse_analysis_time,
        help="ISO8601, UTC if no timezone",
    )
    parser.add_argument(
        "--forecast-duration",
        required=True,
        type=isodate.parse_duration,
        help="ISO8601 duration, e.g. PT18H",
    )
    parser.add_argument("--workdir", required=True, type=Path)
    args = parser.parse_args()

    create_inference_datasets(
        artifact=args.artifact,
        interior_dir=args.interior_dir,
        boundary=args.boundary,
        boundary_source=args.boundary_source,
        analysis_time=args.analysis_time,
        forecast_duration=args.forecast_duration,
        workdir=args.workdir,
    )


if __name__ == "__main__":
    if os.getenv("ANNA_DEBUGGER", "") == "ipdb":
        import ipdb

        with ipdb.launch_ipdb_on_exception():
            main()
    else:
        main()
