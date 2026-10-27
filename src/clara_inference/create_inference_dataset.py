"""
Create the inference datastores and neural-lam config for running CLARA
from regridded DINI (interior) and an IFS forecast (boundary).

Inputs:
- the inference artifact (`clara_inference.artifact`), which describes the
  model as trained: its neural-lam config (named in `configs/model.yaml`),
  the interior and boundary datastore configs, the stats datastores
  (training statistics), the boundary grid with its statics, and the graph
- the regridded interior (`interior_{single,pressure}_levels.zarr` from
  `regrid_dini.py`)
- the IFS boundary forecast following the contract in `ifs_boundary.py`

Written to the inference workdir (neural-lam opens `{name}.zarr` next to each
`{name}.yaml`, and the graph in `graphs/` next to the interior datastore):
- `danra_model1_config_ifs_finetune.{yaml,zarr}`: interior datastore covering
  analysis_time .. analysis_time + forecast_duration (+ extra steps, see
  `regrid_dini.INTERIOR_EXTRA_STEPS`)
- `overlapping_ifs_all2_400km_model1_config.input.zarr`: the selected IFS
  cycle on the boundary grid in training order, with the ERA5 statics
- `overlapping_ifs_all2_400km_model1_config.{yaml,zarr}`: boundary datastore,
  cropped to the 400 km margin around the interior; its points are checked
  against the graph's boundary nodes
- `graphs/<graph name>`: a link to the graph in the artifact
- `config.yaml`: neural-lam config, the training config with the inference
  datastores, normalised with the training statistics in the artifact
  (`overload_stats_path`, see `model_configs.py`)

Statistics are never computed from the inference data.

NB: mllam-data-prep configs must never be written with sorted keys, that
would reorder inputs/variables and so the features the checkpoint expects.

Usage:
    python -m clara_inference.create_inference_dataset \\
        --artifact inference_artifact \\
        --interior-dir inference_workdir/inputs \\
        --ifs-boundary ifs.zarr \\
        --analysis-time 2026-10-26T06:00 --forecast-duration PT18H \\
        --workdir inference_workdir/datastores
"""
import argparse
import datetime
import os
from pathlib import Path

import isodate
import mllam_data_prep as mdp
import mllam_data_prep.config as mdp_config
import xarray as xr
import yaml
from loguru import logger

from clara_inference import ifs_boundary
from clara_inference.convert_output import projection_crs
from clara_inference.model_configs import (
    inference_neural_lam_config,
    training_configs,
)

INTERIOR_STEP = datetime.timedelta(hours=3)
# see regrid_dini.INTERIOR_EXTRA_STEPS
INTERIOR_EXTRA_STEPS = 2
# shortest forecast: neural-lam's ARModel (when loading the checkpoint) builds
# a WeatherDataset per split with its default `ar_steps=3`, which needs
# 2 + 3 + 1 + 1 interior time steps (see INTERIOR_EXTRA_STEPS), i.e. a
# forecast of at least 12h. Shorter forecasts fail in train_model
MIN_FORECAST_DURATION = datetime.timedelta(hours=12)

SPLIT_NAMES = ["train", "val", "test"]
INTERIOR_INPUT_FILES = dict(
    danra_sl_state="interior_single_levels.zarr",
    danra_pl_state="interior_pressure_levels.zarr",
    danra_static="interior_single_levels.zarr",
    danra_forcing="interior_single_levels.zarr",
)
BOUNDARY_STATIC_INPUT = "era_static"


def ar_steps_for(forecast_duration):
    """
    Autoregressive steps for a forecast: the initial states are at the
    analysis time and +3h, the predictions at +6h .. +forecast_duration.
    """
    return int(forecast_duration / INTERIOR_STEP) - 1


def _write_config(config, fp):
    # NB: sort_keys=False, sorting would change the feature order
    config.to_yaml_file(fp, sort_keys=False)


def _splits(start, end):
    """train/val/test splits (neural-lam needs all three), no statistics."""
    return {
        name: mdp_config.Split(start=start, end=end) for name in SPLIT_NAMES
    }


def _check_no_missing(ds, variables, what):
    n_missing = {
        v: int(ds[v].isnull().sum()) for v in variables if v in ds.data_vars
    }
    if any(n_missing.values()):
        raise ValueError(f"missing values in the {what}: {n_missing}")


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
    # neural-lam requires train/val/test splits, the forecast is run on
    # "test". Statistics come from the training stats datastore
    config.output.splitting.splits = _splits(start, end)

    fp_config = workdir / f"{name}.yaml"
    _write_config(config, fp_config)
    ds = mdp.create_dataset(config=config)
    _check_no_missing(ds, ["state", "forcing", "static"], "interior datastore")
    ds.to_zarr(workdir / f"{name}.zarr", mode="w", consolidated=True)
    logger.info(
        f"interior datastore: {ds.time.size} time steps {start} .. {end}, "
        f"{ds.grid_index.size} grid points"
    )
    return fp_config


def _create_boundary_datastore(
    artifact,
    configs,
    fp_ifs,
    fp_interior_config,
    analysis_time,
    forecast_duration,
    workdir,
):
    name = configs.name(configs.boundary_datastore)
    ds_grid = xr.open_zarr(
        artifact / "grids" / f"{name}.grid.zarr", decode_timedelta=True
    ).load()
    ds_input = ifs_boundary.prepare_boundary_input(
        xr.open_zarr(fp_ifs, decode_timedelta=True),
        ds_grid,
        analysis_time,
        forecast_duration,
    )
    fp_input = workdir / f"{name}.input.zarr"
    ds_input.to_zarr(fp_input, mode="w", consolidated=True)

    config = mdp.Config.from_yaml_file(
        artifact / "configs" / configs.boundary_datastore
    )
    # the IFS fields and the statics are on the same grid, in one zarr
    for input_config in config.inputs.values():
        input_config.path = str(fp_input.resolve())
    cycle = ds_input.time.values[0]
    start = str(cycle)[:16]
    config.output.coord_ranges["analysis_time"].start = start
    config.output.coord_ranges["analysis_time"].end = start
    config.output.splitting.splits = _splits(start, start)
    # crop around the inference interior dataset (same grid as in training);
    # mllam-data-prep resolves the path relative to the CWD, so absolute
    config.output.domain_cropping.interior_dataset_config_path = str(
        fp_interior_config.resolve()
    )

    fp_config = workdir / f"{name}.yaml"
    _write_config(config, fp_config)
    ds = mdp.create_dataset(config=config)
    _check_no_missing(ds, ["forcing", "static"], "boundary datastore")

    ifs_boundary.check_point_order(
        ds,
        artifact / "graphs" / _graph_name(artifact),
        projection_crs(fp_interior_config),
    )
    ds.attrs.update(ds_input.attrs)
    ds.to_zarr(workdir / f"{name}.zarr", mode="w", consolidated=True)
    logger.info(
        f"boundary datastore: cycle {start}, "
        f"{ds.elapsed_forecast_duration.size} lead times, "
        f"{ds.grid_index.size} grid points"
    )
    return fp_config


def _graph_name(artifact):
    model = yaml.safe_load((artifact / "configs" / "model.yaml").read_text())
    return model["graph"]["name"]


def _link_graph(artifact, workdir):
    """neural-lam loads the graph from graphs/ next to the interior config."""
    name = _graph_name(artifact)
    link = workdir / "graphs" / name
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to((artifact / "graphs" / name).resolve())
    logger.info(f"graph {name}: {link} -> {link.resolve()}")


def _create_neural_lam_config(
    artifact, configs, fp_interior_config, fp_boundary_config, workdir
):
    configs_dir = (artifact / "configs").resolve()
    nl_config = inference_neural_lam_config(
        yaml.safe_load((configs_dir / configs.neural_lam_config).read_text()),
        interior_config=fp_interior_config.name,
        boundary_config=fp_boundary_config.name,
        interior_stats_config=configs_dir / configs.interior_stats_datastore,
        boundary_stats_config=configs_dir / configs.boundary_stats_datastore,
    )
    fp_config = workdir / "config.yaml"
    fp_config.write_text(yaml.safe_dump(nl_config, sort_keys=False))
    logger.info(f"neural-lam config written to {fp_config}")
    return fp_config


def create_inference_datasets(
    artifact,
    interior_dir,
    ifs_boundary_path,
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
    logger.info(f"training configs: {configs}")
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
        artifact,
        configs,
        ifs_boundary_path,
        fp_interior_config,
        analysis_time,
        forecast_duration,
        workdir,
    )
    _link_graph(artifact, workdir)
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
        "--ifs-boundary",
        required=True,
        help="IFS forecast zarr (contract in ifs_boundary.py)",
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
        ifs_boundary_path=args.ifs_boundary,
        analysis_time=args.analysis_time,
        forecast_duration=args.forecast_duration,
        workdir=args.workdir,
    )


if __name__ == "__main__":
    if os.getenv("CLARA_DEBUGGER", "") == "ipdb":
        import ipdb

        with ipdb.launch_ipdb_on_exception():
            main()
    else:
        main()
