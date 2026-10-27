"""
Compute the training statistics of CLARA's statistics datastores and write
them as the stats-only zarrs neural-lam reads for `overload_stats_path`
(`configs/<name>.zarr` next to `configs/<name>.yaml` in the artifact).

The model's normalisation statistics are not stored in the checkpoint
(neural-lam registers them as non-persistent buffers), and the statistics
computed in training only existed on the training cluster. They are
recomputed here with mllam-data-prep from the public sources the configs
reference, over the training split of each config:

- interior: `danra_model1_config` (DANRA v0.5.0 from the ECMWF object store,
  2016-01-01 .. 2019-10-22, 3-hourly): state, forcing and static statistics
- boundary: `overlapping_era_400km_model1_config` (WeatherBench2 ERA5,
  2016-01-01 .. 2019-10-22, 6-hourly, the 400 km margin around DANRA,
  interior points included): forcing and static statistics

This reads a lot of data (the DANRA state alone is ~11,400 times x 55
features x 465k points), so it is run once, on a machine with good
bandwidth. `--start`/`--end` restrict the period (e.g. for a quick,
approximate run); the period used is recorded in the zarr's attributes.
Only the period changes: the time step stays as in the config, since the
`diff_*` statistics are of one-step differences.

`--from-processed NAME=PATH` instead takes the statistics of a stats
datastore from an existing processed datastore zarr with the same features
(e.g. one made in training, or a small local one for testing).

Usage:
    python -m clara_inference.artifact.compute_stats \\
        --configs artifact/configs --output-dir artifact/build/stats
"""
import argparse
import datetime
import tempfile
from pathlib import Path

import dask.diagnostics
import mllam_data_prep as mdp
import mllam_data_prep.config as mdp_config
import xarray as xr
from loguru import logger

from clara_inference import model_configs as mc

STATS_MARKER = "__train__"


def stats_only(ds):
    """The training statistics and splits of a processed datastore."""
    keep = [v for v in ds.data_vars if STATS_MARKER in v] + ["splits"]
    ds = ds[keep]
    # coords along other dims (e.g. lat/lon per grid_index) aren't needed
    drop = [c for c in ds.coords if not set(ds[c].dims) <= set(ds.dims)]
    return ds.drop_vars(drop)


def _light_interior_config(fp_interior_config, tmpdir):
    """
    A copy of the interior config covering one day, without statistics: the
    cropping reference for the boundary (mllam-data-prep builds the whole
    interior dataset just for its grid).
    """
    config = mdp.Config.from_yaml_file(fp_interior_config)
    start = config.output.splitting.splits["train"].start
    end = (
        datetime.datetime.fromisoformat(str(start)) + datetime.timedelta(days=1)
    ).isoformat()
    config.output.coord_ranges["time"].start = str(start)
    config.output.coord_ranges["time"].end = end
    config.output.splitting.splits = {
        name: mdp_config.Split(start=str(start), end=end)
        for name in ["train", "val", "test"]
    }
    fp = Path(tmpdir) / "interior_reference.yaml"
    config.to_yaml_file(fp, sort_keys=False)
    return fp


def compute_stats(fp_config, start=None, end=None, interior_config=None):
    """
    The statistics (and splits) of the datastore of `fp_config` over its
    training split, optionally restricted to start..end.
    """
    config = mdp.Config.from_yaml_file(fp_config)
    split = config.output.splitting.splits["train"]
    split_dim = config.output.splitting.dim
    if start is not None:
        split.start = start
    if end is not None:
        split.end = end
    coord_range = config.output.coord_ranges[split_dim]
    coord_range.start, coord_range.end = split.start, split.end
    for name in ["val", "test"]:
        config.output.splitting.splits[name] = mdp_config.Split(
            start=split.start, end=split.end
        )
    with tempfile.TemporaryDirectory() as tmpdir:
        if config.output.domain_cropping is not None:
            config.output.domain_cropping.interior_dataset_config_path = str(
                _light_interior_config(interior_config, tmpdir)
            )
        logger.info(
            f"computing statistics of {fp_config} over "
            f"{split.start} .. {split.end}"
        )
        ds = stats_only(mdp.create_dataset(config=config))
        with dask.diagnostics.ProgressBar(dt=30):
            ds = ds.compute()
    ds.attrs.update(
        statistics_period=f"{split.start} .. {split.end}",
        statistics_step=str(coord_range.step),
        description=(
            "training statistics recomputed with "
            "clara_inference.artifact.compute_stats"
        ),
    )
    return ds


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--configs",
        type=Path,
        default=Path("artifact/configs"),
        help="the training configs (with model.yaml)",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--only",
        choices=["interior", "boundary"],
        help="only this statistics datastore",
    )
    parser.add_argument("--start", help="restrict the period (ISO8601)")
    parser.add_argument("--end", help="restrict the period (ISO8601)")
    parser.add_argument(
        "--from-processed",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="take the statistics of stats datastore NAME from the "
        "processed datastore zarr PATH (repeatable)",
    )
    args = parser.parse_args()

    configs = mc.training_configs(args.configs)
    stats_configs = dict(
        interior=configs.interior_stats_datastore,
        boundary=configs.boundary_stats_datastore,
    )
    from_processed = dict(s.split("=", 1) for s in args.from_processed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for which, fn in stats_configs.items():
        if args.only and which != args.only:
            continue
        name = mc.TrainingConfigs.name(fn)
        if name in from_processed:
            ds = stats_only(xr.open_zarr(from_processed[name])).load()
            ds.attrs = dict(
                description=f"statistics taken from {from_processed[name]}",
            )
        else:
            ds = compute_stats(
                args.configs / fn,
                start=args.start,
                end=args.end,
                interior_config=args.configs
                / configs.interior_stats_datastore,
            )
        fp = args.output_dir / f"{name}.zarr"
        ds.to_zarr(fp, mode="w", consolidated=True)
        logger.info(f"wrote {fp}: {list(ds.data_vars)}")


if __name__ == "__main__":
    main()
