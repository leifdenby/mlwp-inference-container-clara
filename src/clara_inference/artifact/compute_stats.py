"""
Compute the training statistics of CLARA's statistics datastores and write
them as the stats-only zarrs neural-lam reads for `overload_stats_path`
(`configs/<name>.zarr` next to `configs/<name>.yaml` in the artifact).

The model's normalisation statistics are not stored in the checkpoint
(neural-lam registers them as non-persistent buffers), and the statistics
computed in training only existed on the training cluster. They are
recomputed here from the public sources the configs reference, over the
training split of each config:

- interior: `danra_model1_config` (DANRA v0.5.0 from the ECMWF object store,
  2016-01-01 .. 2019-10-22, 3-hourly): state, forcing and static statistics
- boundary: `overlapping_era_400km_model1_config` (WeatherBench2 ERA5,
  2016-01-01 .. 2019-10-22, 6-hourly, the 400 km margin around DANRA,
  interior points included): forcing and static statistics

## Exactly as mllam-data-prep computes them

mllam-data-prep (`ops.statistics.calc_stats`, the version CLARA was trained
with) computes, per feature, over the grid points and times of the split:
`mean` and `std` (ddof=0) of the values, `diff_mean` of the first time
differences, and - because `calc_stats` replaces the dataset by its
differences for each `diff_*` op in turn - `diff_std` of the **second** time
differences. (Checked against a datastore made by mllam-data-prep: equal to
~1e-15.) Statics only get `mean` and `std`.

Building the whole training period with mllam-data-prep and reducing it
with dask is far too slow (~35 min for one day), so the variables are
extracted with mllam-data-prep block by block in time (blocks overlap by
two time steps, for the differences), and the statistics of the blocks are
combined exactly (Chan et al.'s parallel mean/variance), in float64.

This reads ~1 TB per datastore, so it is run once, on a machine with good
bandwidth; `--start`/`--end` restrict the period (e.g. for a quick,
approximate run). The period is recorded in the zarr's attributes.

`--from-processed NAME=PATH` instead takes the statistics of a stats
datastore from an existing processed datastore zarr with the same features
(e.g. one made in training, or a small local one for testing).

Usage:
    python -m clara_inference.artifact.compute_stats \\
        --configs artifact/configs --output-dir artifact/build/stats
"""
import argparse
import copy
import datetime
import tempfile
import time
from pathlib import Path

import isodate
import mllam_data_prep as mdp
import mllam_data_prep.config as mdp_config
import numpy as np
import xarray as xr
from loguru import logger

from clara_inference import model_configs as mc

STATS_MARKER = "__train__"
SPLIT_NAMES = ["train", "val", "test"]


def stats_only(ds):
    """The training statistics and splits of a processed datastore."""
    keep = [v for v in ds.data_vars if STATS_MARKER in v] + ["splits"]
    ds = ds[keep]
    # coords along other dims (e.g. lat/lon per grid_index) aren't needed
    drop = [c for c in ds.coords if not set(ds[c].dims) <= set(ds.dims)]
    return ds.drop_vars(drop)


class RunningStats:
    """
    Per-feature count, mean and sum of squared deviations (M2) over all
    values added, combined exactly between batches (Chan et al. 1979).
    """

    def __init__(self, n_features):
        self.n = 0
        self.mean = np.zeros(n_features)
        self.m2 = np.zeros(n_features)

    def add(self, values):
        """`values`: (..., n_features); all but the last axis reduced."""
        x = values.reshape(-1, values.shape[-1]).astype("f8")
        n_b = x.shape[0]
        if n_b == 0:
            return
        mean_b = x.mean(axis=0)
        m2_b = ((x - mean_b) ** 2).sum(axis=0)
        n = self.n + n_b
        delta = mean_b - self.mean
        self.mean = self.mean + delta * n_b / n
        self.m2 = self.m2 + m2_b + delta**2 * self.n * n_b / n
        self.n = n

    @property
    def std(self):
        return np.sqrt(self.m2 / self.n)


class SplitStats:
    """
    mllam-data-prep's statistics of a (time, grid_index, feature) variable,
    from consecutive blocks of times: mean/std of the values, mean of the
    first and std of the second time differences (see module docstring).
    """

    def __init__(self, n_features):
        self.values = RunningStats(n_features)
        self.diff1 = RunningStats(n_features)
        self.diff2 = RunningStats(n_features)
        self.last_time = None

    def add_block(self, times, x):
        """
        `x`: (time, grid_index, feature) at `times`; blocks must overlap the
        previous one by (at least) two times, which are not counted twice.
        """
        if self.last_time is None:
            n_old = 0
        else:
            n_old = int((times <= self.last_time).sum())
            if n_old < 2:
                raise ValueError("blocks must overlap by two time steps")
        self.values.add(x[n_old:])
        d1 = np.diff(x, axis=0)
        # differences ending at a new time
        self.diff1.add(d1[max(n_old - 1, 0):])
        d2 = np.diff(d1, axis=0)
        self.diff2.add(d2[max(n_old - 2, 0):])
        self.last_time = times[-1]

    def result(self):
        return dict(
            mean=self.values.mean,
            std=self.values.std,
            diff_mean=self.diff1.mean,
            diff_std=self.diff2.std,
        )


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
        for name in SPLIT_NAMES
    }
    fp = Path(tmpdir) / "interior_reference.yaml"
    config.to_yaml_file(fp, sort_keys=False)
    return fp


def _block_config(config, start, end):
    """`config` restricted to start..end, without statistics."""
    config = copy.deepcopy(config)
    dim = config.output.splitting.dim
    config.output.coord_ranges[dim].start = start.isoformat()
    config.output.coord_ranges[dim].end = end.isoformat()
    config.output.splitting.splits = {
        name: mdp_config.Split(start=start.isoformat(), end=end.isoformat())
        for name in SPLIT_NAMES
    }
    return config


def compute_stats(
    fp_config, start=None, end=None, interior_config=None, block_days=4
):
    """
    The statistics (and splits) of the datastore of `fp_config` over its
    training split (optionally restricted to start..end), as
    mllam-data-prep computes them.
    """
    config = mdp.Config.from_yaml_file(fp_config)
    dim = config.output.splitting.dim
    split = config.output.splitting.splits["train"]
    start = datetime.datetime.fromisoformat(str(start or split.start))
    end = datetime.datetime.fromisoformat(str(end or split.end))
    step = config.output.coord_ranges[dim].step
    step = isodate.parse_duration(step) if isinstance(step, str) else step
    block = datetime.timedelta(days=block_days)

    stats = {}
    static = None
    with tempfile.TemporaryDirectory() as tmpdir:
        if config.output.domain_cropping is not None:
            config.output.domain_cropping.interior_dataset_config_path = str(
                _light_interior_config(interior_config, tmpdir)
            )
        block_start = start
        t0 = time.perf_counter()
        while block_start < end:
            block_end = min(block_start + block, end)
            ds = mdp.create_dataset(
                config=_block_config(config, block_start, block_end)
            )
            times = ds[dim].values
            for category in ["state", "forcing"]:
                if category not in ds:
                    continue
                x = ds[category].transpose(dim, "grid_index", ...).values
                if category not in stats:
                    stats[category] = (
                        SplitStats(x.shape[-1]),
                        ds[f"{category}_feature"],
                    )
                stats[category][0].add_block(times, x)
            if static is None and "static" in ds:
                static = ds["static"].load()
            elapsed = time.perf_counter() - t0
            done = (block_end - start) / (end - start)
            logger.info(
                f"{fp_config.name}: {block_start:%Y-%m-%d} .. "
                f"{block_end:%Y-%m-%d} done ({100 * done:.1f}%, "
                f"{elapsed / 60:.0f} min, ~{elapsed / done / 60 - elapsed / 60:.0f}"
                " min left)"
            )
            # the next block overlaps this one by two time steps
            block_start = block_end - 2 * step
            if block_end >= end:
                break

    data_vars = {}
    for category, (split_stats, features) in stats.items():
        for op, values in split_stats.result().items():
            data_vars[f"{category}{STATS_MARKER}{op}"] = xr.DataArray(
                values, dims=features.dims, coords=features.coords
            )
    if static is not None:
        x = static.transpose("grid_index", ...).values.astype("f8")
        for op, values in [("mean", x.mean(axis=0)), ("std", x.std(axis=0))]:
            data_vars[f"static{STATS_MARKER}{op}"] = xr.DataArray(
                values,
                dims=["static_feature"],
                coords={"static_feature": static.static_feature},
            )
    period = [start.isoformat(), end.isoformat()]
    data_vars["splits"] = xr.DataArray(
        [period] * len(SPLIT_NAMES),
        dims=["split_name", "split_part"],
        coords=dict(split_name=SPLIT_NAMES, split_part=["start", "end"]),
    )
    ds = xr.Dataset(data_vars)
    ds.attrs.update(
        statistics_period=f"{period[0]} .. {period[1]}",
        statistics_step=isodate.duration_isoformat(step),
        description=(
            "training statistics recomputed with "
            "clara_inference.artifact.compute_stats (as mllam-data-prep)"
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
        "--block-days",
        type=float,
        default=4,
        help="days per block read with mllam-data-prep (default: "
        "%(default)s)",
    )
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
                block_days=args.block_days,
            )
        fp = args.output_dir / f"{name}.zarr"
        ds.to_zarr(fp, mode="w", consolidated=True)
        logger.info(f"wrote {fp}: {list(ds.data_vars)}")


if __name__ == "__main__":
    main()
