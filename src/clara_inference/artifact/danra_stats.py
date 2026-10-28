"""
Recompute CLARA's interior (DANRA) training statistics
(`danra_model1_config`: state, forcing and static) from DANRA v0.5.0 on the
ECMWF object store, resumably, on a machine with good bandwidth.

CLARA's split is 2016-01-01 .. 2019-10-22 (3-hourly). The statistics are
those of mllam-data-prep (`compute_stats` explains them: `diff_mean` of the
first and `diff_std` of the second time differences) and are computed the
same way as there: the variables are extracted with mllam-data-prep for a
few time steps at a time and the statistics combined exactly in float64.

Reading and resuming:
- DANRA is chunked in time by 256 steps (32 days), so the split is processed
  in blocks aligned to the time chunks: every chunk is downloaded once,
  into a temporary local file cache (`--cache-dir`, ~12 GB for one block)
  that is deleted when the block is done. Within a block mllam-data-prep
  extracts `--sub-block-steps` steps at a time. Each extraction decompresses
  whole time chunks, so larger sub-blocks are faster (memory: ~2 GB per 8
  steps; the default of 32 needs ~10 GB)
- each block's accumulators are written (atomically) to `--partials-dir`,
  with its last two time steps, which the next block needs for the first
  and second differences across the block boundary (so it doesn't have to
  download the previous chunk again). Re-running the same command skips
  finished blocks
- extracting is retried with backoff (`--max-retries`) on network errors

About 540 GB is read for the whole split. The output has the layout of
`compute_stats`' output (`{config name}.zarr`, for `assemble --stats-dir`).

`--dry-run [STEPS]` processes only the first few time steps of the split, to
check the setup before the full run, and logs what the full run would
read. As the source is chunked by 32 days, it still downloads one time chunk
of every variable.

Usage:
    uv run python -m clara_inference.artifact.danra_stats \\
        --output stats/danra_model1_config.zarr \\
        --partials-dir partials-danra --cache-dir danra-cache
"""
import argparse
import datetime
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

import mllam_data_prep as mdp
import numpy as np
import pandas as pd
import xarray as xr
from fsspec.implementations.cached import WholeFileCacheFileSystem
from loguru import logger
from tqdm import tqdm

from clara_inference.artifact.compute_stats import (
    SplitStats,
    _block_config,
    _step,
    _use_cache,
    stats_dataset,
)

DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[3]
    / "artifact"
    / "configs"
    / "danra_model1_config.yaml"
)
CATEGORIES = ["state", "forcing"]


def chunk_blocks(source_times, chunk_steps, start, end, block_chunks=1):
    """
    The (first, last) times of the blocks of start..end, each within
    `block_chunks` time chunks of the source (of `chunk_steps` steps from
    the first source time).
    """
    i_start = int(np.searchsorted(source_times, start))
    i_end = int(np.searchsorted(source_times, end, side="right")) - 1
    if source_times[i_start] != start or source_times[i_end] != end:
        raise ValueError(f"{start} or {end} is not a source time")
    size = chunk_steps * block_chunks
    blocks = []
    i0 = i_start
    while i0 <= i_end:
        i1 = min((i0 // size + 1) * size - 1, i_end)
        blocks.append((source_times[i0], source_times[i1]))
        i0 = i1 + 1
    return blocks


def _time_chunking(config):
    """
    Source times, time chunk size (the same for all time inputs) and the
    (uncompressed) bytes of the chunks of one time chunk of the variables
    used: what is downloaded per block.
    """
    times, chunk_steps, nbytes = None, set(), 0
    for input_config in config.inputs.values():
        if "time" not in input_config.dims:
            continue
        if not input_config.variables:
            continue  # derived variables only
        ds = xr.open_zarr(str(input_config.path))
        variables = input_config.variables
        for var in variables:
            da = ds[var]
            chunks = dict(zip(da.dims, da.encoding["chunks"]))
            chunk_steps.add(chunks["time"])
            size = da.isel(time=slice(0, chunks["time"])).nbytes
            if isinstance(variables, dict):
                # selected levels, if the levels are chunked separately
                ((coord, selection),) = variables[var].items()
                if chunks.get(coord) == 1:
                    size = size * len(selection.values) // da.sizes[coord]
            nbytes += size
        if times is None:
            times = ds.time.values
        elif not np.array_equal(times, ds.time.values):
            raise ValueError("the time inputs have different times")
    if len(chunk_steps) != 1:
        raise ValueError(
            f"the time inputs have different chunks {chunk_steps}"
        )
    return times, chunk_steps.pop(), nbytes


def _extract(config, start, end, max_retries):
    """
    The state and forcing values ((time, grid_index, feature) float64) of
    start..end, with their times and feature coordinates, extracted with
    mllam-data-prep, retrying with backoff on errors.
    """
    for attempt in range(max_retries + 1):
        try:
            ds = mdp.create_dataset(
                config=_block_config(config, _py(start), _py(end))
            )
            dim = config.output.splitting.dim
            return {
                category: (
                    ds[dim].values,
                    ds[category].transpose(dim, "grid_index", ...).values,
                    ds[f"{category}_feature"].load(),
                )
                for category in CATEGORIES
                if category in ds
            }
        except Exception as ex:  # network errors come in many types
            if attempt == max_retries:
                raise
            wait = min(30 * 2**attempt, 600)
            logger.warning(
                f"extracting {start}..{end} failed ({type(ex).__name__}: "
                f"{ex}), retry {attempt + 1}/{max_retries} in {wait}s"
            )
            time.sleep(wait)


def _py(t):
    """A numpy datetime64 as datetime.datetime."""
    return pd.Timestamp(t).to_pydatetime()


def _save_atomic(fp, **arrays):
    fp_tmp = fp.with_name(fp.name + ".tmp")
    with open(fp_tmp, "wb") as fh:
        np.savez(fh, **arrays)
    os.replace(fp_tmp, fp)


def _process_block(
    config, start, end, step, sub_block_steps, tail, split_start, max_retries
):
    """
    The SplitStats and feature coordinate per category of the times
    start..end, and the block's last two time steps (`tail`, as the
    previous block's: category -> (times, values), with fewer times only at
    the start of the split). Without the previous block's tail the block
    starts (up to) two time steps earlier, but not before `split_start`.
    """
    stats, features, carry = {}, {}, tail
    sub_start = (
        start if tail is not None else max(start - 2 * step, split_start)
    )
    while sub_start <= end:
        # (the first extraction includes the steps read again)
        sub_end = min(
            max(sub_start + (sub_block_steps - 1) * step, start), end
        )
        extracted = _extract(config, sub_start, sub_end, max_retries)
        new_carry = {}
        for category, (times, x, feature) in extracted.items():
            if category not in stats:
                if carry is not None:
                    n_before = len(carry[category][0])
                    counted = carry[category][0][-1]
                else:
                    # the earlier steps read again (none for the first block)
                    n_before = int((times < start).sum())
                    counted = times[n_before - 1] if n_before else None
                stats[category] = SplitStats(
                    x.shape[-1], counted_until=counted, n_before=n_before
                )
                features[category] = feature
            if carry is not None:
                times = np.concatenate([carry[category][0], times])
                x = np.concatenate([carry[category][1], x])
            stats[category].add_block(times, x)
            new_carry[category] = (times[-2:], x[-2:])
        carry = new_carry
        sub_start = sub_end + step
    return stats, features, carry


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--partials-dir",
        required=True,
        type=Path,
        help="per-block accumulators (and the last block's two last times)",
    )
    parser.add_argument(
        "--cache-dir",
        required=True,
        type=Path,
        help="temporary local file cache of the current block's chunks",
    )
    parser.add_argument(
        "--keep-cache",
        action="store_true",
        help="use --cache-dir itself as a persistent cache, never deleted "
        "(for tests, or with disk space for the whole period: ~540 GB)",
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG, type=Path)
    parser.add_argument("--start", help="split start (default: from config)")
    parser.add_argument("--end", help="split end (default: from config)")
    parser.add_argument(
        "--sub-block-steps",
        type=int,
        default=32,
        help="time steps extracted at a time (default: %(default)s)",
    )
    parser.add_argument(
        "--chunk-steps",
        type=int,
        help="block size in time steps (default: the source's time chunks; "
        "smaller only for testing, as chunks are then downloaded repeatedly)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=8,
        help="retries (with backoff, up to 10 min) when extracting fails",
    )
    parser.add_argument(
        "--dry-run",
        type=int,
        nargs="?",
        const=6,
        metavar="STEPS",
        help="only the first STEPS (default 6) time steps of the split, to "
        "check the setup; logs what the full run would read. NB: still "
        "downloads one time chunk of every variable (the first block). Use "
        "another --partials-dir than for the full run",
    )
    args = parser.parse_args()

    # log through tqdm so log messages don't break the progress bar, and
    # silence mllam-data-prep's per-variable info messages
    logger.remove()
    logger.add(
        lambda msg: tqdm.write(msg, end=""),
        filter={"": "INFO", "mllam_data_prep": "WARNING"},
    )

    config = mdp.Config.from_yaml_file(args.config)
    split = config.output.splitting.splits["train"]
    if list(split.compute_statistics.ops) != [
        "mean",
        "std",
        "diff_mean",
        "diff_std",
    ]:
        raise NotImplementedError(
            f"unsupported statistics ops {split.compute_statistics.ops}"
        )
    start = pd.Timestamp(args.start or split.start).to_datetime64()
    end = pd.Timestamp(args.end or split.end).to_datetime64()
    step = _step(config)

    source_times, chunk_steps, chunk_bytes = _time_chunking(config)
    if args.chunk_steps:
        chunk_steps = args.chunk_steps
    blocks = chunk_blocks(source_times, chunk_steps, start, end)
    logger.info(
        f"split {start} .. {end}: {len(blocks)} blocks within time chunks of "
        f"{chunk_steps} steps, ~{chunk_bytes / 1e9:.0f} GB to read per block, "
        f"~{len(blocks) * chunk_bytes / 1e12:.2f} TB in total (uncompressed)"
    )
    if args.dry_run:
        end = start + (args.dry_run - 1) * pd.Timedelta(step).to_timedelta64()
        blocks = chunk_blocks(source_times, chunk_steps, start, end)
        logger.info(
            f"dry run: only {start} .. {end} ({args.dry_run} steps, "
            f"{len(blocks)} block(s))"
        )

    args.partials_dir.mkdir(parents=True, exist_ok=True)
    run_info = dict(
        config=str(args.config.resolve()),
        start=str(start),
        end=str(end),
        chunk_steps=chunk_steps,
        sub_block_steps=args.sub_block_steps,
    )
    fp_run_info = args.partials_dir / "run_info.json"
    if fp_run_info.exists():
        if json.loads(fp_run_info.read_text()) != run_info:
            raise ValueError(
                f"{args.partials_dir} contains partials from a run with other "
                "arguments, use a new --partials-dir"
            )
    else:
        fp_run_info.write_text(json.dumps(run_info, indent=2))

    def _fp(kind, block):
        first, last = (pd.Timestamp(t).strftime("%Y%m%dT%H") for t in block)
        return args.partials_dir / f"{kind}_{first}_{last}.npz"

    def _load_tail(block):
        fp = _fp("tail", block)
        if not fp.exists():
            return None
        arrays = np.load(fp)
        return {
            c: (arrays[f"{c}_times"], arrays[f"{c}_values"])
            for c in CATEGORIES
            if f"{c}_times" in arrays
        }

    todo = [b for b in blocks if not _fp("block", b).exists()]
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    with tqdm(
        total=len(blocks),
        initial=len(blocks) - len(todo),
        unit="block",
        dynamic_ncols=True,
    ) as pbar:
        for block in todo:
            k = blocks.index(block)
            pbar.set_description(f"{str(block[0])[:13]}..{str(block[1])[:13]}")
            tail = _load_tail(blocks[k - 1]) if k > 0 else None
            if args.keep_cache:
                block_cache = args.cache_dir
            else:
                block_cache = Path(tempfile.mkdtemp(dir=args.cache_dir))
            try:
                # a fresh cache per block: the filesystem instances are cached
                # with their cache directory
                WholeFileCacheFileSystem.clear_instance_cache()
                block_config = mdp.Config.from_yaml_file(args.config)
                _use_cache(block_config, block_cache)
                stats, features, carry = _process_block(
                    block_config,
                    block[0],
                    block[1],
                    pd.Timedelta(step).to_timedelta64(),
                    args.sub_block_steps,
                    tail,
                    start,
                    args.max_retries,
                )
            finally:
                if not args.keep_cache:
                    shutil.rmtree(block_cache, ignore_errors=True)
            _save_atomic(
                _fp("tail", block),
                **{
                    f"{c}_{kind}": v
                    for c, (times, x) in carry.items()
                    for kind, v in [("times", times), ("values", x)]
                },
            )
            _save_atomic(
                _fp("block", block),
                **{
                    k: v
                    for c, s in stats.items()
                    for k, v in s.to_arrays(prefix=f"{c}_").items()
                },
                **{f"{c}_feature": f.values for c, f in features.items()},
            )
            # the previous block's tail is not needed any more
            if k > 0:
                _fp("tail", blocks[k - 1]).unlink(missing_ok=True)
            pbar.update(1)

    # combine the blocks
    stats = {}
    for block in blocks:
        arrays = np.load(_fp("block", block))
        for c in CATEGORIES:
            if f"{c}_feature" not in arrays:
                continue
            block_stats = SplitStats.from_arrays(arrays, prefix=f"{c}_")
            if c not in stats:
                dim, names = f"{c}_feature", arrays[f"{c}_feature"]
                feature = xr.DataArray(names, dims=[dim], coords={dim: names})
                stats[c] = (block_stats, feature)
            else:
                stats[c][0].merge(block_stats)

    # the statics (time independent), from the first two times
    ds_static = mdp.create_dataset(
        config=_block_config(
            mdp.Config.from_yaml_file(args.config),
            _py(start),
            _py(start) + step,
        )
    )
    static = ds_static["static"].load() if "static" in ds_static else None

    ds = stats_dataset(
        stats,
        static,
        _py(start),
        _py(end),
        step,
        description=(
            "Training statistics of CLARA's interior statistics datastore "
            "(danra_model1_config), recomputed from DANRA v0.5.0 with "
            "clara_inference.artifact.danra_stats (as mllam-data-prep)"
        ),
    )
    ds.attrs.update(
        exact=str(not args.dry_run),
        diff_std_note="std of second time-differences (as in training)",
        created_on=datetime.datetime.now().replace(microsecond=0).isoformat(),
    )
    ds.to_zarr(args.output, mode="w", consolidated=True)
    logger.info(f"wrote {args.output}")


if __name__ == "__main__":
    main()
