"""
Assemble a CLARA inference artifact directory (layout in `layout.py`) from:

- the training configs in this repository (`artifact/configs/`)
- the statistics datastores (`compute_stats.py` output)
- the training checkpoint, sanitized (`sanitize_checkpoint.py`)
- the graph the model was trained with (a directory or zip)
- the DANRA grid and statics (from the DANRA v0.5.0 object store, or a
  cached `create-danra-grid` zarr)
- the ERA5 statics (land_sea_mask, geopotential_at_surface; WeatherBench2)
  on the boundary grid

The boundary grid is derived from the graph: its boundary (g2m source) nodes
are the training boundary points in training order, in the interior
projection; they are projected back to lat/lon and snapped to the 0.25 deg
grid, and the boundary grid is the latitude band they span (see
`ifs_boundary.boundary_grid_from_points`).

Usage:
    python -m clara_inference.artifact.assemble \\
        --checkpoint .../last.ckpt --graph .../feature_map_graph_198x148.zip \\
        --stats-dir artifact/build/stats --output artifact/build/clara-local
"""
import argparse
import datetime
import hashlib
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pyproj
import torch
import xarray as xr
import yaml
from loguru import logger

from clara_inference import ifs_boundary
from clara_inference.artifact import layout
from clara_inference.convert_output import projection_crs
from clara_inference.regrid_dini import DANRA_SL_URL, create_danra_grid

REPO_CONFIGS = Path(__file__).parents[3] / "artifact" / "configs"
PACKAGE_README = Path(__file__).parents[3] / "artifact" / "package-README.md"
ERA5_WB2 = "gs://weatherbench2/datasets/era5/1959-2022-6h-1440x721.zarr"


def _md5(fp):
    md5 = hashlib.md5()
    with open(fp, "rb") as fh:
        for chunk in iter(lambda: fh.read(2**20), b""):
            md5.update(chunk)
    return md5.hexdigest()


def _copy_graph(src, dst):
    """Copy the graph files from a directory or zip into `dst`."""
    dst.mkdir(parents=True, exist_ok=True)
    src = Path(src)
    if src.suffix == ".zip":
        with zipfile.ZipFile(src) as zf:
            for name in zf.namelist():
                if Path(name).name in layout.GRAPH_FILES and (
                    "__MACOSX" not in name
                ):
                    (dst / Path(name).name).write_bytes(zf.read(name))
    else:
        for fn in layout.GRAPH_FILES:
            shutil.copy2(src / fn, dst / fn)
    missing = [fn for fn in layout.GRAPH_FILES if not (dst / fn).exists()]
    if missing:
        raise FileNotFoundError(f"graph files missing in {src}: {missing}")


def boundary_points_from_graph(graph_dir, projection):
    """
    lat/lon (on the 0.25 deg grid) of the graph's boundary nodes, in order:
    the g2m source nodes, which come before the mesh nodes.
    """
    g2m = torch.load(graph_dir / "g2m_node_features.pt", weights_only=True)
    m2m = torch.load(graph_dir / "m2m_node_features.pt", weights_only=True)
    g2m = (g2m[0] if isinstance(g2m, list) else g2m).numpy().astype("f8")
    m2m = m2m[0] if isinstance(m2m, list) else m2m
    n_boundary = g2m.shape[0] - m2m.shape[0]
    transformer = pyproj.Transformer.from_crs(
        projection, "EPSG:4326", always_xy=True
    )
    lon, lat = transformer.transform(
        g2m[:n_boundary, 0], g2m[:n_boundary, 1]
    )
    lat_r, lon_r = np.round(lat * 4) / 4, np.round(lon * 4) / 4
    off_grid = max(np.abs(lat - lat_r).max(), np.abs(lon - lon_r).max())
    if off_grid > 1e-3:
        raise ValueError(
            f"graph boundary nodes are not on the 0.25 deg grid (max offset "
            f"{off_grid} deg)"
        )
    return lat_r, lon_r


def _git_commit():
    try:
        return subprocess.run(
            ["git", "-C", str(REPO_CONFIGS), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def assemble(
    output,
    checkpoint,
    graph,
    stats_dir,
    danra_grid=None,
    era5_statics=ERA5_WB2,
    checkpoint_url=None,
    placeholder_stats=False,
):
    output = Path(output)
    configs_dir = output / layout.CONFIGS
    if output.exists():
        raise FileExistsError(f"{output} exists, remove it first")
    shutil.copytree(REPO_CONFIGS, configs_dir)
    training = layout.training_configs(output)
    model = layout.model_description(output)

    # statistics datastores, next to their configs
    for fn in training.stats_datastores:
        name = Path(fn).stem
        shutil.copytree(
            Path(stats_dir) / f"{name}.zarr",
            layout.stats_datastore_path(output, fn),
        )

    # graph
    graph_dir = layout.graph_path(output)
    _copy_graph(graph, graph_dir)

    # interior (DANRA) grid and statics
    fp_danra_grid = layout.grid_path(output, training.interior_datastore)
    fp_danra_grid.parent.mkdir(parents=True, exist_ok=True)
    if danra_grid is None:
        create_danra_grid(fp_danra_grid, DANRA_SL_URL)
    else:
        shutil.copytree(danra_grid, fp_danra_grid)

    # boundary grid with the ERA5 statics, from the graph's boundary nodes
    projection = projection_crs(configs_dir / training.interior_datastore)
    lat, lon = boundary_points_from_graph(graph_dir, projection)
    ds_statics = xr.open_zarr(era5_statics)
    ds_grid = ifs_boundary.boundary_grid_from_points(lat, lon, ds_statics)
    ds_grid.attrs = dict(
        source=str(era5_statics),
        description=(
            "boundary grid of CLARA: the 0.25 deg latitude band of the "
            f"{lat.size} training boundary points (from the graph), "
            "longitude 0..360 increasing, latitude decreasing (training "
            "point order), with the ERA5 statics"
        ),
    )
    _check_no_nan_at(ds_grid, lat, lon)
    ds_grid.to_zarr(
        layout.grid_path(output, training.boundary_datastore),
        mode="w",
        consolidated=True,
    )

    # sanitized checkpoint
    fp_ckpt = layout.checkpoint_path(output)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "clara_inference.artifact.sanitize_checkpoint",
            str(checkpoint),
            str(fp_ckpt),
        ],
        check=True,
    )
    md5 = _md5(fp_ckpt)
    published_md5 = model["checkpoint"].get("md5")
    if published_md5 is not None and published_md5 != md5:
        raise ValueError(
            f"the sanitized checkpoint's md5 {md5} differs from the published "
            f"one in model.yaml ({published_md5}): another training checkpoint?"
        )
    model["checkpoint"]["md5"] = md5
    if checkpoint_url is not None:
        model["checkpoint"]["url"] = checkpoint_url
    _update_model_yaml(configs_dir / "model.yaml", model["checkpoint"])

    meta = dict(
        name=output.name,
        created=datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds"
        ),
        clara_inference_commit=_git_commit(),
        training_checkpoint=str(checkpoint),
        checkpoint_md5=model["checkpoint"]["md5"],
        graph=str(graph),
        stats={
            Path(fn).stem: xr.open_zarr(
                layout.stats_datastore_path(output, fn)
            ).attrs
            for fn in training.stats_datastores
        },
        era5_statics=str(era5_statics),
        boundary_points=int(lat.size),
        placeholder_stats=bool(placeholder_stats),
    )
    (output / layout.ARTIFACT_META).write_text(
        yaml.safe_dump(meta, sort_keys=False)
    )
    shutil.copy2(PACKAGE_README, output / layout.README)
    layout.check_artifact(output, checkpoint=True)
    logger.info(f"assembled and checked the inference artifact {output}")


def _check_no_nan_at(ds_grid, lat, lon):
    ds = ds_grid.sel(
        latitude=xr.DataArray(lat, dims="point"),
        longitude=xr.DataArray(lon % 360, dims="point"),
    )
    for v in ifs_boundary.STATIC_VARS:
        n = int(ds[v].isnull().sum())
        if n:
            raise ValueError(f"{v} is missing at {n} boundary points")


def _update_model_yaml(fp, checkpoint):
    """Fill in the checkpoint url/md5 lines, keeping the comments."""
    lines = fp.read_text().splitlines()
    out, in_checkpoint = [], False
    for line in lines:
        if line.startswith("checkpoint:"):
            in_checkpoint = True
        elif line and not line.startswith((" ", "#")):
            in_checkpoint = False
        if in_checkpoint and line.strip().startswith(("url:", "md5:")):
            key = line.strip().split(":")[0]
            value = checkpoint[key]
            line = f"  {key}: {value if value is not None else 'null'}"
        out.append(line)
    fp.write_text("\n".join(out) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--checkpoint", required=True, type=Path, help="training checkpoint"
    )
    parser.add_argument(
        "--graph", required=True, type=Path, help="graph directory or zip"
    )
    parser.add_argument(
        "--stats-dir",
        required=True,
        type=Path,
        help="compute_stats.py output (<name>.zarr per stats datastore)",
    )
    parser.add_argument(
        "--danra-grid",
        type=Path,
        help="cached DANRA grid zarr (default: from the DANRA object store)",
    )
    parser.add_argument(
        "--era5-statics",
        default=ERA5_WB2,
        help="zarr with ERA5 land_sea_mask and geopotential_at_surface "
        "(default: %(default)s)",
    )
    parser.add_argument(
        "--checkpoint-url",
        help="where the checkpoint is published (default: the url in "
        "artifact/configs/model.yaml)",
    )
    parser.add_argument(
        "--placeholder-stats",
        action="store_true",
        help="mark the statistics as placeholders (testing only)",
    )
    parser.add_argument(
        "--zip", type=Path, help="also write the package zip (no checkpoint)"
    )
    args = parser.parse_args()

    assemble(
        output=args.output,
        checkpoint=args.checkpoint,
        graph=args.graph,
        stats_dir=args.stats_dir,
        danra_grid=args.danra_grid,
        era5_statics=args.era5_statics,
        checkpoint_url=args.checkpoint_url,
        placeholder_stats=args.placeholder_stats,
    )
    if args.zip:
        layout.write_zip(args.output, args.zip)
        logger.info(f"wrote {args.zip}")


if __name__ == "__main__":
    main()
