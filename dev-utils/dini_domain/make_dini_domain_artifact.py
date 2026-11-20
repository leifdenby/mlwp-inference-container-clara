"""
Experiment: derive a CLARA inference artifact for (almost) the whole DINI
domain from the DANRA-domain artifact, without retraining.

The weights of CLARA's model (graph encoder for the boundary, two U-Nets with
self-attention at the coarse levels) don't depend on the grid size; only the
U-Nets' parameter *names* (`enc.790x590_block0...`), their padding and the
choice of attention levels (`--attn_res`, matched against the resolutions)
are derived from it. So the model can be built for another grid, and the
checkpoint's U-Net weights renamed level by level. What does depend on the
domain:

- the interior grid: here 2.5 km (as DANRA) in DINI's own Lambert projection,
  `--nx` x `--ny` points centred in DINI, keeping clear of DINI's edge
  (relaxation) zone. A grid in DANRA's projection can't cover DINI, which is
  rotated against it
- the interior statics, from DINI: `lsm` (made binary, as DANRA's) and
  `orography` (DINI's surface altitude as surface geopotential, as DANRA's)
- the boundary: the 0.25 deg points within 400 km of the interior rectangle,
  in mllam-data-prep's order (longitude 0..360, then latitude decreasing),
  and its grid (the latitude band) with the ERA5 statics
- the graph: built with real-prob-lam's feature-map graph code
  (`neural_lam.build_feature_map_graph`), mesh spacing 4 interior grid
  spacings as in training
- the checkpoint: U-Net weights renamed for the new resolutions, and the
  `--attn_res` that puts the attention at the same levels as in training

The statistics (normalisation) are those of training. Known extrapolations
(not fixed here, they are what the experiment tests): the mesh nodes' static
features are absolute positions (scaled by the largest interior coordinate),
GroupNorm and the attention see a bigger and different domain, the
orography reaches beyond DANRA's (Alps, Iceland, Norway), and the grid's
orientation against north spans another range.

Usage:
    uv run python dev-utils/dini_domain/make_dini_domain_artifact.py \\
        --artifact artifact/build/clara-exact-stats \\
        --output artifact/build/clara-dini-domain
"""
import argparse
import re
import shutil
from pathlib import Path

import cartopy.crs as ccrs
import mllam_data_prep as mdp
import numpy as np
import pyproj
import torch
import weather_model_graphs as wmg
import xarray as xr
import yaml
from loguru import logger
from neural_lam.build_feature_map_graph import (
    create_single_level_2d_mesh_graph_from_interior,
)
from neural_lam.models.edm_networks_2 import SongUNet

from clara_inference import ifs_boundary, storage
from clara_inference.artifact import layout
from clara_inference.artifact.assemble import ERA5_WB2
from clara_inference.convert_output import projection_crs
from clara_inference.regrid_dini import BilinearInterpolator

DINI_ROOT = "s3://harmonie-zarr/dini/control/2026-11-19T180000Z"
GRAVITY = 9.80665
DX = 2500.0
# 4 k + 1 points, as DANRA's 789 x 589: the grid U-Net's bottleneck (k + 1)
# is then the mesh's interior block, as in training. The whole mesh must have
# even sizes (as in training, 278 x 228): neural-lam's SongUNet rounds odd
# resolutions up in place, on the tensor GraphUnet2 also keeps as its mesh
# shape
NX, NY = 1501, 1261
MESH_FACTOR = 4
MARGIN_M = 400e3
GRAPH_PREFIX = "feature_map_graph_dini"
CHECKPOINT_NAME = "clara_model_dini_domain.ckpt"
U_NETS = dict(
    grid_processor=("hidden_dim_grid", "channel_mult_grid"),
    mesh_processor=("hidden_dim", "channel_mult"),
)


def projection_kwargs_from_wkt(crs_wkt: str) -> dict:
    """
    The cartopy LambertConformal arguments (as in a datastore config's
    `extra.projection.kwargs`) of a spherical Lambert CRS.

    Parameters
    ----------
    crs_wkt : str
        The CRS, e.g. DINI's `dini_projection.crs_wkt`.

    Returns
    -------
    dict
        central_longitude, central_latitude, standard_parallels, globe.
    """
    p = pyproj.CRS.from_wkt(crs_wkt).to_dict()
    if p.get("proj") != "lcc" or "R" not in p:
        raise ValueError(f"expected a spherical Lambert projection, got {p}")
    return dict(
        central_longitude=float(p["lon_0"]),
        central_latitude=float(p["lat_0"]),
        standard_parallels=[float(p["lat_1"]), float(p["lat_2"])],
        globe=dict(semimajor_axis=float(p["R"]), semiminor_axis=float(p["R"])),
    )


def create_interior_grid(
    ds_dini: xr.Dataset, nx: int, ny: int, dx: float
) -> xr.Dataset:
    """
    Create the interior grid and statics: `nx` x `ny` points every `dx`,
    centred in DINI, in DINI's projection.

    Parameters
    ----------
    ds_dini : xr.Dataset
        DINI single levels (x, y, lat/lon, lsm, orography, dini_projection).
    nx, ny : int
        Grid size.
    dx : float
        Grid spacing (m).

    Returns
    -------
    xr.Dataset
        x, y, lat, lon, lsm (binary) and orography (surface geopotential),
        laid out as the DANRA grid zarr of the artifact.
    """
    crs = pyproj.CRS.from_wkt(ds_dini["dini_projection"].attrs["crs_wkt"])
    x_dini, y_dini = ds_dini.x.values, ds_dini.y.values
    xc = (x_dini[0] + x_dini[-1]) / 2
    yc = (y_dini[0] + y_dini[-1]) / 2
    x = xc + dx * (np.arange(nx) - (nx - 1) / 2)
    y = yc + dx * (np.arange(ny) - (ny - 1) / 2)
    margin = min(x[0] - x_dini[0], x_dini[-1] - x[-1])
    margin = min(margin, y[0] - y_dini[0], y_dini[-1] - y[-1])
    logger.info(
        f"interior grid {nx} x {ny} every {dx / 1e3} km, {margin / 1e3:.0f} "
        "km inside DINI's edge"
    )
    xx, yy = np.meshgrid(x, y)
    to_lonlat = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    lon, lat = to_lonlat.transform(xx, yy)
    interp = BilinearInterpolator(x_dini, y_dini, xx, yy, outside="error")
    lsm = (interp(ds_dini.lsm.values) >= 0.5).astype("f4")
    orography = (interp(ds_dini.orography.values) * GRAVITY).astype("f4")
    ds = xr.Dataset(
        dict(
            lsm=(("y", "x"), lsm, dict(units="1", long_name="land (1) / sea")),
            orography=(
                ("y", "x"),
                orography,
                dict(units="m**2 s**-2", long_name="surface geopotential"),
            ),
        ),
        coords=dict(
            x=("x", x),
            y=("y", y),
            lat=(("y", "x"), lat, dict(units="degrees_north")),
            lon=(("y", "x"), lon, dict(units="degrees_east")),
        ),
    )
    ds.attrs = dict(
        description=(
            f"CLARA DINI-domain experiment: {nx} x {ny} grid every "
            f"{dx / 1e3} km in DINI's projection, statics from DINI"
        ),
        crs_wkt=crs.to_wkt(),
    )
    return ds


def write_configs(
    src_artifact: Path, output: Path, projection_kwargs: dict
) -> None:
    """
    Copy the artifact's configs (with the statistics datastores) and set the
    interior datastore's projection.

    Parameters
    ----------
    src_artifact : Path
        The DANRA-domain artifact.
    output : Path
        The new artifact.
    projection_kwargs : dict
        `extra.projection.kwargs` of the interior.
    """
    shutil.copytree(src_artifact / layout.CONFIGS, output / layout.CONFIGS)
    model = layout.model_description(output)
    fp = output / layout.CONFIGS / model["interior_datastore"]
    config = mdp.Config.from_yaml_file(fp)
    config.extra["projection"]["kwargs"] = projection_kwargs
    # NB: sort_keys=False, sorting would change the feature order
    config.to_yaml_file(fp, sort_keys=False)


def select_boundary_points(
    ds_statics: xr.Dataset,
    ds_grid: xr.Dataset,
    projection: ccrs.Projection,
    margin: float = MARGIN_M,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Select the 0.25 deg points within `margin` of the interior rectangle (in
    the interior projection), interior points included, in mllam-data-prep's
    order (`crop_rectangular_area` on a grid stacked [longitude, latitude],
    longitude 0..360 increasing, latitude decreasing).

    Parameters
    ----------
    ds_statics : xr.Dataset
        A global 0.25 deg dataset (latitude, longitude).
    ds_grid : xr.Dataset
        The interior grid (lat, lon).
    projection : ccrs.Projection
        The interior projection.
    margin : float
        Margin (m).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        lat, lon of the boundary points, in order.
    """
    lats = np.sort(ds_statics.latitude.values)[::-1]
    lons = np.sort(ds_statics.longitude.values % 360)
    lon2d, lat2d = np.meshgrid(lons, lats, indexing="ij")
    lon_f, lat_f = lon2d.ravel(), lat2d.ravel()
    pc = ccrs.PlateCarree()
    xy = projection.transform_points(pc, lon_f, lat_f)[..., :2]
    ref = projection.transform_points(
        pc, ds_grid.lon.values.ravel(), ds_grid.lat.values.ravel()
    )[..., :2]
    lo, hi = ref.min(axis=0) - margin, ref.max(axis=0) + margin
    # only the hemisphere around the domain (the projection folds far away)
    near = np.abs(((lon_f - float(ds_grid.lon.mean())) + 180) % 360 - 180) < 90
    near &= lat_f > 0
    mask = near & np.isfinite(xy).all(axis=1)
    mask &= (xy[:, 0] >= lo[0]) & (xy[:, 0] <= hi[0])
    mask &= (xy[:, 1] >= lo[1]) & (xy[:, 1] <= hi[1])
    logger.info(
        f"{int(mask.sum())} boundary points within {margin / 1e3:.0f} km, "
        f"lat {lat_f[mask].min()}..{lat_f[mask].max()}, lon (0..360) "
        f"{np.unique(lon_f[mask])[[0, -1]]}"
    )
    return lat_f[mask], lon_f[mask]


def build_graph(
    xy_interior: np.ndarray,
    xy_boundary: np.ndarray,
    nx_i: int,
    ny_i: int,
    graph_dir: Path,
) -> tuple[int, int]:
    """
    Build the feature-map graph as `neural_lam.build_feature_map_graph` does
    from the datastores (same functions and settings).

    Parameters
    ----------
    xy_interior : np.ndarray
        (N, 2) interior grid points (projection coordinates).
    xy_boundary : np.ndarray
        (M, 2) boundary points in datastore order.
    nx_i, ny_i : int
        Mesh nodes over the interior.
    graph_dir : Path
        Output directory.

    Returns
    -------
    tuple[int, int]
        The mesh shape (nx, ny), as GraphUnet2 infers it.
    """
    xy = np.concatenate((xy_interior, xy_boundary), axis=0)
    components = dict(
        m2m=create_single_level_2d_mesh_graph_from_interior(
            xy, xy_interior, nx_i=nx_i, ny_i=ny_i
        )
    )
    g_grid = wmg.create.grid.create_grid_graph_nodes(xy=xy_boundary)
    components["g2m"] = wmg.create.base.connect_nodes_across_graphs(
        G_source=g_grid,
        G_target=components["m2m"],
        method="within_radius",
        rel_max_dist=0.51,
    )
    for name, graph in components.items():
        for edge in graph.edges:
            graph.edges[edge]["component"] = name
    components = {
        name: wmg.networkx_utils.replace_node_labels_with_unique_ids(graph)
        for name, graph in components.items()
    }
    graph_dir.mkdir(parents=True, exist_ok=True)
    for name, graph in components.items():
        kwargs = dict(list_from_attribute="dummy") if name == "m2m" else {}
        wmg.save.to_pyg(
            graph=graph,
            name=name,
            edge_features=["len", "vdiff"],
            output_directory=str(graph_dir),
            **kwargs,
        )
    mesh = torch.load(graph_dir / "m2m_node_features.pt", weights_only=True)
    mesh = mesh[0] if isinstance(mesh, list) else mesh
    shape = (
        int(torch.unique(mesh[:, 0]).numel()),
        int(torch.unique(mesh[:, 1]).numel()),
    )
    logger.info(f"graph {graph_dir.name}: mesh {shape[0]} x {shape[1]}")
    return shape


def create_u_net(
    args: dict, which: str, shape: tuple[int, int], attn_res: list[int]
) -> SongUNet:
    """
    Create one of GraphUnet2's U-Nets (as in its constructor) on the meta
    device.

    Parameters
    ----------
    args : dict
        The checkpoint's training arguments.
    which : str
        "grid_processor" or "mesh_processor".
    shape : tuple[int, int]
        The (x, y) resolution.
    attn_res : list[int]
        Resolutions (x) with self-attention.

    Returns
    -------
    SongUNet
        The U-Net, without weights.
    """
    hidden, channel_mult = (args[k] for k in U_NETS[which])
    # on the CPU: SongUNet computes the level resolutions from it
    resolution = torch.tensor(shape)
    with torch.device("meta"):
        return SongUNet(
            img_resolution=resolution,
            in_channels=hidden,
            out_channels=hidden,
            model_channels=hidden,
            channel_mult=channel_mult,
            channel_mult_emb=args["channel_mult_emb"],
            channel_mult_noise=args["channel_mult_noise"],
            noise_dim=args["noise_dim"],
            attn_resolutions=attn_res,
        )


def level_resolutions(keys: list[str], which: str) -> list[int]:
    """
    Get the x resolution of each encoder level from U-Net parameter names.

    Parameters
    ----------
    keys : list[str]
        Parameter names (with or without the `model.<which>.` prefix).
    which : str
        "grid_processor" or "mesh_processor".

    Returns
    -------
    list[int]
        x resolution per level, finest first.
    """
    res = []
    for k in keys:
        m = re.match(rf"(?:model\.{which}\.)?enc\.(\d+)x\d+_", k)
        if m and int(m.group(1)) not in res:
            res.append(int(m.group(1)))
    return res


def resize_checkpoint(
    fp_in: Path,
    fp_out: Path,
    grid_shape: tuple[int, int],
    mesh_shape: tuple[int, int],
) -> list[int]:
    """
    Rename the U-Net weights of a checkpoint for new grid and mesh shapes.

    Parameters
    ----------
    fp_in, fp_out : Path
        The (sanitized) checkpoint and the resized one.
    grid_shape, mesh_shape : tuple[int, int]
        The new (x, y) shapes of the interior grid and the mesh.

    Returns
    -------
    list[int]
        The `--attn_res` that puts the attention at the trained levels.
    """
    ckpt = torch.load(fp_in, map_location="cpu", weights_only=True)
    args = ckpt["hyper_parameters"]["args"]
    sd = ckpt["state_dict"]
    shapes = dict(grid_processor=grid_shape, mesh_processor=mesh_shape)

    # the attention levels in training, and their new resolutions
    attn_levels = {}
    for which in U_NETS:
        old_res = level_resolutions(list(sd), which)
        new = create_u_net(args, which, shapes[which], [])
        new_res = level_resolutions(list(new.state_dict()), which)
        if len(new_res) != len(old_res):
            raise ValueError(f"{which}: levels {old_res} vs {new_res}")
        attn_levels[which] = (
            [i for i, r in enumerate(old_res) if r in args["attn_res"]],
            new_res,
        )
        logger.info(f"{which}: level resolutions {old_res} -> {new_res}")
    levels, new_res = attn_levels["mesh_processor"]
    attn_res = [new_res[i] for i in levels]
    grid_levels, grid_res = attn_levels["grid_processor"]
    if grid_levels or set(attn_res) & set(grid_res):
        raise ValueError(
            f"attention resolutions {attn_res} would also select grid U-Net "
            f"levels {grid_res}"
        )

    # the U-Nets' parameters are in registration order, the same for any
    # resolution: rename them in order, checking the shapes
    new_names = {}
    for which in U_NETS:
        prefix = f"model.{which}."
        old_keys = [k for k in sd if k.startswith(prefix)]
        new_sd = create_u_net(
            args, which, shapes[which], attn_res
        ).state_dict()
        if len(new_sd) != len(old_keys):
            raise ValueError(
                f"{which}: {len(old_keys)} parameters in the checkpoint, "
                f"{len(new_sd)} in the resized U-Net"
            )
        for k_old, (k_new, p_new) in zip(old_keys, new_sd.items()):
            if tuple(sd[k_old].shape) != tuple(p_new.shape):
                raise ValueError(f"{k_old} -> {k_new}: shapes differ")
            new_names[k_old] = prefix + k_new
    ckpt["state_dict"] = {new_names.get(k, k): v for k, v in sd.items()}
    ckpt["hyper_parameters"]["args"] = dict(args, attn_res=attn_res)
    torch.save(ckpt, fp_out)
    logger.info(
        f"resized checkpoint {fp_out}: {len(ckpt['state_dict'])} tensors, "
        f"--attn_res {attn_res}"
    )
    return attn_res


def write_model_yaml(
    output: Path, graph_name: str, attn_res: list[int], description: str
) -> None:
    """
    Point model.yaml at the new graph and checkpoint, with the new
    `--attn_res`.

    Parameters
    ----------
    output : Path
        The new artifact.
    graph_name : str
        The new graph.
    attn_res : list[int]
        The new attention resolutions.
    description : str
        What this artifact is.
    """
    fp = output / layout.CONFIGS / "model.yaml"
    model = yaml.safe_load(fp.read_text())
    model["checkpoint"] = dict(
        derived_from=dict(model["checkpoint"]),
        artifact_path=CHECKPOINT_NAME,
    )
    model["graph"]["name"] = graph_name
    margs = [str(a) for a in model["train_model_args"]]
    i = margs.index("--attn_res") + 1
    j = i
    while j < len(margs) and not margs[j].startswith("--"):
        j += 1
    model["train_model_args"] = (
        margs[:i] + [str(r) for r in attn_res] + margs[j:]
    )
    model["experiment"] = description
    fp.write_text(yaml.safe_dump(model, sort_keys=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--dini-root",
        default=DINI_ROOT,
        help="any DINI run (only its grid and statics are used; default: "
        "%(default)s)",
    )
    parser.add_argument("--nx", type=int, default=NX)
    parser.add_argument("--ny", type=int, default=NY)
    parser.add_argument("--era5-statics", default=ERA5_WB2)
    args = parser.parse_args()
    for n in (args.nx, args.ny):
        if (n - 1) % MESH_FACTOR:
            raise ValueError(f"grid size {n} must be {MESH_FACTOR} k + 1")
    if args.output.exists():
        raise FileExistsError(f"{args.output} exists")

    src = args.artifact
    model = layout.model_description(src)
    training = layout.training_configs(src)
    ds_dini = storage.open_zarr(
        f"{args.dini_root.rstrip('/')}/single_levels.zarr"
    )

    # interior grid, configs
    ds_grid = create_interior_grid(ds_dini, args.nx, args.ny, DX)
    write_configs(
        src,
        args.output,
        projection_kwargs_from_wkt(
            ds_dini["dini_projection"].attrs["crs_wkt"]
        ),
    )
    fp_grid = layout.grid_path(args.output, training.interior_datastore)
    fp_grid.parent.mkdir(parents=True, exist_ok=True)
    ds_grid.to_zarr(fp_grid, mode="w", consolidated=True)
    projection = projection_crs(
        args.output / layout.CONFIGS / training.interior_datastore
    )

    # boundary points and grid
    ds_statics = xr.open_zarr(args.era5_statics)
    lat_b, lon_b = select_boundary_points(ds_statics, ds_grid, projection)
    ds_bgrid = ifs_boundary.boundary_grid_from_points(lat_b, lon_b, ds_statics)
    ds_bgrid.attrs = dict(
        source=str(args.era5_statics),
        description=(
            "boundary grid of the CLARA DINI-domain experiment: the 0.25 deg "
            f"latitude band of the {lat_b.size} points within 400 km of the "
            "interior, longitude 0..360 increasing, latitude decreasing, "
            "with the ERA5 statics"
        ),
    )
    ds_bgrid.to_zarr(
        layout.grid_path(args.output, training.boundary_datastore),
        mode="w",
        consolidated=True,
    )

    # graph
    nx_i = (args.nx - 1) // MESH_FACTOR + 1
    ny_i = (args.ny - 1) // MESH_FACTOR + 1
    graph_name = f"{GRAPH_PREFIX}_{nx_i}x{ny_i}"
    xx, yy = np.meshgrid(ds_grid.x.values, ds_grid.y.values, indexing="ij")
    xy_interior = np.stack([xx.ravel(), yy.ravel()], axis=1)
    xy_boundary = projection.transform_points(
        ccrs.PlateCarree(), lon_b, lat_b
    )[..., :2]
    mesh_shape = build_graph(
        xy_interior,
        xy_boundary,
        nx_i,
        ny_i,
        args.output / layout.GRAPHS / graph_name,
    )

    if mesh_shape[0] % 2 or mesh_shape[1] % 2:
        raise ValueError(
            f"the mesh is {mesh_shape[0]} x {mesh_shape[1]}, it must have even "
            "sizes (see NX, NY): change --nx/--ny by 4"
        )

    # checkpoint, model.yaml
    attn_res = resize_checkpoint(
        layout.checkpoint_path(src),
        args.output / CHECKPOINT_NAME,
        grid_shape=(args.nx, args.ny),
        mesh_shape=mesh_shape,
    )
    description = (
        "DINI-domain experiment (dev-utils/dini_domain): CLARA without "
        f"retraining on a {args.nx} x {args.ny} 2.5 km grid in DINI's "
        f"projection, graph {graph_name} (mesh {mesh_shape[0]} x "
        f"{mesh_shape[1]}), {lat_b.size} boundary points; the U-Net weights "
        f"of {model['checkpoint']['artifact_path']} renamed for the new "
        "resolutions"
    )
    write_model_yaml(args.output, graph_name, attn_res, description)
    meta = yaml.safe_load((src / layout.ARTIFACT_META).read_text())
    meta = dict(
        name=args.output.name,
        derived_from=str(src.resolve()),
        experiment=description,
        dini_statics=args.dini_root,
        source_artifact=meta,
    )
    (args.output / layout.ARTIFACT_META).write_text(
        yaml.safe_dump(meta, sort_keys=False)
    )
    (args.output / layout.README).write_text(
        f"# {args.output.name}\n\n{description}\n\nMade by "
        "`dev-utils/dini_domain/make_dini_domain_artifact.py`, for "
        "experiments only.\n"
    )
    layout.check_artifact(args.output, checkpoint=True)
    logger.info(f"wrote and checked {args.output}")


if __name__ == "__main__":
    main()
