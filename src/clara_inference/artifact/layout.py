"""
The layout of an ANNA inference artifact, shared by the tools that build it
(`collect` on the training machine, `assemble` for the reconstruction from
gefion-1) and by the checks. The artifact describes the model **as
trained** (DANRA interior, ERA5 boundary), nothing about the operational
boundaries (DINI, IFS), which are part of this package:

    configs/model.yaml              the model: checkpoint, graph recipe,
                                    train_model arguments and the training
                                    neural-lam config
    configs/<neural-lam config>     the training neural-lam config, with
                                    datastore config paths as sibling files
    configs/<datastore config>      the training interior and boundary
                                    datastore configs it names
    configs/<boundary>.zarr         the "stats datastore" neural-lam opens
                                    for `overload_stats_path` (statistics and
                                    splits of the training boundary datastore)
    stats/<datastore>.stats.zarr    training statistics of the training
                                    interior and boundary datastores
    grids/<interior>.grid.zarr      interior grid (x, y, lat, lon) and statics,
                                    the target grid for regridding DINI
    grids/<boundary>.grid.zarr      lat/lon of the training boundary points
    artifact.yaml                   provenance
    README.md                       description (artifact/package-README.md)
    <checkpoint.artifact_path>      the checkpoint (never in the published
                                    package zip)

The datastore names (config file names without `.yaml`) follow from the
training neural-lam config, so the same layout serves any model trained
with an interior and a boundary datastore.
"""
import zipfile
from pathlib import Path

import numpy as np
import yaml

from anna_inference import model_configs as mc

CONFIGS = "configs"
STATS = "stats"
GRIDS = "grids"
ARTIFACT_META = "artifact.yaml"
README = "README.md"


def stats_path(artifact_dir, datastore_config):
    return Path(artifact_dir) / STATS / f"{_name(datastore_config)}.stats.zarr"


def grid_path(artifact_dir, datastore_config):
    return Path(artifact_dir) / GRIDS / f"{_name(datastore_config)}.grid.zarr"


def stats_datastore_path(artifact_dir, datastore_config):
    """The zarr neural-lam opens next to an `overload_stats_path` config."""
    return Path(artifact_dir) / CONFIGS / f"{_name(datastore_config)}.zarr"


def checkpoint_path(artifact_dir):
    model = yaml.safe_load(
        (Path(artifact_dir) / CONFIGS / "model.yaml").read_text()
    )
    return Path(artifact_dir) / model["checkpoint"]["artifact_path"]


def training_configs(artifact_dir):
    """The training configs (named by configs/model.yaml)."""
    return mc.training_configs(Path(artifact_dir) / CONFIGS)


def required_paths(artifact_dir, checkpoint=False):
    """All paths an artifact must have (from its configs/model.yaml)."""
    artifact_dir = Path(artifact_dir)
    configs_dir = artifact_dir / CONFIGS
    training = training_configs(artifact_dir)
    paths = {
        configs_dir / "model.yaml",
        configs_dir / training.neural_lam_config,
        configs_dir / training.interior_datastore,
        configs_dir / training.boundary_datastore,
        stats_datastore_path(artifact_dir, training.boundary_datastore),
        artifact_dir / ARTIFACT_META,
        artifact_dir / README,
    }
    for datastore in [
        training.interior_datastore,
        training.boundary_datastore,
    ]:
        paths |= {
            stats_path(artifact_dir, datastore),
            grid_path(artifact_dir, datastore),
        }
    if checkpoint:
        paths.add(checkpoint_path(artifact_dir))
    return sorted(paths)


def check_artifact(artifact_dir, checkpoint=False):
    """
    Check an artifact directory: all required paths exist, it has nothing
    in configs/ besides the training configs and the stats datastore (no
    operational configs), and the stats datastore loads in neural-lam.
    Raises ValueError on problems.
    """
    artifact_dir = Path(artifact_dir)
    required = required_paths(artifact_dir, checkpoint=checkpoint)
    problems = [
        f"missing {p.relative_to(artifact_dir)}"
        for p in required
        if not p.exists()
    ]
    configs_dir = artifact_dir / CONFIGS
    problems += [
        f"{p.relative_to(artifact_dir)} isn't a training config"
        for p in sorted(configs_dir.iterdir())
        if p not in required
    ]
    training = training_configs(artifact_dir)
    if problems:
        raise ValueError(
            f"inference artifact {artifact_dir}: " + "; ".join(problems)
        )
    check_stats_datastore(configs_dir / training.boundary_datastore)


def check_stats_datastore(fp_boundary_config):
    """
    Open the stats datastore next to `fp_boundary_config` as neural-lam does
    (`overload_stats_path`) and check the boundary statistics it uses.
    """
    from neural_lam.datastore.mdp import MDPDatastore

    fp_zarr = Path(fp_boundary_config).with_suffix(".zarr")
    if not fp_zarr.exists():
        # MDPDatastore would otherwise try to build the full training dataset
        raise FileNotFoundError(fp_zarr)
    datastore = MDPDatastore(config_path=fp_boundary_config)
    for category in ["forcing", "static"]:
        ds = datastore.get_standardization_dataarray(category=category)
        for op in ["mean", "std"]:
            values = ds[f"{category}_{op}"].values
            if values.size == 0 or not np.isfinite(values).all():
                raise ValueError(f"bad {category}_{op} in {fp_zarr}")
        if not (ds[f"{category}_std"].values > 0).all():
            raise ValueError(f"non-positive {category}_std in {fp_zarr}")


def write_zip(artifact_dir, fp_zip):
    """
    Zip the artifact directory with its contents at the zip root, as the
    Containerfile expects (it unzips into `inference_artifact/`), without the
    checkpoint (it is downloaded separately). Written to a temporary file
    first, so an interrupted run doesn't leave a partial zip behind.
    """
    artifact_dir = Path(artifact_dir)
    fp_checkpoint = checkpoint_path(artifact_dir)
    fp_tmp = fp_zip.with_name(fp_zip.name + ".tmp")
    with zipfile.ZipFile(fp_tmp, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fp in sorted(artifact_dir.rglob("*")):
            if fp.is_file() and fp != fp_checkpoint:
                zf.write(fp, fp.relative_to(artifact_dir))
    fp_tmp.replace(fp_zip)


def _name(datastore_config):
    return mc.TrainingConfigs.name(datastore_config)
