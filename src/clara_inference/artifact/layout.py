"""
The layout of a CLARA inference artifact, shared by the tools that build it
(`assemble`, `compute_stats`) and the checks. The artifact describes the model
**as trained**:

    configs/model.yaml              the model: checkpoint, graph, train_model
                                    arguments and the training neural-lam
                                    config
    configs/<neural-lam config>     the training neural-lam config, naming the
                                    datastore configs as sibling files
    configs/<datastore config>      the training interior and boundary
                                    datastore configs, and the two configs of
                                    the datastores the statistics come from
                                    (`overload_stats_path`)
    configs/<stats datastore>.zarr  next to each statistics datastore config:
                                    the training statistics (and splits),
                                    what neural-lam reads for
                                    `overload_stats_path`
    grids/<interior>.grid.zarr      interior (DANRA) grid (x, y, lat, lon) and
                                    statics, the target grid for regridding
                                    DINI
    grids/<boundary>.grid.zarr      the boundary grid: the 0.25 deg latitude
                                    band of the training boundary points in
                                    training order, with the ERA5 statics
    graphs/<graph name>/            the graph the model was trained with
    artifact.yaml                   provenance
    README.md                       description (artifact/package-README.md)
    <checkpoint.artifact_path>      the checkpoint (not in the package zip)
"""
import zipfile
from pathlib import Path

import numpy as np
import yaml

from clara_inference import model_configs as mc

CONFIGS = "configs"
GRIDS = "grids"
GRAPHS = "graphs"
ARTIFACT_META = "artifact.yaml"
README = "README.md"
GRAPH_FILES = [
    "g2m_edge_index.pt",
    "g2m_features.pt",
    "g2m_node_features.pt",
    "m2m_edge_index.pt",
    "m2m_features.pt",
    "m2m_node_features.pt",
]


def _name(datastore_config):
    return mc.TrainingConfigs.name(datastore_config)


def model_description(artifact_dir):
    return yaml.safe_load(
        (Path(artifact_dir) / CONFIGS / "model.yaml").read_text()
    )


def grid_path(artifact_dir, datastore_config):
    return Path(artifact_dir) / GRIDS / f"{_name(datastore_config)}.grid.zarr"


def stats_datastore_path(artifact_dir, datastore_config):
    """The zarr neural-lam opens next to an `overload_stats_path` config."""
    return Path(artifact_dir) / CONFIGS / f"{_name(datastore_config)}.zarr"


def graph_path(artifact_dir):
    name = model_description(artifact_dir)["graph"]["name"]
    return Path(artifact_dir) / GRAPHS / name


def checkpoint_path(artifact_dir):
    model = model_description(artifact_dir)
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
        artifact_dir / ARTIFACT_META,
        artifact_dir / README,
        grid_path(artifact_dir, training.interior_datastore),
        grid_path(artifact_dir, training.boundary_datastore),
    }
    for stats_config in training.stats_datastores:
        paths |= {
            configs_dir / stats_config,
            stats_datastore_path(artifact_dir, stats_config),
        }
    paths |= {graph_path(artifact_dir) / fn for fn in GRAPH_FILES}
    if checkpoint:
        paths.add(checkpoint_path(artifact_dir))
    return sorted(paths)


def check_artifact(artifact_dir, checkpoint=False):
    """
    Check an artifact directory: all required paths exist, nothing in
    configs/ besides the training configs and stats datastores, and the
    stats datastores load in neural-lam with sane statistics. Raises
    ValueError on problems.
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
    if problems:
        raise ValueError(
            f"inference artifact {artifact_dir}: " + "; ".join(problems)
        )
    training = training_configs(artifact_dir)
    check_stats_datastore(
        configs_dir / training.interior_stats_datastore,
        categories=["state", "forcing", "static"],
    )
    check_stats_datastore(
        configs_dir / training.boundary_stats_datastore,
        categories=["forcing", "static"],
    )


def check_stats_datastore(fp_config, categories):
    """
    Open the stats datastore next to `fp_config` as neural-lam does
    (`overload_stats_path`) and check its statistics.
    """
    from neural_lam.datastore.mdp import MDPDatastore

    fp_zarr = Path(fp_config).with_suffix(".zarr")
    if not fp_zarr.exists():
        # MDPDatastore would otherwise try to build the full training dataset
        raise FileNotFoundError(fp_zarr)
    datastore = MDPDatastore(config_path=fp_config)
    for category in categories:
        ds = datastore.get_standardization_dataarray(category=category)
        for v in ds.data_vars:
            values = ds[v].values
            if values.size == 0 or not np.isfinite(values).all():
                raise ValueError(f"bad {v} in {fp_zarr}")
        std = ds[f"{category}_std"].values
        # constant static/forcing features (e.g. a land-sea mask over sea
        # only) can have zero std; the state can't
        if (std < 0).any() or (category == "state" and not (std > 0).all()):
            raise ValueError(f"bad {category}_std in {fp_zarr}")


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
