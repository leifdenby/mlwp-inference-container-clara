"""
Which configs ANNA uses.

- **The trained model** (the inference artifact's `configs/`): `model.yaml`
  names the neural-lam config the model was trained with, which names the
  interior (DANRA) and boundary (ERA5) datastore configs. The artifact has
  their statistics and grids.
- **Operational boundaries** (this package, `anna_inference/boundary/`): the
  boundary datastore configs for running from DINI or IFS forecasts. They are
  not part of the model. The neural-lam config for a forecast is derived from
  the training one (`inference_neural_lam_config`): the same model and
  settings, with the operational boundary normalised with the ERA5 training
  statistics (`overload_stats_path`).
"""
import copy
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import yaml

# operational boundary sources and their datastore configs (package data)
BOUNDARY_SOURCES = {
    "dini": "dini_7deg_model1_config.yaml",
    "ifs": "ifs_7deg_model1_config.yaml",
}


@dataclass
class TrainingConfigs:
    """The training config files (in the artifact's configs directory)."""

    neural_lam_config: str
    interior_datastore: str
    boundary_datastore: str

    @staticmethod
    def name(fn):
        """Datastore name: the config file name without `.yaml`."""
        return Path(fn).stem


def training_configs(configs_dir):
    """The training configs named by `model.yaml` in `configs_dir`."""
    configs_dir = Path(configs_dir)
    model = yaml.safe_load((configs_dir / "model.yaml").read_text())
    fn_nl_config = model["neural_lam_config"]
    nl_config = yaml.safe_load((configs_dir / fn_nl_config).read_text())
    return TrainingConfigs(
        neural_lam_config=fn_nl_config,
        interior_datastore=nl_config["datastore"]["config_path"],
        boundary_datastore=nl_config["datastore_boundary"]["config_path"],
    )


def boundary_config_path(boundary_source):
    """The (packaged) boundary datastore config of an operational source."""
    if boundary_source not in BOUNDARY_SOURCES:
        raise KeyError(
            f"unknown boundary source {boundary_source!r} "
            f"(have {sorted(BOUNDARY_SOURCES)})"
        )
    fp = (
        resources.files("anna_inference")
        / "boundary"
        / BOUNDARY_SOURCES[boundary_source]
    )
    return Path(str(fp))


def inference_neural_lam_config(
    training_nl_config, interior_config, boundary_config, stats_config
):
    """
    The neural-lam config for a forecast: the training config (a dict) with
    the inference interior datastore, the operational boundary datastore, and
    the boundary normalised with the training boundary statistics
    (`overload_stats_path`, the training boundary datastore config next to
    its stats datastore zarr).
    """
    nl_config = copy.deepcopy(training_nl_config)
    nl_config["datastore"]["config_path"] = str(interior_config)
    boundary = nl_config["datastore_boundary"]
    boundary["config_path"] = str(boundary_config)
    boundary["overload_stats_path"] = str(stats_config)
    return nl_config
