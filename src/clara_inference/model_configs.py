"""
Which configs CLARA uses.

The trained model is described by the inference artifact's `configs/`:
`model.yaml` names the neural-lam config the model was trained with
(`danra_model_config_ifs_all.yaml`), which names

- the interior datastore config (DANRA, `danra_model1_config_ifs_finetune`)
  and the config of the datastore its statistics come from
  (`danra_model1_config`, via `overload_stats_path`)
- the boundary datastore config (IFS forecasts,
  `overlapping_ifs_all2_400km_model1_config`) and the config of the datastore
  its statistics come from (ERA5, `overlapping_era_400km_model1_config`)

Unlike ANNA, the operational boundary (IFS forecasts) is what CLARA was
fine-tuned on, so the training boundary datastore config is used as is (with
its input paths replaced). The statistics datastores are shipped in the
artifact as `configs/<name>.zarr` next to their configs, holding only the
training statistics and splits, which is all neural-lam reads for
`overload_stats_path`.
"""
import copy
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class TrainingConfigs:
    """The training config files (in the artifact's configs directory)."""

    neural_lam_config: str
    interior_datastore: str
    interior_stats_datastore: str
    boundary_datastore: str
    boundary_stats_datastore: str

    @staticmethod
    def name(fn):
        """Datastore name: the config file name without `.yaml`."""
        return Path(fn).stem

    @property
    def stats_datastores(self):
        return [self.interior_stats_datastore, self.boundary_stats_datastore]


def training_configs(configs_dir):
    """The training configs named by `model.yaml` in `configs_dir`."""
    configs_dir = Path(configs_dir)
    model = yaml.safe_load((configs_dir / "model.yaml").read_text())
    fn_nl_config = model["neural_lam_config"]
    nl_config = yaml.safe_load((configs_dir / fn_nl_config).read_text())
    interior = nl_config["datastore"]
    boundary = nl_config["datastore_boundary"]
    return TrainingConfigs(
        neural_lam_config=fn_nl_config,
        interior_datastore=interior["config_path"],
        interior_stats_datastore=interior["overload_stats_path"],
        boundary_datastore=boundary["config_path"],
        boundary_stats_datastore=boundary["overload_stats_path"],
    )


def inference_neural_lam_config(
    training_nl_config,
    interior_config,
    boundary_config,
    interior_stats_config,
    boundary_stats_config,
):
    """
    The neural-lam config for a forecast: the training config (a dict) with
    the inference interior and boundary datastores, normalised with the
    training statistics (`overload_stats_path`: the stats datastore configs
    in the artifact, next to their stats-only zarrs). neural-lam joins the
    paths onto the config's directory, so the stats configs are given as
    absolute paths.
    """
    nl_config = copy.deepcopy(training_nl_config)
    nl_config["datastore"]["config_path"] = str(interior_config)
    nl_config["datastore"]["overload_stats_path"] = str(interior_stats_config)
    boundary = nl_config["datastore_boundary"]
    boundary["config_path"] = str(boundary_config)
    boundary["overload_stats_path"] = str(boundary_stats_config)
    return nl_config
